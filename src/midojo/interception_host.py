"""Host-side interception: an MCP interceptor outside the target, bound to one evaluation by the host.

The host (the process that runs the orchestrator) builds a
:class:`HostInterceptor` for each evaluation of a host-mode run
(``midojo-run --interception host``). It gets the evaluation's recording
authority at setup, a ``midojo_sdk.recorder.Recorder`` for the evaluation's
session (in process, or over HTTP with the session token), and the tool-output
part of the episode's injection plan. It mirrors the upstream MCP server (the
real tools), records every call through that recorder and forwards it.

The target gets only a :class:`~midojo.tool_endpoint.ToolEndpoint`: the
interceptor's URL and a bearer token made for this interceptor alone. No
session token, control-plane URL, control token, run ID or evaluation ID
reaches it. A per-evaluation port or path is addressing, not authorization:
the interceptor refuses (401) any request without its own token, so one
evaluation's target can't call another evaluation's tools or add to its records.

Closure is the evaluation's state, not the token's: once the evaluation is
closing or sealed, the control plane refuses the call's start, records it as
a late fact, and the interceptor answers the call with an error without
forwarding it.

Reachability: the interceptor listens on ``bind_host:port`` and is advertised
to the target at ``advertise_host``. For an unmanaged agent on this machine the
defaults (``127.0.0.1``, an ephemeral port) do. An OpenShell sandbox can't
reach the host's loopback: bind an address the sandbox reaches, give a fixed
port, advertise ``host.openshell.internal``, and allow that host and port in
the suite's network policy.

This module needs ``midojo-sdk[mcp]`` (the ``dev`` and ``suites`` extras).
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import socket
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from midojo_sdk.injection import InjectionOutcome, InjectionPlan
from midojo_sdk.mcp import MidojoMCP
from midojo_sdk.recorder import Recorder

from midojo.tool_endpoint import ToolEndpoint

# The path the interceptor serves its tools on.
TOOL_PATH = "/mcp"

# Hosts that mean every interface: a listener can bind them, but a target can't connect to them.
_WILDCARD_HOSTS = {"0.0.0.0", "::", ""}

# How long stopping waits for the target's open connections before closing them.
_SHUTDOWN_SECONDS = 2


@dataclass
class ObservedRequest:
    """One HTTP request the interceptor received, as the target sent it, whether or not it was authorized."""

    method: str
    path: str
    query: str
    headers: list[tuple[str, str]]
    status: int | None = None
    body: bytes = b""
    chunks: list[bytes] = field(default_factory=list, repr=False)


class _Server(uvicorn.Server):
    """A uvicorn server that leaves the process's signal handlers to the host.

    uvicorn's own handlers would take the orchestrator's Ctrl-C. And once a
    stopped server's handler is left installed (two interceptors served at
    once restore each other's), sse_starlette reads that server's
    ``should_exit`` and ends every event stream in the process, the next
    interceptors' included.
    """

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def _url_host(host: str) -> str:
    return f"[{host}]" if ":" in host else host


class HostInterceptor:
    """Mirrors ``upstream``'s tools for one evaluation, recording each call with ``recorder`` and applying ``plan``.

    ``upstream`` is the real MCP server: a URL, or anything a fastmcp
    ``Client`` connects to (a ``FastMCP`` server in this process). ``plan`` is
    the tool-output part of the episode's plan; ``evaluation_id`` scopes its
    occurrence counters and never leaves the host.
    """

    def __init__(
        self,
        recorder: Recorder,
        *,
        upstream: Any,
        evaluation_id: str,
        plan: InjectionPlan = InjectionPlan(),
        bind_host: str = "127.0.0.1",
        port: int = 0,
        advertise_host: str | None = None,
        name: str = "midojo-interceptor",
    ) -> None:
        url, transport = (upstream, None) if isinstance(upstream, str) else (None, upstream)
        self.mcp = MidojoMCP(
            name,
            recorder=recorder,
            upstream_url=url,
            upstream_transport=transport,
            plan=plan,
            evaluation_id=evaluation_id,
        )
        self._token = secrets.token_urlsafe(32)
        self._bind_host = bind_host
        self._port = port
        self._advertise_host = advertise_host or (bind_host if bind_host not in _WILDCARD_HOSTS else "127.0.0.1")
        # Every request the interceptor received, in full: what the target sent, for checking it.
        self.requests: list[ObservedRequest] = []
        self._server: _Server | None = None
        self._serving: asyncio.Task[None] | None = None

    @property
    def injection_outcomes(self) -> tuple[InjectionOutcome, ...]:
        return self.mcp.injection_outcomes

    def app(self) -> Any:
        """The interceptor as an ASGI app: the tools at ``TOOL_PATH``, for this interceptor's token only."""
        mcp_app = self.mcp.http_app(path=TOOL_PATH)
        expected = f"Bearer {self._token}".encode()

        async def intercept(scope, receive, send):
            if scope["type"] != "http":
                await mcp_app(scope, receive, send)
                return
            request = ObservedRequest(
                method=scope["method"],
                path=scope["path"],
                query=scope["query_string"].decode("latin-1"),
                headers=[(name.decode("latin-1"), value.decode("latin-1")) for name, value in scope["headers"]],
            )
            self.requests.append(request)

            async def receive_and_keep():
                message = await receive()
                if message["type"] == "http.request":
                    request.chunks.append(message.get("body", b""))
                    request.body = b"".join(request.chunks)
                return message

            async def send_and_note(message):
                if message["type"] == "http.response.start":
                    request.status = message["status"]
                await send(message)

            if scope["path"].rstrip("/") != TOOL_PATH:
                await _respond(send_and_note, 404, b"Not found")
                return
            authorization = next((value for name, value in scope["headers"] if name.lower() == b"authorization"), b"")
            if not secrets.compare_digest(authorization, expected):
                await _respond(send_and_note, 401, b"This interceptor's bearer token is required")
                return
            await mcp_app(scope, receive_and_keep, send_and_note)

        return intercept

    async def start(self) -> ToolEndpoint:
        """Mirror the upstream's tools and serve them. Returns the endpoint to give the target."""
        await self.mcp.mirror_upstream()
        # Bound here, so a taken port raises OSError now rather than ending the server's task.
        listener = _listen(self._bind_host, self._port)
        config = uvicorn.Config(
            self.app(), log_level="warning", lifespan="on", ws="none", timeout_graceful_shutdown=_SHUTDOWN_SECONDS
        )
        server = self._server = _Server(config)
        serving = self._serving = asyncio.create_task(server.serve(sockets=[listener]))
        while not server.started:
            if serving.done():
                listener.close()
                raise RuntimeError(f"The interceptor stopped before serving on {self._bind_host}:{self._port}")
            await asyncio.sleep(0.01)
        port = listener.getsockname()[1]
        return ToolEndpoint(f"http://{_url_host(self._advertise_host)}:{port}{TOOL_PATH}", self._token)

    async def stop(self) -> None:
        """Stop serving. The target's open connections get a short grace period, then are closed."""
        if self._server is not None and self._serving is not None:
            self._server.should_exit = True
            await self._serving
            self._server = self._serving = None


def _listen(host: str, port: int) -> socket.socket:
    family, kind, proto, _, address = socket.getaddrinfo(
        host or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )[0]
    listener = socket.socket(family, kind, proto)
    try:
        # Each evaluation of a run may listen on the same fixed port as the last one did.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(address)
    except OSError:
        listener.close()
        raise
    return listener


async def _respond(send, status: int, body: bytes) -> None:
    headers = [(b"content-type", b"text/plain; charset=utf-8")]
    if status == 401:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


@dataclass(frozen=True)
class HostInterception:
    """A run's host-mode settings: the upstream every evaluation's interceptor forwards to, and where it listens.

    ``port`` 0 picks a free port for each evaluation. ``advertise_host`` is the
    host the target connects to; by default the bind host, or ``127.0.0.1``
    for a wildcard bind.
    """

    upstream: Any
    bind_host: str = "127.0.0.1"
    port: int = 0
    advertise_host: str | None = None

    def interceptor(self, recorder: Recorder, *, evaluation_id: str, plan: InjectionPlan) -> HostInterceptor:
        """A new interceptor for one evaluation, recording with ``recorder``."""
        return HostInterceptor(
            recorder,
            upstream=self.upstream,
            evaluation_id=evaluation_id,
            plan=plan,
            bind_host=self.bind_host,
            port=self.port,
            advertise_host=self.advertise_host,
        )
