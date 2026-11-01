"""Host-side interception: an MCP interceptor outside the target, bound to one evaluation by the host.

The host (the process that runs the orchestrator) builds a
:class:`HostInterceptor` for each evaluation of a host-mode run
(``midojo-run --interception host``). It gets the evaluation's recording
authority at setup, a ``midojo_sdk.recorder.Recorder`` for the evaluation's
session (in process, or over HTTP with the session token), and the tool-output
part of the episode's injection plan. It mirrors the upstream MCP server (the
real tools), records every call through that recorder and forwards it. A
suite whose tools read or change the simulated environment gives the run its
``tools`` (see :class:`HostInterception`): they are registered on each
evaluation's interceptor as simulated tools, which get the session's
environment on the host, and replace the upstream's tools of the same name;
the upstream's other tools are mirrored as before.

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
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

import uvicorn
from midojo_sdk.injection import InjectionOutcome, InjectionPlan
from midojo_sdk.mcp import MidojoMCP
from midojo_sdk.recorder import Recorder, WorldState

from midojo.tool_endpoint import TOOL_PATH, ToolEndpoint, bare_host, tool_url

# Hosts that mean every interface: a listener can bind them, but a target can't connect to them.
_WILDCARD_HOSTS = {"0.0.0.0", "::", ""}

# How long stopping waits for the target's open connections before closing them.
_SHUTDOWN_SECONDS = 2

HostTools = Callable[[MidojoMCP], None]
"""Registers a suite's simulated tools, blocks and denials on one evaluation's interceptor (its ``MidojoMCP``)."""


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


class HostInterceptor:
    """Mirrors ``upstream``'s tools for one evaluation, recording each call with ``recorder`` and applying ``plan``.

    ``upstream`` is the real MCP server: a URL, or anything a fastmcp
    ``Client`` connects to (a ``FastMCP`` server in this process). ``plan`` is
    the tool-output part of the episode's plan; ``evaluation_id`` scopes its
    occurrence counters and never leaves the host. ``observe_requests`` keeps
    each request it receives, refused ones included, in ``requests``.

    ``tools`` registers simulated tools (``@mcp.tool()``), blocks and denials
    on the interceptor before it mirrors the upstream, so they replace the
    upstream's tools of the same name. A simulated tool reads and changes the
    evaluation's environment through ``world``, which stays on the host with
    the session; without ``world`` it has no environment. A simulated tool
    takes no tool-output instruction (``CapabilityError``), as the upstream
    never produced its result.
    """

    def __init__(
        self,
        recorder: Recorder,
        *,
        upstream: Any,
        evaluation_id: str,
        plan: InjectionPlan = InjectionPlan(),
        world: WorldState | None = None,
        tools: HostTools | None = None,
        bind_host: str = "127.0.0.1",
        port: int = 0,
        advertise_host: str | None = None,
        name: str = "midojo-interceptor",
        observe_requests: bool = False,
        allowed: Iterable[str] | None = None,
        blocked: Iterable[str] = (),
    ) -> None:
        url, transport = (upstream, None) if isinstance(upstream, str) else (None, upstream)
        self.mcp = MidojoMCP(
            name,
            recorder=recorder,
            world=world,
            upstream_url=url,
            upstream_transport=transport,
            plan=plan,
            evaluation_id=evaluation_id,
        )
        if tools is not None:
            tools(self.mcp)
        # An episode's tool policy: mirror only the tools it allows (None mirrors all), and block its writes.
        self._allowed = None if allowed is None else frozenset(allowed)
        self._blocked = tuple(dict.fromkeys(blocked))
        self.mcp.block(*self._blocked)
        self._token = secrets.token_urlsafe(32)
        # A host is kept without URL brackets, and bracketed once when the endpoint's URL is made.
        self._bind_host = bind_host = bare_host(bind_host)
        self._port = port
        default_advertise = bind_host if bind_host not in _WILDCARD_HOSTS else "127.0.0.1"
        self._advertise_host = bare_host(advertise_host) if advertise_host else default_advertise
        # With observe_requests, every request the interceptor received, in full (credentials included): what
        # the target sent, for checking it. Off by default, so a long run keeps none of it.
        self._observe_requests = observe_requests
        self.requests: list[ObservedRequest] = []
        self._server: _Server | None = None
        self._serving: asyncio.Task[None] | None = None

    @property
    def injection_outcomes(self) -> tuple[InjectionOutcome, ...]:
        return self.mcp.injection_outcomes

    async def tool_names(self) -> list[str]:
        """The tools this interceptor lists: the ones it mirrors under its policy, and its simulated ones."""
        return await self.mcp.tool_names()

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
            if self._observe_requests:
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

            async def refuse(status: int, body: bytes) -> None:
                # An observed request keeps all the target sent, so read its body first. Unobserved, answer at
                # once: an unauthorized client can't make the interceptor wait for, or hold, a body it drips.
                if self._observe_requests:
                    while (await receive_and_keep()).get("more_body", False):
                        pass
                await _respond(send_and_note, status, body)

            if scope["path"].rstrip("/") != TOOL_PATH:
                await refuse(404, b"Not found")
                return
            authorization = next((value for name, value in scope["headers"] if name.lower() == b"authorization"), b"")
            if not secrets.compare_digest(authorization, expected):
                await refuse(401, b"This interceptor's bearer token is required")
                return
            await mcp_app(scope, receive_and_keep, send_and_note)

        return intercept

    async def start(self) -> ToolEndpoint:
        """Mirror the upstream's tools and serve them. Returns the endpoint to give the target."""
        await self.mcp.mirror_upstream(allowed=self._allowed)
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
        return ToolEndpoint(tool_url(self._advertise_host, port), self._token)

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
    for a wildcard bind. ``observe_requests`` makes each interceptor keep the
    requests it receives (see :class:`HostInterceptor`), for tests.

    ``tools`` are the suite's simulated tools (a suite's ``HOST_TOOLS``),
    registered on every evaluation's interceptor, which then gets the
    evaluation's environment as well as its recorder.

    ``allowed`` and ``blocked`` are the episode's tool policy: ``allowed``
    lists the tools each interceptor mirrors (None mirrors every upstream
    tool), and ``blocked`` lists the tools it refuses to run, so a write never
    reaches the upstream.
    """

    upstream: Any
    bind_host: str = "127.0.0.1"
    port: int = 0
    advertise_host: str | None = None
    observe_requests: bool = False
    tools: HostTools | None = None
    allowed: tuple[str, ...] | None = None
    blocked: tuple[str, ...] = ()

    def interceptor(
        self, recorder: Recorder, *, evaluation_id: str, plan: InjectionPlan, world: WorldState | None = None
    ) -> HostInterceptor:
        """A new interceptor for one evaluation, recording with ``recorder``; its ``tools`` change ``world``."""
        return HostInterceptor(
            recorder,
            upstream=self.upstream,
            evaluation_id=evaluation_id,
            plan=plan,
            world=world,
            tools=self.tools,
            bind_host=self.bind_host,
            port=self.port,
            advertise_host=self.advertise_host,
            observe_requests=self.observe_requests,
            allowed=self.allowed,
            blocked=self.blocked,
        )
