"""Spike: bindings mounted and unmounted on one uvicorn listener at runtime (bead asago-v9md.2).

A binding is one ``MidojoMCP``'s FastMCP ``http_app`` at ``/b/<id>/mcp``,
behind the binding's own bearer token. ``BindingHost.mount`` appends the mount
to the root app's route table and starts a task that enters the app's own
lifespan: the FastMCP session manager runs only while the binding is mounted.
``BindingHost.unmount`` removes the mount first, so new requests get 404, then
leaves the lifespan. One listener serves every binding, so unbinding one
leaves the others serving.

Spike code, not a library: v9md.3 grows the remote interceptor service from it.
The gotchas it records for that step are in the bead's findings comment.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
import socket
from collections.abc import AsyncIterator, Generator
from dataclasses import dataclass

import uvicorn
from midojo_sdk.mcp import MidojoMCP
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import ASGIApp, Receive, Scope, Send

from midojo.tool_endpoint import TOOL_PATH


@dataclass(frozen=True)
class Listener:
    """One uvicorn listener: its port, and the server object a test can inspect for leaks."""

    server: uvicorn.Server
    port: int

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"


@contextlib.asynccontextmanager
async def serve(app: Starlette) -> AsyncIterator[Listener]:
    """Serve ``app`` on one uvicorn listener on a free loopback port, with WebSockets enabled.

    ``ws="websockets-sansio"`` is deliberate, for two reasons: the
    in-process interceptor sets ``ws="none"``, which the service must not
    inherit (M5b plan rule 8); and uvicorn's ``ws="auto"`` picks its legacy
    ``websockets_impl``, which websockets 16 deprecates.
    """
    listener = _listen("127.0.0.1", 0)
    config = uvicorn.Config(
        app, log_level="warning", lifespan="on", ws="websockets-sansio", timeout_graceful_shutdown=2
    )
    server = _Server(config)
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    try:
        while not server.started:
            if serving.done():
                raise RuntimeError("the listener stopped before it served")
            await asyncio.sleep(0.01)
        yield Listener(server=server, port=listener.getsockname()[1])
    finally:
        server.should_exit = True
        await serving


class _Server(uvicorn.Server):
    """A server that leaves signal handling to the process, as ``HostInterceptor``'s server does."""

    @contextlib.contextmanager
    def capture_signals(self) -> Generator[None, None, None]:
        yield


def _listen(host: str, port: int) -> socket.socket:
    """A bound, not yet listening socket, the same way ``HostInterceptor`` takes its port."""
    family, kind, proto, _, address = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE)[
        0
    ]
    listener = socket.socket(family, kind, proto)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(address)
    except OSError:
        listener.close()
        raise
    return listener


@dataclass
class _Binding:
    route: Mount
    task: asyncio.Task[None]
    stop: asyncio.Event


class BindingHost:
    """The root app's binding table: mount one binding at runtime, then unbind it.

    ``events`` records each binding's lifespan entering and leaving, in order,
    so a test can tell the host entered and left the app's own lifespan rather
    than leaving it to the root app's.
    """

    def __init__(self, app: Starlette) -> None:
        self.app = app
        self.events: list[tuple[str, str]] = []
        self._bindings: dict[str, _Binding] = {}

    @property
    def ids(self) -> tuple[str, ...]:
        """The mounted binding ids, in mount order."""
        return tuple(self._bindings)

    async def mount(self, binding_id: str, *, token: str, mcp: MidojoMCP) -> str:
        """Mount ``mcp`` at ``/b/<binding_id>/mcp`` behind ``token``, entering its lifespan; return the path."""
        if binding_id in self._bindings:
            raise ValueError(f"binding {binding_id!r} is already mounted")
        child = mcp.http_app(path=TOOL_PATH)
        route = Mount(f"/b/{binding_id}", app=_authorized(child, token))
        ready = asyncio.Event()
        stop = asyncio.Event()
        events = self.events

        async def lifespan() -> None:
            entered = False
            try:
                async with child.lifespan(child):
                    entered = True
                    events.append((binding_id, "entered"))
                    ready.set()
                    await stop.wait()
            finally:
                if entered:
                    events.append((binding_id, "left"))
                else:
                    # Let mount see the failure instead of waiting for a lifespan that never entered.
                    ready.set()

        task = asyncio.create_task(lifespan(), name=f"binding {binding_id}")
        await ready.wait()
        if task.done():
            raise RuntimeError(f"binding {binding_id!r} did not enter its lifespan") from task.exception()
        # Published only now: a request between the lifespan entering and the route appearing would reach an app
        # whose session manager doesn't exist yet, while a request before the lifespan enters gets the root's 404.
        self.app.router.routes.append(route)
        self._bindings[binding_id] = _Binding(route=route, task=task, stop=stop)
        return f"/b/{binding_id}{TOOL_PATH}"

    async def unmount(self, binding_id: str) -> None:
        """Stop routing to ``binding_id``, then leave its lifespan and wait for the task to end."""
        binding = self._bindings.pop(binding_id)
        self.app.router.routes.remove(binding.route)
        binding.stop.set()
        await binding.task


def _authorized(app: ASGIApp, token: str) -> ASGIApp:
    """``app`` behind this binding's bearer token, refusing every other credential with 401."""
    expected = f"Bearer {token}".encode()

    async def check(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        authorization = next((value for name, value in scope["headers"] if name.lower() == b"authorization"), b"")
        if not secrets.compare_digest(authorization, expected):
            await _refuse(send, 401, b"This binding's bearer token is required")
            return
        await app(scope, receive, send)

    return check


async def _refuse(send: Send, status: int, body: bytes) -> None:
    headers = [(b"content-type", b"text/plain; charset=utf-8")]
    if status == 401:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})
