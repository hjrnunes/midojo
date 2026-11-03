"""The driver's end of one binding over a real WebSocket: what a remote evaluation's process runs.

The driver owns the evaluation. It opens the evaluation on its own control
plane, upgrades one socket to the interceptor service with the bind
credential, sends ``bind``, and from then on serves the service's frames with
:class:`~midojo.control_plane.link_dispatch.LinkDispatcher`, the v9md.1 link
core's driver side. The service never sees the evaluation's session token.

Tests only: no model, and no network beyond loopback.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import secrets
import socket
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import uvicorn
from midojo_sdk.injection import InjectionPlan
from midojo_sdk.link import API, LinkFault, Message, decode_message, encode_message
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus
from websockets.typing import Subprotocol

from midojo.control_plane.link_dispatch import LinkDispatcher

MAX_FRAME_BYTES = 16 * 1024 * 1024
"""The client's frame limit, set explicitly: the default 1 MiB is below what a tool result can be."""

SERVICE_TIMEOUT = 10.0
"""How long a test waits for the service to answer or to end a binding: generous, so a slow test fails alone."""


@dataclass(frozen=True)
class Listener:
    """One uvicorn listener on a loopback port, for the upstream a test serves."""

    port: int

    def url(self, path: str = "/") -> str:
        return f"http://127.0.0.1:{self.port}{path}"


class _Server(uvicorn.Server):
    """A server that leaves signal handling to the process, as the interceptors' own server does."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


@contextlib.asynccontextmanager
async def serve_app(app: Any) -> AsyncIterator[Listener]:
    """Serve ``app`` on one uvicorn listener on a free loopback port, entering its lifespan."""
    listener = _listen()
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
        yield Listener(listener.getsockname()[1])
    finally:
        server.should_exit = True
        await serving


def _listen() -> socket.socket:
    family, kind, proto, _, address = socket.getaddrinfo(
        "127.0.0.1", 0, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )[0]
    listener = socket.socket(family, kind, proto)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(address)
    except OSError:
        listener.close()
        raise
    return listener


@dataclass(frozen=True)
class Bound:
    """What ``bound`` told the driver: this binding's endpoint, its own tool token, and the tools it serves."""

    tool_url: str
    tool_token: str
    tools: tuple[str, ...]


def _websocket_url(url: str) -> str:
    """The WebSocket scheme of an HTTP URL: the service publishes ``http`` URLs, a driver dials ``ws``."""
    scheme, _, rest = url.partition("://")
    return f"{'wss' if scheme == 'https' else 'ws'}://{rest}"


async def _connect(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    subprotocols: Sequence[str] = (API,),
    timeout: float = SERVICE_TIMEOUT,
) -> ClientConnection:
    return await asyncio.wait_for(
        connect(
            _websocket_url(url),
            additional_headers=dict(headers or {}),
            subprotocols=[Subprotocol(name) for name in subprotocols],
            max_size=MAX_FRAME_BYTES,
            proxy=None,
        ),
        timeout,
    )


async def upgrade(
    service_url: str,
    credential: str,
    *,
    path: str = "/v1/bindings",
    subprotocols: Sequence[str] = (API,),
    extra_headers: Mapping[str, str] | None = None,
) -> ClientConnection:
    """Upgrade one socket to ``path`` with ``credential`` as the bind credential."""
    headers = {"authorization": f"Bearer {credential}", **dict(extra_headers or {})}
    return await _connect(f"{service_url}{path}", headers=headers, subprotocols=subprotocols)


async def refusal(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    subprotocols: Sequence[str] = (API,),
) -> int | None:
    """The HTTP status the upgrade at ``url`` is refused with, or None when the upgrade was accepted."""
    try:
        websocket = await _connect(url, headers=headers, subprotocols=subprotocols)
    except InvalidStatus as refused:
        return int(refused.response.status_code)
    await websocket.close()
    return None


async def raw_upgrade_status(url: str, headers: Mapping[str, str]) -> int:
    """The HTTP status a bare handshake gets, for the headers a WebSocket client won't let a test leave out.

    The client always sends ``Sec-WebSocket-Protocol``, empty when it offers none, and an
    empty header is invalid HTTP: the server refuses such a handshake itself, before any
    app code runs. This sends the header only when ``headers`` carries one.
    """
    parts = urlsplit(url)
    assert parts.hostname is not None and parts.port is not None
    key = base64.b64encode(secrets.token_bytes(16)).decode()
    lines = [
        f"GET {parts.path or '/'} HTTP/1.1",
        f"Host: {parts.netloc}",
        "Upgrade: websocket",
        "Connection: Upgrade",
        f"Sec-WebSocket-Key: {key}",
        "Sec-WebSocket-Version: 13",
        *(f"{name}: {value}" for name, value in headers.items()),
    ]
    reader, writer = await asyncio.open_connection(parts.hostname, parts.port)
    try:
        writer.write(("\r\n".join(lines) + "\r\n\r\n").encode())
        await writer.drain()
        status = await asyncio.wait_for(reader.readline(), SERVICE_TIMEOUT)
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()
    return int(status.split()[1])


async def bind(
    websocket: ClientConnection,
    *,
    upstream_url: str,
    plan: InjectionPlan = InjectionPlan(),
    policy: Mapping[str, Any] | None = None,
    host_tools: Sequence[str] = (),
    faces: Sequence[str] = ("mcp",),
    api: str = API,
    fields: Mapping[str, Any] | None = None,
    op: str = "bind",
) -> Bound:
    """Send one ``bind`` and read its answer, or raise naming the ``rejected`` the service answered."""
    body: dict[str, Any] = {
        "api": api,
        "plan": plan.to_data(),
        "policy": None if policy is None else dict(policy),
        "upstream_url": upstream_url,
        "host_tools": list(host_tools),
        "faces": list(faces),
    }
    body.update(fields or {})
    await websocket.send(encode_message(op, **body))
    answer = await asyncio.wait_for(websocket.recv(), SERVICE_TIMEOUT)
    message = decode_message(answer if isinstance(answer, str) else answer.decode())
    if message.op != "bound":
        raise RefusedError(message)
    return Bound(
        tool_url=message["tool_url"],
        tool_token=message["tool_token"],
        tools=tuple(message["tools"]),
    )


class RefusedError(AssertionError):
    """The service answered ``rejected`` instead of ``bound``: the test reads ``kind`` and ``detail``."""

    def __init__(self, message: Message) -> None:
        super().__init__(f"bind was answered {message.op!r}: {dict(message.fields)}")
        self.message = message

    @property
    def kind(self) -> str:
        return str(self.message["kind"])

    @property
    def detail(self) -> str:
        return str(self.message["detail"])


class _DriverDuplex:
    """The socket as the link core's Duplex: frames in, frames out.

    The service's control messages (``unbound``, ``rejected``) travel on the
    same socket outside the frame stream, so the reader takes them for the
    driver and hands the dispatcher only frames.
    """

    def __init__(self, driver: BindingDriver) -> None:
        self._driver = driver

    async def send(self, data: str) -> None:
        await self._driver.websocket.send(data)

    async def receive(self) -> str:
        while True:
            raw = await self._driver._read()
            if not self._driver._control(raw):
                return raw

    def close(self) -> None:
        self._driver._schedule_close()


@dataclass
class BindingDriver:
    """One bound evaluation's driver: the socket, the bind it answered, and the frames it serves.

    ``dispatcher`` owns the evaluation's records; its serve task runs until
    the service ends the link. A test reads ``unbound``, ``injection_outcomes``,
    ``close_code`` and ``dispatcher.applications`` to see what the service did.
    """

    websocket: ClientConnection
    dispatcher: LinkDispatcher
    bound: Bound
    unbound: asyncio.Event = field(default_factory=asyncio.Event)
    injection_outcomes: tuple[Mapping[str, Any], ...] = ()
    rejected: Message | None = None
    closed: asyncio.Event = field(default_factory=asyncio.Event)
    close_code: int | None = None
    close_reason: str | None = None
    _tasks: list[asyncio.Task[None]] = field(default_factory=list)
    _gate: asyncio.Event = field(default_factory=asyncio.Event)

    def __post_init__(self) -> None:
        self._gate.set()

    def pause(self) -> None:
        """Stop reading the service's frames: a driver that hung, with its socket still open."""
        self._gate.clear()

    def resume(self) -> None:
        """Read the service's frames again, and apply the ones that queued while paused."""
        self._gate.set()

    @classmethod
    async def open(
        cls,
        service_url: str,
        credential: str,
        dispatcher: LinkDispatcher,
        *,
        upstream_url: str,
        plan: InjectionPlan = InjectionPlan(),
        policy: Mapping[str, Any] | None = None,
        host_tools: Sequence[str] = (),
        faces: Sequence[str] = ("mcp",),
        subprotocols: Sequence[str] = (API,),
    ) -> BindingDriver:
        """Upgrade, bind, and start serving frames: the driver's whole side of one binding."""
        websocket = await upgrade(service_url, credential, subprotocols=subprotocols)
        try:
            answered = await bind(
                websocket,
                upstream_url=upstream_url,
                plan=plan,
                policy=policy,
                host_tools=host_tools,
                faces=faces,
            )
        except BaseException:
            await websocket.close()
            raise
        driver = cls(websocket=websocket, dispatcher=dispatcher, bound=answered)
        driver._tasks = [
            asyncio.create_task(driver.dispatcher.serve(_DriverDuplex(driver)), name="binding-driver-serve"),
            asyncio.create_task(driver._watch_close(), name="binding-driver-close"),
        ]
        return driver

    def _schedule_close(self) -> None:
        if self.closed.is_set():
            return
        asyncio.get_running_loop().create_task(self.websocket.close())

    async def _read(self) -> str:
        await self._gate.wait()
        raw = await self.websocket.recv()
        return raw.decode() if isinstance(raw, bytes) else raw

    def _control(self, raw: str) -> bool:
        """Take ``raw`` when it is a control message for the driver; answer whether it did."""
        try:
            message = decode_message(raw)
        except LinkFault:
            return False
        if message.op == "unbound":
            self.injection_outcomes = tuple(message["injection_outcomes"])
            self.unbound.set()
            return True
        if message.op == "rejected":
            self.rejected = message
            return True
        return False

    async def _watch_close(self) -> None:
        await self.websocket.wait_closed()
        self.close_code = self.websocket.close_code
        self.close_reason = self.websocket.close_reason
        self.closed.set()

    async def ping(self) -> None:
        """The driver's heartbeat: one frame the service times its expiry against."""
        await self.websocket.send(encode_message("ping"))

    async def unbind(self) -> None:
        """Release the binding: the service unmounts, answers ``unbound``, and closes normally."""
        await self.websocket.send(encode_message("unbind"))

    async def hang_up(self) -> None:
        """End the socket from the driver's side without unbinding: the loss a dead driver leaves."""
        await self.websocket.close()
        await self.closed.wait()

    async def wait_closed(self) -> None:
        """Wait for the socket to end, as a driver's own process waits for its bindings."""
        await asyncio.wait_for(self.closed.wait(), SERVICE_TIMEOUT)

    async def wait_unbound(self) -> None:
        await asyncio.wait_for(self.unbound.wait(), SERVICE_TIMEOUT)

    async def aclose(self) -> None:
        """Hang up if the socket is still open, then drop the driver's own end."""
        if not self.closed.is_set():
            with contextlib.suppress(Exception):
                await self.websocket.close()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []
        with contextlib.suppress(Exception):
            await self.websocket.close()
