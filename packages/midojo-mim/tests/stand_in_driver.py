"""The driver's end of one binding over a real WebSocket, with nothing of MiDojo's harness behind it.

A remote evaluation's driver upgrades one socket to the interceptor service with the bind
credential, sends ``bind``, and from then on applies the service's ``midojo-binding-v1``
frames to its evaluation and answers each one. :class:`StandInDispatcher` is that
evaluation in memory: it applies the frames in id order, as the real driver's dispatcher
does, and keeps the calls and the environment in plain Python so a test reads them. The
service never sees the evaluation's session token, and neither does this driver's socket.

The repository's own tests drive the same service with the real control plane
(``tests/binding_driver.py``); these tests need none of it.

Tests only: no model, and no network beyond loopback.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import secrets
import socket
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import uvicorn
from midojo_spec.injection import InjectionPlan
from midojo_spec.link import (
    API,
    INVALID_REQUEST,
    OPS,
    SUBPROTOCOL,
    Answer,
    Duplex,
    Frame,
    LinkError,
    LinkFault,
    LinkLost,
    Message,
    Rejection,
    decode_frame,
    decode_message,
    encode_answer,
    encode_message,
)
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus
from websockets.typing import Subprotocol

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
    subprotocols: Sequence[str] = (SUBPROTOCOL,),
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
    subprotocols: Sequence[str] = (SUBPROTOCOL,),
    extra_headers: Mapping[str, str] | None = None,
) -> ClientConnection:
    """Upgrade one socket to ``path`` with ``credential`` as the bind credential."""
    headers = {"authorization": f"Bearer {credential}", **dict(extra_headers or {})}
    return await _connect(f"{service_url}{path}", headers=headers, subprotocols=subprotocols)


async def refusal(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    subprotocols: Sequence[str] = (SUBPROTOCOL,),
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


async def request(
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
) -> None:
    """Send one ``bind`` frame and return without reading: a test holds the service before ``bound``."""
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


async def answered(websocket: ClientConnection) -> Bound:
    """Read the answer to one bind: the ``bound`` it holds, or the ``rejected`` it names."""
    answer = await asyncio.wait_for(websocket.recv(), SERVICE_TIMEOUT)
    message = decode_message(answer if isinstance(answer, str) else answer.decode())
    if message.op != "bound":
        raise RefusedError(message)
    return Bound(
        tool_url=message["tool_url"],
        tool_token=message["tool_token"],
        tools=tuple(message["tools"]),
    )


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
    await request(
        websocket,
        upstream_url=upstream_url,
        plan=plan,
        policy=policy,
        host_tools=host_tools,
        faces=faces,
        api=api,
        fields=fields,
        op=op,
    )
    return await answered(websocket)


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


@dataclass(frozen=True)
class Application:
    """One frame the dispatcher applied: the id it arrived with, its op, and the time the driver stamped it."""

    id: int
    op: str
    at: float


@dataclass
class RecordedCall:
    """One call as the driver's evaluation holds it: started when the service said so, finished once it has a result."""

    call_id: str
    function: str
    args: Any
    disposition: str
    complete: bool = False
    result: Any = None
    error: Any = None
    structured_content: Any = None
    injection_outcomes: tuple[Any, ...] = ()


class StandInDispatcher:
    """The driver's evaluation in memory: applies one binding's frames in id order and answers each.

    It keeps the discipline the real dispatcher keeps. A frame whose id is not the one after the last
    is a link fault that ends the link (the stream starts at id 0), a frame the evaluation refuses is
    answered ``rejected``, and a link that ended stays ended. ``clock`` stamps each applied frame; a
    test passes one that records the order of the driver's own work against the upstream's.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._last_id = -1
        self._lost: LinkError | None = None
        self.applications: list[Application] = []
        self.calls: list[RecordedCall] = []
        self.capture_failures: list[dict[str, Any]] = []
        self.revision = 0
        self.environment: dict[str, Any] = {}

    @property
    def lost(self) -> LinkError | None:
        return self._lost

    @property
    def applied(self) -> tuple[int, ...]:
        return tuple(application.id for application in self.applications)

    async def serve(self, duplex: Duplex) -> None:
        """Apply frames from ``duplex`` until the link is lost, closing the stream with it."""
        try:
            await self._serve(duplex)
        finally:
            with contextlib.suppress(Exception):
                duplex.close()

    async def _serve(self, duplex: Duplex) -> None:
        while self._lost is None:
            try:
                data = await duplex.receive()
            except Exception as e:
                self._lose(LinkLost(f"the binding stream ended: {e}"))
                return
            try:
                frame = decode_frame(data)
            except LinkFault as fault:
                self._lose(fault)
                return
            if frame.id != self._last_id + 1:
                what = "repeated" if frame.id <= self._last_id else "skipped"
                self._lose(LinkFault(f"frame {frame.id} arrived where {self._last_id + 1} was expected: {what}"))
                return
            try:
                answer = self._answer(frame)
            except LinkFault as fault:
                self._lose(fault)
                return
            finally:
                self._last_id = frame.id
            try:
                await duplex.send(encode_answer(answer))
            except Exception as e:
                self._lose(LinkLost(f"the answer to frame {frame.id} couldn't be sent: {e}"))
                return

    def _answer(self, frame: Frame) -> Answer:
        assert frame.op in OPS
        try:
            result = self._apply(frame)
        except _Invalid as invalid:
            self.applications.append(Application(frame.id, frame.op, self._clock()))
            return Answer(frame.id, rejected=Rejection(INVALID_REQUEST, str(invalid)))
        self.applications.append(Application(frame.id, frame.op, self._clock()))
        return Answer(frame.id, ok=result)

    def _apply(self, frame: Frame) -> Any:
        args = frame.args
        if frame.op == "start_call":
            self.calls.append(
                RecordedCall(
                    call_id=_field(args, "call_id"),
                    function=_field(args, "function"),
                    args=_field(args, "args"),
                    disposition=_field(args, "disposition"),
                )
            )
            return None
        if frame.op == "finish_call_detailed":
            call_id = _field(args, "call_id")
            call = next((call for call in self.calls if call.call_id == call_id), None)
            if call is None:
                raise _Invalid(f"call {call_id!r} was never started")
            call.complete = True
            call.result = _field(args, "result")
            call.disposition = _field(args, "disposition")
            call.error = args.get("error")
            call.structured_content = args.get("structured_content")
            call.injection_outcomes = tuple(args.get("injection_outcomes", ()))
            return None
        if frame.op == "report_capture_failure":
            self.capture_failures.append(
                {name: _field(args, name) for name in ("call_id", "function", "phase", "reason")}
            )
            return None
        if frame.op == "env_read":
            return {"revision": self.revision, "environment": dict(self.environment)}
        expected = args.get("expected_revision")
        if expected is not None and expected != self.revision:
            return None
        self.environment.update(_field(args, "set"))
        self.revision += 1
        return {"revision": self.revision, "environment": dict(self.environment)}

    def _lose(self, error: LinkError) -> None:
        if self._lost is None:
            self._lost = error


class _Invalid(Exception):  # noqa: N818 - the evaluation refuses the frame; the driver answers a rejection.
    pass


def _field(args: Mapping[str, Any], name: str) -> Any:
    """The value the frame carries for ``name``: a frame that omits a field broke the protocol."""
    if name not in args:
        raise LinkFault(f"the frame names no {name}")
    return args[name]


class _DriverDuplex:
    """The socket as the link core's Duplex: frames in, frames out.

    The service's control messages (``bound``, ``unbound``, ``rejected``) travel
    on the same socket outside the frame stream, so the reader takes them for
    the driver and hands the dispatcher only frames.
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
    dispatcher: StandInDispatcher
    bound: Bound
    unbound: asyncio.Event = field(default_factory=asyncio.Event)
    bound_seen: asyncio.Event = field(default_factory=asyncio.Event)
    bound_message: Message | None = None
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
        dispatcher: StandInDispatcher,
        *,
        upstream_url: str,
        plan: InjectionPlan = InjectionPlan(),
        policy: Mapping[str, Any] | None = None,
        host_tools: Sequence[str] = (),
        faces: Sequence[str] = ("mcp",),
        subprotocols: Sequence[str] = (SUBPROTOCOL,),
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
        return cls.attach(websocket, dispatcher, answered)

    @classmethod
    def attach(cls, websocket: ClientConnection, dispatcher: StandInDispatcher, bound: Bound) -> BindingDriver:
        """Serve the frames of a socket whose ``bind`` was answered, or whose answer an adapter read."""
        driver = cls(websocket=websocket, dispatcher=dispatcher, bound=bound)
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
        if message.op == "bound":
            # A driver that attached before its answer read it has the answer here, not in ``open``.
            self.bound_message = message
            self.bound_seen.set()
            return True
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
