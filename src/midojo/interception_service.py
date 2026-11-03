"""The interceptor service: one long-lived listener that binds an evaluation's interceptor over a WebSocket.

The agent runs in its own process, maybe on another host, and its tools are
MiDojo's interceptor rather than the upstream: this service is where that
interceptor runs. It serves the MCP endpoint each agent calls
(``/b/<binding id>/mcp``), holds the upstream connection, and keeps no record
of its own. The driver, the process that owns the evaluation, keeps every
record: each call, delivery fact and environment read crosses one WebSocket
per binding as a ``midojo-binding-v1`` frame, and the service never holds the
session token.

In-process host interception (:mod:`midojo.interception_host`) stays the
default. This service is for an agent that can't share the driver's process.

The stream, per design section 2.6:

- ``GET /v1/bindings`` upgrades with ``Authorization: Bearer <bind
  credential>`` and the subprotocol ``midojo-binding.v1``; the service refuses
  anything else with 403 before ``accept()`` and never reads a credential from
  the query string. The subprotocol's name is not the messages' API name:
  every frame in the stream belongs to ``midojo-binding-v1``, which ``bind``
  names.
- The first frame is ``bind {api, plan, policy, upstream_url, host_tools,
  faces}``: document-form data only. The answer is ``bound {tool_url,
  tool_token, tools}``; the endpoint exists before that answer names it, and
  a request that reaches it before the answer waits for the answer rather
  than being refused, so an agent that already read ``bound`` never gets a
  401.
- ``unbind`` is answered by ``unbound {injection_outcomes}``: every outcome of
  the binding, as ``InjectionOutcome.to_data``.
- ``ping`` is the driver's heartbeat: a binding that hears no frame within the
  expiry is lost.
- A loss (the socket closes, a frame goes unanswered past the deadline, or the
  heartbeat expires) fails every pending request, closes with 1011 and unmounts
  the endpoint, so the binding's tool token gets 401 from then on.

The service runs no host tools in v1: ``host_tools`` is ``[]``, and a driver
that names any gets ``rejected``. ``faces`` is ``["mcp"]``, the only face.

This module needs ``midojo-sdk[mcp]`` and ``uvicorn``: it is the remote
interceptor's image, not the driver.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import secrets
import socket
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import uvicorn
from midojo_sdk.binding import ANSWER_DEADLINE_SECONDS, HEARTBEAT_EXPIRY_SECONDS
from midojo_sdk.injection import CapabilityError, Channel, InjectionPlan
from midojo_sdk.link import (
    API,
    BIND_FIELDS,
    SUBPROTOCOL,
    Link,
    LinkFault,
    LinkLost,
    LinkRecorder,
    LinkWorld,
    Message,
    decode_message,
    encode_message,
)
from midojo_sdk.mcp import MidojoMCP
from starlette.applications import Starlette
from starlette.routing import Mount, WebSocketRoute
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket

from midojo.injection_spec import ToolPolicy, tool_policy
from midojo.interception_host import _WILDCARD_HOSTS, CREDENTIAL_ENV, _Server
from midojo.tool_endpoint import TOOL_PATH, bare_host

BIND_PATH = "/v1/bindings"
"""Where a driver upgrades its binding: ``GET`` with the bind credential and the binding subprotocol."""

FACES_PATH = "/b"
"""Where every binding's faces live: ``/b/<binding id>/mcp`` while that binding exists."""

DEFAULT_PORT = 8767
"""The port the service listens on by default: one fixed port a driver's URL can name."""

DEFAULT_ANSWER_DEADLINE_SECONDS = ANSWER_DEADLINE_SECONDS
"""How long one frame may wait for the driver's answer before the binding is a loss."""

DEFAULT_HEARTBEAT_EXPIRY_SECONDS = HEARTBEAT_EXPIRY_SECONDS
"""How long a binding may hear nothing from its driver before it is a loss."""

DEFAULT_BIND_DEADLINE_SECONDS = 10.0
"""How long a driver may take to send its ``bind`` after the upgrade."""

LOSS_CLOSE_CODE = 1011
"""The close code of a lost binding: an internal failure, which is what an unexplained end is."""

RELEASE_CLOSE_CODE = 1000
"""The close code of a binding its driver released: a normal closure."""

READY_PREFIX = "midojo-intercept-serve listening on"
"""The line the service prints once it is listening: what a launcher waits for, then parses."""

_SHUTDOWN_SECONDS = 2
"""How long stopping waits for open connections before closing them, as the host interceptor does."""

_REASON_LIMIT = 120
"""A close reason's byte-ish budget: the WebSocket reason never exceeds 123 bytes."""


class _Refused(Exception):  # noqa: N818 - the service refuses the request, and the driver reads a rejection.
    """The service refuses a request before (or instead of) a binding: the driver reads ``rejected {kind, detail}``."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


@dataclass(frozen=True)
class _BindRequest:
    """One validated ``bind`` message: the document-form data a binding is mounted from."""

    plan: InjectionPlan
    policy: ToolPolicy | None
    upstream_url: str

    @classmethod
    def of(cls, message: Message) -> _BindRequest:
        """Validate ``message``'s body, or raise :class:`_Refused` naming what is wrong."""
        unknown = set(message.fields) - set(BIND_FIELDS)
        if unknown:
            raise _Refused("invalid_bind", f"bind has unknown fields: {sorted(unknown)}")
        missing = set(BIND_FIELDS) - set(message.fields)
        if missing:
            raise _Refused("invalid_bind", f"bind is missing fields: {sorted(missing)}")
        if message["api"] != API:
            raise _Refused("invalid_bind", f"this service speaks {API}, not {message['api']!r}")
        faces = message["faces"]
        if faces != ["mcp"]:
            raise _Refused("unsupported", f"the only face is ['mcp'], not {faces!r}")
        host_tools = message["host_tools"]
        if not isinstance(host_tools, list) or any(type(name) is not str for name in host_tools):
            raise _Refused("invalid_bind", f"host_tools is a list of names, not {host_tools!r}")
        if host_tools:
            raise _Refused("unsupported", f"this service runs no host tools, so host_tools is [], not {host_tools!r}")
        upstream_url = message["upstream_url"]
        if type(upstream_url) is not str or not upstream_url.startswith(("http://", "https://")):
            raise _Refused("invalid_bind", f"upstream_url is an http(s) URL, not {upstream_url!r}")
        try:
            plan = InjectionPlan.from_data(message["plan"]).for_channels(Channel.TOOL_OUTPUT)
        except (TypeError, ValueError) as exc:
            raise _Refused("invalid_bind", f"plan is not an injection plan: {exc}") from None
        policy_data = message["policy"]
        try:
            policy = None if policy_data is None else ToolPolicy.from_data(policy_data)
        except ValueError as exc:
            raise _Refused("invalid_bind", f"policy is not a tool policy: {exc}") from None
        return cls(plan=plan, policy=policy, upstream_url=upstream_url)


@dataclass(frozen=True)
class _Face:
    """One published binding's endpoint: the ASGI app an agent reaches, behind the binding's own token."""

    id: str
    token: str
    app: ASGIApp
    path: str
    bound: asyncio.Event = field(default_factory=asyncio.Event)
    gone: asyncio.Event = field(default_factory=asyncio.Event)


class _Faces:
    """Every ``/b`` endpoint on the one listener: an id reaches its app only while that binding is published.

    Any other id answers 401, with ``www-authenticate``: after an unbind or a
    loss the endpoint says "no such binding is authorized here" rather than
    404, so a client library reads a refusal, not a server that never had the
    binding. The same 401 answers a published id without its own tool token.

    A published face answers only once the service sent its ``bound``: the
    endpoint exists before the answer names it, so a request that arrives in
    that window waits for the answer rather than reading a 401 it would never
    see again. A binding that ends before its answer refuses the request.
    """

    def __init__(self) -> None:
        self._faces: dict[str, _Face] = {}

    @property
    def ids(self) -> tuple[str, ...]:
        """The published binding ids, in publication order."""
        return tuple(self._faces)

    def publish(self, face: _Face) -> None:
        """Make ``face`` reachable at its path."""
        if face.id in self._faces:
            raise ValueError(f"binding {face.id!r} is already published")
        self._faces[face.id] = face

    def remove(self, binding_id: str) -> None:
        """Stop routing to ``binding_id``: its token gets 401 from now on."""
        face = self._faces.pop(binding_id, None)
        if face is not None:
            face.gone.set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # The binding faces are streamable HTTP MCP, nothing else: an upgrade here gets 403, as uvicorn
            # answers a close before accept.
            await send({"type": "websocket.close", "code": 1008})
            return
        # The mount keeps the path whole and names what it matched in root_path: the route is what's left.
        root = scope.get("root_path", "")
        path = scope["path"]
        route = path[len(root) :] if root and path.startswith(root) else path
        binding_id, _, rest = route.lstrip("/").partition("/")
        face = self._faces.get(binding_id)
        if face is None or not rest:
            await _refuse(send, 401, b"This binding's bearer token is required")
            return
        await _named(face)
        # A binding that ended takes precedence over the answer it sent: the answer can go out to a
        # driver that already went, and its app is on its way out with it. An ended binding is a 401.
        if face.gone.is_set() or not face.bound.is_set():
            await _refuse(send, 401, b"This binding's bearer token is required")
            return
        child_scope = dict(scope, path="/" + rest, root_path=root + "/" + binding_id)
        await _authorized(face.app, face.token)(child_scope, receive, send)


async def _named(face: _Face) -> None:
    """Wait until ``face`` has been named to its driver, or until the binding went.

    A face that is removed sets ``gone``, so a held request never outlives its binding.
    """
    if face.bound.is_set() or face.gone.is_set():
        return
    waiting = [
        asyncio.create_task(face.bound.wait(), name="midojo-binding-bound"),
        asyncio.create_task(face.gone.wait(), name="midojo-binding-gone"),
    ]
    try:
        await asyncio.wait(waiting, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for task in waiting:
            task.cancel()
        await asyncio.gather(*waiting, return_exceptions=True)


def _authorized(app: ASGIApp, token: str) -> ASGIApp:
    """``app`` behind this binding's bearer token, refusing every other credential with 401."""
    expected = f"Bearer {token}".encode()

    async def check(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await send({"type": "websocket.close", "code": 1008})
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


class _BindingStream:
    """The service's end of one binding socket: the one sender queue, the reader, and its liveness clock.

    Every frame the link sends goes through the queue, one at a time, so two
    concurrent tool calls never interleave on the socket. The reader hands the
    link the driver's answers and keeps the control messages for the binding:
    a ``ping`` restarts the heartbeat, and an ``unbind`` releases the binding.
    """

    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._out: asyncio.Queue[tuple[str, asyncio.Future[None]] | None] = asyncio.Queue()
        self._in: asyncio.Queue[str | None] = asyncio.Queue()
        self._released = asyncio.Event()
        self._closed = asyncio.Event()
        self._tasks: list[asyncio.Task[None]] = []
        self.last_seen = time.monotonic()
        self.failure: BaseException | None = None

    @property
    def released(self) -> asyncio.Event:
        """Set when the driver sent ``unbind``."""
        return self._released

    @property
    def closed(self) -> asyncio.Event:
        """Set when the socket ended, from either side."""
        return self._closed

    @property
    def detail(self) -> str:
        """What ended the socket, for the outcome a binding ends with."""
        return "the connection closed" if self.failure is None else f"the connection ended: {self.failure}"

    def start(self) -> None:
        """Start reading and writing on the socket."""
        self._tasks = [
            asyncio.create_task(self._read_frames(), name="midojo-binding-reader"),
            asyncio.create_task(self._write_frames(), name="midojo-binding-sender"),
        ]

    # --- the link's Duplex ---

    async def send(self, data: str) -> None:
        """Queue one frame and wait until the socket took it."""
        if self._closed.is_set():
            raise LinkLost("the binding stream is closed")
        delivered: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._out.put((data, delivered))
        await delivered

    async def receive(self) -> str:
        """The next frame the driver sent, waiting until one arrives."""
        data = await self._in.get()
        if data is None:
            raise LinkLost(self.detail)
        return data

    def close(self) -> None:
        """End the stream now: no frame is read or written again, and every waiting one fails."""
        self._end()

    async def aclose(self) -> None:
        """End the stream and wait for its tasks, for a binding that is done."""
        self._end()
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []

    def _end(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._in.put_nowait(None)
        self._out.put_nowait(None)
        while not self._out.empty():
            item = self._out.get_nowait()
            if item is not None and not item[1].done():
                item[1].set_exception(LinkLost("the binding stream is closed"))

    # --- the socket ---

    async def _read_frames(self) -> None:
        try:
            while True:
                raw = await self._websocket.receive_text()
                self.last_seen = time.monotonic()
                try:
                    message = decode_message(raw)
                except LinkFault:
                    self._in.put_nowait(raw)
                    continue
                if message.op == "ping":
                    continue
                if message.op == "unbind":
                    self._released.set()
                    continue
                # Nothing else is a driver-to-service control message: the link refuses it as a fault.
                self._in.put_nowait(raw)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # WebSocketDisconnect, and the runtime error after a close
            self.failure = exc
        finally:
            self._closed.set()
            self._in.put_nowait(None)

    async def _write_frames(self) -> None:
        while True:
            item = await self._out.get()
            if item is None:
                return
            data, delivered = item
            try:
                await self._websocket.send_text(data)
            except Exception as exc:
                if not delivered.done():
                    delivered.set_exception(LinkLost(f"the binding socket couldn't take a frame: {exc}"))
                self.failure = exc
                self._closed.set()
                self._in.put_nowait(None)
                return
            if not delivered.done():
                delivered.set_result(None)


@dataclass(frozen=True)
class Outcome:
    """How one binding ended: released by its driver, or lost, with the detail that ended it."""

    released: bool
    detail: str


@dataclass
class _Mounted:
    """One mounted binding: the app in the table, its lifespan task, and what a release must report."""

    face: _Face
    mcp: MidojoMCP
    tools: tuple[str, ...]
    task: asyncio.Task[None]
    stop: asyncio.Event
    _unmounted: bool = False

    async def unmount(self) -> None:
        """Leave the app's lifespan, as the mount entered it, once."""
        if self._unmounted:
            return
        self._unmounted = True
        self.stop.set()
        await self.task


class InterceptorService:
    """The remote interceptor service: one listener, one MCP endpoint per binding, one socket per binding.

    ``credential`` is the bind credential every driver must present: in v1 the
    service starts with it (``--credential`` or ``MIDOJO_INTERCEPTOR_CREDENTIAL``)
    and the driver reads the same one. ``answer_deadline`` and
    ``heartbeat_expiry`` are the liveness bounds the design fixes at about 10 s
    and 15 s; tests inject short ones. ``advertise_host`` is the host a driver
    reaches the service at, when it differs from the bind host.
    """

    def __init__(
        self,
        credential: str,
        *,
        answer_deadline: float = DEFAULT_ANSWER_DEADLINE_SECONDS,
        heartbeat_expiry: float = DEFAULT_HEARTBEAT_EXPIRY_SECONDS,
        bind_deadline: float = DEFAULT_BIND_DEADLINE_SECONDS,
        advertise_host: str | None = None,
        name: str = "midojo-intercept",
    ) -> None:
        if type(credential) is not str or not credential:
            raise ValueError("the interceptor service needs a bind credential")
        for what, value in (
            ("answer_deadline", answer_deadline),
            ("heartbeat_expiry", heartbeat_expiry),
            ("bind_deadline", bind_deadline),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{what} must be a positive number of seconds, not {value!r}")
        self._credential = credential
        self._answer_deadline = float(answer_deadline)
        self._heartbeat_expiry = float(heartbeat_expiry)
        self._bind_deadline = float(bind_deadline)
        self._advertise_host = bare_host(advertise_host) if advertise_host else None
        self._name = name
        self._faces = _Faces()
        self._listener: socket.socket | None = None
        self._server: _Server | None = None
        self._serving: asyncio.Task[None] | None = None
        self._base_url: str | None = None

    @property
    def url(self) -> str | None:
        """The base URL a driver reaches this service at, once it listens."""
        return self._base_url

    @property
    def published(self) -> tuple[str, ...]:
        """The binding ids whose endpoints exist right now."""
        return self._faces.ids

    def app(self) -> Starlette:
        """The one listener's app: the binding upgrade, and the faces of the bindings that exist."""
        return Starlette(
            routes=[
                WebSocketRoute(BIND_PATH, self._bind),
                Mount(FACES_PATH, app=self._faces),
            ]
        )

    async def start(self, *, host: str = "127.0.0.1", port: int = 0) -> str:
        """Listen on ``host`` and ``port`` (0 picks a free one) and return the driver's base URL."""
        bind_host = bare_host(host)
        advertised = self._advertise_host or ("127.0.0.1" if bind_host in _WILDCARD_HOSTS else bind_host)
        listener = _listen(host, port)
        self._listener = listener
        self._base_url = f"http://{advertised}:{listener.getsockname()[1]}"
        config = uvicorn.Config(
            self.app(),
            log_level="warning",
            lifespan="on",
            ws="websockets-sansio",
            timeout_graceful_shutdown=_SHUTDOWN_SECONDS,
        )
        server = self._server = _Server(config)
        serving = self._serving = asyncio.create_task(server.serve(sockets=[listener]), name="midojo-intercept-serve")
        while not server.started:
            if serving.done():
                listener.close()
                raise RuntimeError(f"the interceptor service stopped before serving on {host}:{port}")
            await asyncio.sleep(0.01)
        return self._base_url

    async def stop(self) -> None:
        """Stop listening. Open bindings get the short grace period, then are closed."""
        if self._server is not None and self._serving is not None:
            self._server.should_exit = True
            await self._serving
        self._server = None
        self._serving = None
        self._listener = None
        self._base_url = None

    async def run(self, *, host: str = "127.0.0.1", port: int = DEFAULT_PORT) -> None:
        """Listen and serve until the process stops, printing the driver's URL first."""
        base = await self.start(host=host, port=port)
        # The ready line is the interface: a launcher waits for it, then parses the driver's URL from it.
        print(f"{READY_PREFIX} {base}{BIND_PATH}", flush=True)
        serving = self._serving
        assert serving is not None
        await serving

    # --- one binding ---

    async def _bind(self, websocket: WebSocket) -> None:
        """The ``GET /v1/bindings`` upgrade: authenticate it, read the bind, mount the endpoint, then serve."""
        if not self._upgrade_authorized(websocket):
            # A close before accept is how this listener refuses the upgrade: uvicorn answers 403.
            await websocket.close()
            return
        await websocket.accept(subprotocol=SUBPROTOCOL)
        request = await self._read_bind(websocket)
        if request is None:
            return
        stream = _BindingStream(websocket)
        link = Link(stream, answer_deadline=self._answer_deadline)
        try:
            mounted = await self._mount(request, link)
        except _Refused as refused:
            await self._reject(websocket, refused)
            return
        # The one sender queue takes every frame from here on: ``bound`` first, then the link's.
        stream.start()
        try:
            await self._serve(websocket, stream, link, mounted)
        finally:
            self._faces.remove(mounted.face.id)
            await mounted.unmount()

    async def _serve(self, websocket: WebSocket, stream: _BindingStream, link: Link, mounted: _Mounted) -> None:
        """Answer ``bound``, then serve the binding until its driver releases it or its link is lost."""
        try:
            await stream.send(self._bound(mounted))
        except LinkLost:
            pass  # the socket went before the answer did; the outcome below ends the binding as a loss
        else:
            # The endpoint answers from here on: a request that waited before the answer goes through now.
            mounted.face.bound.set()
        outcome = await self._outcome(stream, link)
        if outcome.released:
            # Stop routing before leaving the lifespan: a request in that window must not reach a session
            # manager that is shutting down.
            self._faces.remove(mounted.face.id)
            await mounted.unmount()
            outcomes = [record.to_data() for record in mounted.mcp.injection_outcomes]
            with contextlib.suppress(Exception):
                await stream.send(encode_message("unbound", injection_outcomes=outcomes))
            await stream.aclose()
            await _close(websocket, RELEASE_CLOSE_CODE, outcome.detail)
        else:
            link.close(outcome.detail)  # every pending request fails at once, so no call hangs on the loss
            self._faces.remove(mounted.face.id)
            await mounted.unmount()
            await stream.aclose()
            await _close(websocket, LOSS_CLOSE_CODE, outcome.detail)

    def _bound(self, mounted: _Mounted) -> str:
        """The answer to ``bind``: where this binding's endpoint is, its own tool token, and the tools it serves."""
        assert self._base_url is not None
        return encode_message(
            "bound",
            tool_url=f"{self._base_url}{mounted.face.path}",
            tool_token=mounted.face.token,
            tools=list(mounted.tools),
        )

    async def _outcome(self, stream: _BindingStream, link: Link) -> Outcome:
        """Wait for the binding to end: its driver released it, or its link was lost."""
        watch = [
            asyncio.create_task(stream.released.wait(), name="midojo-binding-released"),
            asyncio.create_task(stream.closed.wait(), name="midojo-binding-closed"),
            asyncio.create_task(link.wait_lost(), name="midojo-binding-lost"),
            asyncio.create_task(self._expired(stream), name="midojo-binding-heartbeat"),
        ]
        try:
            await asyncio.wait(watch, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for task in watch:
                task.cancel()
        if stream.released.is_set():
            return Outcome(True, "the driver released the binding")
        if stream.closed.is_set() and not link.lost:
            return Outcome(False, stream.detail)
        if link.lost is not None:
            return Outcome(False, link.lost.detail)
        return Outcome(False, f"no heartbeat within {self._heartbeat_expiry:g}s")

    async def _expired(self, stream: _BindingStream) -> None:
        """Return once the binding heard nothing from its driver for the heartbeat expiry."""
        while True:
            remaining = self._heartbeat_expiry - (time.monotonic() - stream.last_seen)
            if remaining <= 0:
                return
            await asyncio.sleep(remaining)

    def _upgrade_authorized(self, websocket: WebSocket) -> bool:
        """Whether the upgrade carries the bind credential and asks for the binding subprotocol."""
        authorization = websocket.headers.get("authorization", "")
        expected = f"Bearer {self._credential}"
        asked = [name.strip() for name in websocket.headers.get("sec-websocket-protocol", "").split(",")]
        return secrets.compare_digest(authorization, expected) and SUBPROTOCOL in asked

    async def _read_bind(self, websocket: WebSocket) -> _BindRequest | None:
        """Read the first frame and validate the ``bind`` it must hold; refuse and close when it is not one."""
        try:
            raw = await asyncio.wait_for(websocket.receive_text(), self._bind_deadline)
        except TimeoutError:
            await self._reject(websocket, _Refused("invalid_bind", "no bind frame arrived"))
            return None
        except Exception as exc:  # the driver closed before binding
            await _close(websocket, 1008, f"the binding ended before bind: {exc}")
            return None
        try:
            message = decode_message(raw)
        except LinkFault as fault:
            await self._reject(websocket, _Refused("invalid_bind", str(fault)))
            return None
        if message.op != "bind":
            await self._reject(websocket, _Refused("invalid_bind", f"the first frame is bind, not {message.op!r}"))
            return None
        try:
            return _BindRequest.of(message)
        except _Refused as refused:
            await self._reject(websocket, refused)
            return None

    async def _reject(self, websocket: WebSocket, refused: _Refused) -> None:
        """Answer a refused request with ``rejected``, then close the socket that never became a binding."""
        with contextlib.suppress(Exception):
            await websocket.send_text(encode_message("rejected", kind=refused.kind, detail=refused.detail))
        await _close(websocket, 1008, refused.detail)

    async def _mount(self, request: _BindRequest, link: Link) -> _Mounted:
        """Build the binding's interceptor, enter its lifespan and publish its endpoint.

        The endpoint is published before ``bound`` names it, because the answer
        carries its URL and token; it answers nothing until the answer was sent
        (``_Faces`` holds a request until then).
        """
        binding_id = secrets.token_urlsafe(12)
        token = secrets.token_urlsafe(32)
        allowed, blocked = tool_policy(request.policy)
        mcp = MidojoMCP(
            f"{self._name}-{binding_id}",
            recorder=LinkRecorder(link),
            world=LinkWorld(link),
            upstream_url=request.upstream_url,
            plan=request.plan,
            evaluation_id=binding_id,
        )
        mcp.block(*blocked)
        try:
            tools = await mcp.mirror_upstream(allowed=allowed)
        except CapabilityError as exc:
            raise _Refused("invalid_plan", str(exc)) from None
        except Exception as exc:
            raise _Refused("upstream_unavailable", f"the upstream could not be mirrored: {exc}") from None
        child = mcp.http_app(path=TOOL_PATH)
        ready, stop = asyncio.Event(), asyncio.Event()

        async def lifespan() -> None:
            try:
                async with child.lifespan(child):
                    ready.set()
                    await stop.wait()
            finally:
                # A lifespan that failed to enter must not leave the mount waiting on it.
                ready.set()

        task = asyncio.create_task(lifespan(), name=f"midojo-binding {binding_id}")
        await ready.wait()
        if task.done():
            raise _Refused("upstream_unavailable", f"the binding's app did not start: {task.exception()}")
        assert self._base_url is not None
        face = _Face(
            id=binding_id,
            token=token,
            app=_authorized(child, token),
            path=f"{FACES_PATH}/{binding_id}{TOOL_PATH}",
        )
        mounted = _Mounted(face=face, mcp=mcp, tools=tuple(tools), task=task, stop=stop)
        self._faces.publish(face)
        return mounted


async def _close(websocket: WebSocket, code: int, reason: str) -> None:
    """Close the socket with ``code``, once: a peer that already left is not an error."""
    with contextlib.suppress(Exception):
        await websocket.close(code=code, reason=reason[:_REASON_LIMIT])


def _listen(host: str, port: int) -> socket.socket:
    family, kind, proto, _, address = socket.getaddrinfo(
        host or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )[0]
    listener = socket.socket(family, kind, proto)
    try:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(address)
    except OSError:
        listener.close()
        raise
    return listener


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="midojo-intercept-serve",
        description="Serve the remote interceptor: one MCP endpoint per bound evaluation, over one WebSocket each.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="the address to listen on (default: 127.0.0.1)")
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help=f"the port to listen on (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--advertise-host",
        default=None,
        help="the host a driver reaches the service at, when it differs from --host",
    )
    parser.add_argument(
        "--credential",
        default=None,
        help=f"the bind credential every driver presents (default: ${CREDENTIAL_ENV})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the service from the command line: listen, print the driver's URL, and serve until stopped."""
    args = _parser().parse_args(argv)
    credential = args.credential or os.environ.get(CREDENTIAL_ENV, "")
    if not credential:
        _parser().error(f"a bind credential is required: --credential or {CREDENTIAL_ENV}")
    service = InterceptorService(credential, advertise_host=args.advertise_host)
    asyncio.run(service.run(host=args.host, port=args.port))
    return 0


__all__ = [
    "BIND_PATH",
    "CREDENTIAL_ENV",
    "DEFAULT_ANSWER_DEADLINE_SECONDS",
    "DEFAULT_HEARTBEAT_EXPIRY_SECONDS",
    "DEFAULT_PORT",
    "FACES_PATH",
    "LOSS_CLOSE_CODE",
    "READY_PREFIX",
    "SUBPROTOCOL",
    "InterceptorService",
    "Outcome",
    "main",
]
