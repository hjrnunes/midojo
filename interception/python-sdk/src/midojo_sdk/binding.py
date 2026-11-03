"""The driver end of a remote interceptor binding: one outbound WebSocket per evaluation.

A driver dials out to the interceptor service at its binding stream and never
listens. The socket speaks ``midojo-binding-v1``: the first frame is ``bind``,
answered by ``bound``; the service then sends ``{id, op, args}`` frames and the
driver answers each one. :class:`Binding` hands those frames to the driver's
``LinkDispatcher`` through the link core's ``Duplex`` protocol, keeps the
service's control messages (``unbound``, ``rejected``) for its caller, and
answers ``unbind`` and ``ping``.

Liveness (design section 2.6, M5b plan rule 6). The three values are the
binding's wire contract:

- ``HEARTBEAT_INTERVAL_SECONDS`` (5 s): the driver sends ``ping`` this often.
  It is an application frame, not a WebSocket-level ping, so a library cannot
  answer it while the dispatcher is stuck.
- ``HEARTBEAT_EXPIRY_SECONDS`` (15 s): the service's default. A binding that
  hears nothing from its driver for this long is lost.
- ``ANSWER_DEADLINE_SECONDS`` (10 s): the service's default. An answer past
  this deadline loses the binding, so a hung driver fails the call rather than
  keeping it open.

The adapter sets the client's ``max_size`` explicitly: the library's default
is 1 MiB, below the largest frame the service may send. The sender is one
lock, so the dispatcher's answers and the heartbeat's pings never interleave.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus
from websockets.typing import Subprotocol

from .injection import InjectionPlan
from .link import API, Duplex, LinkFault, Message, decode_message, encode_message

BIND_PATH = "/v1/bindings"
MAX_FRAME_BYTES = 16 * 1024 * 1024
BIND_ANSWER_SECONDS = 10.0

# The three liveness values of the binding's wire contract; see the module docstring.
HEARTBEAT_INTERVAL_SECONDS = 5.0
HEARTBEAT_EXPIRY_SECONDS = 15.0
ANSWER_DEADLINE_SECONDS = 10.0

_SCHEMES = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}


def binding_url(service_url: str) -> str:
    """The binding stream's WebSocket URL, from an http(s) or ws(s) service URL.

    This is the one place the scheme is translated and the path is named: a
    driver is configured with the service's base URL, which may carry a proxy
    prefix, and appends the binding stream to it.
    """
    parts = urlsplit(service_url)
    scheme = _SCHEMES.get(parts.scheme)
    if scheme is None or not parts.hostname:
        raise ValueError(f"the interceptor service URL is an http(s) or ws(s) URL with a host, not {service_url!r}")
    if parts.query or parts.fragment:
        raise ValueError(f"the interceptor service URL carries no query or fragment, not {service_url!r}")
    path = parts.path.rstrip("/")
    if not path.endswith(BIND_PATH):
        path += BIND_PATH
    return urlunsplit((scheme, parts.netloc, path, "", ""))


@dataclass(frozen=True)
class Bound:
    """What ``bound`` told the driver: this binding's endpoint, its own tool token, and the tools it serves."""

    tool_url: str
    tool_token: str
    tools: tuple[str, ...]


class BindingRefusedError(RuntimeError):
    """The service refused a binding: the upgrade's HTTP status, or its ``rejected {kind, detail}``."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


class Binding(Duplex):
    """One bound evaluation's driver end: the socket as a ``Duplex``, and the binding's control messages.

    ``bound`` is the answer to the bind. ``unbound`` is set when the service
    acknowledged the release, and ``injection_outcomes`` is what it carried.
    ``closed`` is set when the socket ends; ``close_code`` and ``close_reason``
    are the service's. A ``rejected`` after a bind is kept for the caller.
    """

    def __init__(self, websocket: ClientConnection, bound: Bound, *, heartbeat_interval: float | None) -> None:
        self.websocket = websocket
        self.bound = bound
        self.unbound = asyncio.Event()
        self.injection_outcomes: tuple[Mapping[str, Any], ...] = ()
        self.rejected: Message | None = None
        self.closed = asyncio.Event()
        self.close_code: int | None = None
        self.close_reason: str | None = None
        self._writes = asyncio.Lock()
        self._tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(self._watch_closed(), name="midojo-binding-close"),
        ]
        if heartbeat_interval is not None:
            self._tasks.append(
                asyncio.create_task(self._heartbeat(heartbeat_interval), name="midojo-binding-heartbeat")
            )

    @classmethod
    async def open(
        cls,
        service_url: str,
        credential: str,
        *,
        upstream_url: str,
        plan: InjectionPlan = InjectionPlan(),
        policy: Mapping[str, Any] | None = None,
        host_tools: Sequence[str] = (),
        faces: Sequence[str] = ("mcp",),
        timeout: float = BIND_ANSWER_SECONDS,
        heartbeat_interval: float | None = HEARTBEAT_INTERVAL_SECONDS,
    ) -> Binding:
        """Dial the service, bind one evaluation, and return its driver end.

        ``heartbeat_interval`` is how often the adapter pings the service;
        ``None`` leaves the heartbeat to the caller, and the service expires
        the binding after its own expiry.
        """
        if type(credential) is not str or not credential:
            raise ValueError("the interceptor service needs a bind credential")
        if heartbeat_interval is not None and (type(heartbeat_interval) not in (int, float) or heartbeat_interval <= 0):
            raise ValueError("the heartbeat interval is a positive number of seconds or None")
        url = binding_url(service_url)
        try:
            websocket = await connect(
                url,
                additional_headers={"authorization": f"Bearer {credential}"},
                subprotocols=[Subprotocol(API)],
                max_size=MAX_FRAME_BYTES,
                proxy=None,
                open_timeout=timeout,
            )
        except InvalidStatus as refused:
            raise BindingRefusedError(
                "unauthorized", f"the service refused the upgrade with HTTP {refused.response.status_code}"
            ) from None
        try:
            await websocket.send(
                encode_message(
                    "bind",
                    api=API,
                    plan=plan.to_data(),
                    policy=None if policy is None else dict(policy),
                    upstream_url=upstream_url,
                    host_tools=list(host_tools),
                    faces=list(faces),
                )
            )
            raw = await asyncio.wait_for(websocket.recv(), timeout)
            message = decode_message(raw.decode() if isinstance(raw, bytes) else raw)
            if message.op == "rejected":
                raise BindingRefusedError(str(message["kind"]), str(message["detail"]))
            if message.op != "bound":
                raise LinkFault(f"bind was answered {message.op!r}")
            try:
                bound = Bound(
                    tool_url=str(message["tool_url"]),
                    tool_token=str(message["tool_token"]),
                    tools=tuple(message["tools"]),
                )
            except (KeyError, TypeError) as broken:
                raise LinkFault(f"bound is not a bound message: {broken}") from None
        except BaseException:
            with contextlib.suppress(Exception):
                await websocket.close()
            raise
        return cls(websocket, bound, heartbeat_interval=heartbeat_interval)

    async def send(self, data: str) -> None:
        """Send one frame on the binding stream, after any answer or ping already in flight."""
        async with self._writes:
            await self.websocket.send(data)

    async def receive(self) -> str:
        """The next frame for the dispatcher: the service's control messages are taken here, not handed on."""
        while True:
            raw = await self.websocket.recv()
            data = raw.decode() if isinstance(raw, bytes) else raw
            if self._control(data):
                continue
            return data

    def close(self) -> None:
        """Close the socket without waiting: the duplex's close, as the dispatcher's serve loop calls it."""
        if self.closed.is_set():
            return
        with contextlib.suppress(RuntimeError):
            task = asyncio.get_running_loop().create_task(self.websocket.close(), name="midojo-binding-hand-close")
            self._tasks.append(task)

    async def ping(self) -> None:
        """The heartbeat: one application frame the service times its expiry against."""
        await self.send(encode_message("ping"))

    async def unbind(self) -> None:
        """Release the binding: the service unmounts, answers ``unbound``, and closes normally."""
        await self.send(encode_message("unbind"))

    async def wait_closed(self) -> None:
        """Wait for the socket to end."""
        await self.closed.wait()

    async def wait_unbound(self) -> None:
        """Wait for the service's ``unbound`` answer to the release."""
        await self.unbound.wait()

    async def aclose(self) -> None:
        """Hang up if the socket is still open, then drop the binding's own end."""
        if not self.closed.is_set():
            with contextlib.suppress(Exception):
                await self.websocket.close()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self.closed.wait(), BIND_ANSWER_SECONDS)
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks = []

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

    async def _heartbeat(self, interval: float) -> None:
        while True:
            try:
                await asyncio.wait_for(self.closed.wait(), interval)
            except TimeoutError:
                with contextlib.suppress(Exception):
                    await self.ping()
            else:
                return

    async def _watch_closed(self) -> None:
        await self.websocket.wait_closed()
        self.close_code = self.websocket.close_code
        self.close_reason = self.websocket.close_reason
        self.closed.set()


__all__ = [
    "ANSWER_DEADLINE_SECONDS",
    "API",
    "BIND_ANSWER_SECONDS",
    "BIND_PATH",
    "HEARTBEAT_EXPIRY_SECONDS",
    "HEARTBEAT_INTERVAL_SECONDS",
    "MAX_FRAME_BYTES",
    "Binding",
    "BindingRefusedError",
    "Bound",
    "binding_url",
]
