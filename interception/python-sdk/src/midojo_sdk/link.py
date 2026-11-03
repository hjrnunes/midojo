"""The ``midojo-binding-v1`` frame codec and the service side of a remote interception link.

An interception layer that runs in its own process records through frames
instead of HTTP: it sends ``{"id", "op", "args"}`` to the driver and awaits the
answer, ``{"re", "ok"}`` or ``{"re", "rejected": {"kind", "detail",
"late_fact"}}``. :class:`LinkRecorder` and :class:`LinkWorld` are that side of
the stream: a :class:`~midojo_sdk.recorder.Recorder` (and
:class:`~midojo_sdk.recorder.DetailedRecorder` and
:class:`~midojo_sdk.recorder.CaptureReporter`) and a
:class:`~midojo_sdk.recorder.WorldState` whose every call is one frame.

This module imports only the standard library and ``midojo_sdk`` itself: the
interception layer runs beside the agent, which must not need MiDojo's
driver-side packages. The transport is whatever holds the two ends of the
stream (a WebSocket in production, an in-memory duplex in tests): it only has
to send and receive text frames.

A link is one binding's stream, and it never comes back. A transport error, a
frame the codec refuses, or an answer that matches no request is a *loss*: the
requests in flight fail with :class:`LinkLost` (reason ``link-lost``), every
later request fails at once, and the driver seals the evaluation it was
recording for as incomplete.

Around the frames, one binding exchanges control messages: the driver
``bind``s it and the service answers ``bound``, the driver ``unbind``s it and
the service answers ``unbound``, the driver ``ping``s it, and either side may
``reject`` what it can't accept. :class:`Message` is their envelope, and each
side validates the body its op names: only the two sides of one binding know
one.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol

from .recorder import (
    CAPTURE_REASON_MAX_LENGTH,
    CapturePhase,
    ControlPlaneRejectedError,
    Disposition,
    EnvironmentRevision,
    InjectionRecord,
    InvalidRequestError,
    RejectionKind,
)

API = "midojo-binding-v1"
"""The binding stream's protocol name: what a driver and a service must agree on to talk."""

LINK_LOST_REASON = "link-lost"
"""Why a handle ends when its binding stream is lost: ``HandleError.reason`` on the driver side."""

INVALID_REQUEST = "invalid_request"
"""The rejection kind for a frame the control plane can't validate: the 422 the HTTP route answers.

It is not a ``RejectionKind``: nothing was decided about the evaluation, so the
service raises ``InvalidRequestError``, as the in-process client does.
"""

OPS = (
    "start_call",
    "finish_call_detailed",
    "report_capture_failure",
    "env_read",
    "env_change",
)
"""The ops a service sends: the recorder's and the world's whole surface, one frame each."""


class LinkError(RuntimeError):
    """The binding stream ended. Every subclass names the handle reason in ``reason``.

    ``detail`` says what happened. A link ends for a transport that went away
    or for a peer that broke the protocol, and a service can't tell the two
    apart from outside: both are final, and both are ``link-lost`` to a handle.
    """

    reason = LINK_LOST_REASON

    def __init__(self, detail: str = "the binding stream was lost") -> None:
        super().__init__(f"{self.reason}: {detail}")
        self.detail = detail


class LinkFault(LinkError):  # noqa: N818 - the frame protocol names the fault, not a lint choice.
    """The peer broke the protocol: a malformed frame, a gap or a repeat in the ids, an answer no one awaits."""


class LinkLost(LinkError, ConnectionError):  # noqa: N818 - the loss names itself, and a caller reads ``lost``.
    """The binding stream is gone: the transport failed, or the peer ended it.

    ``detail`` says what happened. The loss is terminal for the binding: the
    requests in flight fail with it, and so does every later one.
    """


class Duplex(Protocol):
    """One end of the binding stream: text frames in, text frames out.

    The driver's end is the same protocol, read the other way. A transport
    that closes raises its own error from :meth:`receive` and :meth:`send`;
    the link reads any such error as a loss.
    """

    async def send(self, data: str) -> None:
        """Send one frame, waiting until the transport takes it."""
        ...

    async def receive(self) -> str:
        """The next frame, waiting until one arrives; raises when the stream ends."""
        ...

    def close(self) -> None:
        """End the stream: nothing more is sent or received, and the peer reads the loss."""
        ...


# --- the codec ---


@dataclass(frozen=True)
class Frame:
    """One service-to-driver request: ``{"id": id, "op": op, "args": args}``.

    ``id`` counts the binding's requests from zero and never repeats, so the
    driver applies frames in id order and answers each by its ``id``.
    """

    id: int
    op: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Rejection:
    """Why the driver refused a frame: the control plane's kind, detail and late-fact flag."""

    kind: str
    detail: str
    late_fact: bool = False


@dataclass(frozen=True)
class Answer:
    """One driver-to-service answer: ``{"re": id, "ok": result}`` or ``{"re": id, "rejected": {...}}``.

    ``ok`` is ``None`` for a request whose answer is nothing: an applied record,
    or an ``env_change`` at a revision other than the one it expected.
    """

    re: int
    ok: Any = None
    rejected: Rejection | None = None


def _is_id(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def encode_frame(frame: Frame) -> str:
    """Frame ``frame`` as one text frame: compact JSON, its keys in a fixed order."""
    return json.dumps({"id": frame.id, "op": frame.op, "args": frame.args}, separators=(",", ":"), sort_keys=True)


def decode_frame(data: str) -> Frame:
    """The frame ``data`` holds, or :class:`LinkFault` when it isn't one."""
    try:
        document = json.loads(data)
    except (TypeError, ValueError) as e:
        raise LinkFault(f"the frame is not JSON: {e}") from e
    if not isinstance(document, dict):
        raise LinkFault(f"a frame is a JSON object, not {type(document).__name__}")
    if set(document) != {"id", "op", "args"}:
        raise LinkFault(f"a frame holds exactly id, op and args, not {sorted(document)}")
    frame_id, op, args = document["id"], document["op"], document["args"]
    if not _is_id(frame_id):
        raise LinkFault(f"a frame id is a non-negative int, not {frame_id!r}")
    if not isinstance(op, str) or op not in OPS:
        raise LinkFault(f"unknown op {op!r}: the binding stream has {list(OPS)}")
    if not isinstance(args, dict):
        raise LinkFault(f"a frame's args are an object, not {type(args).__name__}")
    return Frame(frame_id, op, args)


def encode_answer(answer: Answer) -> str:
    """Answer ``answer`` as one text frame."""
    if answer.rejected is not None:
        body: dict[str, Any] = {
            "re": answer.re,
            "rejected": {
                "kind": answer.rejected.kind,
                "detail": answer.rejected.detail,
                "late_fact": answer.rejected.late_fact,
            },
        }
    else:
        body = {"re": answer.re, "ok": answer.ok}
    return json.dumps(body, separators=(",", ":"), sort_keys=True)


def decode_answer(data: str) -> Answer:
    """The answer ``data`` holds, or :class:`LinkFault` when it isn't one."""
    try:
        document = json.loads(data)
    except (TypeError, ValueError) as e:
        raise LinkFault(f"the answer is not JSON: {e}") from e
    if not isinstance(document, dict) or "re" not in document:
        raise LinkFault("an answer names the frame it answers in re")
    has_ok, has_rejected = "ok" in document, "rejected" in document
    if has_ok == has_rejected or not set(document) <= {"re", "ok", "rejected"}:
        raise LinkFault(f"an answer is ok or rejected, not both or neither: {sorted(document)}")
    re = document["re"]
    if not _is_id(re):
        raise LinkFault(f"an answer's re is a non-negative int, not {re!r}")
    if has_ok:
        return Answer(re, ok=document["ok"])
    rejected = document["rejected"]
    if not isinstance(rejected, dict) or set(rejected) != {"kind", "detail", "late_fact"}:
        raise LinkFault(f"a rejection holds exactly kind, detail and late_fact, not {sorted(rejected)}")
    if not isinstance(rejected["kind"], str) or not isinstance(rejected["detail"], str):
        raise LinkFault("a rejection's kind and detail are strings")
    if not isinstance(rejected["late_fact"], bool):
        raise LinkFault("a rejection's late_fact is a bool")
    return Answer(re, rejected=Rejection(rejected["kind"], rejected["detail"], rejected["late_fact"]))


def rejection_error(rejected: Rejection) -> ControlPlaneRejectedError | InvalidRequestError:
    """The SDK error a rejection is: the refusal's kind, detail and late-fact flag, or the invalid request."""
    if rejected.kind == INVALID_REQUEST:
        return InvalidRequestError(rejected.detail)
    try:
        kind = RejectionKind(rejected.kind)
    except ValueError as e:
        raise LinkFault(f"the driver used an unknown rejection kind {rejected.kind!r}") from e
    return ControlPlaneRejectedError(kind, rejected.detail, late_fact=rejected.late_fact)


# --- the binding control messages ---


MESSAGE_OPS = ("bind", "bound", "unbind", "unbound", "ping", "rejected")
"""The control messages a driver and a service exchange around the frames.

``bind`` (driver), ``bound``, ``unbind`` (driver), ``unbound``, ``ping``
(driver) and ``rejected`` are each one text frame; the two sides of one
binding are the only ones that know a message's body, so the codec checks the
envelope and leaves each body to the side that reads it.
"""

BIND_FIELDS = ("api", "plan", "policy", "upstream_url", "host_tools", "faces")
"""What a ``bind`` message names: the version, the plan, the tool policy, the upstream and the faces served."""

BOUND_FIELDS = ("tool_url", "tool_token", "tools")
"""What a ``bound`` message names: the agent's tool endpoint, its token, and the tools the binding serves."""

UNBOUND_FIELDS = ("injection_outcomes",)
"""What an ``unbound`` message names: every outcome of the released binding, as ``InjectionOutcome.to_data``."""


@dataclass(frozen=True)
class Message:
    """One control message: ``{"op": op, **fields}``, its fields the body its op names."""

    op: str
    fields: Mapping[str, Any]

    def __getitem__(self, name: str) -> Any:
        return self.fields[name]

    def __contains__(self, name: str) -> bool:
        return name in self.fields


def encode_message(op: str, **fields: Any) -> str:
    """Control message ``op`` with ``fields`` as one text frame."""
    if op not in MESSAGE_OPS:
        raise ValueError(f"unknown control message op {op!r}: the binding stream has {list(MESSAGE_OPS)}")
    return json.dumps({"op": op, **fields}, separators=(",", ":"), sort_keys=True)


def decode_message(data: str) -> Message:
    """The control message ``data`` holds, or :class:`LinkFault` when it isn't one."""
    try:
        document = json.loads(data)
    except (TypeError, ValueError) as e:
        raise LinkFault(f"the control message is not JSON: {e}") from e
    if not isinstance(document, dict):
        raise LinkFault(f"a control message is a JSON object, not {type(document).__name__}")
    op = document.get("op")
    if not isinstance(op, str) or op not in MESSAGE_OPS:
        raise LinkFault(f"unknown control message op {op!r}: the binding stream has {list(MESSAGE_OPS)}")
    return Message(op, {name: value for name, value in document.items() if name != "op"})


# --- the service side of one binding ---


class Link:
    """One binding's request stream: ids, answers and the loss that ends it.

    A link sends each request once and waits for its answer. Requests may be
    in flight together: each names its own id, and the driver answers them in
    the order it applies them. The link never resends anything, and it never
    comes back after a loss: a :class:`RetryingRecorder` around it would only
    replay a request the driver may already have applied.

    ``answer_deadline`` is how long one frame may wait for its answer before
    the link is lost: a driver that stopped answering ends the binding instead
    of hanging the call. ``None`` waits for as long as the stream lives.
    """

    def __init__(self, duplex: Duplex, *, answer_deadline: float | None = None) -> None:
        if answer_deadline is not None and (
            isinstance(answer_deadline, bool) or not isinstance(answer_deadline, int | float) or answer_deadline <= 0
        ):
            raise ValueError(f"answer_deadline must be a positive number of seconds or None, not {answer_deadline!r}")
        self._duplex = duplex
        self._answer_deadline = None if answer_deadline is None else float(answer_deadline)
        self._next_id = 0
        self._pending: dict[int, asyncio.Future[Any]] = {}
        # Requests the caller abandoned (a cancelled tool call): their answers arrive to no one, and are dropped.
        self._abandoned: set[int] = set()
        self._reader: asyncio.Task[None] | None = None
        self._lost: LinkError | None = None
        self._lost_event = asyncio.Event()

    @property
    def lost(self) -> LinkError | None:
        """The loss that ended the link, or None while it's live: a transport that went, or a frame the peer broke."""
        return self._lost

    async def wait_lost(self) -> LinkError:
        """The loss that ended the link, waiting for it while the link is still live."""
        await self._lost_event.wait()
        assert self._lost is not None
        return self._lost

    async def request(self, op: str, args: Mapping[str, Any] | None = None) -> Any:
        """Send one frame and return its answer's result, or raise its rejection or the link's loss."""
        if op not in OPS:
            raise ValueError(f"unknown op {op!r}: the binding stream has {list(OPS)}")
        if self._lost is not None:
            raise self._lost
        self._ensure_reader()
        frame_id = self._next_id
        self._next_id += 1
        answer: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[frame_id] = answer
        try:
            await self._duplex.send(encode_frame(Frame(frame_id, op, dict(args or {}))))
        except asyncio.CancelledError:
            self._pending.pop(frame_id, None)
            self._lose(LinkLost(f"frame {frame_id} was cancelled before it was sent"))
            raise
        except Exception as e:
            self._pending.pop(frame_id, None)
            lost = LinkLost(f"frame {frame_id} couldn't be sent: {e}")
            self._lose(lost)
            raise lost from e
        try:
            return await self._answered(frame_id, answer)
        except asyncio.CancelledError:
            self._abandoned.add(frame_id)
            raise
        finally:
            self._pending.pop(frame_id, None)

    async def _answered(self, frame_id: int, answer: asyncio.Future[Any]) -> Any:
        """Wait for ``answer``, within the deadline when the link has one; a deadline that passes loses the link."""
        if self._answer_deadline is None:
            return await answer
        try:
            return await asyncio.wait_for(answer, self._answer_deadline)
        except TimeoutError:
            self._lose(LinkLost(f"frame {frame_id} waited more than {self._answer_deadline:g}s for its answer"))
            # Raise the link's own loss: two requests that timed out together fail as one fact, not two.
            lost = self._lost
            assert lost is not None
            raise lost from None

    def close(self, detail: str = "the service closed the binding") -> None:
        """End the link and its stream: the requests in flight fail, and so does every later one."""
        self._lose(LinkLost(detail))
        with suppress(Exception):
            self._duplex.close()

    async def aclose(self) -> None:
        """End the link and stop its reader, for a service that is done with the binding."""
        self.close("the link was closed")
        reader = self._reader
        if reader is None or reader.done():
            return
        reader.cancel()
        with suppress(asyncio.CancelledError):
            await reader

    def _ensure_reader(self) -> None:
        if self._reader is None:
            self._reader = asyncio.get_running_loop().create_task(self._read_answers())

    async def _read_answers(self) -> None:
        while self._lost is None:
            try:
                data = await self._duplex.receive()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._lose(LinkLost(f"the binding stream ended: {e}"))
                return
            try:
                decoded = decode_answer(data)
            except LinkFault as fault:
                self._lose(fault)
                return
            if decoded.re in self._abandoned:
                self._abandoned.discard(decoded.re)
                continue
            answer = self._pending.get(decoded.re)
            if answer is None or answer.done():
                self._lose(LinkFault(f"the answer to frame {decoded.re} matches no request in flight"))
                return
            if decoded.rejected is None:
                answer.set_result(decoded.ok)
                continue
            try:
                answer.set_exception(rejection_error(decoded.rejected))
            except LinkFault as fault:
                self._lose(fault)
                return

    def _lose(self, error: LinkError) -> None:
        if self._lost is None:
            self._lost = error
        self._lost_event.set()
        for answer in self._pending.values():
            if not answer.done():
                answer.set_exception(self._lost)
        self._pending.clear()


class LinkRecorder:
    """A :class:`~midojo_sdk.recorder.Recorder` whose every call is one frame on a :class:`Link`.

    It records a finish's details, so :func:`~midojo_sdk.recorder.records_details`
    says so and an interception layer never falls back to the text-only finish.
    A refusal raises :class:`~midojo_sdk.recorder.ControlPlaneRejectedError`
    with the driver's kind, detail and late-fact flag; a lost link raises
    :class:`LinkLost`, which is final: the frame is not resent.
    """

    def __init__(self, link: Link) -> None:
        self._link = link

    @property
    def lost(self) -> LinkError | None:
        """The loss that ended the link, or None while it's live."""
        return self._link.lost

    async def start_call(self, *, call_id: str, function: str, args: dict[str, Any], disposition: Disposition) -> None:
        await self._link.request(
            "start_call", {"call_id": call_id, "function": function, "args": args, "disposition": disposition}
        )

    async def finish_call(
        self,
        *,
        call_id: str,
        function: str,
        args: dict[str, Any],
        result: str,
        disposition: Disposition,
        error: str | None = None,
    ) -> None:
        await self._link.request(
            "finish_call_detailed",
            {
                "call_id": call_id,
                "function": function,
                "args": args,
                "result": result,
                "disposition": disposition,
                "error": error,
            },
        )

    async def finish_call_detailed(
        self,
        *,
        call_id: str,
        function: str,
        args: dict[str, Any],
        result: str,
        disposition: Disposition,
        error: str | None = None,
        structured_content: dict[str, Any] | None,
        injection_outcomes: Sequence[InjectionRecord],
    ) -> None:
        await self._link.request(
            "finish_call_detailed",
            {
                "call_id": call_id,
                "function": function,
                "args": args,
                "result": result,
                "disposition": disposition,
                "error": error,
                "structured_content": structured_content,
                "injection_outcomes": list(injection_outcomes),
            },
        )

    async def report_capture_failure(self, *, call_id: str, function: str, phase: CapturePhase, reason: str) -> None:
        await self._link.request(
            "report_capture_failure",
            {
                "call_id": call_id,
                "function": function,
                "phase": phase,
                "reason": reason[:CAPTURE_REASON_MAX_LENGTH],
            },
        )


class LinkWorld:
    """A :class:`~midojo_sdk.recorder.WorldState` whose reads and changes are frames on a :class:`Link`.

    A change at a revision other than the one it expected answers ``None``,
    which is the driver's answer, not a refusal: the environment it read is
    not the one it computed new values from, so it changed nothing.
    """

    def __init__(self, link: Link) -> None:
        self._link = link

    @property
    def lost(self) -> LinkError | None:
        """The loss that ended the link, or None while it's live."""
        return self._link.lost

    async def read(self) -> EnvironmentRevision:
        answer = await self._link.request("env_read")
        return _revision(answer)

    async def change(
        self, values: dict[str, Any], *, expected_revision: int | None = None
    ) -> EnvironmentRevision | None:
        answer = await self._link.request("env_change", {"set": values, "expected_revision": expected_revision})
        return None if answer is None else _revision(answer)


def _revision(answer: object) -> EnvironmentRevision:
    if not isinstance(answer, dict) or set(answer) != {"revision", "environment"}:
        raise LinkFault(f"an environment answer holds a revision and an environment, not {answer!r}")
    return EnvironmentRevision(answer["revision"], answer["environment"])


__all__ = [
    "API",
    "INVALID_REQUEST",
    "LINK_LOST_REASON",
    "MESSAGE_OPS",
    "OPS",
    "Answer",
    "Duplex",
    "Frame",
    "Link",
    "LinkError",
    "LinkFault",
    "LinkLost",
    "LinkRecorder",
    "LinkWorld",
    "Message",
    "Rejection",
    "decode_answer",
    "decode_frame",
    "decode_message",
    "encode_answer",
    "encode_frame",
    "encode_message",
    "rejection_error",
]
