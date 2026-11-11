"""The ``midojo-binding-v1`` frame codec: what a driver and an interceptor service say to each other.

Two names are easy to confuse, and they are different strings: the WebSocket
upgrade offers the subprotocol ``midojo-binding.v1`` (:data:`SUBPROTOCOL`),
and the messages inside the stream carry the API name ``midojo-binding-v1``
(:data:`API`) in their ``bind``.

An interception layer that runs in its own process records through frames
instead of HTTP: it sends ``{"id", "op", "args"}`` to the driver and awaits the
answer, ``{"re", "ok"}`` or ``{"re", "rejected": {"kind", "detail",
"late_fact"}}``. This module holds the frames, their answers, the control
messages and the errors a lost stream raises. The endpoint that sends frames
and awaits answers is the interceptor service's, and the driver's adapter reads
them: each side imports this codec, and neither imports the other.

This module imports only the standard library and the recorder contract beside
it: an interception layer runs next to the agent, which must not need MiDojo's
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

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .recorder import ControlPlaneRejectedError, InvalidRequestError, RejectionKind

API = "midojo-binding-v1"
"""The messages' API name: what a driver and a service agree on inside the stream."""

SUBPROTOCOL = "midojo-binding.v1"
"""The WebSocket subprotocol the upgrade must offer: the handshake's name, not the messages' ``API``."""

HEARTBEAT_INTERVAL_SECONDS = 5.0
"""How often the driver's dispatcher sends the application ``ping`` while it waits for frames.

Only the loop that reads the service's frames sends it: a WebSocket ping/pong
can be answered by the socket library while the dispatcher is stuck, and the
service would never expire an idle binding whose driver stopped applying
frames. The service's expiry is about three times this interval.
"""

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
    except (TypeError, ValueError) as e:  # RecursionError: nested past what the parser holds
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


__all__ = [
    "API",
    "BIND_FIELDS",
    "BOUND_FIELDS",
    "HEARTBEAT_INTERVAL_SECONDS",
    "INVALID_REQUEST",
    "LINK_LOST_REASON",
    "MESSAGE_OPS",
    "OPS",
    "SUBPROTOCOL",
    "UNBOUND_FIELDS",
    "Answer",
    "Duplex",
    "Frame",
    "LinkError",
    "LinkFault",
    "LinkLost",
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
