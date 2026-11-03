"""The driver side of a ``midojo-binding-v1`` link: one session's frames, applied in order, one at a time.

An interception layer in another process sends ``{"id", "op", "args"}`` frames
(see ``midojo_sdk.link``); :class:`LinkDispatcher` applies each one to the
session's :class:`~midojo.control_plane.inprocess.InProcessAgentClient` and
answers ``{"re", "ok"}`` or ``{"re", "rejected": {"kind", "detail",
"late_fact"}}``. It is the session's only writer: the session token and the
records stay on this side of the stream, and nothing is forwarded from here.

The dispatcher validates the frame, not the event's fields: the control
plane's request models do that, as they do for the HTTP route and the
in-process client, so a body either would refuse answers with the same
``invalid_request`` rejection the in-process client raises as
``InvalidRequestError``.

A frame whose id is not the one after the last is a *link fault*: the stream
lost or repeated a frame, and applying the rest would record the evaluation
wrong. The stream starts at id 0, so a first frame with any other id is a
fault too: it skipped the binding's first frame, or repeated one this driver
never applied. A fault ends the link like a lost transport does, and a lost link is
terminal: the requests in flight fail on the service side, nothing is resumed,
and the driver seals the evaluation with ``function_calls`` incomplete
(:meth:`LinkDispatcher.interrupted_evidence`) and raises a handle error with
reason ``link-lost`` (:meth:`LinkDispatcher.error`).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from midojo_sdk.link import (
    INVALID_REQUEST,
    LINK_LOST_REASON,
    Answer,
    Duplex,
    Frame,
    LinkError,
    LinkFault,
    LinkLost,
    Rejection,
    decode_frame,
    encode_answer,
    encode_message,
)
from midojo_sdk.recorder import ControlPlaneRejectedError, EnvironmentRevision, InvalidRequestError

from midojo.observations import FUNCTION_CALLS

from .inprocess import InProcessAgentClient
from .service import ControlPlane

if TYPE_CHECKING:
    from midojo.evaluation_handle import LinkLostError

LINK_LOST = LINK_LOST_REASON
"""The ``HandleError.reason`` a lost link ends the handle with."""


@dataclass(frozen=True)
class Application:
    """One frame the dispatcher applied: the id it arrived with, its op, and the time the driver stamped it."""

    id: int
    op: str
    at: float


class LinkDispatcher:
    """Applies one binding's frames to the session ``session_token`` names, in id order.

    ``clock`` stamps each applied frame with its arrival time; tests pass a
    fake clock, and production passes the wall clock the records already use.
    """

    def __init__(self, control: ControlPlane, session_token: str, *, clock: Callable[[], float] = time.time) -> None:
        # One client serves both seams, the recorder's and the world's: the frames go to the same
        # in-process client the control plane offers, and never over HTTP.
        client = InProcessAgentClient(control, session_token)
        self._recorder = client
        self._world = client
        self._clock = clock
        # The stream's first frame is id 0: one below it is the id no frame may carry.
        self._last_id = -1
        self._lost: LinkError | None = None
        self.applications: list[Application] = []

    @property
    def lost(self) -> LinkError | None:
        """The loss that ended the link, or None while it's live: a transport that went, or a frame the service broke."""
        return self._lost

    @property
    def applied(self) -> tuple[int, ...]:
        """The ids the dispatcher applied, in the order it applied them."""
        return tuple(application.id for application in self.applications)

    def interrupted_evidence(self) -> tuple[str, ...]:
        """The evidence a seal must mark incomplete: ``function_calls`` once the link is lost, else nothing.

        A lost link may have dropped a start or a finish the driver never
        applied (or applied without the service hearing of it), so the
        capture can't be claimed complete, even when no unfinished record
        shows the loss.
        """
        return (FUNCTION_CALLS,) if self._lost is not None else ()

    def error(self) -> LinkLostError | None:
        """The handle error a lost link ends the handle with, or None while it's live.

        Its ``reason`` is ``link-lost``; the handle raises it from whatever
        operation the loss cut short.
        """
        if self._lost is None:
            return None
        from midojo.evaluation_handle import LinkLostError

        return LinkLostError(str(self._lost))

    async def serve(self, duplex: Duplex, *, heartbeat_interval: float | None = None) -> None:
        """Apply frames from ``duplex`` until the link is lost, answering each one.

        ``heartbeat_interval`` is how often the dispatcher sends the
        application ``ping`` while it waits for a frame; ``None`` sends none.
        The ping is only sent from this loop, so a dispatcher that is stuck
        stops pinging and the service expires the binding, as the design's
        liveness rule intends. The driver's remote interception passes the
        wire contract's ``HEARTBEAT_INTERVAL_SECONDS``.

        A link that ends is gone for good, so the stream closes with it: the
        service's requests in flight fail at once rather than waiting for an
        answer this driver will never send.
        """
        try:
            await self._serve(duplex, heartbeat_interval)
        finally:
            with suppress(Exception):
                duplex.close()

    async def _serve(self, duplex: Duplex, heartbeat_interval: float | None) -> None:
        while self._lost is None:
            try:
                data = await self._receive(duplex, heartbeat_interval)
            except LinkLost as lost:
                self._lose(lost)
                return
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
                self._lose(
                    LinkFault(
                        f"frame {frame.id} arrived where {self._last_id + 1} was expected: the stream {what} a frame"
                    )
                )
                return
            try:
                answer = await self._answer(frame)
            except LinkFault as fault:
                self._lose(fault)
                return
            except Exception as failure:
                # A driver that fails to apply a frame can't go on: the evaluation's records would be half-written.
                self._lose(LinkLost(f"the driver failed to apply frame {frame.id}: {failure}"))
                return
            finally:
                self._last_id = frame.id
            try:
                await duplex.send(encode_answer(answer))
            except Exception as e:
                self._lose(LinkLost(f"the answer to frame {frame.id} couldn't be sent: {e}"))
                return

    async def _receive(self, duplex: Duplex, heartbeat_interval: float | None) -> str:
        """The next frame from ``duplex``, pinging the service while the wait lasts."""
        if heartbeat_interval is None:
            return await duplex.receive()
        while True:
            try:
                return await asyncio.wait_for(duplex.receive(), heartbeat_interval)
            except TimeoutError:
                try:
                    await duplex.send(encode_message("ping"))
                except Exception as e:
                    raise LinkLost(f"the heartbeat couldn't be sent: {e}") from e

    async def _answer(self, frame: Frame) -> Answer:
        try:
            result = await self._apply(frame)
        except ControlPlaneRejectedError as refused:
            self.applications.append(Application(frame.id, frame.op, self._clock()))
            rejection = Rejection(refused.kind.value, refused.detail, refused.late_fact)
            return Answer(frame.id, rejected=rejection)
        except InvalidRequestError as invalid:
            # Nothing was decided about the evaluation: the service raises its InvalidRequestError, as in process.
            self.applications.append(Application(frame.id, frame.op, self._clock()))
            return Answer(frame.id, rejected=Rejection(INVALID_REQUEST, invalid.detail))
        self.applications.append(Application(frame.id, frame.op, self._clock()))
        return Answer(frame.id, ok=result)

    async def _apply(self, frame: Frame) -> Any:
        args = frame.args
        if frame.op == "start_call":
            await self._recorder.start_call(
                call_id=_field(args, "call_id"),
                function=_field(args, "function"),
                args=_field(args, "args"),
                disposition=_field(args, "disposition"),
            )
            return None
        if frame.op == "finish_call_detailed":
            fields: dict[str, Any] = {
                "call_id": _field(args, "call_id"),
                "function": _field(args, "function"),
                "args": _field(args, "args"),
                "result": _field(args, "result"),
                "disposition": _field(args, "disposition"),
                "error": args.get("error"),
            }
            if "injection_outcomes" not in args:
                await self._recorder.finish_call(**fields)
                return None
            await self._recorder.finish_call_detailed(
                **fields,
                structured_content=args.get("structured_content"),
                injection_outcomes=_field(args, "injection_outcomes"),
            )
            return None
        if frame.op == "report_capture_failure":
            await self._recorder.report_capture_failure(
                call_id=_field(args, "call_id"),
                function=_field(args, "function"),
                phase=_field(args, "phase"),
                reason=_field(args, "reason"),
            )
            return None
        if frame.op == "env_read":
            return _environment(await self._world.read())
        if frame.op == "env_change":
            changed = await self._world.change(_field(args, "set"), expected_revision=args.get("expected_revision"))
            return None if changed is None else _environment(changed)
        raise LinkFault(f"unknown op {frame.op!r}")

    def _lose(self, error: LinkError) -> None:
        if self._lost is None:
            self._lost = error


def _environment(revision: EnvironmentRevision) -> dict[str, Any]:
    return {"revision": revision.revision, "environment": revision.environment}


def _field(args: Mapping[str, Any], name: str) -> Any:
    """The value the frame carries for ``name``: the control plane's request model is what validates it.

    The frame protocol requires the field to be there, and nothing more: a
    value the model refuses, such as a revision that isn't a number, raises
    the same ``InvalidRequestError`` (and the same rejection frame) as the
    in-process client, not a protocol fault. Sending a field the model
    coerces, such as ``"0"`` for a revision, coerces on both paths alike.
    """
    if name not in args:
        raise LinkFault(f"the frame names no {name}")
    return args[name]
