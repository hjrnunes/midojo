"""The service side of a remote interception link: a request stream, and the recorder and world over it.

:class:`Link` sends the ``midojo-binding-v1`` frames of :mod:`midojo_spec.link` and
awaits their answers. :class:`LinkRecorder` and :class:`LinkWorld` are the
service's end of the stream: a :class:`~midojo_spec.recorder.Recorder` (and
:class:`~midojo_spec.recorder.DetailedRecorder` and
:class:`~midojo_spec.recorder.CaptureReporter`) and a
:class:`~midojo_spec.recorder.WorldState` whose every call is one frame.

This module imports only the standard library and ``midojo_spec``: the
interception layer runs beside the agent, which must not need MiDojo's
driver-side packages. The transport is whatever holds the two ends of the
stream; it only has to send and receive text frames.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import suppress
from typing import Any

from midojo_spec import link as wire
from midojo_spec.recorder import (
    CAPTURE_REASON_MAX_LENGTH,
    CapturePhase,
    Disposition,
    EnvironmentRevision,
    InjectionRecord,
)


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

    def __init__(self, duplex: wire.Duplex, *, answer_deadline: float | None = None) -> None:
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
        self._lost: wire.LinkError | None = None
        self._lost_event = asyncio.Event()

    @property
    def lost(self) -> wire.LinkError | None:
        """The loss that ended the link, or None while it's live: a transport that went, or a frame the peer broke."""
        return self._lost

    async def wait_lost(self) -> wire.LinkError:
        """The loss that ended the link, waiting for it while the link is still live."""
        await self._lost_event.wait()
        assert self._lost is not None
        return self._lost

    async def request(self, op: str, args: Mapping[str, Any] | None = None) -> Any:
        """Send one frame and return its answer's result, or raise its rejection or the link's loss."""
        if op not in wire.OPS:
            raise ValueError(f"unknown op {op!r}: the binding stream has {list(wire.OPS)}")
        if self._lost is not None:
            raise self._lost
        self._ensure_reader()
        frame_id = self._next_id
        self._next_id += 1
        answer: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[frame_id] = answer
        try:
            await self._duplex.send(wire.encode_frame(wire.Frame(frame_id, op, dict(args or {}))))
        except asyncio.CancelledError:
            self._pending.pop(frame_id, None)
            self._lose(wire.LinkLost(f"frame {frame_id} was cancelled before it was sent"))
            raise
        except Exception as e:
            self._pending.pop(frame_id, None)
            lost = wire.LinkLost(f"frame {frame_id} couldn't be sent: {e}")
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
            self._lose(wire.LinkLost(f"frame {frame_id} waited more than {self._answer_deadline:g}s for its answer"))
            # Raise the link's own loss: two requests that timed out together fail as one fact, not two.
            lost = self._lost
            assert lost is not None
            raise lost from None

    def close(self, detail: str = "the service closed the binding") -> None:
        """End the link and its stream: the requests in flight fail, and so does every later one."""
        self._lose(wire.LinkLost(detail))
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
                self._lose(wire.LinkLost(f"the binding stream ended: {e}"))
                return
            try:
                decoded = wire.decode_answer(data)
            except wire.LinkFault as fault:
                self._lose(fault)
                return
            if decoded.re in self._abandoned:
                self._abandoned.discard(decoded.re)
                continue
            answer = self._pending.get(decoded.re)
            if answer is None or answer.done():
                self._lose(wire.LinkFault(f"the answer to frame {decoded.re} matches no request in flight"))
                return
            if decoded.rejected is None:
                answer.set_result(decoded.ok)
                continue
            try:
                answer.set_exception(wire.rejection_error(decoded.rejected))
            except wire.LinkFault as fault:
                self._lose(fault)
                return

    def _lose(self, error: wire.LinkError) -> None:
        if self._lost is None:
            self._lost = error
        self._lost_event.set()
        for answer in self._pending.values():
            if not answer.done():
                answer.set_exception(self._lost)
        self._pending.clear()


class LinkRecorder:
    """A :class:`~midojo_spec.recorder.Recorder` whose every call is one frame on a :class:`Link`.

    It records a finish's details, so :func:`~midojo_spec.recorder.records_details`
    says so and an interception layer never falls back to the text-only finish.
    A refusal raises :class:`~midojo_spec.recorder.ControlPlaneRejectedError`
    with the driver's kind, detail and late-fact flag; a lost link raises
    :class:`LinkLost`, which is final: the frame is not resent.
    """

    def __init__(self, link: Link) -> None:
        self._link = link

    @property
    def lost(self) -> wire.LinkError | None:
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
    """A :class:`~midojo_spec.recorder.WorldState` whose reads and changes are frames on a :class:`Link`.

    A change at a revision other than the one it expected answers ``None``,
    which is the driver's answer, not a refusal: the environment it read is
    not the one it computed new values from, so it changed nothing.
    """

    def __init__(self, link: Link) -> None:
        self._link = link

    @property
    def lost(self) -> wire.LinkError | None:
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
        raise wire.LinkFault(f"an environment answer holds a revision and an environment, not {answer!r}")
    return EnvironmentRevision(answer["revision"], answer["environment"])


__all__ = ["Link", "LinkRecorder", "LinkWorld"]
