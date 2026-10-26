"""The two seams an interception layer talks to: recording calls, and the simulated world.

A :class:`Recorder` records the requests the layer intercepts: each call the
agent made, what the agent saw, and what the layer did with it (its
:data:`Disposition`). A :class:`WorldState` reads and changes the simulated
environment that tools defined by the layer act on. Both are bound to one
evaluation by whoever builds them, so tools never carry or choose an
evaluation. They are separate on purpose: a forwarding or recording-only layer
needs a ``Recorder`` and nothing else, while a tool that simulates state also
takes a ``WorldState``. A recorder that is also a :class:`DetailedRecorder`
records the parts of a result beyond its text (its structured content and the
call's injection outcomes); one that isn't still records every call, with its
text only (:func:`finish_detailed`).

They keep three kinds of evidence apart. A record is an intercepted request,
never proof of an effect. A simulated effect is a change to the simulated
environment, through a ``WorldState``. A real-system effect (what a forwarded
call did to the real tool's system) goes through neither: a runtime observes
it, outside the layer. Nothing here is a transaction across them: a forwarded
call's side effect, its record and an environment change commit separately,
and any one of them can happen without the others (a tool that changed the
environment and then failed to record its finish, a record that never reached
the control plane after the real tool ran).

A record that can't be delivered is evidence too. A recorder that implements
:class:`CaptureReporter` takes a report of each start or finish that never
reached the control plane, which marks the evaluation's ``function_calls``
incomplete, so a check that depends on a call's absence can't pass on a lost
record. :class:`ReportingRecorder` sends that report whenever a record fails
for good; the SDK's interception layers record through one.

Implementations satisfy these protocols structurally and exchange plain values
(dicts, ``None``): :class:`midojo_sdk.client.AgentControlPlaneClient` over HTTP,
or the control plane's in-process client. They raise the same errors: a refusal
by the control plane is a :class:`ControlPlaneRejectedError` carrying its
:class:`RejectionKind`, whichever implementation sent it. This module imports
nothing outside the standard library, so an implementation elsewhere can import
it cheaply.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import warnings
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, TypedDict, TypeVar, cast

if TYPE_CHECKING:
    from midojo_sdk.client import AgentControlPlaneClient

Disposition = Literal["forwarded", "denied", "simulated", "blocked"]
"""What the interception layer did with a call.

- ``forwarded``: sent to the real tool; the agent got the real tool's answer
- ``simulated``: answered by a tool the layer defines, instead of the real tool
- ``denied``: refused by a check on this call's arguments; the tool never ran
- ``blocked``: refused because the tool is blocked for the evaluation; the tool never ran

A denied or blocked call gives the agent an error result, never a success.
"""

REFUSED: frozenset[Disposition] = frozenset({"denied", "blocked"})
"""The dispositions of a call whose tool never ran: the record's ``blocked`` flag."""


class InjectionRecord(TypedDict):
    """How one injection instruction fared on a call, as the call's detailed finish carries it.

    ``status`` is a :class:`midojo_sdk.injection.OutcomeStatus` value as a
    plain ``str``: ``applied``, ``partial`` (structured content only),
    ``fallback``, ``unmatched``, ``too_large``, ``skipped_occurrence`` (the
    instruction matched the call but its occurrence rule skipped it) or
    ``no_result``. ``detail`` says why. A record names neither the counter
    scope (an evaluation ID or a session token) nor the payload.
    """

    instruction_index: int
    probe_key: str | None
    occurrence: int
    status: str
    detail: str


RECORD_ATTEMPTS = 3
"""How many times :class:`RetryingRecorder` sends a call's start or finish before its error is raised."""

RECORD_BACKOFF_SECONDS = 0.1
"""The wait before the first resend of a record, doubled before each later one."""

CapturePhase = Literal["start", "finish"]
"""Which record of a call couldn't be delivered: its start or its finish."""

CAPTURE_REASON_MAX_LENGTH = 2048
"""The longest capture failure ``reason`` the control plane accepts; a longer one is cut to this length."""

logger = logging.getLogger(__name__)


class RejectionKind(enum.StrEnum):
    """Why the control plane refused an event: the same kinds over HTTP (the response's ``kind``) and in process."""

    UNKNOWN_RUN = "unknown_run"
    UNKNOWN_EVALUATION = "unknown_evaluation"
    UNKNOWN_TASK = "unknown_task"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    INVALID_SESSION = "invalid_session"
    SESSION_CLOSED = "session_closed"
    FORBIDDEN = "forbidden"
    CLOSING = "closing"
    SEALED = "sealed"
    CLOSURE_PENDING = "closure_pending"
    NOT_SEALED = "not_sealed"
    CALL_CONFLICT = "call_conflict"
    INVALID_DISPOSITION = "invalid_disposition"
    TURN_PENDING = "turn_pending"
    TURN_NOT_STARTED = "turn_not_started"
    TURN_ANSWERED = "turn_answered"
    REVISION_CONFLICT = "revision_conflict"
    INVALID_ENVIRONMENT = "invalid_environment"
    UNSUPPORTED_VALUE = "unsupported_value"


class ControlPlaneRejectedError(RuntimeError):
    """The control plane refused an event: a record, or a read or change of the environment.

    ``kind`` says why. ``late_fact`` is True when the control plane kept the
    refused write as a late fact: the evaluation was closing or sealed, and its
    evidence is now marked incomplete. A refusal is never resent: sending the
    same event again gets the same answer.
    """

    def __init__(self, kind: RejectionKind, detail: str, *, late_fact: bool = False) -> None:
        super().__init__(f"{kind.value}: {detail}")
        self.kind = kind
        self.detail = detail
        self.late_fact = late_fact


class InvalidRequestError(ValueError):
    """The control plane can't take the request as sent: a field is missing, too long, or of the wrong type.

    Unlike a :class:`ControlPlaneRejectedError`, nothing was decided about the
    evaluation: the request never reached its lifecycle rules. Over HTTP it's
    the 422 the API answers before the control plane decides anything (and is
    also an ``httpx.HTTPStatusError``); in process, the same check fails before
    the event is submitted. Resending it gets the same answer.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(f"invalid request: {detail}")
        self.detail = detail


FINAL_ERRORS: tuple[type[Exception], ...] = (ControlPlaneRejectedError, InvalidRequestError)
"""The errors no resend can change: :func:`retry` raises them at once, whatever its ``retryable`` says."""


class Recorder(Protocol):
    """Records one evaluation's tool calls: each call's start, then its finish with the same ``call_id``.

    Each method sends its event once, and raises :class:`ControlPlaneRejectedError`
    when the control plane refuses it. Wrap a recorder in :class:`RetryingRecorder`
    to resend after a transient failure.
    """

    async def start_call(self, *, call_id: str, function: str, args: dict[str, Any], disposition: Disposition) -> None:
        """Record that a call started: an unfinished record, before the tool runs.

        A started call already has its disposition (``forwarded`` or
        ``simulated``), so a call that never finishes still has one.
        """
        ...

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
        """Finish the call started with ``call_id``, or record it whole when it wasn't started."""
        ...


CallRecorder = Recorder
"""The earlier name of :class:`Recorder`."""


class DetailedRecorder(Recorder, Protocol):
    """A :class:`Recorder` that also records the parts of a result beyond its text: an optional capability.

    ``finish_call_detailed`` finishes a call as ``finish_call`` does, and also
    records the structured content the agent got with ``result`` (its text)
    and how each injection instruction that matched the call fared. A
    recorder has the capability when :func:`records_details` says so: the
    HTTP and in-process clients do, and :class:`RetryingRecorder` and
    :class:`ReportingRecorder` have the capability of the recorder they wrap.

    A recorder without it still records the call, with ``finish_call``
    (:func:`finish_detailed` falls back to it): the text only. The control
    plane keeps such a record with ``injection_outcomes`` None, which says the
    result's other parts and the call's injection outcomes weren't reported,
    not that there were none; a detailed finish always sends a list, empty
    when no instruction matched the call.
    """

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
        """Finish the call started with ``call_id``, with the structured content and injection outcomes too.

        ``structured_content`` is None when the agent got none.
        """
        ...


def records_details(recorder: object) -> bool:
    """Whether ``recorder`` is a :class:`DetailedRecorder`.

    A wrapping recorder says so with its own ``records_details`` attribute,
    taken from the recorder it wraps; any other recorder is one when it has a
    ``finish_call_detailed`` method.
    """
    declared = getattr(recorder, "records_details", None)
    if isinstance(declared, bool):
        return declared
    return callable(getattr(recorder, "finish_call_detailed", None))


async def finish_detailed(
    recorder: Recorder,
    *,
    call_id: str,
    function: str,
    args: dict[str, Any],
    result: str,
    disposition: Disposition,
    error: str | None = None,
    structured_content: dict[str, Any] | None,
    injection_outcomes: Sequence[InjectionRecord],
) -> bool:
    """Finish a call with its details when ``recorder`` records them, else with ``finish_call``: the text only.

    Returns whether the details were recorded. The fallback never fails the
    finish because of the details: a recorder written before them records
    the call as it always did.
    """
    if records_details(recorder):
        await cast("DetailedRecorder", recorder).finish_call_detailed(
            call_id=call_id,
            function=function,
            args=args,
            result=result,
            disposition=disposition,
            error=error,
            structured_content=structured_content,
            injection_outcomes=injection_outcomes,
        )
        return True
    await recorder.finish_call(
        call_id=call_id, function=function, args=args, result=result, disposition=disposition, error=error
    )
    return False


class CaptureReporter(Protocol):
    """Reports a call record that couldn't be delivered, so the evaluation shows its capture is incomplete.

    Optional for a :class:`Recorder`: the HTTP and in-process clients
    implement it, and :class:`RetryingRecorder` and :class:`ReportingRecorder`
    pass it on to the recorder they wrap.
    """

    async def report_capture_failure(self, *, call_id: str, function: str, phase: CapturePhase, reason: str) -> None:
        """Report that the ``phase`` record of the call ``call_id`` never reached the control plane.

        The control plane keeps one report per ``call_id``, function and phase
        (whatever its reason), and marks the evaluation's ``function_calls``
        incomplete; a report kept before the seal is what grading sees. It
        raises :class:`ControlPlaneRejectedError` when the control plane
        refuses it: a new report after the seal, with ``late_fact`` True, is
        still kept as a late fact, which marks the evaluation's evidence status
        but not its grade.
        """
        ...


class EnvironmentRevision(NamedTuple):
    """The simulated environment's fields, as of one revision.

    The revision counts the environment's changes: every change, through any
    caller, gives a new one.
    """

    revision: int
    environment: dict[str, Any]


class WorldState(Protocol):
    """Reads and changes one evaluation's simulated environment.

    A change is atomic on its own: no other change interleaves with it. It is
    not atomic with anything else, such as a record or a real tool's effect.
    The environment is a value the control plane keeps; it says what the
    simulated tools did, not what the real system did.
    """

    async def read(self) -> EnvironmentRevision:
        """The current environment and its revision."""
        ...

    async def change(
        self, values: dict[str, Any], *, expected_revision: int | None = None
    ) -> EnvironmentRevision | None:
        """Set the top-level fields in ``values`` in one atomic step, and return the environment it made.

        With ``expected_revision``, the change applies only at that revision:
        at any other it changes nothing and returns ``None``, so a caller that
        read the environment, computed new values and names the revision it
        read loses no concurrent change. Raises :class:`ControlPlaneRejectedError`
        with ``invalid_environment`` when the changed environment isn't valid
        for the evaluation's suite.
        """
        ...


T = TypeVar("T")


async def retry(
    send: Callable[[], Awaitable[T]],
    *,
    retryable: Callable[[Exception], bool],
    attempts: int | None = None,
    backoff_seconds: float | None = None,
) -> T:
    """Call ``send`` until it returns, resending after an error ``retryable`` accepts.

    Up to ``attempts`` calls in all (``RECORD_ATTEMPTS`` by default), waiting
    ``backoff_seconds`` (``RECORD_BACKOFF_SECONDS``) before the first resend
    and twice as long before each later one. Any other error, and the last
    attempt's, is raised at once, and so is one of :data:`FINAL_ERRORS` (a
    rejection or an invalid request) whatever ``retryable`` says.
    """
    attempts = RECORD_ATTEMPTS if attempts is None else attempts
    backoff = RECORD_BACKOFF_SECONDS if backoff_seconds is None else backoff_seconds
    for attempt in range(1, attempts + 1):
        try:
            return await send()
        except FINAL_ERRORS:
            raise
        except Exception as e:
            if attempt == attempts or not retryable(e):
                raise
        await asyncio.sleep(backoff * 2 ** (attempt - 1))
    raise AssertionError("unreachable")


class RetryingRecorder:
    """A :class:`Recorder` that resends each start and finish after a transient failure.

    ``retryable`` says which errors are transient: for the HTTP recorder,
    :func:`midojo_sdk.client.transient` (a transport error, or a 5xx response
    that isn't the control plane's own rejection). A
    :class:`ControlPlaneRejectedError` or an :class:`InvalidRequestError` is
    never resent, whatever ``retryable`` says.

    Resends keep the ``call_id``, and the control plane keeps at most one
    record per ``call_id``, so a resent event that had already arrived isn't
    recorded twice. This is at most one record per call, not exactly once: a
    finish whose sends all fail leaves its call unfinished, and a call whose
    every send fails is missing.
    """

    def __init__(
        self,
        recorder: Recorder,
        *,
        retryable: Callable[[Exception], bool],
        attempts: int | None = None,
        backoff_seconds: float | None = None,
    ) -> None:
        self.recorder = recorder
        self._retryable = retryable
        self._attempts = attempts
        self._backoff_seconds = backoff_seconds

    async def start_call(self, *, call_id: str, function: str, args: dict[str, Any], disposition: Disposition) -> None:
        await self._retry(
            lambda: self.recorder.start_call(call_id=call_id, function=function, args=args, disposition=disposition)
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
        await self._retry(
            lambda: self.recorder.finish_call(
                call_id=call_id, function=function, args=args, result=result, disposition=disposition, error=error
            )
        )

    @property
    def records_details(self) -> bool:
        """Whether the wrapped recorder records a finish's details (see :class:`DetailedRecorder`)."""
        return records_details(self.recorder)

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
        """Finish the call through the wrapped recorder, with its details when that recorder records them."""
        await self._retry(
            lambda: finish_detailed(
                self.recorder,
                call_id=call_id,
                function=function,
                args=args,
                result=result,
                disposition=disposition,
                error=error,
                structured_content=structured_content,
                injection_outcomes=injection_outcomes,
            )
        )

    async def report_capture_failure(self, *, call_id: str, function: str, phase: CapturePhase, reason: str) -> None:
        """Report a lost record through the wrapped recorder, resent like a record.

        Raises ``NotImplementedError`` when the wrapped recorder can't report one.
        """
        report = _reporter(self.recorder)
        await self._retry(lambda: report(call_id=call_id, function=function, phase=phase, reason=reason))

    async def _retry(self, send: Callable[[], Awaitable[object]]) -> None:
        await retry(send, retryable=self._retryable, attempts=self._attempts, backoff_seconds=self._backoff_seconds)


class ReportingRecorder:
    """A :class:`Recorder` that reports each start or finish it couldn't deliver, then raises its error.

    A record is lost when sending it raised anything but a refusal the control
    plane kept as a late fact (which already marks the evidence incomplete):
    a transport error after every resend of a :class:`RetryingRecorder`, an
    invalid request, or another refusal, such as a ``call_conflict``. The
    report goes through the wrapped recorder's
    :meth:`CaptureReporter.report_capture_failure` once, best effort: when it
    fails too, or the wrapped recorder can't report, it's logged, and the
    record's own error is what's raised. Cancellation isn't a lost record:
    it's raised without a report.
    """

    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder

    async def start_call(self, *, call_id: str, function: str, args: dict[str, Any], disposition: Disposition) -> None:
        await self._record(
            "start",
            call_id,
            function,
            lambda: self.recorder.start_call(call_id=call_id, function=function, args=args, disposition=disposition),
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
        await self._record(
            "finish",
            call_id,
            function,
            lambda: self.recorder.finish_call(
                call_id=call_id, function=function, args=args, result=result, disposition=disposition, error=error
            ),
        )

    @property
    def records_details(self) -> bool:
        """Whether the wrapped recorder records a finish's details (see :class:`DetailedRecorder`)."""
        return records_details(self.recorder)

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
        """Finish the call through the wrapped recorder, with its details when that recorder records them."""
        await self._record(
            "finish",
            call_id,
            function,
            lambda: finish_detailed(
                self.recorder,
                call_id=call_id,
                function=function,
                args=args,
                result=result,
                disposition=disposition,
                error=error,
                structured_content=structured_content,
                injection_outcomes=injection_outcomes,
            ),
        )

    async def report_capture_failure(self, *, call_id: str, function: str, phase: CapturePhase, reason: str) -> None:
        await _reporter(self.recorder)(call_id=call_id, function=function, phase=phase, reason=reason)

    async def _record(
        self, phase: CapturePhase, call_id: str, function: str, send: Callable[[], Awaitable[object]]
    ) -> None:
        try:
            await send()
        except ControlPlaneRejectedError as e:
            if not e.late_fact:
                await self._report(phase, call_id, function, e)
            raise
        except Exception as e:
            await self._report(phase, call_id, function, e)
            raise

    async def _report(self, phase: CapturePhase, call_id: str, function: str, error: Exception) -> None:
        reason = f"{type(error).__name__}: {error}"[:CAPTURE_REASON_MAX_LENGTH]
        try:
            await self.report_capture_failure(call_id=call_id, function=function, phase=phase, reason=reason)
        except ControlPlaneRejectedError as refused:
            if not refused.late_fact:
                logger.warning(
                    "The control plane refused the capture failure of %s (%s): %s", function, call_id, refused
                )
        except Exception:
            logger.warning("The capture failure of %s (%s) wasn't delivered", function, call_id, exc_info=True)


def reporting(recorder: Recorder) -> ReportingRecorder:
    """``recorder`` as a :class:`ReportingRecorder`: itself when it already is one."""
    return recorder if isinstance(recorder, ReportingRecorder) else ReportingRecorder(recorder)


def _reporter(recorder: object) -> Callable[..., Awaitable[None]]:
    """``recorder``'s :meth:`CaptureReporter.report_capture_failure`, or ``NotImplementedError`` without one."""
    report = getattr(recorder, "report_capture_failure", None)
    if not callable(report):
        raise NotImplementedError(f"{type(recorder).__name__} can't report a capture failure")
    return cast("Callable[..., Awaitable[None]]", report)


class ControlPlaneRecorder:
    """Deprecated: records calls through an :class:`~midojo_sdk.client.AgentControlPlaneClient`.

    The client is a :class:`Recorder` itself; wrap it in
    ``RetryingRecorder(client, retryable=midojo_sdk.client.transient)`` for
    resends. This adapter does exactly that, as it resent records before.
    """

    def __init__(self, client: AgentControlPlaneClient) -> None:
        from midojo_sdk.client import transient

        warnings.warn(
            "ControlPlaneRecorder is deprecated: use the AgentControlPlaneClient as the recorder, wrapped in "
            "RetryingRecorder(client, retryable=midojo_sdk.client.transient) for resends.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._recorder = RetryingRecorder(client, retryable=transient)

    async def start_call(self, *, call_id: str, function: str, args: dict[str, Any], disposition: Disposition) -> None:
        await self._recorder.start_call(call_id=call_id, function=function, args=args, disposition=disposition)

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
        await self._recorder.finish_call(
            call_id=call_id, function=function, args=args, result=result, disposition=disposition, error=error
        )

    @property
    def records_details(self) -> bool:
        """Whether the wrapped recorder records a finish's details (see :class:`DetailedRecorder`)."""
        return records_details(self._recorder)

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
        """Finish the call through the wrapped recorder, with its details when that recorder records them."""
        await self._recorder.finish_call_detailed(
            call_id=call_id,
            function=function,
            args=args,
            result=result,
            disposition=disposition,
            error=error,
            structured_content=structured_content,
            injection_outcomes=injection_outcomes,
        )

    async def report_capture_failure(self, *, call_id: str, function: str, phase: CapturePhase, reason: str) -> None:
        await self._recorder.report_capture_failure(call_id=call_id, function=function, phase=phase, reason=reason)
