"""The two seams an interception layer talks to: recording calls, and the simulated world.

A :class:`Recorder` records the requests the layer intercepts: each call the
agent made, what the agent saw, and what the layer did with it (its
:data:`Disposition`). A :class:`WorldState` reads and changes the simulated
environment that tools defined by the layer act on. Both are bound to one
evaluation by whoever builds them, so tools never carry or choose an
evaluation. They are separate on purpose: a forwarding or recording-only layer
needs a ``Recorder`` and nothing else, while a tool that simulates state also
takes a ``WorldState``.

They keep three kinds of evidence apart. A record is an intercepted request,
never proof of an effect. A simulated effect is a change to the simulated
environment, through a ``WorldState``. A real-system effect (what a forwarded
call did to the real tool's system) goes through neither: a runtime observes
it, outside the layer. Nothing here is a transaction across them: a forwarded
call's side effect, its record and an environment change commit separately,
and any one of them can happen without the others (a tool that changed the
environment and then failed to record its finish, a record that never reached
the control plane after the real tool ran).

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
import warnings
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, TypeVar

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

RECORD_ATTEMPTS = 3
"""How many times :class:`RetryingRecorder` sends a call's start or finish before its error is raised."""

RECORD_BACKOFF_SECONDS = 0.1
"""The wait before the first resend of a record, doubled before each later one."""


class RejectionKind(enum.StrEnum):
    """Why the control plane refused an event: the same kinds over HTTP (the response's ``kind``) and in process."""

    UNKNOWN_RUN = "unknown_run"
    UNKNOWN_EVALUATION = "unknown_evaluation"
    UNKNOWN_TASK = "unknown_task"
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

    async def _retry(self, send: Callable[[], Awaitable[None]]) -> None:
        await retry(send, retryable=self._retryable, attempts=self._attempts, backoff_seconds=self._backoff_seconds)


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
