"""Evaluation lifecycle rules as pure transitions.

``transition(evaluation, authority, event)`` decides one event against one
evaluation's current value. It returns:

* :class:`Accepted`: the next value, the event's result, and the effects the
  writer applies outside the evaluation (the session token index);
* :class:`Late`: the event is refused, and the next value records it as a late
  fact, so the evaluation shows its evidence is incomplete;
* :class:`Rejected`: the event is refused and leaves nothing behind.

It never changes its arguments, and what it returns shares no object with them:
it decides on deep copies, and the seal gives the snapshot its own copies too. It reads nothing
but its arguments either: events carry plain values (a time, an assigned call
ID, an already validated environment), never a clock, a store or a suite. A
rejection is a domain :class:`Rejection` kind; only the HTTP edge picks a status
code for it.

The rules kept here:

* sessions: an agent's event is admitted only with the evaluation's session
  (matching token hash, not expired), and only for reads, calls and the
  environment;
* call-ID idempotency and conflicts: one record per ``call_id``; a replay with
  the same body returns the stored record, at any stage, and another body is a
  conflict until the seal, and a late fact after it;
* unfinished calls: a start appends an unfinished record that the finish
  completes in place, keeping its start order and its start's turn;
* dispositions: what the interception layer did with a call, optional on a
  start and a finish. A denied or blocked call is ``blocked`` and a forwarded
  or simulated one isn't; a refused call never runs, so it's never started.
  Any other pairing is ``invalid_disposition``, at any stage. A disposition
  left out (None) says nothing, so it matches any: a finish that leaves it out
  keeps its start's, and a replay without one replays a record with one. Two
  named dispositions must be the same, or it's a ``call_conflict`` like
  another function or args: what the layer did with a call is decided when it
  starts, so a finish never overrides it;
* provenance: a record says how it reached the control plane, from the
  authority it arrived with: ``host`` from the host, ``interceptor`` from the
  session of an evaluation intercepted on the host, ``agent`` from the session
  of one intercepted in the agent. A finish keeps its start's. Nothing the
  caller sends sets it;
* the call capture: an evaluation intercepted in the agent that is sealed
  without any call record has its ``function_calls`` evidence ``incomplete``,
  since a hook that fails open records nothing;
* environment history: a record carries environment snapshots only when the
  evaluation was created to keep them, as what was observed when the call
  started and finished, never as what the call caused;
* the environment's revision: every change counts one, and a change that
  names the revision it was computed from applies only at that revision;
* turns: a call carries the index of the latest started turn, a turn starts
  only once the previous one has its reply, and a turn gets one reply; a
  turn's message and reply, and the agent's output at the seal, are exactly
  ``str``, or ``unsupported_value`` at any stage (never a late fact);
* staged closure: ``open`` admits everything; ``closing`` admits no new call
  but lets admitted calls finish until none is unfinished or the closure wait
  ends, judged by each event's arrival time against the deadline;
  ``collecting`` takes the final observations; ``sealed`` fixes the snapshot
  that grading reads;
* late facts: a write that closure refuses is recorded as a late fact, never
  applied, and marks the evidence it carries ``incomplete``;
* capture failures: a call record the interception layer reports it couldn't
  deliver is kept until the seal, and marks ``function_calls`` ``incomplete``
  in the snapshot; it never adds or changes a record. A report is identified
  by its ``call_id``, function and phase alone (not its reason or sender): a
  report of a kept one is a replay at any stage, even after the seal. A new
  report after the seal is a late fact, kept once per identity, that marks
  only the evaluation's evidence status, never the snapshot or the grade;
* the grade: recorded once, from the snapshot, and never replaced.

The single writer (:mod:`midojo.control_plane.service`) applies transitions:
it holds the evaluation's lock across look-up, transition and commit.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import json
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ValidationError
from pydantic_core import to_jsonable_python

from midojo.observations import FUNCTION_CALLS, EvidenceStatus, evidence_status, require_evidence_keys
from midojo.types import REFUSED_DISPOSITIONS, CallDisposition, CallProvenance, Environment, FunctionCallRecord

from .models import (
    CapturePhase,
    CreateFunctionCallRecord,
    CreateFunctionCallStart,
    FunctionCallBody,
    FunctionCallStart,
    GradeResponse,
)
from .state import (
    CaptureFailure,
    Evaluation,
    LateFact,
    Session,
    Snapshot,
    Stage,
    Turn,
    canonical_json,
    capture_evidence,
    late_evidence,
    unsupported_value,
)

# --- outcomes ---


class Rejection(enum.StrEnum):
    # Outside any one evaluation: what the writer rejects before a transition runs.
    UNKNOWN_RUN = "unknown_run"
    UNKNOWN_EVALUATION = "unknown_evaluation"
    UNKNOWN_TASK = "unknown_task"
    # Sessions.
    INVALID_SESSION = "invalid_session"
    SESSION_CLOSED = "session_closed"
    FORBIDDEN = "forbidden"
    # Closure.
    CLOSING = "closing"
    SEALED = "sealed"
    CLOSURE_PENDING = "closure_pending"
    NOT_SEALED = "not_sealed"
    # Calls and turns.
    CALL_CONFLICT = "call_conflict"
    INVALID_DISPOSITION = "invalid_disposition"
    TURN_PENDING = "turn_pending"
    TURN_NOT_STARTED = "turn_not_started"
    TURN_ANSWERED = "turn_answered"
    # The environment.
    REVISION_CONFLICT = "revision_conflict"
    INVALID_ENVIRONMENT = "invalid_environment"
    # Values a snapshot can't hold (see ``state.unsupported_value``).
    UNSUPPORTED_VALUE = "unsupported_value"


@dataclass(frozen=True)
class Rejected:
    kind: Rejection
    detail: str


@dataclass(frozen=True)
class SessionBound:
    """Effect: index ``token_hash`` to the evaluation."""

    token_hash: str


@dataclass(frozen=True)
class SessionUnbound:
    """Effect: drop ``token_hash`` from the index."""

    token_hash: str


Effect = SessionBound | SessionUnbound


@dataclass(frozen=True)
class Accepted:
    evaluation: Evaluation
    result: Any = None
    effects: tuple[Effect, ...] = ()


@dataclass(frozen=True)
class Late:
    """The event is refused, and ``evaluation`` is the value with it recorded as a late fact."""

    evaluation: Evaluation
    rejected: Rejected


Outcome = Accepted | Late | Rejected


class RejectedError(ValueError):
    """A refused event, raised by the writer. ``late`` is True when it was recorded as a late fact."""

    def __init__(self, rejected: Rejected, *, late: bool = False) -> None:
        super().__init__(rejected.detail)
        self.rejected = rejected
        self.late = late

    @property
    def kind(self) -> Rejection:
        return self.rejected.kind


# --- authorities ---


@dataclass(frozen=True)
class AgentSession:
    """An interception layer holding the evaluation's session token (by hash), at time ``now`` (epoch seconds)."""

    token_hash: str
    now: float


@dataclass(frozen=True)
class Host:
    """The orchestrator or another host-side caller, already authorized at the edge (the control token)."""


HOST = Host()
Authority = AgentSession | Host


# --- events ---


@dataclass(frozen=True)
class Read:
    """Admit a read: the result is the current value, as of ``at`` when it's given (see ``_on_time``)."""

    at: str | None = None


@dataclass(frozen=True)
class OpenSession:
    token_hash: str
    expires_at: float


@dataclass(frozen=True)
class CloseSession:
    pass


@dataclass(frozen=True)
class StartCall:
    req: CreateFunctionCallStart
    at: str


@dataclass(frozen=True)
class FinishCall:
    req: CreateFunctionCallRecord
    at: str
    # The call_id a record sent without one gets. Chosen by the caller, so the transition stays pure.
    assigned_call_id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class ReportCaptureFailure:
    """Report that a call's ``phase`` record never reached the control plane, after the layer's resends.

    The recorded calls may then lack that call, so ``function_calls`` is
    incomplete: a check that depends on a call's absence can't be decided.
    """

    call_id: str
    function: str
    phase: CapturePhase
    reason: str
    at: str


@dataclass(frozen=True)
class StartTurn:
    message: str
    at: str


@dataclass(frozen=True)
class ReplyTurn:
    index: int
    reply: str
    at: str


@dataclass(frozen=True)
class ReplaceEnvironment:
    """Install an environment the caller has already validated against the evaluation's suite."""

    environment: Environment
    at: str


@dataclass(frozen=True)
class ChangeEnvironment:
    """Set top-level fields of the environment in one step, at ``expected_revision`` when it's given.

    The changed environment is validated against the environment's own model,
    the suite's environment type. The event keeps the values as a dict of its
    own, whatever mapping it was given.
    """

    values: Mapping[str, Any]
    at: str
    expected_revision: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", dict(self.values))


@dataclass(frozen=True)
class RecordObservations:
    """Record a source's observations, already validated against the model registered for it."""

    source: str
    observed: BaseModel
    at: str


@dataclass(frozen=True)
class BeginClosure:
    """Stop admitting new calls. Admitted calls may finish until ``deadline`` (epoch seconds)."""

    at: str
    deadline: float


@dataclass(frozen=True)
class EndClosureWait:
    """End the closure wait at ``now`` (epoch seconds): refused while a call is unfinished before the deadline."""

    now: float


@dataclass(frozen=True)
class Seal:
    """Take the snapshot, with the agent's final output.

    ``incomplete`` names evidence-status keys the host knows it didn't
    collect in full (an episode cut short by a failure or a cancellation):
    the snapshot marks them ``incomplete``. A Seal naming an unknown key
    (see ``observations.is_evidence_key``) can't be made: its constructor
    raises ValueError, so no path into the writer gets one. It keeps the
    keys as a tuple of its own, so changing what it was given changes
    nothing in it.
    """

    agent_output: str
    at: str
    incomplete: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        incomplete = tuple(self.incomplete)
        require_evidence_keys(incomplete)
        object.__setattr__(self, "incomplete", incomplete)


@dataclass(frozen=True)
class RecordGrade:
    """Record the grade computed from the snapshot. A recorded grade is never replaced."""

    grade: GradeResponse


Event = (
    Read
    | OpenSession
    | CloseSession
    | StartCall
    | FinishCall
    | ReportCaptureFailure
    | StartTurn
    | ReplyTurn
    | ReplaceEnvironment
    | ChangeEnvironment
    | RecordObservations
    | BeginClosure
    | EndClosureWait
    | Seal
    | RecordGrade
)

# What an agent's session may do. Everything else is the host's.
AGENT_EVENTS = (Read, StartCall, FinishCall, ReportCaptureFailure, ReplaceEnvironment, ChangeEnvironment)

INVALID_SESSION_DETAIL = "Invalid, expired, or closed evaluation session"
SEALED_DETAIL = "Evaluation already completed"
CLOSING_DETAIL = "Evaluation is closing"


# --- the transition ---


def transition(evaluation: Evaluation, authority: Authority, event: Event) -> Outcome:
    if (rejected := _admit(evaluation, authority, event)) is not None:
        return rejected
    # A str is immutable, so the event's own text is the text the rules would decide on.
    if (rejected := _not_text(event)) is not None:
        return rejected
    # The rules work on copies: what a transition returns shares nothing with its arguments, so a
    # caller changing the event or the source evaluation afterwards can't change the outcome.
    evaluation = copy.deepcopy(evaluation)
    try:
        event = copy.deepcopy(event)
    except Exception as error:
        if (refusal := _held_values(event)) is None:
            raise
        kind, what, _ = refusal
        return Rejected(kind, f"The {what} can't be kept: it can't be copied ({type(error).__name__})")
    # Checked on the copy the rules decide on, never on the caller's event, which may still change.
    if (unsupported := _unsupported(event)) is not None:
        return unsupported
    evaluation = _on_time(evaluation, event)
    match event:
        case Read():
            return Accepted(evaluation, evaluation)
        case OpenSession():
            return _open_session(evaluation, event)
        case CloseSession():
            effects = (SessionUnbound(evaluation.session.token_hash),) if evaluation.session else ()
            return Accepted(_replace(evaluation, session=None), None, effects)
        case StartCall():
            return _start_call(evaluation, authority, event)
        case FinishCall():
            return _finish_call(evaluation, authority, event)
        case ReportCaptureFailure():
            return _report_capture_failure(evaluation, authority, event)
        case StartTurn():
            return _start_turn(evaluation, authority, event)
        case ReplyTurn():
            return _reply_turn(evaluation, authority, event)
        case ReplaceEnvironment():
            return _replace_environment(evaluation, authority, event)
        case ChangeEnvironment():
            return _change_environment(evaluation, authority, event)
        case RecordObservations():
            return _record_observations(evaluation, authority, event)
        case BeginClosure():
            if evaluation.stage is not Stage.OPEN:
                return Accepted(evaluation, evaluation)
            next_value = _replace(evaluation, stage=Stage.CLOSING, closure_deadline=event.deadline)
            return Accepted(next_value, next_value)
        case EndClosureWait():
            return _end_closure_wait(evaluation, event)
        case Seal():
            return _seal(evaluation, authority, event)
        case RecordGrade():
            if evaluation.stage is not Stage.SEALED:
                return Rejected(Rejection.NOT_SEALED, "Evaluation not completed. Call complete first.")
            if evaluation.grade is not None:
                return Accepted(evaluation, evaluation)
            next_value = _replace(evaluation, grade=event.grade)
            return Accepted(next_value, next_value)


def _held_values(event: Event) -> tuple[Rejection, str, Callable[[], Any]] | None:
    """For an event carrying values the evaluation would keep: how to refuse one, what it is, and its values."""
    match event:
        case StartCall() | FinishCall():
            return Rejection.UNSUPPORTED_VALUE, "call's record", event.req.model_dump
        case ChangeEnvironment():
            return Rejection.INVALID_ENVIRONMENT, "change", lambda: event.values
        case ReplaceEnvironment():
            return Rejection.INVALID_ENVIRONMENT, "environment", event.environment.model_dump
        case RecordObservations():
            return Rejection.UNSUPPORTED_VALUE, "observations", event.observed.model_dump
        case _:
            return None


def _unsupported(event: Event) -> Rejected | None:
    """Refuse an event carrying a value a snapshot can't hold (``state.unsupported_value``).

    Decided on the transition's own copy of the event: a value the caller
    changes afterwards never reaches the rules unchecked. An event that
    can't be copied at all (one holding a read-only mapping view, say) is
    refused the same way before this.
    """
    if (held := _held_values(event)) is None:
        return None
    kind, what, values = held
    if (problem := unsupported_value(values())) is None:
        return None
    return Rejected(kind, f"The {what} can't be kept: {problem}")


def _not_text(event: Event) -> Rejected | None:
    """Refuse a turn's message or reply, or the agent's output, that isn't exactly a ``str``."""
    match event:
        case StartTurn():
            return not_text("turn's message", event.message)
        case ReplyTurn():
            return not_text("turn's reply", event.reply)
        case Seal():
            return not_text("agent's output", event.agent_output)
        case _:
            return None


def not_text(what: str, value: Any) -> Rejected | None:
    """An ``unsupported_value`` refusal of ``value`` unless it is exactly a ``str`` (not a subclass); else None."""
    if type(value) is str:
        return None
    kind = type(value)
    return Rejected(
        Rejection.UNSUPPORTED_VALUE, f"The {what} must be a str, not a {kind.__module__}.{kind.__qualname__}"
    )


def _admit(evaluation: Evaluation, authority: Authority, event: Event) -> Rejected | None:
    if isinstance(authority, Host):
        return None
    session = evaluation.session
    if session is None or session.token_hash != authority.token_hash or session.expires_at <= authority.now:
        return Rejected(Rejection.INVALID_SESSION, INVALID_SESSION_DETAIL)
    if not isinstance(event, AGENT_EVENTS):
        return Rejected(Rejection.FORBIDDEN, f"An evaluation session cannot {type(event).__name__}")
    # After the seal, the session only delivers late facts: a read is refused like a closed session.
    if isinstance(event, Read) and evaluation.stage is Stage.SEALED:
        return Rejected(Rejection.SESSION_CLOSED, INVALID_SESSION_DETAIL)
    return None


def _replace(evaluation: Evaluation, **changes: Any) -> Evaluation:
    return dataclasses.replace(evaluation, **changes)


def _on_time(evaluation: Evaluation, event: Event) -> Evaluation:
    """The evaluation as of the event's arrival: a closure wait whose deadline has passed is over.

    Nothing else has to end the wait for the deadline to hold: an event that
    arrives after it finds the evaluation collecting, so a late finish or
    environment write is refused and recorded as a late fact.
    """
    at = getattr(event, "at", None)
    deadline = evaluation.closure_deadline
    if evaluation.stage is not Stage.CLOSING or at is None or deadline is None:
        return evaluation
    if datetime.fromisoformat(at).timestamp() < deadline:
        return evaluation
    return _replace(evaluation, stage=Stage.COLLECTING)


def _open_session(evaluation: Evaluation, event: OpenSession) -> Outcome:
    if evaluation.stage is not Stage.OPEN:
        return Rejected(Rejection.CLOSING, "Cannot open a session for this evaluation")
    effects: tuple[Effect, ...] = (SessionBound(event.token_hash),)
    if evaluation.session is not None:
        effects = (SessionUnbound(evaluation.session.token_hash), *effects)
    next_value = _replace(evaluation, session=Session(event.token_hash, event.expires_at))
    return Accepted(next_value, next_value.session, effects)


# --- late facts ---


def _late(
    evaluation: Evaluation,
    authority: Authority,
    event: str,
    at: str,
    evidence: tuple[str, ...],
    payload: Mapping[str, Any],
    call_id: str | None = None,
) -> Late:
    """Refuse a write that closure stopped, and record it as a late fact (once per distinct write)."""
    by = provenance(evaluation, authority)
    if evaluation.stage is not Stage.SEALED:
        rejected = Rejected(Rejection.CLOSING, f"{CLOSING_DETAIL}: the write is recorded as a late fact")
    elif isinstance(authority, AgentSession):
        rejected = Rejected(Rejection.SESSION_CLOSED, INVALID_SESSION_DETAIL)
    else:
        rejected = Rejected(Rejection.SEALED, SEALED_DETAIL)
    fact = LateFact(
        event=event, stage=evaluation.stage, by=by, at=at, evidence=evidence, payload=payload, call_id=call_id
    )
    if any(_same_write(fact, known) for known in evaluation.late_facts):
        return Late(evaluation, rejected)
    return Late(_replace(evaluation, late_facts=(*evaluation.late_facts, fact)), rejected)


def _same_write(a: LateFact, b: LateFact) -> bool:
    return (a.event, a.by, a.call_id, _canonical(a.payload)) == (b.event, b.by, b.call_id, _canonical(b.payload))


def _canonical(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _call_payload(req: BaseModel) -> dict[str, Any]:
    """A call's request as a late fact keeps it: JSON-compatible data, as ``model_dump(mode="json")`` gives it.

    That gives bytes as UTF-8 text, and raises for bytes that aren't UTF-8;
    here their undecodable bytes are escaped (``\\xff``) instead, so a late
    call is kept whatever bytes its args hold.
    """
    return to_jsonable_python(_as_text(req.model_dump()))


def _as_text(value: Any) -> Any:
    """``value`` with every bytes or bytearray in it decoded as UTF-8, undecodable bytes escaped."""
    kind = type(value)
    if kind in (bytes, bytearray):
        return value.decode("utf-8", "backslashreplace")
    if kind is dict:
        return {_as_text(key): _as_text(item) for key, item in value.items()}
    if kind in (list, tuple, set, frozenset):
        return [_as_text(item) for item in value]
    return value


# --- function calls ---

# The disposition is compared on its own (see ``_same_disposition``): a body that leaves it out matches any.
_BODY_FIELDS = set(FunctionCallBody.model_fields) - {"disposition"}
_START_FIELDS = set(FunctionCallStart.model_fields)
_CALL_EVIDENCE = (FUNCTION_CALLS,)


def canonical_body(call: FunctionCallStart | FunctionCallRecord, fields: set[str] = _BODY_FIELDS) -> str:
    """A call's body in a lossless encoding, so a replay must match it type for type.

    The encoding is the snapshot digest's (``state.canonical_json``). Plain ``==`` would let ``true`` match ``1``. The encoding keeps them apart,
    and ``1`` and ``1.0``, bytes and the text they decode to, a list and a
    tuple too: they're different bodies. It encodes any bytes, UTF-8 or not.
    Mapping keys are sorted, so their order doesn't matter, and an omitted
    field and its explicit default are the same body because the model has
    filled in the default by now. ``fields`` picks the fields compared: the
    whole body, or a start's function and args.
    """
    return canonical_json(call.model_dump(include=fields))


def _same_disposition(a: CallDisposition | None, b: CallDisposition | None) -> bool:
    """Whether two dispositions agree: None says nothing, so it agrees with any."""
    return a is None or b is None or a == b


def _invalid_disposition(disposition: CallDisposition | None, blocked: bool) -> str | None:
    """Why ``disposition`` can't go with ``blocked``, or None when it can."""
    if disposition is None or (disposition in REFUSED_DISPOSITIONS) == blocked:
        return None
    return f"A {disposition} call must have blocked={not blocked}"


def _find_call(evaluation: Evaluation, call_id: str) -> int | None:
    return next((i for i, call in enumerate(evaluation.function_calls) if call.call_id == call_id), None)


def provenance(evaluation: Evaluation, authority: Authority) -> CallProvenance:
    """How a write from ``authority`` reached the evaluation: the host, or its session, by interception mode."""
    if isinstance(authority, Host):
        return "host"
    return "interceptor" if evaluation.interception == "host" else "agent"


def _current_turn(evaluation: Evaluation) -> int | None:
    return evaluation.turns[-1].index if evaluation.turns else None


def _observed(evaluation: Evaluation) -> Environment | None:
    """A copy of the live environment for a record, when the evaluation keeps environment history."""
    return evaluation.environment.model_copy(deep=True) if evaluation.environment_history else None


def _start_call(evaluation: Evaluation, authority: Authority, event: StartCall) -> Outcome:
    req = event.req
    if req.disposition in REFUSED_DISPOSITIONS:
        return Rejected(
            Rejection.INVALID_DISPOSITION,
            f"A {req.disposition} call never runs, so it isn't started: its finish records it whole",
        )
    # A start replays any record with its call_id, finished or not, when the function, args and disposition match.
    if (index := _find_call(evaluation, req.call_id)) is not None:
        existing = evaluation.function_calls[index]
        if canonical_body(existing, _START_FIELDS) == canonical_body(req, _START_FIELDS) and _same_disposition(
            existing.disposition, req.disposition
        ):
            return Accepted(evaluation, (existing, False))
        if evaluation.stage is not Stage.SEALED:
            return Rejected(
                Rejection.CALL_CONFLICT,
                f"call_id {req.call_id!r} is already recorded with another function, args or disposition",
            )
    if evaluation.stage is not Stage.OPEN:
        return _late(evaluation, authority, "start_call", event.at, _CALL_EVIDENCE, _call_payload(req), req.call_id)
    # With environment history, the pre-env is the live environment when the call starts; the finish adds the
    # post-env.
    record = FunctionCallRecord(
        **req.model_dump(),
        complete=False,
        provenance=provenance(evaluation, authority),
        timestamp=event.at,
        # A started call belongs to the turn it started in, even if it finishes later.
        turn=_current_turn(evaluation),
        pre_environment=_observed(evaluation),
    )
    return Accepted(_replace(evaluation, function_calls=(*evaluation.function_calls, record)), (record, True))


def _finish_call(evaluation: Evaluation, authority: Authority, event: FinishCall) -> Outcome:
    req = event.req
    if (invalid := _invalid_disposition(req.disposition, req.blocked)) is not None:
        return Rejected(Rejection.INVALID_DISPOSITION, invalid)
    index = _find_call(evaluation, req.call_id) if req.call_id is not None else None
    if index is not None:
        existing = evaluation.function_calls[index]
        same_disposition = _same_disposition(existing.disposition, req.disposition)
        if existing.complete and canonical_body(existing) == canonical_body(req) and same_disposition:
            return Accepted(evaluation, (existing, False))
        # After the seal, a conflict is a late fact like any other write that isn't an exact replay.
        if evaluation.stage is not Stage.SEALED:
            if existing.complete:
                return Rejected(
                    Rejection.CALL_CONFLICT, f"call_id {req.call_id!r} is already recorded with another body"
                )
            if canonical_body(existing, _START_FIELDS) != canonical_body(req, _START_FIELDS) or not same_disposition:
                return Rejected(
                    Rejection.CALL_CONFLICT,
                    f"call_id {req.call_id!r} was started with another function, args or disposition",
                )
            # A finish that leaves the disposition out keeps the start's: a blocked finish contradicts it.
            if _invalid_disposition(existing.disposition, req.blocked) is not None:
                return Rejected(
                    Rejection.CALL_CONFLICT,
                    f"call_id {req.call_id!r} was started {existing.disposition}, so its finish can't be blocked",
                )
    # An admitted call may finish until the closure wait ends; a new call only while the evaluation is open.
    admitted = (Stage.OPEN, Stage.CLOSING) if index is not None else (Stage.OPEN,)
    if evaluation.stage not in admitted:
        return _late(evaluation, authority, "finish_call", event.at, _CALL_EVIDENCE, _call_payload(req), req.call_id)
    if index is not None:
        # Finish a started call in place, so the record keeps its place in start order.
        started = evaluation.function_calls[index]
        finished = started.model_copy(
            update={
                "result": req.result,
                "error": req.error,
                "blocked": req.blocked,
                "disposition": req.disposition or started.disposition,
                "structured_content": req.structured_content,
                "injection_outcomes": req.injection_outcomes,
                "complete": True,
                "post_environment": _observed(evaluation),
            }
        )
        calls = list(evaluation.function_calls)
        calls[index] = finished
        return Accepted(_replace(evaluation, function_calls=tuple(calls)), (finished, True))
    # With environment history, each call's pre-env is the previous call's
    # post-env, chaining from the eval's initial environment. An unfinished
    # previous call has no post-env, and changes made since its start may be
    # its own, so the pre-env is the live environment: this call's record then
    # shows only what changed while it was being recorded. Each snapshot is its
    # own copy.
    if not evaluation.environment_history:
        pre_env = None
    elif not evaluation.function_calls:
        pre_env = evaluation.pre_environment
    elif (previous_post := evaluation.function_calls[-1].post_environment) is not None:
        pre_env = previous_post
    else:
        pre_env = evaluation.environment.model_copy(deep=True)
    record = FunctionCallRecord(
        **req.model_dump(exclude={"call_id"}),
        call_id=req.call_id or event.assigned_call_id,
        provenance=provenance(evaluation, authority),
        timestamp=event.at,
        turn=_current_turn(evaluation),
        pre_environment=pre_env,
        post_environment=_observed(evaluation),
    )
    return Accepted(_replace(evaluation, function_calls=(*evaluation.function_calls, record)), (record, True))


def _report_capture_failure(evaluation: Evaluation, authority: Authority, event: ReportCaptureFailure) -> Outcome:
    # One lost record is one report, whatever its reason (another attempt's error, say) and whoever sent it.
    identity = (event.call_id, event.function, event.phase)
    for known in evaluation.capture_failures:
        if (known.call_id, known.function, known.phase) == identity:
            return Accepted(evaluation, (known, False))
    if evaluation.stage is Stage.SEALED:
        payload = {"call_id": event.call_id, "function": event.function, "phase": event.phase, "reason": event.reason}
        late = _late(evaluation, authority, "report_capture_failure", event.at, _CALL_EVIDENCE, payload, event.call_id)
        if any(_capture_identity(fact) == identity for fact in evaluation.late_facts):
            return Late(evaluation, late.rejected)
        return late
    by = provenance(evaluation, authority)
    failure = CaptureFailure(
        call_id=event.call_id,
        function=event.function,
        phase=event.phase,
        reason=event.reason,
        stage=evaluation.stage,
        by=by,
        at=event.at,
    )
    next_value = _replace(evaluation, capture_failures=(*evaluation.capture_failures, failure))
    return Accepted(next_value, (failure, True))


def _capture_identity(fact: LateFact) -> tuple[Any, Any, Any] | None:
    """The ``call_id``, function and phase of a late capture failure report, or None for another late fact."""
    if fact.event != "report_capture_failure":
        return None
    return (fact.payload.get("call_id"), fact.payload.get("function"), fact.payload.get("phase"))


# --- turns ---

_TURN_EVIDENCE = ("turns",)


def _start_turn(evaluation: Evaluation, authority: Authority, event: StartTurn) -> Outcome:
    if evaluation.stage is not Stage.OPEN:
        return _late(evaluation, authority, "start_turn", event.at, _TURN_EVIDENCE, {"message": event.message})
    if evaluation.turns and evaluation.turns[-1].ended_at is None:
        return Rejected(Rejection.TURN_PENDING, f"Turn {evaluation.turns[-1].index} has no reply yet")
    turn = Turn(index=len(evaluation.turns), message=event.message, started_at=event.at)
    return Accepted(_replace(evaluation, turns=(*evaluation.turns, turn)), turn)


def _reply_turn(evaluation: Evaluation, authority: Authority, event: ReplyTurn) -> Outcome:
    if evaluation.stage is not Stage.OPEN:
        payload = {"index": event.index, "reply": event.reply}
        return _late(evaluation, authority, "reply_turn", event.at, _TURN_EVIDENCE, payload)
    if event.index < 0 or event.index >= len(evaluation.turns):
        return Rejected(Rejection.TURN_NOT_STARTED, f"Turn not started: {event.index}")
    if evaluation.turns[event.index].ended_at is not None:
        return Rejected(Rejection.TURN_ANSWERED, f"Turn {event.index} already has a reply")
    answered = dataclasses.replace(evaluation.turns[event.index], reply=event.reply, ended_at=event.at)
    turns = list(evaluation.turns)
    turns[event.index] = answered
    return Accepted(_replace(evaluation, turns=tuple(turns)), answered)


# --- environment and observations ---


def _replace_environment(evaluation: Evaluation, authority: Authority, event: ReplaceEnvironment) -> Outcome:
    # A call finishing during the closure wait may still change the environment.
    if evaluation.stage not in (Stage.OPEN, Stage.CLOSING):
        payload = event.environment.model_dump(mode="json")
        return _late(evaluation, authority, "replace_environment", event.at, ("environment",), payload)
    next_value = _replace(
        evaluation, environment=event.environment, environment_revision=evaluation.environment_revision + 1
    )
    return Accepted(next_value, event.environment)


def _change_environment(evaluation: Evaluation, authority: Authority, event: ChangeEnvironment) -> Outcome:
    if evaluation.stage not in (Stage.OPEN, Stage.CLOSING):
        # The payload as JSON-compatible data, whatever an in-process caller sent.
        payload = json.loads(
            json.dumps({"set": event.values, "expected_revision": event.expected_revision}, default=str)
        )
        return _late(evaluation, authority, "change_environment", event.at, ("environment",), payload)
    if event.expected_revision is not None and event.expected_revision != evaluation.environment_revision:
        return Rejected(
            Rejection.REVISION_CONFLICT,
            f"The environment is at revision {evaluation.environment_revision}, not {event.expected_revision}",
        )
    model = type(evaluation.environment)
    if model.model_config.get("extra") != "allow" and (unknown := sorted(set(event.values) - set(model.model_fields))):
        return Rejected(Rejection.INVALID_ENVIRONMENT, f"Unknown environment fields: {unknown}")
    try:
        environment = model.model_validate({**evaluation.environment.model_dump(), **event.values})
    except ValidationError as exc:
        return Rejected(Rejection.INVALID_ENVIRONMENT, str(exc))
    if (problem := unsupported_value(environment.model_dump())) is not None:
        return Rejected(Rejection.INVALID_ENVIRONMENT, f"The changed environment can't be kept: {problem}")
    revision = evaluation.environment_revision + 1
    next_value = _replace(evaluation, environment=environment, environment_revision=revision)
    return Accepted(next_value, (revision, environment))


def _record_observations(evaluation: Evaluation, authority: Authority, event: RecordObservations) -> Outcome:
    if evaluation.stage is Stage.SEALED:
        evidence = tuple(evidence_status({event.source: event.observed})) or (event.source,)
        payload = {"source": event.source, "data": event.observed.model_dump(mode="json")}
        return _late(evaluation, authority, "record_observations", event.at, evidence, payload)
    next_value = _replace(evaluation, observations={**evaluation.observations, event.source: event.observed})
    return Accepted(next_value, next_value.observations)


# --- closure and the seal ---


def _end_closure_wait(evaluation: Evaluation, event: EndClosureWait) -> Outcome:
    if evaluation.stage is not Stage.CLOSING:
        return Accepted(evaluation, evaluation)
    unfinished = [call.call_id for call in evaluation.function_calls if not call.complete]
    deadline = evaluation.closure_deadline if evaluation.closure_deadline is not None else event.now
    if unfinished and event.now < deadline:
        return Rejected(
            Rejection.CLOSURE_PENDING, f"Waiting for {len(unfinished)} admitted call(s) to finish: {unfinished}"
        )
    next_value = _replace(evaluation, stage=Stage.COLLECTING)
    return Accepted(next_value, next_value)


def _capture_status(evaluation: Evaluation) -> dict[str, EvidenceStatus]:
    """The call capture's status, when the records alone can't show it is complete.

    An interceptor on the host sees every call the target makes through it, so
    no record means no call. An interception layer in the agent may fail open:
    when it can't reach the control plane, the tool still runs and nothing is
    recorded. So an evaluation intercepted in the agent with no record at all
    has an incomplete capture, never a clean one.
    """
    if evaluation.interception == "agent" and not evaluation.function_calls:
        return {"function_calls": "incomplete"}
    return {}


def _seal(evaluation: Evaluation, authority: Authority, event: Seal) -> Outcome:
    if evaluation.stage is Stage.SEALED:
        assert evaluation.snapshot is not None
        # Completing again with the same output is a retry; another output is a late fact.
        if evaluation.snapshot.agent_output == event.agent_output:
            return Accepted(evaluation, evaluation.snapshot)
        payload = {"agent_output": event.agent_output}
        return _late(evaluation, authority, "seal", event.at, ("agent_output",), payload)
    if evaluation.stage is not Stage.COLLECTING:
        return Rejected(Rejection.CLOSURE_PENDING, "Closure has not finished waiting for admitted calls")
    # The snapshot keeps its own copies, so nothing reached through the sealed evaluation changes it.
    pre_environment, environment, function_calls, turns, observations, capture_failures = copy.deepcopy(
        (
            evaluation.pre_environment,
            evaluation.environment,
            evaluation.function_calls,
            evaluation.turns,
            dict(evaluation.observations),
            evaluation.capture_failures,
        )
    )
    snapshot = Snapshot(
        evaluation_id=evaluation.id,
        run_id=evaluation.run_id,
        user_task_id=evaluation.user_task_id,
        injection_task_id=evaluation.injection_task_id,
        agent_output=event.agent_output,
        pre_environment=pre_environment,
        environment=environment,
        function_calls=function_calls,
        turns=turns,
        observations=observations,
        evidence_status={
            **evidence_status(evaluation.observations),
            **_capture_status(evaluation),
            **late_evidence(evaluation.late_facts),
            **capture_evidence(evaluation.capture_failures),
            **{key: "incomplete" for key in event.incomplete},
        },
        sealed_at=event.at,
        capture_failures=capture_failures,
    )
    next_value = _replace(evaluation, stage=Stage.SEALED, snapshot=snapshot)
    return Accepted(next_value, snapshot)
