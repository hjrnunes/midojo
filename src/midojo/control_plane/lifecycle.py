"""Evaluation lifecycle rules as pure transitions.

``transition(evaluation, authority, event)`` decides one event against one
evaluation's current value. It returns:

* :class:`Accepted`: the next value, the event's result, and the effects the
  writer applies outside the evaluation (the session token index);
* :class:`Late`: the event is refused, and the next value records it as a late
  fact, so the evaluation shows its evidence is incomplete;
* :class:`Rejected`: the event is refused and leaves nothing behind.

It never changes its arguments: every change is a new value. It reads nothing
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
  conflict;
* unfinished calls: a start appends an unfinished record that the finish
  completes in place, keeping its start order and its start's turn;
* turns: a call carries the index of the latest started turn, a turn starts
  only once the previous one has its reply, and a turn gets one reply;
* staged closure: ``open`` admits everything; ``closing`` admits no new call
  but lets admitted calls finish until none is unfinished or the closure wait
  ends, judged by each event's arrival time against the deadline;
  ``collecting`` takes the final observations; ``sealed`` fixes the snapshot
  that grading reads;
* late facts: a write that closure refuses is recorded as a late fact, never
  applied, and marks the evidence it carries ``incomplete``;
* the grade: recorded once, from the snapshot, and never replaced.

The single writer (:mod:`midojo.control_plane.service`) applies transitions:
it holds the evaluation's lock across look-up, transition and commit.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from midojo.observations import evidence_status
from midojo.types import Environment, FunctionCallRecord

from .models import (
    CreateFunctionCallRecord,
    CreateFunctionCallStart,
    FunctionCallBody,
    FunctionCallStart,
    GradeResponse,
)
from .state import Evaluation, LateFact, Session, Snapshot, Stage, Turn, late_evidence

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
    TURN_PENDING = "turn_pending"
    TURN_NOT_STARTED = "turn_not_started"
    TURN_ANSWERED = "turn_answered"


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
    """Take the snapshot, with the agent's final output."""

    agent_output: str
    at: str


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
    | StartTurn
    | ReplyTurn
    | ReplaceEnvironment
    | RecordObservations
    | BeginClosure
    | EndClosureWait
    | Seal
    | RecordGrade
)

# What an agent's session may do. Everything else is the host's.
AGENT_EVENTS = (Read, StartCall, FinishCall, ReplaceEnvironment)

INVALID_SESSION_DETAIL = "Invalid, expired, or closed evaluation session"
SEALED_DETAIL = "Evaluation already completed"
CLOSING_DETAIL = "Evaluation is closing"


# --- the transition ---


def transition(evaluation: Evaluation, authority: Authority, event: Event) -> Outcome:
    if (rejected := _admit(evaluation, authority, event)) is not None:
        return rejected
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
        case StartTurn():
            return _start_turn(evaluation, authority, event)
        case ReplyTurn():
            return _reply_turn(evaluation, authority, event)
        case ReplaceEnvironment():
            return _replace_environment(evaluation, authority, event)
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
    by = "agent" if isinstance(authority, AgentSession) else "host"
    if evaluation.stage is not Stage.SEALED:
        rejected = Rejected(Rejection.CLOSING, f"{CLOSING_DETAIL}: the write is recorded as a late fact")
    elif by == "agent":
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


# --- function calls ---

_BODY_FIELDS = set(FunctionCallBody.model_fields)
_START_FIELDS = set(FunctionCallStart.model_fields)
_CALL_EVIDENCE = ("function_calls",)


def canonical_body(call: FunctionCallStart | FunctionCallRecord, fields: set[str] = _BODY_FIELDS) -> str:
    """A call's body as canonical JSON, so a replay must match it type for type.

    Plain ``==`` would let ``true`` match ``1``. JSON keeps them apart, and ``1``
    and ``1.0`` too: they're different bodies. Object keys are sorted, so their
    order doesn't matter, and an omitted field and its explicit default are the
    same body because the model has filled in the default by now. ``fields``
    picks the fields compared: the whole body, or a start's function and args.
    """
    return json.dumps(call.model_dump(mode="json", include=fields), sort_keys=True, separators=(",", ":"))


def _find_call(evaluation: Evaluation, call_id: str) -> int | None:
    return next((i for i, call in enumerate(evaluation.function_calls) if call.call_id == call_id), None)


def _current_turn(evaluation: Evaluation) -> int | None:
    return evaluation.turns[-1].index if evaluation.turns else None


def _start_call(evaluation: Evaluation, authority: Authority, event: StartCall) -> Outcome:
    req = event.req
    # A start replays any record with its call_id, finished or not, when the function and args match.
    if (index := _find_call(evaluation, req.call_id)) is not None:
        existing = evaluation.function_calls[index]
        if canonical_body(existing, _START_FIELDS) != canonical_body(req, _START_FIELDS):
            return Rejected(
                Rejection.CALL_CONFLICT, f"call_id {req.call_id!r} is already recorded with another function or args"
            )
        return Accepted(evaluation, (existing, False))
    if evaluation.stage is not Stage.OPEN:
        return _late(
            evaluation, authority, "start_call", event.at, _CALL_EVIDENCE, req.model_dump(mode="json"), req.call_id
        )
    # The pre-env is the live environment when the call starts; the finish adds the post-env.
    record = FunctionCallRecord(
        **req.model_dump(),
        complete=False,
        timestamp=event.at,
        # A started call belongs to the turn it started in, even if it finishes later.
        turn=_current_turn(evaluation),
        pre_environment=evaluation.environment.model_copy(deep=True),
    )
    return Accepted(_replace(evaluation, function_calls=(*evaluation.function_calls, record)), (record, True))


def _finish_call(evaluation: Evaluation, authority: Authority, event: FinishCall) -> Outcome:
    req = event.req
    index = _find_call(evaluation, req.call_id) if req.call_id is not None else None
    if index is not None:
        existing = evaluation.function_calls[index]
        if existing.complete:
            if canonical_body(existing) != canonical_body(req):
                return Rejected(
                    Rejection.CALL_CONFLICT, f"call_id {req.call_id!r} is already recorded with another body"
                )
            return Accepted(evaluation, (existing, False))
        if canonical_body(existing, _START_FIELDS) != canonical_body(req, _START_FIELDS):
            return Rejected(
                Rejection.CALL_CONFLICT, f"call_id {req.call_id!r} was started with another function or args"
            )
    # An admitted call may finish until the closure wait ends; a new call only while the evaluation is open.
    admitted = (Stage.OPEN, Stage.CLOSING) if index is not None else (Stage.OPEN,)
    if evaluation.stage not in admitted:
        return _late(
            evaluation, authority, "finish_call", event.at, _CALL_EVIDENCE, req.model_dump(mode="json"), req.call_id
        )
    if index is not None:
        # Finish a started call in place, so the record keeps its place in start order.
        finished = evaluation.function_calls[index].model_copy(
            update={
                "result": req.result,
                "error": req.error,
                "blocked": req.blocked,
                "complete": True,
                "post_environment": evaluation.environment.model_copy(deep=True),
            }
        )
        calls = list(evaluation.function_calls)
        calls[index] = finished
        return Accepted(_replace(evaluation, function_calls=tuple(calls)), (finished, True))
    # Each call's pre-env is the previous call's post-env, chaining from the
    # eval's initial environment. An unfinished previous call has no post-env,
    # and changes made since its start may be its own, so the pre-env is the
    # live environment: this call's record then shows only what changed while
    # it was being recorded. Each snapshot is its own copy.
    if not evaluation.function_calls:
        pre_env = evaluation.pre_environment
    elif (previous_post := evaluation.function_calls[-1].post_environment) is not None:
        pre_env = previous_post
    else:
        pre_env = evaluation.environment.model_copy(deep=True)
    record = FunctionCallRecord(
        **req.model_dump(exclude={"call_id"}),
        call_id=req.call_id or event.assigned_call_id,
        timestamp=event.at,
        turn=_current_turn(evaluation),
        pre_environment=pre_env,
        post_environment=evaluation.environment.model_copy(deep=True),
    )
    return Accepted(_replace(evaluation, function_calls=(*evaluation.function_calls, record)), (record, True))


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
    next_value = _replace(evaluation, environment=event.environment)
    return Accepted(next_value, event.environment)


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
    snapshot = Snapshot(
        user_task_id=evaluation.user_task_id,
        injection_task_id=evaluation.injection_task_id,
        agent_output=event.agent_output,
        pre_environment=evaluation.pre_environment,
        environment=evaluation.environment,
        function_calls=evaluation.function_calls,
        turns=evaluation.turns,
        observations=evaluation.observations,
        evidence_status={**evidence_status(evaluation.observations), **late_evidence(evaluation.late_facts)},
        sealed_at=event.at,
    )
    next_value = _replace(evaluation, stage=Stage.SEALED, snapshot=snapshot)
    return Accepted(next_value, snapshot)
