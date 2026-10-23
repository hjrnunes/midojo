"""In-memory persistence seam with immutable evaluation session bindings.

The store has no locks. It relies on the control plane declaring every route
handler and dependency ``async def``: FastAPI runs those on a single event loop,
and a handler only gives up control at an ``await``. Store methods never await,
so each handler's session check and the write that depends on it run without
another request in between.

A ``def`` handler would instead run on FastAPI's thread pool, where Python can
switch threads between any two steps. A late agent callback could then race the
orchestrator completing its evaluation::

    thread 1: POST /agent/function-calls   thread 2: POST /runs/.../complete
      look up token -> valid
            -- thread switch --
                                             revoke token, completed = True
                                             (orchestrator calls /grade)
            -- switch back --
      append function call   <- writes to an evaluation that's already completed

On the event loop the callback either finishes before completion or finds its
token revoked and gets a 401. Keep a check and its dependent write in the same
handler, with no ``await`` between them. ``tests/test_sessions.py`` rejects sync
handlers and dependencies. A persistent store can implement the same Store
contract with transactions.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
import uuid
from datetime import UTC, datetime
from typing import Protocol

from pydantic import BaseModel

from midojo.observations import EvidenceStatus
from midojo.types import CheckValidity, Environment, FunctionCallRecord, GradeValidity, SuiteName, ValidityReason

from .models import CreateFunctionCallRecord, CreateFunctionCallStart, FunctionCallBody, FunctionCallStart
from .state import Evaluation, Run, Turn


def _new_id() -> str:
    return uuid.uuid4().hex


class Store(Protocol):
    """Interface for run/evaluation persistence."""

    # --- runs ---
    def create_run(self, suite_name: SuiteName) -> Run: ...
    def get_run(self, run_id: str) -> Run | None: ...
    def list_runs(self) -> list[Run]: ...

    # --- evaluations ---
    def create_evaluation(
        self,
        run_id: str,
        *,
        user_task_id: str,
        injection_task_id: str | None,
        pre_environment: Environment,
        environment: Environment,
        active_injections: dict[str, str],
        agent_input: str | None = None,
    ) -> Evaluation: ...
    def get_evaluation(self, run_id: str, eval_id: str) -> Evaluation | None: ...
    def create_session(self, run_id: str, eval_id: str, ttl_seconds: int) -> tuple[str, str]: ...
    def close_session(self, run_id: str, eval_id: str) -> None: ...
    def session_evaluation(self, session_token: str) -> Evaluation: ...

    # --- per-evaluation mutations (ID-based) ---
    # Each returns None if (run_id, eval_id) is unknown, else the mutated
    # evaluation, except start_function_call and append_function_call: they
    # return (record, created), with created False when req replays a stored
    # call_id, and raise FunctionCallConflictError when the call_id is stored
    # with another body. append_function_call finishes an unfinished record
    # with req's call_id in place (created True) instead of appending.
    def start_function_call(
        self, run_id: str, eval_id: str, req: CreateFunctionCallStart
    ) -> tuple[FunctionCallRecord, bool] | None: ...
    def append_function_call(
        self, run_id: str, eval_id: str, req: CreateFunctionCallRecord
    ) -> tuple[FunctionCallRecord, bool] | None: ...
    def start_turn(self, run_id: str, eval_id: str, message: str) -> Evaluation | None: ...
    def record_turn_reply(self, run_id: str, eval_id: str, index: int, reply: str) -> Evaluation | None: ...
    def set_environment(self, run_id: str, eval_id: str, environment: Environment) -> Evaluation | None: ...
    def record_observations(self, run_id: str, eval_id: str, source: str, observed: BaseModel) -> Evaluation | None: ...
    def set_grade(
        self,
        run_id: str,
        eval_id: str,
        *,
        utility: bool | None,
        security: bool | None,
        security_reason: str | None = None,
        utility_reason: str | None = None,
        grade_validity: GradeValidity | None = None,
        evidence_status: dict[str, EvidenceStatus] | None = None,
    ) -> Evaluation | None:
        """Record the evaluation's grades and mark it graded.

        Without ``grade_validity``, the validity is derived from the values: a
        decided grade is valid and an unknown (None) one invalid, and security is
        not applicable without an injection task. Reasons are stored as given; the
        control plane's grade route always passes them, but a caller that omits
        them stores None.
        """
        ...

    def complete_evaluation(self, run_id: str, eval_id: str, agent_output: str) -> Evaluation | None: ...


class InvalidSessionError(ValueError):
    """A callback has no live evaluation binding."""


class FunctionCallConflictError(ValueError):
    """A function call record reuses a call_id already stored with a different body."""


_BODY_FIELDS = set(FunctionCallBody.model_fields)
_START_FIELDS = set(FunctionCallStart.model_fields)


def _canonical_body(call: FunctionCallStart | FunctionCallRecord, fields: set[str] = _BODY_FIELDS) -> str:
    """A call's body as canonical JSON, so a replay must match it type for type.

    Plain ``==`` would let ``true`` match ``1``. JSON keeps them apart, and ``1``
    and ``1.0`` too: they're different bodies. Object keys are sorted, so their
    order doesn't matter, and an omitted field and its explicit default are the
    same body because the model has filled in the default by now. ``fields``
    picks the fields compared: the whole body, or a start's function and args.
    """
    return json.dumps(call.model_dump(mode="json", include=fields), sort_keys=True, separators=(",", ":"))


def _find_call(evaluation: Evaluation, call_id: str) -> FunctionCallRecord | None:
    return next((call for call in evaluation.function_calls if call.call_id == call_id), None)


def _derived_validity(evaluation: Evaluation, utility: bool | None, security: bool | None) -> GradeValidity:
    """The validity of grades recorded without one (see :meth:`Store.set_grade`)."""

    def check_validity(check: str, value: bool | None, reason: str | None) -> CheckValidity:
        if value is not None:
            return CheckValidity(status="valid")
        return CheckValidity(
            status="invalid", reasons=[ValidityReason(check=reason or check, reason="the grade is unknown")]
        )

    return GradeValidity(
        utility=check_validity("utility", utility, evaluation.utility_reason),
        security=(
            check_validity("security", security, evaluation.security_reason)
            if evaluation.injection_task_id is not None
            else CheckValidity(status="not_applicable")
        ),
    )


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class InMemoryStore:
    """Process-local, in-memory :class:`Store`. State is lost on restart.

    Not thread-safe: callers must run on one event loop (see the module docstring).
    """

    def __init__(self) -> None:
        self._runs: dict[str, Run] = {}
        self._evaluation_ids: set[str] = set()
        self._sessions: dict[str, tuple[str, str, float]] = {}
        self._session_tokens: dict[tuple[str, str], set[str]] = {}

    def _remove_sessions(self, run_id: str, eval_id: str) -> None:
        for token_hash in self._session_tokens.pop((run_id, eval_id), set()):
            self._sessions.pop(token_hash, None)

    # --- runs ---

    def create_run(self, suite_name: SuiteName) -> Run:
        run = Run(id=_new_id(), suite_name=suite_name)
        self._runs[run.id] = run
        return run

    def get_run(self, run_id: str) -> Run | None:
        return self._runs.get(run_id)

    def list_runs(self) -> list[Run]:
        return list(self._runs.values())

    # --- evaluations ---

    def create_evaluation(
        self,
        run_id: str,
        *,
        user_task_id: str,
        injection_task_id: str | None,
        pre_environment: Environment,
        environment: Environment,
        active_injections: dict[str, str],
        agent_input: str | None = None,
    ) -> Evaluation:
        run = self._runs[run_id]
        # Keep evaluation IDs short and unique across every run in this store.
        eval_id = secrets.token_hex(5)
        while eval_id in self._evaluation_ids:
            eval_id = secrets.token_hex(5)
        evaluation = Evaluation(
            id=eval_id,
            run_id=run_id,
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            pre_environment=pre_environment,
            environment=environment,
            active_injections=active_injections,
            agent_input=agent_input,
        )
        run.evaluations[evaluation.id] = evaluation
        self._evaluation_ids.add(evaluation.id)
        return evaluation

    def get_evaluation(self, run_id: str, eval_id: str) -> Evaluation | None:
        run = self._runs.get(run_id)
        if run is None:
            return None
        return run.evaluations.get(eval_id)

    def create_session(self, run_id: str, eval_id: str, ttl_seconds: int) -> tuple[str, str]:
        evaluation = self.get_evaluation(run_id, eval_id)
        if evaluation is None or evaluation.completed:
            raise InvalidSessionError("Cannot open a session for this evaluation")
        if ttl_seconds <= 0:
            raise ValueError("Session TTL must be positive")
        self._remove_sessions(run_id, eval_id)
        token = secrets.token_urlsafe(32)
        expires = time.time() + ttl_seconds
        token_hash = _token_hash(token)
        self._sessions[token_hash] = (run_id, eval_id, expires)
        self._session_tokens.setdefault((run_id, eval_id), set()).add(token_hash)
        return token, datetime.fromtimestamp(expires, UTC).isoformat()

    def close_session(self, run_id: str, eval_id: str) -> None:
        self._remove_sessions(run_id, eval_id)

    def session_evaluation(self, session_token: str) -> Evaluation:
        """Return the evaluation bound to a live session token."""
        token_hash = _token_hash(session_token)
        binding = self._sessions.get(token_hash)
        if binding is None or binding[2] <= time.time():
            if binding is not None:
                self._sessions.pop(token_hash)
                self._session_tokens.get(binding[:2], set()).discard(token_hash)
            raise InvalidSessionError("Invalid, expired, or closed evaluation session")
        evaluation = self.get_evaluation(binding[0], binding[1])
        if evaluation is None or evaluation.completed:
            raise InvalidSessionError("Invalid, expired, or closed evaluation session")
        return evaluation

    # --- per-evaluation mutations ---

    def start_function_call(
        self, run_id: str, eval_id: str, req: CreateFunctionCallStart
    ) -> tuple[FunctionCallRecord, bool] | None:
        evaluation = self.get_evaluation(run_id, eval_id)
        if evaluation is None:
            return None
        # A start replays any record with its call_id, finished or not, when
        # the function and args match. As in append_function_call, the lookup
        # and the append run without an await between them.
        existing = _find_call(evaluation, req.call_id)
        if existing is not None:
            if _canonical_body(existing, _START_FIELDS) != _canonical_body(req, _START_FIELDS):
                raise FunctionCallConflictError(
                    f"call_id {req.call_id!r} is already recorded with another function or args"
                )
            return existing, False
        # The pre-env is the live environment when the call starts; the finish
        # adds the post-env.
        record = FunctionCallRecord(
            **req.model_dump(),
            complete=False,
            timestamp=datetime.now(UTC).isoformat(),
            # A started call belongs to the turn it started in, even if it finishes later.
            turn=evaluation.turns[-1].index if evaluation.turns else None,
            pre_environment=evaluation.environment.model_copy(deep=True),
        )
        evaluation.function_calls.append(record)
        return record, True

    def append_function_call(
        self, run_id: str, eval_id: str, req: CreateFunctionCallRecord
    ) -> tuple[FunctionCallRecord, bool] | None:
        evaluation = self.get_evaluation(run_id, eval_id)
        if evaluation is None:
            return None
        # At most one record per call_id. The lookup and the append below run
        # without an await between them, so a concurrent retry can't slip in.
        existing = _find_call(evaluation, req.call_id) if req.call_id is not None else None
        if existing is not None and existing.complete:
            if _canonical_body(existing) != _canonical_body(req):
                raise FunctionCallConflictError(f"call_id {req.call_id!r} is already recorded with another body")
            return existing, False
        if existing is not None:
            # Finish a started call in place, so the record keeps its place in start order.
            if _canonical_body(existing, _START_FIELDS) != _canonical_body(req, _START_FIELDS):
                raise FunctionCallConflictError(f"call_id {req.call_id!r} was started with another function or args")
            existing.result = req.result
            existing.error = req.error
            existing.blocked = req.blocked
            existing.complete = True
            existing.post_environment = evaluation.environment.model_copy(deep=True)
            return existing, True
        # Each call's pre-env is the previous call's post-env, chaining from the
        # eval's initial environment; post-env is a deep copy so later mutations
        # don't retroactively change the recorded snapshot. An unfinished
        # previous call has no post-env, and changes made since its start may be
        # its own, so the pre-env is the live environment: this call's record
        # then shows only what changed while it was being recorded.
        if not evaluation.function_calls:
            pre_env = evaluation.pre_environment
        elif (previous_post := evaluation.function_calls[-1].post_environment) is not None:
            pre_env = previous_post
        else:
            pre_env = evaluation.environment.model_copy(deep=True)
        record = FunctionCallRecord(
            **req.model_dump(exclude={"call_id"}),
            call_id=req.call_id or str(uuid.uuid4()),
            timestamp=datetime.now(UTC).isoformat(),
            turn=evaluation.turns[-1].index if evaluation.turns else None,
            pre_environment=pre_env,
            post_environment=evaluation.environment.model_copy(deep=True),
        )
        evaluation.function_calls.append(record)
        return record, True

    def start_turn(self, run_id: str, eval_id: str, message: str) -> Evaluation | None:
        evaluation = self.get_evaluation(run_id, eval_id)
        if evaluation is None:
            return None
        evaluation.turns.append(Turn(index=len(evaluation.turns), message=message))
        return evaluation

    def record_turn_reply(self, run_id: str, eval_id: str, index: int, reply: str) -> Evaluation | None:
        evaluation = self.get_evaluation(run_id, eval_id)
        if evaluation is None:
            return None
        turn = evaluation.turns[index]
        turn.reply = reply
        turn.ended_at = datetime.now(UTC).isoformat()
        return evaluation

    def set_environment(self, run_id: str, eval_id: str, environment: Environment) -> Evaluation | None:
        evaluation = self.get_evaluation(run_id, eval_id)
        if evaluation is None:
            return None
        evaluation.environment = environment
        return evaluation

    def record_observations(self, run_id: str, eval_id: str, source: str, observed: BaseModel) -> Evaluation | None:
        evaluation = self.get_evaluation(run_id, eval_id)
        if evaluation is None:
            return None
        evaluation.observations[source] = observed
        return evaluation

    def set_grade(
        self,
        run_id: str,
        eval_id: str,
        *,
        utility: bool | None,
        security: bool | None,
        security_reason: str | None = None,
        utility_reason: str | None = None,
        grade_validity: GradeValidity | None = None,
        evidence_status: dict[str, EvidenceStatus] | None = None,
    ) -> Evaluation | None:
        evaluation = self.get_evaluation(run_id, eval_id)
        if evaluation is None:
            return None
        evaluation.utility = utility
        evaluation.utility_reason = utility_reason
        evaluation.security = security
        evaluation.security_reason = security_reason
        evaluation.grade_validity = grade_validity or _derived_validity(evaluation, utility, security)
        evaluation.evidence_status = dict(evidence_status or {})
        evaluation.graded = True
        return evaluation

    def complete_evaluation(self, run_id: str, eval_id: str, agent_output: str) -> Evaluation | None:
        evaluation = self.get_evaluation(run_id, eval_id)
        if evaluation is None:
            return None
        self._remove_sessions(run_id, eval_id)
        evaluation.agent_output = agent_output
        evaluation.completed = True
        return evaluation
