"""The control plane's operations, applied by a single writer per evaluation.

:class:`ControlPlane` is what the HTTP routes call, and what an in-process
caller (a host-side recorder, an orchestrator in the same process) calls in
their place. Every write to an evaluation goes through :meth:`ControlPlane.apply`,
which holds that evaluation's ``threading.Lock`` across the store's look-up,
the pure transition (:mod:`midojo.control_plane.lifecycle`) and the commit. So
no two events on one evaluation interleave, whatever thread or event loop sends
them, and events on different evaluations don't wait for each other.

``apply`` is synchronous and only decides: the lock is never held across an
``await``, a tool call or grading. The closure wait (:meth:`ControlPlane.close`)
sleeps between short ``apply`` calls, and grading reads the sealed snapshot,
grades it with no lock held, then records the grade in one more ``apply``.

The stored evaluation never aliases a caller's objects, in either direction:
``apply`` copies each event in before the transition, and copies its result out
before it returns; every read returns a copy.

Outcomes are domain values: a refusal raises :class:`RejectedError` with its
:class:`Rejection` kind, and only the HTTP edge (``dependencies.py``) turns a
kind into a status code.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from midojo.types import Environment, FunctionCallRecord, SuiteName
from midojo.yaml_task_suite import YAMLTaskSuite

from . import lifecycle
from .config import AppConfig
from .grading import grade_snapshot
from .lifecycle import HOST, Accepted, AgentSession, Event, Late, Rejected, RejectedError, Rejection
from .models import CreateEvaluationResponse, CreateFunctionCallRecord, CreateFunctionCallStart, GradeResponse
from .state import Evaluation, Run, Snapshot, Stage, Turn
from .store import InMemoryStore, Store, detached

# How often the closure wait checks for unfinished calls, in seconds.
CLOSURE_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class ById:
    """An evaluation addressed by its IDs, by the host (authorized at the edge)."""

    run_id: str
    eval_id: str


@dataclass(frozen=True)
class BySession:
    """The evaluation an interception layer's session token is bound to."""

    token: str


Target = ById | BySession


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _reject(kind: Rejection, detail: str) -> RejectedError:
    return RejectedError(Rejected(kind, detail))


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class ControlPlane:
    def __init__(
        self,
        suites: Mapping[SuiteName, YAMLTaskSuite],
        *,
        store: Store | None = None,
        config: AppConfig | None = None,
    ) -> None:
        self.suites = dict(suites)
        self.store: Store = store if store is not None else InMemoryStore()
        self.config = config if config is not None else AppConfig()
        self._locks: dict[tuple[str, str], threading.Lock] = {}
        self._locks_guard = threading.Lock()

    # --- the single writer ---

    def _lock(self, run_id: str, eval_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault((run_id, eval_id), threading.Lock())

    def _resolve(self, target: Target) -> tuple[str, str, lifecycle.Authority]:
        if isinstance(target, BySession):
            hashed = token_hash(target.token)
            bound = self.store.session_target(hashed)
            if bound is None:
                raise _reject(Rejection.INVALID_SESSION, lifecycle.INVALID_SESSION_DETAIL)
            return bound[0], bound[1], AgentSession(hashed, time.time())
        if self.store.get_run(target.run_id) is None:
            raise _reject(Rejection.UNKNOWN_RUN, f"Unknown run: {target.run_id}")
        return target.run_id, target.eval_id, HOST

    def apply(self, target: Target, event: Event) -> Any:
        """Decide ``event`` on the target's evaluation and commit it, as one step; return its result.

        Raises :class:`RejectedError` when the event is refused, with ``late``
        True when it was recorded as a late fact.
        """
        run_id, eval_id, authority = self._resolve(target)
        event = detached(event)
        with self._lock(run_id, eval_id):
            evaluation = self.store.evaluation_value(run_id, eval_id)
            if evaluation is None:
                raise _reject(Rejection.UNKNOWN_EVALUATION, f"Unknown evaluation: {eval_id}")
            outcome = lifecycle.transition(evaluation, authority, event)
            if isinstance(outcome, Rejected):
                raise RejectedError(outcome)
            if outcome.evaluation is not evaluation:
                self.store.commit_evaluation(outcome.evaluation)
            if isinstance(outcome, Late):
                raise RejectedError(outcome.rejected, late=True)
            self._apply_effects(outcome, run_id, eval_id)
            return detached(outcome.result)

    def _apply_effects(self, outcome: Accepted, run_id: str, eval_id: str) -> None:
        for effect in outcome.effects:
            if isinstance(effect, lifecycle.SessionBound):
                self.store.bind_session(effect.token_hash, run_id, eval_id)
            else:
                self.store.unbind_session(effect.token_hash)

    # --- reads ---

    def evaluation(self, target: Target) -> Evaluation:
        """A copy of the target's evaluation, as of now."""
        result = self.apply(target, lifecycle.Read(_now()))
        assert isinstance(result, Evaluation)
        return result

    def run(self, run_id: str) -> Run:
        run = self.store.get_run(run_id)
        if run is None:
            raise _reject(Rejection.UNKNOWN_RUN, f"Unknown run: {run_id}")
        return run

    def evaluations(self, run_id: str) -> list[Evaluation]:
        self.run(run_id)
        return self.store.list_evaluations(run_id)

    def suite(self, run_id: str) -> YAMLTaskSuite:
        """The suite of ``run_id``. A run's suite never changes."""
        return self.suites[self.run(run_id).suite_name]

    # --- runs and evaluations ---

    def create_run(self, suite_name: SuiteName) -> Run:
        return self.store.create_run(suite_name)

    def create_evaluation(
        self,
        run_id: str,
        user_task_id: str,
        injection_task_id: str | None = None,
        injections: Mapping[str, str] | None = None,
    ) -> CreateEvaluationResponse:
        """Create an evaluation and open its session. The response is the only place the token appears."""
        injections = detached(dict(injections or {}))
        suite = self.suite(run_id)
        if user_task_id not in suite.user_tasks:
            raise _reject(Rejection.UNKNOWN_TASK, f"Unknown user task: {user_task_id}")
        if injection_task_id is not None and injection_task_id not in suite.injection_tasks:
            raise _reject(Rejection.UNKNOWN_TASK, f"Unknown injection task: {injection_task_id}")
        environment = suite.provision_environment(injections)
        turns = suite.inject_user_task_turns(user_task_id, injections)
        evaluation = Evaluation(
            id=self.store.new_evaluation_id(),
            run_id=run_id,
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            pre_environment=environment.model_copy(deep=True),
            environment=environment,
            active_injections=injections,
            agent_input=turns[0],
        )
        self.store.insert_evaluation(evaluation)
        token = secrets.token_urlsafe(32)
        expires = time.time() + self.config.session_ttl_seconds
        self.apply(ById(run_id, evaluation.id), lifecycle.OpenSession(token_hash(token), expires))
        return CreateEvaluationResponse(
            id=evaluation.id,
            prompt=turns[0],
            turns=turns,
            session_token=token,
            session_expires_at=datetime.fromtimestamp(expires, UTC).isoformat(),
        )

    def close_session(self, run_id: str, eval_id: str) -> None:
        self.apply(ById(run_id, eval_id), lifecycle.CloseSession())

    # --- turns, calls, environment and observations ---

    def start_turn(self, run_id: str, eval_id: str, message: str) -> Turn:
        return self.apply(ById(run_id, eval_id), lifecycle.StartTurn(message, _now()))

    def record_turn_reply(self, run_id: str, eval_id: str, index: int, reply: str) -> Turn:
        return self.apply(ById(run_id, eval_id), lifecycle.ReplyTurn(index, reply, _now()))

    def start_call(self, target: Target, req: CreateFunctionCallStart) -> tuple[FunctionCallRecord, bool]:
        """Start a call: ``(record, created)``, with created False when ``req`` replays its ``call_id``."""
        return self.apply(target, lifecycle.StartCall(req, _now()))

    def finish_call(self, target: Target, req: CreateFunctionCallRecord) -> tuple[FunctionCallRecord, bool]:
        """Record a call, or finish a started one: ``(record, created)``, created False on a replay."""
        return self.apply(target, lifecycle.FinishCall(req, _now()))

    def replace_environment(self, target: Target, environment: Environment) -> Environment:
        """Install ``environment``, already validated against the evaluation's suite."""
        return self.apply(target, lifecycle.ReplaceEnvironment(environment, _now()))

    def record_observations(self, run_id: str, eval_id: str, source: str, observed: BaseModel) -> dict[str, BaseModel]:
        """Record ``source``'s observations, already validated against the model registered for it."""
        return dict(self.apply(ById(run_id, eval_id), lifecycle.RecordObservations(source, observed, _now())))

    # --- staged closure ---

    async def close(self, run_id: str, eval_id: str) -> Evaluation:
        """Stop admitting new calls, and wait for admitted calls to finish, up to the closure wait.

        Returns once the evaluation is collecting its final observations (or
        already sealed). If this is cancelled, the evaluation stays closing, and
        the next call waits for what is left of the same closure wait.
        """
        target = ById(run_id, eval_id)
        deadline = time.time() + self.config.closure_wait_seconds
        self.apply(target, lifecycle.BeginClosure(_now(), deadline))
        while True:
            try:
                return self.apply(target, lifecycle.EndClosureWait(time.time()))
            except RejectedError as exc:
                if exc.kind is not Rejection.CLOSURE_PENDING:
                    raise
            evaluation = self.evaluation(target)
            remaining = (evaluation.closure_deadline or 0.0) - time.time()
            await asyncio.sleep(max(0.0, min(CLOSURE_POLL_SECONDS, remaining)))

    async def complete(self, run_id: str, eval_id: str, agent_output: str) -> Snapshot:
        """Seal the evaluation with the agent's final output, closing it first if needed.

        Completing a sealed evaluation again with the same output returns its
        snapshot; another output is refused and recorded as a late fact.
        """
        if self.evaluation(ById(run_id, eval_id)).stage is not Stage.SEALED:
            await self.close(run_id, eval_id)
        return self.apply(ById(run_id, eval_id), lifecycle.Seal(agent_output, _now()))

    # --- grading ---

    def grade(self, run_id: str, eval_id: str) -> GradeResponse:
        """Grade the sealed snapshot, once. The grade is computed with no lock held, and never replaced.

        The response's evidence status also marks every key a late fact has
        marked, including facts that arrived after the grade.
        """
        target = ById(run_id, eval_id)
        evaluation = self.evaluation(target)
        if evaluation.grade is None:
            if evaluation.snapshot is None:
                raise _reject(Rejection.NOT_SEALED, "Evaluation not completed. Call complete first.")
            graded = grade_snapshot(self.suite(run_id), evaluation.snapshot)
            evaluation = self.apply(target, lifecycle.RecordGrade(graded))
        assert evaluation.grade is not None
        return evaluation.grade.model_copy(update={"evidence_status": evaluation.evidence_status})
