"""The control plane's operations, applied by a single writer per evaluation.

:class:`ControlPlane` is what the HTTP routes call, and what an in-process
caller (a host-side recorder, an orchestrator in the same process) calls in
their place. Every event on an evaluation goes through :meth:`ControlPlane.apply`,
which runs the pure transition (:mod:`midojo.control_plane.lifecycle`) as one
step of the store's writer capability (:meth:`Store.writer`, which only the
control plane holds): under that evaluation's lock, from look-up to commit. So
no two events on one evaluation interleave, whatever thread or event loop sends
them, and events on different evaluations don't wait for each other.

``apply`` is synchronous and only decides: the lock is never held across an
``await``, a tool call or grading. The closure wait (:meth:`ControlPlane.close`)
sleeps between short ``apply`` calls, and grading reads the sealed snapshot,
grades it with no lock held, then records the grade in one more ``apply``.

The stored evaluation never aliases a caller's objects, in either direction:
the transition decides on copies of the event and the evaluation, the writer
keeps a copy of the next value, and every read returns a copy.

Outcomes are domain values: a refusal raises :class:`RejectedError` with its
:class:`Rejection` kind, and only the HTTP edge (``dependencies.py``) turns a
kind into a status code.

An evaluation is one run of an :class:`~midojo.episode.Episode`.
:meth:`ControlPlane.open_evaluation` creates one from any episode, with no
suite or run registered (``run_id`` None); :meth:`ControlPlane.create_evaluation`
compiles a registered suite's task pair into an episode and opens it in a run.
Grading reads the episode's checks.
"""

from __future__ import annotations

import asyncio
import hashlib
import secrets
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from midojo.episode import Episode, UnknownTaskError
from midojo.observations import require_evidence_keys
from midojo.types import Environment, FunctionCallRecord, InterceptionMode, SuiteName
from midojo.yaml_task_suite import YAMLTaskSuite

from . import lifecycle
from .config import AppConfig
from .grading import grade_snapshot
from .lifecycle import HOST, Accepted, AgentSession, Event, Late, Rejected, RejectedError, Rejection
from .models import (
    CapturePhase,
    CreateEvaluationResponse,
    CreateFunctionCallRecord,
    CreateFunctionCallStart,
    GradeResponse,
)
from .state import CaptureFailure, Creation, Evaluation, Run, Snapshot, Stage, Turn, unsupported_value
from .store import InMemoryStore, Store, detached

# How often the closure wait checks for unfinished calls, in seconds.
CLOSURE_POLL_SECONDS = 0.05


@dataclass(frozen=True)
class ById:
    """An evaluation addressed by its IDs, by the host (authorized at the edge). ``run_id`` is None outside a run."""

    run_id: str | None
    eval_id: str


@dataclass(frozen=True)
class BySession:
    """The evaluation an interception layer's session token is bound to."""

    token: str


Target = ById | BySession


@dataclass(frozen=True)
class OpenedEvaluation:
    """A new evaluation and its open session. This is the only place the session token appears."""

    id: str
    run_id: str | None
    # The episode's turns, or None when the host drives the conversation.
    turns: tuple[str, ...] | None
    session_token: str
    session_expires_at: str


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _reject(kind: Rejection, detail: str) -> RejectedError:
    return RejectedError(Rejected(kind, detail))


CONTROL_TOKEN_REQUIRED_DETAIL = (
    "Host interception needs a control plane served with a control token (MIDOJO_CONTROL_TOKEN): without one, "
    "its run-scoped routes are open to anything that reaches it, the agent included."
)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class ControlPlane:
    def __init__(
        self,
        suites: Mapping[SuiteName, YAMLTaskSuite],
        *,
        store: Store | None = None,
        config: AppConfig | None = None,
        served: bool = False,
    ) -> None:
        """A control plane over ``store``, for ``suites``.

        ``served`` says an HTTP app serves it (``create_app`` sets it). Served
        without a control token, its run-scoped routes take any request, so it
        opens no host-mode evaluation (``control_token_required``), whatever
        path asks, and refuses (ValueError) a store that already holds one. A
        control plane no app serves has no such routes and opens host mode
        without a token.

        Its wiring is fixed once it is built: ``suites`` (a read-only copy of
        the mapping given), ``store``, ``config`` (a frozen ``AppConfig``) and
        ``served`` are read-only, so no caller can, say, take the control token
        away from a served control plane that already holds host-mode
        evaluations.
        """
        self._suites: Mapping[SuiteName, YAMLTaskSuite] = MappingProxyType(dict(suites))
        self._store: Store = store if store is not None else InMemoryStore()
        self._config = config if config is not None else AppConfig()
        self._served = served
        if self._refuses_host_mode():
            runs = [run.id for run in self.store.list_runs()]
            held = [
                e.id
                for run_id in [None, *runs]
                for e in self.store.list_evaluations(run_id)
                if e.interception == "host"
            ]
            if held:
                raise ValueError(
                    f"A control plane served without a control token can't hold host-mode evaluations: {held}"
                )
        self._writer = self.store.writer()
        # Held while a create under an idempotency key looks up, opens and keeps its evaluation, or rotates its
        # session's token.
        self._creating = threading.Lock()

    @property
    def suites(self) -> Mapping[SuiteName, YAMLTaskSuite]:
        """The suites this control plane compiles episodes from, read-only."""
        return self._suites

    @property
    def store(self) -> Store:
        return self._store

    @property
    def config(self) -> AppConfig:
        return self._config

    @property
    def served(self) -> bool:
        """Whether an HTTP app serves this control plane (see ``create_app``)."""
        return self._served

    def _refuses_host_mode(self) -> bool:
        """Whether host mode is refused: served over HTTP with no control token to guard the run-scoped routes."""
        return self.served and self.config.control_token is None

    def _check_interception(self, interception: InterceptionMode) -> None:
        if interception == "host" and self._refuses_host_mode():
            raise _reject(Rejection.CONTROL_TOKEN_REQUIRED, CONTROL_TOKEN_REQUIRED_DETAIL)

    # --- the single writer ---

    def _resolve(self, target: Target) -> tuple[str | None, str, lifecycle.Authority]:
        if isinstance(target, BySession):
            hashed = token_hash(target.token)
            bound = self.store.session_target(hashed)
            if bound is None:
                raise _reject(Rejection.INVALID_SESSION, lifecycle.INVALID_SESSION_DETAIL)
            return bound[0], bound[1], AgentSession(hashed, time.time())
        if target.run_id is not None and self.store.get_run(target.run_id) is None:
            raise _reject(Rejection.UNKNOWN_RUN, f"Unknown run: {target.run_id}")
        return target.run_id, target.eval_id, HOST

    def apply(self, target: Target, event: Event) -> Any:
        """Decide ``event`` on the target's evaluation and commit it, as one step; return its result.

        Raises :class:`RejectedError` when the event is refused, with ``late``
        True when it was recorded as a late fact.
        """
        run_id, eval_id, authority = self._resolve(target)
        decided: list[lifecycle.Outcome] = []

        def step(evaluation: Evaluation | None) -> Evaluation | None:
            if evaluation is None:
                raise _reject(Rejection.UNKNOWN_EVALUATION, f"Unknown evaluation: {eval_id}")
            outcome = lifecycle.transition(evaluation, authority, event)
            decided.append(outcome)
            if isinstance(outcome, Rejected):
                return None
            if isinstance(outcome, Accepted):
                self._apply_effects(outcome, run_id, eval_id)
            return outcome.evaluation

        self._writer.apply(run_id, eval_id, step)
        [outcome] = decided
        if isinstance(outcome, Rejected):
            raise RejectedError(outcome)
        if isinstance(outcome, Late):
            raise RejectedError(outcome.rejected, late=True)
        # The writer kept its own copy of the evaluation, so the result shares nothing with what's stored.
        return outcome.result

    def _apply_effects(self, outcome: Accepted, run_id: str | None, eval_id: str) -> None:
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
        interception: InterceptionMode = "agent",
        idempotency_key: str | None = None,
    ) -> CreateEvaluationResponse:
        """Compile the run's suite pair into an episode and open it.

        The response is the only place the session token appears. With an
        ``idempotency_key`` (the caller's choice, one per request), the run
        also keeps the request and the evaluation it opened under the key, but
        never the token: sending the same request with the same key again
        opens nothing and returns the same evaluation with a freshly issued
        token for its session (same expiry), and the token issued before
        stops resolving, so a create whose response was lost can be resent
        and only its last response's token works. It rotates at any stage,
        changing nothing else: once the evaluation is closing or sealed, the
        new token is held to that stage's rules (a write is a late fact). A
        resend is refused as ``session_closed`` once the session was revoked
        or has expired, which leaves it as it was. The same key with another
        request is refused as ``idempotency_conflict``. A create refused
        outright (an unknown task, say) keeps nothing, so resending it tries
        it afresh. On a control plane served without a control token, a
        host-mode create is refused as ``control_token_required`` before its
        key is read, and keeps nothing.
        """
        self._check_interception(interception)
        # Read once, into a value of its own: the key's request and the compiled episode are the same injections,
        # whatever the caller does with its mapping meanwhile.
        injected: dict[str, str] = detached(dict(injections or {}))
        if idempotency_key is None:
            return self._create_evaluation(run_id, user_task_id, injection_task_id, injected, interception)
        request = {
            "user_task_id": user_task_id,
            "injection_task_id": injection_task_id,
            "injections": injected,
            "interception": interception,
        }
        with self._creating:
            kept = self.store.kept_creation(run_id, idempotency_key)
            if kept is not None:
                if kept.request != request:
                    raise _reject(
                        Rejection.IDEMPOTENCY_CONFLICT,
                        f"Idempotency key {idempotency_key!r} already created an evaluation for another request",
                    )
                # Under the same lock as the look-up: two resends rotate one after the other, the last one's token
                # the one that resolves.
                return self._reissue(run_id, kept)
            response = self._create_evaluation(run_id, user_task_id, injection_task_id, injected, interception)
            self.store.keep_creation(run_id, idempotency_key, Creation(request, response.id, tuple(response.turns)))
            return response

    def _reissue(self, run_id: str, kept: Creation) -> CreateEvaluationResponse:
        """The kept creation's response again, with a fresh session token: the one issued before stops resolving."""
        token = secrets.token_urlsafe(32)
        rotate = lifecycle.RotateSession(token_hash(token), time.time())
        session = self.apply(ById(run_id, kept.evaluation_id), rotate)
        return CreateEvaluationResponse(
            id=kept.evaluation_id,
            prompt=kept.turns[0],
            turns=list(kept.turns),
            session_token=token,
            session_expires_at=datetime.fromtimestamp(session.expires_at, UTC).isoformat(),
        )

    def _create_evaluation(
        self,
        run_id: str,
        user_task_id: str,
        injection_task_id: str | None,
        injections: dict[str, str],
        interception: InterceptionMode,
    ) -> CreateEvaluationResponse:
        """Compile and open the evaluation; ``injections`` is already the caller's, read once (see above)."""
        try:
            episode = self.suite(run_id).episode(user_task_id, injection_task_id, injections)
        except UnknownTaskError as exc:
            raise _reject(Rejection.UNKNOWN_TASK, exc.detail) from None
        opened = self.open_evaluation(episode, run_id=run_id, interception=interception)
        assert opened.turns is not None  # A suite's episode always has its user task's turns.
        return CreateEvaluationResponse(
            id=opened.id,
            prompt=opened.turns[0],
            turns=list(opened.turns),
            session_token=opened.session_token,
            session_expires_at=opened.session_expires_at,
        )

    def open_evaluation(
        self, episode: Episode, *, run_id: str | None = None, interception: InterceptionMode = "agent"
    ) -> OpenedEvaluation:
        """Create an evaluation of ``episode`` and open its session.

        Needs no registered suite or run: without ``run_id``, the evaluation is
        outside any run, and addressed as ``ById(None, eval_id)`` (or by its
        session). With one, the run must exist, and the episode must name its
        ``user_task_id`` (the run's routes report every evaluation by it), or
        this raises ValueError. The episode's suite and task IDs are kept as
        provenance; grading uses its checks.

        ``interception`` is where its tool calls are intercepted: ``agent`` (by
        an interception layer given the session) or ``host`` (by an interceptor
        the host runs outside the target, the only holder of the session). It
        labels each record the session makes (``FunctionCallRecord.provenance``).
        A control plane served without a control token refuses ``host`` as
        ``control_token_required``. An environment holding a value a snapshot
        can't hold (``state.unsupported_value``) raises ValueError too.
        """
        self._check_interception(interception)
        run = self.run(run_id) if run_id is not None else None
        if run is not None and episode.user_task_id is None:
            raise ValueError("An episode opened in a run needs a user_task_id")
        environment = episode.provision_environment()
        if (problem := unsupported_value(environment.model_dump())) is not None:
            raise ValueError(f"The episode's environment can't be kept: {problem}")
        turns = episode.turns
        evaluation = Evaluation(
            id=self.store.new_evaluation_id(),
            run_id=run_id,
            user_task_id=episode.user_task_id,
            injection_task_id=episode.injection_task_id,
            pre_environment=environment.model_copy(deep=True),
            environment=environment,
            active_injections=dict(episode.injections),
            agent_input=turns[0] if turns else None,
            episode=episode,
            suite_name=episode.suite_name if episode.suite_name is not None else (run.suite_name if run else None),
            environment_history=self.config.environment_history,
            interception=interception,
        )
        self.store.insert_evaluation(evaluation)
        token = secrets.token_urlsafe(32)
        expires = time.time() + self.config.session_ttl_seconds
        self.apply(ById(run_id, evaluation.id), lifecycle.OpenSession(token_hash(token), expires))
        return OpenedEvaluation(
            id=evaluation.id,
            run_id=run_id,
            turns=None if turns is None else tuple(turns),
            session_token=token,
            session_expires_at=datetime.fromtimestamp(expires, UTC).isoformat(),
        )

    def close_session(self, run_id: str | None, eval_id: str) -> None:
        self.apply(ById(run_id, eval_id), lifecycle.CloseSession())

    # --- turns, calls, environment and observations ---

    def start_turn(self, run_id: str | None, eval_id: str, message: str) -> Turn:
        return self.apply(ById(run_id, eval_id), lifecycle.StartTurn(message, _now()))

    def record_turn_reply(self, run_id: str | None, eval_id: str, index: int, reply: str) -> Turn:
        return self.apply(ById(run_id, eval_id), lifecycle.ReplyTurn(index, reply, _now()))

    def start_call(self, target: Target, req: CreateFunctionCallStart) -> tuple[FunctionCallRecord, bool]:
        """Start a call: ``(record, created)``, with created False when ``req`` replays its ``call_id``."""
        return self.apply(target, lifecycle.StartCall(req, _now()))

    def finish_call(self, target: Target, req: CreateFunctionCallRecord) -> tuple[FunctionCallRecord, bool]:
        """Record a call, or finish a started one: ``(record, created)``, created False on a replay."""
        return self.apply(target, lifecycle.FinishCall(req, _now()))

    def report_capture_failure(
        self, target: Target, *, call_id: str, function: str, phase: CapturePhase, reason: str = ""
    ) -> tuple[CaptureFailure, bool]:
        """Report a call record the interception layer couldn't deliver: ``(failure, created)``.

        Before the seal, the report marks ``function_calls`` incomplete in the
        snapshot, so a check that depends on a call's absence grades unknown.
        ``created`` is False when the same ``call_id``, function and phase were
        kept before the seal, whatever the reason or sender, even when the
        repeat arrives after it. A new report after the seal is refused and
        recorded as a late fact: it marks only the evaluation's evidence
        status, never the grade. Repeating one is refused the same way, and
        adds no other late fact.
        """
        event = lifecycle.ReportCaptureFailure(call_id, function, phase, reason, _now())
        return self.apply(target, event)

    def replace_environment(self, target: Target, environment: Environment) -> Environment:
        """Install ``environment``, already validated against the evaluation's episode (``environment_type``)."""
        return self.apply(target, lifecycle.ReplaceEnvironment(environment, _now()))

    def change_environment(
        self, target: Target, values: Mapping[str, Any], expected_revision: int | None = None
    ) -> tuple[int, Environment]:
        """Set top-level fields of the environment in one step: ``(revision, environment)`` after the change.

        With ``expected_revision``, the change applies only at that revision,
        and is refused as a ``revision_conflict`` at any other. The changed
        environment is validated against the evaluation's environment type, and
        refused as ``invalid_environment`` when it doesn't fit.
        """
        return self.apply(target, lifecycle.ChangeEnvironment(dict(values), _now(), expected_revision))

    def record_observations(
        self, run_id: str | None, eval_id: str, source: str, observed: BaseModel
    ) -> dict[str, BaseModel]:
        """Record ``source``'s observations, already validated against the model registered for it."""
        return dict(self.apply(ById(run_id, eval_id), lifecycle.RecordObservations(source, observed, _now())))

    # --- staged closure ---

    async def close(self, run_id: str | None, eval_id: str) -> Evaluation:
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

    async def complete(
        self, run_id: str | None, eval_id: str, agent_output: str, incomplete: Sequence[str] = ()
    ) -> Snapshot:
        """Seal the evaluation with the agent's final output, closing it first if needed.

        Completing a sealed evaluation again with the same output returns its
        snapshot; another output is refused and recorded as a late fact.
        ``incomplete`` names evidence the host didn't collect in full, such as
        an episode a failure cut short: the snapshot marks it ``incomplete``.
        Each key must be a known evidence key (``observations.is_evidence_key``),
        or this raises ValueError; ``agent_output`` must be exactly a ``str``,
        or it's refused as ``unsupported_value``.
        """
        # Before closing: a refused seal leaves the evaluation as it was.
        require_evidence_keys(incomplete)
        if (refused := lifecycle.not_text("agent's output", agent_output)) is not None:
            raise RejectedError(refused)
        if self.evaluation(ById(run_id, eval_id)).stage is not Stage.SEALED:
            await self.close(run_id, eval_id)
        return self.apply(ById(run_id, eval_id), lifecycle.Seal(agent_output, _now(), incomplete=tuple(incomplete)))

    def snapshot(self, run_id: str | None, eval_id: str) -> Snapshot:
        """The evaluation's sealed snapshot, refused as ``not_sealed`` before the seal. It shares nothing stored."""
        snapshot = self.evaluation(ById(run_id, eval_id)).snapshot
        if snapshot is None:
            raise _reject(Rejection.NOT_SEALED, "Evaluation not completed. Call complete first.")
        return snapshot

    # --- grading ---

    def grade(self, run_id: str | None, eval_id: str) -> GradeResponse:
        """Grade the sealed snapshot with the episode's checks, once: with no lock held, and never replaced.

        The response's evidence status also marks every key a late fact has
        marked, including facts that arrived after the grade.
        """
        target = ById(run_id, eval_id)
        evaluation = self.evaluation(target)
        if evaluation.grade is None:
            if evaluation.snapshot is None:
                raise _reject(Rejection.NOT_SEALED, "Evaluation not completed. Call complete first.")
            if evaluation.episode is None:
                raise RuntimeError(f"Evaluation {eval_id} has no episode to grade it with")
            graded = grade_snapshot(evaluation.episode, evaluation.snapshot)
            evaluation = self.apply(target, lifecycle.RecordGrade(graded))
        assert evaluation.grade is not None
        return evaluation.grade.model_copy(update={"evidence_status": evaluation.evidence_status})
