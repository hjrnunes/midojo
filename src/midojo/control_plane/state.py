"""Run and evaluation values. Application handles live on FastAPI app.state.

An :class:`Evaluation` is a value: a frozen dataclass whose sequences are
tuples. The lifecycle rules (:mod:`midojo.control_plane.lifecycle`) never change
one; they build the next value, and the writer
(:mod:`midojo.control_plane.service`) replaces the stored value with it. The
pydantic models inside (environments, call records, observations) are mutable
types, so the writer copies them in and out: no caller ever holds an object the
store keeps.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from midojo.episode import Episode
from midojo.observations import FUNCTION_CALLS, EvidenceStatus
from midojo.types import Environment, FunctionCallRecord, InterceptionMode, SuiteName

from .models import CapturePhase, GradeResponse


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Stage(enum.StrEnum):
    """Where an evaluation is in its lifecycle.

    ``open``: everything is admitted. ``closing``: no new call is admitted, and
    calls already admitted may still finish, until they all have or the closure
    wait runs out. ``collecting``: the final observations are recorded; no call
    finishes any more. ``sealed``: the snapshot is taken, and nothing changes it.
    """

    OPEN = "open"
    CLOSING = "closing"
    COLLECTING = "collecting"
    SEALED = "sealed"


@dataclass(frozen=True)
class Turn:
    """One user message in an evaluation's conversation and the agent's reply to it."""

    index: int
    message: str
    reply: str | None = None
    started_at: str = field(default_factory=_now)
    ended_at: str | None = None


@dataclass(frozen=True)
class Session:
    """The evaluation's live session: its token's hash and when it expires (epoch seconds)."""

    token_hash: str
    expires_at: float


@dataclass(frozen=True)
class LateFact:
    """A write that closure refused, kept so the evaluation shows its evidence is incomplete.

    ``evidence`` names the evidence-status keys it marks ``incomplete``
    (``function_calls``, ``environment``, ``turns``, ``agent_output``, or an
    observation source's ``<source>.<part>``). ``payload`` is the write as sent,
    as JSON-compatible data.
    """

    event: str
    stage: Stage
    by: str
    at: str
    evidence: tuple[str, ...]
    payload: Mapping[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class CaptureFailure:
    """A call record an interception layer couldn't deliver: the recorded calls may lack that call.

    ``phase`` is the record that was lost: ``start`` (the call may be missing
    altogether) or ``finish`` (its record may be unfinished or missing).
    ``reason`` is the last error the layer got. A capture failure marks
    ``function_calls`` ``incomplete``; it never adds, changes or removes a
    record.
    """

    call_id: str
    function: str
    phase: CapturePhase
    reason: str
    stage: Stage
    by: str
    at: str


@dataclass(frozen=True)
class Snapshot:
    """What the seal fixes: everything grading reads, and how completely it was collected.

    ``evidence_status`` is the observations' own status, with every key a late
    fact recorded before the seal marked ``incomplete``, and ``function_calls``
    marked ``incomplete`` when a capture failure was reported before the seal.

    ``evaluation_id`` and ``run_id`` name the evaluation it seals (``run_id``
    None outside a run), and ``snapshot_id`` this snapshot: a result graded
    from it names what it graded, never "the latest evaluation".
    """

    evaluation_id: str
    run_id: str | None
    snapshot_id: str
    user_task_id: str | None
    injection_task_id: str | None
    agent_output: str
    pre_environment: Environment
    environment: Environment
    function_calls: tuple[FunctionCallRecord, ...]
    turns: tuple[Turn, ...]
    observations: Mapping[str, BaseModel]
    evidence_status: Mapping[str, EvidenceStatus]
    sealed_at: str
    capture_failures: tuple[CaptureFailure, ...] = ()


@dataclass(frozen=True)
class Evaluation:
    """One run of an episode: its records while it's open, then its sealed snapshot and grade.

    ``run_id``, ``suite_name`` and the task IDs are provenance: set when the
    episode came from a suite pair in a run, None for a direct caller's episode.
    ``episode`` is what was run; grading reads its checks.
    """

    id: str
    run_id: str | None
    user_task_id: str | None
    injection_task_id: str | None
    pre_environment: Environment
    environment: Environment
    function_calls: tuple[FunctionCallRecord, ...] = ()
    # Turns in the order they were started; function calls name the latest one.
    turns: tuple[Turn, ...] = ()
    # Runtime observations keyed by source (e.g. "openshell" -> OpenShellObservations),
    # recorded by the orchestrator and read by verifiers at grade time.
    observations: Mapping[str, BaseModel] = field(default_factory=dict)
    agent_input: str | None = None
    created_at: str = field(default_factory=_now)
    active_injections: Mapping[str, str] = field(default_factory=dict)
    session: Session | None = None
    stage: Stage = Stage.OPEN
    # When the closure wait ends (epoch seconds), once closure has begun.
    closure_deadline: float | None = None
    snapshot: Snapshot | None = None
    late_facts: tuple[LateFact, ...] = ()
    # Call records the interception layer reported it couldn't deliver, in arrival order.
    capture_failures: tuple[CaptureFailure, ...] = ()
    # Set once, from the snapshot, and never replaced.
    grade: GradeResponse | None = None
    # Read-only: copying the evaluation shares it (see Episode.__deepcopy__).
    episode: Episode | None = None
    suite_name: SuiteName | None = None
    # How many times the environment has changed since the evaluation was created.
    environment_revision: int = 0
    # Whether call records carry environment snapshots: the environment observed when each call started and
    # finished, which shows what changed during the call, not what the call caused.
    environment_history: bool = False
    # Where the tool calls are intercepted, chosen by the host at creation: it sets the provenance of the
    # records the session makes, and whether a capture without records is complete (see lifecycle._seal).
    interception: InterceptionMode = "agent"

    @property
    def completed(self) -> bool:
        return self.stage is Stage.SEALED

    @property
    def graded(self) -> bool:
        return self.grade is not None

    @property
    def environment_type(self) -> type[Environment]:
        """The model a replacement environment is validated against: the episode's."""
        if self.episode is not None and self.episode.environment_type is not None:
            return self.episode.environment_type
        return type(self.environment)

    @property
    def agent_output(self) -> str | None:
        return self.snapshot.agent_output if self.snapshot is not None else None

    @property
    def evidence_status(self) -> dict[str, EvidenceStatus]:
        """The grade's evidence status, with every key a late fact or a capture failure marks ``incomplete``.

        Empty before grading unless late facts or capture failures were
        recorded. The grade itself never changes: a fact that arrives after the
        seal shows up here only.
        """
        status: dict[str, EvidenceStatus] = dict(self.grade.evidence_status) if self.grade is not None else {}
        status.update(late_evidence(self.late_facts))
        status.update(capture_evidence(self.capture_failures))
        return status


def late_evidence(facts: tuple[LateFact, ...]) -> dict[str, EvidenceStatus]:
    """The evidence-status keys ``facts`` mark incomplete."""
    return {key: "incomplete" for fact in facts for key in fact.evidence}


def capture_evidence(failures: tuple[CaptureFailure, ...]) -> dict[str, EvidenceStatus]:
    """The evidence-status keys ``failures`` mark incomplete: ``function_calls``, once any was reported."""
    return {FUNCTION_CALLS: "incomplete"} if failures else {}


@dataclass(frozen=True)
class Run:
    id: str
    suite_name: SuiteName
    created_at: str = field(default_factory=_now)
