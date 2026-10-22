"""Run and evaluation records. Application handles live on FastAPI app.state."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from midojo.observations import EvidenceStatus
from midojo.types import Environment, FunctionCallRecord, GradeValidity, SuiteName

# --- State models ---


@dataclass
class Evaluation:
    """A single task execution within a Run. Captures the environment before and after tool execution, the function call trace, and grading results."""

    id: str
    run_id: str
    user_task_id: str
    injection_task_id: str | None
    pre_environment: Environment
    environment: Environment
    function_calls: list[FunctionCallRecord] = field(default_factory=list)
    # Runtime observations keyed by source (e.g. "openshell" -> OpenShellObservations),
    # recorded by the orchestrator and read by verifiers at grade time.
    observations: dict[str, BaseModel] = field(default_factory=dict)
    agent_input: str | None = None
    agent_output: str | None = None
    completed: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    active_injections: dict[str, str] = field(default_factory=dict)
    # Grades: None before grading. After grading, None means the check couldn't be
    # decided, and grade_validity (None until graded) marks that grade invalid.
    utility: bool | None = None
    utility_reason: str | None = None
    security: bool | None = None
    security_reason: str | None = None
    grade_validity: GradeValidity | None = None
    evidence_status: dict[str, EvidenceStatus] = field(default_factory=dict)

    @property
    def graded(self) -> bool:
        return self.grade_validity is not None


class Run(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    suite_name: SuiteName
    evaluations: dict[str, Evaluation] = {}
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
