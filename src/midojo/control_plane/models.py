from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from midojo.observations import EvidenceStatus
from midojo.types import Environment, GradeValidity, SuiteName

# --- Run / Evaluation request/response models ---


class CreateFunctionCallRecord(BaseModel):
    """Request body for recording a function call (server fills in timestamp + env snapshots)."""

    function: str
    args: dict
    result: str
    error: str | None = None
    blocked: bool = Field(
        default=False,
        description="The tool never ran: the agent's call was blocked, and `result` is what the agent was told instead.",
    )


class FunctionCallResponse(CreateFunctionCallRecord):
    """Public shape of a recorded function call.

    Excludes the internal pre/post environment snapshots that the domain
    ``FunctionCallRecord`` carries for grading — no API client consumes them.
    ``from_attributes`` lets it be built directly from a domain record.
    """

    model_config = ConfigDict(from_attributes=True)

    timestamp: str


class CreateEvaluationRequest(BaseModel):
    user_task_id: str
    injection_task_id: str | None = None
    injections: dict[str, str] = {}


class CreateRunRequest(BaseModel):
    suite_name: SuiteName


class CreateRunResponse(BaseModel):
    id: str
    suite_name: SuiteName


class CreateEvaluationResponse(BaseModel):
    id: str
    prompt: str
    session_token: str
    session_expires_at: str


class CompleteRequest(BaseModel):
    agent_output: str


class RecordObservationsRequest(BaseModel):
    """Record an evaluation's runtime observations from one source.

    e.g. ``{"source": "openshell", "data": {"network_calls_blocked": ["audit.ext-log.com:443"], ...}}``.
    ``data`` must match the model registered for ``source`` in :mod:`midojo.observations`.
    Verifiers read that model from ``VerificationContext.observations[source]`` at grade time.
    """

    source: str
    data: dict[str, Any]


_EVIDENCE_STATUS_DESCRIPTION = (
    "How completely each evidence source was collected, keyed by `<source>.<part>` "
    "(e.g. `openshell.ocsf`): complete, incomplete, failed or not_collected."
)


class GradeResponse(BaseModel):
    """An evaluation's grades.

    ``utility`` and ``security`` are None when the check couldn't be decided
    from the evidence collected; ``grade_validity`` then marks that grade
    invalid. Without an injection task, ``security`` is False and its validity
    ``not_applicable``.
    """

    utility: bool | None
    utility_reason: str | None = Field(default=None, description="The criterion that decided the utility grade.")
    security: bool | None
    security_reason: str | None = Field(
        default=None,
        description=(
            "The criterion that decided the security grade, whatever the outcome "
            "(e.g. the AnyOf branch that fired, or the unknown checks); None without an injection task."
        ),
    )
    grade_validity: GradeValidity
    evidence_status: dict[str, EvidenceStatus] = Field(default_factory=dict, description=_EVIDENCE_STATUS_DESCRIPTION)


class EvaluationSummary(BaseModel):
    id: str
    user_task_id: str
    injection_task_id: str | None
    completed: bool
    graded: bool = Field(description="Whether the evaluation was graded. A graded None is unknown, not ungraded.")
    utility: bool | None
    security: bool | None
    grade_validity: GradeValidity | None = None


class RunResponse(BaseModel):
    id: str
    suite_name: SuiteName
    created_at: str
    evaluations: list[EvaluationSummary]


class EvaluationResponse(BaseModel):
    id: str
    user_task_id: str
    injection_task_id: str | None
    completed: bool
    graded: bool = Field(description="Whether the evaluation was graded. A graded None is unknown, not ungraded.")
    utility: bool | None
    utility_reason: str | None = None
    security: bool | None
    security_reason: str | None = None
    grade_validity: GradeValidity | None = None
    evidence_status: dict[str, EvidenceStatus] = Field(default_factory=dict, description=_EVIDENCE_STATUS_DESCRIPTION)
    agent_input: str | None
    agent_output: str | None
    function_calls: list[FunctionCallResponse]


# --- Suite / task response models ---


class SuiteInfoResponse(BaseModel):
    name: SuiteName
    user_tasks: list[str]
    injection_tasks: list[str]
    environment: SerializeAsAny[Environment]


class TaskDetailResponse(BaseModel):
    id: str
    type: str
    prompt: str | None = None
    description: str | None = None
