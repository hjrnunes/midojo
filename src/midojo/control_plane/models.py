from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from midojo.types import Environment, SuiteName

# --- Run / Evaluation request/response models ---


CALL_ID_MAX_LENGTH = 256


class FunctionCallBody(BaseModel):
    """The fields a client records for a function call. A replayed ``call_id`` must repeat them exactly."""

    function: str
    args: dict
    result: str
    error: str | None = None
    blocked: bool = Field(
        default=False,
        description="The tool never ran: the agent's call was blocked, and `result` is what the agent was told instead.",
    )


class CreateFunctionCallRecord(FunctionCallBody):
    """Request body for recording a function call (server fills in timestamp + env snapshots)."""

    call_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=CALL_ID_MAX_LENGTH,
        description=(
            "Client-chosen ID for this record, unique within the evaluation. Resending a record with the same "
            "call_id and body returns the stored record instead of appending a duplicate; the same call_id with "
            "a different body is a 409. When omitted, the server assigns a UUID and the record isn't deduplicated."
        ),
    )


class FunctionCallResponse(FunctionCallBody):
    """Public shape of a recorded function call.

    Excludes the internal pre/post environment snapshots that the domain
    ``FunctionCallRecord`` carries for grading — no API client consumes them.
    ``from_attributes`` lets it be built directly from a domain record.
    """

    model_config = ConfigDict(from_attributes=True)

    call_id: str = Field(description="The client's call_id, or a UUID the server assigned when the client sent none.")
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


class GradeResponse(BaseModel):
    utility: bool
    security: bool
    security_reason: str | None = Field(
        default=None,
        description=(
            "When the attack succeeded, the criterion that graded it "
            "(e.g. the AnyOf branch that fired); None when the attack failed."
        ),
    )


class EvaluationSummary(BaseModel):
    id: str
    user_task_id: str
    injection_task_id: str | None
    completed: bool
    utility: bool | None
    security: bool | None


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
    utility: bool | None
    security: bool | None
    security_reason: str | None = None
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
