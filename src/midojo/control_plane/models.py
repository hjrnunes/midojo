from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny

from midojo.observations import EvidenceStatus
from midojo.types import Environment, GradeValidity, SuiteName

# --- Run / Evaluation request/response models ---


CALL_ID_MAX_LENGTH = 256


class FunctionCallStart(BaseModel):
    """The fields that identify a call when it starts. Its finish must repeat them exactly."""

    function: str
    args: dict


class FunctionCallBody(FunctionCallStart):
    """The fields a client records for a function call. A replayed ``call_id`` must repeat them exactly."""

    result: str
    error: str | None = None
    blocked: bool = Field(
        default=False,
        description="The tool never ran: the agent's call was blocked, and `result` is what the agent was told instead.",
    )


class CreateFunctionCallRecord(FunctionCallBody):
    """Request body for recording a function call (server fills in timestamp + env snapshots).

    Also finishes a call started with ``CreateFunctionCallStart`` under the same ``call_id``.
    """

    call_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=CALL_ID_MAX_LENGTH,
        description=(
            "Client-chosen ID for this record, unique within the evaluation. When the call_id was started and "
            "not finished, this finishes that record in place; its function and args must match the start, or "
            "it's a 409. Resending a finished record with the same call_id and body returns the stored record "
            "instead of appending a duplicate; the same call_id with a different body is a 409. When omitted, "
            "the server assigns a UUID and the record isn't deduplicated."
        ),
    )


class CreateFunctionCallStart(FunctionCallStart):
    """Request body for starting a function call: an unfinished record, finished later by its ``call_id``."""

    call_id: str = Field(
        min_length=1,
        max_length=CALL_ID_MAX_LENGTH,
        description=(
            "Client-chosen ID for the call, unique within the evaluation. The finish sends the same call_id. "
            "Resending a start with the same call_id, function and args returns the stored record, finished or "
            "not; the same call_id with another function or args is a 409."
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
    result: str | None = Field(default=None, description="What the agent saw. None while the call is unfinished.")
    complete: bool = Field(
        default=True, description="False while the call is started but not finished: it has no result yet."
    )
    timestamp: str
    turn: int | None = Field(
        default=None, description="Index of the turn most recently started when the call was recorded, if any."
    )


class StartTurnRequest(BaseModel):
    message: str


class TurnReplyRequest(BaseModel):
    reply: str


class TurnResponse(BaseModel):
    """One user message in an evaluation's conversation and the agent's reply, once recorded."""

    model_config = ConfigDict(from_attributes=True)

    index: int
    message: str
    reply: str | None
    started_at: str
    ended_at: str | None


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
    prompt: str = Field(description="The first user message, the same as turns[0].")
    turns: list[str] = Field(description="The user's messages, one per turn, with probes substituted.")
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
    turns: list[TurnResponse]
    function_calls: list[FunctionCallResponse]
    observations: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Runtime observations keyed by source, as recorded (see the evaluation's /observations).",
    )
    pre_environment: dict
    environment: dict


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
    turns: list[str] | None = None
    description: str | None = None
