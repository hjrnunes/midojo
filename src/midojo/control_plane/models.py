from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SerializeAsAny, field_validator

from midojo.observations import EvidenceStatus, require_evidence_keys
from midojo.types import (
    CallDisposition,
    CallInjectionOutcome,
    CallProvenance,
    Environment,
    GradeValidity,
    InterceptionMode,
    SuiteName,
)

# --- Run / Evaluation request/response models ---


CALL_ID_MAX_LENGTH = 256

_DISPOSITION_DESCRIPTION = (
    "What the interception layer did with the call: forwarded, simulated, denied or blocked. Optional: None "
    "says nothing. A denied or blocked call has `blocked` true, a forwarded or simulated one false; any other "
    "pairing is a 422 `invalid_disposition`. A finish that leaves it out keeps its start's; one that names "
    "another than its start's is a 409 `call_conflict`."
)


_PROVENANCE_DESCRIPTION = (
    "How the record reached the control plane, set by the control plane from the authority it arrived with, never "
    "by the caller: host (a run-scoped route, or the host in process), interceptor (the session of an evaluation "
    "intercepted on the host, which the target never holds) or agent (the session of an evaluation intercepted in "
    "the agent: lower trust, since the target may hold the session). None in older records."
)

_INTERCEPTION_DESCRIPTION = (
    "Where the evaluation's tool calls are intercepted: agent (an interception layer in the agent records them with "
    "the evaluation's session, which the target may hold) or host (an interceptor the host runs outside the target "
    "records them; the target is given only its tool endpoint). It sets the provenance of the records the session "
    "makes. With agent, an evaluation sealed without any call record has function_calls evidence incomplete: a "
    "hook that failed open records nothing, so no record is not evidence of no call."
)


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
    disposition: CallDisposition | None = Field(default=None, description=_DISPOSITION_DESCRIPTION)
    structured_content: dict[str, Any] | None = Field(
        default=None,
        description=(
            "The structured content the agent got with `result` (its text), exactly as it got it. Optional: None "
            "when the result had none or the layer reports the text only."
        ),
    )
    injection_outcomes: list[CallInjectionOutcome] | None = Field(
        default=None,
        description=(
            "How each injection instruction that matched the call fared: those the interception layer admitted for "
            "it (applied, partial: the payload is in `structured_content` only, as the text isn't a JSON mirror of "
            "it; fallback, unmatched, too_large or no_result) and those whose occurrence rule skipped it "
            "(skipped_occurrence). Empty when none matched. Optional: left out (None), the record carries the "
            "result's text only and says nothing about its structured content or injections."
        ),
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
            "not finished, this finishes that record in place; its function and args must match the start, and "
            "its disposition, when both name one, or it's a 409. Resending a finished record with the same "
            "call_id and body returns the stored record instead of appending a duplicate (a disposition left out "
            "on either side matches any); the same call_id with a different body is a 409. When omitted, the "
            "server assigns a UUID and the record isn't deduplicated."
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
            "not (a disposition left out on either side matches any); the same call_id with another function, "
            "args or disposition is a 409."
        ),
    )
    disposition: CallDisposition | None = Field(
        default=None,
        description=(
            "What the interception layer does with the call: forwarded or simulated. Optional: None says nothing. "
            "A denied or blocked call never runs, so it's recorded whole, never started: starting one is a 422 "
            "`invalid_disposition`."
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
    disposition: CallDisposition | None = Field(
        default=None,
        description=(
            "What the interception layer did with the call: forwarded, simulated, denied or blocked. None when "
            "neither its start nor its finish said."
        ),
    )
    provenance: CallProvenance | None = Field(default=None, description=_PROVENANCE_DESCRIPTION)
    timestamp: str
    turn: int | None = Field(
        default=None, description="Index of the turn most recently started when the call was recorded, if any."
    )


class ChangeEnvironmentRequest(BaseModel):
    """Request body for setting top-level fields of an environment in one atomic step."""

    set: dict[str, Any] = Field(description="The fields to set, each to its new value. The other fields are kept.")
    expected_revision: int | None = Field(
        default=None,
        description=(
            "The revision the change was computed from: it applies only at that revision, and is refused with "
            "409 revision_conflict at any other. Omitted, the change applies at whatever revision is current."
        ),
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
    interception: InterceptionMode = Field(default="agent", description=_INTERCEPTION_DESCRIPTION)
    idempotency_key: str | None = Field(
        default=None,
        min_length=1,
        max_length=256,
        description=(
            "Optional, the caller's choice, one per request. The run keeps the evaluation it opens under it, never "
            "the session token: the same request sent again with the same key returns the same evaluation with a "
            "freshly issued session token, and the token issued before stops resolving (a create whose response was "
            "lost can be resent), at any stage; once the session was revoked or has expired, the resend is refused "
            "(401, session_closed). The same key with another request is refused (409, idempotency_conflict)."
        ),
    )


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
    incomplete: list[str] = Field(
        default_factory=list,
        description=(
            "Evidence-status keys the host didn't collect in full, such as an episode a failure cut short "
            "(function_calls, turns, agent_output): the snapshot marks them incomplete. Each must be a known "
            "evidence key: function_calls, environment, turns, agent_output, or an observation source's part."
        ),
    )

    @field_validator("incomplete")
    @classmethod
    def _known_evidence(cls, keys: list[str]) -> list[str]:
        require_evidence_keys(keys)
        return keys


_STAGE_DESCRIPTION = (
    "Where the evaluation is in its lifecycle: open (everything is admitted), closing (no new call is admitted; "
    "calls already admitted may finish during the closure wait), collecting (the final observations are "
    "recorded; no call finishes any more) or sealed (completed: the snapshot grading reads is fixed)."
)


class CloseResponse(BaseModel):
    stage: str = Field(description=_STAGE_DESCRIPTION)
    unfinished_calls: list[str] = Field(
        description="The call_ids still unfinished when the closure wait ended: they stay unfinished in the snapshot."
    )


CapturePhase = Literal["start", "finish"]
"""Which record of a call an interception layer couldn't deliver."""

CAPTURE_REASON_MAX_LENGTH = 2048


class ReportCaptureFailureRequest(BaseModel):
    """Request body for reporting a call record the interception layer couldn't deliver, after its resends."""

    call_id: str = Field(
        min_length=1,
        max_length=CALL_ID_MAX_LENGTH,
        description=(
            "The call_id of the record that was lost. Reporting the same call_id, function and phase again, "
            "with any reason, returns the stored report (a 200) instead of keeping another, when it was kept "
            "before the seal. One first received after the seal is only a late fact, and stays refused."
        ),
    )
    function: str
    phase: CapturePhase = Field(
        description="start: the call's start was lost, so the call may be missing. finish: its finish was lost."
    )
    reason: str = Field(default="", max_length=CAPTURE_REASON_MAX_LENGTH, description="The last error the layer got.")


class CaptureFailureResponse(BaseModel):
    """A call record the interception layer couldn't deliver. It marks function_calls incomplete."""

    model_config = ConfigDict(from_attributes=True)

    call_id: str
    function: str
    phase: CapturePhase
    reason: str
    stage: str = Field(description="The evaluation's stage when the report arrived.")
    by: str = Field(
        description=(
            "Who sent it: agent or interceptor (the evaluation's session, by interception mode) or host (a "
            "run-scoped route, or the host in process)."
        )
    )
    at: str


class LateFactResponse(BaseModel):
    """A write that closure refused. It never changes the evaluation's records; it marks their evidence incomplete."""

    model_config = ConfigDict(from_attributes=True)

    event: str = Field(
        description=(
            "What was refused: start_call, finish_call, report_capture_failure, replace_environment, "
            "change_environment, record_observations, start_turn, reply_turn or seal."
        )
    )
    stage: str = Field(description="The evaluation's stage when it arrived.")
    by: str = Field(
        description=(
            "Who sent it: agent or interceptor (the evaluation's session, by interception mode) or host (a "
            "run-scoped route, or the host in process)."
        )
    )
    at: str
    evidence: list[str] = Field(description="The evidence_status keys it marks incomplete.")
    payload: dict[str, Any] = Field(description="The write as sent.")
    call_id: str | None = None


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
    stage: str = Field(default="open", description=_STAGE_DESCRIPTION)
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
    interception: InterceptionMode = Field(default="agent", description=_INTERCEPTION_DESCRIPTION)
    completed: bool
    stage: str = Field(default="open", description=_STAGE_DESCRIPTION)
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
    late_facts: list[LateFactResponse] = Field(
        default_factory=list,
        description=(
            "Writes that closure refused, in arrival order. They never change the records above; each marks the "
            "evidence it carries incomplete in evidence_status."
        ),
    )
    capture_failures: list[CaptureFailureResponse] = Field(
        default_factory=list,
        description=(
            "Call records the interception layer reported it couldn't deliver, in arrival order. Each marks "
            "function_calls incomplete in evidence_status, so a check that depends on a call's absence is unknown."
        ),
    )


class SnapshotResponse(BaseModel):
    """A sealed evaluation's snapshot: everything grading reads, and how completely it was collected.

    ``function_calls`` are the full call records, environment history
    included. The snapshot never changes: a fact that arrives after the seal
    shows up only in the evaluation's ``evidence_status``.
    """

    evaluation_id: str
    run_id: str | None
    snapshot_id: str = Field(description="Names this snapshot: what a grade of it graded.")
    user_task_id: str | None
    injection_task_id: str | None
    agent_output: str
    pre_environment: dict[str, Any]
    environment: dict[str, Any]
    function_calls: list[dict[str, Any]]
    turns: list[TurnResponse]
    observations: dict[str, dict[str, Any]]
    evidence_status: dict[str, EvidenceStatus] = Field(description=_EVIDENCE_STATUS_DESCRIPTION)
    sealed_at: str
    capture_failures: list[CaptureFailureResponse]


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
