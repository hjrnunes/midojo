from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from midojo.types import FunctionCallRecord, SuiteName
from midojo.yaml_task_suite import YAMLTaskSuite

from ..config import AppConfig
from ..dependencies import (
    RECORD_FUNCTION_CALL_RESPONSES,
    START_FUNCTION_CALL_RESPONSES,
    control_token_required,
    created_or_replayed,
    exposed_without_control_token,
    get_config,
    get_control,
    get_evaluation_by_id,
    get_run,
    get_suites,
    refuse_unguarded_host_evaluation,
    resolve_suite,
    validate_environment,
    validate_observations,
)
from ..lifecycle import json_data
from ..models import (
    CaptureFailureResponse,
    CloseResponse,
    CompleteRequest,
    CreateEvaluationRequest,
    CreateEvaluationResponse,
    CreateFunctionCallRecord,
    CreateFunctionCallStart,
    CreateRunRequest,
    CreateRunResponse,
    EvaluationResponse,
    EvaluationSummary,
    FunctionCallResponse,
    GradeResponse,
    LateFactResponse,
    RecordObservationsRequest,
    RunResponse,
    SnapshotResponse,
    StartTurnRequest,
    TurnReplyRequest,
    TurnResponse,
)
from ..service import ById, ControlPlane
from ..state import Evaluation, Run, Snapshot

# Every route here, and any added later, refuses a host-mode evaluation on an app without a control token.
router = APIRouter(prefix="/runs", dependencies=[Depends(refuse_unguarded_host_evaluation)])


def _user_task_id(evaluation: Evaluation) -> str:
    # ControlPlane.open_evaluation puts an episode in a run only with its user task ID.
    assert evaluation.user_task_id is not None
    return evaluation.user_task_id


@router.post("", response_model=CreateRunResponse, status_code=status.HTTP_201_CREATED)
async def create_run(
    req: CreateRunRequest,
    suites: Annotated[Mapping[SuiteName, YAMLTaskSuite], Depends(get_suites)],
    control: Annotated[ControlPlane, Depends(get_control)],
):
    resolve_suite(suites, req.suite_name)
    run = control.create_run(req.suite_name)
    return CreateRunResponse(id=run.id, suite_name=run.suite_name)


@router.get("/{run_id}", response_model=RunResponse, status_code=status.HTTP_200_OK)
async def retrieve_run(
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
    config: Annotated[AppConfig, Depends(get_config)],
):
    """The run and a summary of each evaluation; without a control token, of its agent-mode evaluations only."""
    return RunResponse(
        id=run.id,
        suite_name=run.suite_name,
        created_at=run.created_at,
        evaluations=[
            EvaluationSummary(
                id=e.id,
                user_task_id=_user_task_id(e),
                injection_task_id=e.injection_task_id,
                completed=e.completed,
                stage=e.stage.value,
                graded=e.graded,
                utility=e.grade.utility if e.grade else None,
                security=e.grade.security if e.grade else None,
                grade_validity=e.grade.grade_validity if e.grade else None,
            )
            for e in control.evaluations(run.id)
            if exposed_without_control_token(config, e)
        ],
    )


@router.post(
    "/{run_id}/evaluations",
    response_model=CreateEvaluationResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        status.HTTP_403_FORBIDDEN: {
            "description": (
                "`kind` control_token_required: `interception` is host and the server has no control token. "
                "Nothing is created or kept."
            )
        }
    },
)
async def create_evaluation(
    req: CreateEvaluationRequest,
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
    config: Annotated[AppConfig, Depends(get_config)],
):
    """Create an evaluation in the run, and issue its session token.

    A host-mode evaluation (``interception`` host) needs a server started with
    a control token: without one, the run-scoped routes take any request, so a
    target that reached the server could read and change its evaluation there
    as the host. It is refused with 403 ``control_token_required``, and nothing
    is kept. One created in process on this app's control plane is refused on
    every run-scoped route instead (``refuse_unguarded_host_evaluation``).
    """
    if req.interception == "host" and config.control_token is None:
        raise control_token_required()
    return control.create_evaluation(
        run.id,
        req.user_task_id,
        req.injection_task_id,
        req.injections,
        interception=req.interception,
        idempotency_key=req.idempotency_key,
    )


@router.get(
    "/{run_id}/evaluations/{eval_id}",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
)
async def retrieve_evaluation(evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)]):
    return evaluation_response(evaluation)


def evaluation_response(evaluation: Evaluation) -> EvaluationResponse:
    """The evaluation as ``GET /runs/{run_id}/evaluations/{eval_id}`` returns it."""
    grade = evaluation.grade
    return EvaluationResponse(
        id=evaluation.id,
        user_task_id=_user_task_id(evaluation),
        injection_task_id=evaluation.injection_task_id,
        interception=evaluation.interception,
        completed=evaluation.completed,
        stage=evaluation.stage.value,
        graded=evaluation.graded,
        utility=grade.utility if grade else None,
        utility_reason=grade.utility_reason if grade else None,
        security=grade.security if grade else None,
        security_reason=grade.security_reason if grade else None,
        grade_validity=grade.grade_validity if grade else None,
        evidence_status=evaluation.evidence_status,
        agent_input=evaluation.agent_input,
        agent_output=evaluation.agent_output,
        turns=[TurnResponse.model_validate(turn) for turn in evaluation.turns],
        function_calls=[FunctionCallResponse.model_validate(fc) for fc in evaluation.function_calls],
        observations=_observations_json(evaluation.observations),
        pre_environment=evaluation.pre_environment.model_dump(),
        environment=evaluation.environment.model_dump(),
        late_facts=[LateFactResponse.model_validate(fact) for fact in evaluation.late_facts],
        capture_failures=[CaptureFailureResponse.model_validate(f) for f in evaluation.capture_failures],
    )


# --- Closure ---
#
# Closure is staged: POST .../close stops admitting new calls and waits, up to
# the control plane's closure wait (midojo-serve --closure-wait), for calls
# already admitted to finish; then the orchestrator records the final
# observations; POST .../complete seals the snapshot that grading reads. A
# complete without a close closes first. Writes that closure refuses are
# recorded as late facts.


@router.post("/{run_id}/evaluations/{eval_id}/close", response_model=CloseResponse, status_code=status.HTTP_200_OK)
async def close_evaluation(
    eval_id: str,
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> CloseResponse:
    """Stop admitting new calls, and wait for admitted calls to finish, up to the closure wait.

    Idempotent: closing again waits for what is left of the same closure wait,
    or returns at once when it's over.
    """
    return close_response(await control.close(run.id, eval_id))


def close_response(evaluation: Evaluation) -> CloseResponse:
    """The closed evaluation as ``POST .../close`` returns it."""
    return CloseResponse(
        stage=evaluation.stage.value,
        unfinished_calls=[call.call_id for call in evaluation.function_calls if not call.complete],
    )


@router.post("/{run_id}/evaluations/{eval_id}/complete", status_code=status.HTTP_200_OK)
async def complete_evaluation(
    eval_id: str,
    req: CompleteRequest,
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
):
    """Seal the evaluation with the agent's final output, closing it first if it isn't closed.

    Completing again with the same output is a no-op; another output is a 409,
    recorded as a late fact.
    """
    await control.complete(run.id, eval_id, req.agent_output, req.incomplete)
    return {"status": "completed"}


@router.post("/{run_id}/evaluations/{eval_id}/grade", response_model=GradeResponse, status_code=status.HTTP_200_OK)
async def grade_evaluation(
    eval_id: str,
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
):
    """Grade the sealed snapshot. The first grade is kept: grading again returns it."""
    return control.grade(run.id, eval_id)


@router.get("/{run_id}/evaluations/{eval_id}/snapshot", response_model=SnapshotResponse, status_code=status.HTTP_200_OK)
async def get_snapshot(
    eval_id: str,
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
):
    """The sealed snapshot grading reads; refused (400) before the seal."""
    return snapshot_response(control.snapshot(run.id, eval_id))


def snapshot_response(snapshot: Snapshot) -> SnapshotResponse:
    """The snapshot as ``GET .../snapshot`` returns it.

    Bytes that aren't UTF-8 are escaped (``\\xff``), not refused: the response
    is served, and a client rebuilding the snapshot from it finds its digest
    differs (``client.snapshot_from_response``).
    """
    return SnapshotResponse(
        evaluation_id=snapshot.evaluation_id,
        run_id=snapshot.run_id,
        snapshot_id=snapshot.snapshot_id,
        user_task_id=snapshot.user_task_id,
        injection_task_id=snapshot.injection_task_id,
        agent_output=snapshot.agent_output,
        pre_environment=json_data(snapshot.pre_environment),
        environment=json_data(snapshot.environment),
        function_calls=[json_data(call) for call in snapshot.function_calls],
        turns=[TurnResponse.model_validate(turn) for turn in snapshot.turns],
        observations=_observations_json(snapshot.observations),
        evidence_status=dict(snapshot.evidence_status),
        sealed_at=snapshot.sealed_at,
        capture_failures=[CaptureFailureResponse.model_validate(f) for f in snapshot.capture_failures],
    )


# --- Environment endpoints (nested under evaluation) ---


@router.get("/{run_id}/evaluations/{eval_id}/environment", status_code=status.HTTP_200_OK)
async def get_environment(evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)]) -> dict:
    return evaluation.environment.model_dump()


@router.put("/{run_id}/evaluations/{eval_id}/environment")
async def update_environment(
    body: dict,
    evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> dict:
    env = validate_environment(evaluation, body)
    return control.replace_environment(ById(evaluation.run_id, evaluation.id), env).model_dump()


@router.delete("/{run_id}/evaluations/{eval_id}/session", status_code=204)
async def close_session(
    eval_id: str,
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> None:
    control.close_session(run.id, eval_id)


# --- Function call endpoints ---


@router.get(
    "/{run_id}/evaluations/{eval_id}/function-calls",
    response_model=list[FunctionCallResponse],
    status_code=status.HTTP_200_OK,
)
async def list_function_calls(
    evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)],
) -> list[FunctionCallRecord]:
    return list(evaluation.function_calls)


@router.get(
    "/{run_id}/evaluations/{eval_id}/function-calls/{idx}",
    response_model=FunctionCallResponse,
    status_code=status.HTTP_200_OK,
)
async def get_function_call(
    idx: int, evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)]
) -> FunctionCallRecord:
    if idx < 0 or idx >= len(evaluation.function_calls):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Function call index out of range: {idx}")
    return evaluation.function_calls[idx]


@router.post(
    "/{run_id}/evaluations/{eval_id}/function-calls",
    response_model=FunctionCallResponse,
    status_code=status.HTTP_201_CREATED,
    responses=RECORD_FUNCTION_CALL_RESPONSES,
    # The ID FastAPI generated before this handler was renamed: keep it stable for API clients.
    operation_id="record_function_call_runs__run_id__evaluations__eval_id__function_calls_post",
)
async def record_run_function_call(
    eval_id: str,
    req: CreateFunctionCallRecord,
    response: Response,
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> FunctionCallRecord:
    """Record a call in the evaluation, at most once per ``call_id``.

    201 when recorded, or when it finishes a started ``call_id`` in place; 200
    with the stored record when ``call_id`` is already finished with the same
    body; 409 when it's finished with another body, or was started with
    another function or args, or when closure no longer admits it (recorded as
    a late fact), as on ``POST /agent/function-calls``.
    """
    return created_or_replayed(control.finish_call(ById(run.id, eval_id), req), response)


@router.post(
    "/{run_id}/evaluations/{eval_id}/function-calls/start",
    response_model=FunctionCallResponse,
    status_code=status.HTTP_201_CREATED,
    responses=START_FUNCTION_CALL_RESPONSES,
)
async def start_run_function_call(
    eval_id: str,
    req: CreateFunctionCallStart,
    response: Response,
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> FunctionCallRecord:
    """Start a call in the evaluation: an unfinished record that ``POST .../function-calls`` finishes.

    201 when started; 200 with the stored record when ``call_id`` is already
    recorded, finished or not, with the same function and args; 409 when it's
    recorded with another function or args, or once closure has begun
    (recorded as a late fact), as on ``POST /agent/function-calls/start``.
    """
    return created_or_replayed(control.start_call(ById(run.id, eval_id), req), response)


# --- Turn endpoints ---
#
# The orchestrator starts each turn before sending its message to the agent and
# records the reply once the agent answers. Function calls recorded in between
# carry that turn's index. Turns run one at a time: the next starts only after
# the previous one's reply is recorded.


@router.get(
    "/{run_id}/evaluations/{eval_id}/turns",
    response_model=list[TurnResponse],
    status_code=status.HTTP_200_OK,
)
async def list_turns(evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)]) -> list[TurnResponse]:
    return [TurnResponse.model_validate(turn) for turn in evaluation.turns]


@router.post(
    "/{run_id}/evaluations/{eval_id}/turns",
    response_model=TurnResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start_turn(
    eval_id: str,
    req: StartTurnRequest,
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> TurnResponse:
    return TurnResponse.model_validate(control.start_turn(run.id, eval_id, req.message))


@router.post(
    "/{run_id}/evaluations/{eval_id}/turns/{index}/reply",
    response_model=TurnResponse,
    status_code=status.HTTP_200_OK,
)
async def record_turn_reply(
    eval_id: str,
    index: int,
    req: TurnReplyRequest,
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> TurnResponse:
    return TurnResponse.model_validate(control.record_turn_reply(run.id, eval_id, index, req.reply))


# --- Observation endpoints ---
#
# Runtime observations: evidence the orchestrator collects outside the agent
# (e.g. OpenShell's workdir diff and OCSF events), keyed by source. There is no
# /agent route for them, so an agent's session token can't write them. Each
# source's data is validated against its registered model (midojo.observations)
# when it's recorded. They're accepted until the seal; after it, they're late
# facts. Verifiers read the models from VerificationContext.observations at
# grade time.


def _observations_json(observations: Mapping) -> dict:
    return {source: json_data(observed) for source, observed in observations.items()}


@router.get("/{run_id}/evaluations/{eval_id}/observations", status_code=status.HTTP_200_OK)
async def get_observations(evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)]) -> dict:
    return _observations_json(evaluation.observations)


@router.post("/{run_id}/evaluations/{eval_id}/observations", status_code=status.HTTP_200_OK)
async def record_observations(
    eval_id: str,
    req: RecordObservationsRequest,
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> dict:
    observed = validate_observations(req.source, req.data)
    return _observations_json(control.record_observations(run.id, eval_id, req.source, observed))
