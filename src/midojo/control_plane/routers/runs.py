from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from midojo.types import FunctionCallRecord, SuiteName
from midojo.verifiers import ConversationTurn
from midojo.yaml_task_suite import YAMLTaskSuite

from ..config import AppConfig
from ..dependencies import (
    RECORD_FUNCTION_CALL_RESPONSES,
    get_config,
    get_evaluation_by_id,
    get_run,
    get_run_suite,
    get_store,
    get_suites,
    record_function_call,
    resolve_suite,
    validate_environment,
    validate_observations,
)
from ..models import (
    CompleteRequest,
    CreateEvaluationRequest,
    CreateEvaluationResponse,
    CreateFunctionCallRecord,
    CreateRunRequest,
    CreateRunResponse,
    EvaluationResponse,
    EvaluationSummary,
    FunctionCallResponse,
    GradeResponse,
    RecordObservationsRequest,
    RunResponse,
    StartTurnRequest,
    TurnReplyRequest,
    TurnResponse,
)
from ..state import Evaluation, Run
from ..store import Store

router = APIRouter(prefix="/runs")


def _require_eval(evaluation: Evaluation | None, eval_id: str) -> Evaluation:
    """404 when an ID-based store mutation reports the evaluation doesn't exist."""
    if evaluation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown evaluation: {eval_id}")
    return evaluation


@router.post("", response_model=CreateRunResponse, status_code=status.HTTP_201_CREATED)
async def create_run(
    req: CreateRunRequest,
    suites: Annotated[Mapping[SuiteName, YAMLTaskSuite], Depends(get_suites)],
    store: Annotated[Store, Depends(get_store)],
):
    resolve_suite(suites, req.suite_name)
    run = store.create_run(req.suite_name)
    return CreateRunResponse(id=run.id, suite_name=run.suite_name)


@router.get("/{run_id}", response_model=RunResponse, status_code=status.HTTP_200_OK)
async def retrieve_run(run: Annotated[Run, Depends(get_run)]):
    return RunResponse(
        id=run.id,
        suite_name=run.suite_name,
        created_at=run.created_at,
        evaluations=[
            EvaluationSummary(
                id=e.id,
                user_task_id=e.user_task_id,
                injection_task_id=e.injection_task_id,
                completed=e.completed,
                utility=e.utility,
                security=e.security,
            )
            for e in list(run.evaluations.values())
        ],
    )


@router.post("/{run_id}/evaluations", response_model=CreateEvaluationResponse, status_code=status.HTTP_201_CREATED)
async def create_evaluation(
    req: CreateEvaluationRequest,
    config: Annotated[AppConfig, Depends(get_config)],
    run: Annotated[Run, Depends(get_run)],
    suite: Annotated[YAMLTaskSuite, Depends(get_run_suite)],
    store: Annotated[Store, Depends(get_store)],
):
    if req.user_task_id not in suite.user_tasks:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown user task: {req.user_task_id}")
    if req.injection_task_id is not None and req.injection_task_id not in suite.injection_tasks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown injection task: {req.injection_task_id}"
        )

    environment = suite.provision_environment(req.injections)
    pre_environment = environment.model_copy(deep=True)
    turns = suite.inject_user_task_turns(req.user_task_id, req.injections)
    prompt = turns[0]

    evaluation = store.create_evaluation(
        run.id,
        user_task_id=req.user_task_id,
        injection_task_id=req.injection_task_id,
        pre_environment=pre_environment,
        environment=environment,
        active_injections=req.injections,
        agent_input=prompt,
    )
    token, expires_at = store.create_session(run.id, evaluation.id, config.session_ttl_seconds)
    return CreateEvaluationResponse(
        id=evaluation.id, prompt=prompt, turns=turns, session_token=token, session_expires_at=expires_at
    )


@router.get(
    "/{run_id}/evaluations/{eval_id}",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
)
async def retrieve_evaluation(evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)]):
    return EvaluationResponse(
        id=evaluation.id,
        user_task_id=evaluation.user_task_id,
        injection_task_id=evaluation.injection_task_id,
        completed=evaluation.completed,
        utility=evaluation.utility,
        security=evaluation.security,
        security_reason=evaluation.security_reason,
        agent_input=evaluation.agent_input,
        agent_output=evaluation.agent_output,
        turns=[TurnResponse.model_validate(turn) for turn in evaluation.turns],
        function_calls=[FunctionCallResponse.model_validate(fc) for fc in evaluation.function_calls],
        pre_environment=evaluation.pre_environment.model_dump(),
        environment=evaluation.environment.model_dump(),
    )


@router.post("/{run_id}/evaluations/{eval_id}/complete", status_code=status.HTTP_200_OK)
async def complete_evaluation(
    eval_id: str,
    req: CompleteRequest,
    run: Annotated[Run, Depends(get_run)],
    store: Annotated[Store, Depends(get_store)],
):
    _require_eval(store.complete_evaluation(run.id, eval_id, req.agent_output), eval_id)
    return {"status": "completed"}


@router.post("/{run_id}/evaluations/{eval_id}/grade", response_model=GradeResponse, status_code=status.HTTP_200_OK)
async def grade_evaluation(
    evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)],
    suite: Annotated[YAMLTaskSuite, Depends(get_run_suite)],
    store: Annotated[Store, Depends(get_store)],
):
    if not evaluation.completed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Evaluation not completed. Call complete first."
        )

    result = suite.grade(
        user_task_id=evaluation.user_task_id,
        injection_task_id=evaluation.injection_task_id,
        agent_output=evaluation.agent_output or "",
        pre_environment=evaluation.pre_environment,
        post_environment=evaluation.environment,
        function_calls=evaluation.function_calls,
        observations=evaluation.observations,
        turns=[ConversationTurn(message=turn.message, reply=turn.reply) for turn in evaluation.turns],
    )
    graded = GradeResponse.model_validate(result)
    store.set_grade(
        evaluation.run_id,
        evaluation.id,
        utility=graded.utility,
        security=graded.security,
        security_reason=graded.security_reason,
    )
    return graded


# --- Environment endpoints (nested under evaluation) ---


@router.get("/{run_id}/evaluations/{eval_id}/environment", status_code=status.HTTP_200_OK)
async def get_environment(evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)]) -> dict:
    return evaluation.environment.model_dump()


@router.put("/{run_id}/evaluations/{eval_id}/environment")
async def update_environment(
    body: dict,
    evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)],
    suite: Annotated[YAMLTaskSuite, Depends(get_run_suite)],
    store: Annotated[Store, Depends(get_store)],
) -> dict:
    env = validate_environment(suite, body)
    updated = _require_eval(store.set_environment(evaluation.run_id, evaluation.id, env), evaluation.id)
    return updated.environment.model_dump()


@router.delete("/{run_id}/evaluations/{eval_id}/session", status_code=204)
async def close_session(
    evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)],
    store: Annotated[Store, Depends(get_store)],
) -> None:
    store.close_session(evaluation.run_id, evaluation.id)


# --- Function call endpoints ---


@router.get(
    "/{run_id}/evaluations/{eval_id}/function-calls",
    response_model=list[FunctionCallResponse],
    status_code=status.HTTP_200_OK,
)
async def list_function_calls(
    evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)],
) -> list[FunctionCallRecord]:
    return evaluation.function_calls


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
    store: Annotated[Store, Depends(get_store)],
) -> FunctionCallRecord:
    """Record a call in the evaluation, at most once per ``call_id``.

    201 when recorded; 200 with the stored record when ``call_id`` is already
    recorded with the same body; 409 when it's recorded with another body.
    """
    record = record_function_call(store, run.id, eval_id, req, response)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Unknown evaluation: {eval_id}")
    return record


# --- Turn endpoints ---
#
# The orchestrator starts each turn before sending its message to the agent and
# records the reply once the agent answers. Function calls recorded in between
# carry that turn's index. Turns run one at a time: the next starts only after
# the previous one's reply is recorded.


def _require_open(evaluation: Evaluation) -> None:
    if evaluation.completed:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Evaluation already completed")


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
    req: StartTurnRequest,
    evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)],
    store: Annotated[Store, Depends(get_store)],
) -> TurnResponse:
    _require_open(evaluation)
    if evaluation.turns and evaluation.turns[-1].ended_at is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Turn {evaluation.turns[-1].index} has no reply yet",
        )
    updated = _require_eval(store.start_turn(evaluation.run_id, evaluation.id, req.message), evaluation.id)
    return TurnResponse.model_validate(updated.turns[-1])


@router.post(
    "/{run_id}/evaluations/{eval_id}/turns/{index}/reply",
    response_model=TurnResponse,
    status_code=status.HTTP_200_OK,
)
async def record_turn_reply(
    index: int,
    req: TurnReplyRequest,
    evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)],
    store: Annotated[Store, Depends(get_store)],
) -> TurnResponse:
    _require_open(evaluation)
    if index < 0 or index >= len(evaluation.turns):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Turn not started: {index}")
    if evaluation.turns[index].ended_at is not None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=f"Turn {index} already has a reply")
    updated = _require_eval(store.record_turn_reply(evaluation.run_id, evaluation.id, index, req.reply), evaluation.id)
    return TurnResponse.model_validate(updated.turns[index])


# --- Observation endpoints ---
#
# Runtime observations: evidence the orchestrator collects outside the agent
# (e.g. OpenShell's workdir diff and OCSF events), keyed by source. There is no
# /agent route for them, so an agent's session token can't write them. Each
# source's data is validated against its registered model (midojo.observations)
# when it's recorded. Verifiers read the models from
# VerificationContext.observations at grade time.


def _observations_json(evaluation: Evaluation) -> dict:
    return {source: observed.model_dump(mode="json") for source, observed in evaluation.observations.items()}


@router.get("/{run_id}/evaluations/{eval_id}/observations", status_code=status.HTTP_200_OK)
async def get_observations(evaluation: Annotated[Evaluation, Depends(get_evaluation_by_id)]) -> dict:
    return _observations_json(evaluation)


@router.post("/{run_id}/evaluations/{eval_id}/observations", status_code=status.HTTP_200_OK)
async def record_observations(
    eval_id: str,
    req: RecordObservationsRequest,
    run: Annotated[Run, Depends(get_run)],
    store: Annotated[Store, Depends(get_store)],
) -> dict:
    observed = validate_observations(req.source, req.data)
    evaluation = _require_eval(store.record_observations(run.id, eval_id, req.source, observed), eval_id)
    return _observations_json(evaluation)
