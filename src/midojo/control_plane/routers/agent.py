"""Callbacks used by the interception layer (MCP and framework SDKs).

Each callback's session token authorizes access to one evaluation.
"""

from collections.abc import Mapping
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from midojo.types import SuiteName
from midojo.yaml_task_suite import YAMLTaskSuite

from ..dependencies import (
    RECORD_FUNCTION_CALL_RESPONSES,
    get_session_token,
    get_store,
    get_suites,
    record_function_call,
    resolve_suite,
    validate_environment,
)
from ..models import CreateFunctionCallRecord, FunctionCallResponse
from ..store import Store

router = APIRouter(prefix="/agent")


@router.get("/environment")
async def get_environment(
    session_token: Annotated[str, Depends(get_session_token)], store: Annotated[Store, Depends(get_store)]
) -> dict:
    evaluation = store.session_evaluation(session_token)
    return evaluation.environment.model_dump()


@router.put("/environment")
async def put_environment(
    body: dict,
    session_token: Annotated[str, Depends(get_session_token)],
    store: Annotated[Store, Depends(get_store)],
    suites: Annotated[Mapping[SuiteName, YAMLTaskSuite], Depends(get_suites)],
) -> dict:
    evaluation = store.session_evaluation(session_token)
    run = store.get_run(evaluation.run_id)
    assert run is not None
    suite = resolve_suite(suites, run.suite_name)
    env = validate_environment(suite, body)
    store.set_environment(run.id, evaluation.id, env)
    return env.model_dump()


@router.post(
    "/function-calls",
    response_model=FunctionCallResponse,
    status_code=201,
    responses=RECORD_FUNCTION_CALL_RESPONSES,
    # The ID FastAPI generated before this handler was renamed: keep it stable for API clients.
    operation_id="record_function_call_agent_function_calls_post",
)
async def record_agent_function_call(
    req: CreateFunctionCallRecord,
    response: Response,
    session_token: Annotated[str, Depends(get_session_token)],
    store: Annotated[Store, Depends(get_store)],
) -> FunctionCallResponse:
    """Record a call in the session's evaluation, at most once per ``call_id``.

    201 when recorded; 200 with the stored record when ``call_id`` is already
    recorded with the same body; 409 when it's recorded with another body.
    """
    evaluation = store.session_evaluation(session_token)
    record = record_function_call(store, evaluation.run_id, evaluation.id, req, response)
    assert record is not None
    return FunctionCallResponse.model_validate(record)


@router.get("/function-calls", response_model=list[FunctionCallResponse])
async def list_function_calls(
    session_token: Annotated[str, Depends(get_session_token)], store: Annotated[Store, Depends(get_store)]
) -> list[FunctionCallResponse]:
    evaluation = store.session_evaluation(session_token)
    return [FunctionCallResponse.model_validate(call) for call in evaluation.function_calls]


@router.get("/function-calls/{idx}", response_model=FunctionCallResponse)
async def get_function_call(
    idx: int, session_token: Annotated[str, Depends(get_session_token)], store: Annotated[Store, Depends(get_store)]
) -> FunctionCallResponse:
    evaluation = store.session_evaluation(session_token)
    if idx < 0 or idx >= len(evaluation.function_calls):
        raise HTTPException(404, f"Function call index out of range: {idx}")
    return FunctionCallResponse.model_validate(evaluation.function_calls[idx])
