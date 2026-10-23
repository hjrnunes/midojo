"""Callbacks used by the interception layer (MCP and framework SDKs).

Each callback's session token authorizes access to one evaluation. Every
callback goes through the control plane's writer, with the same lifecycle
rules as the run-scoped routes; only the status codes are chosen here, at the
edge (``dependencies.REJECTION_STATUS``).
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response

from ..dependencies import (
    RECORD_FUNCTION_CALL_RESPONSES,
    START_FUNCTION_CALL_RESPONSES,
    created_or_replayed,
    get_control,
    get_session_token,
    validate_environment,
)
from ..models import CreateFunctionCallRecord, CreateFunctionCallStart, FunctionCallResponse
from ..service import BySession, ControlPlane

router = APIRouter(prefix="/agent")


@router.get("/environment")
async def get_environment(
    session_token: Annotated[str, Depends(get_session_token)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> dict:
    return control.evaluation(BySession(session_token)).environment.model_dump()


@router.put("/environment")
async def put_environment(
    body: dict,
    session_token: Annotated[str, Depends(get_session_token)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> dict:
    target = BySession(session_token)
    evaluation = control.evaluation(target)
    env = validate_environment(control.suite(evaluation.run_id), body)
    return control.replace_environment(target, env).model_dump()


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
    control: Annotated[ControlPlane, Depends(get_control)],
) -> FunctionCallResponse:
    """Record a call in the session's evaluation, at most once per ``call_id``.

    201 when recorded, or when it finishes a started ``call_id`` in place; 200
    with the stored record when ``call_id`` is already finished with the same
    body; 409 when it's finished with another body, or was started with
    another function or args, or when closure no longer admits it (recorded as
    a late fact); 401 after the seal (also recorded as a late fact).
    """
    record = created_or_replayed(control.finish_call(BySession(session_token), req), response)
    return FunctionCallResponse.model_validate(record)


@router.post(
    "/function-calls/start",
    response_model=FunctionCallResponse,
    status_code=201,
    responses=START_FUNCTION_CALL_RESPONSES,
)
async def start_agent_function_call(
    req: CreateFunctionCallStart,
    response: Response,
    session_token: Annotated[str, Depends(get_session_token)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> FunctionCallResponse:
    """Start a call in the session's evaluation: an unfinished record that ``POST /function-calls`` finishes.

    201 when started; 200 with the stored record when ``call_id`` is already
    recorded, finished or not, with the same function and args; 409 when it's
    recorded with another function or args, or once closure has begun (recorded
    as a late fact); 401 after the seal (also recorded as a late fact).
    """
    record = created_or_replayed(control.start_call(BySession(session_token), req), response)
    return FunctionCallResponse.model_validate(record)


@router.get("/function-calls", response_model=list[FunctionCallResponse])
async def list_function_calls(
    session_token: Annotated[str, Depends(get_session_token)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> list[FunctionCallResponse]:
    evaluation = control.evaluation(BySession(session_token))
    return [FunctionCallResponse.model_validate(call) for call in evaluation.function_calls]


@router.get("/function-calls/{idx}", response_model=FunctionCallResponse)
async def get_function_call(
    idx: int,
    session_token: Annotated[str, Depends(get_session_token)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> FunctionCallResponse:
    evaluation = control.evaluation(BySession(session_token))
    if idx < 0 or idx >= len(evaluation.function_calls):
        raise HTTPException(404, f"Function call index out of range: {idx}")
    return FunctionCallResponse.model_validate(evaluation.function_calls[idx])
