from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ValidationError

from midojo.observations import observation_type
from midojo.types import Environment, FunctionCallRecord, SuiteName
from midojo.yaml_task_suite import YAMLTaskSuite

from .config import AppConfig
from .models import CreateFunctionCallRecord, CreateFunctionCallStart
from .state import Evaluation, Run
from .store import FunctionCallConflictError, Store


async def get_store(request: Request) -> Store:
    return request.app.state.store


async def get_config(request: Request) -> AppConfig:
    return request.app.state.config


async def get_suites(request: Request) -> Mapping[SuiteName, YAMLTaskSuite]:
    return request.app.state.suites


def resolve_suite(suites: Mapping[SuiteName, YAMLTaskSuite], suite_name: SuiteName) -> YAMLTaskSuite:
    suite = suites.get(suite_name)
    if suite is None:
        raise HTTPException(404, f"Unknown suite: {suite_name}")
    return suite


async def get_suite(
    suite_name: SuiteName, suites: Annotated[Mapping[SuiteName, YAMLTaskSuite], Depends(get_suites)]
) -> YAMLTaskSuite:
    return resolve_suite(suites, suite_name)


async def get_run(run_id: str, store: Annotated[Store, Depends(get_store)]) -> Run:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, f"Unknown run: {run_id}")
    return run


async def get_run_suite(
    run: Annotated[Run, Depends(get_run)], suites: Annotated[Mapping[SuiteName, YAMLTaskSuite], Depends(get_suites)]
) -> YAMLTaskSuite:
    return resolve_suite(suites, run.suite_name)


async def get_evaluation_by_id(
    eval_id: str,
    run: Annotated[Run, Depends(get_run)],
    store: Annotated[Store, Depends(get_store)],
) -> Evaluation:
    evaluation = store.get_evaluation(run.id, eval_id)
    if evaluation is None:
        raise HTTPException(404, f"Unknown evaluation: {eval_id}")
    return evaluation


async def get_session_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(HTTPBearer(auto_error=False))],
) -> str:
    if credentials is None:
        raise HTTPException(401, "Evaluation session token required", headers={"WWW-Authenticate": "Bearer"})
    return credentials.credentials


def validate_environment(suite: YAMLTaskSuite, body: dict) -> Environment:
    try:
        return suite.environment_type.model_validate(body)
    except ValidationError as exc:
        errors = [{**error, "loc": ("body", *error["loc"])} for error in exc.errors()]
        raise RequestValidationError(errors) from exc


def validate_observations(source: str, data: dict) -> BaseModel:
    model = observation_type(source)
    if model is None:
        error = {"type": "value_error", "loc": ("body", "source"), "msg": f"Unknown observation source: {source!r}"}
        raise RequestValidationError([{**error, "input": source}])
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        errors = [{**error, "loc": ("body", "data", *error["loc"])} for error in exc.errors()]
        raise RequestValidationError(errors) from exc


RECORD_FUNCTION_CALL_RESPONSES: dict[int | str, dict] = {
    status.HTTP_200_OK: {
        "description": "Replay: the call_id is already finished with this body. Nothing is appended or changed."
    },
    status.HTTP_201_CREATED: {
        "description": "The call is recorded: appended, or, when the call_id was started, its record finished."
    },
    status.HTTP_409_CONFLICT: {
        "description": (
            "The call_id is already finished with a different body, or was started with another function or "
            "args. Nothing is appended or changed."
        )
    },
}

START_FUNCTION_CALL_RESPONSES: dict[int | str, dict] = {
    status.HTTP_200_OK: {
        "description": (
            "Replay: the call_id is already recorded, finished or not, with this function and args. "
            "Nothing is appended."
        )
    },
    status.HTTP_201_CREATED: {"description": "The call is started: an unfinished record is appended."},
    status.HTTP_409_CONFLICT: {
        "description": "The call_id is already recorded with another function or args. Nothing is appended."
    },
}


def record_function_call(
    store: Store, run_id: str, eval_id: str, req: CreateFunctionCallRecord, response: Response
) -> FunctionCallRecord | None:
    """Record a function call idempotently by ``call_id``, for both record routes.

    Returns the new record (201), or the stored one when ``req`` replays its
    ``call_id`` with the same body (200); raises a 409 when the body differs.
    When ``call_id`` was started and not finished, finishes that record in
    place (201), or raises a 409 when the function or args differ from the
    start. Returns None if the evaluation is unknown.
    """
    try:
        appended = store.append_function_call(run_id, eval_id, req)
    except FunctionCallConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return _created_or_replayed(appended, response)


def start_function_call(
    store: Store, run_id: str, eval_id: str, req: CreateFunctionCallStart, response: Response
) -> FunctionCallRecord | None:
    """Start a function call idempotently by ``call_id``, for both start routes.

    Returns the new, unfinished record (201), or the stored one, finished or
    not, when ``req`` replays its ``call_id`` with the same function and args
    (200); raises a 409 when they differ. Returns None if the evaluation is
    unknown.
    """
    try:
        started = store.start_function_call(run_id, eval_id, req)
    except FunctionCallConflictError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return _created_or_replayed(started, response)


def _created_or_replayed(
    stored: tuple[FunctionCallRecord, bool] | None, response: Response
) -> FunctionCallRecord | None:
    if stored is None:
        return None
    record, created = stored
    if not created:
        response.status_code = status.HTTP_200_OK
    return record
