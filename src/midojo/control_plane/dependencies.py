from __future__ import annotations

import secrets
from collections.abc import Mapping
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ValidationError

from midojo.observations import observation_type
from midojo.types import Environment, FunctionCallRecord, SuiteName
from midojo.yaml_task_suite import YAMLTaskSuite

from .config import AppConfig
from .lifecycle import RejectedError, Rejection
from .service import ById, ControlPlane
from .state import Evaluation, Run

# --- the edge: domain rejections as HTTP responses ---

# The one place a rejection kind becomes a status code.
REJECTION_STATUS: dict[Rejection, int] = {
    Rejection.UNKNOWN_RUN: status.HTTP_404_NOT_FOUND,
    Rejection.UNKNOWN_EVALUATION: status.HTTP_404_NOT_FOUND,
    Rejection.UNKNOWN_TASK: status.HTTP_400_BAD_REQUEST,
    Rejection.IDEMPOTENCY_CONFLICT: status.HTTP_409_CONFLICT,
    Rejection.CONTROL_TOKEN_REQUIRED: status.HTTP_403_FORBIDDEN,
    Rejection.INVALID_SESSION: status.HTTP_401_UNAUTHORIZED,
    Rejection.SESSION_CLOSED: status.HTTP_401_UNAUTHORIZED,
    Rejection.FORBIDDEN: status.HTTP_403_FORBIDDEN,
    Rejection.CLOSING: status.HTTP_409_CONFLICT,
    Rejection.SEALED: status.HTTP_409_CONFLICT,
    Rejection.CLOSURE_PENDING: status.HTTP_409_CONFLICT,
    Rejection.NOT_SEALED: status.HTTP_400_BAD_REQUEST,
    Rejection.CALL_CONFLICT: status.HTTP_409_CONFLICT,
    Rejection.INVALID_DISPOSITION: status.HTTP_422_UNPROCESSABLE_CONTENT,
    Rejection.TURN_PENDING: status.HTTP_409_CONFLICT,
    Rejection.TURN_NOT_STARTED: status.HTTP_404_NOT_FOUND,
    Rejection.TURN_ANSWERED: status.HTTP_409_CONFLICT,
    Rejection.REVISION_CONFLICT: status.HTTP_409_CONFLICT,
    Rejection.INVALID_ENVIRONMENT: status.HTTP_422_UNPROCESSABLE_CONTENT,
    Rejection.UNSUPPORTED_VALUE: status.HTTP_422_UNPROCESSABLE_CONTENT,
}


def rejection_response(exc: RejectedError) -> JSONResponse:
    """A refused event as a response: ``detail``, the domain ``kind``, and whether it was kept as a late fact."""
    code = REJECTION_STATUS[exc.kind]
    headers = {"WWW-Authenticate": "Bearer"} if code == status.HTTP_401_UNAUTHORIZED else None
    content = {"detail": exc.rejected.detail, "kind": exc.kind.value, "late_fact": exc.late}
    return JSONResponse(status_code=code, content=content, headers=headers)


# --- dependencies ---


async def get_control(request: Request) -> ControlPlane:
    return request.app.state.control


# The settings and suites are the control plane's own: one source for the control token a route requires and the
# one the control plane opens host mode under.
async def get_config(control: Annotated[ControlPlane, Depends(get_control)]) -> AppConfig:
    return control.config


async def get_suites(control: Annotated[ControlPlane, Depends(get_control)]) -> Mapping[SuiteName, YAMLTaskSuite]:
    return control.suites


def resolve_suite(suites: Mapping[SuiteName, YAMLTaskSuite], suite_name: SuiteName) -> YAMLTaskSuite:
    suite = suites.get(suite_name)
    if suite is None:
        raise HTTPException(404, f"Unknown suite: {suite_name}")
    return suite


async def get_suite(
    suite_name: SuiteName, suites: Annotated[Mapping[SuiteName, YAMLTaskSuite], Depends(get_suites)]
) -> YAMLTaskSuite:
    return resolve_suite(suites, suite_name)


async def get_run(run_id: str, control: Annotated[ControlPlane, Depends(get_control)]) -> Run:
    return control.run(run_id)


async def get_evaluation_by_id(
    eval_id: str,
    run: Annotated[Run, Depends(get_run)],
    control: Annotated[ControlPlane, Depends(get_control)],
) -> Evaluation:
    """A copy of the evaluation, taken through the writer: a response never reads a value being committed."""
    return control.evaluation(ById(run.id, eval_id))


async def get_session_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(HTTPBearer(auto_error=False))],
) -> str:
    if credentials is None:
        raise HTTPException(401, "Evaluation session token required", headers={"WWW-Authenticate": "Bearer"})
    return credentials.credentials


async def require_control_token(
    config: Annotated[AppConfig, Depends(get_config)],
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(HTTPBearer(auto_error=False))],
) -> None:
    if config.control_token is None:
        return
    if credentials is None or not secrets.compare_digest(
        credentials.credentials.encode(), config.control_token.encode()
    ):
        raise HTTPException(401, "Control token required", headers={"WWW-Authenticate": "Bearer"})


def validate_environment(evaluation: Evaluation, body: dict) -> Environment:
    """``body`` as the evaluation's environment type (its episode's), or a 422."""
    try:
        return evaluation.environment_type.model_validate(body)
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


_CLOSED_RESPONSES: dict[int | str, dict] = {
    status.HTTP_401_UNAUTHORIZED: {
        "description": (
            "/agent routes: the session is invalid, expired or closed. After the evaluation is sealed, a call "
            "sent with its session that isn't an exact replay (a new call, a finish, or another body under a "
            "recorded call_id) is still recorded as a late fact (`late_fact` true)."
        )
    },
}

RECORD_FUNCTION_CALL_RESPONSES: dict[int | str, dict] = {
    status.HTTP_200_OK: {
        "description": (
            "Replay: the call_id is already finished with this body, at any stage (a disposition left out on "
            "either side matches any). Nothing is appended or changed."
        )
    },
    status.HTTP_201_CREATED: {
        "description": "The call is recorded: appended, or, when the call_id was started, its record finished."
    },
    **_CLOSED_RESPONSES,
    status.HTTP_409_CONFLICT: {
        "description": (
            "`kind` call_conflict: before the seal, the call_id is already finished with a different body, or was "
            "started with another function, args or disposition, or started forwarded or simulated and finished "
            "blocked; nothing is appended or changed. `kind` closing or sealed: the evaluation no longer admits "
            "this call (a new call once closure has begun, a finish once the closure wait is over, any body but "
            "an exact replay after the seal); it is recorded as a late fact, and the evaluation's records don't "
            "change."
        )
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "description": (
            "`kind` invalid_disposition: the disposition doesn't go with `blocked` (a denied or blocked call is "
            "blocked, a forwarded or simulated one isn't), at any stage; nothing is appended, changed or recorded "
            "as a late fact. Without a `kind`, the request body itself is malformed."
        )
    },
}

START_FUNCTION_CALL_RESPONSES: dict[int | str, dict] = {
    status.HTTP_200_OK: {
        "description": (
            "Replay: the call_id is already recorded, finished or not, with this function and args, at any "
            "stage, and the same disposition when both name one. Nothing is appended."
        )
    },
    status.HTTP_201_CREATED: {"description": "The call is started: an unfinished record is appended."},
    **_CLOSED_RESPONSES,
    status.HTTP_409_CONFLICT: {
        "description": (
            "`kind` call_conflict: before the seal, the call_id is already recorded with another function, args "
            "or disposition; nothing is appended. `kind` closing or sealed: closure has begun and no new call is "
            "admitted, or the evaluation is sealed and the start isn't an exact replay; the start is recorded as "
            "a late fact."
        )
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "description": (
            "`kind` invalid_disposition: the disposition is denied or blocked; a refused call never runs, so it "
            "is recorded whole, never started. Nothing is appended. Without a `kind`, the request body itself is "
            "malformed."
        )
    },
}

REPORT_CAPTURE_FAILURE_RESPONSES: dict[int | str, dict] = {
    status.HTTP_200_OK: {
        "description": (
            "Replay: a report of the same call_id, function and phase was kept before the seal, whatever the "
            "reason, even if this one arrives after it. Nothing is kept. A report first received after the seal "
            "is only a late fact: sending it again is refused as it was, and adds no other late fact."
        )
    },
    status.HTTP_201_CREATED: {
        "description": "The report is kept, and the evaluation's function_calls evidence is marked incomplete."
    },
    **_CLOSED_RESPONSES,
}


CHANGE_ENVIRONMENT_RESPONSES: dict[int | str, dict] = {
    **_CLOSED_RESPONSES,
    status.HTTP_409_CONFLICT: {
        "description": (
            "`kind` revision_conflict: the environment is no longer at `expected_revision`; nothing is changed. "
            "`kind` closing or sealed: the closure wait is over; the change is recorded as a late fact."
        )
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "description": (
            "`kind` invalid_environment: a field is unknown, or the changed environment doesn't fit the suite's "
            "environment type; nothing is changed. Without a `kind`, the request body itself is malformed."
        )
    },
}


def created_or_replayed(stored: tuple[FunctionCallRecord, bool], response: Response) -> FunctionCallRecord:
    """The record of a start or finish, with the response's status set to 200 for a replay."""
    record, created = stored
    if not created:
        response.status_code = status.HTTP_200_OK
    return record
