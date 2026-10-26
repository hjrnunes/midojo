"""The interception layer's seams, in the control plane's own process: no HTTP layer, the same single writer.

:class:`InProcessAgentClient` is a ``midojo_sdk.recorder.Recorder`` (and
``CaptureReporter``) and ``WorldState`` for the evaluation a session token is bound to, as
``midojo_sdk.client.AgentControlPlaneClient`` is over HTTP. It calls
:class:`~midojo.control_plane.service.ControlPlane` directly, so every event
goes through the same lifecycle transitions, under the same lock, as the
``/agent`` routes; it never makes an HTTP request. Only the edge that turns a
rejection into a status code is missing: a refusal raises the SDK's
``ControlPlaneRejectedError`` with the same kind the HTTP client reads from the
response body, and a request the ``/agent`` API would refuse as malformed (a
422 over HTTP) raises the SDK's ``InvalidRequestError``, as the HTTP client
does.

This module needs ``midojo-sdk`` installed (the ``dev`` and ``suites``
extras); nothing else in ``midojo`` imports it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar

from midojo_sdk.recorder import (
    CAPTURE_REASON_MAX_LENGTH,
    REFUSED,
    CapturePhase,
    ControlPlaneRejectedError,
    Disposition,
    EnvironmentRevision,
    InjectionRecord,
    InvalidRequestError,
    RejectionKind,
    finish_fields,
)
from pydantic import BaseModel, ValidationError

from .lifecycle import RejectedError, Rejection
from .models import (
    ChangeEnvironmentRequest,
    CreateFunctionCallRecord,
    CreateFunctionCallStart,
    ReportCaptureFailureRequest,
)
from .service import BySession, ControlPlane

RequestT = TypeVar("RequestT", bound=BaseModel)


def sdk_error(error: RejectedError) -> ControlPlaneRejectedError:
    """The SDK's error for a refusal by the control plane: the same kind, detail and late-fact flag."""
    return ControlPlaneRejectedError(RejectionKind(error.kind.value), error.rejected.detail, late_fact=error.late)


def _request(model: type[RequestT], **fields: Any) -> RequestT:
    """The request body the ``/agent`` route would validate, or the SDK's ``InvalidRequestError`` for its 422."""
    try:
        return model.model_validate(fields)
    except ValidationError as e:
        raise InvalidRequestError(str(e.errors(include_url=False))) from e


class InProcessAgentClient:
    """Records calls in, and reads or changes the environment of, the evaluation ``session_token`` is bound to.

    Each method submits one event to the control plane's writer and returns
    once it's committed or refused. The records it makes are the ones the HTTP
    client makes for the same calls: each start and finish carries its
    disposition, and a finish the ``blocked`` flag the disposition implies.
    Nothing is resent: a refusal is final, and there is no transport to fail.
    """

    def __init__(self, control: ControlPlane, session_token: str) -> None:
        self._control = control
        self._target = BySession(session_token)

    # --- Recorder ---

    async def start_call(self, *, call_id: str, function: str, args: dict[str, Any], disposition: Disposition) -> None:
        req = _request(CreateFunctionCallStart, call_id=call_id, function=function, args=args, disposition=disposition)
        try:
            self._control.start_call(self._target, req)
        except RejectedError as e:
            raise sdk_error(e) from e

    async def finish_call(
        self,
        *,
        call_id: str,
        function: str,
        args: dict[str, Any],
        result: str,
        disposition: Disposition,
        error: str | None = None,
        structured_content: dict[str, Any] | None = None,
        injection_outcomes: Sequence[InjectionRecord] = (),
    ) -> None:
        req = _request(
            CreateFunctionCallRecord,
            call_id=call_id,
            function=function,
            args=args,
            result=result,
            error=error,
            blocked=disposition in REFUSED,
            disposition=disposition,
            **finish_fields(structured_content, injection_outcomes),
        )
        try:
            self._control.finish_call(self._target, req)
        except RejectedError as e:
            raise sdk_error(e) from e

    # --- CaptureReporter ---

    async def report_capture_failure(self, *, call_id: str, function: str, phase: CapturePhase, reason: str) -> None:
        """Report a record that never reached the control plane: the evaluation's ``function_calls`` is incomplete."""
        # Cut as the HTTP client cuts it.
        reason = reason[:CAPTURE_REASON_MAX_LENGTH]
        req = _request(ReportCaptureFailureRequest, call_id=call_id, function=function, phase=phase, reason=reason)
        try:
            self._control.report_capture_failure(
                self._target, call_id=req.call_id, function=req.function, phase=req.phase, reason=req.reason
            )
        except RejectedError as e:
            raise sdk_error(e) from e

    # --- WorldState ---

    async def read(self) -> EnvironmentRevision:
        try:
            evaluation = self._control.evaluation(self._target)
        except RejectedError as e:
            raise sdk_error(e) from e
        # As JSON data, as the HTTP client gets it.
        return EnvironmentRevision(evaluation.environment_revision, evaluation.environment.model_dump(mode="json"))

    async def change(
        self, values: dict[str, Any], *, expected_revision: int | None = None
    ) -> EnvironmentRevision | None:
        # The body the PATCH route validates: the same inputs are accepted, coerced or refused as over HTTP.
        req = _request(ChangeEnvironmentRequest, set=values, expected_revision=expected_revision)
        try:
            revision, environment = self._control.change_environment(self._target, req.set, req.expected_revision)
        except RejectedError as e:
            if e.kind is Rejection.REVISION_CONFLICT and req.expected_revision is not None:
                return None
            raise sdk_error(e) from e
        return EnvironmentRevision(revision, environment.model_dump(mode="json"))
