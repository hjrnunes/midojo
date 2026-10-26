"""The orchestrator's control client: what it needs from a control plane, and the HTTP implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol

import httpx
from pydantic import BaseModel

from midojo.control_plane.models import (
    CloseResponse,
    CreateEvaluationResponse,
    CreateRunResponse,
    EvaluationResponse,
    GradeResponse,
    SnapshotResponse,
    TurnResponse,
)
from midojo.control_plane.state import CaptureFailure, Snapshot, Stage, Turn
from midojo.observations import observation_type
from midojo.types import Environment, FunctionCallRecord, InterceptionMode

if TYPE_CHECKING:
    from midojo_sdk.recorder import Recorder


class ControlClient(Protocol):
    """Drives runs and evaluations on one control plane, for the orchestrator.

    :class:`ControlPlaneClient` implements it over HTTP, and
    :class:`midojo.control_plane.inprocess_control.InProcessControlClient` in
    the control plane's own process. Both return the same response models; a
    failure raises an ``httpx`` error over HTTP, and the control plane's
    ``RejectedError`` (or a ``LookupError`` or ``ValueError`` for a request
    the routes would refuse as malformed) in process.

    An evaluation is addressed by its run and its ID. ``run_id`` is None only
    for an evaluation outside any run, which only a control plane in this
    process can open (``InProcessControlClient.open_episode``); over HTTP that
    raises ValueError.
    """

    async def aclose(self) -> None: ...

    async def create_run(self, suite_name: str) -> CreateRunResponse: ...

    async def create_evaluation(
        self,
        run_id: str,
        user_task_id: str,
        injection_task_id: str | None,
        injections: dict[str, str],
        interception: InterceptionMode = "agent",
        idempotency_key: str | None = None,
    ) -> CreateEvaluationResponse:
        """Open an evaluation of the run's suite pair; resent with the same ``idempotency_key``, the same one."""
        ...

    async def start_turn(self, run_id: str | None, eval_id: str, message: str) -> TurnResponse: ...

    async def record_turn_reply(self, run_id: str | None, eval_id: str, index: int, reply: str) -> TurnResponse: ...

    async def close_evaluation(self, run_id: str | None, eval_id: str) -> CloseResponse: ...

    async def complete_evaluation(
        self, run_id: str | None, eval_id: str, agent_output: str, incomplete: Sequence[str] = ()
    ) -> None:
        """Seal the evaluation with ``agent_output``; ``incomplete`` names evidence the host didn't collect in full."""
        ...

    async def grade_evaluation(self, run_id: str | None, eval_id: str) -> GradeResponse: ...

    async def evaluation(self, run_id: str | None, eval_id: str) -> EvaluationResponse: ...

    async def record_observations(self, run_id: str | None, eval_id: str, source: str, observed: BaseModel) -> None: ...

    async def revoke_session(self, run_id: str | None, eval_id: str) -> None: ...

    async def snapshot(self, run_id: str | None, eval_id: str, *, environment_type: type[Environment]) -> Snapshot:
        """The sealed evaluation's snapshot, its environments as ``environment_type`` (the episode's)."""
        ...

    def recorder(self, session_token: str) -> Recorder:
        """A recorder for the evaluation ``session_token`` is bound to, for a host-side interceptor.

        It records on the same control plane, with the session as its only
        authority: the host keeps the token, and the target never sees it.
        Needs ``midojo-sdk`` (the ``dev`` and ``suites`` extras).
        """
        ...


class ControlPlaneClient:
    """Access orchestrator endpoints on one control plane, over HTTP."""

    def __init__(self, base_url: str, *, token: str | None = None, http: httpx.AsyncClient | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"} if token else {}
        self._http = http or httpx.AsyncClient(timeout=300.0)
        self._owns_http = http is None

    async def __aenter__(self) -> ControlPlaneClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    def _evaluation_url(self, run_id: str | None, eval_id: str) -> str:
        if run_id is None:
            raise ValueError("Over HTTP, an evaluation is addressed in its run: run_id is required")
        return f"{self._base_url}/runs/{run_id}/evaluations/{eval_id}"

    async def create_run(self, suite_name: str) -> CreateRunResponse:
        response = await self._http.post(
            f"{self._base_url}/runs", headers=self._headers, json={"suite_name": suite_name}
        )
        response.raise_for_status()
        return CreateRunResponse.model_validate(response.json())

    async def create_evaluation(
        self,
        run_id: str,
        user_task_id: str,
        injection_task_id: str | None,
        injections: dict[str, str],
        interception: InterceptionMode = "agent",
        idempotency_key: str | None = None,
    ) -> CreateEvaluationResponse:
        # Sent only when given: a request without one is the same as before keys existed.
        keyed = {"idempotency_key": idempotency_key} if idempotency_key is not None else {}
        response = await self._http.post(
            f"{self._base_url}/runs/{run_id}/evaluations",
            headers=self._headers,
            json={
                "user_task_id": user_task_id,
                "injection_task_id": injection_task_id,
                "injections": injections,
                "interception": interception,
                **keyed,
            },
        )
        response.raise_for_status()
        return CreateEvaluationResponse.model_validate(response.json())

    async def start_turn(self, run_id: str | None, eval_id: str, message: str) -> TurnResponse:
        response = await self._http.post(
            f"{self._evaluation_url(run_id, eval_id)}/turns",
            headers=self._headers,
            json={"message": message},
        )
        response.raise_for_status()
        return TurnResponse.model_validate(response.json())

    async def record_turn_reply(self, run_id: str | None, eval_id: str, index: int, reply: str) -> TurnResponse:
        response = await self._http.post(
            f"{self._evaluation_url(run_id, eval_id)}/turns/{index}/reply",
            headers=self._headers,
            json={"reply": reply},
        )
        response.raise_for_status()
        return TurnResponse.model_validate(response.json())

    async def close_evaluation(self, run_id: str | None, eval_id: str) -> CloseResponse:
        """Stop admitting new calls, and wait (up to the control plane's closure wait) for admitted ones to finish."""
        response = await self._http.post(f"{self._evaluation_url(run_id, eval_id)}/close", headers=self._headers)
        response.raise_for_status()
        return CloseResponse.model_validate(response.json())

    async def complete_evaluation(
        self, run_id: str | None, eval_id: str, agent_output: str, incomplete: Sequence[str] = ()
    ) -> None:
        body: dict[str, Any] = {"agent_output": agent_output}
        if incomplete:
            body["incomplete"] = list(incomplete)
        response = await self._http.post(
            f"{self._evaluation_url(run_id, eval_id)}/complete", headers=self._headers, json=body
        )
        response.raise_for_status()

    async def grade_evaluation(self, run_id: str | None, eval_id: str) -> GradeResponse:
        response = await self._http.post(f"{self._evaluation_url(run_id, eval_id)}/grade", headers=self._headers)
        response.raise_for_status()
        return GradeResponse.model_validate(response.json())

    async def evaluation(self, run_id: str | None, eval_id: str) -> EvaluationResponse:
        response = await self._http.get(f"{self._evaluation_url(run_id, eval_id)}", headers=self._headers)
        response.raise_for_status()
        return EvaluationResponse.model_validate(response.json())

    async def record_observations(self, run_id: str | None, eval_id: str, source: str, observed: BaseModel) -> None:
        response = await self._http.post(
            f"{self._evaluation_url(run_id, eval_id)}/observations",
            headers=self._headers,
            json={"source": source, "data": observed.model_dump(mode="json")},
        )
        response.raise_for_status()

    async def snapshot(self, run_id: str | None, eval_id: str, *, environment_type: type[Environment]) -> Snapshot:
        response = await self._http.get(f"{self._evaluation_url(run_id, eval_id)}/snapshot", headers=self._headers)
        response.raise_for_status()
        return snapshot_from_response(SnapshotResponse.model_validate(response.json()), environment_type)

    def recorder(self, session_token: str) -> Recorder:
        """The session's recorder over HTTP, on this client's connection pool, resending after a transient failure."""
        from midojo_sdk.client import AgentControlPlaneClient, transient
        from midojo_sdk.recorder import RetryingRecorder

        return RetryingRecorder(
            AgentControlPlaneClient(self._base_url, session_token, http=self._http), retryable=transient
        )

    async def revoke_session(self, run_id: str | None, eval_id: str) -> None:
        response = await self._http.delete(f"{self._evaluation_url(run_id, eval_id)}/session", headers=self._headers)
        response.raise_for_status()


def snapshot_from_response(response: SnapshotResponse, environment_type: type[Environment]) -> Snapshot:
    """The snapshot ``response`` carries, as the control plane holds it: environments as ``environment_type``.

    Observations are validated against the model registered for their source;
    an unregistered source raises ValueError. So does a rebuilt snapshot whose
    content digest isn't the ``snapshot_id`` the control plane sent: it isn't
    the snapshot that was sealed (another environment type, say).
    """

    def call(data: dict[str, Any]) -> FunctionCallRecord:
        # The record's environment history is typed as the base Environment: give it the episode's type.
        history = {
            key: environment_type.model_validate(data[key])
            for key in ("pre_environment", "post_environment")
            if data.get(key) is not None
        }
        return FunctionCallRecord.model_validate(data).model_copy(update=history)

    observations: dict[str, BaseModel] = {}
    for source, data in response.observations.items():
        model = observation_type(source)
        if model is None:
            raise ValueError(f"Unknown observation source: {source!r}")
        observations[source] = model.model_validate(data)
    snapshot = Snapshot(
        evaluation_id=response.evaluation_id,
        run_id=response.run_id,
        user_task_id=response.user_task_id,
        injection_task_id=response.injection_task_id,
        agent_output=response.agent_output,
        pre_environment=environment_type.model_validate(response.pre_environment),
        environment=environment_type.model_validate(response.environment),
        function_calls=tuple(call(data) for data in response.function_calls),
        turns=tuple(Turn(**turn.model_dump()) for turn in response.turns),
        observations=observations,
        evidence_status=dict(response.evidence_status),
        sealed_at=response.sealed_at,
        capture_failures=tuple(
            CaptureFailure(**{**failure.model_dump(), "stage": Stage(failure.stage)})
            for failure in response.capture_failures
        ),
    )
    if snapshot.snapshot_id != response.snapshot_id:
        raise ValueError(f"Snapshot {response.snapshot_id} didn't survive the round trip: its content differs")
    return snapshot
