"""The orchestrator's control client: what it needs from a control plane, and the HTTP implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import httpx
from pydantic import BaseModel

from midojo.control_plane.models import (
    CloseResponse,
    CreateEvaluationResponse,
    CreateRunResponse,
    EvaluationResponse,
    GradeResponse,
    TurnResponse,
)
from midojo.types import InterceptionMode

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
    ) -> CreateEvaluationResponse: ...

    async def start_turn(self, run_id: str, eval_id: str, message: str) -> TurnResponse: ...

    async def record_turn_reply(self, run_id: str, eval_id: str, index: int, reply: str) -> TurnResponse: ...

    async def close_evaluation(self, run_id: str, eval_id: str) -> CloseResponse: ...

    async def complete_evaluation(self, run_id: str, eval_id: str, agent_output: str) -> None: ...

    async def grade_evaluation(self, run_id: str, eval_id: str) -> GradeResponse: ...

    async def evaluation(self, run_id: str, eval_id: str) -> EvaluationResponse: ...

    async def record_observations(self, run_id: str, eval_id: str, source: str, observed: BaseModel) -> None: ...

    async def revoke_session(self, run_id: str, eval_id: str) -> None: ...

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
    ) -> CreateEvaluationResponse:
        response = await self._http.post(
            f"{self._base_url}/runs/{run_id}/evaluations",
            headers=self._headers,
            json={
                "user_task_id": user_task_id,
                "injection_task_id": injection_task_id,
                "injections": injections,
                "interception": interception,
            },
        )
        response.raise_for_status()
        return CreateEvaluationResponse.model_validate(response.json())

    async def start_turn(self, run_id: str, eval_id: str, message: str) -> TurnResponse:
        response = await self._http.post(
            f"{self._base_url}/runs/{run_id}/evaluations/{eval_id}/turns",
            headers=self._headers,
            json={"message": message},
        )
        response.raise_for_status()
        return TurnResponse.model_validate(response.json())

    async def record_turn_reply(self, run_id: str, eval_id: str, index: int, reply: str) -> TurnResponse:
        response = await self._http.post(
            f"{self._base_url}/runs/{run_id}/evaluations/{eval_id}/turns/{index}/reply",
            headers=self._headers,
            json={"reply": reply},
        )
        response.raise_for_status()
        return TurnResponse.model_validate(response.json())

    async def close_evaluation(self, run_id: str, eval_id: str) -> CloseResponse:
        """Stop admitting new calls, and wait (up to the control plane's closure wait) for admitted ones to finish."""
        response = await self._http.post(
            f"{self._base_url}/runs/{run_id}/evaluations/{eval_id}/close", headers=self._headers
        )
        response.raise_for_status()
        return CloseResponse.model_validate(response.json())

    async def complete_evaluation(self, run_id: str, eval_id: str, agent_output: str) -> None:
        response = await self._http.post(
            f"{self._base_url}/runs/{run_id}/evaluations/{eval_id}/complete",
            headers=self._headers,
            json={"agent_output": agent_output},
        )
        response.raise_for_status()

    async def grade_evaluation(self, run_id: str, eval_id: str) -> GradeResponse:
        response = await self._http.post(
            f"{self._base_url}/runs/{run_id}/evaluations/{eval_id}/grade", headers=self._headers
        )
        response.raise_for_status()
        return GradeResponse.model_validate(response.json())

    async def evaluation(self, run_id: str, eval_id: str) -> EvaluationResponse:
        response = await self._http.get(f"{self._base_url}/runs/{run_id}/evaluations/{eval_id}", headers=self._headers)
        response.raise_for_status()
        return EvaluationResponse.model_validate(response.json())

    async def record_observations(self, run_id: str, eval_id: str, source: str, observed: BaseModel) -> None:
        response = await self._http.post(
            f"{self._base_url}/runs/{run_id}/evaluations/{eval_id}/observations",
            headers=self._headers,
            json={"source": source, "data": observed.model_dump(mode="json")},
        )
        response.raise_for_status()

    def recorder(self, session_token: str) -> Recorder:
        """The session's recorder over HTTP, on this client's connection pool, resending after a transient failure."""
        from midojo_sdk.client import AgentControlPlaneClient, transient
        from midojo_sdk.recorder import RetryingRecorder

        return RetryingRecorder(
            AgentControlPlaneClient(self._base_url, session_token, http=self._http), retryable=transient
        )

    async def revoke_session(self, run_id: str, eval_id: str) -> None:
        response = await self._http.delete(
            f"{self._base_url}/runs/{run_id}/evaluations/{eval_id}/session", headers=self._headers
        )
        response.raise_for_status()
