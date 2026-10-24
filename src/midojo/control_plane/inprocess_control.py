"""The orchestrator's control client, in the control plane's own process: no HTTP layer, the same single writer.

:class:`InProcessControlClient` is a :class:`~midojo.control_plane.client.ControlClient`
like :class:`~midojo.control_plane.client.ControlPlaneClient`, which reaches the
run-scoped routes over HTTP. It calls
:class:`~midojo.control_plane.service.ControlPlane` directly, so every event
goes through the same lifecycle transitions, under the same lock, as those
routes, and it returns the same response models; it never makes an HTTP
request. Request bodies are validated with the routes' request models, and
observations against the model registered for their source, as the routes do.

What differs is how a failure surfaces, since there is no status code: a
refusal raises the control plane's own
:class:`~midojo.control_plane.lifecycle.RejectedError` (with its ``kind``),
an unknown suite a ``LookupError``, and an invalid request pydantic's
``ValidationError`` (a ``ValueError``). There is no control token: whoever
holds the ``ControlPlane`` is the host.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel

from midojo.episode import Episode
from midojo.observations import observation_type
from midojo.types import Environment, InterceptionMode

from .models import (
    CloseResponse,
    CompleteRequest,
    CreateEvaluationRequest,
    CreateEvaluationResponse,
    CreateRunRequest,
    CreateRunResponse,
    EvaluationResponse,
    GradeResponse,
    RecordObservationsRequest,
    StartTurnRequest,
    TurnReplyRequest,
    TurnResponse,
)
from .routers.runs import close_response, evaluation_response
from .service import ById, ControlPlane, OpenedEvaluation
from .state import Snapshot

if TYPE_CHECKING:
    from midojo_sdk.recorder import Recorder


class InProcessControlClient:
    """Drives runs and evaluations on ``control``, as the orchestrator does over HTTP with ``ControlPlaneClient``."""

    def __init__(self, control: ControlPlane) -> None:
        self._control = control

    async def __aenter__(self) -> InProcessControlClient:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Nothing to release: the control plane belongs to whoever built it."""

    async def create_run(self, suite_name: str) -> CreateRunResponse:
        req = CreateRunRequest(suite_name=suite_name)
        if req.suite_name not in self._control.suites:
            raise LookupError(f"Unknown suite: {req.suite_name}")
        run = self._control.create_run(req.suite_name)
        return CreateRunResponse(id=run.id, suite_name=run.suite_name)

    async def create_evaluation(
        self,
        run_id: str,
        user_task_id: str,
        injection_task_id: str | None,
        injections: dict[str, str],
        interception: InterceptionMode = "agent",
    ) -> CreateEvaluationResponse:
        req = CreateEvaluationRequest(
            user_task_id=user_task_id,
            injection_task_id=injection_task_id,
            injections=injections,
            interception=interception,
        )
        return self._control.create_evaluation(
            run_id, req.user_task_id, req.injection_task_id, req.injections, interception=req.interception
        )

    async def open_episode(self, episode: Episode, *, interception: InterceptionMode = "agent") -> OpenedEvaluation:
        """Open an evaluation of ``episode`` itself, outside any run: no suite or run registered.

        Its ``run_id`` is None, which is how the other methods address it.
        """
        return self._control.open_evaluation(episode, interception=interception)

    async def start_turn(self, run_id: str | None, eval_id: str, message: str) -> TurnResponse:
        req = StartTurnRequest(message=message)
        return TurnResponse.model_validate(self._control.start_turn(run_id, eval_id, req.message))

    async def record_turn_reply(self, run_id: str | None, eval_id: str, index: int, reply: str) -> TurnResponse:
        req = TurnReplyRequest(reply=reply)
        return TurnResponse.model_validate(self._control.record_turn_reply(run_id, eval_id, index, req.reply))

    async def close_evaluation(self, run_id: str | None, eval_id: str) -> CloseResponse:
        """Stop admitting new calls, and wait (up to the control plane's closure wait) for admitted ones to finish."""
        return close_response(await self._control.close(run_id, eval_id))

    async def complete_evaluation(
        self, run_id: str | None, eval_id: str, agent_output: str, incomplete: Sequence[str] = ()
    ) -> None:
        req = CompleteRequest(agent_output=agent_output, incomplete=list(incomplete))
        await self._control.complete(run_id, eval_id, req.agent_output, req.incomplete)

    async def grade_evaluation(self, run_id: str | None, eval_id: str) -> GradeResponse:
        return self._control.grade(run_id, eval_id)

    async def evaluation(self, run_id: str | None, eval_id: str) -> EvaluationResponse:
        return evaluation_response(self._control.evaluation(ById(run_id, eval_id)))

    async def record_observations(self, run_id: str | None, eval_id: str, source: str, observed: BaseModel) -> None:
        # As JSON data, as the HTTP client sends it: the recorded model shares nothing with ``observed``.
        req = RecordObservationsRequest(source=source, data=observed.model_dump(mode="json"))
        model = observation_type(req.source)
        if model is None:
            raise ValueError(f"Unknown observation source: {req.source!r}")
        self._control.record_observations(run_id, eval_id, req.source, model.model_validate(req.data))

    async def revoke_session(self, run_id: str | None, eval_id: str) -> None:
        self._control.close_session(run_id, eval_id)

    async def snapshot(self, run_id: str | None, eval_id: str, *, environment_type: type[Environment]) -> Snapshot:
        """The snapshot the control plane holds (a copy): its environments are already the episode's type."""
        return self._control.snapshot(run_id, eval_id)

    def recorder(self, session_token: str) -> Recorder:
        """The session's recorder in this process: no HTTP request, the same single writer."""
        from .inprocess import InProcessAgentClient

        return InProcessAgentClient(self._control, session_token)
