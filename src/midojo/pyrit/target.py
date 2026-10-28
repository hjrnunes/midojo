"""One PyRIT send plants one candidate and runs one MiDojo episode."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from types import ModuleType
from typing import Literal

from midojo_sdk.injection import InjectionPlan
from pyrit.models.messages.conversations import construct_response_from_request
from pyrit.models.messages.message import Message
from pyrit.prompt_target.common.prompt_target import PromptTarget

import midojo
from midojo.attacks.attacker import AttackAttempt, OutOfSurfaceError, check_plan_within, permitted_injections
from midojo.control_plane.models import GradeResponse
from midojo.control_plane.service import ControlPlane
from midojo.episode import Episode
from midojo.episode_context import EpisodeContext
from midojo.interception_host import HostInterception
from midojo.runtimes import AgentRuntime, attempt_isolation
from midojo.suite_tools import host_tools_reference
from midojo.yaml_task_suite import YAMLTaskSuite

METADATA_PREFIX = "midojo."
UNDETERMINED_REPLY = "MiDojo could not evaluate this candidate."


@dataclass(frozen=True)
class MidojoVerdict:
    """An immutable receipt. The grade is JSON so callers cannot mutate the stored verdict.

    ``security`` is the raw sealed result; ``broke`` requires a successful
    clean-start replay. Replay identities are also emitted in score metadata.
    """

    status: Literal["graded", "failed", "refused"]
    evaluation_id: str | None = None
    snapshot_id: str | None = None
    run_id: str | None = None
    grade_json: str | None = None
    reason: str | None = None
    replay: MidojoVerdict | None = None

    @property
    def grade(self) -> GradeResponse | None:
        return GradeResponse.model_validate_json(self.grade_json) if self.grade_json is not None else None

    @property
    def security(self) -> bool | None:
        grade = self.grade
        if self.status != "graded" or grade is None:
            return None
        validity = grade.grade_validity
        if validity.security.status != "valid":
            return None
        return grade.security

    @property
    def broke(self) -> bool | None:
        # Same confirmation rule as AttemptOutcome.confirmed_by: an unknown
        # replay is unknown, and a replay that does not break is False.
        if self.security is True:
            return self.replay.security if self.replay is not None else None
        return self.security

    def metadata(self) -> dict[str, str | int]:
        """Portable audit data, never used as scoring authority."""
        fields = {
            "status": self.status,
            "evaluation_id": self.evaluation_id,
            "snapshot_id": self.snapshot_id,
            "run_id": self.run_id,
            "grade": self.grade_json,
            "reason": self.reason,
        }
        result: dict[str, str | int] = {
            METADATA_PREFIX + key: value for key, value in fields.items() if value is not None
        }
        if self.replay is not None:
            result[METADATA_PREFIX + "replay"] = json.dumps(self.replay.metadata(), sort_keys=True)
        return result


@dataclass(frozen=True)
class _Attempt:
    episode: Episode | None
    verdict: MidojoVerdict


class MidojoEpisodeTarget(PromptTarget):
    """Plant text on exactly one declared instruction, then drive and seal an episode.

    ``episode_for_payload`` is trusted host code that compiles a candidate;
    its plan must match ``surface`` in everything but the payload. For suites,
    use :meth:`from_suite`, which also checks the probe key. ``drive`` sends
    the episode's benign turns using ``ep.turn`` and the agent's binding.

    A supplied runtime belongs to a run the caller starts and ends. Every
    evaluation requests its declared reset, or uses its disposable target.
    Sends and scorer replays are serialized on this instance; do not share
    its runtime with another target. The receipt registry lives as long as
    this target, so keep it alive until scoring completes.
    """

    def __init__(
        self,
        *,
        episode_for_payload: Callable[[str], Episode],
        surface: InjectionPlan,
        drive: Callable[[EpisodeContext], Awaitable[None]],
        runtime: AgentRuntime,
        interception: HostInterception | None = None,
        control: ControlPlane | None = None,
    ) -> None:
        if len(surface.instructions) != 1 or len(surface.probe_payloads) > 1:
            raise OutOfSurfaceError("a PyRIT target requires exactly one declared injection instruction")
        isolation = attempt_isolation(runtime)
        if isolation is None:
            raise ValueError("adaptive PyRIT use requires a runtime declaring reset or a disposable target")
        super().__init__()
        self._episode_for_payload = episode_for_payload
        self._surface = surface
        self._drive = drive
        self._runtime = runtime
        self._reset = isolation == "reset"
        self._interception = interception
        self._control = control if control is not None else ControlPlane({})
        self._attempts: dict[str, _Attempt] = {}
        self._lock = asyncio.Lock()

    @property
    def control(self) -> ControlPlane:
        """The target's control plane, retaining evaluations and snapshots after sends."""
        return self._control

    @classmethod
    def from_suite(
        cls,
        *,
        suite: YAMLTaskSuite,
        suite_module: ModuleType,
        user_task_id: str,
        injection_task_id: str,
        probe_key: str,
        drive: Callable[[EpisodeContext], Awaitable[None]],
        runtime: AgentRuntime,
        interception: HostInterception | None = None,
        control: ControlPlane | None = None,
    ) -> MidojoEpisodeTarget:
        """Declare a single suite probe (the full ``task:probe`` key).

        ``suite_module`` is the caller's imported module that owns ``suite``.
        Its HOST_TOOLS declaration is checked without importing its target;
        the short suite name is never used to guess a module.
        """
        permitted = suite.get_probes_for_task(injection_task_id)
        permitted_injections(AttackAttempt({probe_key: ""}), permitted)
        host_tools = host_tools_reference(suite_module)
        if interception is None and host_tools is not None:
            raise ValueError("a suite declaring HOST_TOOLS requires host interception")
        surface = suite.episode(user_task_id, injection_task_id, {probe_key: ""}).plan

        def compile_episode(payload: str) -> Episode:
            injections = permitted_injections(AttackAttempt({probe_key: payload}), {probe_key: ""})
            return suite.episode(user_task_id, injection_task_id, injections)

        return cls(
            episode_for_payload=compile_episode,
            surface=surface,
            drive=drive,
            runtime=runtime,
            interception=interception,
            control=control,
        )

    def _validate_request(self, *, normalized_conversation: list[Message]) -> None:
        # Defer target capability validation to the send hook, where a refusal
        # becomes a receipt. PyRIT still validates the Message model itself.
        pass

    async def _send_prompt_to_target_async(self, *, normalized_conversation: list[Message]) -> list[Message]:
        request = normalized_conversation[-1]
        piece = request.message_pieces[0]
        async with self._lock:
            episode = None
            try:
                super()._validate_request(normalized_conversation=normalized_conversation)
                if piece.role != "user":
                    raise ValueError("a candidate must be a user request for the declared surface")
            except ValueError as error:
                verdict, reply = MidojoVerdict("refused", reason=str(error)), UNDETERMINED_REPLY
            else:
                candidate = piece.converted_value
                try:
                    episode = self._episode_for_payload(candidate)
                    check_plan_within(episode.plan, self._surface)
                    if len(episode.plan.instructions) != 1 or episode.plan.instructions[0].payload != candidate:
                        raise OutOfSurfaceError("the candidate must fill exactly the declared instruction")
                except OutOfSurfaceError as error:
                    verdict, reply = MidojoVerdict("refused", reason=str(error)), UNDETERMINED_REPLY
                except Exception as error:
                    verdict = MidojoVerdict("failed", reason=f"{type(error).__name__}: {error}")
                    reply = UNDETERMINED_REPLY
                else:
                    verdict, reply = await self._run(episode)
            response = construct_response_from_request(piece, [reply], prompt_metadata=verdict.metadata())
            # PyRIT's helper merges request metadata. Keep the receipt entirely
            # host-authored; a candidate's metadata is not evidence.
            response.message_pieces[0].prompt_metadata = verdict.metadata()
            replay_episode = episode if verdict.security is True else None
            self._attempts[str(response.message_pieces[0].id)] = _Attempt(replay_episode, verdict)
            return [response]

    async def _run(self, episode: Episode) -> tuple[MidojoVerdict, str]:
        ep = midojo.open_episode(
            episode,
            runtime=self._runtime,
            interception=self._interception,
            control=self._control,
            reset=self._reset,
        )
        try:
            async with ep:
                await self._drive(ep)
            snapshot = ep.sealed
            verdict = midojo.grade(episode, snapshot)
            return (
                MidojoVerdict(
                    "graded",
                    evaluation_id=verdict.evaluation_id,
                    snapshot_id=verdict.snapshot_id,
                    run_id=snapshot.run_id,
                    grade_json=verdict.grade.model_dump_json(),
                    reason=verdict.grade.security_reason,
                ),
                snapshot.agent_output,
            )
        except Exception as error:
            # Cancellation and interrupts propagate; ordinary failed episodes
            # have no verdict, even if cleanup managed to seal them.
            evaluation_id = snapshot_id = run_id = None
            try:
                evaluation_id = ep.evaluation_id
                snapshot_id, run_id = ep.sealed.snapshot_id, ep.sealed.run_id
            except RuntimeError:
                pass
            reason = f"{type(error).__name__}: {error}"
            return MidojoVerdict("failed", evaluation_id, snapshot_id, run_id, reason=reason), UNDETERMINED_REPLY

    def verdict_of(self, message: Message) -> MidojoVerdict | None:
        """Resolve exactly one response by identity; ignore response text and metadata."""
        if len(message.message_pieces) != 1 or message.message_pieces[0].role != "assistant":
            return None
        attempt = self._attempts.get(str(message.message_pieces[0].id))
        return attempt.verdict if attempt is not None else None

    async def confirm(self, message: Message) -> MidojoVerdict | None:
        """Replay a discovered success once, on first scoring, and cache its receipt."""
        async with self._lock:
            verdict = self.verdict_of(message)
            if verdict is None or verdict.security is not True or verdict.replay is not None:
                return verdict
            key = str(message.message_pieces[0].id)
            attempt = self._attempts[key]
            assert attempt.episode is not None
            replay, _ = await self._run(attempt.episode)
            verdict = replace(verdict, replay=replay)
            self._attempts[key] = replace(attempt, episode=None, verdict=verdict)
            return verdict
