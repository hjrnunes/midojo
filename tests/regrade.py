"""Re-grade recorded evaluations with the current code, offline.

A recorded evaluation is what a control plane kept for one run of a suite
pair: its turns, its function call records, the agent's last reply and the
verdict it gave, with the security check the pair was graded by. Re-grading
replays it through today's control plane: a new in-process evaluation of an
episode with the same turns and check, every call record sent again through
the evaluation's session (as the in-agent layer that recorded it did), the
turns and the reply, then the seal and the grade. Nothing the recording
didn't hold is invented: an unfinished call is only started, a blocked one
is recorded whole, and the evidence status is whatever the current lifecycle
derives from the replayed records.

The verdict that comes back is compared with the recorded one; each change
needs a cause (see ``tests/test_regrade_recorded.py``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from midojo.control_plane.config import AppConfig
from midojo.control_plane.models import CreateFunctionCallRecord, CreateFunctionCallStart, GradeResponse
from midojo.control_plane.service import BySession, ControlPlane
from midojo.episode import Episode
from midojo.types import InterceptionMode


@dataclass(frozen=True)
class RecordedEvaluation:
    """One recorded evaluation: what is replayed, and the verdict recorded for it."""

    id: str
    security: Mapping[str, Any]
    turns: list[dict[str, Any]]
    function_calls: list[dict[str, Any]]
    agent_output: str
    recorded_security: bool | None
    recorded_security_reason: str | None

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> RecordedEvaluation:
        return cls(
            id=data["id"],
            security=data["security"],
            turns=list(data["turns"]),
            function_calls=list(data["function_calls"]),
            agent_output=data["agent_output"],
            recorded_security=data["recorded"]["security"],
            recorded_security_reason=data["recorded"]["security_reason"],
        )


def load(path: Path) -> list[RecordedEvaluation]:
    return [RecordedEvaluation.from_data(item) for item in json.loads(path.read_text())["evaluations"]]


async def regrade(recorded: RecordedEvaluation, *, interception: InterceptionMode = "agent") -> GradeResponse:
    """Replay ``recorded`` through a new in-process control plane, seal it and grade it with the current code.

    The recordings were made by an interception layer in the agent, hence
    ``agent``; ``host`` replays the same records as a host interceptor's.
    """
    messages = [turn["message"] for turn in recorded.turns]
    episode = Episode(turns=messages, security=recorded.security)
    control = ControlPlane({}, config=AppConfig(closure_wait_seconds=0))
    opened = control.open_evaluation(episode, interception=interception)
    session = BySession(opened.session_token)

    def replay_calls(turn: int | None) -> None:
        for call in recorded.function_calls:
            if call.get("turn") != turn:
                continue
            body = {"call_id": call["call_id"], "function": call["function"], "args": call["args"]}
            if call.get("blocked"):
                # A blocked call never ran: it is recorded whole, never started.
                finish = {"result": call.get("result") or "", "error": call.get("error"), "blocked": True}
                control.finish_call(session, CreateFunctionCallRecord(**body, **finish))
                continue
            control.start_call(session, CreateFunctionCallStart(**body))
            if call.get("complete", True):
                finish = {"result": call.get("result") or "", "error": call.get("error")}
                control.finish_call(session, CreateFunctionCallRecord(**body, **finish))

    # Calls before the first turn carry no turn index.
    replay_calls(None)
    for index, turn in enumerate(recorded.turns):
        control.start_turn(None, opened.id, turn["message"])
        replay_calls(index)
        if turn.get("reply") is not None:
            control.record_turn_reply(None, opened.id, index, turn["reply"])
    await control.complete(None, opened.id, recorded.agent_output)
    return control.grade(None, opened.id)
