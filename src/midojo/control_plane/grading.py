"""Grading as a pure function of a sealed snapshot."""

from __future__ import annotations

from midojo.verifiers import ConversationTurn
from midojo.yaml_task_suite import YAMLTaskSuite

from .models import GradeResponse
from .state import Snapshot
from .store import detached


def grade_snapshot(suite: YAMLTaskSuite, snapshot: Snapshot) -> GradeResponse:
    """Grade ``snapshot`` with ``suite``'s checks. Reads nothing else, and changes nothing in ``snapshot``.

    The evidence status is the suite's, with every key the snapshot marks (a
    late fact recorded before the seal) taken from the snapshot.
    """
    snapshot = detached(snapshot)
    result = suite.grade(
        user_task_id=snapshot.user_task_id,
        injection_task_id=snapshot.injection_task_id,
        agent_output=snapshot.agent_output,
        pre_environment=snapshot.pre_environment,
        post_environment=snapshot.environment,
        function_calls=list(snapshot.function_calls),
        observations=dict(snapshot.observations),
        turns=[ConversationTurn(message=turn.message, reply=turn.reply) for turn in snapshot.turns],
    )
    graded = GradeResponse.model_validate(result)
    return graded.model_copy(update={"evidence_status": {**graded.evidence_status, **snapshot.evidence_status}})
