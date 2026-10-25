"""Grading as a pure function of a sealed snapshot.

:func:`grade` needs no control plane and no episode context: only the episode
whose checks grade, and the sealed snapshot. The control plane grades with the
same function (:meth:`midojo.control_plane.service.ControlPlane.grade`), so a
snapshot gets the same grade wherever it's graded; only the control plane's
response adds late facts to its evidence status (see :func:`grade`).
"""

from __future__ import annotations

from dataclasses import dataclass

from midojo.episode import Episode
from midojo.verifiers import ConversationTurn

from .models import GradeResponse
from .state import Snapshot
from .store import detached


def grade_snapshot(episode: Episode, snapshot: Snapshot) -> GradeResponse:
    """Grade ``snapshot`` with ``episode``'s checks. Reads nothing else, and changes nothing in ``snapshot``.

    The checks see the snapshot's evidence status, so a check that depends on a
    call's absence is unknown when ``function_calls`` is incomplete. The
    evidence status is the episode's, with every key the snapshot marks (a late
    fact or a capture failure recorded before the seal, or a call capture the
    seal found incomplete) taken from the snapshot.
    """
    snapshot = detached(snapshot)
    result = episode.grade(
        agent_output=snapshot.agent_output,
        pre_environment=snapshot.pre_environment,
        post_environment=snapshot.environment,
        function_calls=list(snapshot.function_calls),
        observations=dict(snapshot.observations),
        turns=[ConversationTurn(message=turn.message, reply=turn.reply) for turn in snapshot.turns],
        evidence=dict(snapshot.evidence_status),
    )
    graded = GradeResponse.model_validate(result)
    return graded.model_copy(update={"evidence_status": {**graded.evidence_status, **snapshot.evidence_status}})


@dataclass(frozen=True)
class Verdict:
    """The grade of one sealed snapshot, naming the evaluation and the snapshot it graded."""

    evaluation_id: str
    snapshot_id: str
    grade: GradeResponse


def grade(episode: Episode, snapshot: Snapshot) -> Verdict:
    """Grade ``snapshot`` with ``episode``'s checks: :func:`grade_snapshot`, naming what was graded.

    The verdict's ``evidence_status`` is the one fixed in the snapshot. The
    control plane grades the same snapshot with the same function, but its
    response (``ControlPlane.grade``, ``POST .../grade``) shows the
    evaluation's current evidence status, which also marks ``incomplete``
    every key a late fact carries, including facts that arrived after the
    seal. So with no late fact the two are equal; with one, the grades
    (``utility``, ``security``, their reasons and validity) are still equal,
    and ``evidence_status`` differs only in the late facts' keys, which the
    control plane's response marks ``incomplete``.
    """
    return Verdict(snapshot.evaluation_id, snapshot.snapshot_id, grade_snapshot(episode, snapshot))
