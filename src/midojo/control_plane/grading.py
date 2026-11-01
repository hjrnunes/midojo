"""Grading as a pure function of a sealed snapshot.

:func:`grade` needs no control plane and no episode context: only the episode
whose checks grade, and the sealed snapshot. The control plane grades with the
same function (:meth:`midojo.control_plane.service.ControlPlane.grade`), so a
snapshot gets the same grade wherever it's graded; only the control plane's
response adds late facts to its evidence status (see :func:`grade`).

A ``reply_judge`` check is the exception: it asks a judge service
(:mod:`midojo.verifiers.judge`), so its grade is the service's answer as well
as the snapshot's, and grading again may get another answer. The control
plane grades once and keeps that grade; the grade's ``judgements`` record
what was asked and answered. Passing those records back as ``judgements=``
replays the grade offline: each request is answered by request id, no
service is asked, and a request the records don't hold is unknown, never
False (see :func:`grade`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from midojo.episode import Episode
from midojo.verifiers import ConversationTurn
from midojo.verifiers.judge import JudgeRecord

from .models import GradeResponse
from .state import Snapshot

VERDICT_DOCUMENT_V1 = "midojo-verdict-v1"
"""The verdict document version :meth:`Verdict.to_document` writes; a versionless document reads as v0."""


class VerdictDocumentError(ValueError):
    """A verdict document MiDojo refuses: a missing member, a wrong type, or a grade that does not validate."""


class UnsupportedVerdictVersion(VerdictDocumentError):  # noqa: N818 - named by the document contract, not a lint choice.
    """A verdict document whose ``schema_version`` MiDojo does not read. ``version`` names it."""

    def __init__(self, version: object) -> None:
        self.version = version
        self.supported = (VERDICT_DOCUMENT_V1,)
        super().__init__(
            f"verdict document version {version!r} is not supported; supported: {VERDICT_DOCUMENT_V1}, "
            "or no schema_version for the versionless v0 form"
        )


def grade_snapshot(
    episode: Episode, snapshot: Snapshot, *, judgements: Sequence[JudgeRecord | Mapping[str, Any]] | None = None
) -> GradeResponse:
    """Grade ``snapshot`` with ``episode``'s checks. Reads nothing else, and changes nothing in ``snapshot``.

    The checks see the snapshot's evidence status, so a check that depends on a
    call's absence is unknown when ``function_calls`` is incomplete. The
    evidence status is the episode's, with every key the snapshot marks (a late
    fact or a capture failure recorded before the seal, or a call capture the
    seal found incomplete) taken from the snapshot.

    ``judgements`` replays a judge-graded snapshot instead of asking a judge
    service: it takes records an earlier grade kept (its ``judgements``) and
    answers each judge request by request id (:class:`midojo.verifiers.judge.RecordedJudge`).

    It grades only the content ``snapshot.snapshot_id`` names, read once
    (:meth:`Snapshot.verified`): a snapshot changed after it was made (its
    private copies, say) raises ValueError, and is never graded.
    """
    snapshot = snapshot.verified()
    result = episode.grade(
        agent_output=snapshot.agent_output,
        pre_environment=snapshot.pre_environment,
        post_environment=snapshot.environment,
        function_calls=list(snapshot.function_calls),
        observations=dict(snapshot.observations),
        turns=[ConversationTurn(message=turn.message, reply=turn.reply) for turn in snapshot.turns],
        evidence=dict(snapshot.evidence_status),
        judgements=judgements,
    )
    graded = GradeResponse.model_validate(result)
    return graded.model_copy(update={"evidence_status": {**graded.evidence_status, **snapshot.evidence_status}})


@dataclass(frozen=True)
class Verdict:
    """The grade of one sealed snapshot, naming the evaluation and the snapshot it graded."""

    evaluation_id: str
    snapshot_id: str
    grade: GradeResponse

    def to_document(self) -> dict[str, Any]:
        """This verdict as a ``midojo-verdict-v1`` document: the version, the two IDs, and the grade's fields."""
        return {
            "schema_version": VERDICT_DOCUMENT_V1,
            "evaluation_id": self.evaluation_id,
            "snapshot_id": self.snapshot_id,
            "grade": self.grade.model_dump(mode="json"),
        }

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> Verdict:
        """A verdict from a v1 document, or from the versionless v0 form the cm.58 artifacts carry.

        An unknown version raises :class:`UnsupportedVerdictVersion`, naming
        it; anything else MiDojo cannot read raises :class:`VerdictDocumentError`.
        """
        if not isinstance(document, Mapping):
            raise VerdictDocumentError(f"verdict document must be a JSON object, not {type(document).__name__}")
        if "schema_version" in document:
            version = document["schema_version"]
            if type(version) is not str:
                raise VerdictDocumentError(
                    f"verdict document: schema_version must be a string, not {type(version).__name__}"
                )
            if version != VERDICT_DOCUMENT_V1:
                raise UnsupportedVerdictVersion(version)
        allowed = {"schema_version", "evaluation_id", "snapshot_id", "grade"}
        if unknown := [key for key in document if key not in allowed]:
            raise VerdictDocumentError(f"verdict document: unknown key {unknown[0]!r}")
        for name in ("evaluation_id", "snapshot_id", "grade"):
            if name not in document:
                raise VerdictDocumentError(f"verdict document: {name} is required")
        values: dict[str, str] = {}
        for name in ("evaluation_id", "snapshot_id"):
            value = document[name]
            if type(value) is not str:
                raise VerdictDocumentError(f"verdict document: {name} must be a string, not {type(value).__name__}")
            values[name] = value
        if not isinstance(document["grade"], Mapping):
            raise VerdictDocumentError("verdict document: grade must be a JSON object")
        try:
            grade = GradeResponse.model_validate(document["grade"])
        except ValidationError as exc:
            raise VerdictDocumentError(f"verdict document: grade: {exc}") from None
        return cls(values["evaluation_id"], values["snapshot_id"], grade)


def grade(
    episode: Episode, snapshot: Snapshot, *, judgements: Sequence[JudgeRecord | Mapping[str, Any]] | None = None
) -> Verdict:
    """Grade ``snapshot`` with ``episode``'s checks: :func:`grade_snapshot`, naming what was graded.

    ``judgements=`` replays a judge-graded snapshot offline from records an
    earlier grade kept (``grade.judgements``, as :class:`JudgeRecord` objects
    or their JSON): each judge request is answered by request id, no judge
    service is asked, and a recorded failure replays as unknown. A request the
    records don't hold makes its check unknown, with the reason
    ``judge-replay-unrecorded:<id8>``, never False: the grade is then invalid
    for that check, as a judge that gave no answer leaves it. Without
    ``judgements``, grading is unchanged and asks the service.

    The verdict's ``evidence_status`` is the one fixed in the snapshot. The
    control plane grades the same snapshot with the same function, but its
    response (``ControlPlane.grade``, ``POST .../grade``) shows the
    evaluation's current evidence status, which also marks ``incomplete``
    every key a late fact carries. A late fact recorded before the seal is
    already in the snapshot, so it changes neither: only late facts that
    arrive after the seal (even after the grade) make the two differ. With
    none, they are equal; with one, the grades (``utility``, ``security``,
    their reasons and validity) are still equal, and ``evidence_status``
    differs only in those facts' keys, which the control plane's response
    marks ``incomplete``.
    """
    return Verdict(
        snapshot.evaluation_id, snapshot.snapshot_id, grade_snapshot(episode, snapshot, judgements=judgements)
    )
