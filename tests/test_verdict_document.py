"""Verdict documents: the ``midojo-verdict-v1`` shape, the only one MiDojo reads (architecture 3.4).

A document with no ``schema_version`` is refused, never read as an older form.
"""

from __future__ import annotations

from typing import Any

import pytest

from midojo.control_plane.grading import (
    VERDICT_DOCUMENT_V1,
    UnsupportedVerdictVersion,
    Verdict,
    VerdictDocumentError,
)
from midojo.control_plane.models import GradeResponse

GRADE: dict[str, Any] = {
    "attempted": False,
    "blocked": False,
    "effect_observed": False,
    "evidence_status": {"openshell.ocsf": "complete", "openshell.workdir": "complete"},
    "grade_validity": {
        "security": {"reasons": [], "status": "valid"},
        "utility": {"reasons": [], "status": "not_applicable"},
    },
    "injection_delivered": True,
    "security": False,
    "security_reason": "the captured calls do not satisfy the condition",
    "utility": False,
    "utility_reason": None,
}


def a_verdict() -> Verdict:
    """A verdict of a delivered injection that did not break the agent."""
    return Verdict(
        "47ee75fed7",
        "sha256:490adb33e785a3862031342b3fee004f08634471bfbbb1fdded08839ce5272d6",
        GradeResponse.model_validate(GRADE),
    )


def test_a_verdict_round_trips_through_its_v1_document() -> None:
    verdict = a_verdict()

    document = verdict.to_document()

    assert document["schema_version"] == "midojo-verdict-v1"
    assert set(document) == {"schema_version", "evaluation_id", "snapshot_id", "grade"}
    assert document["evaluation_id"] == verdict.evaluation_id
    assert document["snapshot_id"] == verdict.snapshot_id
    assert document["grade"] == verdict.grade.model_dump(mode="json")
    assert Verdict.from_document(document) == verdict


def test_a_v1_document_reads_the_grade_it_holds() -> None:
    verdict = Verdict.from_document(a_verdict().to_document())

    assert verdict.evaluation_id == "47ee75fed7"
    assert verdict.snapshot_id == "sha256:490adb33e785a3862031342b3fee004f08634471bfbbb1fdded08839ce5272d6"
    assert verdict.grade.model_dump(mode="json") == GRADE
    assert verdict.grade.security is False
    assert verdict.grade.security_reason == "the captured calls do not satisfy the condition"
    assert verdict.grade.utility is False
    assert verdict.grade.utility_reason is None
    assert (verdict.grade.attempted, verdict.grade.blocked, verdict.grade.effect_observed) == (False, False, False)
    assert verdict.grade.injection_delivered is True
    assert verdict.grade.evidence_status == {"openshell.ocsf": "complete", "openshell.workdir": "complete"}
    assert verdict.grade.grade_validity.security.status == "valid"


def test_a_versioned_document_pins_the_version() -> None:
    document = a_verdict().to_document()

    assert Verdict.from_document(document).to_document() == document
    assert VERDICT_DOCUMENT_V1 == "midojo-verdict-v1"


def test_a_document_without_a_version_is_refused() -> None:
    document = a_verdict().to_document()
    del document["schema_version"]

    with pytest.raises(VerdictDocumentError, match="schema_version is required") as raised:
        Verdict.from_document(document)

    assert not isinstance(raised.value, UnsupportedVerdictVersion)


@pytest.mark.parametrize("version", ["midojo-verdict-v9", "midojo-verdict-v1.1", ""])
def test_an_unknown_verdict_version_is_refused_naming_it(version: str) -> None:
    document = {**a_verdict().to_document(), "schema_version": version}

    with pytest.raises(UnsupportedVerdictVersion) as raised:
        Verdict.from_document(document)

    assert raised.value.version == version
    assert raised.value.supported == (VERDICT_DOCUMENT_V1,)
    assert repr(version) in str(raised.value)
    assert VERDICT_DOCUMENT_V1 in str(raised.value)


def test_a_non_string_version_is_refused() -> None:
    document = {**a_verdict().to_document(), "schema_version": 1}

    with pytest.raises(VerdictDocumentError, match="schema_version") as raised:
        Verdict.from_document(document)

    assert not isinstance(raised.value, UnsupportedVerdictVersion)


@pytest.mark.parametrize("member", ["evaluation_id", "snapshot_id", "grade"])
def test_a_missing_member_is_refused_naming_it(member: str) -> None:
    document = a_verdict().to_document()
    del document[member]

    with pytest.raises(VerdictDocumentError, match=member):
        Verdict.from_document(document)


def test_an_unknown_key_is_refused_naming_it() -> None:
    document = {**a_verdict().to_document(), "extensions": {}}

    with pytest.raises(VerdictDocumentError, match="extensions"):
        Verdict.from_document(document)


def test_a_grade_that_does_not_validate_is_refused() -> None:
    document = a_verdict().to_document()
    document["grade"] = {"security": "yes"}

    with pytest.raises(VerdictDocumentError, match="grade"):
        Verdict.from_document(document)


def test_the_grade_fields_are_unchanged() -> None:
    # The document's grade is the GradeResponse's own fields: the document version adds no field to it.
    assert set(GradeResponse.model_fields) == {
        "utility",
        "utility_reason",
        "security",
        "security_reason",
        "grade_validity",
        "evidence_status",
        "attempted",
        "blocked",
        "effect_observed",
        "injection_delivered",
        "judgements",
    }
    # A grade no judge took part in dumps without its empty judgements list, as it always did.
    document = a_verdict().to_document()
    assert set(document["grade"]) == set(GradeResponse.model_fields) - {"judgements"}
