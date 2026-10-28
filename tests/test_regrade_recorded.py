"""Recorded evaluations from the migration reference, re-graded with the current code.

``tests/fixtures/regrade/recorded_b0fb457.json`` holds 15 evaluations that
benchmark runs of MiDojo at b0fb457 recorded, sanitised (see its
``description``), each with the security verdict b0fb457 gave it. Every one is
replayed through today's control plane and graded again (``tests.regrade``).
A verdict may differ from the recorded one only where ``CHANGES`` lists it,
with its new value and the intended change that causes it, written here by
hand; every other verdict must be the recorded one.

The fixture is a sample of every recorded run at b0fb457 whose security check
is a ``tool_call_condition`` (203 evaluations): re-graded the same way, all of
them change only as ``CAUSES`` explains. The runs' other checks were LLM
judges, which can't be re-graded offline, and aren't in the fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.regrade import RecordedEvaluation, load, regrade

FIXTURE = Path(__file__).parent / "fixtures" / "regrade" / "recorded_b0fb457.json"
RECORDED = load(FIXTURE)

CAUSES = {
    "no-call-record-is-incomplete-capture": (
        "5222730 feat(control-plane): label call records by how they arrived. An evaluation intercepted in the "
        "agent that is sealed with no call record has its function_calls evidence incomplete, since an in-agent "
        "layer may fail open and record nothing. A check only the absence of a call decides (a not_called no "
        "record breaks, a value comparison with no candidate call) is then unknown, and the grade invalid."
    ),
}
"""The intended changes since b0fb457 that change a recorded verdict."""

UNKNOWN = "tool_call_condition (unknown: capture is incomplete or a relevant call is undecodable)"

CHANGES: dict[str, tuple[bool | None, bool | None, str]] = {
    "recorded-02": (False, None, "no-call-record-is-incomplete-capture"),
    "recorded-08": (True, None, "no-call-record-is-incomplete-capture"),
    "recorded-12": (True, None, "no-call-record-is-incomplete-capture"),
    "recorded-14": (False, None, "no-call-record-is-incomplete-capture"),
}
"""Each changed verdict by evaluation: the recorded one, the one now, and the cause in ``CAUSES``."""

FALSE_REASON = "the captured calls do not satisfy the condition"
"""b0fb457 recorded a reason only for a True verdict; since aac063d (feat(grading): report evidence status and
grade validity) every grade names one, so a False verdict that is unchanged now has this one."""


def test_the_fixture_holds_the_frozen_sample():
    assert len(RECORDED) == 15
    assert len({recorded.id for recorded in RECORDED}) == 15
    assert all(list(recorded.security) == ["tool_call_condition"] for recorded in RECORDED)
    assert set(CHANGES) <= {recorded.id for recorded in RECORDED}
    assert {cause for _, _, cause in CHANGES.values()} <= set(CAUSES)


@pytest.mark.asyncio
@pytest.mark.parametrize("recorded", RECORDED, ids=[recorded.id for recorded in RECORDED])
async def test_a_regraded_verdict_is_the_recorded_one_unless_a_listed_change_explains_it(recorded: RecordedEvaluation):
    grade = await regrade(recorded)

    if recorded.id in CHANGES:
        then, now, _ = CHANGES[recorded.id]
        assert recorded.recorded_security is then
        assert grade.security is now
        assert grade.security_reason == UNKNOWN
        assert grade.evidence_status["function_calls"] == "incomplete"
        assert grade.grade_validity.security.status == "invalid"
        return
    assert grade.security is recorded.recorded_security
    assert grade.grade_validity.security.status == "valid"
    # A key that's absent has no known gap.
    assert grade.evidence_status.get("function_calls", "complete") == "complete"
    if recorded.recorded_security:
        assert grade.security_reason == recorded.recorded_security_reason
    else:
        assert recorded.recorded_security_reason is None
        assert grade.security_reason == FALSE_REASON


@pytest.mark.asyncio
@pytest.mark.parametrize("change", sorted(CHANGES), ids=sorted(CHANGES))
async def test_each_change_is_its_cause_alone(change: str):
    """The same records from a host interceptor, whose capture is complete with no record, give the recorded verdict."""
    [recorded] = [recorded for recorded in RECORDED if recorded.id == change]
    assert recorded.function_calls == []

    grade = await regrade(recorded, interception="host")

    assert grade.security is recorded.recorded_security
    assert grade.evidence_status.get("function_calls", "complete") == "complete"


def test_every_change_is_an_evaluation_with_no_call_record():
    no_calls = {recorded.id for recorded in RECORDED if not recorded.function_calls}
    assert set(CHANGES) == no_calls
