"""The evaluation lifecycle rules, as pure transitions on evaluation values."""

from __future__ import annotations

import copy
import dataclasses
import uuid
from datetime import datetime

import pytest

from midojo.control_plane import lifecycle
from midojo.control_plane.lifecycle import (
    HOST,
    Accepted,
    AgentSession,
    BeginClosure,
    ChangeEnvironment,
    CloseSession,
    EndClosureWait,
    FinishCall,
    Late,
    OpenSession,
    Read,
    RecordGrade,
    RecordObservations,
    Rejected,
    Rejection,
    ReplaceEnvironment,
    ReplyTurn,
    RotateSession,
    Seal,
    StartCall,
    StartTurn,
    transition,
)
from midojo.control_plane.models import CreateFunctionCallRecord, CreateFunctionCallStart, GradeResponse
from midojo.control_plane.state import Evaluation, Session, Stage
from midojo.runtimes.openshell import OpenShellObservations
from midojo.types import CheckValidity, Environment, GradeValidity

AT = "2026-01-01T00:00:00+00:00"
# AT as epoch seconds: closure deadlines are compared with the time an event arrived.
T = datetime.fromisoformat(AT).timestamp()
TOKEN = "hash-of-token"
AGENT = AgentSession(TOKEN, now=100.0)


class _Env(Environment):
    counter: int = 0


def _evaluation(**changes) -> Evaluation:
    evaluation = Evaluation(
        id="e",
        run_id="r",
        user_task_id="ut",
        injection_task_id=None,
        pre_environment=_Env(),
        environment=_Env(),
        session=Session(TOKEN, expires_at=200.0),
    )
    return dataclasses.replace(evaluation, **changes)


def _start(call_id: str = "c1", **args) -> StartCall:
    return StartCall(CreateFunctionCallStart(call_id=call_id, function="a", args=args), AT)


def _finish(call_id: str | None = "c1", result: str = "r", **args) -> FinishCall:
    return FinishCall(CreateFunctionCallRecord(call_id=call_id, function="a", args=args, result=result), AT)


def _accept(evaluation: Evaluation, event, authority: lifecycle.Authority = HOST) -> Accepted:
    outcome = transition(evaluation, authority, event)
    assert isinstance(outcome, Accepted), outcome
    return outcome


def _apply(evaluation: Evaluation, *events, authority: lifecycle.Authority = HOST) -> Evaluation:
    for event in events:
        evaluation = _accept(evaluation, event, authority).evaluation
    return evaluation


def _closing(evaluation: Evaluation, deadline: float = T + 50) -> Evaluation:
    return _apply(evaluation, BeginClosure(AT, deadline))


def _collecting(evaluation: Evaluation) -> Evaluation:
    return _apply(_closing(evaluation, deadline=0.0), EndClosureWait(now=1.0))


def _sealed(evaluation: Evaluation, output: str = "out") -> Evaluation:
    return _apply(_collecting(evaluation), Seal(output, AT))


def _grade() -> GradeResponse:
    valid = CheckValidity(status="valid")
    return GradeResponse(
        utility=True, security=False, grade_validity=GradeValidity(utility=valid, security=valid), evidence_status={}
    )


ALL_EVENTS = [
    Read(),
    OpenSession("other", 300.0),
    RotateSession("other"),
    CloseSession(),
    _start(),
    _finish(),
    _finish(call_id=None),
    StartTurn("m", AT),
    ReplyTurn(0, "r", AT),
    ReplaceEnvironment(_Env(counter=5), AT),
    ChangeEnvironment({"counter": 6}, AT),
    ChangeEnvironment({"counter": 6}, AT, expected_revision=0),
    RecordObservations("openshell", OpenShellObservations(), AT),
    BeginClosure(AT, T + 50),
    EndClosureWait(now=T + 60),
    Seal("out", AT),
    RecordGrade(_grade()),
]


def _staged() -> list[Evaluation]:
    started = _apply(_evaluation(), StartTurn("m", AT), _start())
    return [started, _closing(started), _collecting(started), _sealed(started)]


# --- purity ---


@pytest.mark.parametrize("authority", [HOST, AGENT, AgentSession("wrong", now=100.0)], ids=["host", "agent", "other"])
@pytest.mark.parametrize("event", ALL_EVENTS, ids=lambda event: type(event).__name__)
def test_a_transition_is_deterministic_and_never_changes_its_inputs(event, authority):
    for evaluation in _staged():
        before, event_before = copy.deepcopy(evaluation), copy.deepcopy(event)
        first = transition(evaluation, authority, event)
        second = transition(evaluation, authority, event)
        assert first == second
        assert evaluation == before
        assert event == event_before


def test_an_accepted_event_changed_afterwards_changes_nothing_returned():
    environment, observed = _Env(counter=1), OpenShellObservations(network_calls_blocked=["a"])
    start = StartCall(CreateFunctionCallStart(call_id="c1", function="a", args={"x": [1]}), AT)
    evaluation = _apply(
        _evaluation(), start, ReplaceEnvironment(environment, AT), RecordObservations("openshell", observed, AT)
    )
    environment.counter = 99
    observed.network_calls_blocked.append("b")
    start.req.args["x"].append(2)
    assert evaluation.environment == _Env(counter=1)
    assert evaluation.observations["openshell"] == OpenShellObservations(network_calls_blocked=["a"])
    assert evaluation.function_calls[0].args == {"x": [1]}


def test_a_source_evaluation_changed_afterwards_changes_nothing_returned():
    environment, pre_environment = _Env(counter=1), _Env(counter=0)
    source = _apply(_evaluation(environment=environment, pre_environment=pre_environment), _start("a", x=[1]))
    returned = _apply(source, _finish("b"))
    environment.counter = 99
    pre_environment.counter = 98
    source.function_calls[0].args["x"].append(2)
    assert returned.environment == _Env(counter=1)
    assert returned.pre_environment == _Env(counter=0)
    assert returned.function_calls[0].args == {"x": [1]}
    # And the other way round: changing what was returned leaves the source alone.
    source_before = copy.deepcopy(source)
    returned.function_calls[0].args["x"].append(3)
    returned.environment.counter = 5  # type: ignore[attr-defined]
    assert source == source_before


def test_the_snapshot_cannot_be_changed_through_the_sealed_evaluation_or_its_source():
    environment = _Env(counter=1)
    observed = OpenShellObservations(network_calls_blocked=["a"])
    collecting = _collecting(
        _apply(
            _evaluation(environment=environment),
            _start("a", x=[1]),
            RecordObservations("openshell", observed, AT),
        )
    )
    sealed = _apply(collecting, Seal("out", AT))
    snapshot = sealed.snapshot
    assert snapshot is not None
    expected = copy.deepcopy(snapshot)
    environment.counter = 99
    observed.network_calls_blocked.append("b")
    for value in [collecting, sealed]:
        value.function_calls[0].args["x"].append(2)
        value.environment.counter = 97  # type: ignore[attr-defined]
        value.observations["openshell"].network_calls_blocked.append("c")  # type: ignore[attr-defined]
    assert snapshot == expected


def test_rejections_are_domain_kinds_not_status_codes():
    rejected = transition(_sealed(_evaluation()), HOST, RecordGrade(_grade()))
    assert isinstance(rejected, Accepted)
    rejected = transition(_evaluation(), HOST, RecordGrade(_grade()))
    assert rejected == Rejected(Rejection.NOT_SEALED, "Evaluation not completed. Call complete first.")
    assert {field.name for field in dataclasses.fields(Rejected)} == {"kind", "detail"}


def test_the_evaluation_value_is_frozen():
    evaluation = _evaluation()
    with pytest.raises(dataclasses.FrozenInstanceError):
        evaluation.stage = Stage.SEALED  # type: ignore[misc]
    assert isinstance(_apply(evaluation, _start()).function_calls, tuple)


# --- sessions ---


@pytest.mark.parametrize(
    "authority",
    [AgentSession("wrong", now=100.0), AgentSession(TOKEN, now=200.0)],
    ids=["wrong token", "expired"],
)
def test_an_agent_event_needs_the_live_session(authority):
    for evaluation in [*_staged(), _evaluation(session=None)]:
        assert transition(evaluation, authority, _start("x")) == Rejected(
            Rejection.INVALID_SESSION, lifecycle.INVALID_SESSION_DETAIL
        )


@pytest.mark.parametrize(
    "event",
    [
        CloseSession(),
        RotateSession("new"),
        StartTurn("m", AT),
        Seal("o", AT),
        RecordGrade(_grade()),
        BeginClosure(AT, 1.0),
    ],
)
def test_a_session_cannot_drive_the_host_lifecycle(event):
    outcome = transition(_evaluation(), AGENT, event)
    assert isinstance(outcome, Rejected) and outcome.kind is Rejection.FORBIDDEN


def test_after_the_seal_a_session_reads_nothing():
    assert transition(_sealed(_evaluation()), AGENT, Read()) == Rejected(
        Rejection.SESSION_CLOSED, lifecycle.INVALID_SESSION_DETAIL
    )


def test_opening_a_session_replaces_the_previous_one_only_while_open():
    outcome = _accept(_evaluation(), OpenSession("new", 300.0))
    assert outcome.evaluation.session == Session("new", 300.0)
    assert outcome.effects == (lifecycle.SessionUnbound(TOKEN), lifecycle.SessionBound("new"))
    assert isinstance(transition(_closing(_evaluation()), HOST, OpenSession("new", 300.0)), Rejected)
    closed = _accept(_sealed(_evaluation()), CloseSession())
    assert (closed.evaluation.session, closed.effects) == (None, (lifecycle.SessionUnbound(TOKEN),))


def test_rotating_a_session_replaces_its_token_hash_and_keeps_its_expiry_only_while_open():
    outcome = _accept(_evaluation(), RotateSession("new"))
    assert outcome.evaluation.session == Session("new", 200.0) and outcome.result == Session("new", 200.0)
    assert outcome.effects == (lifecycle.SessionUnbound(TOKEN), lifecycle.SessionBound("new"))
    for staged in (_closing(_evaluation()), _collecting(_evaluation()), _sealed(_evaluation())):
        assert transition(staged, HOST, RotateSession("new")) == Rejected(
            Rejection.CLOSING, "Cannot issue a session for this evaluation"
        )
    # A revoked session stays revoked: rotating never opens one.
    revoked = _accept(_evaluation(), CloseSession()).evaluation
    refused = transition(revoked, HOST, RotateSession("new"))
    assert isinstance(refused, Rejected) and refused.kind is Rejection.SESSION_CLOSED


# --- the environment: revisions and atomic changes ---


def test_every_environment_change_counts_a_revision():
    evaluation = _apply(_evaluation(), ReplaceEnvironment(_Env(counter=1), AT), ChangeEnvironment({"counter": 2}, AT))
    assert (evaluation.environment_revision, evaluation.environment) == (2, _Env(counter=2))


def test_a_change_returns_the_new_revision_and_environment():
    outcome = _accept(_evaluation(), ChangeEnvironment({"counter": 4}, AT, expected_revision=0), AGENT)
    assert outcome.result == (1, _Env(counter=4))
    assert outcome.evaluation.environment == _Env(counter=4)


def test_a_change_at_a_stale_revision_is_refused_and_changes_nothing():
    evaluation = _apply(_evaluation(), ChangeEnvironment({"counter": 1}, AT))
    outcome = transition(evaluation, AGENT, ChangeEnvironment({"counter": 9}, AT, expected_revision=0))
    assert isinstance(outcome, Rejected) and outcome.kind is Rejection.REVISION_CONFLICT
    # A rejection leaves nothing behind: the evaluation is still at revision 1, with counter 1.
    assert (evaluation.environment_revision, evaluation.environment) == (1, _Env(counter=1))
    after = _accept(evaluation, ChangeEnvironment({}, AT, expected_revision=1), AGENT)
    assert after.result == (2, _Env(counter=1))


def test_a_change_without_a_revision_applies_at_any_revision():
    evaluation = _apply(_evaluation(), ReplaceEnvironment(_Env(counter=1), AT), ChangeEnvironment({"counter": 5}, AT))
    assert evaluation.environment == _Env(counter=5)


@pytest.mark.parametrize("values", [{"unknown": 1}, {"counter": "not a number"}], ids=["unknown", "invalid"])
def test_a_change_that_does_not_fit_the_environment_is_refused(values):
    evaluation = _apply(_evaluation(), ChangeEnvironment({"counter": 1}, AT))
    outcome = transition(evaluation, AGENT, ChangeEnvironment(values, AT))
    assert isinstance(outcome, Rejected) and outcome.kind is Rejection.INVALID_ENVIRONMENT
    # A rejection leaves nothing behind: the next change still applies at the same revision.
    after = _accept(evaluation, ChangeEnvironment({"counter": 2}, AT, expected_revision=1), AGENT)
    assert after.result == (2, _Env(counter=2))


# --- calls: idempotency, conflicts and unfinished calls ---


def test_records_carry_no_environment_snapshots_by_default():
    evaluation = _apply(_evaluation(environment=_Env(counter=3)), _start("a"), _finish("b"), _finish("a"))
    assert [(call.pre_environment, call.post_environment) for call in evaluation.function_calls] == [(None, None)] * 2


def test_the_first_call_chains_from_the_pre_environment():
    evaluation = _evaluation(pre_environment=_Env(counter=7), environment=_Env(counter=9), environment_history=True)
    outcome = _accept(evaluation, _finish())
    record, created = outcome.result
    assert created is True
    assert outcome.evaluation.function_calls == (record,)
    assert record.pre_environment == _Env(counter=7)
    assert record.post_environment == _Env(counter=9)


def test_each_call_chains_from_the_previous_post_environment():
    evaluation = _apply(
        _evaluation(environment_history=True), _finish("a"), ReplaceEnvironment(_Env(counter=1), AT), _finish("b")
    )
    first, second = evaluation.function_calls
    assert second.pre_environment == first.post_environment
    assert second.post_environment == _Env(counter=1)
    assert second.pre_environment is not second.post_environment


def test_a_call_without_call_id_gets_the_assigned_one_and_is_not_deduplicated():
    evaluation = _apply(_evaluation(), _finish(call_id=None), _finish(call_id=None))
    first, second = evaluation.function_calls
    assert str(uuid.UUID(first.call_id)) == first.call_id
    assert first.call_id != second.call_id


def test_a_replay_returns_the_stored_record_and_changes_nothing():
    evaluation = _apply(_evaluation(), _finish(x=1))
    outcome = _accept(evaluation, _finish(x=1))
    assert outcome.evaluation == evaluation
    assert outcome.result == (evaluation.function_calls[0], False)


@pytest.mark.parametrize("change", [{"function": "b"}, {"args": {"x": 2}}, {"result": "other"}, {"blocked": True}])
def test_a_replay_with_another_body_is_a_conflict(change):
    evaluation = _apply(_evaluation(), _finish(x=1))
    req = CreateFunctionCallRecord(call_id="c1", function="a", args={"x": 1}, result="r").model_copy(update=change)
    outcome = transition(evaluation, HOST, FinishCall(req, AT))
    assert isinstance(outcome, Rejected) and outcome.kind is Rejection.CALL_CONFLICT


def test_a_start_is_finished_in_place_keeping_its_order_and_turn():
    evaluation = _apply(
        _evaluation(environment=_Env(counter=3), environment_history=True),
        StartTurn("m", AT),
        _start("a"),
        _finish("b"),
    )
    started = evaluation.function_calls[0]
    assert (started.complete, started.result, started.post_environment, started.turn) == (False, None, None, 0)
    assert started.pre_environment == _Env(counter=3)
    evaluation = _apply(
        evaluation, ReplyTurn(0, "r", AT), StartTurn("n", AT), ReplaceEnvironment(_Env(counter=5), AT), _finish("a")
    )
    finished = evaluation.function_calls[0]
    assert [call.call_id for call in evaluation.function_calls] == ["a", "b"]
    assert (finished.complete, finished.result, finished.turn) == (True, "r", 0)
    assert finished.post_environment == _Env(counter=5)


@pytest.mark.parametrize("change", [{"function": "b"}, {"args": {"x": 2}}])
def test_a_start_and_its_finish_must_name_the_same_call(change):
    evaluation = _apply(_evaluation(), _start(x=1))
    start = CreateFunctionCallStart(call_id="c1", function="a", args={"x": 1}).model_copy(update=change)
    finish = CreateFunctionCallRecord(call_id="c1", function="a", args={"x": 1}, result="r").model_copy(update=change)
    for event in [StartCall(start, AT), FinishCall(finish, AT)]:
        outcome = transition(evaluation, HOST, event)
        assert isinstance(outcome, Rejected) and outcome.kind is Rejection.CALL_CONFLICT


def test_a_record_after_an_unfinished_call_starts_from_the_live_environment():
    evaluation = _apply(
        _evaluation(environment_history=True), _start("a"), ReplaceEnvironment(_Env(counter=4), AT), _finish("b")
    )
    a, b = evaluation.function_calls
    assert (a.complete, a.pre_environment, b.pre_environment) == (False, _Env(counter=0), _Env(counter=4))


# --- turns ---


def test_turns_run_one_at_a_time_and_take_one_reply():
    evaluation = _apply(_evaluation(), StartTurn("m", AT))
    assert transition(evaluation, HOST, StartTurn("n", AT)) == Rejected(
        Rejection.TURN_PENDING, "Turn 0 has no reply yet"
    )
    assert transition(evaluation, HOST, ReplyTurn(1, "r", AT)) == Rejected(
        Rejection.TURN_NOT_STARTED, "Turn not started: 1"
    )
    evaluation = _apply(evaluation, ReplyTurn(0, "r", AT))
    assert transition(evaluation, HOST, ReplyTurn(0, "s", AT)) == Rejected(
        Rejection.TURN_ANSWERED, "Turn 0 already has a reply"
    )


# --- staged closure ---


def test_closure_moves_through_its_stages_in_order():
    evaluation = _evaluation()
    assert [s.stage for s in _staged()] == [Stage.OPEN, Stage.CLOSING, Stage.COLLECTING, Stage.SEALED]
    closing = _closing(evaluation, deadline=T + 50)
    # Beginning closure again keeps the first deadline.
    assert _closing(closing, deadline=T + 999).closure_deadline == T + 50
    assert transition(evaluation, HOST, Seal("o", AT)) == transition(closing, HOST, Seal("o", AT))
    assert isinstance(transition(closing, HOST, Seal("o", AT)), Rejected)


def test_the_closure_wait_ends_when_admitted_calls_finish_or_the_deadline_passes():
    closing = _closing(_apply(_evaluation(), _start("a")), deadline=T + 50)
    pending = transition(closing, HOST, EndClosureWait(now=T + 49))
    assert isinstance(pending, Rejected) and pending.kind is Rejection.CLOSURE_PENDING
    assert _apply(closing, EndClosureWait(now=T + 50)).stage is Stage.COLLECTING
    assert _apply(closing, _finish("a"), EndClosureWait(now=T)).stage is Stage.COLLECTING


def test_closing_admits_finishes_of_admitted_calls_and_no_new_call():
    closing = _closing(_apply(_evaluation(), _start("a")))
    finished = _accept(closing, _finish("a"), AGENT)
    assert finished.evaluation.function_calls[0].complete is True
    for event, name in [(_start("b"), "start_call"), (_finish("b"), "finish_call")]:
        late = transition(closing, AGENT, event)
        assert isinstance(late, Late) and late.rejected.kind is Rejection.CLOSING
        assert late.evaluation.function_calls == closing.function_calls
        assert [fact.event for fact in late.evaluation.late_facts] == [name]
    # The environment stays writable for calls finishing during the wait.
    assert _apply(closing, ReplaceEnvironment(_Env(counter=2), AT), authority=AGENT).environment == _Env(counter=2)
    assert _apply(closing, ChangeEnvironment({"counter": 3}, AT), authority=AGENT).environment == _Env(counter=3)


@pytest.mark.parametrize(
    ("event", "name"),
    [
        (_finish("a"), "finish_call"),
        (ReplaceEnvironment(_Env(counter=2), AT), "replace_environment"),
        (ChangeEnvironment({"counter": 2}, AT), "change_environment"),
    ],
    ids=["finish", "environment", "change"],
)
def test_a_write_after_the_closure_deadline_is_late_and_moves_closure_on(event, name):
    """The stage may still say closing (no one ended the wait), but the event arrived after the deadline."""
    closing = _closing(_apply(_evaluation(), _start("a")), deadline=T - 1)
    for authority in [HOST, AGENT]:
        late = transition(closing, authority, event)
        assert isinstance(late, Late) and late.rejected.kind is Rejection.CLOSING
        assert late.evaluation.stage is Stage.COLLECTING
        assert [fact.event for fact in late.evaluation.late_facts] == [name]
        assert late.evaluation.late_facts[0].stage is Stage.COLLECTING
        assert late.evaluation.function_calls[0].complete is False
        assert late.evaluation.environment == closing.environment


def test_a_write_just_before_the_closure_deadline_is_admitted():
    closing = _closing(_apply(_evaluation(), _start("a")), deadline=T + 1)
    assert _apply(closing, _finish("a"), authority=AGENT).function_calls[0].complete is True


def test_any_event_after_the_closure_deadline_moves_closure_on():
    closing = _closing(_evaluation(), deadline=T - 1)
    assert _accept(closing, Read(AT)).evaluation.stage is Stage.COLLECTING
    assert _accept(closing, Read()).evaluation.stage is Stage.CLOSING
    # The seal no longer waits for a closure wait that is over.
    assert _apply(closing, Seal("out", AT)).stage is Stage.SEALED


def test_collecting_takes_observations_and_no_call_or_environment():
    collecting = _collecting(_apply(_evaluation(), _start("a")))
    assert "openshell" in _apply(collecting, RecordObservations("openshell", OpenShellObservations(), AT)).observations
    for event in [
        _finish("a"),
        ReplaceEnvironment(_Env(counter=2), AT),
        ChangeEnvironment({"counter": 2}, AT),
        StartTurn("m", AT),
    ]:
        late = transition(collecting, HOST, event)
        assert isinstance(late, Late) and late.rejected.kind is Rejection.CLOSING


def test_the_seal_fixes_the_snapshot_and_unfinished_calls_stay_unfinished():
    sealed = _sealed(_apply(_evaluation(), StartTurn("m", AT), _start("a"), _finish("b")), output="final")
    snapshot = sealed.snapshot
    assert snapshot is not None
    assert (snapshot.agent_output, snapshot.function_calls, snapshot.turns) == (
        "final",
        sealed.function_calls,
        sealed.turns,
    )
    assert [call.complete for call in snapshot.function_calls] == [False, True]
    assert snapshot.evidence_status == {}
    assert sealed.completed and sealed.agent_output == "final"


# --- late facts and the seal ---


@pytest.mark.parametrize(
    ("event", "kind", "evidence"),
    [
        (_start("new"), Rejection.SEALED, ("function_calls",)),
        (_finish("c1"), Rejection.SEALED, ("function_calls",)),
        (ReplaceEnvironment(_Env(counter=9), AT), Rejection.SEALED, ("environment",)),
        (ChangeEnvironment({"counter": 9}, AT), Rejection.SEALED, ("environment",)),
        (
            RecordObservations("openshell", OpenShellObservations(), AT),
            Rejection.SEALED,
            ("openshell.ocsf", "openshell.workdir"),
        ),
        (StartTurn("m", AT), Rejection.SEALED, ("turns",)),
        (Seal("another", AT), Rejection.SEALED, ("agent_output",)),
    ],
    ids=lambda value: type(value).__name__ if not isinstance(value, Rejection | tuple) else None,
)
def test_after_the_seal_every_write_is_a_late_fact_that_leaves_the_snapshot_alone(event, kind, evidence):
    sealed = _apply(_sealed(_apply(_evaluation(), StartTurn("m", AT), _start("c1"))), RecordGrade(_grade()))
    late = transition(sealed, HOST, event)
    assert isinstance(late, Late)
    assert late.rejected.kind is kind
    assert late.evaluation.snapshot == sealed.snapshot
    assert late.evaluation.grade == sealed.grade
    [fact] = late.evaluation.late_facts
    assert (fact.stage, fact.by, fact.evidence) == (Stage.SEALED, "host", evidence)
    assert late.evaluation.evidence_status == {key: "incomplete" for key in evidence}


def test_an_agent_write_after_the_seal_is_a_late_fact_refused_as_a_closed_session():
    late = transition(_sealed(_apply(_evaluation(), _start("c1"))), AGENT, _finish("c1"))
    assert isinstance(late, Late)
    assert late.rejected == Rejected(Rejection.SESSION_CLOSED, lifecycle.INVALID_SESSION_DETAIL)
    assert late.evaluation.late_facts[0].by == "agent"


@pytest.mark.parametrize(
    "event",
    [
        _start("done", x=2),
        _finish("done", x=2),
        _finish("done", result="other"),
        _start("open", x=2),
        _finish("open", x=2),
    ],
    ids=["start-args", "finish-args", "finish-result", "unfinished-start-args", "unfinished-finish-args"],
)
@pytest.mark.parametrize(
    ("authority", "kind", "by"), [(HOST, Rejection.SEALED, "host"), (AGENT, Rejection.SESSION_CLOSED, "agent")]
)
def test_a_conflicting_write_after_the_seal_is_a_late_fact(event, authority, kind, by):
    """After the seal, only an exact replay is a replay; anything else under a known call_id is a late fact."""
    sealed = _sealed(_apply(_evaluation(), _start("done", x=1), _finish("done", x=1), _start("open", x=1)))
    late = transition(sealed, authority, event)
    assert isinstance(late, Late)
    assert late.rejected.kind is kind
    [fact] = late.evaluation.late_facts
    assert (fact.by, fact.evidence, fact.call_id) == (by, ("function_calls",), event.req.call_id)
    assert late.evaluation.function_calls == sealed.function_calls
    assert late.evaluation.evidence_status == {"function_calls": "incomplete"}


def test_a_late_fact_is_recorded_once_per_distinct_write():
    sealed = _sealed(_evaluation())
    once = transition(sealed, AGENT, _finish("late"))
    assert isinstance(once, Late)
    twice = transition(once.evaluation, AGENT, _finish("late"))
    assert isinstance(twice, Late) and twice.evaluation == once.evaluation
    other = transition(once.evaluation, AGENT, _finish("late", result="other"))
    assert isinstance(other, Late) and len(other.evaluation.late_facts) == 2


def test_replays_after_the_seal_are_not_late_facts():
    sealed = _sealed(_apply(_evaluation(), _start("a"), _finish("b")), output="out")
    for event in [_start("a"), _start("b"), _finish("b")]:
        outcome = _accept(sealed, event, AGENT)
        assert outcome.evaluation == sealed and outcome.result[1] is False
    assert _accept(sealed, Seal("out", AT)).result == sealed.snapshot


def test_late_facts_before_the_seal_mark_the_snapshot_incomplete():
    closing = _closing(_evaluation())
    late = transition(closing, AGENT, _start("refused"))
    assert isinstance(late, Late)
    sealed = _apply(late.evaluation, EndClosureWait(now=200.0), Seal("out", AT))
    assert sealed.snapshot is not None
    assert sealed.snapshot.evidence_status == {"function_calls": "incomplete"}
    assert sealed.snapshot.function_calls == ()


def test_the_grade_is_recorded_once():
    sealed = _sealed(_evaluation())
    graded = _apply(sealed, RecordGrade(_grade()))
    other = _grade().model_copy(update={"utility": False})
    again = _accept(graded, RecordGrade(other))
    assert again.evaluation == graded
    assert again.evaluation.grade == _grade()


def test_observations_are_kept_per_source_and_replaced_until_the_seal():
    first, second, other = OpenShellObservations(), OpenShellObservations(image="b"), OpenShellObservations(image="c")
    evaluation = _apply(_evaluation(), RecordObservations("openshell", first, AT), RecordObservations("acs", other, AT))
    evaluation = _apply(_collecting(evaluation), RecordObservations("openshell", second, AT))
    assert dict(evaluation.observations) == {"openshell": second, "acs": other}
