"""Unit tests for the in-memory Store implementation (PR1 store seam)."""

from __future__ import annotations

import re
import uuid

import pytest

from midojo.control_plane.models import CreateFunctionCallRecord, CreateFunctionCallStart
from midojo.control_plane.store import FunctionCallConflictError, InMemoryStore
from midojo.types import Environment


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


class _Env(Environment):
    counter: int = 0


def _fc(function: str, result: str) -> CreateFunctionCallRecord:
    return CreateFunctionCallRecord(function=function, args={}, result=result)


def _make_eval(
    store: InMemoryStore,
    run_id: str,
    *,
    environment: Environment | None = None,
    pre_environment: Environment | None = None,
    agent_input: str | None = None,
):
    return store.create_evaluation(
        run_id,
        user_task_id="ut",
        injection_task_id=None,
        pre_environment=pre_environment if pre_environment is not None else _Env(),
        environment=environment if environment is not None else _Env(),
        active_injections={},
        agent_input=agent_input,
    )


# --- runs ---


def test_create_run_round_trip(store):
    r1 = store.create_run("test")
    r2 = store.create_run("test")
    # Each run is retrievable by its own id: two creates coexist without clobbering
    # (if create_run reused an id, get_run(r1.id) would return r2).
    assert store.get_run(r1.id) is r1
    assert store.get_run(r2.id) is r2


def test_get_run_unknown_returns_none(store):
    # None (not a raise) is the contract the dependency layer turns into a 404.
    assert store.get_run("nope") is None


def test_list_runs(store):
    assert store.list_runs() == []
    r1 = store.create_run("test")
    r2 = store.create_run("test")
    assert {r.id for r in store.list_runs()} == {r1.id, r2.id}


# --- evaluations ---


def test_create_evaluation_stored_under_run(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id, agent_input="prompt")
    assert ev.run_id == run.id
    assert store.get_evaluation(run.id, ev.id) is ev
    assert re.fullmatch(r"[0-9a-f]{10}", ev.id)
    assert ev.agent_input == "prompt"  # kwarg is plumbed through to the Evaluation


def test_evaluation_id_collision_preserves_existing_records_and_sessions(store, monkeypatch):
    run = store.create_run("test")
    first = _make_eval(store, run.id, agent_input="first")
    token, _ = store.create_session(run.id, first.id, ttl_seconds=60)
    next_run = store.create_run("other_suite")
    new_id = "0000000000" if first.id != "0000000000" else "1111111111"
    candidates = iter([first.id, first.id, new_id])
    monkeypatch.setattr("midojo.control_plane.store.secrets.token_hex", lambda n: next(candidates))

    second = _make_eval(store, next_run.id, agent_input="second")

    assert second.id == new_id
    assert store.get_evaluation(run.id, first.id) is first
    assert store.get_evaluation(next_run.id, second.id) is second
    assert store.session_evaluation(token) is first


def test_get_evaluation_unknown_returns_none(store):
    run = store.create_run("test")
    # Two distinct not-found branches: known run/unknown eval, and unknown run.
    assert store.get_evaluation(run.id, "nope") is None
    assert store.get_evaluation("nope", "nope") is None


# --- function calls ---


def test_append_function_call_first_call_chains_from_pre_environment(store):
    run = store.create_run("test")
    # pre_environment and the live environment differ so the assertion can tell
    # which one the first call's pre-env is taken from.
    ev = _make_eval(store, run.id, pre_environment=_Env(counter=7), environment=_Env(counter=9))
    appended = store.append_function_call(run.id, ev.id, _fc("a", "r0"))
    rec = ev.function_calls[-1]
    assert appended == (rec, True)  # returns the new record, and that it was created
    assert isinstance(rec.pre_environment, _Env)
    assert rec.pre_environment.counter == 7  # first call chains from pre_environment, not the live env
    assert isinstance(rec.post_environment, _Env)
    assert rec.post_environment.counter == 9
    assert ev.function_calls == [rec]


def test_append_function_call_chains_pre_env_across_environment_change(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id, environment=_Env(counter=0))
    store.append_function_call(run.id, ev.id, _fc("a", "r0"))
    store.set_environment(run.id, ev.id, _Env(counter=1))
    store.append_function_call(run.id, ev.id, _fc("b", "r1"))
    r0, r1 = ev.function_calls
    # Second call's pre-env is the first call's post-env...
    assert r1.pre_environment == r0.post_environment
    # ...and its post-env reflects the set_environment that happened in between.
    assert isinstance(r1.post_environment, _Env)
    assert r1.post_environment.counter == 1


def test_append_function_call_post_env_is_deep_copy(store):
    run = store.create_run("test")
    env = _Env(counter=0)
    ev = _make_eval(store, run.id, environment=env)
    store.append_function_call(run.id, ev.id, _fc("a", "r"))
    rec = ev.function_calls[-1]
    env.counter = 99  # mutate the live env in place after recording
    # The recorded snapshot is a deep copy, so it doesn't track later mutations.
    assert isinstance(rec.post_environment, _Env)
    assert rec.post_environment.counter == 0


def test_append_function_call_without_call_id_assigns_a_uuid(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    store.append_function_call(run.id, ev.id, _fc("a", "r"))
    store.append_function_call(run.id, ev.id, _fc("a", "r"))
    first, second = ev.function_calls  # no call_id, no dedupe: identical bodies both append
    assert str(uuid.UUID(first.call_id)) == first.call_id
    assert first.call_id != second.call_id


def test_append_function_call_replay_returns_stored_record(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    req = CreateFunctionCallRecord(call_id="c1", function="a", args={"x": 1}, result="r")
    appended = store.append_function_call(run.id, ev.id, req)
    assert appended is not None
    record, created = appended
    assert created is True
    assert record.call_id == "c1"
    assert store.append_function_call(run.id, ev.id, req.model_copy()) == (record, False)
    assert ev.function_calls == [record]


@pytest.mark.parametrize(
    "change", [{"function": "b"}, {"args": {"x": 2}}, {"result": "other"}, {"error": "boom"}, {"blocked": True}]
)
def test_append_function_call_conflicting_body_raises(store, change):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    req = CreateFunctionCallRecord(call_id="c1", function="a", args={"x": 1}, result="r")
    store.append_function_call(run.id, ev.id, req)
    with pytest.raises(FunctionCallConflictError, match="c1"):
        store.append_function_call(run.id, ev.id, req.model_copy(update=change))
    assert len(ev.function_calls) == 1


def test_append_function_call_call_ids_are_per_evaluation(store):
    run = store.create_run("test")
    first, second = _make_eval(store, run.id), _make_eval(store, run.id)
    req = CreateFunctionCallRecord(call_id="c1", function="a", args={}, result="r")
    store.append_function_call(run.id, first.id, req)
    appended = store.append_function_call(run.id, second.id, req)
    assert appended is not None
    assert appended[1] is True
    assert len(second.function_calls) == 1


def test_start_function_call_snapshots_the_live_environment(store):
    run = store.create_run("test")
    env = _Env(counter=3)
    ev = _make_eval(store, run.id, pre_environment=_Env(counter=0), environment=env)
    started = store.start_function_call(run.id, ev.id, CreateFunctionCallStart(call_id="c1", function="a", args={}))
    assert started is not None
    record, created = started
    env.counter = 4  # the start's snapshot is a deep copy
    assert created is True
    assert (record.complete, record.result, record.post_environment) == (False, None, None)
    assert isinstance(record.pre_environment, _Env)
    assert record.pre_environment.counter == 3
    assert ev.function_calls == [record]


def test_finish_completes_the_started_record_in_place(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    store.start_function_call(run.id, ev.id, CreateFunctionCallStart(call_id="c1", function="a", args={"x": 1}))
    store.set_environment(run.id, ev.id, _Env(counter=5))
    finish = CreateFunctionCallRecord(call_id="c1", function="a", args={"x": 1}, result="r", error="e", blocked=True)
    finished = store.append_function_call(run.id, ev.id, finish)
    assert finished is not None
    record, created = finished
    assert created is True
    assert ev.function_calls == [record]
    assert (record.complete, record.result, record.error, record.blocked) == (True, "r", "e", True)
    assert isinstance(record.post_environment, _Env)
    assert record.post_environment.counter == 5
    assert store.append_function_call(run.id, ev.id, finish.model_copy()) == (record, False)


@pytest.mark.parametrize("change", [{"function": "b"}, {"args": {"x": 2}}])
def test_start_and_finish_conflicts_raise(store, change):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    start = CreateFunctionCallStart(call_id="c1", function="a", args={"x": 1})
    store.start_function_call(run.id, ev.id, start)
    with pytest.raises(FunctionCallConflictError, match="c1"):
        store.start_function_call(run.id, ev.id, start.model_copy(update=change))
    finish = CreateFunctionCallRecord(call_id="c1", function="a", args={"x": 1}, result="r")
    with pytest.raises(FunctionCallConflictError, match="c1"):
        store.append_function_call(run.id, ev.id, finish.model_copy(update=change))
    [record] = ev.function_calls
    assert record.complete is False


def test_start_function_call_unknown_returns_none(store):
    assert (
        store.start_function_call("nope", "nope", CreateFunctionCallStart(call_id="c", function="a", args={})) is None
    )


# --- environment / observations / grade / complete ---


def test_set_environment_replaces(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id, environment=_Env(counter=1))
    new_env = _Env(counter=2)
    # Returns the mutated evaluation; the new env is installed on it.
    assert store.set_environment(run.id, ev.id, new_env) is ev
    assert ev.environment is new_env


def test_record_observations_keyed_by_source(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    e1, e2, acs = _Env(counter=1), _Env(counter=2), _Env(counter=3)
    assert store.record_observations(run.id, ev.id, "openshell", e1) is ev
    assert ev.observations == {"openshell": e1}
    store.record_observations(run.id, ev.id, "acs", acs)
    assert ev.observations == {"openshell": e1, "acs": acs}
    # A second write to the same source replaces that source's value only.
    store.record_observations(run.id, ev.id, "openshell", e2)
    assert ev.observations == {"openshell": e2, "acs": acs}


def test_set_grade(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    # Distinct values guard against the two flags being swapped.
    assert store.set_grade(run.id, ev.id, utility=True, security=True, security_reason="output contains X") is ev
    assert ev.utility is True
    assert ev.security is True
    assert ev.security_reason == "output contains X"


def test_set_grade_security_reason_defaults_to_none(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    # Reason is optional: an attack that failed (or a utility-only grade) leaves it None.
    assert store.set_grade(run.id, ev.id, utility=True, security=False) is ev
    assert ev.security_reason is None


def test_complete_evaluation(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    assert store.complete_evaluation(run.id, ev.id, "final answer") is ev
    # Completing records both the flag and the agent's final output.
    assert ev.completed is True
    assert ev.agent_output == "final answer"


# --- mutations return None for an unknown (run, eval) ---
#
# Every per-eval mutation honors the uniform `Evaluation | None` contract (the
# HTTP layer turns that None into a 404 / 400). Per-method coverage so the
# contract stays pinned at each entry point for future Store implementations.


def test_append_function_call_unknown_returns_none(store):
    assert store.append_function_call("nope", "nope", _fc("a", "r")) is None


def test_set_environment_unknown_returns_none(store):
    assert store.set_environment("nope", "nope", _Env()) is None


def test_record_observations_unknown_returns_none(store):
    assert store.record_observations("nope", "nope", "src", _Env()) is None


def test_set_grade_unknown_returns_none(store):
    assert store.set_grade("nope", "nope", utility=True, security=True) is None


def test_complete_evaluation_unknown_returns_none(store):
    assert store.complete_evaluation("nope", "nope", "out") is None
