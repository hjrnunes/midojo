"""Unit tests for the in-memory Store: a map of evaluation values that only the writer replaces.

The rules for calls, turns and closure are the lifecycle's (tests/test_lifecycle.py);
the store only keeps values and hands out copies.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from midojo.control_plane.state import Evaluation, Stage
from midojo.control_plane.store import InMemoryStore
from midojo.types import Environment, FunctionCallRecord


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


class _Env(Environment):
    counter: int = 0


def _make_eval(store: InMemoryStore, run_id: str, **changes) -> Evaluation:
    evaluation = Evaluation(
        id=store.new_evaluation_id(),
        run_id=run_id,
        user_task_id="ut",
        injection_task_id=None,
        pre_environment=_Env(),
        environment=_Env(),
    )
    evaluation = dataclasses.replace(evaluation, **changes)
    store.insert_evaluation(evaluation)
    return evaluation


# --- runs ---


def test_create_run_round_trip(store):
    r1 = store.create_run("test")
    r2 = store.create_run("test")
    # Each run is retrievable by its own id: two creates coexist without clobbering.
    assert store.get_run(r1.id) == r1
    assert store.get_run(r2.id) == r2
    assert r1.id != r2.id


def test_get_run_unknown_returns_none(store):
    # None (not a raise) is the contract the writer turns into an unknown_run rejection.
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
    assert store.get_evaluation(run.id, ev.id) == ev
    assert re.fullmatch(r"[0-9a-f]{10}", ev.id)
    assert [e.id for e in store.list_evaluations(run.id)] == [ev.id]
    assert store.list_evaluations(store.create_run("other").id) == []


def test_evaluation_id_collision_draws_another_id(store, monkeypatch):
    run = store.create_run("test")
    first = _make_eval(store, run.id)
    new_id = "0000000000" if first.id != "0000000000" else "1111111111"
    candidates = iter([first.id, first.id, new_id])
    monkeypatch.setattr("midojo.control_plane.store.secrets.token_hex", lambda n: next(candidates))
    assert store.new_evaluation_id() == new_id


def test_inserting_an_existing_evaluation_raises(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    with pytest.raises(ValueError, match=ev.id):
        store.insert_evaluation(ev)


def test_get_evaluation_unknown_returns_none(store):
    run = store.create_run("test")
    # Two distinct not-found branches: known run/unknown eval, and unknown run.
    assert store.get_evaluation(run.id, "nope") is None
    assert store.get_evaluation("nope", "nope") is None


def test_reads_are_copies_that_share_nothing_with_the_stored_value(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id, environment=_Env(counter=1))
    read = store.get_evaluation(run.id, ev.id)
    assert read is not None
    assert read is not store.evaluation_value(run.id, ev.id)
    read.environment.counter = 99  # type: ignore[attr-defined]
    [listed] = store.list_evaluations(run.id)
    listed.pre_environment.counter = 98  # type: ignore[attr-defined]
    assert store.get_evaluation(run.id, ev.id) == ev


def test_an_inserted_value_is_copied_in(store):
    run = store.create_run("test")
    environment = _Env(counter=1)
    ev = _make_eval(store, run.id, environment=environment)
    environment.counter = 2
    stored = store.get_evaluation(run.id, ev.id)
    assert stored is not None
    assert stored.environment == _Env(counter=1)


def test_commit_replaces_the_value(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    record = FunctionCallRecord(function="a", args={}, result="r", timestamp="t", pre_environment=_Env())
    committed = dataclasses.replace(ev, function_calls=(record,), stage=Stage.CLOSING)
    store.commit_evaluation(committed)
    assert store.get_evaluation(run.id, ev.id) == committed
    assert store.evaluation_value(run.id, ev.id) is committed
    # The value read before the commit is unchanged: values are never changed in place.
    assert ev.function_calls == () and ev.stage is Stage.OPEN


def test_commit_of_an_unknown_evaluation_raises(store):
    run = store.create_run("test")
    ev = Evaluation(
        id="nope", run_id=run.id, user_task_id="ut", injection_task_id=None, pre_environment=_Env(), environment=_Env()
    )
    with pytest.raises(KeyError):
        store.commit_evaluation(ev)


# --- the session token index ---


def test_session_index_binds_and_unbinds(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    assert store.session_target("h") is None
    store.bind_session("h", run.id, ev.id)
    assert store.session_target("h") == (run.id, ev.id)
    store.unbind_session("h")
    store.unbind_session("h")
    assert store.session_target("h") is None
