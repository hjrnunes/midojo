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


def _make_eval(store: InMemoryStore, run_id: str | None, **changes) -> Evaluation:
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


def test_evaluations_outside_any_run_are_kept_apart_from_the_runs(store):
    assert store.list_evaluations(None) == []
    run = store.create_run("test")
    inside = _make_eval(store, run.id)
    outside = _make_eval(store, None)
    assert [e.id for e in store.list_evaluations(None)] == [outside.id]
    assert [e.id for e in store.list_evaluations(run.id)] == [inside.id]
    assert store.get_evaluation(None, outside.id) == outside
    assert store.get_evaluation(None, inside.id) is None
    assert store.get_evaluation(run.id, outside.id) is None
    # Outside a run is not a run: it isn't listed among them.
    assert [r.id for r in store.list_runs()] == [run.id]
    # The writer reaches an evaluation outside a run by the same (None, eval_id) key.
    writer = store.writer()
    writer.apply(None, outside.id, lambda value: dataclasses.replace(value, agent_input="changed") if value else None)
    assert store.get_evaluation(None, outside.id).agent_input == "changed"  # type: ignore[union-attr]
    assert store.get_evaluation(run.id, inside.id) == inside
    with pytest.raises(KeyError):
        writer.apply(run.id, outside.id, lambda value: outside)
    store.bind_session("h", None, outside.id)
    assert store.session_target("h") == (None, outside.id)


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


def test_no_public_store_method_hands_out_or_keeps_a_resident_value(store):
    """Only the writer capability reaches stored values; every public method returns or takes copies."""
    public = {name for name in dir(store) if not name.startswith("_") and callable(getattr(store, name))}
    assert public == {
        "create_run",
        "get_run",
        "list_runs",
        "new_evaluation_id",
        "insert_evaluation",
        "get_evaluation",
        "list_evaluations",
        "writer",
        "bind_session",
        "unbind_session",
        "session_target",
    }


def test_the_writer_capability_has_one_holder(store):
    store.writer()
    with pytest.raises(RuntimeError, match="writer"):
        store.writer()


def test_the_writer_hands_out_and_keeps_copies(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    writer = store.writer()
    seen: dict[str, Evaluation] = {}

    def step(value: Evaluation | None) -> Evaluation:
        assert value is not None and value == ev
        record = FunctionCallRecord(function="a", args={}, result="r", timestamp="t", pre_environment=_Env())
        seen["given"], seen["kept"] = value, dataclasses.replace(value, function_calls=(record,), stage=Stage.CLOSING)
        return seen["kept"]

    writer.apply(run.id, ev.id, step)
    committed = store.get_evaluation(run.id, ev.id)
    assert committed == seen["kept"]
    # Neither the value the step was given nor the one it returned is the stored value.
    seen["given"].environment.counter = 99  # type: ignore[attr-defined]
    seen["kept"].function_calls[0].args["x"] = 1
    seen["kept"].environment.counter = 98  # type: ignore[attr-defined]
    assert store.get_evaluation(run.id, ev.id) == committed
    # A step that returns None keeps the value as it is; an unknown evaluation is None.
    writer.apply(run.id, ev.id, lambda value: None)
    assert store.get_evaluation(run.id, ev.id) == committed
    unknown: list[Evaluation | None] = []
    writer.apply(run.id, "nope", lambda value: unknown.append(value))
    assert unknown == [None]


def test_the_writer_cannot_create_an_evaluation(store):
    run = store.create_run("test")
    ev = _make_eval(store, run.id)
    with pytest.raises(KeyError):
        store.writer().apply(run.id, "nope", lambda value: dataclasses.replace(ev, id="nope"))


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
