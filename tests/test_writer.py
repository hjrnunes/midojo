"""The single writer: one lock per evaluation, applied the same way for HTTP and in-process callers."""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from midojo.control_plane import lifecycle, service
from midojo.control_plane.config import AppConfig
from midojo.control_plane.lifecycle import HOST, RejectedError, Rejection
from midojo.control_plane.models import CreateFunctionCallRecord, CreateFunctionCallStart
from midojo.control_plane.service import ById, BySession, ControlPlane
from midojo.control_plane.state import Stage

THREADS = 8


@pytest.fixture
def control(suite) -> ControlPlane:
    return ControlPlane({"weather": suite}, config=AppConfig(closure_wait_seconds=0))


def _new(control: ControlPlane) -> tuple[str, str, str]:
    run = control.create_run("weather")
    created = control.create_evaluation(run.id, "weather_new_york")
    return run.id, created.id, created.session_token


def _start(call_id: str = "same") -> CreateFunctionCallStart:
    return CreateFunctionCallStart(call_id=call_id, function="get_weather", args={"city": "New York"})


def _finish(call_id: str) -> CreateFunctionCallRecord:
    return CreateFunctionCallRecord(call_id=call_id, function="get_weather", args={"city": "New York"}, result="72°F")


@pytest.fixture
def widened(monkeypatch):
    """Every transition takes a while after it decides, so concurrent writers overlap between decision and commit."""
    transition = lifecycle.transition

    def slow(evaluation, authority, event):
        outcome = transition(evaluation, authority, event)
        time.sleep(0.02)
        return outcome

    monkeypatch.setattr(lifecycle, "transition", slow)


def _unsynchronised_start(
    control: ControlPlane, run_id: str, eval_id: str, req: CreateFunctionCallStart, decided: threading.Barrier
):
    """The writer's steps without its lock: look up, decide, commit. Every caller decides before any commits."""
    evaluation = control.store.evaluation_value(run_id, eval_id)
    assert evaluation is not None
    outcome = lifecycle.transition(evaluation, HOST, lifecycle.StartCall(req, "now"))
    assert isinstance(outcome, lifecycle.Accepted)
    decided.wait(5)
    control.store.commit_evaluation(outcome.evaluation)
    return outcome.result


def test_control_without_the_writer_concurrent_starts_admit_one_call_id_twice(control):
    run_id, eval_id, _ = _new(control)
    decided = threading.Barrier(THREADS)
    with ThreadPoolExecutor(THREADS) as pool:
        results = list(
            pool.map(lambda _: _unsynchronised_start(control, run_id, eval_id, _start(), decided), range(THREADS))
        )
    # Every caller was told its start was admitted, and all but the last commit were lost.
    assert [created for _, created in results] == [True] * THREADS
    assert len(control.evaluation(ById(run_id, eval_id)).function_calls) == 1


@pytest.mark.parametrize("by", ["session", "id"])
def test_concurrent_starts_of_one_call_id_admit_it_once(control, widened, by):
    run_id, eval_id, token = _new(control)
    target = BySession(token) if by == "session" else ById(run_id, eval_id)
    with ThreadPoolExecutor(THREADS) as pool:
        results = list(pool.map(lambda _: control.start_call(target, _start()), range(THREADS)))
    assert sorted(created for _, created in results) == [False] * (THREADS - 1) + [True]
    assert len({record.timestamp for record, _ in results}) == 1
    assert len(control.evaluation(ById(run_id, eval_id)).function_calls) == 1


def test_concurrent_writes_on_both_paths_are_all_kept(control, widened):
    run_id, eval_id, token = _new(control)

    def write(i: int):
        target = BySession(token) if i % 2 else ById(run_id, eval_id)
        return control.finish_call(target, _finish(f"call-{i}"))

    with ThreadPoolExecutor(THREADS) as pool:
        list(pool.map(write, range(THREADS)))
    calls = control.evaluation(ById(run_id, eval_id)).function_calls
    assert sorted(call.call_id for call in calls) == sorted(f"call-{i}" for i in range(THREADS))


def test_http_routes_and_in_process_callers_share_one_writer(app, widened):
    control: ControlPlane = app.state.control
    run_id, eval_id, token = _new(control)
    body = _start().model_dump()

    def start(i: int) -> bool:
        if i % 2:
            response = client.post(
                "/agent/function-calls/start", headers={"Authorization": f"Bearer {token}"}, json=body
            )
            assert response.status_code in (200, 201)
            return response.status_code == 201
        return control.start_call(ById(run_id, eval_id), _start())[1]

    with TestClient(app) as client, ThreadPoolExecutor(THREADS) as pool:
        created = list(pool.map(start, range(THREADS)))
    assert created.count(True) == 1
    assert len(control.evaluation(ById(run_id, eval_id)).function_calls) == 1


def test_late_calls_cannot_slip_into_the_snapshot_from_other_threads(control, widened):
    run_id, eval_id, token = _new(control)
    admitted: list[str] = []
    refused: list[str] = []

    def record(i: int) -> None:
        try:
            control.finish_call(BySession(token), _finish(f"call-{i}"))
            admitted.append(f"call-{i}")
        except RejectedError as exc:
            assert exc.late
            refused.append(f"call-{i}")

    with ThreadPoolExecutor(THREADS + 1) as pool:
        futures = [pool.submit(record, i) for i in range(THREADS)]
        sealing = pool.submit(asyncio.run, control.complete(run_id, eval_id, "done"))
        futures += [pool.submit(record, i) for i in range(THREADS, 2 * THREADS)]
        for future in [*futures, sealing]:
            future.result()

    evaluation = control.evaluation(ById(run_id, eval_id))
    assert evaluation.snapshot is not None
    assert sorted(call.call_id for call in evaluation.snapshot.function_calls) == sorted(admitted)
    assert evaluation.snapshot.function_calls == evaluation.function_calls
    assert sorted(fact.call_id or "" for fact in evaluation.late_facts) == sorted(refused)
    assert len(admitted) + len(refused) == 2 * THREADS


def test_each_evaluation_has_its_own_writer(control, monkeypatch):
    run_a, eval_a, _ = _new(control)
    run_b, eval_b, _ = _new(control)
    entered, release = threading.Event(), threading.Event()
    transition = lifecycle.transition

    def blocking(evaluation, authority, event):
        if evaluation.id == eval_a and isinstance(event, lifecycle.StartCall):
            entered.set()
            release.wait(5)
        return transition(evaluation, authority, event)

    monkeypatch.setattr(lifecycle, "transition", blocking)
    holder = threading.Thread(target=control.start_call, args=(ById(run_a, eval_a), _start()))
    holder.start()
    try:
        assert entered.wait(5)
        # A holds its writer; B's write and read don't wait for it.
        writer_b = threading.Thread(target=control.start_call, args=(ById(run_b, eval_b), _start()))
        writer_b.start()
        writer_b.join(2)
        assert not writer_b.is_alive()
        assert len(control.evaluation(ById(run_b, eval_b)).function_calls) == 1
    finally:
        release.set()
        holder.join(5)


def test_values_the_writer_returns_share_nothing_with_the_stored_evaluation(control):
    run_id, eval_id, token = _new(control)
    target = ById(run_id, eval_id)
    record, _ = control.finish_call(target, _finish("c1"))
    record.args["city"] = "changed"
    assert record.post_environment is not None
    record.post_environment.weather_alerts.append({"city": "x", "message": "changed"})  # type: ignore[attr-defined]
    evaluation = control.evaluation(target)
    evaluation.environment.weather_alerts.append({"city": "y", "message": "changed"})  # type: ignore[attr-defined]
    evaluation.function_calls[0].args["city"] = "changed again"
    evaluation.observations["openshell"] = None  # type: ignore[index]

    stored = control.evaluation(target)
    assert stored.function_calls[0].args == {"city": "New York"}
    assert stored.function_calls[0].post_environment == stored.environment
    assert stored.environment == control.evaluation(BySession(token)).environment
    assert stored.environment.weather_alerts == []  # type: ignore[attr-defined]
    assert dict(stored.observations) == {}


def test_an_event_payload_changed_after_submission_changes_nothing_stored(control):
    run_id, eval_id, _ = _new(control)
    target = ById(run_id, eval_id)
    req = _finish("c1")
    control.finish_call(target, req)
    req.args["city"] = "changed"
    environment = control.evaluation(target).environment
    control.replace_environment(target, environment)
    environment.weather_alerts.append({"city": "z", "message": "changed"})  # type: ignore[attr-defined]
    stored = control.evaluation(target)
    assert stored.function_calls[0].args == {"city": "New York"}
    assert stored.environment.weather_alerts == []  # type: ignore[attr-defined]


def test_grading_runs_without_holding_the_writer(control, monkeypatch):
    run_id, eval_id, token = _new(control)
    asyncio.run(control.complete(run_id, eval_id, "New York is 72°F and sunny"))
    grade_snapshot = service.grade_snapshot
    writes: list[str] = []

    def grader(suite, snapshot):
        # Another thread writes to the same evaluation while grading runs (a late call).
        def late():
            with pytest.raises(RejectedError) as refused:
                control.finish_call(BySession(token), _finish("late"))
            writes.append(refused.value.kind)

        writer = threading.Thread(target=late)
        writer.start()
        writer.join(2)
        assert not writer.is_alive(), "the writer was held across grading"
        return grade_snapshot(suite, snapshot)

    monkeypatch.setattr(service, "grade_snapshot", grader)
    graded = control.grade(run_id, eval_id)
    assert writes == [Rejection.SESSION_CLOSED]
    assert graded.utility is True
    # The late call arrived while grading: the grade is the snapshot's, its evidence marked incomplete.
    assert graded.evidence_status == {
        "openshell.ocsf": "not_collected",
        "openshell.workdir": "not_collected",
        "function_calls": "incomplete",
    }
    assert control.evaluation(ById(run_id, eval_id)).snapshot.function_calls == ()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_the_closure_wait_never_holds_the_writer(suite):
    control = ControlPlane({"weather": suite}, config=AppConfig(closure_wait_seconds=5))
    run_id, eval_id, token = _new(control)
    control.start_call(BySession(token), _start("admitted"))
    began = time.monotonic()
    closing = asyncio.create_task(control.close(run_id, eval_id))
    await asyncio.sleep(0.1)
    # The call finishes from another thread during the wait, and a new call is refused.
    await asyncio.to_thread(control.finish_call, BySession(token), _finish("admitted"))
    with pytest.raises(RejectedError) as refused:
        await asyncio.to_thread(control.start_call, BySession(token), _start("new"))
    assert (refused.value.kind, refused.value.late) == (Rejection.CLOSING, True)
    evaluation = await closing
    assert time.monotonic() - began < 2
    assert evaluation.stage is Stage.COLLECTING
    assert [call.complete for call in evaluation.function_calls] == [True]


@pytest.mark.asyncio
async def test_a_cancelled_closure_wait_resumes_with_the_same_deadline(suite):
    control = ControlPlane({"weather": suite}, config=AppConfig(closure_wait_seconds=0.5))
    run_id, eval_id, token = _new(control)
    control.start_call(BySession(token), _start("admitted"))
    closing = asyncio.create_task(control.close(run_id, eval_id))
    await asyncio.sleep(0.1)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    evaluation = control.evaluation(ById(run_id, eval_id))
    assert evaluation.stage is Stage.CLOSING
    deadline = evaluation.closure_deadline
    assert deadline is not None

    evaluation = await control.close(run_id, eval_id)
    assert evaluation.closure_deadline == deadline
    assert time.time() >= deadline
    assert evaluation.stage is Stage.COLLECTING
    assert [call.complete for call in evaluation.function_calls] == [False]


@pytest.mark.asyncio
async def test_a_finish_after_the_deadline_of_a_cancelled_closure_is_late(suite):
    """No one ends the closure wait (its close was cancelled): the deadline still applies to what arrives."""
    control = ControlPlane({"weather": suite}, config=AppConfig(closure_wait_seconds=0.2))
    run_id, eval_id, token = _new(control)
    control.start_call(BySession(token), _start("admitted"))
    closing = asyncio.create_task(control.close(run_id, eval_id))
    await asyncio.sleep(0.05)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    await asyncio.sleep(0.3)

    with pytest.raises(RejectedError) as refused:
        control.finish_call(BySession(token), _finish("admitted"))
    assert (refused.value.kind, refused.value.late) == (Rejection.CLOSING, True)
    evaluation = control.evaluation(ById(run_id, eval_id))
    assert evaluation.stage is Stage.COLLECTING
    assert [call.complete for call in evaluation.function_calls] == [False]
    assert [fact.event for fact in evaluation.late_facts] == ["finish_call"]
