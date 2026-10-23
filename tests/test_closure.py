"""Staged closure, the seal and late facts, over the control plane's HTTP routes."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

from midojo.control_plane.config import AppConfig
from midojo.control_plane.main import create_app


def _completed(client: TestClient, agent_output: str = "New York is 72°F and sunny") -> str:
    run_id = client.post("/runs", json={"suite_name": "weather"}).json()["id"]
    eval_id = client.post(f"/runs/{run_id}/evaluations", json={"user_task_id": "weather_new_york"}).json()["id"]
    url = f"/runs/{run_id}/evaluations/{eval_id}"
    assert client.post(f"{url}/complete", json={"agent_output": agent_output}).status_code == 200
    return url


@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("function-calls", {"call_id": "late", "function": "get_weather", "args": {}, "result": "72°F"}),
        ("function-calls/start", {"call_id": "late", "function": "get_weather", "args": {}}),
    ],
)
def test_run_scoped_call_routes_refuse_a_completed_evaluation(client, path, body):
    url = _completed(client)
    response = client.post(f"{url}/{path}", json=body)
    assert response.status_code == 409
    assert response.json() == {"detail": "Evaluation already completed", "kind": "sealed", "late_fact": True}
    assert client.get(url).json()["function_calls"] == []


def test_a_completed_evaluation_keeps_its_environment(client):
    url = _completed(client)
    environment = client.get(f"{url}/environment").json()
    changed = {**environment, "weather_alerts": [{"city": "Boston", "message": "late"}]}
    assert client.put(f"{url}/environment", json=changed).status_code == 409
    assert client.get(f"{url}/environment").json() == environment


def test_a_completed_evaluation_keeps_its_observations(client):
    url = _completed(client)
    observed = {"evidence_status": {"ocsf": "complete", "workdir": "failed"}}
    assert client.post(f"{url}/observations", json={"source": "openshell", "data": observed}).status_code == 409
    assert client.get(f"{url}/observations").json() == {}


def test_a_grade_cannot_be_replaced(client):
    url = _completed(client, "New York is 72°F and sunny")
    assert client.post(f"{url}/grade").json()["utility"] is True
    assert client.post(f"{url}/complete", json={"agent_output": "no idea"}).status_code == 409
    assert client.post(f"{url}/grade").json()["utility"] is True
    assert client.get(url).json()["agent_output"] == "New York is 72°F and sunny"


# --- staged closure ---


def _evaluation(client: TestClient) -> tuple[str, dict[str, str]]:
    run_id = client.post("/runs", json={"suite_name": "weather"}).json()["id"]
    created = client.post(f"/runs/{run_id}/evaluations", json={"user_task_id": "weather_new_york"}).json()
    return f"/runs/{run_id}/evaluations/{created['id']}", {"Authorization": f"Bearer {created['session_token']}"}


START = {"call_id": "c1", "function": "get_weather", "args": {"city": "New York"}}
FINISH = {**START, "result": "72°F, sunny"}


def _app(suite, wait: float) -> TestClient:
    return TestClient(create_app({"weather": suite}, config=AppConfig(closure_wait_seconds=wait)))


def test_closure_lets_admitted_calls_finish_and_admits_no_new_call(suite):
    with _app(suite, wait=5) as client:
        url, session = _evaluation(client)
        assert client.post("/agent/function-calls/start", headers=session, json=START).status_code == 201
        closed: dict = {}
        closing = threading.Thread(target=lambda: closed.update(client.post(f"{url}/close").json()))
        began = time.monotonic()
        closing.start()
        while client.get(url).json()["stage"] != "closing":
            time.sleep(0.01)

        refused = client.post("/agent/function-calls/start", headers=session, json={**START, "call_id": "c2"})
        assert refused.status_code == 409
        assert refused.json()["kind"] == "closing" and refused.json()["late_fact"] is True
        refused = client.post(f"{url}/function-calls", json={**FINISH, "call_id": "c3"})
        assert (refused.status_code, refused.json()["kind"]) == (409, "closing")
        assert client.post("/agent/function-calls", headers=session, json=FINISH).status_code == 201
        closing.join(5)

        assert time.monotonic() - began < 4  # the wait ended when the admitted call finished
        assert closed == {"stage": "collecting", "unfinished_calls": []}
        evaluation = client.get(url).json()
        assert [(call["call_id"], call["complete"]) for call in evaluation["function_calls"]] == [("c1", True)]
        assert [(fact["event"], fact["call_id"], fact["by"]) for fact in evaluation["late_facts"]] == [
            ("start_call", "c2", "agent"),
            ("finish_call", "c3", "host"),
        ]
        assert evaluation["evidence_status"] == {"function_calls": "incomplete"}


def test_when_the_closure_wait_runs_out_a_call_stays_unfinished_and_its_finish_is_late(suite):
    with _app(suite, wait=0.2) as client:
        url, session = _evaluation(client)
        client.post("/agent/function-calls/start", headers=session, json=START)
        assert client.post(f"{url}/close").json() == {"stage": "collecting", "unfinished_calls": ["c1"]}
        # Closing again returns at once.
        assert client.post(f"{url}/close").json()["stage"] == "collecting"

        late = client.post("/agent/function-calls", headers=session, json=FINISH)
        assert (late.status_code, late.json()["kind"], late.json()["late_fact"]) == (409, "closing", True)
        # The final observations are still recorded while collecting.
        observed = {"evidence_status": {"ocsf": "complete", "workdir": "complete"}}
        assert client.post(f"{url}/observations", json={"source": "openshell", "data": observed}).status_code == 200
        assert client.post(f"{url}/complete", json={"agent_output": "72 and sunny"}).status_code == 200

        evaluation = client.get(url).json()
        assert evaluation["stage"] == "sealed" and evaluation["completed"] is True
        [call] = evaluation["function_calls"]
        assert (call["complete"], call["result"]) == (False, None)
        graded = client.post(f"{url}/grade").json()
        assert graded["evidence_status"] == {
            "openshell.ocsf": "complete",
            "openshell.workdir": "complete",
            "function_calls": "incomplete",
        }


def test_complete_without_close_closes_first(suite):
    with _app(suite, wait=0.2) as client:
        url, session = _evaluation(client)
        client.post("/agent/function-calls/start", headers=session, json=START)
        began = time.monotonic()
        assert client.post(f"{url}/complete", json={"agent_output": "done"}).status_code == 200
        assert time.monotonic() - began >= 0.15
        assert client.get(url).json()["stage"] == "sealed"


# --- the seal and late facts ---


def test_a_late_fact_marks_the_snapshot_incomplete_and_never_changes_it(client):
    url, session = _evaluation(client)
    client.post("/agent/function-calls", headers=session, json=FINISH)
    client.post(f"{url}/complete", json={"agent_output": "New York is 72°F and sunny"})
    graded = client.post(f"{url}/grade").json()
    sealed = client.get(url).json()
    assert "function_calls" not in sealed["evidence_status"]

    late = client.post("/agent/function-calls", headers=session, json={**FINISH, "call_id": "late"})
    assert late.status_code == 401
    assert late.json() == {
        "detail": "Invalid, expired, or closed evaluation session",
        "kind": "session_closed",
        "late_fact": True,
    }
    assert late.headers["WWW-Authenticate"] == "Bearer"
    observed = {"evidence_status": {"ocsf": "failed", "workdir": "failed"}}
    assert client.post(f"{url}/observations", json={"source": "openshell", "data": observed}).status_code == 409

    after = client.get(url).json()
    for field in ["function_calls", "observations", "environment", "turns", "agent_output", "utility", "security"]:
        assert after[field] == sealed[field], field
    assert after["grade_validity"] == graded["grade_validity"]
    assert after["evidence_status"] == {
        **graded["evidence_status"],
        "function_calls": "incomplete",
        "openshell.ocsf": "incomplete",
        "openshell.workdir": "incomplete",
    }
    assert [fact["event"] for fact in after["late_facts"]] == ["finish_call", "record_observations"]
    assert after["late_facts"][0]["payload"]["call_id"] == "late"
    regraded = client.post(f"{url}/grade").json()
    assert {key: regraded[key] for key in ["utility", "security", "grade_validity"]} == {
        key: graded[key] for key in ["utility", "security", "grade_validity"]
    }
    assert regraded["evidence_status"] == after["evidence_status"]


def test_idempotent_retries_after_the_seal_return_the_stored_records(client):
    url, session = _evaluation(client)
    started = client.post("/agent/function-calls/start", headers=session, json=START).json()
    finished = client.post("/agent/function-calls", headers=session, json=FINISH).json()
    client.post(f"{url}/complete", json={"agent_output": "done"})

    for send in [
        lambda: client.post("/agent/function-calls/start", headers=session, json=START),
        lambda: client.post(f"{url}/function-calls/start", json=START),
    ]:
        assert (send().status_code, send().json()) == (200, finished)
    for send in [
        lambda: client.post("/agent/function-calls", headers=session, json=FINISH),
        lambda: client.post(f"{url}/function-calls", json=FINISH),
    ]:
        assert (send().status_code, send().json()) == (200, finished)
    assert started["timestamp"] == finished["timestamp"]
    assert client.post(f"{url}/complete", json={"agent_output": "done"}).status_code == 200
    evaluation = client.get(url).json()
    assert (len(evaluation["function_calls"]), evaluation["late_facts"]) == (1, [])


def test_a_conflicting_write_after_the_seal_is_a_late_fact(client):
    """Only an exact replay is answered from the record after the seal; any other body is a late fact."""
    url, session = _evaluation(client)
    client.post("/agent/function-calls", headers=session, json=FINISH)
    client.post(f"{url}/complete", json={"agent_output": "done"})

    conflict = client.post(f"{url}/function-calls", json={**FINISH, "result": "other"})
    assert (conflict.status_code, conflict.json()) == (
        409,
        {"detail": "Evaluation already completed", "kind": "sealed", "late_fact": True},
    )
    conflict = client.post("/agent/function-calls/start", headers=session, json={**START, "args": {"city": "Paris"}})
    assert (conflict.status_code, conflict.json()["kind"], conflict.json()["late_fact"]) == (
        401,
        "session_closed",
        True,
    )
    evaluation = client.get(url).json()
    assert [call["result"] for call in evaluation["function_calls"]] == ["72°F, sunny"]
    assert [(fact["event"], fact["by"]) for fact in evaluation["late_facts"]] == [
        ("finish_call", "host"),
        ("start_call", "agent"),
    ]
    assert evaluation["evidence_status"]["function_calls"] == "incomplete"


def test_a_late_fact_retried_is_recorded_once(client):
    url, session = _evaluation(client)
    client.post(f"{url}/complete", json={"agent_output": "done"})
    for _ in range(3):
        assert client.post("/agent/function-calls", headers=session, json=FINISH).status_code == 401
        assert client.post(f"{url}/function-calls", json=FINISH).status_code == 409
    assert [fact["by"] for fact in client.get(url).json()["late_facts"]] == ["agent", "host"]


def test_turns_after_the_seal_are_late_facts(client):
    url, _ = _evaluation(client)
    client.post(f"{url}/turns", json={"message": "first"})
    client.post(f"{url}/complete", json={"agent_output": "x"})
    for path, body in [("turns", {"message": "y"}), ("turns/0/reply", {"reply": "y"})]:
        response = client.post(f"{url}/{path}", json=body)
        assert (response.status_code, response.json()["kind"]) == (409, "sealed")
    evaluation = client.get(url).json()
    assert [turn["reply"] for turn in evaluation["turns"]] == [None]
    assert evaluation["evidence_status"] == {"turns": "incomplete"}


# --- cancellation ---


def test_a_revoked_session_records_nothing(client):
    """The orchestrator revokes the session when it gives up on an evaluation: later callbacks leave no trace."""
    url, session = _evaluation(client)
    client.post("/agent/function-calls/start", headers=session, json=START)
    assert client.delete(f"{url}/session").status_code == 204
    for path, body in [("function-calls", FINISH), ("function-calls/start", {**START, "call_id": "c2"})]:
        response = client.post(f"/agent/{path}", headers=session, json=body)
        assert (response.status_code, response.json()["kind"], response.json()["late_fact"]) == (
            401,
            "invalid_session",
            False,
        )
    evaluation = client.get(url).json()
    assert evaluation["stage"] == "open"
    assert ([call["complete"] for call in evaluation["function_calls"]], evaluation["late_facts"]) == ([False], [])
