"""Staged closure, the seal and late facts, over the control plane's HTTP routes."""

import pytest
from fastapi.testclient import TestClient

# Writes the control plane accepted after completion, before closure was a lifecycle rule.
_PRE_FIX = pytest.mark.xfail(strict=True, reason="a completed evaluation still accepts this write")


def _completed(client: TestClient, agent_output: str = "New York is 72°F and sunny") -> str:
    run_id = client.post("/runs", json={"suite_name": "weather"}).json()["id"]
    eval_id = client.post(f"/runs/{run_id}/evaluations", json={"user_task_id": "weather_new_york"}).json()["id"]
    url = f"/runs/{run_id}/evaluations/{eval_id}"
    assert client.post(f"{url}/complete", json={"agent_output": agent_output}).status_code == 200
    return url


@_PRE_FIX
@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("function-calls", {"call_id": "late", "function": "get_weather", "args": {}, "result": "72°F"}),
        ("function-calls/start", {"call_id": "late", "function": "get_weather", "args": {}}),
    ],
)
def test_run_scoped_call_routes_refuse_a_completed_evaluation(client, path, body):
    url = _completed(client)
    assert client.post(f"{url}/{path}", json=body).status_code == 409
    assert client.get(url).json()["function_calls"] == []


@_PRE_FIX
def test_a_completed_evaluation_keeps_its_environment(client):
    url = _completed(client)
    environment = client.get(f"{url}/environment").json()
    changed = {**environment, "weather_alerts": [{"city": "Boston", "message": "late"}]}
    assert client.put(f"{url}/environment", json=changed).status_code == 409
    assert client.get(f"{url}/environment").json() == environment


@_PRE_FIX
def test_a_completed_evaluation_keeps_its_observations(client):
    url = _completed(client)
    observed = {"evidence_status": {"ocsf": "complete", "workdir": "failed"}}
    assert client.post(f"{url}/observations", json={"source": "openshell", "data": observed}).status_code == 409
    assert client.get(f"{url}/observations").json() == {}


@_PRE_FIX
def test_a_grade_cannot_be_replaced(client):
    url = _completed(client, "New York is 72°F and sunny")
    assert client.post(f"{url}/grade").json()["utility"] is True
    assert client.post(f"{url}/complete", json={"agent_output": "no idea"}).status_code == 409
    assert client.post(f"{url}/grade").json()["utility"] is True
    assert client.get(url).json()["agent_output"] == "New York is 72°F and sunny"
