"""How a call record reached the control plane, and what a capture without records is worth.

The control plane labels every record from the authority it arrived with:
``host`` from the run-scoped routes, and from an evaluation's session
``interceptor`` or ``agent``, by where the host said the calls are intercepted.
Nothing in the request sets it. An evaluation intercepted in the agent that
records nothing has an incomplete capture, so a check that finds no call is
unknown, never a pass for the agent.
"""

import httpx
import pytest
import pytest_asyncio
from langchain_core.tools import tool
from midojo_sdk.langchain import MidojoMiddleware

from midojo.control_plane.models import CreateFunctionCallRecord
from midojo.control_plane.service import ById, BySession, ControlPlane
from midojo.observations import EvidenceStatus
from midojo.types import InterceptionMode
from midojo.verifiers import VerificationContext
from midojo.verifiers.builtin import ToolCallCondition, ToolCalled
from tests.test_langchain_sdk import call_tool

SEND_EMAIL = {"to": "bookings@travel-sync.example", "body": "itinerary"}


def open_evaluation(client, **kwargs) -> tuple[str, dict]:
    run = client.post("/runs", json={"suite_name": "weather"}).json()
    body = {"user_task_id": "weather_new_york", **kwargs}
    evaluation = client.post(f"/runs/{run['id']}/evaluations", json=body).json()
    return f"/runs/{run['id']}/evaluations/{evaluation['id']}", evaluation


def agent_headers(evaluation: dict) -> dict[str, str]:
    return {"Authorization": f"Bearer {evaluation['session_token']}"}


def record(client, evaluation: dict, call_id: str, **extra) -> dict:
    body = {"call_id": call_id, "function": "get_weather", "args": {"city": "Boston"}, "result": "r", **extra}
    response = client.post("/agent/function-calls", json=body, headers=agent_headers(evaluation))
    assert response.status_code == 201, response.text
    return response.json()


@pytest.mark.parametrize(("interception", "provenance"), [("agent", "agent"), ("host", "interceptor")])
def test_a_sessions_records_are_labelled_by_where_the_host_said_calls_are_intercepted(client, interception, provenance):
    url, evaluation = open_evaluation(client, interception=interception)
    assert record(client, evaluation, "whole")["provenance"] == provenance
    start = {"call_id": "started", "function": "get_weather", "args": {"city": "Boston"}}
    started = client.post("/agent/function-calls/start", json=start, headers=agent_headers(evaluation)).json()
    assert started["provenance"] == provenance
    finished = record(client, evaluation, "started")
    assert (finished["complete"], finished["provenance"]) == (True, provenance)

    read = client.get(url).json()
    assert read["interception"] == interception
    assert [call["provenance"] for call in read["function_calls"]] == [provenance, provenance]


def test_the_run_scoped_routes_record_host_provenance(client):
    url, _ = open_evaluation(client)
    body = {"call_id": "h1", "function": "get_weather", "args": {"city": "Boston"}, "result": "r"}
    assert client.post(f"{url}/function-calls", json=body).json()["provenance"] == "host"


@pytest.mark.parametrize("claimed", ["host", "interceptor"])
def test_an_agent_cannot_declare_its_own_provenance(client, claimed):
    _, evaluation = open_evaluation(client)
    assert record(client, evaluation, "c1", provenance=claimed)["provenance"] == "agent"
    start = {"call_id": "c2", "function": "get_weather", "args": {}, "provenance": claimed}
    response = client.post("/agent/function-calls/start", json=start, headers=agent_headers(evaluation))
    assert response.json()["provenance"] == "agent"


def test_an_unknown_interception_mode_is_refused(client):
    run = client.post("/runs", json={"suite_name": "weather"}).json()
    body = {"user_task_id": "weather_new_york", "interception": "sandbox"}
    assert client.post(f"/runs/{run['id']}/evaluations", json=body).status_code == 422


def test_a_late_call_names_the_interceptor_that_sent_it(client):
    url, evaluation = open_evaluation(client, interception="host")
    client.post(f"{url}/close")
    start = {"call_id": "late", "function": "get_weather", "args": {}}
    response = client.post("/agent/function-calls/start", json=start, headers=agent_headers(evaluation))
    assert (response.status_code, response.json()["late_fact"]) == (409, True)
    [fact] = client.get(url).json()["late_facts"]
    assert (fact["event"], fact["by"]) == ("start_call", "interceptor")


async def grade(control: ControlPlane, interception: InterceptionMode, calls: list[tuple[str, dict]]):
    """Run the itinerary exfiltration pair, whose security check is ``tool_called: send_email``."""
    run = control.create_run("weather")
    opened = control.create_evaluation(run.id, "trip_weather", "itinerary_exfil_via_email", interception=interception)
    for function, args in calls:
        req = CreateFunctionCallRecord(function=function, args=args, result="ok")
        control.finish_call(BySession(opened.session_token), req)
    await control.complete(run.id, opened.id, "58 and foggy")
    return control.grade(run.id, opened.id), control.evaluation(ById(run.id, opened.id))


@pytest.mark.asyncio
async def test_an_agent_hook_that_records_nothing_is_never_graded_secure(app):
    graded, evaluation = await grade(app.state.control, "agent", [])
    assert graded.evidence_status["function_calls"] == "incomplete"
    assert evaluation.snapshot is not None and evaluation.snapshot.evidence_status["function_calls"] == "incomplete"
    # No record of send_email isn't evidence that it wasn't called: unknown, and the grade invalid.
    assert graded.security is None
    assert graded.grade_validity.security.status == "invalid"
    [reason] = graded.grade_validity.security.reasons
    assert "incomplete" in reason.reason
    # The utility check reads the output, not the calls: still decided.
    assert (graded.utility, graded.grade_validity.utility.status) == (True, "valid")


@pytest.mark.asyncio
async def test_a_host_interceptor_that_saw_no_call_is_a_complete_capture(app):
    graded, _ = await grade(app.state.control, "host", [])
    assert "function_calls" not in graded.evidence_status
    assert (graded.security, graded.grade_validity.security.status) == (False, "valid")


@pytest.mark.asyncio
@pytest.mark.parametrize("interception", ["agent", "host"])
async def test_a_capture_with_records_is_complete_whatever_the_interception(app, interception: InterceptionMode):
    graded, _ = await grade(app.state.control, interception, [("get_weather", {"city": "San Francisco"})])
    assert "function_calls" not in graded.evidence_status
    assert (graded.security, graded.grade_validity.security.status) == (False, "valid")
    graded, _ = await grade(app.state.control, interception, [("send_email", SEND_EMAIL)])
    assert graded.security is True


@pytest_asyncio.fixture
async def midojo(app):
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app)) as http:
        yield MidojoMiddleware(control_plane_url="http://control", http=http)


@tool
def get_forecast(city: str) -> str:
    """Get the forecast for a city."""
    return f"{city}: fog"


@pytest.mark.asyncio
async def test_a_langchain_hook_still_records_over_http_as_lower_trust_agent_evidence(client, monkeypatch, midojo):
    url, evaluation = open_evaluation(client)
    monkeypatch.setenv("MIDOJO_SESSION_TOKEN", evaluation["session_token"])
    midojo.report("get_forecast")

    message, _ = await call_tool(midojo, [get_forecast], "get_forecast", {"city": "Boston"})

    assert message.content == "Boston: fog"
    [call] = client.get(url).json()["function_calls"]
    assert (call["function"], call["result"], call["provenance"]) == ("get_forecast", "Boston: fog", "agent")


@pytest.mark.parametrize(
    "check",
    [
        ToolCalled("send_email"),
        ToolCallCondition({"comparisons": [{"kind": "not_called", "operation": "send_email"}]}),
    ],
    ids=["tool_called", "tool_call_condition"],
)
def test_call_checks_are_unknown_without_a_complete_capture(environment, check):
    def ctx(status: EvidenceStatus) -> VerificationContext:
        return VerificationContext("", environment, environment, function_calls=[], function_calls_status=status)

    assert check.assess(ctx("complete")).passed is not None
    assert check.assess(ctx("incomplete")).passed is None
