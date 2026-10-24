"""The in-process implementations of the SDK's seams and the control client make no HTTP request."""

import pytest
from midojo_sdk.mcp import MidojoMCP, ToolContext

from midojo.control_plane.inprocess import InProcessAgentClient
from midojo.control_plane.inprocess_control import InProcessControlClient
from midojo.control_plane.lifecycle import RejectedError, Rejection
from midojo.control_plane.service import BySession
from midojo.runtimes.openshell import OpenShellObservations
from tests.backends import Backend
from tests.scripted_mcp import ScriptedMCP, wire_session


@pytest.mark.asyncio
async def test_the_in_process_client_records_and_simulates_without_http(app, no_http):
    control = app.state.control
    created = control.create_evaluation(control.create_run("weather").id, "weather_new_york")
    agent = InProcessAgentClient(control, created.session_token)
    mcp = MidojoMCP("proxy", recorder=agent, world=agent, upstream_transport=ScriptedMCP().transport)

    @mcp.tool()
    async def send_alert(ctx: ToolContext, message: str) -> str:
        """Add an alert."""
        revision, _ = await ctx.read()
        alerts = [{"city": "Lisbon", "message": message}]
        assert await ctx.env_update("weather_alerts", alerts, expected_revision=revision)
        return "sent"

    await mcp.mirror_upstream()
    await wire_session(
        mcp._fastmcp._mcp_server,
        [
            {"method": "tools/call", "params": {"name": "send_alert", "arguments": {"message": "storm"}}},
            {"method": "tools/call", "params": {"name": "ping", "arguments": {}}},
        ],
    )

    evaluation = control.evaluation(BySession(created.session_token))
    assert [(call.function, call.result, call.complete) for call in evaluation.function_calls] == [
        ("send_alert", "sent", True),
        ("ping", "pong", True),
    ]
    assert evaluation.environment.model_dump()["weather_alerts"] == [{"city": "Lisbon", "message": "storm"}]


@pytest.mark.asyncio
async def test_the_in_process_backend_fails_on_any_http_request(app, monkeypatch):
    backend = Backend("in-process", app, monkeypatch)
    with pytest.raises(pytest.fail.Exception, match="sent an HTTP request"):
        await backend.http().get("http://control/agent/environment")
    assert [request.url.path for request in backend.forbidden_requests] == ["/agent/environment"]
    with pytest.raises(AssertionError, match="sent HTTP requests"):
        await backend.aclose()


@pytest.mark.asyncio
async def test_the_in_process_control_client_refuses_as_the_routes_do(app, no_http):
    control = InProcessControlClient(app.state.control)
    with pytest.raises(LookupError, match="Unknown suite"):
        await control.create_run("nope")
    with pytest.raises(RejectedError) as unknown:
        await control.create_evaluation("no-run", "weather_new_york", None, {})
    assert unknown.value.kind is Rejection.UNKNOWN_RUN
    run = await control.create_run("weather")
    with pytest.raises(RejectedError) as unknown_task:
        await control.create_evaluation(run.id, "no-task", None, {})
    assert unknown_task.value.kind is Rejection.UNKNOWN_TASK
    created = await control.create_evaluation(run.id, "weather_new_york", None, {})
    with pytest.raises(RejectedError) as not_sealed:
        await control.grade_evaluation(run.id, created.id)
    assert not_sealed.value.kind is Rejection.NOT_SEALED
    with pytest.raises(ValueError, match="Unknown observation source"):
        await control.record_observations(run.id, created.id, "nope", OpenShellObservations())

    await control.record_observations(run.id, created.id, "openshell", OpenShellObservations())
    await control.complete_evaluation(run.id, created.id, "72°F")
    grade = await control.grade_evaluation(run.id, created.id)
    evaluation = await control.evaluation(run.id, created.id)
    assert (evaluation.stage, evaluation.graded, evaluation.utility) == ("sealed", True, grade.utility)
    assert evaluation.observations == {"openshell": OpenShellObservations().model_dump(mode="json")}
