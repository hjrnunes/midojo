"""The in-process implementation of the SDK's seams makes no HTTP request, and the test backend proves it."""

import httpx
import pytest
from midojo_sdk.mcp import MidojoMCP, ToolContext

from midojo.control_plane.inprocess import InProcessAgentClient
from midojo.control_plane.service import BySession
from tests.backends import Backend
from tests.scripted_mcp import ScriptedMCP, wire_session


@pytest.fixture
def no_http(monkeypatch):
    """Fail on any request any httpx client sends, synchronous or not."""

    def refuse(self, request, *args, **kwargs):
        pytest.fail(f"An HTTP request was sent: {request.method} {request.url}")

    async def refuse_async(self, request, *args, **kwargs):
        refuse(self, request)

    monkeypatch.setattr(httpx.Client, "send", refuse)
    monkeypatch.setattr(httpx.AsyncClient, "send", refuse_async)


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
