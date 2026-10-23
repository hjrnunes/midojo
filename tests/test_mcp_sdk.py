"""Tests for MCP tool registration, forwarding and call recording."""

import asyncio
import json
import uuid

import httpx
import pytest
from fastmcp.exceptions import ToolError
from midojo_sdk import recorder as sdk_recorder
from midojo_sdk.client import AgentControlPlaneClient
from midojo_sdk.mcp import BLOCKED_REASON, MidojoMCP, ToolContext
from midojo_sdk.recorder import ControlPlaneRejectedError, RejectionKind


def test_midojo_mcp_tool_registration():
    mcp = MidojoMCP("test", control_plane_url="http://localhost:9999")

    @mcp.tool()
    async def my_tool(ctx: ToolContext, name: str) -> str:
        """A test tool."""
        return f"hello {name}"

    tools = asyncio.run(mcp._fastmcp.list_tools())
    assert len(tools) == 1
    tool = tools[0]
    assert tool.name == "my_tool"
    assert "name" in tool.parameters.get("properties", {})
    assert "ctx" not in tool.parameters.get("properties", {})


def test_midojo_mcp_tool_requires_ctx():
    mcp = MidojoMCP("test", control_plane_url="http://localhost:9999")

    with pytest.raises(TypeError, match="ToolContext"):

        @mcp.tool()
        async def bad_tool(name: str) -> str:
            """Missing ctx."""
            return name


@pytest.mark.asyncio
async def test_tool_context_forward_raises_without_upstream():
    async with httpx.AsyncClient() as http:
        ctx = ToolContext(AgentControlPlaneClient("http://localhost:9999", "unused", http=http))
        with pytest.raises(RuntimeError, match="no upstream is configured"):
            await ctx.forward("get_weather", {"city": "New York"})


@pytest.mark.asyncio
async def test_midojo_mcp_starts_and_finishes_each_call_with_a_uuid_call_id(monkeypatch):
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.path, json.loads(request.read())))
        return httpx.Response(201, json={})

    monkeypatch.setenv("MIDOJO_SESSION_TOKEN", "session")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        mcp = MidojoMCP("test", control_plane_url="http://control", http=http)

        @mcp.tool()
        async def my_tool(ctx: ToolContext, name: str) -> str:
            """A test tool."""
            return f"hello {name}"

        await mcp._fastmcp.call_tool("my_tool", {"name": "Ada"})
        await mcp._fastmcp.call_tool("my_tool", {"name": "Ada"})

    paths = [path for path, _ in sent]
    assert paths == ["/agent/function-calls/start", "/agent/function-calls"] * 2
    first_start, first_finish, second_start, _ = (body for _, body in sent)
    assert first_start == {
        "call_id": first_start["call_id"],
        "function": "my_tool",
        "args": {"name": "Ada"},
        "disposition": "simulated",
    }
    assert first_finish["call_id"] == first_start["call_id"]
    assert (first_finish["result"], first_finish["disposition"]) == ("hello Ada", "simulated")
    assert str(uuid.UUID(first_start["call_id"])) == first_start["call_id"]
    assert second_start["call_id"] != first_start["call_id"]


class FailingFinish:
    """Records through ``recorder``, except that every finish fails as an unreachable control plane would."""

    def __init__(self, recorder) -> None:
        self.recorder = recorder

    async def start_call(self, **kwargs) -> None:
        await self.recorder.start_call(**kwargs)

    async def finish_call(self, **kwargs) -> None:
        raise httpx.ConnectError("control plane unreachable")


def _new_evaluation(client) -> tuple[dict, dict]:
    run = client.post("/runs", json={"suite_name": "weather"}).json()
    evaluation = client.post(f"/runs/{run['id']}/evaluations", json={"user_task_id": "weather_new_york"}).json()
    return run, evaluation


@pytest.mark.asyncio
async def test_midojo_mcp_leaves_an_unfinished_call_when_the_finish_fails_over_http(client, monkeypatch, app):
    run, evaluation = _new_evaluation(client)
    monkeypatch.setenv("MIDOJO_SESSION_TOKEN", evaluation["session_token"])
    control = httpx.ASGITransport(app)

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/agent/function-calls":
            raise httpx.ConnectError("control plane unreachable")
        return await control.handle_async_request(request)

    monkeypatch.setattr(sdk_recorder, "RECORD_BACKOFF_SECONDS", 0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        mcp = MidojoMCP("test", control_plane_url="http://control", http=http)

        @mcp.tool()
        async def my_tool(ctx: ToolContext, name: str) -> str:
            """A test tool."""
            return f"hello {name}"

        with pytest.raises(Exception, match="control plane unreachable"):
            await mcp._fastmcp.call_tool("my_tool", {"name": "Ada"})

    [call] = client.get(f"/runs/{run['id']}/evaluations/{evaluation['id']}/function-calls").json()
    assert (call["function"], call["args"], call["complete"], call["result"]) == (
        "my_tool",
        {"name": "Ada"},
        False,
        None,
    )


@pytest.mark.asyncio
async def test_midojo_mcp_leaves_an_unfinished_call_when_the_finish_fails(client, backend):
    run, evaluation = _new_evaluation(client)
    agent = backend.agent(evaluation["session_token"])
    mcp = MidojoMCP("test", recorder=FailingFinish(agent), world=agent)

    @mcp.tool()
    async def my_tool(ctx: ToolContext, name: str) -> str:
        """A test tool."""
        return f"hello {name}"

    with pytest.raises(Exception, match="control plane unreachable"):
        await mcp._fastmcp.call_tool("my_tool", {"name": "Ada"})

    [call] = client.get(f"/runs/{run['id']}/evaluations/{evaluation['id']}/function-calls").json()
    assert (call["function"], call["args"], call["complete"], call["result"]) == (
        "my_tool",
        {"name": "Ada"},
        False,
        None,
    )


@pytest.mark.asyncio
async def test_midojo_mcp_leaves_a_cancelled_call_unfinished(client, backend):
    run, evaluation = _new_evaluation(client)
    running = asyncio.Event()
    mcp = backend.mcp("test", evaluation["session_token"])

    @mcp.tool()
    async def my_tool(ctx: ToolContext, name: str) -> str:
        """A test tool that never returns."""
        running.set()
        await asyncio.Event().wait()
        return name

    call = asyncio.create_task(mcp._fastmcp.call_tool("my_tool", {"name": "Ada"}))
    await running.wait()
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    [recorded] = client.get(f"/runs/{run['id']}/evaluations/{evaluation['id']}/function-calls").json()
    assert (recorded["function"], recorded["complete"], recorded["result"]) == ("my_tool", False, None)


@pytest.mark.asyncio
async def test_midojo_mcp_records_and_simulates_on_the_control_plane(client, backend):
    run, evaluation = _new_evaluation(client)
    mcp = backend.mcp("test", evaluation["session_token"])

    @mcp.tool()
    async def send_alert(ctx: ToolContext, city: str, message: str) -> str:
        """Send a weather alert."""
        assert city in await ctx.env("cities")
        await ctx.env_update("weather_alerts", [{"city": city, "message": message}])
        return "sent"

    mcp.block("delete_city")

    @mcp.tool()
    async def delete_city(ctx: ToolContext, city: str) -> str:
        """Delete a city."""
        raise AssertionError("a blocked tool never runs")

    await mcp._fastmcp.call_tool("send_alert", {"city": "New York", "message": "storm"})
    with pytest.raises(ToolError, match=BLOCKED_REASON):
        await mcp._fastmcp.call_tool("delete_city", {"city": "New York"})

    url = f"/runs/{run['id']}/evaluations/{evaluation['id']}"
    calls = client.get(f"{url}/function-calls").json()
    assert [(call["function"], call["result"], call["blocked"], call["complete"]) for call in calls] == [
        ("send_alert", "sent", False, True),
        ("delete_city", BLOCKED_REASON, True, True),
    ]
    assert client.get(f"{url}/environment").json()["weather_alerts"] == [{"city": "New York", "message": "storm"}]


@pytest.mark.asyncio
async def test_midojo_mcp_raises_the_same_rejection_on_both_implementations(client, backend):
    run, evaluation = _new_evaluation(client)
    mcp = backend.mcp("test", evaluation["session_token"])

    @mcp.tool()
    async def my_tool(ctx: ToolContext, name: str) -> str:
        """A test tool."""
        return name

    client.delete(f"/runs/{run['id']}/evaluations/{evaluation['id']}/session")
    with pytest.raises(ToolError) as refused:
        await mcp._fastmcp.call_tool("my_tool", {"name": "Ada"})
    # FastMCP gives the agent an error result, raised from the recorder's error.
    cause = refused.value.__cause__
    assert isinstance(cause, ControlPlaneRejectedError)
    assert cause.kind is RejectionKind.INVALID_SESSION


@pytest.mark.asyncio
async def test_tool_context_still_takes_the_deprecated_client_keyword(client, backend):
    _, evaluation = _new_evaluation(client)
    agent = backend.agent(evaluation["session_token"])

    with pytest.warns(DeprecationWarning, match=r"ToolContext\(client=\.\.\.\) is deprecated"):
        ctx = ToolContext(client=agent)

    assert "New York" in await ctx.env("cities")
    with pytest.raises(TypeError, match="world or client, not both"):
        ToolContext(agent, client=agent)
