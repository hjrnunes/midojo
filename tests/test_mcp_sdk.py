"""Tests for MCP tool registration, forwarding and call recording."""

import asyncio
import json
import uuid

import httpx
import pytest
from midojo_sdk import client as sdk_client
from midojo_sdk.client import AgentControlPlaneClient
from midojo_sdk.mcp import MidojoMCP, ToolContext


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
    assert first_start == {"call_id": first_start["call_id"], "function": "my_tool", "args": {"name": "Ada"}}
    assert first_finish["call_id"] == first_start["call_id"]
    assert first_finish["result"] == "hello Ada"
    assert str(uuid.UUID(first_start["call_id"])) == first_start["call_id"]
    assert second_start["call_id"] != first_start["call_id"]


@pytest.mark.asyncio
async def test_midojo_mcp_leaves_an_unfinished_call_when_the_finish_fails(client, monkeypatch, app):
    run = client.post("/runs", json={"suite_name": "weather"}).json()
    evaluation = client.post(f"/runs/{run['id']}/evaluations", json={"user_task_id": "weather_new_york"}).json()
    monkeypatch.setenv("MIDOJO_SESSION_TOKEN", evaluation["session_token"])
    control = httpx.ASGITransport(app)

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/agent/function-calls":
            raise httpx.ConnectError("control plane unreachable")
        return await control.handle_async_request(request)

    monkeypatch.setattr(sdk_client, "RECORD_BACKOFF_SECONDS", 0)
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
async def test_midojo_mcp_leaves_a_cancelled_call_unfinished(client, monkeypatch, app):
    run = client.post("/runs", json={"suite_name": "weather"}).json()
    evaluation = client.post(f"/runs/{run['id']}/evaluations", json={"user_task_id": "weather_new_york"}).json()
    monkeypatch.setenv("MIDOJO_SESSION_TOKEN", evaluation["session_token"])
    running = asyncio.Event()

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app)) as http:
        mcp = MidojoMCP("test", control_plane_url="http://control", http=http)

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
