"""Tests for MCP tool registration and forwarding independent of the control plane."""

import asyncio
import json
import uuid

import httpx
import pytest
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
async def test_midojo_mcp_records_with_a_uuid_call_id(monkeypatch):
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
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

    first, second = (body["call_id"] for body in sent)
    assert str(uuid.UUID(first)) == first
    assert first != second
