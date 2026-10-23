"""A scripted MCP server, and a wire-level session against any MCP server in the same process.

The server is built on the MCP SDK's low-level ``Server`` with its own
request handlers, so every tool definition and result is the literal JSON
below: no framework derives a schema from annotations or reshapes a result.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
from fastmcp.client.transports import ClientTransport
from mcp import ClientSession
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_client_server_memory_streams
from mcp.shared.message import SessionMessage
from mcp.types import (
    LATEST_PROTOCOL_VERSION,
    CallToolRequest,
    CallToolResult,
    ErrorData,
    JSONRPCMessage,
    JSONRPCNotification,
    JSONRPCRequest,
    ListToolsRequest,
    ListToolsResult,
    ServerResult,
    Tool,
)

TOOLS: list[dict[str, Any]] = [
    {
        "name": "ping",
        "description": "Check the service is up. Takes no parameters at all.",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "get_time",
        "title": "Current time",
        "description": "The service's clock, as ISO 8601.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        "annotations": {"readOnlyHint": True, "idempotentHint": True},
    },
    {
        "name": "search_flights",
        "title": "Search flights",
        "description": "Find flights on a route.\n\nPrices are per passenger.",
        "inputSchema": {
            "type": "object",
            "$defs": {
                "airport": {"type": "string", "pattern": "^[A-Z]{3}$", "description": "IATA airport code"},
            },
            "properties": {
                "route": {
                    "type": "object",
                    "properties": {"from": {"$ref": "#/$defs/airport"}, "to": {"$ref": "#/$defs/airport"}},
                    "required": ["from", "to"],
                    "additionalProperties": False,
                },
                "cabin": {"type": "string", "enum": ["economy", "premium", "business", "first"], "default": "economy"},
                "passengers": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": ["integer", "null"], "minimum": 0},
                            "loyalty": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                        },
                        "required": ["name"],
                    },
                },
                "max_price": {"anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}]},
                "flexible_days": {"type": "integer", "enum": [0, 1, 3]},
            },
            "required": ["route", "passengers"],
            "additionalProperties": False,
        },
        "outputSchema": {
            "type": "object",
            "properties": {
                "currency": {"type": "string", "enum": ["EUR", "USD"]},
                "flights": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "number": {"type": "string"},
                            "price": {"type": "number"},
                            "seats_left": {"type": ["integer", "null"]},
                        },
                        "required": ["number", "price"],
                    },
                },
                "query": {"type": "object"},
            },
            "required": ["currency", "flights", "query"],
        },
        "annotations": {"title": "Flight search", "readOnlyHint": True, "openWorldHint": True},
        "_meta": {"example.com/owner": "flights", "example.com/rate_limit": {"per_minute": 30}},
    },
    {
        "name": "get_forecast",
        "description": "The forecast for a city, as text and a chart.",
        "inputSchema": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "days": {"type": "integer", "minimum": 1, "maximum": 7}},
            "required": ["city"],
        },
    },
    {
        "name": "transfer_funds",
        "description": "Move money between accounts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "from_account": {"type": "string"},
                "to_account": {"type": "string"},
                "amount": {"type": "number"},
                "memo": {"type": ["string", "null"]},
            },
            "required": ["from_account", "to_account", "amount"],
        },
        "annotations": {"destructiveHint": True},
    },
    {
        "name": "close_account",
        "description": "Close an account for good.",
        "inputSchema": {"type": "object", "properties": {"account": {"type": "string"}}, "required": ["account"]},
        "annotations": {"destructiveHint": True, "idempotentHint": False},
    },
]
"""The tools as the server lists them, in this order."""

PROTOCOL_ERROR = ErrorData(code=-32602, message="Unknown account: acc-404", data={"account": "acc-404"})
"""The JSON-RPC error ``close_account`` answers with."""


def _result(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The result the server sends for a call, as literal JSON. Arguments are echoed as received."""
    if name == "ping":
        return {"content": [{"type": "text", "text": "pong"}], "isError": False}
    if name == "get_time":
        return {"content": [{"type": "text", "text": "2026-01-02T03:04:05Z"}], "isError": False}
    if name == "search_flights":
        structured = {
            "currency": "EUR",
            "flights": [
                {"number": "TP 1234", "price": 129.5, "seats_left": 4},
                {"number": "LH 1167", "price": 211.0, "seats_left": None},
            ],
            "query": arguments,
        }
        return {
            "content": [
                {"type": "text", "text": json.dumps(structured)},
                {"type": "text", "text": "2 flights found", "annotations": {"audience": ["user"], "priority": 0.5}},
            ],
            "structuredContent": structured,
            "isError": False,
            "_meta": {"example.com/cache": "miss"},
        }
    if name == "get_forecast":
        return {
            "content": [
                {"type": "text", "text": f"{arguments['city']}: sunny, 24°C — light wind"},
                {"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"},
                {
                    "type": "resource",
                    "resource": {"uri": "forecast://lisbon/today", "mimeType": "application/json", "text": "{}"},
                },
            ],
            "isError": False,
        }
    if name == "transfer_funds":
        return {
            "content": [{"type": "text", "text": "Insufficient funds in " + arguments["from_account"]}],
            "structuredContent": {"code": "insufficient_funds", "balance": 12.25},
            "isError": True,
        }
    raise AssertionError(f"no scripted result for {name}")


class ScriptedMCP:
    """An MCP server that lists ``TOOLS`` and answers each call with its scripted result.

    ``close_account`` answers with the JSON-RPC error ``PROTOCOL_ERROR``.
    Every call it receives is kept in ``calls`` as ``(name, arguments)``.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.server: Server = Server("scripted")
        self.server.request_handlers[ListToolsRequest] = self._list_tools
        self.server.request_handlers[CallToolRequest] = self._call_tool

    async def _list_tools(self, request: ListToolsRequest) -> ServerResult:
        return ServerResult(ListToolsResult(tools=[Tool.model_validate(tool) for tool in TOOLS]))

    async def _call_tool(self, request: CallToolRequest) -> ServerResult:
        name, arguments = request.params.name, request.params.arguments or {}
        self.calls.append((name, arguments))
        if name == "close_account":
            raise McpError(PROTOCOL_ERROR)
        return ServerResult(CallToolResult.model_validate(_result(name, arguments)))


class InProcessTransport(ClientTransport):
    """A fastmcp client transport to a low-level MCP ``Server`` in the same process."""

    def __init__(self, server: Server) -> None:
        self.server = server

    @contextlib.asynccontextmanager
    async def connect_session(self, **session_kwargs: Any) -> AsyncIterator[ClientSession]:
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(self.server.run, *server_streams, self.server.create_initialization_options())
                try:
                    async with ClientSession(*client_streams, **session_kwargs) as session:
                        yield session
                finally:
                    tasks.cancel_scope.cancel()


INITIALIZE = {
    "method": "initialize",
    "params": {
        "protocolVersion": LATEST_PROTOCOL_VERSION,
        "capabilities": {},
        "clientInfo": {"name": "parity-test", "version": "0"},
    },
}


async def wire_session(server: Server, requests: list[dict[str, Any]]) -> list[str]:
    """Run one session against ``server`` and return each request's response as sent on the wire.

    The session initializes, then sends ``requests`` (``method`` and
    ``params``) one at a time with ids 1, 2, ... Each response is its JSON-RPC
    message serialized the way the MCP SDK's stdio and HTTP transports write
    it: ``model_dump_json(by_alias=True, exclude_none=True)``.
    """
    async with create_client_server_memory_streams() as ((client_read, client_write), server_streams):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(server.run, *server_streams, server.create_initialization_options())

            async def send(request_id: int, request: dict[str, Any]) -> str:
                message = JSONRPCRequest(jsonrpc="2.0", id=request_id, **request)
                await client_write.send(SessionMessage(JSONRPCMessage(message)))
                while True:
                    received = await client_read.receive()
                    if isinstance(received, Exception):
                        raise received
                    if getattr(received.message.root, "id", None) == request_id:
                        return received.message.model_dump_json(by_alias=True, exclude_none=True)

            await send(0, INITIALIZE)
            initialized = JSONRPCNotification(jsonrpc="2.0", method="notifications/initialized")
            await client_write.send(SessionMessage(JSONRPCMessage(initialized)))
            responses = [await send(request_id, request) for request_id, request in enumerate(requests, start=1)]
            tasks.cancel_scope.cancel()
    return responses
