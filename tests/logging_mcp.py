"""An MCP server that logs exactly what it served, for checking delivery facts against a record of its own.

The server is built on the MCP SDK's low-level ``Server``, so each result is
the literal JSON below. Every call it answers is appended to ``served`` as the
tool, the arguments it received and the result as it goes on the wire
(``model_dump(by_alias=True, exclude_none=True)``, the way the SDK's
transports write it). Nothing here imports MiDojo: the log is independent of
the interceptor whose delivery facts it checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from mcp.server.lowlevel import Server
from mcp.types import CallToolRequest, CallToolResult, ListToolsRequest, ListToolsResult, ServerResult, Tool

from tests.scripted_mcp import InProcessTransport

NOTES = {"notes": [{"id": "NOTE-1", "body": "first"}, {"id": "NOTE-2", "body": "second"}]}

TOOLS: list[dict[str, Any]] = [
    {"name": "list_notes", "description": "Every note, as indented JSON.", "inputSchema": {"type": "object"}},
    {
        "name": "list_notes_summary",
        "description": "Every note, with a text summary.",
        "inputSchema": {"type": "object"},
    },
    {
        "name": "lookup_order",
        "description": "One order, as compact JSON.",
        "inputSchema": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    },
    {
        "name": "get_referral",
        "description": "A patient's referral; its id is in the result only.",
        "inputSchema": {"type": "object", "properties": {"patient_id": {"type": "string"}}, "required": ["patient_id"]},
    },
    {
        "name": "read_page",
        "description": "A page of text, with no structured content.",
        "inputSchema": {"type": "object", "properties": {"page": {"type": "integer"}}, "required": ["page"]},
    },
]


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def scripted_result(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """The result the server sends for a call, before the wire dump."""
    if name == "list_notes":
        return {"content": [_text(json.dumps(NOTES, indent=2))], "structuredContent": NOTES, "isError": False}
    if name == "list_notes_summary":
        return {
            "content": [_text("2 notes"), _text("NOTE-1, NOTE-2")],
            "structuredContent": NOTES,
            "isError": False,
        }
    if name == "lookup_order":
        order = {"item": {"order_id": arguments["order_id"], "status": "shipped", "note": "Leave at door."}}
        return {"content": [_text(json.dumps(order, separators=(",", ":")))], "structuredContent": order}
    if name == "get_referral":
        referral_id = "REF-1" if arguments["patient_id"] == "P1" else "REF-2"
        referral = {"referral_id": referral_id, "reason": "eye check"}
        return {"content": [_text(json.dumps(referral, separators=(",", ":")))], "structuredContent": referral}
    if name == "read_page":
        return {"content": [_text(f"Page {arguments['page']}: opening hours are 9-17.")]}
    raise AssertionError(f"no scripted result for {name}")


@dataclass(frozen=True)
class Served:
    """One call as the server saw and answered it."""

    tool: str
    arguments: dict[str, Any]
    result: dict[str, Any]


class LoggingMCP:
    """A low-level MCP server answering each call with its scripted result, and logging it in ``served``."""

    def __init__(self) -> None:
        self.served: list[Served] = []
        self.server: Server = Server("logging")
        self.server.request_handlers[ListToolsRequest] = self._list_tools
        self.server.request_handlers[CallToolRequest] = self._call_tool
        self.transport = InProcessTransport(self.server)

    async def _list_tools(self, request: ListToolsRequest) -> ServerResult:
        return ServerResult(ListToolsResult(tools=[Tool.model_validate(tool) for tool in TOOLS]))

    async def _call_tool(self, request: CallToolRequest) -> ServerResult:
        name, arguments = request.params.name, dict(request.params.arguments or {})
        result = CallToolResult.model_validate(scripted_result(name, arguments))
        wire = result.model_dump(mode="json", by_alias=True, exclude_none=True)
        self.served.append(Served(name, json.loads(json.dumps(arguments)), wire))
        return ServerResult(result)
