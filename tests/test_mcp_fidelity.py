"""Proxy fidelity: with no injection, MidojoMCP mirroring an MCP server is indistinguishable from the server.

Each test runs sessions over the MCP wire format (see ``tests.scripted_mcp``)
against the scripted server directly and through a ``MidojoMCP`` that mirrors
it, and compares what the agent receives.
"""

import asyncio
import json
import typing
from collections.abc import Sequence
from typing import Any

import httpx
import pytest
from midojo_sdk.client import AgentControlPlaneClient
from midojo_sdk.mcp import BLOCKED_REASON, MidojoMCP, ToolContext, UpstreamClient
from midojo_sdk.recorder import Disposition, InjectionRecord

from midojo.types import CallDisposition, FunctionCallRecord
from tests.scripted_mcp import PROTOCOL_ERROR, SESSION_CALL, TOOLS, ScriptedMCP, wire_session

SEARCH = {
    "route": {"from": "LIS", "to": "FRA"},
    "cabin": "business",
    "passengers": [{"name": "Ada", "age": None}, {"name": "Grace", "age": 41, "loyalty": "gold"}],
    "max_price": None,
}
TRANSFER = {"from_account": "acc-1", "to_account": "acc-2", "amount": 250.0, "memo": None}

SESSION: list[dict[str, Any]] = [
    {"method": "tools/list", "params": {}},
    {"method": "tools/call", "params": {"name": "ping", "arguments": {}}},
    # A call to a tool without parameters may leave out the arguments.
    {"method": "tools/call", "params": {"name": "get_time"}},
    {"method": "tools/call", "params": {"name": "search_flights", "arguments": SEARCH}},
    {"method": "tools/call", "params": {"name": "get_forecast", "arguments": {"city": "Lisbon", "days": 2}}},
    {"method": "tools/call", "params": {"name": "transfer_funds", "arguments": TRANSFER}},
    {"method": "tools/call", "params": {"name": "close_account", "arguments": {"account": "acc-404"}}},
]
"""One scripted session: the tool list, then a call to every tool."""


class CapturingRecorder:
    """A recorder that keeps every start and finish it's given."""

    def __init__(self) -> None:
        self.starts: list[dict[str, Any]] = []
        self.finishes: list[dict[str, Any]] = []

    async def start_call(self, *, call_id: str, function: str, args: dict[str, Any], disposition: Disposition) -> None:
        self.starts.append({"call_id": call_id, "function": function, "args": args, "disposition": disposition})

    async def finish_call(
        self,
        *,
        call_id: str,
        function: str,
        args: dict[str, Any],
        result: str,
        disposition: Disposition,
        error: str | None = None,
        structured_content: dict[str, Any] | None = None,
        injection_outcomes: Sequence[InjectionRecord] = (),
    ) -> None:
        self.finishes.append(
            {
                "call_id": call_id,
                "function": function,
                "args": args,
                "result": result,
                "error": error,
                "disposition": disposition,
                "structured_content": structured_content,
                "injection_outcomes": list(injection_outcomes),
            }
        )


async def _mirror(*, hang: tuple[str, ...] = ()) -> tuple[MidojoMCP, ScriptedMCP, CapturingRecorder]:
    """A MidojoMCP for a fresh scripted server, with no injection, recording into a capturing recorder.

    It has a recorder and nothing else: no control plane and no environment.
    """
    upstream = ScriptedMCP(hang=hang)
    recorder = CapturingRecorder()
    mcp = MidojoMCP("proxy", upstream_transport=upstream.transport, recorder=recorder)
    return mcp, upstream, recorder


def _without_request_id(response: str, request_id: int) -> str:
    """The response with its JSON-RPC id replaced, and its other bytes as sent.

    The id is the only normalization: it numbers the requests of one session,
    so it identifies the request rather than describing the result. Both
    sessions send the same ids, so this checks the id is where it should be
    and that nothing else is touched.
    """
    prefix = f'{{"jsonrpc":"2.0","id":{request_id},'
    assert response.startswith(prefix), response
    return '{"jsonrpc":"2.0","id":"<request>",' + response.removeprefix(prefix)


async def _session(mcp: MidojoMCP, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [json.loads(response) for response in await wire_session(mcp._fastmcp._mcp_server, requests)]


def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"method": "tools/call", "params": {"name": name, "arguments": arguments}}


# --- parity ---


@pytest.mark.asyncio
async def test_a_mirrored_session_matches_the_direct_session_byte_for_byte():
    direct = ScriptedMCP()
    direct_responses = await wire_session(direct.server, SESSION)

    mcp, upstream, _ = await _mirror()
    assert await mcp.mirror_upstream() == [tool["name"] for tool in TOOLS]
    proxied_responses = await wire_session(mcp._fastmcp._mcp_server, SESSION)

    assert len(proxied_responses) == len(direct_responses) == len(SESSION)
    for request_id, (direct_response, proxied_response) in enumerate(
        zip(direct_responses, proxied_responses, strict=True), start=1
    ):
        assert _without_request_id(proxied_response, request_id) == _without_request_id(direct_response, request_id)
    # The upstream got every call, with its arguments as the agent sent them,
    # in one upstream session, as the direct session made them. Each result
    # numbers its call in its session, so the responses would differ too if
    # the mirror reconnected between calls.
    assert upstream.sessions == direct.sessions == [direct.calls]
    assert [name for name, _ in upstream.calls] == [tool["name"] for tool in TOOLS]
    # One connection listed the tools to mirror and one served the session; both are closed.
    assert (upstream.transport.opened, upstream.transport.open) == (2, 0)


@pytest.mark.asyncio
async def test_each_downstream_session_gets_its_own_upstream_session():
    mcp, upstream, _ = await _mirror()
    await mcp.mirror_upstream()
    connected = upstream.transport.opened

    first = await _session(mcp, [_call("ping", {}), _call("get_time", {})])
    second = await _session(mcp, [_call("ping", {})])
    # A session that makes no call never connects to the upstream.
    await _session(mcp, [{"method": "tools/list", "params": {}}])

    assert [response["result"]["_meta"][SESSION_CALL] for response in first + second] == [1, 2, 1]
    assert upstream.sessions == [[("ping", {}), ("get_time", {})], [("ping", {})]]
    assert (upstream.transport.opened - connected, upstream.transport.open) == (2, 0)


@pytest.mark.asyncio
async def test_the_session_covers_every_kind_of_tool_and_result():
    """The parity session isn't vacuous: it lists and gets what the scripted server is meant to send."""
    tools, *results = [json.loads(response) for response in await wire_session(ScriptedMCP().server, SESSION)]

    assert [tool["inputSchema"] for tool in tools["result"]["tools"][:2]] == [
        {"type": "object"},
        {"type": "object", "properties": {}, "additionalProperties": False},
    ]
    ping, get_time, search, forecast, transfer, close = results
    assert [result["result"]["_meta"][SESSION_CALL] for result in results[:-1]] == [1, 2, 3, 4, 5]
    assert ping["result"]["content"] == [{"type": "text", "text": "pong"}]
    assert get_time["result"]["content"] == [{"type": "text", "text": "2026-01-02T03:04:05Z"}]
    assert search["result"]["structuredContent"]["query"] == SEARCH
    assert [block["type"] for block in forecast["result"]["content"]] == ["text", "image", "resource"]
    assert transfer["result"]["isError"] is True
    assert close["error"] == PROTOCOL_ERROR.model_dump() | {"data": {"account": "acc-404", SESSION_CALL: 6}}


# --- dispositions ---


@pytest.mark.asyncio
async def test_every_forwarded_call_is_recorded_as_forwarded_with_what_the_agent_saw():
    mcp, _, recorder = await _mirror()
    await mcp.mirror_upstream()
    await _session(mcp, SESSION)

    calls = SESSION[1:]
    assert [start["function"] for start in recorder.starts] == [call["params"]["name"] for call in calls]
    assert {start["disposition"] for start in recorder.starts} == {"forwarded"}
    assert [finish["call_id"] for finish in recorder.finishes] == [start["call_id"] for start in recorder.starts]
    assert {finish["disposition"] for finish in recorder.finishes} == {"forwarded"}
    by_tool = {finish["function"]: finish for finish in recorder.finishes}
    assert by_tool["ping"] == by_tool["ping"] | {"args": {}, "result": "pong", "error": None}
    assert by_tool["get_forecast"]["result"].splitlines() == [
        "Lisbon: sunny, 24°C — light wind",
        '{"type":"image","data":"iVBORw0KGgo=","mimeType":"image/png"}',
        '{"type":"resource","resource":{"uri":"forecast://lisbon/today","mimeType":"application/json","text":"{}"}}',
    ]
    # The upstream's tool error and its protocol error are both errors in the record.
    assert by_tool["transfer_funds"]["error"] == by_tool["transfer_funds"]["result"] == "Insufficient funds in acc-1"
    assert by_tool["close_account"]["error"] == by_tool["close_account"]["result"] == PROTOCOL_ERROR.message


@pytest.mark.asyncio
async def test_a_blocked_call_gets_an_error_result_and_is_recorded_as_blocked():
    mcp, upstream, recorder = await _mirror()
    mcp.block("transfer_funds")
    await mcp.mirror_upstream()

    tools, blocked = await _session(mcp, [{"method": "tools/list", "params": {}}, _call("transfer_funds", TRANSFER)])

    # The agent still sees the tool as the upstream lists it.
    assert tools["result"]["tools"] == TOOLS
    assert blocked["result"] == {"content": [{"type": "text", "text": BLOCKED_REASON}], "isError": True}
    assert upstream.calls == []
    # A blocked call never runs, so it's recorded once, whole.
    assert recorder.starts == []
    [record] = recorder.finishes
    assert record | {"call_id": None} == {
        "call_id": None,
        "function": "transfer_funds",
        "args": TRANSFER,
        "result": BLOCKED_REASON,
        "error": None,
        "disposition": "blocked",
        "structured_content": None,
        "injection_outcomes": [],
    }


@pytest.mark.asyncio
async def test_a_denied_call_gets_its_reason_as_an_error_result_and_is_recorded_as_denied():
    mcp, upstream, recorder = await _mirror()

    @mcp.deny("transfer_funds")
    def over_limit(args: dict[str, Any]) -> str | None:
        return "Transfers over 100 need approval" if args["amount"] > 100 else None

    await mcp.mirror_upstream()
    small = TRANSFER | {"amount": 5.0}
    denied, forwarded = await _session(mcp, [_call("transfer_funds", TRANSFER), _call("transfer_funds", small)])

    assert denied["result"] == {
        "content": [{"type": "text", "text": "Transfers over 100 need approval"}],
        "isError": True,
    }
    # The check lets the small transfer through to the upstream, which answers it.
    assert forwarded["result"]["structuredContent"] == {"code": "insufficient_funds", "balance": 12.25}
    assert upstream.calls == [("transfer_funds", small)]
    assert [(finish["args"], finish["disposition"]) for finish in recorder.finishes] == [
        (TRANSFER, "denied"),
        (small, "forwarded"),
    ]
    assert [start["args"] for start in recorder.starts] == [small]


@pytest.mark.asyncio
async def test_a_tool_the_layer_defines_is_recorded_as_simulated_and_not_mirrored(monkeypatch):
    monkeypatch.setenv("MIDOJO_SESSION_TOKEN", "session")
    mcp, upstream, recorder = await _mirror()

    @mcp.tool()
    async def transfer_funds(ctx: ToolContext, from_account: str, to_account: str, amount: float) -> str:
        """Pretend to move money."""
        return f"Moved {amount}"

    mirrored = await mcp.mirror_upstream()
    [result] = await _session(mcp, [_call("transfer_funds", {"from_account": "a", "to_account": "b", "amount": 3})])

    assert "transfer_funds" not in mirrored
    assert result["result"]["content"] == [{"type": "text", "text": "Moved 3.0"}]
    assert upstream.calls == []
    [start] = recorder.starts
    assert start["disposition"] == "simulated"
    [finish] = recorder.finishes
    assert (finish["function"], finish["result"], finish["disposition"]) == ("transfer_funds", "Moved 3.0", "simulated")


@pytest.mark.asyncio
async def test_a_cancelled_simulated_call_stays_started_with_its_disposition(monkeypatch):
    monkeypatch.setenv("MIDOJO_SESSION_TOKEN", "session")
    mcp, _, recorder = await _mirror()
    running = asyncio.Event()

    @mcp.tool()
    async def send_alert(ctx: ToolContext, text: str) -> str:
        """Send an alert, and never return."""
        running.set()
        await asyncio.Event().wait()
        return "sent"

    call = asyncio.create_task(mcp._fastmcp.call_tool("send_alert", {"text": "hi"}))
    await running.wait()
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    [start] = recorder.starts
    assert (start["function"], start["args"], start["disposition"]) == ("send_alert", {"text": "hi"}, "simulated")
    assert recorder.finishes == []


@pytest.mark.asyncio
async def test_a_cancelled_forwarded_call_stays_started_with_its_disposition():
    mcp, upstream, recorder = await _mirror(hang=("get_forecast",))
    await mcp.mirror_upstream()

    call = asyncio.create_task(mcp._fastmcp.call_tool("get_forecast", {"city": "Lisbon"}))
    await upstream.hanging.wait()
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call

    [start] = recorder.starts
    assert (start["function"], start["args"], start["disposition"]) == ("get_forecast", {"city": "Lisbon"}, "forwarded")
    assert recorder.finishes == []
    assert upstream.transport.open == 0


@pytest.mark.asyncio
async def test_a_blocked_tool_the_layer_defines_never_runs(monkeypatch):
    monkeypatch.setenv("MIDOJO_SESSION_TOKEN", "session")
    mcp, _, recorder = await _mirror()
    ran = []

    @mcp.tool()
    async def send_alert(ctx: ToolContext, text: str) -> str:
        """Send an alert."""
        ran.append(text)
        return "sent"

    mcp.block("send_alert")
    [result] = await _session(mcp, [_call("send_alert", {"text": "hi"})])

    assert result["result"] == {"content": [{"type": "text", "text": BLOCKED_REASON}], "isError": True}
    assert ran == []
    assert [finish["disposition"] for finish in recorder.finishes] == ["blocked"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("disposition", "blocked"), [("forwarded", False), ("simulated", False), ("denied", True), ("blocked", True)]
)
async def test_the_http_recorder_sends_the_disposition_and_the_blocked_flag_it_implies(disposition, blocked):
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
        return httpx.Response(201, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        recorder = AgentControlPlaneClient("http://control", "session", http=http)
        if not blocked:
            await recorder.start_call(call_id="c1", function="f", args={}, disposition=disposition)
        await recorder.finish_call(call_id="c1", function="f", args={}, result="r", disposition=disposition)

    # The finish keeps the blocked flag alongside, for a control plane that predates the disposition field.
    start = {"call_id": "c1", "function": "f", "args": {}, "disposition": disposition}
    finish = {**start, "result": "r", "error": None, "blocked": blocked}
    assert sent == ([] if blocked else [start]) + [finish]


@pytest.mark.asyncio
async def test_a_refused_call_is_blocked_on_the_control_plane(client, backend):
    run = client.post("/runs", json={"suite_name": "weather"}).json()
    evaluation = client.post(f"/runs/{run['id']}/evaluations", json={"user_task_id": "weather_new_york"}).json()

    mcp = backend.mcp("proxy", evaluation["session_token"], upstream_transport=ScriptedMCP().transport)
    mcp.block("transfer_funds")
    await mcp.mirror_upstream()
    await _session(mcp, [_call("transfer_funds", TRANSFER), _call("ping", {})])

    calls = client.get(f"/runs/{run['id']}/evaluations/{evaluation['id']}/function-calls").json()
    assert [(call["function"], call["blocked"], call["result"]) for call in calls] == [
        ("transfer_funds", True, BLOCKED_REASON),
        ("ping", False, "pong"),
    ]


# --- what an annotation-based proxy gets wrong ---


@pytest.mark.asyncio
async def test_mirrored_schemas_are_the_upstreams_not_rebuilt_from_python_annotations():
    """An annotation-based proxy rebuilds each input schema from Python types: str, list, dict, ``X | None``."""
    mcp, _, _ = await _mirror()
    await mcp.mirror_upstream()
    [tools] = await _session(mcp, [{"method": "tools/list", "params": {}}])

    search = {tool["name"]: tool for tool in tools["result"]["tools"]}["search_flights"]
    assert search == TOOLS[2]
    schema = search["inputSchema"]
    assert schema["properties"]["cabin"]["enum"] == ["economy", "premium", "business", "first"]
    assert schema["properties"]["route"]["properties"]["from"] == {"$ref": "#/$defs/airport"}
    assert schema["$defs"]["airport"]["pattern"] == "^[A-Z]{3}$"
    passenger = schema["properties"]["passengers"]["items"]
    assert passenger["properties"]["age"]["type"] == ["integer", "null"]
    assert passenger["required"] == ["name"]
    assert schema["properties"]["max_price"] == {"anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"type": "null"}]}
    assert schema["required"] == ["route", "passengers"]
    assert search["outputSchema"] == TOOLS[2]["outputSchema"]
    assert (search["title"], search["annotations"], search["_meta"]) == (
        TOOLS[2]["title"],
        TOOLS[2]["annotations"],
        TOOLS[2]["_meta"],
    )


@pytest.mark.asyncio
async def test_a_tool_without_parameters_gets_no_dummy_parameter():
    """An annotation-based proxy gives a tool without parameters a ``_dummy`` one, to have a signature."""
    mcp, upstream, _ = await _mirror()
    await mcp.mirror_upstream()
    tools, *_ = await _session(mcp, [{"method": "tools/list", "params": {}}, _call("ping", {}), _call("get_time", {})])

    listed = {tool["name"]: tool["inputSchema"] for tool in tools["result"]["tools"]}
    assert listed["ping"] == {"type": "object"}
    assert listed["get_time"] == {"type": "object", "properties": {}, "additionalProperties": False}
    assert upstream.calls == [("ping", {}), ("get_time", {})]


@pytest.mark.asyncio
async def test_results_are_not_flattened_to_text():
    """A text-only proxy joins a result's text blocks into one string and drops everything else."""
    mcp, _, _ = await _mirror()
    await mcp.mirror_upstream()
    search, forecast, transfer = await _session(
        mcp,
        [
            _call("search_flights", SEARCH),
            _call("get_forecast", {"city": "Lisbon"}),
            _call("transfer_funds", TRANSFER),
        ],
    )

    assert search["result"]["structuredContent"]["flights"][1] == {
        "number": "LH 1167",
        "price": 211.0,
        "seats_left": None,
    }
    assert search["result"]["_meta"] == {"example.com/cache": "miss", SESSION_CALL: 1}
    assert search["result"]["content"][1] == {
        "type": "text",
        "text": "2 flights found",
        "annotations": {"audience": ["user"], "priority": 0.5},
    }
    assert [block["type"] for block in forecast["result"]["content"]] == ["text", "image", "resource"]
    assert forecast["result"]["content"][1] == {"type": "image", "data": "iVBORw0KGgo=", "mimeType": "image/png"}
    assert transfer["result"]["isError"] is True
    assert transfer["result"]["structuredContent"] == {"code": "insufficient_funds", "balance": 12.25}


@pytest.mark.asyncio
async def test_a_call_that_is_not_forwarded_gets_no_success_shaped_result():
    """A stubbing proxy answers a tool it doesn't forward with ``{"status": "ok"}``, and records that as its result."""
    mcp, upstream, recorder = await _mirror()
    mcp.block("close_account")
    await mcp.mirror_upstream()
    [result] = await _session(mcp, [_call("close_account", {"account": "acc-1"})])

    assert result["result"]["isError"] is True
    assert "structuredContent" not in result["result"]
    assert [block["text"] for block in result["result"]["content"]] == [BLOCKED_REASON]
    assert upstream.calls == []
    [record] = recorder.finishes
    assert (record["result"], record["disposition"]) == (BLOCKED_REASON, "blocked")
    assert '"status"' not in record["result"]


# --- the upstream client, the mirror and the record type ---


def test_upstream_client_still_takes_an_upstream_url():
    client = UpstreamClient(upstream_url="http://localhost:8081/mcp")
    assert (client.upstream_url, client.transport, client.target) == (
        "http://localhost:8081/mcp",
        None,
        client.upstream_url,
    )
    assert UpstreamClient("http://localhost:8081/mcp").upstream_url == "http://localhost:8081/mcp"


def test_midojo_mcp_still_takes_an_upstream_url():
    mcp = MidojoMCP("proxy", control_plane_url="http://control.invalid", upstream_url="http://localhost:8081/mcp")
    assert mcp._upstream is not None
    assert mcp._upstream.upstream_url == "http://localhost:8081/mcp"


@pytest.mark.parametrize("kwargs", [{}, {"upstream_url": "http://localhost:8081/mcp", "transport": object()}])
def test_upstream_client_needs_exactly_one_upstream(kwargs):
    with pytest.raises(ValueError, match="exactly one of upstream_url and transport"):
        UpstreamClient(**kwargs)


@pytest.mark.asyncio
async def test_mirror_upstream_raises_without_an_upstream():
    mcp = MidojoMCP("proxy", recorder=CapturingRecorder())
    with pytest.raises(RuntimeError, match="no upstream is configured"):
        await mcp.mirror_upstream()


def test_midojo_mcp_needs_a_control_plane_or_a_recorder():
    with pytest.raises(ValueError, match="control_plane_url or a recorder"):
        MidojoMCP("proxy", upstream_url="http://localhost:8081/mcp")


@pytest.mark.asyncio
async def test_a_simulated_tool_without_a_world_has_no_environment():
    mcp = MidojoMCP("proxy", recorder=(recorder := CapturingRecorder()))

    @mcp.tool()
    async def read_alerts(ctx: ToolContext) -> str:
        """Read the alerts."""
        return await ctx.env("weather_alerts")

    with pytest.raises(Exception, match="no WorldState"):
        await mcp._fastmcp.call_tool("read_alerts", {})
    [finish] = recorder.finishes
    assert (finish["function"], finish["disposition"]) == ("read_alerts", "simulated")
    assert "no WorldState" in (finish["error"] or "")


@pytest.mark.asyncio
async def test_a_simulated_tool_reads_and_changes_the_world_it_is_given(client, backend):
    run = client.post("/runs", json={"suite_name": "weather"}).json()
    evaluation = client.post(f"/runs/{run['id']}/evaluations", json={"user_task_id": "weather_new_york"}).json()
    mcp = backend.mcp("proxy", evaluation["session_token"], upstream_transport=ScriptedMCP().transport)

    @mcp.tool()
    async def send_alert(ctx: ToolContext, text: str) -> str:
        """Add an alert to the simulated environment."""
        revision, environment = await ctx.read()
        alerts = [*environment["weather_alerts"], {"city": "Lisbon", "message": text}]
        assert await ctx.env_update("weather_alerts", alerts, expected_revision=revision)
        return "sent"

    await mcp.mirror_upstream()
    await _session(mcp, [_call("send_alert", {"text": "storm"}), _call("ping", {})])

    url = f"/runs/{run['id']}/evaluations/{evaluation['id']}"
    assert client.get(f"{url}/environment").json()["weather_alerts"] == [{"city": "Lisbon", "message": "storm"}]
    calls = client.get(f"{url}/function-calls").json()
    assert [(call["function"], call["result"], call["blocked"]) for call in calls] == [
        ("send_alert", "sent", False),
        ("ping", "pong", False),
    ]


def test_the_sdk_and_the_record_name_the_same_dispositions():
    # The SDK doesn't import midojo, so it names the dispositions itself.
    assert typing.get_args(Disposition) == typing.get_args(CallDisposition)


def test_a_record_without_a_disposition_still_loads():
    record = FunctionCallRecord.model_validate(
        {"function": "f", "args": {}, "result": "r", "timestamp": "2026-01-01T00:00:00+00:00", "pre_environment": {}}
    )
    assert record.disposition is None


def test_a_blocked_record_without_a_disposition_still_loads():
    record = FunctionCallRecord.model_validate(
        {
            "function": "f",
            "args": {},
            "result": BLOCKED_REASON,
            "blocked": True,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "pre_environment": {},
        }
    )
    assert (record.blocked, record.disposition) == (True, None)


@pytest.mark.parametrize(
    ("disposition", "blocked"), [("forwarded", False), ("simulated", False), ("denied", True), ("blocked", True)]
)
def test_a_records_disposition_matches_its_blocked_flag(disposition, blocked):
    base = {"function": "f", "args": {}, "result": "r", "timestamp": "2026-01-01T00:00:00+00:00", "pre_environment": {}}
    assert FunctionCallRecord.model_validate(base | {"disposition": disposition, "blocked": blocked}).blocked is blocked
    with pytest.raises(ValueError, match=f"A {disposition} call must have blocked={blocked}"):
        FunctionCallRecord.model_validate(base | {"disposition": disposition, "blocked": not blocked})
