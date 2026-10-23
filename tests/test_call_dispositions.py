"""Each call's disposition, carried from the interception layer through the /agent routes into the stored record.

The rules are the lifecycle's (see ``midojo.control_plane.lifecycle``): a
disposition is optional, goes with the ``blocked`` flag, is decided when the
call starts, and a disposition left out matches any. The tests check them as
pure transitions, over both implementations of the SDK's seams, and over the
MCP wire: stateful and stateless streamable HTTP sessions, and a call the
agent cancels mid-call.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from midojo_sdk.client import AgentControlPlaneClient
from midojo_sdk.mcp import BLOCKED_REASON, MidojoMCP, ToolContext
from midojo_sdk.recorder import ControlPlaneRejectedError, RejectionKind

from midojo.control_plane.config import AppConfig
from midojo.control_plane.lifecycle import (
    HOST,
    Accepted,
    FinishCall,
    Late,
    Outcome,
    Rejected,
    Rejection,
    Seal,
    StartCall,
    transition,
)
from midojo.control_plane.main import create_app
from midojo.control_plane.models import CreateFunctionCallRecord, CreateFunctionCallStart, FunctionCallResponse
from midojo.control_plane.service import ById
from midojo.types import FunctionCallRecord
from tests.scripted_mcp import ScriptedMCP
from tests.test_lifecycle import AT, _apply, _collecting, _evaluation, _sealed

DISPOSITIONS = [("forwarded", False), ("simulated", False), ("denied", True), ("blocked", True)]


def _start(disposition: Any = None, call_id: str = "c1") -> StartCall:
    return StartCall(CreateFunctionCallStart(call_id=call_id, function="a", args={}, disposition=disposition), AT)


def _finish(disposition: Any = None, blocked: bool = False, call_id: str = "c1", result: str = "r") -> FinishCall:
    req = CreateFunctionCallRecord(
        call_id=call_id, function="a", args={}, result=result, blocked=blocked, disposition=disposition
    )
    return FinishCall(req, AT)


def _decide(evaluation, event) -> Outcome:
    return transition(evaluation, HOST, event)


def _rejected(evaluation, event) -> Rejection:
    outcome = _decide(evaluation, event)
    assert isinstance(outcome, Rejected), outcome
    return outcome.kind


# --- the rules, as pure transitions ---


@pytest.mark.parametrize("disposition", ["forwarded", "simulated"])
def test_a_start_carries_its_disposition_and_a_finish_without_one_keeps_it(disposition):
    evaluation = _apply(_evaluation(), _start(disposition))
    assert evaluation.function_calls[0].disposition == disposition
    finished = _apply(evaluation, _finish())
    assert (finished.function_calls[0].disposition, finished.function_calls[0].complete) == (disposition, True)


def test_a_finish_names_the_disposition_a_start_left_out():
    evaluation = _apply(_evaluation(), _start(), _finish("simulated"))
    assert evaluation.function_calls[0].disposition == "simulated"


def test_a_finish_naming_another_disposition_than_its_start_is_a_conflict():
    evaluation = _apply(_evaluation(), _start("forwarded"))
    assert _rejected(evaluation, _finish("simulated")) is Rejection.CALL_CONFLICT
    assert _rejected(evaluation, _finish("denied", blocked=True)) is Rejection.CALL_CONFLICT
    # A finish without a disposition keeps the start's, so it can't be blocked either.
    assert _rejected(evaluation, _finish(blocked=True)) is Rejection.CALL_CONFLICT


@pytest.mark.parametrize("disposition", ["denied", "blocked"])
def test_a_refused_call_is_never_started(disposition):
    evaluation = _evaluation()
    assert _rejected(evaluation, _start(disposition)) is Rejection.INVALID_DISPOSITION
    # Recorded whole, it's a blocked record with its disposition.
    [record] = _apply(evaluation, _finish(disposition, blocked=True)).function_calls
    assert (record.disposition, record.blocked, record.complete) == (disposition, True, True)


@pytest.mark.parametrize(("disposition", "blocked"), DISPOSITIONS)
def test_a_disposition_that_does_not_go_with_blocked_is_invalid_at_any_stage(disposition, blocked):
    started = _apply(_evaluation(), _start("forwarded"))
    for evaluation in [_evaluation(), started, _collecting(started), _sealed(started)]:
        outcome = _decide(evaluation, _finish(disposition, blocked=not blocked))
        # Refused outright: nothing is recorded, not even a late fact.
        assert isinstance(outcome, Rejected) and outcome.kind is Rejection.INVALID_DISPOSITION, outcome


def test_a_replay_without_a_disposition_replays_a_record_with_one():
    evaluation = _apply(_evaluation(), _start("forwarded"), _finish("forwarded"))
    for event in [_start(), _start("forwarded"), _finish(), _finish("forwarded")]:
        outcome = _decide(evaluation, event)
        assert isinstance(outcome, Accepted) and outcome.evaluation == evaluation
        assert outcome.result == (evaluation.function_calls[0], False)


def test_a_replay_with_a_disposition_replays_a_record_without_one_and_changes_nothing():
    evaluation = _apply(_evaluation(), _start(), _finish())
    outcome = _decide(evaluation, _finish("forwarded"))
    assert isinstance(outcome, Accepted) and outcome.evaluation == evaluation
    assert evaluation.function_calls[0].disposition is None


def test_a_replay_naming_another_disposition_is_a_conflict():
    unfinished = _apply(_evaluation(), _start("forwarded"))
    finished = _apply(unfinished, _finish())
    assert _rejected(unfinished, _start("simulated")) is Rejection.CALL_CONFLICT
    assert _rejected(finished, _start("simulated")) is Rejection.CALL_CONFLICT
    assert _rejected(finished, _finish("simulated")) is Rejection.CALL_CONFLICT


def test_after_the_seal_a_disposition_conflict_is_a_late_fact_that_carries_it():
    sealed = _sealed(_apply(_evaluation(), _start("forwarded"), _finish()))
    outcome = _decide(sealed, _finish("simulated"))
    assert isinstance(outcome, Late) and outcome.rejected.kind is Rejection.SEALED
    [fact] = outcome.evaluation.late_facts
    assert (fact.event, fact.payload["disposition"]) == ("finish_call", "simulated")


def test_the_snapshot_keeps_each_records_disposition():
    evaluation = _apply(
        _evaluation(),
        _start("forwarded", call_id="a"),
        _finish("forwarded", call_id="a"),
        _start("simulated", call_id="b"),
        _finish("denied", blocked=True, call_id="c"),
        _finish(call_id="d"),
    )
    sealed = _apply(_collecting(evaluation), Seal("out", AT))
    assert sealed.snapshot is not None
    assert [(call.call_id, call.disposition, call.complete) for call in sealed.snapshot.function_calls] == [
        ("a", "forwarded", True),
        ("b", "simulated", False),
        ("c", "denied", True),
        ("d", None, True),
    ]


# --- what older clients send, and records made before the field ---


def test_request_bodies_without_a_disposition_still_validate():
    start = CreateFunctionCallStart.model_validate({"call_id": "c1", "function": "f", "args": {}})
    finish = CreateFunctionCallRecord.model_validate(
        {"call_id": "c1", "function": "f", "args": {}, "result": "r", "blocked": True}
    )
    assert (start.disposition, finish.disposition, finish.blocked) == (None, None, True)


def test_a_record_without_a_disposition_still_loads_and_responds_with_none():
    record = FunctionCallRecord.model_validate_json(
        '{"call_id": "c1", "function": "f", "args": {}, "result": "r", "timestamp": "2026-01-01T00:00:00+00:00"}'
    )
    assert record.disposition is None
    assert FunctionCallResponse.model_validate(record).model_dump()["disposition"] is None
    # And a record with one round-trips through JSON.
    with_one = record.model_copy(update={"disposition": "forwarded"})
    assert FunctionCallRecord.model_validate_json(with_one.model_dump_json()).disposition == "forwarded"


def test_legacy_bodies_are_recorded_without_a_disposition_over_http(client, app):
    run_id, eval_id, token = new_evaluation(client)
    headers = {"Authorization": f"Bearer {token}"}

    started = client.post(
        "/agent/function-calls/start", headers=headers, json={"call_id": "c1", "function": "f", "args": {}}
    )
    recorded = client.post(
        "/agent/function-calls",
        headers=headers,
        json={"call_id": "c2", "function": "f", "args": {}, "result": BLOCKED_REASON, "blocked": True},
    )

    assert (started.status_code, started.json()["disposition"]) == (201, None)
    assert (recorded.status_code, recorded.json()["disposition"], recorded.json()["blocked"]) == (201, None, True)
    assert [(call.call_id, call.disposition) for call in stored(app, run_id, eval_id)] == [("c1", None), ("c2", None)]


# --- both implementations of the SDK's seams ---


def new_evaluation(client) -> tuple[str, str, str]:
    run = client.post("/runs", json={"suite_name": "weather"}).json()
    evaluation = client.post(f"/runs/{run['id']}/evaluations", json={"user_task_id": "weather_new_york"}).json()
    return run["id"], evaluation["id"], evaluation["session_token"]


def stored(app, run_id: str, eval_id: str) -> tuple[FunctionCallRecord, ...]:
    """The evaluation's stored records, as the control plane's writer keeps them."""
    return app.state.control.evaluation(ById(run_id, eval_id)).function_calls


@pytest.mark.asyncio
@pytest.mark.parametrize(("disposition", "blocked"), DISPOSITIONS)
async def test_a_start_stores_its_disposition_on_both_implementations(client, app, backend, disposition, blocked):
    run_id, eval_id, token = new_evaluation(client)
    agent = backend.agent(token)

    if blocked:
        with pytest.raises(ControlPlaneRejectedError) as refused:
            await agent.start_call(call_id="c1", function="f", args={"x": 1}, disposition=disposition)
        assert (refused.value.kind, refused.value.late_fact) == (RejectionKind.INVALID_DISPOSITION, False)
        assert stored(app, run_id, eval_id) == ()
        return
    await agent.start_call(call_id="c1", function="f", args={"x": 1}, disposition=disposition)

    [record] = stored(app, run_id, eval_id)
    assert (record.call_id, record.disposition, record.blocked, record.complete) == ("c1", disposition, False, False)
    [call] = client.get(f"/runs/{run_id}/evaluations/{eval_id}/function-calls").json()
    assert (call["disposition"], call["complete"]) == (disposition, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(("disposition", "blocked"), DISPOSITIONS)
async def test_a_finish_stores_its_disposition_on_both_implementations(client, app, backend, disposition, blocked):
    run_id, eval_id, token = new_evaluation(client)
    agent = backend.agent(token)

    # A forwarded or simulated call is started first and finished in place; a refused one is recorded whole.
    if not blocked:
        await agent.start_call(call_id="c1", function="f", args={"x": 1}, disposition=disposition)
    await agent.finish_call(call_id="c1", function="f", args={"x": 1}, result="r", disposition=disposition)
    # A resend is a replay: still one record.
    await agent.finish_call(call_id="c1", function="f", args={"x": 1}, result="r", disposition=disposition)

    [record] = stored(app, run_id, eval_id)
    assert (record.call_id, record.disposition, record.blocked, record.complete) == ("c1", disposition, blocked, True)
    [call] = client.get(f"/runs/{run_id}/evaluations/{eval_id}/function-calls").json()
    assert (call["disposition"], call["blocked"]) == (disposition, blocked)


@pytest.mark.asyncio
async def test_a_finish_naming_another_disposition_is_the_same_conflict_on_both_implementations(client, app, backend):
    run_id, eval_id, token = new_evaluation(client)
    agent = backend.agent(token)
    await agent.start_call(call_id="c1", function="f", args={}, disposition="forwarded")

    with pytest.raises(ControlPlaneRejectedError) as refused:
        await agent.finish_call(call_id="c1", function="f", args={}, result="r", disposition="simulated")

    assert refused.value.kind is RejectionKind.CALL_CONFLICT
    [record] = stored(app, run_id, eval_id)
    assert (record.disposition, record.complete) == ("forwarded", False)


def test_the_http_edge_answers_an_invalid_disposition_with_422_and_its_kind(client, app):
    run_id, eval_id, token = new_evaluation(client)
    headers = {"Authorization": f"Bearer {token}"}

    start = client.post(
        "/agent/function-calls/start",
        headers=headers,
        json={"call_id": "c1", "function": "f", "args": {}, "disposition": "denied"},
    )
    finish = client.post(
        "/agent/function-calls",
        headers=headers,
        json={"call_id": "c2", "function": "f", "args": {}, "result": "r", "blocked": False, "disposition": "blocked"},
    )
    unknown = client.post(
        "/agent/function-calls/start",
        headers=headers,
        json={"call_id": "c3", "function": "f", "args": {}, "disposition": "allowed"},
    )

    for response in [start, finish]:
        assert response.status_code == 422
        assert (response.json()["kind"], response.json()["late_fact"]) == ("invalid_disposition", False)
    # A disposition that isn't one at all is a malformed body: a 422 without a kind.
    assert unknown.status_code == 422 and "kind" not in unknown.json()
    assert stored(app, run_id, eval_id) == ()


def test_the_run_scoped_call_routes_carry_the_disposition(client, app):
    run_id, eval_id, _ = new_evaluation(client)
    url = f"/runs/{run_id}/evaluations/{eval_id}/function-calls"

    started = client.post(
        f"{url}/start", json={"call_id": "c1", "function": "f", "args": {}, "disposition": "simulated"}
    )
    finished = client.post(url, json={"call_id": "c1", "function": "f", "args": {}, "result": "r"})
    refused = client.post(
        url,
        json={"call_id": "c2", "function": "f", "args": {}, "result": "no", "blocked": True, "disposition": "denied"},
    )

    assert [response.status_code for response in [started, finished, refused]] == [201, 201, 201]
    assert [response.json()["disposition"] for response in [started, finished, refused]] == [
        "simulated",
        "simulated",
        "denied",
    ]
    assert client.get(f"{url}/0").json()["disposition"] == "simulated"
    assert [call.disposition for call in stored(app, run_id, eval_id)] == ["simulated", "denied"]


@pytest.mark.asyncio
async def test_the_sealed_snapshot_and_the_evaluation_response_keep_the_disposition(client, app, backend):
    run_id, eval_id, token = new_evaluation(client)
    agent = backend.agent(token)
    await agent.start_call(call_id="c1", function="f", args={}, disposition="forwarded")
    await agent.finish_call(call_id="c1", function="f", args={}, result="r", disposition="forwarded")
    await agent.finish_call(call_id="c2", function="f", args={}, result="no", disposition="denied")

    assert client.post(f"/runs/{run_id}/evaluations/{eval_id}/complete", json={"agent_output": "done"}).is_success
    assert client.post(f"/runs/{run_id}/evaluations/{eval_id}/grade").is_success

    snapshot = app.state.control.evaluation(ById(run_id, eval_id)).snapshot
    assert [call.disposition for call in snapshot.function_calls] == ["forwarded", "denied"]
    response = client.get(f"/runs/{run_id}/evaluations/{eval_id}").json()
    assert [call["disposition"] for call in response["function_calls"]] == ["forwarded", "denied"]


# --- over the MCP wire ---

ACCEPT = "application/json, text/event-stream"
TRANSFER = {"from_account": "acc-1", "to_account": "acc-2", "amount": 250.0, "memo": None}
DENIAL = "Closing accounts is not allowed"


def _message(response: httpx.Response) -> dict[str, Any]:
    """The JSON-RPC message of a streamable HTTP response, sent as JSON or as one SSE event."""
    assert response.status_code == 200, response.text
    if response.headers["content-type"].startswith("application/json"):
        return response.json()
    [data] = [line.removeprefix("data: ") for line in response.text.splitlines() if line.startswith("data: ")]
    return json.loads(data)


def _call(request_id: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    }


class WireSession:
    """One agent's streamable HTTP MCP session, sending its evaluation session on every request."""

    def __init__(self, http: httpx.AsyncClient, token: str) -> None:
        self._http = http
        self._headers = {"Accept": ACCEPT, "X-Midojo-Session": token}

    async def open(self) -> None:
        """Start a stateful MCP session: the server keeps it under the ID it answers with."""
        params = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "agent", "version": "1"}}
        response = await self._post({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": params})
        _message(response)
        self._headers["mcp-session-id"] = response.headers["mcp-session-id"]
        assert (await self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})).status_code == 202

    async def call(self, request_id: int, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return _message(await self._post(_call(request_id, name, arguments)))

    async def cancel(self, request_id: int) -> None:
        params = {"requestId": request_id, "reason": "the agent gave up"}
        response = await self._post({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": params})
        assert response.status_code == 202

    async def _post(self, body: dict[str, Any]) -> httpx.Response:
        return await self._http.post("/mcp", headers=self._headers, json=body)


class Proxy:
    """A MidojoMCP mirroring a scripted server, recording on ``app``'s control plane over HTTP.

    ``send_alert`` is simulated, and waits for its text's gate when it has one;
    ``transfer_funds`` is blocked and ``close_account`` denied; the scripted
    server's other tools are forwarded.
    """

    def __init__(self, app, upstream: ScriptedMCP) -> None:
        self.control = httpx.AsyncClient(transport=httpx.ASGITransport(app))
        self.mcp = MidojoMCP(
            "proxy", control_plane_url="http://control", http=self.control, upstream_transport=upstream.transport
        )
        self.gates: dict[str, asyncio.Event] = {}
        self.running: dict[str, asyncio.Event] = {}

        @self.mcp.tool()
        async def send_alert(ctx: ToolContext, text: str) -> str:
            """Send an alert."""
            self.running.setdefault(text, asyncio.Event()).set()
            if text in self.gates:
                await self.gates[text].wait()
            return f"sent {text}"

        self.mcp.block("transfer_funds")
        self.mcp.deny("close_account")(lambda args: DENIAL)

    async def serve(self, *, stateless: bool = False) -> Any:
        """The proxy's streamable HTTP app at ``/mcp``, mirroring the upstream."""
        await self.mcp.mirror_upstream()
        return self.mcp._fastmcp.http_app(path="/mcp", stateless_http=stateless)


def _records(app, run_id: str, eval_id: str) -> list[tuple[str, str | None, bool, bool]]:
    return [(call.function, call.disposition, call.blocked, call.complete) for call in stored(app, run_id, eval_id)]


@pytest.mark.asyncio
async def test_concurrent_stateful_sessions_record_each_call_in_its_own_evaluation(client, app):
    run_a, eval_a, token_a = new_evaluation(client)
    run_b, eval_b, token_b = new_evaluation(client)
    proxy = Proxy(app, ScriptedMCP())
    proxy.gates["A"] = asyncio.Event()
    mcp_app = await proxy.serve()

    async with mcp_app.router.lifespan_context(mcp_app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(mcp_app), base_url="http://mcp") as http:
            a, b = WireSession(http, token_a), WireSession(http, token_b)
            await asyncio.gather(a.open(), b.open())

            # A's simulated call is in flight while B's session makes every kind of call.
            a_alert = asyncio.create_task(a.call(1, "send_alert", {"text": "A"}))
            await proxy.running.setdefault("A", asyncio.Event()).wait()
            b_results = [
                await b.call(1, "ping", {}),
                await b.call(2, "transfer_funds", TRANSFER),
                await b.call(3, "close_account", {"account": "acc-1"}),
                await b.call(4, "send_alert", {"text": "B"}),
            ]
            assert _records(app, run_a, eval_a) == [("send_alert", "simulated", False, False)]
            proxy.gates["A"].set()
            a_results = [await a_alert, await a.call(2, "ping", {})]

    await proxy.control.aclose()
    assert [result["result"]["content"][0]["text"] for result in a_results] == ["sent A", "pong"]
    assert [result["result"].get("isError", False) for result in b_results] == [False, True, True, False]
    assert _records(app, run_a, eval_a) == [
        ("send_alert", "simulated", False, True),
        ("ping", "forwarded", False, True),
    ]
    assert _records(app, run_b, eval_b) == [
        ("ping", "forwarded", False, True),
        ("transfer_funds", "blocked", True, True),
        ("close_account", "denied", True, True),
        ("send_alert", "simulated", False, True),
    ]
    assert [call.result for call in stored(app, run_b, eval_b)] == ["pong", BLOCKED_REASON, DENIAL, "sent B"]


@pytest.mark.asyncio
async def test_stateless_requests_record_each_call_in_the_evaluation_it_names(client, app):
    evaluations = [new_evaluation(client) for _ in range(2)]
    proxy = Proxy(app, ScriptedMCP())
    mcp_app = await proxy.serve(stateless=True)
    calls = [
        ("ping", {}),
        ("transfer_funds", TRANSFER),
        ("close_account", {"account": "acc-1"}),
        ("send_alert", {"text": "hi"}),
    ]

    async with mcp_app.router.lifespan_context(mcp_app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(mcp_app), base_url="http://mcp") as http:
            # No MCP session: each request stands alone, with its evaluation session in its own header.
            responses = await asyncio.gather(
                *[
                    http.post(
                        "/mcp",
                        headers={"Accept": ACCEPT, "X-Midojo-Session": token},
                        json=_call(request_id, name, arguments),
                    )
                    for _, _, token in evaluations
                    for request_id, (name, arguments) in enumerate(calls, start=1)
                ]
            )

    await proxy.control.aclose()
    assert all("result" in _message(response) for response in responses)
    for run_id, eval_id, _ in evaluations:
        assert sorted(_records(app, run_id, eval_id)) == [
            ("close_account", "denied", True, True),
            ("ping", "forwarded", False, True),
            ("send_alert", "simulated", False, True),
            ("transfer_funds", "blocked", True, True),
        ]


@pytest.fixture
def quick_app(suite):
    """A control plane whose closure doesn't wait for unfinished calls."""
    return create_app({"weather": suite}, config=AppConfig(closure_wait_seconds=0))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "arguments", "disposition"),
    [("send_alert", {"text": "hang"}, "simulated"), ("get_forecast", {"city": "Lisbon"}, "forwarded")],
)
async def test_a_call_cancelled_over_the_wire_stays_unfinished_with_its_disposition(
    quick_app, tool, arguments, disposition
):
    app = quick_app
    run_id, eval_id, token = new_evaluation(TestClient(app))
    upstream = ScriptedMCP(hang=("get_forecast",))
    proxy = Proxy(app, upstream)
    proxy.gates["hang"] = asyncio.Event()  # never set: the simulated call never returns
    mcp_app = await proxy.serve()

    async with mcp_app.router.lifespan_context(mcp_app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(mcp_app), base_url="http://mcp") as http:
            session = WireSession(http, token)
            await session.open()
            call = asyncio.create_task(session.call(1, tool, arguments))
            if disposition == "simulated":
                await proxy.running.setdefault("hang", asyncio.Event()).wait()
            else:
                await upstream.hanging.wait()
            await session.cancel(1)
            cancelled = await asyncio.wait_for(call, timeout=5)
            # The session goes on after the cancelled call.
            after = await session.call(2, "ping", {})

    await proxy.control.aclose()
    assert cancelled["error"]["message"] == "Request cancelled"
    assert after["result"]["content"][0]["text"] == "pong"
    assert _records(app, run_id, eval_id) == [(tool, disposition, False, False), ("ping", "forwarded", False, True)]
    assert stored(app, run_id, eval_id)[0].result is None

    # Sealed, the call stays unfinished in the snapshot, with its disposition.
    await app.state.control.complete(run_id, eval_id, "done")
    snapshot = app.state.control.evaluation(ById(run_id, eval_id)).snapshot
    assert [(call.function, call.disposition, call.complete) for call in snapshot.function_calls] == [
        (tool, disposition, False),
        ("ping", "forwarded", True),
    ]


@pytest.mark.asyncio
async def test_the_sdk_client_keeps_the_blocked_flag_for_an_older_control_plane():
    # An older control plane ignores the disposition it doesn't know, and still reads blocked.
    sent: list[dict[str, Any]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
        return httpx.Response(201, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        agent = AgentControlPlaneClient("http://control", "session", http=http)
        await agent.finish_call(call_id="c1", function="f", args={}, result="no", disposition="denied")

    [body] = sent
    assert (body["disposition"], body["blocked"]) == ("denied", True)
    legacy = CreateFunctionCallRecord.model_validate(
        {key: value for key, value in body.items() if key != "disposition"}
    )
    assert legacy.blocked is True
