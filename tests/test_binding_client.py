"""The driver's WebSocket adapter over real localhost sockets (bead asago-v9md.3, part 2).

The shipping adapter lives in the SDK's ``mcp`` extra: a driver dials out to
the interceptor service, binds one evaluation, and serves the service's frames
through the v9md.1 ``LinkDispatcher``. These tests use the real service app on
a loopback port and a scripted upstream, and read what the dispatcher recorded
into the driver's control plane.

What the groups show:

- the URL: an http(s) or ws(s) service URL becomes the binding stream's
  ws(s) URL in one place, with the path kept.
- the binding: the adapter answers ``bound``, serves frames, carries the
  release's outcomes, and refuses a wrong credential or a refused bind without
  publishing an endpoint.
- the liveness: pings keep an idle binding, silence ends it as a loss, and an
  answer past the service's deadline ends it as a loss with the capture
  incomplete and no second forward.
- the transport: the client's ``max_size`` is explicit and a frame between
  1 MiB and 16 MiB passes in both directions.
- parity: a binding made through the console script records the same facts as
  the in-process interceptor for the same scripted evaluation.

No model, and no network beyond loopback.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import NoReturn

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from midojo_sdk.binding import (
    ANSWER_DEADLINE_SECONDS,
    HEARTBEAT_EXPIRY_SECONDS,
    HEARTBEAT_INTERVAL_SECONDS,
    MAX_FRAME_BYTES,
    WEBSOCKETS_LOGGER,
    Binding,
    BindingRefusedError,
    binding_url,
)
from midojo_sdk.injection import Channel, InjectionInstruction, InjectionPlan, InjectionTarget

from midojo.control_plane.inprocess_control import InProcessControlClient
from midojo.delivery import delivery_facts, exposure
from midojo.interception_host import HostInterceptor
from midojo.interception_service import (
    DEFAULT_ANSWER_DEADLINE_SECONDS,
    DEFAULT_HEARTBEAT_EXPIRY_SECONDS,
    LOSS_CLOSE_CODE,
    RELEASE_CLOSE_CODE,
)
from tests.test_interception_service import (
    CREDENTIAL,
    PLAN,
    _spec,
    call,
    refused_http,
    scripted_upstream,
    serving,
)

EXPECTED = f"41F, rain\n{PLAN.instructions[0].payload}"


# --- the URL ---


@pytest.mark.parametrize(
    ("service_url", "expected"),
    [
        ("http://127.0.0.1:8767", "ws://127.0.0.1:8767/v1/bindings"),
        ("http://127.0.0.1:8767/", "ws://127.0.0.1:8767/v1/bindings"),
        ("https://interceptor.example", "wss://interceptor.example/v1/bindings"),
        ("https://interceptor.example/", "wss://interceptor.example/v1/bindings"),
        ("ws://127.0.0.1:8767", "ws://127.0.0.1:8767/v1/bindings"),
        ("wss://interceptor.example", "wss://interceptor.example/v1/bindings"),
        ("wss://interceptor.example/", "wss://interceptor.example/v1/bindings"),
        ("http://127.0.0.1:8767/base", "ws://127.0.0.1:8767/base/v1/bindings"),
        ("http://127.0.0.1:8767/v1/bindings", "ws://127.0.0.1:8767/v1/bindings"),
    ],
)
def test_a_service_url_becomes_the_binding_stream_url(service_url, expected):
    """One place maps http(s) to ws(s) and names the binding stream's path."""
    assert binding_url(service_url) == expected


@pytest.mark.parametrize(
    "service_url",
    [
        "",
        "127.0.0.1:8767",
        "ftp://interceptor.example",
        "http:///v1/bindings",
        "http://host?credential=x",
        "http://host#x",
    ],
)
def test_a_service_url_that_is_not_a_web_socket_base_is_refused(service_url):
    """A URL with no host or no http(s)/ws(s) scheme is a settings error, not a dialled socket."""
    with pytest.raises(ValueError):
        binding_url(service_url)


# --- the binding ---


@pytest.mark.asyncio
async def test_the_adapter_binds_serves_and_releases_one_evaluation():
    """The adapter answers ``bound``, serves frames through the dispatcher, and carries the release's outcomes."""
    async with serving() as remote:
        opened, dispatcher = remote.open()
        binding = await Binding.open(remote.url, CREDENTIAL, upstream_url=remote.upstream_url, plan=PLAN)
        serving_frames = asyncio.create_task(dispatcher.serve(binding))
        try:
            assert binding.bound.tool_token
            assert binding.bound.tools == ("get_weather", "list_alerts")
            assert binding.bound.tool_url.startswith(f"{remote.url}/b/")
            assert binding.websocket.subprotocol == "midojo-binding.v1"
            assert binding.websocket.protocol.max_message_size == MAX_FRAME_BYTES
            assert remote.service.published == (binding.bound.tool_url.split("/")[4],)

            assert await call(binding.bound.tool_url, binding.bound.tool_token) == EXPECTED
            [record] = remote.evaluation(opened).function_calls
            assert (record.function, record.result, record.complete) == ("get_weather", EXPECTED, True)

            await binding.unbind()
            await asyncio.wait_for(binding.wait_unbound(), 5)
            [outcome] = binding.injection_outcomes
            assert (outcome["tool"], outcome["status"]) == ("get_weather", "applied")
            await asyncio.wait_for(binding.wait_closed(), 5)
            assert binding.close_code == RELEASE_CLOSE_CODE
            unmounted = await refused_http(binding.bound.tool_url, binding.bound.tool_token)
            assert (unmounted.status_code, unmounted.headers["www-authenticate"]) == (401, "Bearer")
        finally:
            await binding.aclose()
            with contextlib.suppress(asyncio.CancelledError):
                serving_frames.cancel()
                await serving_frames


@pytest.mark.asyncio
async def test_the_adapter_refuses_a_wrong_credential_and_a_refused_bind():
    """A wrong bind credential is an ``unauthorized`` refusal; a refused bind names the service's own kind."""
    async with serving() as remote:
        with pytest.raises(BindingRefusedError) as unauthorized:
            await Binding.open(remote.url, "wrong", upstream_url=remote.upstream_url)
        assert unauthorized.value.kind == "unauthorized"
        assert "403" in unauthorized.value.detail

        with pytest.raises(BindingRefusedError) as refused:
            await Binding.open(remote.url, CREDENTIAL, upstream_url="http://127.0.0.1:9/mcp")
        assert refused.value.kind == "upstream_unavailable"
        assert remote.service.published == ()


@pytest.mark.asyncio
async def test_the_adapter_keeps_the_bind_credential_out_of_the_websockets_debug_log():
    """websockets logs every handshake header at DEBUG; the adapter redacts the bearer credential in them."""
    lines: list[str] = []

    class Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    handler = Collect()
    level = WEBSOCKETS_LOGGER.level
    WEBSOCKETS_LOGGER.setLevel(logging.DEBUG)
    WEBSOCKETS_LOGGER.addHandler(handler)
    try:
        async with serving() as remote:
            binding = await Binding.open(remote.url, CREDENTIAL, upstream_url=remote.upstream_url)
            await binding.aclose()
    finally:
        WEBSOCKETS_LOGGER.removeHandler(handler)
        WEBSOCKETS_LOGGER.setLevel(level)

    headers = [line for line in lines if "authorization" in line.lower()]
    assert headers, "the handshake headers are logged, so the redaction has something to hide"
    assert [line for line in lines if CREDENTIAL in line] == []
    assert all("<redacted>" in line for line in headers)


@pytest.mark.asyncio
async def test_frames_reach_the_dispatcher_in_id_order():
    """Two calls in flight cross one socket as consecutive ids, and the dispatcher applies each once."""
    async with serving() as remote:
        opened, dispatcher = remote.open()
        binding = await Binding.open(remote.url, CREDENTIAL, upstream_url=remote.upstream_url, plan=PLAN)
        serving_frames = asyncio.create_task(dispatcher.serve(binding))
        try:
            async with Client(binding.bound.tool_url, auth=binding.bound.tool_token) as target:
                await asyncio.gather(
                    target.call_tool("get_weather", {"city": "Boston"}),
                    target.call_tool("get_weather", {"city": "New York"}),
                )
            assert [application.id for application in dispatcher.applications] == [0, 1, 2, 3]
            assert [application.op for application in dispatcher.applications] == [
                "start_call",
                "start_call",
                "finish_call_detailed",
                "finish_call_detailed",
            ]
            assert len(remote.evaluation(opened).function_calls) == 2
        finally:
            await binding.aclose()
            with contextlib.suppress(asyncio.CancelledError):
                serving_frames.cancel()
                await serving_frames


# --- the transport ---


@pytest.mark.asyncio
async def test_a_frame_between_one_and_sixteen_mebibytes_passes_in_both_directions():
    """The bind carries a >1 MiB plan and the finish a >1 MiB result, both past the client's 1 MiB default."""
    payload = "x" * (2 * 1024 * 1024)
    plan = InjectionPlan(
        (
            InjectionInstruction(
                channel=Channel.TOOL_OUTPUT,
                payload=payload,
                target=InjectionTarget(tool="get_weather"),
                probe_key="w:big",
            ),
        )
    )
    # The frames the adapter carries: the bind's plan, and the injected result the service sends back.
    assert len(json.dumps(plan.to_data()).encode()) > 1024 * 1024
    assert len(f"41F, rain\n{payload}".encode()) > 1024 * 1024

    async with serving() as remote:
        opened, dispatcher = remote.open()
        binding = await Binding.open(remote.url, CREDENTIAL, upstream_url=remote.upstream_url, plan=plan)
        serving_frames = asyncio.create_task(dispatcher.serve(binding))
        try:
            assert await call(binding.bound.tool_url, binding.bound.tool_token) == f"41F, rain\n{payload}"
            [record] = remote.evaluation(opened).function_calls
            assert (record.function, record.complete) == ("get_weather", True)
            assert record.result == f"41F, rain\n{payload}"
        finally:
            await binding.aclose()
            with contextlib.suppress(asyncio.CancelledError):
                serving_frames.cancel()
                await serving_frames


# --- the liveness ---


def test_the_three_liveness_values_are_the_wire_contract():
    """The driver's interval sits well inside the service's expiry, and the deadline inside both."""
    assert (HEARTBEAT_INTERVAL_SECONDS, HEARTBEAT_EXPIRY_SECONDS, ANSWER_DEADLINE_SECONDS) == (5.0, 15.0, 10.0)
    assert HEARTBEAT_INTERVAL_SECONDS * 2 < HEARTBEAT_EXPIRY_SECONDS
    # The service enforces the same expiry and deadline, from the one SDK contract.
    assert (DEFAULT_ANSWER_DEADLINE_SECONDS, DEFAULT_HEARTBEAT_EXPIRY_SECONDS) == (
        ANSWER_DEADLINE_SECONDS,
        HEARTBEAT_EXPIRY_SECONDS,
    )


@pytest.mark.asyncio
async def test_the_dispatchers_heartbeat_keeps_an_idle_binding_alive():
    """The dispatcher pings while it waits for frames, so an idle binding outlives several expiries.

    The binding is opened with no heartbeat of its own: the pings the service
    sees are the dispatcher's, through the same duplex that answers the
    service's frames.
    """
    async with serving(heartbeat_expiry=0.4) as remote:
        opened, dispatcher = remote.open()
        binding = await Binding.open(remote.url, CREDENTIAL, upstream_url=remote.upstream_url)
        serving_frames = asyncio.create_task(dispatcher.serve(binding, heartbeat_interval=0.05))
        try:
            # Three expiries with no call: only the dispatcher's pings keep the binding alive.
            await asyncio.sleep(0.9)
            assert not binding.closed.is_set()
            assert remote.service.published
            assert await call(binding.bound.tool_url, binding.bound.tool_token) == "41F, rain"
        finally:
            await binding.aclose()
            with contextlib.suppress(asyncio.CancelledError):
                serving_frames.cancel()
                await serving_frames


class Stalled:
    """The duplex of a dispatcher that stopped: nothing it reads ever arrives and nothing it sends leaves."""

    def __init__(self, binding: Binding) -> None:
        self.binding = binding

    async def send(self, data: str) -> None:
        await asyncio.Event().wait()

    async def receive(self) -> NoReturn:
        await asyncio.Event().wait()
        raise AssertionError("a stalled dispatcher never receives a frame")

    def close(self) -> None:
        self.binding.close()


@pytest.mark.asyncio
async def test_a_stalled_dispatcher_lets_the_binding_expire():
    """The ping is the dispatcher's work, not a task of its own: a dispatcher stuck in its wait lets the service expire the binding."""
    async with serving(heartbeat_expiry=0.4) as remote:
        opened, dispatcher = remote.open()
        binding = await Binding.open(remote.url, CREDENTIAL, upstream_url=remote.upstream_url)
        stalled = Stalled(binding)
        serving_frames = asyncio.create_task(dispatcher.serve(stalled, heartbeat_interval=0.05))
        try:
            assert remote.service.published == (binding.bound.tool_url.split("/")[4],)
            await asyncio.wait_for(binding.wait_closed(), 5)
            assert binding.close_code == LOSS_CLOSE_CODE
            assert "no heartbeat within 0.4s" in (binding.close_reason or ""), binding.close_reason
            assert remote.service.published == ()
            assert (await refused_http(binding.bound.tool_url, binding.bound.tool_token)).status_code == 401
        finally:
            await binding.aclose()
            with contextlib.suppress(asyncio.CancelledError):
                serving_frames.cancel()
                await serving_frames


class SlowAnswers:
    """The duplex the dispatcher serves, with a switchable delay before each answer leaves."""

    def __init__(self, binding: Binding) -> None:
        self.binding = binding
        self.delay = 0.0

    async def send(self, data: str) -> None:
        await asyncio.sleep(self.delay)
        await self.binding.send(data)

    async def receive(self) -> str:
        return await self.binding.receive()

    def close(self) -> None:
        self.binding.close()


@pytest.mark.asyncio
async def test_an_answer_past_the_service_deadline_ends_the_capture_incomplete_without_a_second_forward():
    """A driver that stops answering costs the binding: the seal says the capture is incomplete, and nothing re-forwards."""
    async with serving(answer_deadline=0.2) as remote:
        opened, dispatcher = remote.open()
        binding = await Binding.open(remote.url, CREDENTIAL, upstream_url=remote.upstream_url, heartbeat_interval=None)
        answers = SlowAnswers(binding)
        serving_frames = asyncio.create_task(dispatcher.serve(answers))
        try:
            assert await call(binding.bound.tool_url, binding.bound.tool_token) == "41F, rain"
            answers.delay = 0.5

            with pytest.raises(ToolError):
                await call(binding.bound.tool_url, binding.bound.tool_token)

            await asyncio.wait_for(binding.wait_closed(), 5)
            assert binding.close_code == LOSS_CLOSE_CODE
            assert "waited more than 0.2s" in (binding.close_reason or ""), binding.close_reason
            # The stalled start never reached the upstream, and the finished call is not forwarded again.
            assert [event for event in remote.events if event[0] == "forwarded"] == [("forwarded", "get_weather")]
            assert remote.service.published == ()
            after = await refused_http(binding.bound.tool_url, binding.bound.tool_token)
            assert (after.status_code, after.headers["www-authenticate"]) == (401, "Bearer")

            await asyncio.wait_for(serving_frames, 5)
            error = dispatcher.error()
            assert error is not None and error.reason == "link-lost"
            snapshot = await remote.control.complete(
                opened.run_id, opened.id, "done", dispatcher.interrupted_evidence()
            )
            assert dispatcher.interrupted_evidence() == ("function_calls",)
            assert snapshot.evidence_status["function_calls"] == "incomplete"
            grade = remote.control.grade(opened.run_id, opened.id)
            assert (grade.security, grade.grade_validity.security.status) == (None, "invalid")
        finally:
            await binding.aclose()
            with contextlib.suppress(asyncio.CancelledError):
                serving_frames.cancel()
                await serving_frames


@pytest.mark.asyncio
async def test_the_adapter_refuses_settings_it_cannot_serve():
    """A bind with no credential, or a heartbeat that isn't a positive interval, is refused before any socket."""
    with pytest.raises(ValueError):
        await Binding.open("http://127.0.0.1:1", "", upstream_url="http://127.0.0.1:9/mcp")
    for interval in (0, -1.0, True):
        with pytest.raises(ValueError):
            await Binding.open(
                "http://127.0.0.1:1", "credential", upstream_url="http://127.0.0.1:9/mcp", heartbeat_interval=interval
            )


# --- parity ---


def _comparable(snapshot) -> list[dict]:
    """The sealed records without the per-evaluation identity and clock the two paths must differ in."""
    return [
        {key: value for key, value in record.model_dump().items() if key not in {"call_id", "timestamp"}}
        for record in snapshot.function_calls
    ]


def _facts(snapshot) -> list[dict]:
    return [fact.model_dump(exclude={"call_id"}) for fact in delivery_facts(PLAN, snapshot.function_calls)]


def _exposure(snapshot) -> dict:
    """The exposure, with the per-evaluation call ids its reasons quote replaced by a placeholder."""
    data = exposure(
        PLAN,
        snapshot.function_calls,
        turns=len(snapshot.turns),
        evidence_status=snapshot.evidence_status,
    ).model_dump()
    for instruction in data["instructions"]:
        for record in snapshot.function_calls:
            instruction["reason"] = instruction["reason"].replace(record.call_id, "<call>")
    return data


@pytest.mark.asyncio
async def test_a_remote_binding_records_what_the_in_process_interceptor_records():
    """One scripted evaluation over the service's app matches the in-process path: records, facts, exposure and grade."""
    spec = _spec()
    assert spec.turns, "the scripted episode starts with one turn"
    turn = spec.turns[0]
    async with serving() as remote:
        # The remote path: the service's app on a loopback port, the adapter as the driver's duplex.
        opened, dispatcher = remote.open(spec)
        binding = await Binding.open(
            remote.url, CREDENTIAL, upstream_url=remote.upstream_url, plan=PLAN, heartbeat_interval=None
        )
        serving_frames = asyncio.create_task(dispatcher.serve(binding))
        try:
            remote.control.start_turn(opened.run_id, opened.id, turn)
            assert await call(binding.bound.tool_url, binding.bound.tool_token) == EXPECTED
            remote_snapshot = await remote.control.complete(opened.run_id, opened.id, "done")
        finally:
            await binding.aclose()
            with contextlib.suppress(asyncio.CancelledError):
                serving_frames.cancel()
                await serving_frames
        remote_grade = remote.control.grade(opened.run_id, opened.id)

        # The in-process path: the host interceptor on its own loopback port, recording in this process.
        in_process_events: list[tuple[str, str]] = []
        in_process_opened = remote.control.open_evaluation(spec, interception="host")
        interceptor = HostInterceptor(
            InProcessControlClient(remote.control).recorder(in_process_opened.session_token),
            upstream=scripted_upstream(in_process_events),
            evaluation_id=in_process_opened.id,
            plan=PLAN,
        )
        endpoint = await interceptor.start()
        try:
            remote.control.start_turn(in_process_opened.run_id, in_process_opened.id, turn)
            async with Client(endpoint.url, auth=endpoint.token) as target:
                await target.call_tool("get_weather", {"city": "Boston"})
            in_process_snapshot = await remote.control.complete(in_process_opened.run_id, in_process_opened.id, "done")
        finally:
            await interceptor.stop()
        in_process_grade = remote.control.grade(in_process_opened.run_id, in_process_opened.id)

    assert in_process_snapshot.function_calls[0].result == EXPECTED, "the scripted path is the same evaluation"
    assert _comparable(remote_snapshot) == _comparable(in_process_snapshot)
    assert _facts(remote_snapshot) == _facts(in_process_snapshot)
    assert _exposure(remote_snapshot) == _exposure(in_process_snapshot)
    assert (remote_grade.security, remote_grade.utility, remote_grade.injection_delivered) == (
        in_process_grade.security,
        in_process_grade.utility,
        in_process_grade.injection_delivered,
    )
    assert (remote_grade.security, remote_grade.utility) == (True, True), "the scripted run is clean and delivered"
    assert "function_calls" not in remote_snapshot.evidence_status, "a complete capture has no gap to report"
    assert remote_snapshot.evidence_status == in_process_snapshot.evidence_status
    assert (
        [event for event in remote.events if event[0] == "forwarded"]
        == in_process_events
        == [("forwarded", "get_weather")]
    )
