"""Tests for how the Python SDK identifies, resends and reports function-call records."""

import json

import httpx
import pytest
from midojo_sdk import client as sdk_client
from midojo_sdk import recorder as sdk_recorder
from midojo_sdk.client import RECORD_ATTEMPTS, AgentControlPlaneClient, CallIds, transient
from midojo_sdk.recorder import (
    ControlPlaneRecorder,
    ControlPlaneRejectedError,
    InvalidRequestError,
    RejectionKind,
    RetryingRecorder,
    retry,
)

from midojo.control_plane.lifecycle import Rejection


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(sdk_recorder, "RECORD_BACKOFF_SECONDS", 0)


def scripted(responses):
    """An httpx client that answers each request with the next of ``responses``, and the bodies it was sent.

    A response is a status code (answered with an empty JSON object), an
    ``httpx.Response``, or an exception to raise.
    """
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request.read())
        response = responses[len(sent) - 1]
        if isinstance(response, Exception):
            raise response
        if isinstance(response, httpx.Response):
            return response
        return httpx.Response(response, json={})

    return httpx.AsyncClient(transport=httpx.MockTransport(handle)), sent


async def record(http, **kwargs):
    agent = AgentControlPlaneClient("http://control", "session", http=http)
    await agent.record_function_call(function="get_weather", args={"city": "Boston"}, result="sunny", **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [503, 500, httpx.ConnectError("refused"), httpx.ReadTimeout("lost response")])
async def test_record_is_resent_with_the_same_call_id(failure):
    http, sent = scripted([failure, failure, 201])

    await record(http, call_id="call-1")

    assert len(sent) == 3
    assert len(set(sent)) == 1  # every resend has the same body
    assert b'"call_id":"call-1"' in sent[0]


@pytest.mark.asyncio
async def test_record_without_call_id_is_resent_with_one_uuid():
    http, sent = scripted([503, 200])

    await record(http)

    assert len(sent) == 2
    assert sent[0] == sent[1]
    assert b'"call_id":"' in sent[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 401, 404, 409, 422])
async def test_record_is_not_resent_after_a_4xx(status):
    http, sent = scripted([status, 201])

    with pytest.raises(httpx.HTTPStatusError):
        await record(http, call_id="call-1")
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_record_attempts_are_bounded():
    http, sent = scripted([503] * RECORD_ATTEMPTS)
    with pytest.raises(httpx.HTTPStatusError):
        await record(http)
    assert len(sent) == RECORD_ATTEMPTS

    http, sent = scripted([httpx.ConnectError("refused")] * RECORD_ATTEMPTS)
    with pytest.raises(httpx.ConnectError):
        await record(http)
    assert len(sent) == RECORD_ATTEMPTS


async def start(http):
    agent = AgentControlPlaneClient("http://control", "session", http=http)
    await agent.start_function_call(function="get_weather", args={"city": "Boston"}, call_id="call-1")


@pytest.mark.asyncio
async def test_start_is_resent_with_the_same_body():
    http, sent = scripted([503, httpx.ConnectError("refused"), 201])

    await start(http)

    assert len(sent) == 3
    assert len(set(sent)) == 1
    assert json.loads(sent[0]) == {"call_id": "call-1", "function": "get_weather", "args": {"city": "Boston"}}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [409, 422])
async def test_start_is_not_resent_after_a_4xx(status):
    http, sent = scripted([status, 201])

    with pytest.raises(httpx.HTTPStatusError):
        await start(http)
    assert len(sent) == 1


def test_call_ids_namespace_harness_ids():
    first, second = CallIds("langchain"), CallIds("langchain")

    call_id = first.for_call("call-1")

    assert call_id.startswith("langchain:") and call_id.endswith(":call-1")
    assert second.for_call("call-1") != call_id  # another SDK instance can't collide


@pytest.mark.parametrize("harness_id", [None, "", "x" * 300])
def test_call_ids_fall_back_to_a_uuid(harness_id):
    call_ids = CallIds("langchain")
    assert ":" not in call_ids.for_call(harness_id)


def test_call_ids_fall_back_when_a_harness_id_repeats():
    call_ids = CallIds("langchain")
    first = call_ids.for_call("call-0")
    again = call_ids.for_call("call-0")
    assert again != first
    assert ":" not in again


def test_call_ids_pick_a_new_namespace_in_a_forked_process(monkeypatch):
    call_ids = CallIds("langchain")
    parent = call_ids.for_call("call-1")

    monkeypatch.setattr(sdk_client.os, "getpid", lambda: -1)  # as in a worker forked after the instance was built
    child = call_ids.for_call("call-1")

    assert child.endswith(":call-1")  # the used-ID set starts empty again, so it isn't a UUID fallback
    assert child.rsplit(":", 1)[0] != parent.rsplit(":", 1)[0]
    assert call_ids.for_call("call-1") != child  # and the new namespace's IDs are tracked


# --- retries are a wrapping recorder ---


def retrying(http) -> RetryingRecorder:
    return RetryingRecorder(AgentControlPlaneClient("http://control", "session", http=http), retryable=transient)


async def finish(recorder, **kwargs):
    await recorder.finish_call(
        call_id="call-1", function="get_weather", args={"city": "Boston"}, result="sunny", disposition="forwarded"
    )


@pytest.mark.asyncio
async def test_the_http_recorder_sends_each_event_once():
    http, sent = scripted([503, 201])
    with pytest.raises(httpx.HTTPStatusError):
        await finish(AgentControlPlaneClient("http://control", "session", http=http))
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [503, 500, httpx.ConnectError("refused"), httpx.ReadTimeout("lost response")])
async def test_a_retrying_recorder_resends_a_transient_failure_with_the_same_body(failure):
    http, sent = scripted([failure, failure, 201])

    await finish(retrying(http))

    assert len(sent) == 3
    assert len(set(sent)) == 1


@pytest.mark.asyncio
async def test_a_retrying_recorder_resends_a_start_like_a_finish():
    http, sent = scripted([503, httpx.ConnectError("refused"), 201])

    await retrying(http).start_call(call_id="call-1", function="get_weather", args={}, disposition="simulated")

    assert len(sent) == 3
    assert len(set(sent)) == 1
    assert json.loads(sent[0]) == {
        "call_id": "call-1",
        "function": "get_weather",
        "args": {},
        "disposition": "simulated",
    }


@pytest.mark.asyncio
async def test_a_retrying_recorder_gives_up_after_its_attempts():
    http, sent = scripted([503] * RECORD_ATTEMPTS)
    with pytest.raises(httpx.HTTPStatusError):
        await finish(retrying(http))
    assert len(sent) == RECORD_ATTEMPTS


def rejection(status: int, kind: str, late_fact: bool = False) -> httpx.Response:
    return httpx.Response(status, json={"detail": f"refused: {kind}", "kind": kind, "late_fact": late_fact})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response", [rejection(409, "call_conflict"), rejection(401, "session_closed", True), 400, 404, 422]
)
async def test_a_retrying_recorder_never_resends_a_rejection_or_another_4xx(response):
    http, sent = scripted([response, 201])
    with pytest.raises((ControlPlaneRejectedError, httpx.HTTPStatusError)):
        await finish(retrying(http))
    assert len(sent) == 1


# --- only the control plane's own rejections are rejections ---


@pytest.mark.asyncio
async def test_a_rejection_carries_its_kind_detail_and_late_fact():
    http, _ = scripted([rejection(401, "session_closed", late_fact=True)])

    with pytest.raises(ControlPlaneRejectedError) as refused:
        await finish(AgentControlPlaneClient("http://control", "session", http=http))

    assert (refused.value.kind, refused.value.detail, refused.value.late_fact) == (
        RejectionKind.SESSION_CLOSED,
        "refused: session_closed",
        True,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503, text="Service Unavailable"),
        httpx.Response(503, json={"detail": "upstream connect error"}),
        httpx.Response(503, json={"detail": "x", "kind": "not_a_control_plane_kind"}),
        httpx.Response(409, json=["call_conflict"]),
        httpx.Response(401, json={"detail": "Evaluation session token required"}),
    ],
    ids=["text", "no-kind", "unknown-kind", "not-an-object", "edge-401"],
)
async def test_an_error_without_a_control_plane_kind_is_an_http_error(response):
    http, _ = scripted([response])
    agent = AgentControlPlaneClient("http://control", "session", http=http)
    with pytest.raises(httpx.HTTPStatusError) as failure:
        await finish(agent)
    assert not isinstance(failure.value, ControlPlaneRejectedError)
    assert failure.value.response.status_code == response.status_code


@pytest.mark.asyncio
async def test_only_a_revision_conflict_on_a_change_with_a_revision_is_none():
    conflict = rejection(409, "revision_conflict")
    http, _ = scripted([conflict, conflict, httpx.Response(409, json={"detail": "conflict"})])
    agent = AgentControlPlaneClient("http://control", "session", http=http)

    assert await agent.change({"weather_alerts": []}, expected_revision=3) is None
    with pytest.raises(ControlPlaneRejectedError) as refused:
        await agent.change({"weather_alerts": []})
    assert refused.value.kind is RejectionKind.REVISION_CONFLICT
    with pytest.raises(httpx.HTTPStatusError):
        await agent.change({"weather_alerts": []}, expected_revision=3)


def test_the_sdk_names_the_control_planes_rejection_kinds():
    # The SDK doesn't import midojo, so it names the kinds itself.
    assert {kind.value for kind in RejectionKind} == {kind.value for kind in Rejection}


def test_transient_failures_are_transport_errors_and_5xx_responses():
    request = httpx.Request("POST", "http://control/agent/function-calls")

    def status_error(code: int) -> httpx.HTTPStatusError:
        return httpx.HTTPStatusError("failed", request=request, response=httpx.Response(code, request=request))

    assert transient(httpx.ConnectError("refused"))
    assert transient(status_error(503))
    assert not transient(status_error(409))
    assert not transient(ControlPlaneRejectedError(RejectionKind.SEALED, "sealed"))
    assert not transient(ValueError("not a transport failure"))


# --- the same outcomes on both implementations ---


def new_evaluation(client) -> tuple[str, str, str]:
    run = client.post("/runs", json={"suite_name": "weather"}).json()
    evaluation = client.post(f"/runs/{run['id']}/evaluations", json={"user_task_id": "weather_new_york"}).json()
    return run["id"], evaluation["id"], evaluation["session_token"]


class Counting:
    """Passes each start and finish to ``recorder``, counting them."""

    def __init__(self, recorder) -> None:
        self.recorder = recorder
        self.sent = 0

    async def start_call(self, **kwargs) -> None:
        self.sent += 1
        await self.recorder.start_call(**kwargs)

    async def finish_call(self, **kwargs) -> None:
        self.sent += 1
        await self.recorder.finish_call(**kwargs)


@pytest.mark.asyncio
async def test_a_conflicting_record_is_the_same_rejection_and_is_never_resent(client, backend):
    _, _, token = new_evaluation(client)
    counting = Counting(backend.agent(token))
    recorder = RetryingRecorder(counting, retryable=transient)
    await recorder.finish_call(call_id="c1", function="f", args={}, result="first", disposition="simulated")

    with pytest.raises(ControlPlaneRejectedError) as refused:
        await recorder.finish_call(call_id="c1", function="f", args={}, result="second", disposition="simulated")

    assert (refused.value.kind, refused.value.late_fact) == (RejectionKind.CALL_CONFLICT, False)
    assert counting.sent == 2
    # A replay with the same body is accepted again: at most one record per call_id.
    await recorder.finish_call(call_id="c1", function="f", args={}, result="first", disposition="simulated")


@pytest.mark.asyncio
async def test_a_record_after_the_seal_is_the_same_late_fact(client, backend):
    run_id, eval_id, token = new_evaluation(client)
    agent = backend.agent(token)
    client.post(f"/runs/{run_id}/evaluations/{eval_id}/complete", json={"agent_output": "done"})

    with pytest.raises(ControlPlaneRejectedError) as refused:
        await agent.finish_call(call_id="c1", function="f", args={}, result="late", disposition="forwarded")

    assert (refused.value.kind, refused.value.late_fact) == (RejectionKind.SESSION_CLOSED, True)
    [fact] = client.get(f"/runs/{run_id}/evaluations/{eval_id}").json()["late_facts"]
    assert (fact["event"], fact["call_id"]) == ("finish_call", "c1")


@pytest.mark.asyncio
@pytest.mark.parametrize(("disposition", "blocked"), [("simulated", False), ("denied", True), ("blocked", True)])
async def test_a_refused_disposition_is_recorded_as_blocked(client, backend, disposition, blocked):
    run_id, eval_id, token = new_evaluation(client)

    await backend.agent(token).finish_call(call_id="c1", function="f", args={}, result="r", disposition=disposition)

    [call] = client.get(f"/runs/{run_id}/evaluations/{eval_id}/function-calls").json()
    assert (call["call_id"], call["blocked"], call["complete"]) == ("c1", blocked, True)


@pytest.mark.asyncio
async def test_both_implementations_record_the_same_call(client, backend):
    run_id, eval_id, token = new_evaluation(client)
    agent = backend.agent(token)

    await agent.start_call(call_id="c1", function="get_weather", args={"city": "Boston"}, disposition="forwarded")
    await agent.finish_call(
        call_id="c1", function="get_weather", args={"city": "Boston"}, result="r", error="e", disposition="forwarded"
    )

    [call] = client.get(f"/runs/{run_id}/evaluations/{eval_id}/function-calls").json()
    del call["timestamp"]
    assert call == {
        "call_id": "c1",
        "function": "get_weather",
        "args": {"city": "Boston"},
        "result": "r",
        "error": "e",
        "blocked": False,
        "disposition": "forwarded",
        "complete": True,
        "turn": None,
    }


class Refusing:
    """A recorder whose every start and finish raises ``error``, counting the attempts."""

    def __init__(self, error: Exception) -> None:
        self.error = error
        self.attempts = 0

    async def start_call(self, **kwargs) -> None:
        self.attempts += 1
        raise self.error

    async def finish_call(self, **kwargs) -> None:
        self.attempts += 1
        raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [ControlPlaneRejectedError(RejectionKind.CALL_CONFLICT, "conflict"), InvalidRequestError("call_id too long")],
    ids=["rejection", "invalid-request"],
)
async def test_a_retrying_recorder_never_resends_a_final_error_whatever_its_predicate(error):
    refusing = Refusing(error)
    recorder = RetryingRecorder(refusing, retryable=lambda e: True)

    with pytest.raises(type(error)):
        await recorder.start_call(call_id="c1", function="f", args={}, disposition="simulated")
    with pytest.raises(type(error)):
        await recorder.finish_call(call_id="c1", function="f", args={}, result="r", disposition="simulated")

    assert refusing.attempts == 2  # one attempt each


@pytest.mark.asyncio
async def test_the_retry_helper_never_resends_a_rejection_whatever_its_predicate():
    refusing = Refusing(ControlPlaneRejectedError(RejectionKind.SEALED, "sealed"))
    with pytest.raises(ControlPlaneRejectedError):
        await retry(lambda: refusing.finish_call(), retryable=lambda e: True)
    assert refusing.attempts == 1


@pytest.mark.asyncio
async def test_a_lost_start_response_is_resent_and_records_one_call(client, app):
    """The start commits, its response is lost, and the resend replays it: one record, not two."""
    run_id, eval_id, token = new_evaluation(client)
    control = httpx.ASGITransport(app)
    lost: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        response = await control.handle_async_request(request)
        if request.url.path.endswith("/start") and not lost:
            lost.append(request.url.path)  # committed on the control plane; the response never arrives
            raise httpx.ReadError("connection reset after the commit")
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        recorder = RetryingRecorder(AgentControlPlaneClient("http://control", token, http=http), retryable=transient)
        await recorder.start_call(call_id="c1", function="f", args={"x": 1}, disposition="forwarded")
        await recorder.finish_call(call_id="c1", function="f", args={"x": 1}, result="r", disposition="forwarded")

    assert lost == ["/agent/function-calls/start"]
    calls = client.get(f"/runs/{run_id}/evaluations/{eval_id}/function-calls").json()
    assert [(call["call_id"], call["complete"], call["result"]) for call in calls] == [("c1", True, "r")]


@pytest.mark.asyncio
async def test_the_deprecated_control_plane_recorder_still_records_and_resends():
    http, sent = scripted([503, 201, 201])
    with pytest.warns(DeprecationWarning, match="ControlPlaneRecorder is deprecated"):
        recorder = ControlPlaneRecorder(AgentControlPlaneClient("http://control", "session", http=http))

    await recorder.start_call(call_id="c1", function="f", args={}, disposition="simulated")
    await recorder.finish_call(call_id="c1", function="f", args={}, result="r", disposition="denied")

    start = {"call_id": "c1", "function": "f", "args": {}, "disposition": "simulated"}
    finish = {**start, "result": "r", "error": None, "blocked": True, "disposition": "denied"}
    assert [json.loads(body) for body in sent] == [start, start, finish]


# --- a malformed request is the same error on both implementations ---


@pytest.mark.asyncio
@pytest.mark.parametrize("call_id", ["", "x" * 257], ids=["empty", "too-long"])
@pytest.mark.parametrize("event", ["start", "finish"])
async def test_a_malformed_call_is_an_invalid_request_on_both_implementations(client, backend, call_id, event):
    run_id, eval_id, token = new_evaluation(client)
    agent = backend.agent(token)

    with pytest.raises(InvalidRequestError) as invalid:
        if event == "start":
            await agent.start_call(call_id=call_id, function="f", args={}, disposition="forwarded")
        else:
            await agent.finish_call(call_id=call_id, function="f", args={}, result="r", disposition="forwarded")

    assert "call_id" in invalid.value.detail
    # Over HTTP it is still the API's 422, for callers that catch httpx errors.
    assert isinstance(invalid.value, httpx.HTTPStatusError) is (backend.name == "http")
    assert client.get(f"/runs/{run_id}/evaluations/{eval_id}/function-calls").json() == []
