"""Tests for how the SDK reports a call record it couldn't deliver, and how the control plane takes the report."""

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest
from midojo_sdk import recorder as sdk_recorder
from midojo_sdk.client import AgentControlPlaneClient, transient
from midojo_sdk.recorder import (
    CAPTURE_REASON_MAX_LENGTH,
    ControlPlaneRejectedError,
    RejectionKind,
    ReportingRecorder,
    RetryingRecorder,
    reporting,
)

from midojo.control_plane import models


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(sdk_recorder, "RECORD_BACKOFF_SECONDS", 0)


def new_evaluation(client) -> tuple[str, str, str]:
    run = client.post("/runs", json={"suite_name": "weather"}).json()
    evaluation = client.post(f"/runs/{run['id']}/evaluations", json={"user_task_id": "weather_new_york"}).json()
    return run["id"], evaluation["id"], evaluation["session_token"]


def evaluation(client, run_id: str, eval_id: str) -> dict:
    return client.get(f"/runs/{run_id}/evaluations/{eval_id}").json()


REPORT: dict[str, Any] = {
    "call_id": "c1",
    "function": "process_refund",
    "phase": "finish",
    "reason": "ConnectError: unreachable",
}


def test_the_sdk_and_the_control_plane_agree_on_the_report_limits():
    assert sdk_recorder.CAPTURE_REASON_MAX_LENGTH == models.CAPTURE_REASON_MAX_LENGTH
    assert sdk_recorder.CapturePhase == models.CapturePhase


# --- the /agent/capture-failures route ---


def test_the_route_keeps_a_report_once_and_replays_it(client):
    run_id, eval_id, token = new_evaluation(client)
    auth = {"Authorization": f"Bearer {token}"}

    created = client.post("/agent/capture-failures", json=REPORT, headers=auth)
    # A resend with another attempt's reason is the same lost record.
    replayed = client.post("/agent/capture-failures", json={**REPORT, "reason": "ReadTimeout"}, headers=auth)

    assert created.status_code == 201
    assert replayed.status_code == 200
    assert created.json() == replayed.json()
    assert {k: created.json()[k] for k in ("call_id", "function", "phase", "stage", "by")} == {
        "call_id": "c1",
        "function": "process_refund",
        "phase": "finish",
        "stage": "open",
        "by": "agent",
    }
    body = evaluation(client, run_id, eval_id)
    assert len(body["capture_failures"]) == 1
    assert body["evidence_status"] == {"function_calls": "incomplete"}
    assert body["function_calls"] == []  # a report never adds a record


def test_the_route_refuses_a_report_after_the_seal_as_a_late_fact(client):
    run_id, eval_id, token = new_evaluation(client)
    client.post(f"/runs/{run_id}/evaluations/{eval_id}/complete", json={"agent_output": "done"})

    refused = client.post("/agent/capture-failures", json=REPORT, headers={"Authorization": f"Bearer {token}"})

    assert refused.status_code == 401
    assert (refused.json()["kind"], refused.json()["late_fact"]) == ("session_closed", True)
    body = evaluation(client, run_id, eval_id)
    assert body["capture_failures"] == []
    [fact] = body["late_facts"]
    assert (fact["event"], fact["call_id"]) == ("report_capture_failure", "c1")
    assert body["evidence_status"]["function_calls"] == "incomplete"


@pytest.mark.parametrize(
    "report",
    [
        {**REPORT, "call_id": ""},
        {**REPORT, "phase": "middle"},
        {**REPORT, "reason": "x" * (CAPTURE_REASON_MAX_LENGTH + 1)},
        {k: v for k, v in REPORT.items() if k != "function"},
    ],
)
def test_the_route_refuses_a_malformed_report(client, report):
    run_id, eval_id, token = new_evaluation(client)

    response = client.post("/agent/capture-failures", json=report, headers={"Authorization": f"Bearer {token}"})

    assert response.status_code == 422
    assert evaluation(client, run_id, eval_id)["capture_failures"] == []


def test_the_route_needs_a_session(client):
    assert client.post("/agent/capture-failures", json=REPORT).status_code == 401


# --- the HTTP and in-process clients ---


@pytest.mark.asyncio
async def test_both_clients_report_a_lost_record_the_same_way(client, backend):
    run_id, eval_id, token = new_evaluation(client)
    agent = backend.agent(token)

    await agent.report_capture_failure(call_id="c1", function="f", phase="start", reason="x" * 5000)
    await agent.report_capture_failure(call_id="c1", function="f", phase="start", reason="resent")

    [failure] = evaluation(client, run_id, eval_id)["capture_failures"]
    # Both clients cut a long reason to what the control plane accepts.
    assert (failure["phase"], failure["reason"]) == ("start", "x" * CAPTURE_REASON_MAX_LENGTH)


@pytest.mark.asyncio
async def test_both_clients_raise_the_late_fact_after_the_seal(client, backend):
    run_id, eval_id, token = new_evaluation(client)
    agent = backend.agent(token)
    client.post(f"/runs/{run_id}/evaluations/{eval_id}/complete", json={"agent_output": "done"})

    with pytest.raises(ControlPlaneRejectedError) as refused:
        await agent.report_capture_failure(call_id="c1", function="f", phase="finish", reason="lost")

    assert (refused.value.kind, refused.value.late_fact) == (RejectionKind.SESSION_CLOSED, True)
    assert evaluation(client, run_id, eval_id)["evidence_status"]["function_calls"] == "incomplete"


def scripted(responses):
    """An httpx client that answers each request with the next of ``responses``, and the paths and bodies it got."""
    sent: list[tuple[str, dict]] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append((request.url.path, json.loads(request.read())))
        response = responses[len(sent) - 1]
        if isinstance(response, Exception):
            raise response
        return httpx.Response(response, json={})

    return httpx.AsyncClient(transport=httpx.MockTransport(handle)), sent


@pytest.mark.asyncio
async def test_the_http_client_sends_a_report_once():
    http, sent = scripted([503])
    agent = AgentControlPlaneClient("http://control", "session", http=http)

    with pytest.raises(httpx.HTTPStatusError):
        await agent.report_capture_failure(**REPORT)

    assert sent == [("/agent/capture-failures", REPORT)]


@pytest.mark.asyncio
async def test_a_retrying_recorder_resends_a_report_like_a_record():
    http, sent = scripted([httpx.ConnectError("refused"), 503, 201])
    recorder = RetryingRecorder(AgentControlPlaneClient("http://control", "session", http=http), retryable=transient)

    await recorder.report_capture_failure(**REPORT)

    assert [body for _, body in sent] == [REPORT] * 3


# --- the reporting recorder ---


class Failing:
    """A recorder whose every start and finish raises ``error``, and that notes the reports it gets."""

    def __init__(self, error: BaseException, report_error: Exception | None = None) -> None:
        self.error = error
        self.report_error = report_error
        self.reports: list[dict] = []

    async def start_call(self, **kwargs) -> None:
        raise self.error

    async def finish_call(self, **kwargs) -> None:
        raise self.error

    async def report_capture_failure(self, **kwargs) -> None:
        self.reports.append(kwargs)
        if self.report_error is not None:
            raise self.report_error


async def start(recorder) -> None:
    await recorder.start_call(call_id="c1", function="f", args={}, disposition="forwarded")


async def finish(recorder) -> None:
    await recorder.finish_call(call_id="c1", function="f", args={}, result="r", disposition="forwarded")


@pytest.mark.asyncio
@pytest.mark.parametrize(("send", "phase"), [(start, "start"), (finish, "finish")])
@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("unreachable"),
        ControlPlaneRejectedError(RejectionKind.CALL_CONFLICT, "another body"),
        sdk_recorder.InvalidRequestError("call_id too long"),
    ],
)
async def test_a_record_that_fails_for_good_is_reported_then_raised(send, phase, error):
    failing = Failing(error)

    with pytest.raises(type(error)) as raised:
        await send(ReportingRecorder(failing))

    assert raised.value is error
    [report] = failing.reports
    assert (report["call_id"], report["function"], report["phase"]) == ("c1", "f", phase)
    assert report["reason"] == f"{type(error).__name__}: {error}"


@pytest.mark.asyncio
async def test_a_refusal_kept_as_a_late_fact_is_not_reported_again():
    failing = Failing(ControlPlaneRejectedError(RejectionKind.CLOSING, "closing", late_fact=True))

    with pytest.raises(ControlPlaneRejectedError):
        await finish(ReportingRecorder(failing))

    assert failing.reports == []


@pytest.mark.asyncio
async def test_a_cancelled_record_is_not_reported():
    failing = Failing(asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        await finish(ReportingRecorder(failing))

    assert failing.reports == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "report_error",
    [httpx.ConnectError("still unreachable"), ControlPlaneRejectedError(RejectionKind.SESSION_CLOSED, "closed")],
)
async def test_a_report_that_fails_too_is_logged_and_the_records_error_is_raised(report_error, caplog):
    error = httpx.ConnectError("unreachable")
    failing = Failing(error, report_error)

    with caplog.at_level(logging.WARNING, logger="midojo_sdk.recorder"), pytest.raises(httpx.ConnectError) as raised:
        await finish(ReportingRecorder(failing))

    assert raised.value is error
    assert len(failing.reports) == 1
    assert "capture failure of f (c1)" in caplog.text


@pytest.mark.asyncio
async def test_a_recorder_that_cannot_report_still_raises_the_records_error(caplog):
    class Unreachable:
        async def start_call(self, **kwargs) -> None:
            raise httpx.ConnectError("unreachable")

    with caplog.at_level(logging.WARNING, logger="midojo_sdk.recorder"), pytest.raises(httpx.ConnectError):
        await start(ReportingRecorder(Unreachable()))  # type: ignore[arg-type]

    assert "capture failure of f (c1)" in caplog.text


@pytest.mark.asyncio
async def test_the_report_follows_every_resend_of_a_retrying_recorder():
    http, sent = scripted([httpx.ConnectError("unreachable")] * sdk_recorder.RECORD_ATTEMPTS + [201])
    agent = AgentControlPlaneClient("http://control", "session", http=http)

    with pytest.raises(httpx.ConnectError):
        await finish(reporting(RetryingRecorder(agent, retryable=transient)))

    paths = [path for path, _ in sent]
    assert paths == ["/agent/function-calls"] * sdk_recorder.RECORD_ATTEMPTS + ["/agent/capture-failures"]
    assert sent[-1][1] == {
        "call_id": "c1",
        "function": "f",
        "phase": "finish",
        "reason": "ConnectError: unreachable",
    }


def test_reporting_wraps_a_recorder_once():
    wrapped = reporting(Failing(RuntimeError()))
    assert reporting(wrapped) is wrapped


@pytest.mark.asyncio
async def test_a_delivered_record_is_not_reported(client, backend):
    run_id, eval_id, token = new_evaluation(client)
    recorder = reporting(backend.agent(token))

    await start(recorder)
    await finish(recorder)

    body = evaluation(client, run_id, eval_id)
    assert (len(body["function_calls"]), body["capture_failures"]) == (1, [])
    assert "function_calls" not in body["evidence_status"]


@pytest.mark.asyncio
async def test_a_refused_record_is_reported_through_either_client(client, backend):
    run_id, eval_id, token = new_evaluation(client)
    recorder = reporting(backend.agent(token))
    await finish(recorder)

    with pytest.raises(ControlPlaneRejectedError):
        await recorder.finish_call(call_id="c1", function="f", args={}, result="other", disposition="forwarded")

    body = evaluation(client, run_id, eval_id)
    [failure] = body["capture_failures"]
    assert (failure["call_id"], failure["phase"]) == ("c1", "finish")
    assert failure["reason"].startswith("ControlPlaneRejectedError: call_conflict")
    assert body["evidence_status"] == {"function_calls": "incomplete"}
