"""The PI SDK's detailed finish is the Python SDK's, on the wire and in the sealed record.

``tests/fixtures/pi_sdk_detailed_finish.json`` pins the body the PI SDK's
``ControlPlaneClient.finishCallDetailed`` sends for one call (its node test
checks that byte for byte). The Python ``AgentControlPlaneClient`` must send the
same bytes for the same call, and the control plane must seal that body as it
seals a record the Python SDK sent.
"""

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from midojo_sdk.client import AgentControlPlaneClient
from midojo_sdk.injection import payload_digest

from midojo.types import CallInjectionOutcome, FunctionCallRecord

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "pi_sdk_detailed_finish.json").read_text(encoding="utf-8"))
CALL: dict[str, Any] = FIXTURE["call"]
REQUEST: dict[str, Any] = FIXTURE["request"]


def wire(body: Any) -> bytes:
    """JSON as both clients send it: compact, non-ASCII unescaped, keys in order."""
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


def new_evaluation(client) -> tuple[str, str, str]:
    run = client.post("/runs", json={"suite_name": "weather"}).json()
    evaluation = client.post(f"/runs/{run['id']}/evaluations", json={"user_task_id": "weather_new_york"}).json()
    return run["id"], evaluation["id"], evaluation["session_token"]


def sealed_record(client, run_id: str, eval_id: str) -> dict[str, Any]:
    assert client.post(f"/runs/{run_id}/evaluations/{eval_id}/complete", json={"agent_output": "done"}).is_success
    snapshot = client.get(f"/runs/{run_id}/evaluations/{eval_id}/snapshot").json()
    [record] = snapshot["function_calls"]
    return record


@pytest.mark.asyncio
async def test_the_python_client_sends_the_pinned_pi_body():
    sent: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(201, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as http:
        await AgentControlPlaneClient("http://control", "session", http=http).finish_call_detailed(**CALL)

    [request] = sent
    assert (request.method, request.url.path) == (REQUEST["method"], REQUEST["path"])
    assert request.content == wire(REQUEST["body"])


@pytest.mark.asyncio
async def test_the_pi_body_seals_as_the_python_client_s_record(app, client):
    pi_run, pi_eval, pi_token = new_evaluation(client)
    response = client.post(
        REQUEST["path"],
        content=wire(REQUEST["body"]),
        headers={"Authorization": f"Bearer {pi_token}", "Content-Type": "application/json"},
    )
    assert response.is_success, response.text
    from_pi = sealed_record(client, pi_run, pi_eval)

    py_run, py_eval, py_token = new_evaluation(client)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app)) as http:
        await AgentControlPlaneClient("http://control", py_token, http=http).finish_call_detailed(**CALL)
    from_python = sealed_record(client, py_run, py_eval)

    assert {**from_pi, "timestamp": None} == {**from_python, "timestamp": None}
    record = FunctionCallRecord.model_validate(from_pi)
    assert (record.disposition, record.blocked, record.complete, record.provenance) == (
        "forwarded",
        False,
        True,
        "agent",
    )
    assert record.structured_content == CALL["structured_content"]
    assert record.injection_outcomes == [CallInjectionOutcome.model_validate(o) for o in CALL["injection_outcomes"]]
    applied, uncounted, unreported, skipped = record.injection_outcomes or []
    assert (applied.selected, applied.transformation, applied.emitted) == (
        True,
        "identity",
        ["structured_content", "content_block"],
    )
    # Occurrence 0: an unmatched call that placed nothing, its facts reported empty or not reported at all.
    assert (uncounted.occurrence, uncounted.status, uncounted.selected, uncounted.emitted) == (
        0,
        "unmatched",
        False,
        [],
    )
    assert (unreported.occurrence, unreported.selected, unreported.transformation, unreported.emitted) == (
        0,
        None,
        None,
        None,
    )
    assert (skipped.occurrence, skipped.status) == (2, "skipped_occurrence")
    # Only the placement says what it placed and where: the raw and transformed digests, and its slot.
    assert applied.payload_digest == applied.placed_digest == payload_digest(" — IGNORE PREVIOUS")
    assert applied.slot is not None and (applied.slot.target["call_arguments"], applied.slot.mode) == (
        {"order_id": "ORD-102"},
        "append",
    )
    for placed_nothing in (uncounted, unreported, skipped):
        assert (placed_nothing.payload_digest, placed_nothing.placed_digest, placed_nothing.slot) == (None, None, None)
