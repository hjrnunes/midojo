"""Tests for how the Python SDK identifies and resends function-call records."""

import httpx
import pytest
from midojo_sdk import client as sdk_client
from midojo_sdk.client import RECORD_ATTEMPTS, AgentControlPlaneClient, CallIds


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(sdk_client, "RECORD_BACKOFF_SECONDS", 0)


def scripted(responses):
    """An httpx client that answers each POST with the next of ``responses``, and the bodies it was sent."""
    sent = []

    def handle(request: httpx.Request) -> httpx.Response:
        sent.append(request.read())
        response = responses[len(sent) - 1]
        if isinstance(response, Exception):
            raise response
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
