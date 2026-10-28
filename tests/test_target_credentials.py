"""A host-mode target can't read or change its own measurement with what it is given.

The target's credentials are taken from where each kind of target gets them:
an OpenShell sandbox's environment (a faked gateway, whose suite's
``env_vars`` try to hand it the control plane), a PI subprocess's environment
(whose host's environment names the session and the control plane) and the
Responses API's MCP tool block. Each credential is then presented, as a bearer
token, to every control plane route: the ``/agent`` routes that read or write
an evaluation's environment and records, with valid bodies, and every
run-scoped route. All of them refuse it, and the evaluation's environment and
records stay as they were, through the seal. The interceptor's token calls
tools, and nothing else.

The run-scoped routes take only the control token, and a server without one
is open to anything that reaches it. So a control plane an app serves without
a control token opens no host-mode evaluation, whatever path asks: over HTTP,
in process on the app's control plane, or through the episode context. A
control plane no app serves has no routes, and opens host mode as before.

No model, no gateway, no network beyond loopback.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from fastapi.routing import APIRoute
from fastmcp import Client
from openshell import ExecResult, SandboxRef, SandboxStatusRef

from midojo.agent_client import OpenAIResponsesAgentClient, PIAgentClient
from midojo.control_plane.client import ControlPlaneClient
from midojo.control_plane.config import CONTROL_TOKEN_ENV, AppConfig
from midojo.control_plane.inprocess_control import InProcessControlClient
from midojo.control_plane.lifecycle import RejectedError, Rejection
from midojo.control_plane.main import create_app
from midojo.control_plane.service import ById, ControlPlane
from midojo.control_plane.store import InMemoryStore
from midojo.episode_context import open_episode
from midojo.interception_host import HostInterception, HostInterceptor
from midojo.runtimes import UnmanagedRuntime
from midojo.runtimes.openshell import OpenShellRuntime
from midojo.tool_endpoint import ToolEndpoint
from tests.fixtures.scripted_target.upstream import scripted_upstream

CONTROL_TOKEN = "control-secret-7f3a"
CONTROL_URL = "http://control"


class _FakeSandboxClient:
    """A gateway that keeps each sandbox's spec and runs nothing."""

    def __init__(self) -> None:
        self.specs: list = []

    def create(self, *, spec, name, workspace, labels=None, **kwargs):
        self.specs.append(spec)
        return SandboxRef(id="sb-1", name=name, workspace=workspace, status=SandboxStatusRef(0, 0))

    def wait_ready(self, name, **kwargs):
        pass

    def exec(self, sandbox_name, command, **kwargs):
        return ExecResult(exit_code=0, stdout="", stderr="")


@pytest_asyncio.fixture
async def host_mode(suite):
    """A host-mode evaluation on a control plane app with a control token, and its running interceptor."""
    app = create_app({"weather": suite}, config=AppConfig(control_token=CONTROL_TOKEN, closure_wait_seconds=0))
    control = app.state.control
    run = control.create_run("weather")
    evaluation = control.create_evaluation(run.id, "weather_new_york", interception="host")
    recorder = InProcessControlClient(control).recorder(evaluation.session_token)
    interceptor = HostInterceptor(recorder, upstream=scripted_upstream(), evaluation_id=evaluation.id)
    endpoint = await interceptor.start()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url=CONTROL_URL) as http:
        yield SimpleNamespace(app=app, control=control, run=run, evaluation=evaluation, endpoint=endpoint, http=http)
    await interceptor.stop()


def _openshell_credentials(endpoint: ToolEndpoint, session_token: str, monkeypatch) -> list[str]:
    """The environment a host-mode OpenShell sandbox gets, its suite trying to name the control plane in it."""
    monkeypatch.setenv(CONTROL_TOKEN_ENV, CONTROL_TOKEN)
    env_vars = {
        "LITELLM_MODEL": "m",
        "MIDOJO_URL": CONTROL_URL,
        "MIDOJO_CONTROL_TOKEN": CONTROL_TOKEN,
        "UPSTREAM_MCP_URL": "http://upstream:9000/mcp",
    }
    runtime = OpenShellRuntime("weather", image="pi", env_vars=env_vars)
    client = _FakeSandboxClient()
    runtime._client, runtime._workspace_name = client, "midojo-weath-abc123"
    runtime.configure(cluster="test", interceptor_url=endpoint.url)
    runtime.setup({}, tool_endpoint=endpoint, eval_id="e1", user_task_id="weather_new_york", injection_task_id=None)
    [spec] = client.specs
    return list(dict(spec.environment).values())


async def _pi_credentials(endpoint: ToolEndpoint, session_token: str, monkeypatch) -> list[str]:
    """What a host-mode PI subprocess's environment adds to, or keeps from, a host that names the session."""
    planted = {
        "MIDOJO_SESSION_TOKEN": session_token,
        "MIDOJO_URL": CONTROL_URL,
        CONTROL_TOKEN_ENV: CONTROL_TOKEN,
        "UPSTREAM_MCP_URL": "http://upstream:9000/mcp",
    }
    inherited = dict(os.environ)
    for name, value in planted.items():
        monkeypatch.setenv(name, value)
    launches: list[dict] = []

    class Process:
        returncode = 0

        async def communicate(self):
            return b"done", b""

    async def launch(*args, **kwargs):
        launches.append(kwargs["env"])
        return Process()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", launch)
    await PIAgentClient("agent", CONTROL_URL).send_task("task", tool_endpoint=endpoint)
    [env] = launches
    return [value for name, value in env.items() if name in planted or inherited.get(name) != value]


async def _responses_credentials(endpoint: ToolEndpoint, session_token: str, monkeypatch) -> list[str]:
    """The MCP tool block a host-mode Responses API request carries: its URL and every header value."""
    instance = MagicMock()
    instance.responses.create = AsyncMock(return_value=MagicMock(output_text="ok", id="resp-1"))
    instance.__aenter__ = AsyncMock(return_value=instance)
    instance.__aexit__ = AsyncMock(return_value=None)
    module = MagicMock(AsyncOpenAI=MagicMock(return_value=instance))
    client = OpenAIResponsesAgentClient("http://localhost:8321/v1", "m", None, api_key="x")
    with patch.dict(sys.modules, {"openai": module}):
        await client.send_task("task", tool_endpoint=endpoint)
    [tool] = instance.responses.create.call_args.kwargs["tools"]
    return [tool["server_url"], *tool["headers"].values()]


async def _credentials(source: str, endpoint: ToolEndpoint, session_token: str, monkeypatch) -> list[str]:
    if source == "openshell":
        return _openshell_credentials(endpoint, session_token, monkeypatch)
    if source == "pi":
        return await _pi_credentials(endpoint, session_token, monkeypatch)
    return await _responses_credentials(endpoint, session_token, monkeypatch)


def _bearers(credentials: list[str]) -> set[str]:
    """Every value the target holds, as it would present it: whole, and without a ``Bearer`` prefix."""
    return {value.removeprefix("Bearer ") for value in credentials} | set(credentials)


def _agent_requests(environment: dict) -> list[tuple[str, str, dict | None]]:
    """The /agent routes, each with a body the control plane would accept with the evaluation's own session."""
    forged = {**environment, "weather_alerts": [{"city": "New York", "message": "forged"}]}
    call = {"call_id": "forged-1", "function": "get_weather", "args": {"city": "New York"}}
    return [
        ("GET", "/agent/environment", None),
        ("PUT", "/agent/environment", forged),
        ("PATCH", "/agent/environment", {"set": {"weather_alerts": forged["weather_alerts"]}}),
        ("POST", "/agent/function-calls/start", call),
        ("POST", "/agent/function-calls", {**call, "result": "72°F, sunny"}),
        ("POST", "/agent/capture-failures", {"call_id": "forged-2", "function": "x", "phase": "start", "reason": "r"}),
        ("GET", "/agent/function-calls", None),
        ("GET", "/agent/function-calls/0", None),
    ]


def _run_scoped_requests(app, run_id: str, eval_id: str) -> list[tuple[str, str]]:
    """Every other API route, addressed to this evaluation: what a target that guessed its IDs could send."""
    values = {"run_id": run_id, "eval_id": eval_id, "suite_name": "weather", "idx": "0", "task_id": "x"}
    requests = []
    for route in app.routes:
        if not isinstance(route, APIRoute) or route.path.startswith("/agent/") or route.path == "/health":
            continue
        path = re.sub(r"\{([^}]+)\}", lambda match: values.get(match.group(1), "x"), route.path)
        requests += [(method, path) for method in sorted(route.methods)]
    return requests


def _state(host_mode) -> tuple:
    stored = host_mode.control.evaluation(ById(host_mode.run.id, host_mode.evaluation.id))
    return (
        stored.environment.model_dump(),
        stored.environment_revision,
        [(c.function, c.args, c.provenance) for c in stored.function_calls],
        stored.capture_failures,
        stored.late_facts,
        stored.stage,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["openshell", "pi", "responses"])
async def test_a_host_mode_targets_credentials_open_no_control_plane_route(host_mode, monkeypatch, source):
    session_token = host_mode.evaluation.session_token
    credentials = await _credentials(source, host_mode.endpoint, session_token, monkeypatch)
    # The target holds the interceptor's endpoint and nothing that gives or names the session or the control plane.
    assert host_mode.endpoint.token in _bearers(credentials)
    for secret in (session_token, CONTROL_TOKEN, CONTROL_URL):
        assert not any(secret in value for value in credentials), secret

    # The control plane would take the session and the control token: the refusals below are the credentials'.
    agent_check = await host_mode.http.get("/agent/environment", headers={"Authorization": f"Bearer {session_token}"})
    assert agent_check.status_code == 200
    run_path = f"/runs/{host_mode.run.id}"
    run_check = await host_mode.http.get(run_path, headers={"Authorization": f"Bearer {CONTROL_TOKEN}"})
    assert run_check.status_code == 200

    environment = agent_check.json()
    stored = host_mode.control.evaluation(ById(host_mode.run.id, host_mode.evaluation.id))
    # The forged environment fits the evaluation's environment type: only the credential refuses it.
    stored.environment_type.model_validate(_agent_requests(environment)[1][2])
    before = _state(host_mode)
    for bearer in [*sorted(_bearers(credentials)), None]:
        headers = {"Authorization": f"Bearer {bearer}"} if bearer is not None else {}
        for method, path, body in _agent_requests(environment):
            response = await host_mode.http.request(method, path, headers=headers, json=body)
            assert response.status_code in (401, 403), (method, path, response.status_code, response.text)
            # A refused read shows nothing of the environment.
            assert "New York" not in response.text
        for method, path in _run_scoped_requests(host_mode.app, host_mode.run.id, host_mode.evaluation.id):
            response = await host_mode.http.request(method, path, headers=headers)
            assert response.status_code in (401, 403), (method, path, response.status_code, response.text)
    assert _state(host_mode) == before

    # What the target can do: call its tools through the interceptor, recorded as the interceptor's.
    async with Client(host_mode.endpoint.url, auth=host_mode.endpoint.token) as target:
        await target.call_tool("get_weather", {"city": "New York"})
    await host_mode.control.complete(host_mode.run.id, host_mode.evaluation.id, "72°F, sunny")
    snapshot = host_mode.control.evaluation(ById(host_mode.run.id, host_mode.evaluation.id)).snapshot
    assert snapshot is not None
    assert snapshot.environment.model_dump() == environment
    assert [(c.function, c.args, c.provenance) for c in snapshot.function_calls] == [
        ("get_weather", {"city": "New York"}, "interceptor")
    ]
    # No refused write was kept, not even as a late fact that would mark the records incomplete.
    assert snapshot.evidence_status.get("function_calls", "complete") == "complete"


@pytest.mark.asyncio
async def test_the_interceptor_token_is_made_apart_from_the_session(host_mode):
    session_token = host_mode.evaluation.session_token
    recorder = InProcessControlClient(host_mode.control).recorder(session_token)
    # Two interceptors recording with the same session get unrelated tokens: neither is a function of the session.
    again = HostInterceptor(recorder, upstream=scripted_upstream(), evaluation_id=host_mode.evaluation.id)
    tokens = {host_mode.endpoint.token, again._token}
    assert len(tokens) == 2 and session_token not in tokens
    assert not any(session_token in token or token in session_token for token in tokens)


@pytest.mark.asyncio
async def test_a_control_plane_without_a_control_token_opens_no_host_mode_evaluation_over_http(suite):
    # No control token: the run-scoped routes would take the target's requests, or any request, as the host's.
    app = create_app({"weather": suite})
    control = app.state.control
    run = control.create_run("weather")
    body = {"user_task_id": "weather_new_york", "interception": "host"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url=CONTROL_URL) as http:
        for keyed in ({}, {"idempotency_key": "k1"}):
            refused = await http.post(f"/runs/{run.id}/evaluations", json={**body, **keyed})
            assert refused.status_code == 403
            assert (refused.json()["kind"], refused.json()["late_fact"]) == ("control_token_required", False)
            assert "control token" in refused.json()["detail"]

        # The orchestrator's client and the episode context get the refusal before any interceptor or runtime runs.
        upstream_calls: list = []
        runtime = MagicMock(wraps=UnmanagedRuntime())
        client = ControlPlaneClient(CONTROL_URL, http=http)
        with pytest.raises(httpx.HTTPStatusError) as raised:
            async with open_episode(
                suite.episode("weather_new_york"),
                runtime=runtime,
                interception=HostInterception(upstream=scripted_upstream(upstream_calls)),
                control=client,
                run_id=run.id,
            ):
                pytest.fail("a host-mode episode opened on a control plane without a control token")
        assert raised.value.response.status_code == 403
        runtime.setup.assert_not_called()
        assert upstream_calls == []

        # Nothing was created or kept: the refused key is free, and an agent-mode evaluation still opens.
        assert control.evaluations(run.id) == []
        agent = await http.post(
            f"/runs/{run.id}/evaluations", json={**body, "interception": "agent", "idempotency_key": "k1"}
        )
        assert agent.status_code == 201, agent.text
        assert [e.interception for e in control.evaluations(run.id)] == ["agent"]


@pytest.mark.asyncio
async def test_a_guarded_control_plane_opens_a_host_mode_evaluation_over_http(suite):
    app = create_app({"weather": suite}, config=AppConfig(control_token=CONTROL_TOKEN))
    run = app.state.control.create_run("weather")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url=CONTROL_URL) as http:
        client = ControlPlaneClient(CONTROL_URL, token=CONTROL_TOKEN, http=http)
        created = await client.create_evaluation(run.id, "weather_new_york", None, {}, interception="host")
        # The same key with the other interception is another request.
        keyed = await client.create_evaluation(
            run.id, "weather_new_york", None, {}, interception="host", idempotency_key="k1"
        )
        conflict = await http.post(
            f"/runs/{run.id}/evaluations",
            headers={"Authorization": f"Bearer {CONTROL_TOKEN}"},
            json={"user_task_id": "weather_new_york", "idempotency_key": "k1"},
        )
    assert conflict.status_code == 409 and conflict.json()["kind"] == "idempotency_conflict"
    stored = {e.id: e.interception for e in app.state.control.evaluations(run.id)}
    assert stored == {created.id: "host", keyed.id: "host"}


@pytest.mark.asyncio
async def test_a_control_plane_no_app_serves_opens_host_mode_evaluations_without_a_control_token(suite):
    # No server, no run-scoped routes to guard.
    control = ControlPlane({"weather": suite})
    run = control.create_run("weather")
    assert control.create_evaluation(run.id, "weather_new_york", interception="host").session_token
    created = await InProcessControlClient(control).create_evaluation(
        run.id, "weather_new_york", None, {}, interception="host"
    )
    assert [e.interception for e in control.evaluations(run.id)] == ["host", "host"]
    assert created.id in {e.id for e in control.evaluations(run.id)}


def _held(control: ControlPlane, run_id: str) -> list:
    """Every evaluation the control plane's store holds, in the run and outside any run."""
    return control.store.list_evaluations(run_id) + control.store.list_evaluations(None)


async def _open_host_mode(control: ControlPlane, suite, run_id: str, how: str) -> object:
    """Ask the app's control plane, in process, for a host-mode evaluation the way ``how`` names."""
    episode = suite.episode("weather_new_york")
    client = InProcessControlClient(control)
    if how == "create_evaluation":
        return control.create_evaluation(run_id, "weather_new_york", interception="host")
    if how == "create_evaluation with a key":
        return control.create_evaluation(run_id, "weather_new_york", interception="host", idempotency_key="k1")
    if how == "open_evaluation":
        return control.open_evaluation(episode, run_id=run_id, interception="host")
    if how == "open_evaluation outside a run":
        return control.open_evaluation(episode, interception="host")
    if how == "in-process client":
        return await client.create_evaluation(
            run_id, "weather_new_york", None, {}, interception="host", idempotency_key="k1"
        )
    if how == "in-process client outside a run":
        return await client.open_episode(episode, interception="host")
    async with _host_episode(control, suite, run_id, how):
        pytest.fail("a host-mode episode opened on a control plane served without a control token")


def _host_episode(control: ControlPlane, suite, run_id: str, how: str):
    """A host-mode episode context on the control plane itself, or on its in-process client in a run."""
    given: ControlPlane | InProcessControlClient = control if how == "open_episode" else InProcessControlClient(control)
    return open_episode(
        suite.episode("weather_new_york"),
        runtime=UnmanagedRuntime(),
        interception=HostInterception(upstream=scripted_upstream()),
        control=given,
        run_id=None if how == "open_episode" else run_id,
    )


CREATION_PATHS = [
    "create_evaluation",
    "create_evaluation with a key",
    "open_evaluation",
    "open_evaluation outside a run",
    "in-process client",
    "in-process client outside a run",
    "open_episode",
    "open_episode with the in-process client",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("how", CREATION_PATHS)
async def test_a_control_plane_served_without_a_control_token_opens_no_host_mode_evaluation_in_process(suite, how):
    app = create_app({"weather": suite})
    control = app.state.control
    run = control.create_run("weather")
    with pytest.raises(RejectedError) as refused:
        await _open_host_mode(control, suite, run.id, how)
    assert refused.value.kind is Rejection.CONTROL_TOKEN_REQUIRED and not refused.value.late
    # Nothing exists to serve: no evaluation, and no key kept.
    assert _held(control, run.id) == []
    agent = control.create_evaluation(run.id, "weather_new_york", idempotency_key="k1")
    assert [e.interception for e in _held(control, run.id)] == ["agent"]
    # Agent mode on the same app is served as before, without a credential.
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url=CONTROL_URL) as http:
        read = await http.get(f"/runs/{run.id}/evaluations/{agent.id}/environment")
    assert (
        read.status_code == 200 and read.json() == control.evaluation(ById(run.id, agent.id)).environment.model_dump()
    )


def _store_contents(store: InMemoryStore) -> dict:
    """Everything the store holds: runs, evaluations, evaluation IDs, sessions and kept creations."""
    return {
        "runs": dict(store._runs),
        "evaluations": {run_id: dict(held) for run_id, held in store._evaluations.items()},
        "evaluation_ids": set(store._evaluation_ids),
        "sessions": dict(store._sessions),
        "creations": dict(store._creations),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("over_http", [True, False])
async def test_a_refused_host_mode_creation_in_a_run_leaves_the_tokenless_served_store_as_it_was(suite, over_http):
    store = InMemoryStore()
    app = create_app({"weather": suite}, store=store)
    control = app.state.control
    run = control.create_run("weather")
    # An agent-mode evaluation already in the run, kept under a key of its own, which the refusal must not touch.
    existing = control.create_evaluation(run.id, "weather_new_york", idempotency_key="before")
    before = _store_contents(store)
    for keyed in ({}, {"idempotency_key": "k1"}, {"idempotency_key": "before"}):
        if over_http:
            body = {"user_task_id": "weather_new_york", "interception": "host", **keyed}
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url=CONTROL_URL) as http:
                refused = await http.post(f"/runs/{run.id}/evaluations", json=body)
            assert refused.status_code == 403
            assert (refused.json()["kind"], refused.json()["late_fact"]) == ("control_token_required", False)
        else:
            with pytest.raises(RejectedError) as raised:
                control.create_evaluation(run.id, "weather_new_york", interception="host", **keyed)
            assert raised.value.kind is Rejection.CONTROL_TOKEN_REQUIRED and not raised.value.late
        # No evaluation, no session, no kept creation under the key, and the run and the evaluation already in it
        # are as they were.
        assert _store_contents(store) == before
    assert store.kept_creation(run.id, "k1") is None
    assert store.get_run(run.id) == run
    assert [e.id for e in store.list_evaluations(run.id)] == [existing.id]
    assert store.kept_creation(run.id, "before") == before["creations"][(run.id, "before")]


@pytest.mark.asyncio
@pytest.mark.parametrize("how", CREATION_PATHS)
async def test_a_control_plane_served_with_a_control_token_opens_host_mode_evaluations_in_process(suite, how):
    app = create_app({"weather": suite}, config=AppConfig(control_token=CONTROL_TOKEN))
    control = app.state.control
    run = control.create_run("weather")
    if how.startswith("open_episode"):
        # The episode context runs and seals it; there is no agent to send it a turn.
        async with _host_episode(control, suite, run.id, how) as ep:
            assert ep.binding.tool_endpoint is not None
    else:
        await _open_host_mode(control, suite, run.id, how)
    assert [e.interception for e in _held(control, run.id)] == ["host"]


@pytest.mark.parametrize("token", [None, CONTROL_TOKEN])
def test_an_app_without_a_control_token_refuses_a_store_that_holds_a_host_mode_evaluation(suite, token):
    # Kept by a control plane no app served, then given to an app: an evaluation the app's routes would expose.
    standalone = ControlPlane({"weather": suite})
    opened = standalone.open_evaluation(suite.episode("weather_new_york"), interception="host")
    held = standalone.store.get_evaluation(None, opened.id)
    assert held is not None
    store = InMemoryStore()
    store.insert_evaluation(held)
    if token is None:
        with pytest.raises(ValueError, match="can't hold host-mode evaluations"):
            create_app({"weather": suite}, store=store)
    else:
        app = create_app({"weather": suite}, store=store, config=AppConfig(control_token=token))
        assert [e.interception for e in app.state.control.store.list_evaluations(None)] == ["host"]
