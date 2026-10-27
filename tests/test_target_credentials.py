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
from midojo.control_plane.config import CONTROL_TOKEN_ENV, AppConfig
from midojo.control_plane.inprocess_control import InProcessControlClient
from midojo.control_plane.main import create_app
from midojo.control_plane.service import ById
from midojo.interception_host import HostInterceptor
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
