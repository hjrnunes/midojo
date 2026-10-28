"""midojo-run in host mode: one interceptor per evaluation, the agent given only its endpoint, stopped after it.

The upstream is the scripted MCP server in this process and the agent an MCP
client on loopback. No model and no network beyond loopback.
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock

import click
import httpx
import pytest
from click.testing import CliRunner
from fastmcp import Client
from fastmcp.exceptions import ToolError

from midojo import orchestrator
from midojo.agent_client import A2AAgentClient, AgentClient, SimpleHTTPAgentClient
from midojo.control_plane.config import CONTROL_TOKEN_ENV, AppConfig
from midojo.control_plane.inprocess_control import InProcessControlClient
from midojo.control_plane.service import ControlPlane
from midojo.interception_host import HostInterception
from midojo.orchestrator import run_benchmark
from midojo.runtimes import UnmanagedRuntime
from midojo.runtimes.openshell import OpenShellRuntime
from midojo.tool_endpoint import ToolEndpoint
from tests.fixtures.scripted_target.upstream import scripted_upstream


class ToolCallingAgent(AgentClient):
    """Calls ``get_weather`` at the tool endpoint it's given, and keeps what it was given."""

    supports_host_interception = True

    def __init__(self, fail: bool = False) -> None:
        self.given: list[dict] = []
        self.fail = fail

    async def send_task(self, prompt, *, session_token=None, tool_endpoint=None):
        self.given.append({"session_token": session_token, "tool_endpoint": tool_endpoint})
        assert tool_endpoint is not None
        async with Client(tool_endpoint.url, auth=tool_endpoint.token) as tools:
            weather = await tools.call_tool("get_weather", {"city": "New York"})
        if self.fail:
            raise RuntimeError("Agent failed")
        return f"New York is {weather.content[0].text}"  # type: ignore[union-attr]


class SetupRecordingRuntime(UnmanagedRuntime):
    """An unmanaged runtime that keeps what each setup was given."""

    def __init__(self) -> None:
        self.setups: list[dict] = []

    def setup(self, injections, **kwargs) -> None:
        self.setups.append(kwargs)


@pytest.fixture
def control(suite) -> ControlPlane:
    return ControlPlane({"weather": suite}, config=AppConfig(closure_wait_seconds=0))


async def benchmark(control, suite, agent, runtime, tmp_path, interception, **kwargs):
    control_client = kwargs.pop("control_client", InProcessControlClient)
    async with control_client(control) as in_process:
        return await run_benchmark(
            control_url="http://control",
            agent_client=agent,
            agent_uri="http://agent",
            protocol=kwargs.pop("protocol", "openai"),
            suite=suite,
            suite_name="weather",
            user_task_ids=["weather_new_york", "warmest_city"],
            injection_task_ids=[],
            logdir=tmp_path,
            control=in_process,
            runtime=runtime,
            interception=interception,
        )


async def refused(endpoint: ToolEndpoint) -> bool:
    """Whether nothing serves the endpoint any more."""
    try:
        async with httpx.AsyncClient() as http:
            await http.post(endpoint.url, headers=endpoint.headers, json={})
    except httpx.ConnectError:
        return True
    return False


@pytest.mark.asyncio
async def test_each_evaluation_gets_its_own_interceptor_and_only_its_endpoint(control, suite, tmp_path):
    agent, runtime = ToolCallingAgent(), SetupRecordingRuntime()
    await benchmark(control, suite, agent, runtime, tmp_path, HostInterception(upstream=scripted_upstream()))

    endpoints = [given["tool_endpoint"] for given in agent.given]
    assert [given["session_token"] for given in agent.given] == [None, None]
    assert len({e.url for e in endpoints}) == len({e.token for e in endpoints}) == 2
    # The runtime was given the same endpoint as the agent, and no session.
    assert [setup.get("tool_endpoint") for setup in runtime.setups] == endpoints
    assert all("session_token" not in setup for setup in runtime.setups)
    # Every interceptor was stopped after its evaluation.
    assert [await refused(endpoint) for endpoint in endpoints] == [True, True]

    [run] = control.store.list_runs()
    evaluations = control.evaluations(run.id)
    assert [(e.interception, e.stage.value, e.graded) for e in evaluations] == [("host", "sealed", True)] * 2
    for evaluation in evaluations:
        assert [(c.function, c.provenance) for c in evaluation.function_calls] == [("get_weather", "interceptor")]
    assert evaluations[0].grade is not None and evaluations[0].grade.utility is True


@pytest.mark.asyncio
async def test_a_call_after_the_seal_is_refused_by_a_live_interceptor_and_kept_as_a_late_fact(control, suite, tmp_path):
    agent = ToolCallingAgent()
    after_seal: list[str] = []

    class CallingAfterTheSeal(InProcessControlClient):
        """Once each evaluation is sealed, the target tries one more call, as a slow agent would."""

        async def complete_evaluation(self, run_id, eval_id, agent_output):
            result = await super().complete_evaluation(run_id, eval_id, agent_output)
            endpoint = agent.given[-1]["tool_endpoint"]
            async with Client(endpoint.url, auth=endpoint.token) as tools:
                with pytest.raises(ToolError):
                    await tools.call_tool("get_weather", {"city": "Chicago"})
            after_seal.append(eval_id)
            return result

    interception = HostInterception(upstream=scripted_upstream())
    await benchmark(
        control, suite, agent, SetupRecordingRuntime(), tmp_path, interception, control_client=CallingAfterTheSeal
    )

    [run] = control.store.list_runs()
    evaluations = control.evaluations(run.id)
    assert after_seal == [e.id for e in evaluations]
    for evaluation in evaluations:
        # The refused call is a late fact, not a record, and the sealed grade is unchanged.
        assert [c.args for c in evaluation.function_calls] == [{"city": "New York"}]
        [fact] = evaluation.late_facts
        assert (fact.event, fact.by, fact.stage.value) == ("start_call", "interceptor", "sealed")
        assert fact.payload["args"] == {"city": "Chicago"}
        assert evaluation.grade is not None and evaluation.grade.utility is True
    # The interceptors still stopped once their evaluations were done.
    assert [await refused(given["tool_endpoint"]) for given in agent.given] == [True, True]


@pytest.mark.asyncio
async def test_the_interceptor_stops_when_the_agent_fails(control, suite, tmp_path):
    agent = ToolCallingAgent(fail=True)
    outcome = await benchmark(
        control, suite, agent, SetupRecordingRuntime(), tmp_path, HostInterception(upstream=scripted_upstream())
    )
    # Each pair failed and was recorded as failed; the run went on to the next.
    assert (outcome.graded, outcome.failed) == (0, 2)
    assert [await refused(given["tool_endpoint"]) for given in agent.given] == [True, True]


@pytest.mark.asyncio
async def test_host_mode_is_refused_before_a_run_for_a_client_that_cant_pass_an_endpoint(control, suite, tmp_path):
    with pytest.raises(click.UsageError, match="can't give the agent a tool endpoint"):
        await benchmark(
            control,
            suite,
            SimpleHTTPAgentClient("http://agent"),
            UnmanagedRuntime(),
            tmp_path,
            HostInterception(upstream=scripted_upstream()),
            protocol="http",
        )
    assert control.store.list_runs() == []


# --- the CLI ---


@pytest.fixture
def cli_benchmark(monkeypatch, plain_suite):
    for name in ("MODEL_NAME", "MCP_SERVER_URL", "OPENAI_API_KEY", "UPSTREAM_MCP_URL"):
        monkeypatch.delenv(name, raising=False)
    # Host mode needs the control plane's token; agent mode passes it through as it always did.
    monkeypatch.setenv(CONTROL_TOKEN_ENV, CONTROL_TOKEN)
    mock = AsyncMock(return_value=orchestrator.BenchmarkOutcome(graded=1, failed=0))
    monkeypatch.setattr(orchestrator, "run_benchmark", mock)
    return mock


CONTROL_TOKEN = "control-secret"
PI = ["--suite", "plain", "--agent-uri", "/agent", "--protocol", "pi"]
OPENSHELL = ["--suite", "document_assistant", "--gateway", "g"]
UPSTREAM = ["--upstream-mcp-url", "http://localhost:9000/mcp"]


def test_the_cli_runs_an_unmanaged_agent_in_host_mode_on_loopback(cli_benchmark):
    result = CliRunner().invoke(orchestrator.main, [*PI, "--interception", "host", *UPSTREAM])
    assert result.exit_code == 0, result.output
    interception = cli_benchmark.call_args.kwargs["interception"]
    assert interception == HostInterception(
        upstream="http://localhost:9000/mcp", bind_host="127.0.0.1", port=0, advertise_host=None
    )
    assert cli_benchmark.call_args.kwargs["control_token"] == CONTROL_TOKEN


@pytest.mark.parametrize("args", [[*PI, "--interception", "host", *UPSTREAM], [*OPENSHELL, "--interception", "host"]])
def test_the_cli_refuses_host_mode_without_a_control_token(cli_benchmark, args):
    # Without one, the control plane's run-scoped routes are open to anything that reaches it, the agent included.
    result = CliRunner().invoke(orchestrator.main, args, env={CONTROL_TOKEN_ENV: None})
    assert result.exit_code == 2
    assert f"--interception host requires --control-token or {CONTROL_TOKEN_ENV}" in result.output
    cli_benchmark.assert_not_called()


def test_the_cli_runs_in_agent_mode_without_a_control_token(cli_benchmark):
    result = CliRunner().invoke(orchestrator.main, PI, env={CONTROL_TOKEN_ENV: None})
    assert result.exit_code == 0, result.output
    assert cli_benchmark.call_args.kwargs["control_token"] is None


def test_the_cli_runs_in_agent_mode_by_default(cli_benchmark):
    result = CliRunner().invoke(orchestrator.main, PI)
    assert result.exit_code == 0, result.output
    assert cli_benchmark.call_args.kwargs["interception"] is None


def test_the_cli_gives_an_openshell_sandbox_the_interceptor_and_not_the_control_plane(cli_benchmark):
    args = [*OPENSHELL, "--interception", "host", *UPSTREAM, "--interceptor-bind", "0.0.0.0"]
    result = CliRunner().invoke(orchestrator.main, [*args, "--interceptor-port", "8765"])
    assert result.exit_code == 0, result.output
    kwargs = cli_benchmark.call_args.kwargs
    assert kwargs["interception"] == HostInterception(
        upstream="http://localhost:9000/mcp", bind_host="0.0.0.0", port=8765, advertise_host="host.openshell.internal"
    )
    runtime = kwargs["runtime"]
    assert isinstance(runtime, OpenShellRuntime)
    assert (runtime._interceptor_url, runtime._control_url) == ("http://host.openshell.internal:8765/mcp", "")


@pytest.mark.parametrize("advertise", ["2001:db8::1", "[2001:db8::1]"])
def test_the_cli_gives_an_openshell_sandbox_an_ipv6_interceptor_in_brackets_once(cli_benchmark, advertise):
    args = [*OPENSHELL, "--interception", "host", *UPSTREAM, "--interceptor-bind", "::", "--interceptor-port", "8765"]
    result = CliRunner().invoke(orchestrator.main, [*args, "--interceptor-advertise-host", advertise])
    assert result.exit_code == 0, result.output
    kwargs = cli_benchmark.call_args.kwargs
    assert (kwargs["interception"].bind_host, kwargs["interception"].advertise_host) == ("::", "2001:db8::1")
    assert kwargs["runtime"]._interceptor_url == "http://[2001:db8::1]:8765/mcp"


def test_a_responses_protocol_needs_no_mcp_server_url_in_host_mode(cli_benchmark):
    args = ["--suite", "plain", "--agent-uri", "http://agent", "--protocol", "openai", "--model-name", "m"]
    result = CliRunner().invoke(
        orchestrator.main, [*args, "--interception", "host", *UPSTREAM], env={"OPENAI_API_KEY": "k"}
    )
    assert result.exit_code == 0, result.output
    assert cli_benchmark.call_args.kwargs["agent_client"].mcp_server_url is None


@pytest.mark.parametrize(
    "args,message",
    [
        ([*PI, "--interception", "host"], "requires --upstream-mcp-url"),
        ([*OPENSHELL, "--interception", "host", *UPSTREAM, "--interceptor-bind", "0.0.0.0"], "--interceptor-port"),
        ([*OPENSHELL, "--interception", "host", *UPSTREAM, "--interceptor-port", "8765"], "host's loopback"),
        *[
            (
                [
                    *OPENSHELL,
                    "--interception",
                    "host",
                    *UPSTREAM,
                    "--interceptor-port",
                    "8765",
                    "--interceptor-bind",
                    h,
                ],
                "host's loopback",
            )
            for h in ("127.0.0.2", "::1", "localhost", "::ffff:127.0.0.1")
        ],
        *[
            (
                [
                    *OPENSHELL,
                    "--interception",
                    "host",
                    *UPSTREAM,
                    "--interceptor-port",
                    "8765",
                    "--interceptor-bind",
                    "0.0.0.0",
                    "--interceptor-advertise-host",
                    h,
                ],
                "can't reach an interceptor advertised at",
            )
            for h in ("127.0.0.1", "127.1.2.3", "::1", "[::1]", "0.0.0.0", "::", "localhost", "LocalHost")
        ],
        ([*PI, "--interceptor-port", "8765"], "apply only to --interception host"),
        ([*PI, "--interceptor-advertise-host", "h"], "apply only to --interception host"),
    ],
)
def test_the_cli_refuses_host_mode_settings_a_target_couldnt_reach(cli_benchmark, args, message):
    result = CliRunner().invoke(orchestrator.main, args)
    assert result.exit_code == 2
    assert message in result.output
    cli_benchmark.assert_not_called()


# --- a suite's host-side tools ---

WEATHER = ["--suite", "weather", "--gateway", "g"]
WEATHER_HOST = [*WEATHER, "--interception", "host", *UPSTREAM, "--interceptor-bind", "0.0.0.0"]


def test_the_cli_gives_every_interceptor_the_suites_host_tools(cli_benchmark):
    from suites.weather import host_tools

    result = CliRunner().invoke(orchestrator.main, [*WEATHER_HOST, "--interceptor-port", "8766"])
    assert result.exit_code == 0, result.output
    interception = cli_benchmark.call_args.kwargs["interception"]
    assert interception.tools is host_tools.register
    assert (interception.port, interception.advertise_host) == (8766, "host.openshell.internal")


def test_the_cli_refuses_a_suite_with_host_tools_in_agent_mode(cli_benchmark):
    # Its tools live on the host-side interceptor: in agent mode the agent would have none that record.
    result = CliRunner().invoke(orchestrator.main, WEATHER)
    assert result.exit_code == 2
    assert "serves its tools on the host-side interceptor (HOST_TOOLS)" in result.output
    assert "--interception host" in result.output
    cli_benchmark.assert_not_called()


MINIBANK = ["--suite", "minibank", "--agent-uri", "http://agent", "--protocol", "a2a"]


def test_the_cli_runs_the_minibank_a2a_agent_in_host_mode_with_its_host_tools(cli_benchmark):
    from suites.minibank import host_tools

    result = CliRunner().invoke(orchestrator.main, [*MINIBANK, "--interception", "host", *UPSTREAM])
    assert result.exit_code == 0, result.output
    kwargs = cli_benchmark.call_args.kwargs
    assert kwargs["interception"] == HostInterception(upstream="http://localhost:9000/mcp", tools=host_tools.register)
    assert isinstance(kwargs["agent_client"], A2AAgentClient)
    assert kwargs["agent_client"].supports_host_interception


@pytest.mark.parametrize("protocol", ["a2a", "http", "openai"])
def test_the_cli_refuses_minibank_in_agent_mode(cli_benchmark, protocol):
    args = ["--suite", "minibank", "--agent-uri", "http://agent", "--protocol", protocol, "--model-name", "m"]
    result = CliRunner().invoke(orchestrator.main, args, env={"OPENAI_API_KEY": "k", "MCP_SERVER_URL": "http://t/mcp"})
    assert result.exit_code == 2
    assert "Suite 'minibank' serves its tools on the host-side interceptor (HOST_TOOLS)" in result.output
    cli_benchmark.assert_not_called()


def test_a_suite_without_host_tools_gives_its_interceptors_none(cli_benchmark):
    result = CliRunner().invoke(orchestrator.main, [*PI, "--interception", "host", *UPSTREAM])
    assert result.exit_code == 0, result.output
    assert cli_benchmark.call_args.kwargs["interception"].tools is None


@pytest.mark.parametrize("reference", [42, "tests.test_orchestrator_host_mode", ":register", "module:"])
def test_a_suites_host_tools_declaration_must_be_module_and_function(cli_benchmark, monkeypatch, reference):
    import suites.weather

    monkeypatch.setattr(suites.weather, "HOST_TOOLS", reference)
    result = CliRunner().invoke(orchestrator.main, WEATHER)
    assert result.exit_code == 2
    assert f"suites.weather.HOST_TOOLS must be a 'module:function' string, got {reference!r}." in result.output
    cli_benchmark.assert_not_called()


@pytest.mark.parametrize(
    "reference", ["tests.test_orchestrator_host_mode:CONTROL_TOKEN", "tests.test_orchestrator_host_mode:missing"]
)
def test_a_suites_host_tools_must_name_a_function(reference):
    with pytest.raises(click.UsageError, match="is not a function"):
        orchestrator._load_host_tools(reference)


@pytest.fixture
def declared_host_tools(tmp_path, monkeypatch):
    """An out-of-tree suite declaring HOST_TOOLS in a module whose import raises: the two modules' names."""
    package = tmp_path / "host_tools_probe"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "suite.py").write_text('HOST_TOOLS = "host_tools_probe.tools:register"\n')
    (package / "tools.py").write_text('raise RuntimeError("the host tools were imported")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    # The run's suite is the weather suite's definition, under the probe module's name.
    weather = orchestrator.get_suite("weather")
    monkeypatch.setattr(orchestrator, "get_suite", lambda _name: weather)
    yield "host_tools_probe.suite", "host_tools_probe.tools"
    for name in ("host_tools_probe", "host_tools_probe.suite", "host_tools_probe.tools"):
        sys.modules.pop(name, None)


def test_a_refused_agent_mode_run_never_imports_the_host_tools(cli_benchmark, declared_host_tools):
    suite_module, tools_module = declared_host_tools
    result = CliRunner().invoke(orchestrator.main, ["--suite", suite_module, "--gateway", "g"])
    # Refused for its declaration alone: the module it names, whose import raises, was never imported.
    assert result.exit_code == 2, result.output
    assert "serves its tools on the host-side interceptor (HOST_TOOLS)" in result.output
    assert suite_module in sys.modules and tools_module not in sys.modules
    cli_benchmark.assert_not_called()


def test_a_host_mode_run_imports_the_host_tools(cli_benchmark, declared_host_tools):
    suite_module, tools_module = declared_host_tools
    args = ["--suite", suite_module, "--gateway", "g", "--interception", "host", *UPSTREAM]
    result = CliRunner().invoke(
        orchestrator.main, [*args, "--interceptor-bind", "0.0.0.0", "--interceptor-port", "8766"]
    )
    assert isinstance(result.exception, RuntimeError) and "the host tools were imported" in str(result.exception)
    cli_benchmark.assert_not_called()
