from unittest.mock import AsyncMock

import pytest
import yaml
from click.testing import CliRunner

from midojo import orchestrator
from midojo.yaml_task_suite import YAMLTaskSuite


@pytest.fixture
def benchmark(monkeypatch):
    for name in ("MODEL_NAME", "MCP_SERVER_URL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    benchmark = AsyncMock(return_value=orchestrator.BenchmarkOutcome(graded=1, failed=0))
    monkeypatch.setattr(orchestrator, "run_benchmark", benchmark)
    return benchmark


@pytest.mark.parametrize(
    "protocol,missing",
    [
        ("ogx", "MODEL_NAME"),
        ("openai", "MODEL_NAME"),
        ("ogx", "MCP_SERVER_URL"),
        ("openai", "MCP_SERVER_URL"),
        ("openai", "OPENAI_API_KEY"),
    ],
)
def test_missing_inference_config_stops_before_benchmark(benchmark, protocol, missing):
    env = {"MODEL_NAME": "model-id", "MCP_SERVER_URL": "http://tools/mcp", "OPENAI_API_KEY": "test-key"}
    env.pop(missing)
    result = CliRunner().invoke(
        orchestrator.main,
        ["--suite", "minibank", "--agent-uri", "http://agent", "--protocol", protocol],
        env=env,
    )
    assert result.exit_code == 2
    assert missing in result.output
    benchmark.assert_not_called()


@pytest.mark.parametrize("protocol", ["ogx", "openai", "http"])
def test_cli_resolves_inference_config_only_for_responses_protocols(benchmark, protocol):
    args = ["--suite", "minibank", "--agent-uri", "http://agent", "--protocol", protocol]
    env = {}
    if protocol != "http":
        env = {"MODEL_NAME": "env-model", "MCP_SERVER_URL": "http://env-tools/mcp"}
    if protocol == "openai":
        env["OPENAI_API_KEY"] = "test-key"
        args += ["--model-name", "cli-model", "--mcp-server-url", "http://cli-tools/mcp"]
    result = CliRunner().invoke(orchestrator.main, args, env=env)
    assert result.exit_code == 0, result.exception
    benchmark.assert_awaited_once()
    agent = benchmark.call_args.kwargs["agent_client"]
    if protocol == "ogx":
        assert agent.model == "env-model"
        assert agent.mcp_server_url == "http://env-tools/mcp"
    elif protocol == "openai":
        assert agent.model == "cli-model"
        assert agent.mcp_server_url == "http://cli-tools/mcp"
        assert agent.api_key == "test-key"


@pytest.mark.parametrize(
    "args,message",
    [
        (["--suite", "document_assistant"], "requires --gateway"),
        (["--suite", "document_assistant", "--gateway", "g", "--protocol", "http"], "instead of --agent-uri"),
        (["--suite", "minibank", "--gateway", "g", "--agent-uri", "http://agent", "--protocol", "http"], "unmanaged"),
        (["--suite", "minibank", "--agent-uri", "http://agent"], "requires --agent-uri and --protocol"),
    ],
)
def test_cli_flags_must_match_the_suite_runtime(benchmark, args, message):
    result = CliRunner().invoke(orchestrator.main, args)
    assert result.exit_code == 2
    assert message in result.output
    benchmark.assert_not_called()


def test_openshell_suite_runs_on_the_gateway(benchmark):
    result = CliRunner().invoke(orchestrator.main, ["--suite", "document_assistant", "--gateway", "g"])
    assert result.exit_code == 0, result.exception
    kwargs = benchmark.call_args.kwargs
    assert isinstance(kwargs["agent_client"], orchestrator.OpenShellAgentClient)
    assert (kwargs["agent_uri"], kwargs["protocol"]) == ("g", "openshell")


@pytest.mark.parametrize(
    "suite_name,task_id,args,via",
    [
        ("minibank", "account_balance", ["--agent-uri", "http://agent", "--protocol", "http"], "--protocol http"),
        ("minibank", "account_balance", ["--agent-uri", "http://agent", "--protocol", "a2a"], "--protocol a2a"),
    ],
)
def test_multiturn_task_exits_before_creating_a_run(monkeypatch, suite_name, task_id, args, via):
    suite = orchestrator.get_suite(suite_name)
    monkeypatch.setattr(suite.user_tasks[task_id], "turns", ["First message", "Follow-up"])
    monkeypatch.setattr(orchestrator, "get_suite", lambda name: suite)
    create_run = AsyncMock()
    monkeypatch.setattr(orchestrator.ControlPlaneClient, "create_run", create_run)
    result = CliRunner().invoke(orchestrator.main, ["--suite", suite_name, "-ut", task_id, *args])
    assert result.exit_code == 2
    assert f"User task '{task_id}' has 2 turns" in result.output
    assert via in result.output
    create_run.assert_not_called()


@pytest.mark.parametrize(
    "args,env,expected",
    [
        (["--control-token", "cli-token"], {"MIDOJO_CONTROL_TOKEN": "env-token"}, "cli-token"),
        ([], {"MIDOJO_CONTROL_TOKEN": "env-token"}, "env-token"),
        ([], {}, None),
    ],
)
def test_cli_passes_the_control_token_to_the_benchmark(benchmark, monkeypatch, args, env, expected):
    monkeypatch.delenv("MIDOJO_CONTROL_TOKEN", raising=False)
    args = ["--suite", "minibank", "--agent-uri", "http://agent", "--protocol", "http", *args]
    result = CliRunner().invoke(orchestrator.main, args, env=env)
    assert result.exit_code == 0, result.exception
    assert benchmark.call_args.kwargs["control_token"] == expected


def test_multiturn_task_on_the_openshell_runtime_is_not_refused(monkeypatch):
    suite = orchestrator.get_suite("document_assistant")
    monkeypatch.setattr(suite.user_tasks["summarize_q4_report"], "turns", ["First message", "Follow-up"])
    monkeypatch.setattr(orchestrator, "get_suite", lambda name: suite)
    create_run = AsyncMock(side_effect=RuntimeError("stop after the turn check"))
    monkeypatch.setattr(orchestrator.ControlPlaneClient, "create_run", create_run)
    result = CliRunner().invoke(
        orchestrator.main, ["--suite", "document_assistant", "-ut", "summarize_q4_report", "--gateway", "g"]
    )
    assert "turns" not in result.output
    assert isinstance(result.exception, RuntimeError)
    create_run.assert_awaited_once()


@pytest.fixture
def slow_suite(monkeypatch, tmp_path):
    """An openshell suite that sets its own agent timeout."""
    path = tmp_path / "suite.yaml"
    path.write_text(yaml.safe_dump({"agent_runtime": {"image": "pi", "timeout_seconds": 900}}))
    monkeypatch.setattr(orchestrator, "get_suite", lambda _name: YAMLTaskSuite("slow", path))


@pytest.mark.parametrize("args,timeout", [([], 900), (["--agent-timeout", "1.5"], 1.5)])
def test_agent_timeout_flag_overrides_the_suite(benchmark, slow_suite, args, timeout):
    result = CliRunner().invoke(orchestrator.main, ["--suite", "slow", "--gateway", "g", *args])
    assert result.exit_code == 0, result.exception
    assert benchmark.call_args.kwargs["agent_client"].timeout == timeout


def test_openshell_agent_timeout_defaults_to_300(benchmark):
    result = CliRunner().invoke(orchestrator.main, ["--suite", "document_assistant", "--gateway", "g"])
    assert result.exit_code == 0, result.exception
    assert benchmark.call_args.kwargs["agent_client"].timeout == 300


@pytest.mark.parametrize("protocol", ["http", "a2a", "pi", "ogx", "openai"])
@pytest.mark.parametrize("args,timeout", [([], 120), (["--agent-timeout", "600"], 600)])
def test_agent_timeout_flag_reaches_unmanaged_clients(benchmark, protocol, args, timeout):
    env = {"MODEL_NAME": "model-id", "MCP_SERVER_URL": "http://tools/mcp", "OPENAI_API_KEY": "test-key"}
    cli = ["--suite", "minibank", "--agent-uri", "http://agent", "--protocol", protocol, *args]
    result = CliRunner().invoke(orchestrator.main, cli, env=env)
    assert result.exit_code == 0, result.exception
    assert benchmark.call_args.kwargs["agent_client"].timeout == timeout


@pytest.mark.parametrize("value", ["0", "-5", "soon", "nan", "inf"])
def test_agent_timeout_flag_must_be_positive_and_finite(benchmark, value):
    args = ["--suite", "document_assistant", "--gateway", "g", "--agent-timeout", value]
    result = CliRunner().invoke(orchestrator.main, args)
    assert result.exit_code == 2
    assert "--agent-timeout" in result.output
    benchmark.assert_not_called()
