from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openshell._proto import openshell_pb2

import midojo.runtimes.openshell as openshell_runtime
from midojo.agent_client import OpenShellAgentClient
from midojo.runtimes import build_runtime
from midojo.runtimes.openshell import OpenShellRuntime, _resolve_policy
from midojo.runtimes.openshell.logs import parse_ocsf_lines
from midojo.suite_definition import OpenShellRuntimeDefinition

POLICY = {"networkPolicies": {"api": {"endpoints": [{"host": "api.example.com", "port": 443}]}}}


def test_build_runtime_passes_the_definition_through():
    definition = OpenShellRuntimeDefinition(image="pi", agent_command=["pi", "-p", "--no-session"], policy=POLICY)
    runtime = build_runtime("shell_suite", definition)
    assert isinstance(runtime, OpenShellRuntime)
    assert runtime.image == "pi"
    assert runtime.agent_command == ["pi", "-p", "--no-session"]
    assert runtime.policy == POLICY


def test_ocsf_pairs_each_exit_with_the_latest_launch_of_its_pid():
    lines = [
        "PROC:LAUNCH [INFO] curl(41) [cmd:curl https://example.com]",
        "PROC:TERMINATE [INFO] curl(41) [exit:6]",
        "PROC:LAUNCH [INFO] cat(41) [cmd:cat notes.txt]",
        "PROC:TERMINATE [INFO] cat(41) [exit:0]",
        "PROC:LAUNCH [INFO] sh(42) [cmd:sh]",
    ]
    assert [p.exit_code for p in parse_ocsf_lines(lines).processes] == [6, 0, None]


def test_configure_sets_gateway_and_control_url():
    runtime = OpenShellRuntime("shell_suite", image="pi")
    runtime.configure(cluster="my-gateway", control_url="http://localhost:8080")
    assert (runtime._cluster, runtime._control_url) == ("my-gateway", "http://localhost:8080")


class TestPolicy:
    """_resolve_policy accepts None (no-op) or an inline dict."""

    def test_none_is_noop(self):
        spec = openshell_pb2.SandboxSpec()
        _resolve_policy(None, spec)
        assert not spec.HasField("policy")

    def test_inline_dict_fills_policy(self):
        spec = openshell_pb2.SandboxSpec()
        inline = {"networkPolicies": {"allow_all": {"endpoints": [{"host": "api.example.com", "port": 443}]}}}
        _resolve_policy(inline, spec)
        assert spec.policy.network_policies["allow_all"].endpoints[0].host == "api.example.com"


class _FakeSandboxClient:
    """Records the SDK calls the runtime makes against one sandbox."""

    def __init__(self):
        self.exec_calls = []
        self.exec_results = []
        self.logs_requests = []
        self.specs = []
        self._stub = self

    def create(self, *, spec, name, **kwargs):
        self.specs.append(spec)
        return SimpleNamespace(name=name)

    def wait_ready(self, name, **kwargs):
        pass

    def exec(self, sandbox_name, command, **kwargs):
        self.exec_calls.append((sandbox_name, command, kwargs))
        return self.exec_results.pop(0) if self.exec_results else SimpleNamespace(exit_code=0, stdout="", stderr="")

    def GetSandboxLogs(self, request, timeout):  # noqa: N802 - mirrors the gRPC stub
        self.logs_requests.append(request)
        return openshell_pb2.GetSandboxLogsResponse()


def _runtime_with_sandbox(**options) -> tuple[OpenShellRuntime, _FakeSandboxClient]:
    runtime = OpenShellRuntime("shell_suite", image="pi", **options)
    client = _FakeSandboxClient()
    runtime._client = client
    runtime._workspace_name = "midojo-shell-abc123"
    runtime._ref = SimpleNamespace(name="midojo-eval-1")
    runtime._start_ms = 1_790_000_000_123
    return runtime, client


def test_exec_addresses_the_sandbox_by_name_within_the_workspace():
    runtime, client = _runtime_with_sandbox()
    runtime.exec_agent("hello", timeout_seconds=30)
    name, _, kwargs = client.exec_calls[0]
    assert name == "midojo-eval-1"
    assert kwargs["workspace"] == "midojo-shell-abc123"


def test_sandbox_environment_never_carries_the_control_token(monkeypatch):
    monkeypatch.setenv("MIDOJO_CONTROL_TOKEN", "control-secret")
    runtime, client = _runtime_with_sandbox(env_vars={"LITELLM_MODEL": "m"})
    runtime.configure(cluster="test", control_url="http://localhost:8090")
    runtime.setup({}, session_token="session", eval_id="e1", user_task_id="t", injection_task_id=None)
    assert dict(client.specs[0].environment) == {
        "LITELLM_MODEL": "m",
        "MIDOJO_SESSION_TOKEN": "session",
        "MIDOJO_URL": "http://host.openshell.internal:8090",
    }


_AGENT_RUN = {"workspace": "midojo-shell-abc123", "workdir": "/sandbox/workdir"}


@pytest.mark.asyncio
async def test_multiturn_conversation_runs_the_agent_command_once_per_turn_in_one_sandbox():
    runtime, client = _runtime_with_sandbox()
    runtime._agent_command = ["agent", "-p"]
    client.exec_results = [
        SimpleNamespace(exit_code=0, stdout=" Hi there \n", stderr=""),
        SimpleNamespace(exit_code=0, stdout="Still here\n", stderr=""),
    ]
    agent = OpenShellAgentClient(runtime, timeout=60)
    assert agent.supports_multiturn
    conversation = agent.start_conversation(session_token="session-a", turn_count=2)
    assert await conversation.send("Hello") == "Hi there"
    assert await conversation.send("Are you there?") == "Still here"
    assert client.exec_calls == [
        (
            "midojo-eval-1",
            ["agent", "-p", "Hello"],
            {**_AGENT_RUN, "timeout_seconds": 60, "env": {"MIDOJO_TURN_INDEX": "0", "MIDOJO_TURN_COUNT": "2"}},
        ),
        (
            "midojo-eval-1",
            ["agent", "-p", "Are you there?"],
            {**_AGENT_RUN, "timeout_seconds": 60, "env": {"MIDOJO_TURN_INDEX": "1", "MIDOJO_TURN_COUNT": "2"}},
        ),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("via", ["send_task", "conversation"])
async def test_single_turn_runs_the_agent_command_as_before(via):
    runtime, client = _runtime_with_sandbox()
    runtime._agent_command = ["agent", "-p"]
    client.exec_results = [SimpleNamespace(exit_code=0, stdout="Done\n", stderr="")]
    agent = OpenShellAgentClient(runtime)
    if via == "send_task":
        reply = await agent.send_task("Hello", session_token="session-a")
    else:
        reply = await agent.start_conversation(session_token="session-a", turn_count=1).send("Hello")
    assert reply == "Done"
    # No turn variables: the same command line and exec options as before multi-turn support.
    assert client.exec_calls == [("midojo-eval-1", ["agent", "-p", "Hello"], {**_AGENT_RUN, "timeout_seconds": 300})]


def test_log_fetch_scopes_to_the_sandbox_since_the_evaluation_started():
    runtime, client = _runtime_with_sandbox()
    runtime._read_ocsf_messages()
    request = client.logs_requests[0]
    assert request.sandbox == "midojo-eval-1"
    assert request.workspace_scope.workspace == "midojo-shell-abc123"
    assert request.since_time.ToMilliseconds() == 1_790_000_000_123
    assert list(request.sources) == ["sandbox"]


class TestLogSync:
    """The OCSF read waits for a marker so late-pushed agent events are included."""

    def _runtime(self, reads):
        runtime = OpenShellRuntime("shell_suite", image="pi")
        runtime._exec = MagicMock()
        runtime._read_ocsf_messages = MagicMock(side_effect=reads)
        return runtime

    def test_waits_for_marker_and_drops_its_events(self, monkeypatch):
        monkeypatch.setattr(openshell_runtime.time, "sleep", lambda _s: None)
        agent_event = "NET:OPEN [MED] DENIED /usr/bin/curl(0) -> evil.test:443 [reason:transparent_tcp_policy_denied]"

        def marker_from_exec():
            script = runtime._exec.call_args.args[0][2]
            host = script.split()[2]
            return f"NET:REFUSE [MED] DENIED {host} [reason:policy_dns_ineligible]"

        reads = iter([[], [agent_event], None])

        def read():
            batch = next(reads)
            return batch if batch is not None else [agent_event, marker_from_exec()]

        runtime = self._runtime(read)
        assert runtime._sync_ocsf_messages() == [agent_event]
        assert runtime._read_ocsf_messages.call_count == 3

    def test_returns_last_read_when_marker_never_arrives(self, monkeypatch):
        clock = iter(range(0, 100, 5))
        monkeypatch.setattr(openshell_runtime.time, "time", lambda: next(clock))
        monkeypatch.setattr(openshell_runtime.time, "sleep", lambda _s: None)
        allowed = "NET:OPEN [INFO] ALLOWED /usr/bin/curl(0) -> ok.test:443"
        runtime = self._runtime(lambda: [allowed])
        assert runtime._sync_ocsf_messages() == [allowed]
