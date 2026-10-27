import json
import os
import re
import subprocess
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock

import pytest
from openshell import ExecResult, SandboxRef, SandboxStatusRef
from openshell._proto import openshell_pb2
from pydantic import ValidationError

import midojo.runtimes.openshell as openshell_runtime
from midojo.agent_client import OpenShellAgentClient
from midojo.runtimes import build_runtime
from midojo.runtimes.openshell import (
    BUILTIN_POLICY_DIGEST,
    OpenShellObservations,
    OpenShellRuntime,
    _policy_digest,
    _resolve_policy,
)
from midojo.runtimes.openshell.logs import parse_ocsf_lines
from midojo.suite_definition import OpenShellRuntimeDefinition
from midojo.tool_endpoint import ToolEndpoint

POLICY = {"networkPolicies": {"api": {"endpoints": [{"host": "api.example.com", "port": 443}]}}}


def test_build_runtime_passes_the_definition_through():
    definition = OpenShellRuntimeDefinition(image="pi", agent_command=["pi", "-p", "--no-session"], policy=POLICY)
    runtime = build_runtime("shell_suite", definition)
    assert isinstance(runtime, OpenShellRuntime)
    assert runtime.image == "pi"
    assert runtime.agent_command == ["pi", "-p", "--no-session"]
    assert runtime.policy == POLICY
    assert runtime.agent_timeout_seconds == 300


@pytest.mark.parametrize("inject_session", [True, False])
def test_build_runtime_passes_inject_session_through(inject_session):
    definition = OpenShellRuntimeDefinition(image="pi", inject_session=inject_session)
    runtime = build_runtime("shell_suite", definition)
    assert isinstance(runtime, OpenShellRuntime)
    assert runtime.inject_session is inject_session


def test_a_sandbox_gets_the_session_unless_the_suite_says_otherwise():
    assert OpenShellRuntimeDefinition(image="pi").inject_session is True
    with pytest.raises(ValidationError):
        OpenShellRuntimeDefinition.model_validate({"image": "pi", "inject_session": "no"})


def test_build_runtime_passes_the_agent_timeout_through():
    runtime = build_runtime("shell_suite", OpenShellRuntimeDefinition(image="pi", timeout_seconds=900))
    assert isinstance(runtime, OpenShellRuntime)
    assert runtime.agent_timeout_seconds == 900


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

    def test_empty_dict_is_noop(self):
        # `policy: {}` sets no field, so it means no policy, as when it is omitted.
        spec = openshell_pb2.SandboxSpec()
        _resolve_policy({}, spec)
        assert not spec.HasField("policy")

    def test_inline_dict_fills_policy(self):
        spec = openshell_pb2.SandboxSpec()
        inline = {"networkPolicies": {"allow_all": {"endpoints": [{"host": "api.example.com", "port": 443}]}}}
        _resolve_policy(inline, spec)
        assert spec.policy.network_policies["allow_all"].endpoints[0].host == "api.example.com"


def _digest(policy: dict | None) -> str:
    spec = openshell_pb2.SandboxSpec()
    _resolve_policy(policy, spec)
    return _policy_digest(spec)


TWO_RULES = {
    "networkPolicies": {
        "api": {"endpoints": [{"host": "api.example.com", "port": 443}]},
        "control": {"endpoints": [{"host": "host.openshell.internal", "port": 8090}]},
    }
}
# The same policy, with every mapping's keys in reverse order.
TWO_RULES_REORDERED = {
    "networkPolicies": {
        "control": {"endpoints": [{"port": 8090, "host": "host.openshell.internal"}]},
        "api": {"endpoints": [{"port": 443, "host": "api.example.com"}]},
    }
}


class TestPolicyDigest:
    """The policy digest identifies the effective policy, not its YAML spelling."""

    def test_is_a_sha256_of_the_policy(self):
        assert _digest(POLICY).startswith("sha256:")
        assert len(_digest(POLICY).removeprefix("sha256:")) == 64

    def test_stable_across_key_order(self):
        assert _digest(TWO_RULES) == _digest(TWO_RULES_REORDERED)

    def test_changes_when_the_policy_changes(self):
        changed = {"networkPolicies": {"api": {"endpoints": [{"host": "api.example.com", "port": 8443}]}}}
        assert _digest(POLICY) != _digest(changed)
        assert _digest(POLICY) != _digest(TWO_RULES)

    def test_no_policy_is_the_builtin_marker(self):
        assert _digest(None) == BUILTIN_POLICY_DIGEST == "builtin"

    def test_empty_policy_is_the_builtin_marker(self):
        # `policy: {}` sends no policy, so the image's built-in policy applies.
        assert _digest({}) == BUILTIN_POLICY_DIGEST


class _FakeSandboxClient:
    """Records the SDK calls the runtime makes against one sandbox."""

    def __init__(self):
        self.exec_calls = []
        self.exec_results = []
        self.logs_requests = []
        self.specs = []
        self.created_specs = self.specs
        # Each sandbox's name and labels, as created.
        self.created = []
        self._stub = self

    def create(self, *, spec, name, workspace="midojo-shell-abc123", labels=None, **kwargs):
        self.specs.append(spec)
        self.created.append((name, dict(labels or {})))
        return SandboxRef(
            id="sb-1", name=name, workspace=workspace, status=SandboxStatusRef(phase=0, current_policy_version=0)
        )

    def wait_ready(self, name, **kwargs):
        pass

    def exec(self, sandbox_name, command, **kwargs):
        self.exec_calls.append((sandbox_name, command, kwargs))
        return self.exec_results.pop(0) if self.exec_results else ExecResult(exit_code=0, stdout="", stderr="")

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


# What a suite's env_vars may try to hand the sandbox: the session, the control plane and the real tools.
_STALE_ENV = {
    "MIDOJO_URL": "http://control",
    "MIDOJO_SESSION_TOKEN": "stale",
    "MIDOJO_CONTROL_TOKEN": "secret",
    "UPSTREAM_MCP_URL": "http://upstream:9000/mcp",
    "MCP_SERVER_URL": "http://upstream:9000/mcp",
}


def _sandbox_texts(client: _FakeSandboxClient) -> list[str]:
    """Everything the sandbox was given: its environment, name and labels, and every command and input it ran."""
    texts = [f"{key}={value}" for spec in client.specs for key, value in spec.environment.items()]
    texts += [f"{name} {labels}" for name, labels in client.created]
    texts += [repr(call) for call in client.exec_calls]
    return texts


def test_without_inject_session_an_agent_mode_sandbox_gets_no_session_and_no_control_plane():
    runtime, client = _runtime_with_sandbox(
        env_vars={"LITELLM_MODEL": "m", **_STALE_ENV}, files={"notes.txt": "hi"}, inject_session=False
    )
    runtime._run_labels = {"midojo.suite": "shell_suite", "midojo.run-id": "r" * 32}
    runtime._agent_command = ["agent", "-p"]
    runtime.configure(cluster="test", control_url="http://localhost:8090")
    runtime.setup({}, session_token="session-secret", eval_id="e1", user_task_id="t", injection_task_id="attack")
    runtime.exec_agent("hello", timeout_seconds=30, env={"MIDOJO_TURN_INDEX": "0", "MIDOJO_TURN_COUNT": "2"})
    # The real tools' URL stays: there is no interceptor for the agent to bypass, and an agent observed only by
    # OpenShell may need it.
    assert dict(client.specs[0].environment) == {
        "LITELLM_MODEL": "m",
        "UPSTREAM_MCP_URL": "http://upstream:9000/mcp",
        "MCP_SERVER_URL": "http://upstream:9000/mcp",
    }
    # Nor does the sandbox's name or a label name the run or the evaluation the control plane addresses it by.
    [(name, labels)] = client.created
    assert re.fullmatch(r"sbx-[0-9a-f]{12}", name) and labels == {"midojo.suite": "shell_suite"}
    texts = _sandbox_texts(client)
    for secret in ("session-secret", "host.openshell.internal:8090", "localhost:8090", "r" * 32, "eval-e1"):
        assert not any(secret in text for text in texts), secret


@pytest.mark.parametrize("inject_session", [True, False])
def test_host_mode_sandbox_environment_has_the_tool_endpoint_and_nothing_of_the_control_plane(inject_session):
    stale = {
        "MIDOJO_URL": "http://control",
        "MIDOJO_SESSION_TOKEN": "stale",
        "MIDOJO_CONTROL_TOKEN": "secret",
        # A direct route to the real tools would bypass the interceptor and its records.
        "UPSTREAM_MCP_URL": "http://upstream:9000/mcp",
        "MCP_SERVER_URL": "http://upstream:9000/mcp",
    }
    # A host-mode sandbox never gets the session, whatever the suite's inject_session says.
    runtime, client = _runtime_with_sandbox(env_vars={"LITELLM_MODEL": "m", **stale}, inject_session=inject_session)
    runtime.configure(cluster="test", interceptor_url="http://host.openshell.internal:8765/mcp")
    endpoint = ToolEndpoint("http://host.openshell.internal:8765/mcp", "tool-token")
    runtime.setup({}, tool_endpoint=endpoint, eval_id="e1", user_task_id="t", injection_task_id=None)
    assert dict(client.specs[0].environment) == {
        "LITELLM_MODEL": "m",
        "MIDOJO_TOOL_URL": "http://host.openshell.internal:8765/mcp",
        "MIDOJO_TOOL_TOKEN": "tool-token",
    }


def test_a_host_mode_sandbox_is_named_and_labelled_with_nothing_of_its_evaluation():
    runtime, client = _runtime_with_sandbox()
    runtime._run_labels = {"midojo.suite": "shell_suite", "midojo.run-id": "r" * 32}
    runtime.configure(cluster="test", interceptor_url="http://host.openshell.internal:8765/mcp")
    endpoint = ToolEndpoint("http://host.openshell.internal:8765/mcp", "tool-token")
    for _ in range(2):
        runtime.setup({}, tool_endpoint=endpoint, eval_id="e1", user_task_id="t", injection_task_id="attack")
    names = [name for name, _ in client.created]
    assert all(re.fullmatch(r"sbx-[0-9a-f]{12}", name) for name in names) and len(set(names)) == 2
    assert [labels for _, labels in client.created] == [{"midojo.suite": "shell_suite"}] * 2
    # The host, not the sandbox, knows which sandbox runs the evaluation.
    assert runtime.sandbox_name == names[-1]


def test_an_agent_mode_sandbox_keeps_its_evaluation_name_and_labels():
    runtime, client = _runtime_with_sandbox()
    runtime._run_labels = {"midojo.suite": "shell_suite", "midojo.run-id": "r" * 32}
    runtime.setup({}, session_token="s", eval_id="e1", user_task_id="t", injection_task_id="attack")
    [(name, labels)] = client.created
    assert name == "eval-e1"
    assert labels == {
        "midojo.suite": "shell_suite",
        "midojo.run-id": "r" * 32,
        "midojo.eval-id": "e1",
        "midojo.user-task": "t",
        "midojo.injection-task": "attack",
    }


@pytest.mark.parametrize("access", [{}, {"session_token": "s", "tool_endpoint": ToolEndpoint("http://h/mcp", "t")}])
def test_setup_needs_exactly_one_of_a_session_and_a_tool_endpoint(access):
    runtime, _ = _runtime_with_sandbox()
    with pytest.raises(ValueError, match="exactly one"):
        runtime.setup({}, **access, eval_id="e1", user_task_id="t", injection_task_id=None)


@pytest.mark.parametrize(
    "policy,warning",
    [
        (None, "No sandbox network policy defined"),
        (POLICY, "does not allow the host-side interceptor at host.openshell.internal:8765"),
        ({"networkPolicies": {"tools": {"endpoints": [{"host": "host.openshell.internal", "port": 8765}]}}}, None),
    ],
)
@pytest.mark.parametrize("host", ["host.openshell.internal", "[2001:db8::1]"])
def test_start_run_warns_when_the_policy_does_not_allow_the_interceptor(monkeypatch, caplog, policy, warning, host):
    # A policy names an IPv6 host without the brackets its URL needs.
    bare = host.strip("[]")
    policy = json.loads(json.dumps(policy).replace("host.openshell.internal", bare)) if policy else policy
    warning = warning and warning.replace("host.openshell.internal", bare)
    runtime = OpenShellRuntime("shell_suite", image="pi", policy=policy)
    workspaces = MagicMock()
    workspaces.create.side_effect = lambda name, **kw: SimpleNamespace(name=name)
    monkeypatch.setattr(openshell_runtime.SandboxClient, "from_active_cluster", lambda **kw: MagicMock())
    monkeypatch.setattr(openshell_runtime.WorkspaceClient, "from_sandbox_client", lambda client: workspaces)
    runtime.configure(cluster="test", interceptor_url=f"http://{host}:8765/mcp")
    with caplog.at_level("WARNING", logger="midojo.runtimes.openshell"):
        runtime.start_run("a" * 32, suite_name="shell_suite")
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    if warning is None:
        assert warnings == []
    else:
        [message] = warnings
        assert warning in message and "control plane" not in message


@pytest.mark.parametrize("inject_session,warned", [(True, True), (False, False)])
def test_start_run_checks_the_policy_reaches_the_control_plane_only_for_a_sandbox_with_the_session(
    monkeypatch, caplog, inject_session, warned
):
    runtime = OpenShellRuntime("shell_suite", image="pi", policy=POLICY, inject_session=inject_session)
    workspaces = MagicMock()
    workspaces.create.side_effect = lambda name, **kw: SimpleNamespace(name=name)
    monkeypatch.setattr(openshell_runtime.SandboxClient, "from_active_cluster", lambda **kw: MagicMock())
    monkeypatch.setattr(openshell_runtime.WorkspaceClient, "from_sandbox_client", lambda client: workspaces)
    runtime.configure(cluster="test", control_url="http://localhost:8090")
    with caplog.at_level("WARNING", logger="midojo.runtimes.openshell"):
        runtime.start_run("a" * 32, suite_name="shell_suite")
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert bool(warnings) is warned
    assert all("control plane at host.openshell.internal:8090" in message for message in warnings)


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


def test_exec_agent_rounds_a_fractional_timeout_up_to_whole_seconds():
    runtime, client = _runtime_with_sandbox()
    for timeout, sent in [(0.5, 1), (30, 30), (30.2, 31)]:
        runtime.exec_agent("hello", timeout_seconds=timeout)
        assert client.exec_calls[-1][2]["timeout_seconds"] == sent


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
            script = cast(MagicMock, runtime._exec).call_args.args[0][2]
            host = script.split()[2]
            return f"NET:REFUSE [MED] DENIED {host} [reason:policy_dns_ineligible]"

        reads = iter([[], [agent_event], None])

        def read():
            batch = next(reads)
            return batch if batch is not None else [agent_event, marker_from_exec()]

        runtime = self._runtime(read)
        assert runtime._sync_ocsf_messages() == ([agent_event], True)
        assert cast(MagicMock, runtime._read_ocsf_messages).call_count == 3

    def test_returns_last_read_as_unsynced_when_marker_never_arrives(self, monkeypatch):
        clock = iter(range(0, 100, 5))
        monkeypatch.setattr(openshell_runtime.time, "time", lambda: next(clock))
        monkeypatch.setattr(openshell_runtime.time, "sleep", lambda _s: None)
        allowed = "NET:OPEN [INFO] ALLOWED /usr/bin/curl(0) -> ok.test:443"
        runtime = self._runtime(lambda: [allowed])
        assert runtime._sync_ocsf_messages() == ([allowed], False)


class TestObservedRun:
    """observe() records what ran: the image and the effective policy's digest."""

    def _observe(self, monkeypatch, policy: dict | None) -> OpenShellObservations:
        runtime = OpenShellRuntime("shell_suite", image="localhost/agent:latest", policy=policy)
        client = _FakeSandboxClient()
        runtime._client = client
        runtime._workspace_name = "midojo-shell-abc123"
        monkeypatch.setattr(runtime, "_sync_ocsf_messages", lambda: ([], True))
        runtime.setup({}, session_token="tok", eval_id="e1", user_task_id="u1", injection_task_id=None)
        assert client.created_specs[0].template.image == "localhost/agent:latest"
        observed = runtime.observe()["openshell"]
        assert isinstance(observed, OpenShellObservations)
        return observed

    def test_records_image_and_policy_digest(self, monkeypatch):
        observed = self._observe(monkeypatch, POLICY)
        assert observed.image == "localhost/agent:latest"
        assert observed.policy_digest == _digest(POLICY)

    @pytest.mark.parametrize("policy", [None, {}])
    def test_builtin_policy_records_the_marker(self, monkeypatch, policy):
        assert self._observe(monkeypatch, policy).policy_digest == BUILTIN_POLICY_DIGEST

    def test_image_digest_is_null_because_openshell_does_not_report_one(self, monkeypatch):
        # OpenShell 0.1.2 reports no resolved image digest: the SDK's SandboxRef
        # carries no image, and the Sandbox proto (spec and status) only the
        # requested SandboxTemplate.image. If a digest appears, record it.
        def field_names(descriptor, seen):
            if descriptor.full_name in seen:
                return set()
            seen.add(descriptor.full_name)
            names = {f.name for f in descriptor.fields}
            for f in descriptor.fields:
                if f.message_type is not None:
                    names |= field_names(f.message_type, seen)
            return names

        reported = field_names(openshell_pb2.Sandbox.DESCRIPTOR, set()) | set(SandboxRef.__dataclass_fields__)
        assert not [name for name in reported if "digest" in name]
        assert {name for name in reported if "image" in name} == {"image"}
        assert self._observe(monkeypatch, POLICY).image_digest is None

    @pytest.mark.parametrize("field", ["image", "image_digest", "policy_digest"])
    def test_fields_are_optional(self, field):
        assert getattr(OpenShellObservations(), field) is None


class _FakeSandboxFiles:
    """A sandbox filesystem that answers the commands ``setup()`` and ``observe()`` run.

    ``changed`` holds the paths written after the baseline was touched. Set
    ``status_result`` to answer the outside-workdir status check with it instead.
    """

    def __init__(self):
        self.files: dict[str, str] = {}
        self.changed: set[str] = set()
        self.commands: list[list[str]] = []
        self.status_result: ExecResult | None = None

    def create(self, *, workspace, spec, name, labels):
        return SimpleNamespace(name=name)

    def wait_ready(self, name, *, workspace, timeout_seconds):
        pass

    def exec(self, sandbox_name, command, *, workspace, stdin=None, **kwargs):
        self.commands.append(command)
        stdout = ""
        match command:
            case ["tee", path]:
                self.write(path, (stdin or b"").decode())
            case ["touch", "/tmp/.midojo_baseline"]:
                self.changed.clear()
            case ["find", root, "-type", "f", *rest]:
                paths = [p for p in self.files if p.startswith(f"{root}/")]
                if "-newer" in rest:
                    paths = [p for p in paths if p in self.changed]
                stdout = "".join(f"{p}\n" for p in paths)
            case ["cat", path]:
                stdout = self.files[path]
            case ["sh", "-c", _script, _name, _baseline, *paths]:
                if self.status_result is not None:
                    return self.status_result
                stdout = "".join(f"{self._status(p)}\n" for p in paths)
        return ExecResult(exit_code=0, stdout=stdout, stderr="")

    def _status(self, path):
        if path not in self.files:
            return "deleted"
        return "modified" if path in self.changed else "unchanged"

    def write(self, path, content):
        self.files[path] = content
        self.changed.add(path)

    def delete(self, path):
        del self.files[path]


def _seeded_runtime(files: dict[str, str]) -> tuple[OpenShellRuntime, _FakeSandboxFiles]:
    """A runtime whose sandbox has been set up with these seed files."""
    runtime = OpenShellRuntime("shell_suite", image="pi", files=files)
    sandbox = _FakeSandboxFiles()
    runtime._client = sandbox
    runtime._workspace_name = "midojo-shell-abc123"
    runtime.setup({}, session_token="tok", eval_id="1", user_task_id="user_task_0", injection_task_id=None)
    runtime._cached_ocsf = parse_ocsf_lines([])  # skip the OCSF log sync
    sandbox.commands.clear()
    return runtime, sandbox


def _file_changes(runtime: OpenShellRuntime) -> tuple[list[str], list[str], list[str]]:
    observed = runtime.observe()["openshell"]
    assert isinstance(observed, OpenShellObservations)
    return observed.files_created, observed.files_modified, observed.files_deleted


@pytest.mark.parametrize("seed", ["/tmp/.midojo_baseline", "//tmp/.midojo_baseline", "/tmp/./.midojo_baseline"])
def test_seeding_the_baseline_path_is_rejected_at_suite_load(seed):
    definition = OpenShellRuntimeDefinition(image="pi", files={"notes.txt": "n", seed: "x"})
    with pytest.raises(ValueError, match=f"files entry '{re.escape(seed)}' .* baseline /tmp/.midojo_baseline"):
        build_runtime("shell_suite", definition)


class TestSeededFileChanges:
    """observe() reports seeded files at the paths setup() seeded them to."""

    def test_absolute_seed_left_alone_is_unchanged(self):
        runtime, sandbox = _seeded_runtime({"/etc/app.conf": "debug = false"})
        assert "/etc/app.conf" in sandbox.files
        assert _file_changes(runtime) == ([], [], [])

    @pytest.mark.parametrize(
        ("seed", "dest"),
        [("notes.txt", "/sandbox/workdir/notes.txt"), ("/etc/app.conf", "/etc/app.conf")],
        ids=["relative", "absolute"],
    )
    def test_seed_left_alone_modified_and_deleted(self, seed, dest):
        runtime, sandbox = _seeded_runtime({seed: "original"})
        assert _file_changes(runtime) == ([], [], [])

        sandbox.write(dest, "changed")
        assert _file_changes(runtime) == ([], [dest], [])

        sandbox.delete(dest)
        assert _file_changes(runtime) == ([], [], [dest])

    def test_mixed_seeds_in_one_evaluation(self):
        runtime, sandbox = _seeded_runtime(
            {
                "kept.txt": "a",
                "edited.txt": "b",
                "removed.txt": "c",
                "/etc/kept.conf": "d",
                "/etc/edited.conf": "e",
                "/opt/app/removed.conf": "f",
            }
        )
        sandbox.write("/sandbox/workdir/edited.txt", "changed")
        sandbox.delete("/sandbox/workdir/removed.txt")
        sandbox.write("/etc/edited.conf", "changed")
        sandbox.delete("/opt/app/removed.conf")
        sandbox.write("/sandbox/workdir/new.txt", "created")

        observed = runtime.observe()["openshell"]
        assert isinstance(observed, OpenShellObservations)
        assert observed.files_created == ["/sandbox/workdir/new.txt"]
        assert observed.workdir_new_file_contents == {"/sandbox/workdir/new.txt": "created"}
        assert sorted(observed.files_modified) == ["/etc/edited.conf", "/sandbox/workdir/edited.txt"]
        assert sorted(observed.files_deleted) == ["/opt/app/removed.conf", "/sandbox/workdir/removed.txt"]

    def test_files_created_outside_the_workdir_are_not_reported(self):
        runtime, sandbox = _seeded_runtime({"/etc/app.conf": "x"})
        sandbox.write("/etc/other.conf", "y")
        assert _file_changes(runtime) == ([], [], [])

    @pytest.mark.parametrize(
        ("seed", "dest"),
        [
            ("a", "/sandbox/workdir/a"),
            ("./sub/../a", "/sandbox/workdir/a"),
            ("//sandbox/workdir/a", "/sandbox/workdir/a"),
            ("///etc/app.conf", "/etc/app.conf"),
        ],
    )
    def test_seed_path_is_one_normalized_absolute_path(self, seed, dest):
        assert openshell_runtime._seed_path(seed) == dest

    @pytest.mark.parametrize("seed", ["/sandbox/workdir/notes.txt", "//sandbox/workdir/notes.txt"])
    def test_absolute_seed_inside_the_workdir_is_a_workdir_file(self, seed):
        runtime, sandbox = _seeded_runtime({seed: "x", "notes.txt": "x"})
        assert list(sandbox.files) == ["/sandbox/workdir/notes.txt"]
        assert _file_changes(runtime) == ([], [], [])
        assert all(command[0] == "find" for command in sandbox.commands)

        sandbox.write("/sandbox/workdir/notes.txt", "changed")
        assert _file_changes(runtime) == ([], ["/sandbox/workdir/notes.txt"], [])

        sandbox.delete("/sandbox/workdir/notes.txt")
        assert _file_changes(runtime) == ([], [], ["/sandbox/workdir/notes.txt"])


class TestSeededFileExecs:
    """Seeded files outside the workdir cost one exec in total, and only when present."""

    def test_no_extra_exec_without_absolute_seeds(self):
        runtime, sandbox = _seeded_runtime({"a.txt": "a", "sub/b.txt": "b"})
        runtime.observe()
        assert [command[0] for command in sandbox.commands] == ["find", "find"]

    def test_one_exec_checks_every_absolute_seed(self):
        paths = ["/etc/a.conf", "/etc/b.conf", "/opt/app/c.conf"]
        runtime, sandbox = _seeded_runtime({"notes.txt": "n", **{p: "x" for p in paths}})
        runtime.observe()
        assert len(sandbox.commands) == 3
        batch = sandbox.commands[2]
        assert batch[:2] == ["sh", "-c"]
        assert batch[3:] == ["sh", "/tmp/.midojo_baseline", *paths]

    @pytest.mark.parametrize(
        "result",
        [
            ExecResult(exit_code=127, stdout="", stderr="sh: not found"),
            ExecResult(exit_code=1, stdout="deleted\nmodified\n", stderr="find: Permission denied"),
            ExecResult(exit_code=0, stdout="deleted\n", stderr=""),
            ExecResult(exit_code=0, stdout="deleted\ngone\n", stderr=""),
        ],
        ids=["no-output", "nonzero-exit-full-output", "too-few-lines", "unknown-status"],
    )
    def test_failed_check_reports_no_outside_seeds_and_warns(self, result, caplog):
        runtime, sandbox = _seeded_runtime({"notes.txt": "n", "/etc/a.conf": "a", "/etc/b.conf": "b"})
        sandbox.delete("/etc/a.conf")
        sandbox.delete("/sandbox/workdir/notes.txt")
        sandbox.status_result = result
        assert _file_changes(runtime) == ([], [], ["/sandbox/workdir/notes.txt"])
        assert [r.levelname for r in caplog.records] == ["WARNING"]
        assert f"Seeded file check failed (exit {result.exit_code}" in caplog.text

    def test_failed_check_makes_the_file_evidence_incomplete(self):
        runtime, sandbox = _seeded_runtime({"notes.txt": "n", "/etc/a.conf": "a"})
        observed = runtime.observe()["openshell"]
        assert isinstance(observed, OpenShellObservations)
        assert observed.evidence_status.workdir == "complete"

        sandbox.status_result = ExecResult(exit_code=127, stdout="", stderr="sh: not found")
        observed = runtime.observe()["openshell"]
        assert isinstance(observed, OpenShellObservations)
        assert observed.evidence_status.workdir == "incomplete"

    def test_check_that_raises_reports_no_outside_seeds_and_incomplete_evidence(self, monkeypatch):
        runtime, sandbox = _seeded_runtime({"notes.txt": "n", "/etc/a.conf": "a"})
        sandbox.delete("/etc/a.conf")
        exec_in_sandbox = sandbox.exec

        def exec_raising_on_the_check(sandbox_name, command, **kwargs):
            if command[:2] == ["sh", "-c"]:
                raise RuntimeError("exec failed")
            return exec_in_sandbox(sandbox_name, command, **kwargs)

        monkeypatch.setattr(sandbox, "exec", exec_raising_on_the_check)
        observed = runtime.observe()["openshell"]
        assert isinstance(observed, OpenShellObservations)
        assert observed.files_deleted == []
        assert observed.evidence_status.workdir == "incomplete"


class TestSeedStatusScript:
    """The outside-workdir check script, run by a real local shell."""

    def _statuses(self, baseline, paths):
        result = subprocess.run(
            ["sh", "-c", openshell_runtime._SEED_STATUS_SCRIPT, "sh", str(baseline), *map(str, paths)],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.splitlines()

    def test_reports_each_path_in_order_without_running_its_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        baseline = tmp_path / "baseline"
        baseline.touch()
        os.utime(baseline, (1000, 1000))
        unchanged = tmp_path / "$(touch pwned) 'q\"; touch pwned2"
        unchanged.write_text("x")
        os.utime(unchanged, (500, 500))
        modified = tmp_path / "with space\nand newline.conf"
        modified.write_text("x")
        os.utime(modified, (2000, 2000))
        deleted = tmp_path / "gone.conf"

        assert self._statuses(baseline, [unchanged, modified, deleted]) == ["unchanged", "modified", "deleted"]
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["baseline", unchanged.name, modified.name])

    def test_a_seed_replaced_by_a_directory_is_deleted(self, tmp_path):
        baseline = tmp_path / "baseline"
        baseline.touch()
        os.utime(baseline, (1000, 1000))
        replaced = tmp_path / "app.conf"
        replaced.mkdir()
        os.utime(replaced, (500, 500))

        assert self._statuses(baseline, [replaced]) == ["deleted"]

    def test_a_seed_behind_a_symlink_is_checked_at_its_target(self, tmp_path):
        baseline = tmp_path / "baseline"
        baseline.touch()
        os.utime(baseline, (1000, 1000))
        target = tmp_path / "target.conf"
        target.write_text("x")
        link = tmp_path / "app.conf"
        link.symlink_to(target)  # the link itself is newer than the baseline

        os.utime(target, (500, 500))
        assert self._statuses(baseline, [link]) == ["unchanged"]
        os.utime(target, (2000, 2000))
        assert self._statuses(baseline, [link]) == ["modified"]
