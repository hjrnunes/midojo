"""The host-mode contract every runtime passes: the target gets its tool endpoint, and nothing about its evaluation.

Each runtime runs a real host-mode benchmark (``run_benchmark`` with an
in-process control plane and one host-side interceptor per evaluation) whose
target is the scripted one (tests/fixtures/scripted_target/target.py), a
separate process: launched as the PI binary for an unmanaged agent, and by a
faked OpenShell gateway's exec for the openshell runtime. Everything the target
was given or sent is then searched for what must never reach it: its arguments,
environment, files, and the request lines, headers and bodies the interceptor
received. Planted leaks prove the search finds a secret in each channel.

No model, no gateway, no network beyond loopback.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openshell import SandboxClient, WorkspaceClient

from midojo.agent_client import AgentClient, Conversation, OpenShellAgentClient, PIAgentClient
from midojo.control_plane.config import CONTROL_TOKEN_ENV, AppConfig
from midojo.control_plane.inprocess_control import InProcessControlClient
from midojo.control_plane.service import ById, ControlPlane
from midojo.interception_host import HostInterception, HostInterceptor, ObservedRequest
from midojo.orchestrator import run_benchmark
from midojo.runtimes import UnmanagedRuntime
from midojo.runtimes.openshell import OpenShellRuntime
from midojo.tool_endpoint import TOOL_TOKEN_ENV, TOOL_URL_ENV
from tests.fixtures.scripted_target.target import SCRIPT
from tests.fixtures.scripted_target.upstream import scripted_upstream

TARGET = Path(__file__).parent / "fixtures" / "scripted_target" / "target.py"
CONTROL_TOKEN = "control-secret-7f3a"
CONTROL_URL = "http://control.example:8090"
SANDBOX_WORKDIR = "/sandbox/workdir"


def leaks(channels: dict[str, list[str]], forbidden: dict[str, str]) -> list[tuple[str, str]]:
    """Each (channel, sentinel) where a forbidden value appears, ignoring case."""
    return sorted(
        {
            (channel, name)
            for channel, texts in channels.items()
            for name, value in forbidden.items()
            if any(value.lower() in text.lower() for text in texts)
        }
    )


def target_channels(workdir: Path, requests: list[ObservedRequest]) -> dict[str, list[str]]:
    """Everything the target was given or sent, by channel: arguments, environment, files and requests.

    The files are those in the target's working directory but its transcript,
    which only repeats its arguments and environment.
    """
    transcript = json.loads((workdir / "transcript.json").read_text())
    files = [path for path in workdir.rglob("*") if path.is_file() and path.name != "transcript.json"]
    return {
        "argv": list(transcript["argv"]),
        "env name": list(transcript["env"]),
        "env value": list(transcript["env"].values()),
        "file name": [str(path.relative_to(workdir)) for path in files],
        "file content": [path.read_text(errors="replace") for path in files],
        "request line": [f"{r.method} {r.path}?{r.query}" for r in requests],
        "header name": [name for r in requests for name, _ in r.headers],
        "header value": [value for r in requests for _, value in r.headers],
        "request body": [r.body.decode("utf-8", errors="replace") for r in requests],
    }


CLEAN = {
    "argv": ["target.py", "-p", "What is the weather?"],
    "env name": ["PATH", TOOL_URL_ENV],
    "env value": ["/usr/bin", "http://127.0.0.1:1/mcp"],
    "file name": ["notes.md"],
    "file content": ["{}"],
    "request line": ["POST /mcp?"],
    "header name": ["host", "authorization"],
    "header value": ["127.0.0.1:1", "Bearer abc"],
    "request body": ['{"method": "tools/list"}'],
}
FORBIDDEN = {
    "session token": "Tok3n-abc",
    "control token": CONTROL_TOKEN,
    "control URL": CONTROL_URL,
    "run ID": "a" * 32,
    "evaluation ID": "0123456789",
}


@pytest.mark.parametrize("channel", list(CLEAN))
@pytest.mark.parametrize("sentinel", list(FORBIDDEN))
def test_the_leak_check_catches_any_sentinel_in_any_channel(channel, sentinel):
    assert leaks(CLEAN, FORBIDDEN) == []
    leaked = {**CLEAN, channel: [*CLEAN[channel], f"x-{FORBIDDEN[sentinel].upper()}-x"]}
    assert leaks(leaked, FORBIDDEN) == [(channel, sentinel)]


@dataclass(frozen=True)
class KeptInterception(HostInterception):
    """Host-mode settings that keep each interceptor they build, to read what it received."""

    built: list[HostInterceptor] = field(default_factory=list)

    def interceptor(self, recorder, *, evaluation_id, plan) -> HostInterceptor:
        interceptor = super().interceptor(recorder, evaluation_id=evaluation_id, plan=plan)
        self.built.append(interceptor)
        return interceptor


class WatchedControl(InProcessControlClient):
    """An in-process control client that keeps the evaluations it created, and runs ``on_evaluation`` on each."""

    def __init__(self, control: ControlPlane, on_evaluation) -> None:
        super().__init__(control)
        self.created: list = []
        self.on_evaluation = on_evaluation

    async def create_evaluation(self, *args, **kwargs):
        evaluation = await super().create_evaluation(*args, **kwargs)
        self.created.append(evaluation)
        self.on_evaluation(evaluation)
        return evaluation


class ArgvLeakingClient(AgentClient):
    """Wraps a client and appends ``secret()`` to every message: a planted leak in the target's arguments."""

    def __init__(self, inner: AgentClient, secret) -> None:
        self.inner, self.secret = inner, secret
        self.supports_host_interception = inner.supports_host_interception
        self.supports_multiturn = inner.supports_multiturn

    async def send_task(self, prompt, **access):
        return await self.inner.send_task(f"{prompt} {self.secret()}", **access)

    def start_conversation(self, **access) -> Conversation:
        inner, secret = self.inner.start_conversation(**access), self.secret

        class Leaking(Conversation):
            async def send(self, message: str) -> str:
                return await inner.send(f"{message} {secret()}")

            async def aclose(self) -> None:
                await inner.aclose()

        return Leaking()


@dataclass
class Harness:
    """One runtime running the scripted target in host mode, and where to plant a leak in what it's given."""

    runtime: object
    agent_client: AgentClient
    protocol: str
    workdir: Path
    plant_env: object  # (name, value) -> None
    interception: KeptInterception
    specs: list = field(default_factory=list)


@pytest.fixture
def unmanaged(tmp_path, monkeypatch) -> Harness:
    """An unmanaged agent over --protocol pi, whose ``pi`` binary is the scripted target."""
    bin_dir, agent_dir = tmp_path / "bin", tmp_path / "agent"
    bin_dir.mkdir()
    agent_dir.mkdir()
    pi = bin_dir / "pi"
    pi.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{TARGET}" "$@"\n')
    pi.chmod(pi.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    return Harness(
        runtime=UnmanagedRuntime(),
        agent_client=PIAgentClient(str(agent_dir), CONTROL_URL),
        protocol="pi",
        workdir=agent_dir,
        # The PI subprocess inherits the orchestrator's environment, so a leak there reaches it.
        plant_env=lambda name, value: monkeypatch.setenv(name, value),
        interception=KeptInterception(upstream=scripted_upstream()),
    )


@pytest.fixture
def openshell(tmp_path, monkeypatch) -> Harness:
    """The openshell runtime on a faked gateway, whose exec runs the scripted target on this machine.

    The fake sandbox's working directory is a local directory, its environment
    the sandbox spec's plus the run's, and its network maps
    host.openshell.internal to this machine's loopback.
    """
    workdir = tmp_path / "sandbox"
    workdir.mkdir()
    planted: dict[str, str] = {}
    specs: list = []
    sandboxes = MagicMock()
    sandboxes.list_ids.return_value = []

    def create(**kwargs):
        specs.append(kwargs["spec"])
        return SimpleNamespace(name=kwargs["name"], id=kwargs["name"])

    def exec_in_sandbox(sandbox, command, **kwargs):
        if command[0] == "tee":
            dest = workdir / Path(command[1]).relative_to(SANDBOX_WORKDIR)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(kwargs["stdin"])
        elif command[:2] == ["agent", "-p"]:
            env = {**specs[-1].environment, **(kwargs.get("env") or {}), **planted}
            if TOOL_URL_ENV in env:
                env[TOOL_URL_ENV] = env[TOOL_URL_ENV].replace("host.openshell.internal", "127.0.0.1")
            ran = subprocess.run(
                [sys.executable, "-I", str(TARGET), *command[1:]],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            return SimpleNamespace(exit_code=ran.returncode, stdout=ran.stdout, stderr=ran.stderr)
        return SimpleNamespace(exit_code=0, stdout="", stderr="")

    sandboxes.create.side_effect = create
    sandboxes.exec.side_effect = exec_in_sandbox
    workspaces = MagicMock()
    workspaces.create.side_effect = lambda name, **kw: SimpleNamespace(name=name)
    monkeypatch.setattr(SandboxClient, "from_active_cluster", lambda **kw: sandboxes)
    monkeypatch.setattr(WorkspaceClient, "from_sandbox_client", lambda client: workspaces)

    runtime = OpenShellRuntime(
        "weather",
        image="base",
        agent_command=["agent", "-p"],
        files={"notes.md": "Trip notes.\n"},
        # A suite's own env_vars may name the control plane: host mode drops them.
        env_vars={"MIDOJO_URL": CONTROL_URL, "MIDOJO_SESSION_TOKEN": "stale", CONTROL_TOKEN_ENV: CONTROL_TOKEN},
    )
    monkeypatch.setattr(runtime, "_sync_ocsf_messages", lambda: ([], True))
    runtime.configure(cluster="test", interceptor_url="http://host.openshell.internal:8765/mcp")
    return Harness(
        runtime=runtime,
        agent_client=OpenShellAgentClient(runtime),
        protocol="openshell",
        workdir=workdir,
        plant_env=planted.__setitem__,
        interception=KeptInterception(upstream=scripted_upstream(), advertise_host="host.openshell.internal"),
        specs=specs,
    )


@pytest.fixture(params=["unmanaged", "openshell"])
def harness(request) -> Harness:
    return request.getfixturevalue(request.param)


async def run_host_mode(harness: Harness, suite, monkeypatch, tmp_path, plant=None):
    """Run one host-mode evaluation on the harness, ``plant`` (evaluation, run ID) -> None at its creation.

    Returns the evaluation's stored record, its sentinels, and its channels.
    """
    # The host's own environment holds everything a target must not see.
    monkeypatch.setenv(CONTROL_TOKEN_ENV, CONTROL_TOKEN)
    monkeypatch.setenv("MIDOJO_URL", CONTROL_URL)
    control = ControlPlane({"weather": suite}, config=AppConfig(control_token=CONTROL_TOKEN, closure_wait_seconds=0))
    run_ids: list[str] = []
    original_create_run = control.create_run

    def create_run(*args, **kwargs):
        run = original_create_run(*args, **kwargs)
        run_ids.append(run.id)
        return run

    monkeypatch.setattr(control, "create_run", create_run)
    monkeypatch.setattr(suite, "new_runtime", lambda: harness.runtime)

    def on_evaluation(evaluation):
        # The session token is in the host's environment too, under the name an in-agent SDK reads.
        monkeypatch.setenv("MIDOJO_SESSION_TOKEN", evaluation.session_token)
        if plant is not None:
            plant(evaluation, run_ids[0])

    watched = WatchedControl(control, on_evaluation)
    async with watched:
        await run_benchmark(
            control_url=CONTROL_URL,
            agent_client=harness.agent_client,
            agent_uri="target",
            protocol=harness.protocol,
            suite=suite,
            suite_name="weather",
            user_task_ids=["weather_new_york"],
            injection_task_ids=[],
            logdir=tmp_path / "logs",
            control=watched,
            runtime=harness.runtime,  # type: ignore[arg-type]
            interception=harness.interception,
        )
    [evaluation] = watched.created
    [interceptor] = harness.interception.built
    sentinels = {
        "session token": evaluation.session_token,
        "control token": CONTROL_TOKEN,
        "control URL": CONTROL_URL,
        "control host": "control.example",
        "run ID": run_ids[0],
        "evaluation ID": evaluation.id,
    }
    stored = control.evaluation(ById(run_ids[0], evaluation.id))
    return stored, sentinels, target_channels(harness.workdir, interceptor.requests), interceptor


@pytest.mark.asyncio
async def test_a_host_mode_target_gets_only_its_tool_endpoint(harness, suite, monkeypatch, tmp_path):
    stored, sentinels, channels, interceptor = await run_host_mode(harness, suite, monkeypatch, tmp_path)

    assert leaks(channels, sentinels) == []
    midojo_env = {name for name in channels["env name"] if name.upper().startswith("MIDOJO")}
    assert midojo_env == {TOOL_URL_ENV, TOOL_TOKEN_ENV}
    assert interceptor.requests
    for request in interceptor.requests:
        assert "x-midojo-session" not in {name.lower() for name, _ in request.headers}
    for spec in harness.specs:
        assert {"MIDOJO_SESSION_TOKEN", "MIDOJO_URL", CONTROL_TOKEN_ENV}.isdisjoint(spec.environment)

    # The target ran its whole script through the interceptor, recorded in its evaluation by the interceptor.
    assert stored.interception == "host"
    assert [(c.function, c.provenance) for c in stored.function_calls] == [(n, "interceptor") for n, _ in SCRIPT]
    assert (stored.stage.value, stored.graded) == ("sealed", True)


@pytest.mark.asyncio
@pytest.mark.parametrize("channel", ["env", "argv", "file"])
async def test_a_planted_leak_fails_the_contract(harness, suite, monkeypatch, tmp_path, channel):
    created: list = []
    if channel == "argv":
        harness.agent_client = ArgvLeakingClient(harness.agent_client, lambda: created[-1].session_token)

    def plant(evaluation, run_id):
        created.append(evaluation)
        if channel == "env":
            # The target sends it along in a header, too.
            harness.plant_env("TARGET_DEBUG", f"run={run_id}")
        elif channel == "file":
            (harness.workdir / "leak.txt").write_text(f"eval {evaluation.id}")

    _, sentinels, channels, _ = await run_host_mode(harness, suite, monkeypatch, tmp_path, plant)
    expected = {
        "env": [("env value", "run ID"), ("header value", "run ID")],
        "argv": [("argv", "session token")],
        "file": [("file content", "evaluation ID")],
    }[channel]
    assert leaks(channels, sentinels) == expected
