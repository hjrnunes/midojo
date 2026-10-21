import os
import subprocess
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from openshell import ExecResult
from openshell._proto import openshell_pb2

import midojo.runtimes.openshell as openshell_runtime
from midojo.runtimes import build_runtime
from midojo.runtimes.openshell import OpenShellObservations, OpenShellRuntime, _resolve_policy
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
        self.logs_requests = []
        self._stub = self

    def exec(self, sandbox_name, command, **kwargs):
        self.exec_calls.append((sandbox_name, command, kwargs))

    def GetSandboxLogs(self, request, timeout):  # noqa: N802 - mirrors the gRPC stub
        self.logs_requests.append(request)
        return openshell_pb2.GetSandboxLogsResponse()


def _runtime_with_sandbox() -> tuple[OpenShellRuntime, _FakeSandboxClient]:
    runtime = OpenShellRuntime("shell_suite", image="pi")
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
