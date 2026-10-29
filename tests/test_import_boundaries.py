"""Reusable logic imports none of the OpenShell runtime's dependencies, which load only when that runtime is chosen.

The light path must import neither the OpenShell runtime nor gRPC, protobuf
or the OpenShell SDK: the verifiers, the observation models, suite reading,
the episode, its context and result (:mod:`midojo.episode_context`), the
sealed snapshot (:mod:`midojo.control_plane.state`), grading
(:mod:`midojo.control_plane.grading`), the evaluation address
(:class:`~midojo.control_plane.service.ById`) and the in-process control
plane ``open_episode`` uses by default (:mod:`midojo.control_plane.service`,
``lifecycle``, ``store``, ``inprocess_control``, ``inprocess``). Each check
runs in a fresh interpreter, so nothing another test imported hides an
import.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from midojo.capabilities import MissingCapabilityError, import_provider
from midojo.episode import Episode
from midojo.runtimes import DEFAULT_RUNTIMES, RuntimeRegistry, UnmanagedRuntime, build_runtime
from midojo.suite_definition import OpenShellRuntimeDefinition
from midojo.verifiers import DEFAULT_VERIFIERS, VerificationResult, VerifierRegistry, parse_check
from midojo.verifiers.builtin import BuiltinVerifier
from midojo.yaml_task_suite import YAMLTaskSuite

ROOT = Path(__file__).resolve().parent.parent

# The OpenShell runtime and the modules only it needs.
HEAVY = ("grpc", "google.protobuf", "openshell", "midojo.runtimes.openshell")

# The light path: every module here, and all it imports, must load without the OpenShell runtime.
LIGHT_MODULES = (
    "midojo",
    "midojo.capabilities",
    "midojo.observations",
    "midojo.observations.openshell",
    "midojo.verifiers",
    "midojo.verifiers.builtin",
    "midojo.verifiers.openshell",
    "midojo.verifier",
    "midojo.suite_definition",
    "midojo.yaml_task_suite",
    "midojo.suites",
    "midojo.types",
    "midojo.delivery",
    "midojo.runtimes",
    # The episode, its context and result (the handle), the sealed snapshot and grading.
    "midojo.episode",
    "midojo.episode_context",
    "midojo.control_plane.state",
    "midojo.control_plane.grading",
    # The in-process control plane open_episode uses by default, which holds ById, and what it imports.
    "midojo.control_plane.service",
    "midojo.control_plane.lifecycle",
    "midojo.control_plane.store",
    "midojo.control_plane.models",
    "midojo.control_plane.config",
    "midojo.control_plane.client",
    "midojo.control_plane.inprocess",
    "midojo.control_plane.inprocess_control",
    # midojo-run's module, home of run_task, a client of the episode context.
    "midojo.orchestrator",
)

# What the weather suite (openshell runtime) needs in the environment to load.
SUITE_ENV = {"LITELLM_API_HOST": "models.invalid", "LITELLM_MODEL": "test-model"}

# Unimportable in the child: a meta path finder ahead of every other refuses them, after any copy already
# imported at start-up is purged from sys.modules.
BLOCKER = """
import importlib.abc
import sys

BLOCKED = ("grpc", "google.protobuf", "openshell")


def blocked(name):
    return any(name == top or name.startswith(top + ".") for top in BLOCKED)


class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if blocked(fullname):
            raise ModuleNotFoundError(f"No module named {fullname!r} (blocked by the test)", name=fullname)
        return None


for name in [name for name in sys.modules if blocked(name)]:
    del sys.modules[name]
sys.meta_path.insert(0, Unavailable())
"""


def _run(code: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, **SUITE_ENV}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120
    )


LIGHT_SCRIPTED_EPISODE = """
import asyncio
import importlib

# The light path imports with the runtime's dependencies unavailable.
for module in LIGHT_MODULES:
    importlib.import_module(module)

from midojo import EpisodeResult, Snapshot, Verdict, grade, open_episode
from midojo.capabilities import MissingCapabilityError
from midojo.control_plane.service import ById, ControlPlane
from midojo.observations.openshell import OpenShellObservations
from midojo.suites import get_suite

# A bundled suite reads: the weather suite asks for the openshell runtime, the minibank suite for an unmanaged one.
weather = get_suite("weather")
assert weather.definition.agent_runtime.type == "openshell"
minibank = get_suite("minibank")
episode = minibank.episode("account_balance", "ssn_disclosure", minibank.get_probes_for_task("ssn_disclosure"))
assert OpenShellObservations.not_collected().evidence_status.ocsf == "not_collected"


async def scripted() -> None:
    # The episode builds its own unmanaged runtime and runs on an in-process control plane: no server, no network.
    plane = ControlPlane({})
    async with open_episode(episode, control=plane) as ep:
        recorder = ep.control.recorder(ep.binding.session_token)

        async def agent(message: str) -> str:
            await recorder.start_call(
                call_id="c1", function="get_balance", args={"account_id": "ACC001"}, disposition="forwarded"
            )
            await recorder.finish_call(
                call_id="c1",
                function="get_balance",
                args={"account_id": "ACC001"},
                result="15420.50",
                disposition="forwarded",
            )
            return "ACC001 holds 15420.50. Alice's SSN is 123-45-6789."

        await ep.turn(ep.turns[0], agent)
    # The handle: the result names the evaluation, and the snapshot it sealed.
    result = ep.result
    assert isinstance(result, EpisodeResult) and not result.interrupted
    snapshot = ep.sealed
    assert isinstance(snapshot, Snapshot) and isinstance(result.snapshot, Snapshot)
    assert result.snapshot_id == snapshot.snapshot_id and snapshot.snapshot_id.startswith("sha256:")
    assert [call.function for call in snapshot.function_calls] == ["get_balance"]
    address = ById(None, result.evaluation_id)
    assert plane.evaluation(address).snapshot.snapshot_id == snapshot.snapshot_id
    # Graded by the pure function and by the control plane, alike.
    verdict = grade(episode, snapshot)
    assert isinstance(verdict, Verdict)
    assert (verdict.evaluation_id, verdict.snapshot_id) == (result.evaluation_id, snapshot.snapshot_id)
    assert verdict.grade.utility is True, verdict.grade
    assert verdict.grade.security is True, verdict.grade
    recorded = plane.grade(address.run_id, address.eval_id)
    assert (recorded.utility, recorded.security) == (True, True), recorded
    assert recorded.utility_reason == verdict.grade.utility_reason


asyncio.run(scripted())
loaded = sorted(name for name in HEAVY if name in sys.modules)
assert not loaded, f"loaded {loaded}"

# Choosing the OpenShell runtime then names the missing capability and what to install.
for choose in (weather.new_runtime, weather.episode(next(iter(weather.user_tasks))).new_runtime):
    try:
        choose()
    except MissingCapabilityError as error:
        assert isinstance(error, ImportError)
        assert error.capability == "the 'openshell' runtime", error.capability
        assert "openshell" in error.packages, error.packages
        assert "the 'openshell' runtime is not available" in str(error), str(error)
        assert error.name is not None and error.name.split(".")[0] in ("grpc", "google", "openshell"), error.name
    else:
        raise AssertionError("building the openshell runtime should fail without its dependencies")
try:
    from midojo.runtimes import OpenShellRuntime  # noqa: F401
except MissingCapabilityError as error:
    assert error.capability == "the 'openshell' runtime"
else:
    raise AssertionError("OpenShellRuntime should be unavailable without its dependencies")
print("ok")
"""


def test_with_the_openshell_dependencies_unavailable_a_suite_reads_and_a_scripted_episode_runs():
    result = _run(BLOCKER + f"HEAVY = {HEAVY!r}\nLIGHT_MODULES = {LIGHT_MODULES!r}\n" + LIGHT_SCRIPTED_EPISODE)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("ok")


@pytest.mark.parametrize("module", LIGHT_MODULES)
def test_reusable_modules_do_not_import_the_openshell_runtime(module):
    result = _run(
        f"""
        import importlib, sys
        importlib.import_module({module!r})
        print(sorted(name for name in {HEAVY!r} if name in sys.modules))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"importing {module} loaded {result.stdout.strip()}"


@pytest.mark.parametrize("suite_name", ["weather", "minibank", "document_assistant"])
def test_reading_a_bundled_suite_and_compiling_its_episodes_does_not_import_the_openshell_runtime(suite_name):
    result = _run(
        f"""
        import sys
        from midojo.suites import get_suite
        suite = get_suite({suite_name!r})
        for user_task in suite.user_tasks:
            for injection_task in [None, *suite.injection_tasks]:
                probes = suite.get_probes_for_task(injection_task) if injection_task else None
                suite.episode(user_task, injection_task, probes)
        print(sorted(name for name in {HEAVY!r} if name in sys.modules))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"reading {suite_name} loaded {result.stdout.strip()}"


def test_building_the_openshell_runtime_imports_it():
    result = _run(
        f"""
        import sys
        from midojo.suites import get_suite
        runtime = get_suite("weather").new_runtime()
        assert type(runtime).__name__ == "OpenShellRuntime"
        print(sorted(name for name in {HEAVY!r} if name in sys.modules))
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(sorted(HEAVY))


def test_a_missing_module_that_is_not_a_declared_dependency_is_raised_as_it_is():
    with pytest.raises(ModuleNotFoundError) as error:
        import_provider("midojo_no_such_provider", capability="x", packages=["x"], dependencies=["grpc"])
    assert not isinstance(error.value, MissingCapabilityError)


def test_a_provider_whose_declared_dependency_is_missing_names_the_capability(tmp_path, monkeypatch):
    (tmp_path / "boundary_provider.py").write_text("import boundary_missing_sdk.client\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(MissingCapabilityError) as error:
        import_provider(
            "boundary_provider", capability="the 'x' runtime", packages=["x-sdk"], dependencies=["boundary_missing_sdk"]
        )
    assert (error.value.capability, error.value.packages, error.value.name) == (
        "the 'x' runtime",
        ("x-sdk",),
        "boundary_missing_sdk",
    )
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert "pip install x-sdk" in str(error.value)
    assert "boundary_provider" not in sys.modules


class _Marker(UnmanagedRuntime):
    pass


def _suite(tmp_path: Path, utility: str = "output_contains: done", **kwargs) -> YAMLTaskSuite:
    path = tmp_path / "suite.yaml"
    path.write_text(
        textwrap.dedent(
            f"""
            agent_runtime:
              type: unmanaged
            environment: {{}}
            user_tasks:
              - id: task
                prompt: hi
                utility:
                  {utility}
            injection_tasks: []
            """
        )
    )
    return YAMLTaskSuite("boundary", suite_yaml_path=path, **kwargs)


def test_a_runtime_registry_can_be_injected_without_changing_the_default(tmp_path):
    registry = DEFAULT_RUNTIMES.copy()
    registry.register("unmanaged", lambda name, definition: _Marker(), replace=True)
    suite = _suite(tmp_path)
    assert isinstance(suite.new_runtime(registry=registry), _Marker)
    assert isinstance(suite.episode("task").new_runtime(registry=registry), _Marker)
    assert type(suite.new_runtime()) is UnmanagedRuntime
    with pytest.raises(ValueError, match="already registered"):
        registry.register("unmanaged", lambda name, definition: _Marker())
    with pytest.raises(ValueError, match="No runtime is registered for agent_runtime type 'openshell'"):
        build_runtime("s", OpenShellRuntimeDefinition(image="pi"), registry=RuntimeRegistry())


class _Custom:
    name = "boundary_custom"

    def parse(self, check_spec):
        return check_spec

    def assess(self, check, ctx):
        return VerificationResult(ctx.agent_output == check, "custom check")

    def evaluate(self, check, ctx):
        return ctx.agent_output == check


def test_a_verifier_registry_can_be_injected_into_a_suite_and_its_episodes(tmp_path):
    registry = DEFAULT_VERIFIERS.copy()
    registry.register(_Custom())
    assert DEFAULT_VERIFIERS.get("boundary_custom") is None
    suite = _suite(tmp_path, utility="boundary_custom: done", verifiers=registry)
    episode = suite.episode("task")
    assert episode.verifiers is registry
    env = suite.provision_environment({})
    graded = episode.grade("done", env, env, [])
    assert (graded["utility"], graded["utility_reason"]) == (True, "custom check")
    # The default registry knows no such verifier, so the built-in predicates refuse the block.
    with pytest.raises(ValueError):
        _suite(tmp_path, utility="boundary_custom: done")
    with pytest.raises(ValueError):
        parse_check({"boundary_custom": "done"})
    assert parse_check({"boundary_custom": "done"}, registry=registry).verifier.name == "boundary_custom"
    # An injected registry keeps the built-in predicates of the default it was copied from.
    assert parse_check({"output_contains": "x"}, registry=registry).verifier is DEFAULT_VERIFIERS.default


def test_an_episode_refuses_a_verifier_registry_of_another_type():
    with pytest.raises(TypeError, match="VerifierRegistry"):
        Episode(verifiers={})  # type: ignore[arg-type]


def test_an_empty_verifier_registry_has_no_default():
    with pytest.raises(ValueError, match="No default verifier"):
        VerifierRegistry().parse_check({"output_contains": "x"})


# ---------------------------------------------------------------------------
# Compatibility with the names and module state earlier versions exposed
# ---------------------------------------------------------------------------


def test_replacing_the_module_verifier_mapping_changes_what_the_default_path_dispatches_to(monkeypatch):
    import midojo.verifiers as verifiers

    replaced: dict = {"boundary_custom": _Custom()}
    monkeypatch.setattr(verifiers, "_VERIFIERS", replaced)
    assert parse_check({"boundary_custom": "done"}).verifier is replaced["boundary_custom"]
    assert DEFAULT_VERIFIERS.get("boundary_custom") is replaced["boundary_custom"]

    class Another(_Custom):
        name = "boundary_another"

    verifiers.register_verifier(Another())
    assert set(replaced) == {"boundary_custom", "boundary_another"}
    # An injected registry still resolves against its own verifiers.
    with pytest.raises(ValueError):
        parse_check({"boundary_custom": "done"}, registry=VerifierRegistry(default=DEFAULT_VERIFIERS.default))


def test_replacing_the_module_default_verifier_changes_the_default_path(monkeypatch):
    import midojo.verifiers as verifiers

    monkeypatch.setattr(verifiers, "_default_verifier", None)
    with pytest.raises(ValueError, match="No default verifier"):
        parse_check({"output_contains": "x"})
    builtin = BuiltinVerifier()
    verifiers.register_verifier(builtin, default=True)
    assert verifiers._default_verifier is builtin
    assert parse_check({"output_contains": "x"}).verifier is builtin


@pytest.mark.parametrize("default_first", [True, False])
def test_a_named_verifier_and_a_default_that_claims_its_name_conflict_in_either_order(default_first):
    class OutputContains(_Custom):
        name = "output_contains"

    registry = VerifierRegistry()
    steps = [lambda: registry.register(BuiltinVerifier(), default=True), lambda: registry.register(OutputContains())]
    first, second = steps if default_first else steps[::-1]
    first()
    with pytest.raises(ValueError, match="'output_contains' shadows a default-verifier check type"):
        second()


def test_the_runtimes_star_import_lists_the_openshell_runtime_and_loads_it_only_when_run():
    import midojo.runtimes as runtimes

    assert {"OpenShellRuntime", "openshell"} <= set(runtimes.__all__)
    result = _run(
        """
        import sys
        import midojo.runtimes as runtimes
        assert {"OpenShellRuntime", "openshell"} <= set(dir(runtimes))
        assert "midojo.runtimes.openshell" not in sys.modules
        namespace = {}
        exec("from midojo.runtimes import *", namespace)
        import midojo.runtimes.openshell as openshell
        assert namespace["OpenShellRuntime"] is openshell.OpenShellRuntime
        assert namespace["openshell"] is openshell
        assert runtimes.openshell is openshell
        assert set(runtimes.__all__) <= set(namespace)
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_the_runtimes_star_import_names_the_missing_capability_without_the_dependencies():
    result = _run(
        BLOCKER
        + textwrap.dedent(
            """
            from midojo.capabilities import MissingCapabilityError
            try:
                exec("from midojo.runtimes import *", {})
            except MissingCapabilityError as error:
                assert error.capability == "the 'openshell' runtime"
            else:
                raise AssertionError("the star import loads the openshell runtime, which needs its dependencies")
            print("ok")
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# Public names each module had at the base of this change that it no longer defines up front.
_KEPT_NAMES = {
    "midojo.runtimes": ("OpenShellRuntime", "openshell"),
    "midojo.runtimes.openshell": (
        "ClassVar",
        "ConfigDict",
        "Field",
        "HttpEvent",
        "Literal",
        "register_observation_type",
        "BUILTIN_POLICY_DIGEST",
        "OBSERVATIONS_SOURCE",
        "HttpRequestRecord",
        "OpenShellEvidence",
        "OpenShellObservations",
        "ProcessRecord",
    ),
    "midojo.orchestrator": ("OpenShellRuntime", "SANDBOX_HOST"),
}


@pytest.mark.parametrize("module", sorted(_KEPT_NAMES))
def test_names_a_module_exported_before_are_still_importable_from_it_and_listed(module):
    result = _run(
        f"""
        import importlib
        module = importlib.import_module({module!r})
        names = {_KEPT_NAMES[module]!r}
        missing = [name for name in names if name not in dir(module)]
        assert not missing, missing
        for name in names:
            exec(f"from {module} import {{name}}", {{}})
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# What ``from midojo.orchestrator import *`` exported at the base of this change: every public global it had.
_BASE_ORCHESTRATOR_STAR = (
    "A2AAgentClient",
    "ATTACK_OUTCOME",
    "AgentClient",
    "AgentRuntime",
    "Any",
    "Attacker",
    "AttackerFactory",
    "AttemptIsolation",
    "AttemptOutcome",
    "AttemptRun",
    "BenchmarkOutcome",
    "CONTROL_TOKEN_ENV",
    "Callable",
    "Console",
    "ControlClient",
    "ControlPlaneClient",
    "DeliveryFact",
    "EXIT_ALL_FAILED",
    "EXIT_SOME_FAILED",
    "EXPOSURE_BOUNDARIES",
    "Episode",
    "EpisodeContext",
    "EpisodeRun",
    "EpisodeStep",
    "Exchange",
    "Exposure",
    "ExposureStatus",
    "GradeResponse",
    "NamedTuple",
    "OGXResponsesClient",
    "OpenAIResponsesAgentClient",
    "OpenShellAgentClient",
    "OpenShellRuntime",
    "OutOfSurfaceError",
    "PIAgentClient",
    "PairFailure",
    "Panel",
    "Path",
    "SANDBOX_HOST",
    "Sequence",
    "SimpleHTTPAgentClient",
    "Snapshot",
    "StaticAttacker",
    "TYPE_CHECKING",
    "Table",
    "TaskPair",
    "TaskResult",
    "Text",
    "YAMLTaskSuite",
    "annotations",
    "asdict",
    "asyncio",
    "attempt_isolation",
    "bare_host",
    "cast",
    "check_plan_within",
    "click",
    "console",
    "contextlib",
    "delivery_facts",
    "exit_status",
    "exposure",
    "get_suite",
    "host_tools_reference",
    "httpx",
    "import_suite_module",
    "importlib",
    "ipaddress",
    "is_adaptive",
    "is_disposable",
    "json",
    "list_suites",
    "main",
    "math",
    "open_episode",
    "os",
    "permitted_injections",
    "run_benchmark",
    "run_task",
    "sys",
    "tool_url",
)


def test_the_orchestrator_star_import_exports_everything_it_did_and_named_imports_stay_light():
    import midojo.orchestrator as orchestrator

    assert set(_BASE_ORCHESTRATOR_STAR) <= set(orchestrator.__all__)
    result = _run(
        f"""
        import sys
        from midojo.orchestrator import main, run_task
        import midojo.orchestrator as orchestrator
        assert {{"OpenShellRuntime", "SANDBOX_HOST"}} <= set(dir(orchestrator))
        print(sorted(name for name in {HEAVY!r} if name in sys.modules))
        namespace = {{}}
        exec("from midojo.orchestrator import *", namespace)
        import midojo.runtimes.openshell as openshell
        assert namespace["OpenShellRuntime"] is openshell.OpenShellRuntime
        assert namespace["SANDBOX_HOST"] == openshell.SANDBOX_HOST
        assert set({_BASE_ORCHESTRATOR_STAR!r}) <= set(namespace)
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["[]", "ok"]


def test_the_orchestrator_star_import_names_the_missing_capability_without_the_dependencies():
    result = _run(
        BLOCKER
        + textwrap.dedent(
            """
            import midojo.orchestrator
            from midojo.orchestrator import run_task
            from midojo.capabilities import MissingCapabilityError
            try:
                exec("from midojo.orchestrator import *", {})
            except MissingCapabilityError as error:
                assert error.capability == "the 'openshell' runtime"
            else:
                raise AssertionError("the star import loads the openshell runtime, which needs its dependencies")
            print("ok")
            """
        )
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# The submodules a bare ``import midojo`` made reachable as attributes at the base of this change, and its __all__.
_BASE_REACHABLE = (
    "midojo.control_plane",
    "midojo.control_plane.config",
    "midojo.observations",
    "midojo.probes",
    "midojo.runtimes",
    "midojo.runtimes.openshell",
    "midojo.runtimes.openshell.logs",
    "midojo.suite_definition",
    "midojo.tool_endpoint",
    "midojo.types",
    "midojo.verifiers",
    "midojo.verifiers.builtin",
    "midojo.verifiers.combinators",
    "midojo.verifiers.judge",
    "midojo.verifiers.openshell",
    "midojo.verifiers.tool_call_condition",
)
_BASE_MIDOJO_ALL = (
    "AgentBinding",
    "Episode",
    "EpisodeContext",
    "EpisodeResult",
    "Snapshot",
    "Verdict",
    "grade",
    "open_episode",
)


def test_a_bare_import_reaches_every_submodule_it_did_and_loads_them_only_when_read():
    import midojo

    assert set(_BASE_MIDOJO_ALL) <= set(midojo.__all__)
    light = [name for name in _BASE_REACHABLE if not name.startswith("midojo.runtimes.openshell")]
    result = _run(
        BLOCKER
        + f"""
import sys
import midojo
lazy = sorted(name for name in {light!r} if name not in sys.modules)
assert lazy, "a bare import loads every one of them"
assert {{name.split(".")[1] for name in {_BASE_REACHABLE!r}}} <= set(dir(midojo))
for name in {light!r}:
    value = midojo
    for part in name.split(".")[1:]:
        value = getattr(value, part)
    assert value is sys.modules[name], name
loaded = sorted(name for name in {HEAVY!r} if name in sys.modules)
assert not loaded, f"reading the light submodules loaded {{loaded}}"
print("ok")
"""
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_a_bare_import_reaches_the_openshell_runtime_modules_when_read():
    result = _run(
        f"""
        import sys
        import midojo
        assert "midojo.runtimes.openshell" not in sys.modules
        for name in {[n for n in _BASE_REACHABLE if n.startswith("midojo.runtimes.openshell")]!r}:
            value = midojo
            for part in name.split(".")[1:]:
                value = getattr(value, part)
            assert value is sys.modules[name], name
        print("ok")
        """
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
