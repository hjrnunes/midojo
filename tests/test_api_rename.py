"""The D23 names are the only names: the D24 aliases are gone (M9).

``InjectionSpec`` and ``Evaluation*`` are the names new code uses. The
``Episode*`` names, ``open_episode``, the ``midojo.episode*`` modules and the
SDK's ``midojo_sdk.episode_schema`` were deprecated aliases of them; importing
or naming any of them now fails. Two host-facing keyword arguments, the
``episode=`` of ``open_evaluation`` and the ``episode_for_payload=`` of
``MidojoInjectionSpecTarget``, keep their pre-rename names for one cycle, with
a ``DeprecationWarning``. The readers of recorded data keep the old document
versions, the old schema file and the old notes: those are data, not code
names.
"""

from __future__ import annotations

import importlib
import importlib.util
from collections.abc import Iterator
from typing import Any

import pytest
from midojo_sdk.injection import Channel, InjectionInstruction, InjectionPlan, InjectionTarget

import midojo
import midojo.control_plane.grading
import midojo.control_plane.inprocess_control
import midojo.control_plane.service
import midojo.control_plane.state
import midojo.evaluation_context
import midojo.evaluation_handle
import midojo.injection_spec
import midojo.injection_spec_document
import midojo.orchestrator
import midojo.yaml_task_suite
from midojo.runtimes import UnmanagedRuntime

PYRIT_SURFACE = InjectionPlan((InjectionInstruction(Channel.TOOL_OUTPUT, "", InjectionTarget(tool="read_note")),))


class _Disposable(UnmanagedRuntime):
    """An unmanaged target PyRIT may replace per evaluation; the constructor only checks the declaration."""

    disposable = True


async def _never_driven(context: object) -> None:
    raise AssertionError("the constructor test never drives the target")


@pytest.fixture
def pyrit_memory(monkeypatch) -> Iterator[Any]:
    """The central memory PyRIT's target base class reads; a fresh one per test, disposed afterwards."""
    pytest.importorskip("pyrit")
    from pyrit.common.singleton import Singleton  # pyright: ignore[reportMissingImports]
    from pyrit.memory.central_memory import CentralMemory  # pyright: ignore[reportMissingImports]
    from pyrit.memory.sqlite_memory import SQLiteMemory  # pyright: ignore[reportMissingImports]

    monkeypatch.setattr(Singleton, "_instances", {})
    monkeypatch.setattr(CentralMemory, "_memory_instance", None)
    memory: Any = SQLiteMemory(db_path=":memory:", silent=True)
    CentralMemory.set_memory_instance(memory)
    yield memory
    memory.dispose_engine()


REMOVED_MODULES = (
    "midojo.episode",
    "midojo.episode_aliases",
    "midojo.episode_context",
    "midojo.episode_document",
    "midojo.episode_handle",
    "midojo_sdk.episode_schema",
)

REMOVED_NAMES = (
    "Episode",
    "EpisodeContext",
    "EpisodeHandle",
    "EpisodeResult",
    "EpisodeRun",
    "EpisodeStep",
    "open_episode",
)

# The modules that exported an old name, in place of the import that used to resolve it.
ALIASED_MODULES = (
    midojo,
    midojo.control_plane.grading,
    midojo.control_plane.inprocess_control,
    midojo.control_plane.service,
    midojo.control_plane.state,
    midojo.orchestrator,
    midojo.yaml_task_suite,
)


def test_the_new_names_are_exported_by_the_package() -> None:
    from midojo import (
        EvaluationContext,
        EvaluationHandle,
        EvaluationResult,
        EvaluationRun,
        EvaluationStep,
        InjectionSpec,
        open_evaluation,
    )

    assert {
        "EvaluationContext",
        "EvaluationHandle",
        "EvaluationResult",
        "EvaluationRun",
        "EvaluationStep",
        "InjectionSpec",
        "open_evaluation",
    } <= set(midojo.__all__)
    assert midojo.InjectionSpec is InjectionSpec
    assert midojo.open_evaluation is open_evaluation
    assert midojo.EvaluationContext is EvaluationContext
    assert midojo.EvaluationHandle is EvaluationHandle
    assert midojo.EvaluationResult is EvaluationResult
    assert midojo.EvaluationRun is EvaluationRun
    assert midojo.EvaluationStep is EvaluationStep


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_an_old_module_is_gone(name: str) -> None:
    assert importlib.util.find_spec(name) is None
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


@pytest.mark.parametrize("module", ALIASED_MODULES, ids=lambda module: module.__name__)
def test_no_module_answers_to_an_old_name(module) -> None:
    for name in REMOVED_NAMES:
        assert not hasattr(module, name), f"{module.__name__} still answers to {name}"
        assert name not in dir(module), f"{module.__name__} still lists {name}"


def test_the_package_does_not_list_an_old_name() -> None:
    for name in REMOVED_NAMES:
        assert name not in midojo.__all__


def test_a_star_import_of_the_orchestrator_names_no_old_name() -> None:
    namespace: dict[str, object] = {}
    exec("from midojo.orchestrator import *", namespace)
    for name in REMOVED_NAMES:
        assert name not in namespace
    assert namespace["InjectionSpec"] is midojo.InjectionSpec
    assert namespace["EvaluationRun"] is midojo.orchestrator.EvaluationRun


def test_the_pyrit_target_has_no_old_names() -> None:
    pytest.importorskip("pyrit")
    import midojo.pyrit.target

    for name in REMOVED_NAMES:
        assert not hasattr(midojo.pyrit.target, name)


def test_open_evaluation_keeps_the_deprecated_episode_keyword() -> None:
    spec = midojo.InjectionSpec()
    with pytest.warns(DeprecationWarning, match=r"episode= is deprecated; use injection_spec="):
        context = midojo.open_evaluation(episode=spec, runtime=UnmanagedRuntime())

    assert context.episode is spec


def test_open_evaluation_refuses_both_keywords() -> None:
    spec = midojo.InjectionSpec()
    with pytest.raises(TypeError, match="injection_spec and its deprecated alias episode"):
        midojo.open_evaluation(spec, episode=spec)


def test_open_evaluation_still_requires_an_injection_spec() -> None:
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'injection_spec'"):
        midojo.open_evaluation()


def test_the_pyrit_target_keeps_the_deprecated_episode_for_payload_keyword(pyrit_memory) -> None:
    from midojo.pyrit.target import MidojoInjectionSpecTarget

    def compile_injection_spec(payload: str) -> midojo.InjectionSpec:
        return midojo.InjectionSpec()

    with pytest.warns(DeprecationWarning, match=r"episode_for_payload= is deprecated; use injection_spec_for_payload="):
        target = MidojoInjectionSpecTarget(
            episode_for_payload=compile_injection_spec,
            surface=PYRIT_SURFACE,
            drive=_never_driven,
            runtime=_Disposable(),
        )

    assert target is not None


def test_the_pyrit_target_refuses_both_payload_keywords(pyrit_memory) -> None:
    from midojo.pyrit.target import MidojoInjectionSpecTarget

    def compile_injection_spec(payload: str) -> midojo.InjectionSpec:
        return midojo.InjectionSpec()

    with pytest.raises(TypeError, match="injection_spec_for_payload and its deprecated alias episode_for_payload"):
        MidojoInjectionSpecTarget(
            injection_spec_for_payload=compile_injection_spec,
            episode_for_payload=compile_injection_spec,
            surface=PYRIT_SURFACE,
            drive=_never_driven,
            runtime=_Disposable(),
        )


def test_the_pyrit_target_still_requires_a_payload_compiler(pyrit_memory) -> None:
    from midojo.pyrit.target import MidojoInjectionSpecTarget

    with pytest.raises(TypeError, match="missing 1 required keyword-only argument: 'injection_spec_for_payload'"):
        MidojoInjectionSpecTarget(
            surface=PYRIT_SURFACE,
            drive=_never_driven,
            runtime=_Disposable(),
        )


def test_the_in_process_client_has_no_open_episode() -> None:
    from midojo.control_plane.inprocess_control import InProcessControlClient
    from midojo.control_plane.service import ControlPlane

    client = InProcessControlClient(ControlPlane({}))
    assert not hasattr(client, "open_episode")
    assert hasattr(client, "open_evaluation")


def test_the_old_document_constant_is_gone() -> None:
    import midojo_sdk.injection_spec_schema as schema

    assert not hasattr(schema, "EPISODE_DOCUMENT_V1")
    assert "EPISODE_DOCUMENT_V1" not in schema.__all__


def test_the_readers_keep_the_old_recorded_names() -> None:
    """The old names are data: recorded documents, the schema file and the notes keep reading."""
    import midojo_sdk.injection_spec_schema as schema

    assert midojo.injection_spec_document.LEGACY_DOCUMENT_VERSIONS == ("midojo-episode-v1", "midojo-episode-v2")
    assert schema.LEGACY_INJECTION_SPEC_DOCUMENT_V1 == "midojo-episode-v1"
    assert schema.LEGACY_SCHEMA_FILE == "schemas/midojo-episode-v1.schema.json"
    assert midojo.injection_spec.MAX_NESTING_DEPTH > 0
