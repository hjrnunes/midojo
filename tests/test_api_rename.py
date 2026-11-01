"""The D23 names are primary, and the D24 aliases still resolve to them.

``InjectionSpec`` and ``Evaluation*`` are the names new code uses; the
``Episode*`` names and ``open_episode`` stay importable, and every one of them
is the same object as the new name it aliases, so a caller on either spelling
behaves the same until the aliases go (M9).
"""

from __future__ import annotations

import pytest

import midojo
import midojo.episode
import midojo.episode_context
import midojo.episode_document
import midojo.episode_handle
import midojo.evaluation_context
import midojo.evaluation_handle
import midojo.injection_spec
import midojo.injection_spec_document
import midojo.orchestrator


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


def test_each_old_name_is_its_new_name() -> None:
    from midojo import (
        Episode,
        EpisodeContext,
        EpisodeHandle,
        EpisodeResult,
        EpisodeRun,
        EpisodeStep,
        open_episode,
    )
    from midojo.evaluation_context import EvaluationContext, EvaluationResult, EvaluationStep, open_evaluation
    from midojo.evaluation_handle import EvaluationHandle
    from midojo.injection_spec import InjectionSpec
    from midojo.injection_spec_document import InjectionSpecDocumentError

    assert Episode is InjectionSpec
    assert EpisodeContext is EvaluationContext
    assert EpisodeHandle is EvaluationHandle
    assert EpisodeResult is EvaluationResult
    assert EpisodeRun is midojo.orchestrator.EvaluationRun
    assert EpisodeStep is EvaluationStep
    assert open_episode is open_evaluation
    assert midojo.Episode is InjectionSpec
    assert midojo.episode.Episode is InjectionSpec
    assert midojo.episode_context.EpisodeContext is EvaluationContext
    assert midojo.episode_context.EpisodeResult is EvaluationResult
    assert midojo.episode_context.EpisodeStep is EvaluationStep
    assert midojo.episode_context.open_episode is open_evaluation
    assert midojo.episode_handle.EpisodeHandle is EvaluationHandle
    assert midojo.episode_document.EpisodeDocumentError is InjectionSpecDocumentError
    assert midojo.EvaluationRun is midojo.orchestrator.EvaluationRun


def test_each_old_name_imports_from_the_module_that_exported_it() -> None:
    """Every old import path names the new object, not only the package root (AGENTS.md invariant 5)."""
    from midojo.control_plane.grading import Episode as GradingEpisode
    from midojo.control_plane.inprocess_control import Episode as InProcessEpisode
    from midojo.control_plane.service import Episode as ServiceEpisode
    from midojo.control_plane.state import Episode as StateEpisode
    from midojo.episode_context import Episode as ContextEpisode
    from midojo.episode_document import Episode as DocumentEpisode
    from midojo.episode_handle import Episode as HandleEpisode
    from midojo.evaluation_context import EvaluationContext, EvaluationStep, open_evaluation
    from midojo.injection_spec import InjectionSpec
    from midojo.orchestrator import Episode as OrchestratorEpisode
    from midojo.orchestrator import EpisodeContext, EpisodeRun, EpisodeStep, open_episode
    from midojo.yaml_task_suite import Episode as YamlSuiteEpisode

    assert OrchestratorEpisode is InjectionSpec
    assert EpisodeContext is EvaluationContext
    assert EpisodeRun is midojo.orchestrator.EvaluationRun
    assert EpisodeStep is EvaluationStep
    assert open_episode is open_evaluation
    assert GradingEpisode is InjectionSpec
    assert InProcessEpisode is InjectionSpec
    assert ServiceEpisode is InjectionSpec
    assert StateEpisode is InjectionSpec
    assert ContextEpisode is InjectionSpec
    assert DocumentEpisode is InjectionSpec
    assert HandleEpisode is InjectionSpec
    assert YamlSuiteEpisode is InjectionSpec


def test_the_pyrit_target_keeps_its_old_names() -> None:
    pytest.importorskip("pyrit")
    from midojo.pyrit.target import Episode, EpisodeContext

    assert Episode is midojo.InjectionSpec
    assert EpisodeContext is midojo.EvaluationContext


def test_a_star_import_of_the_orchestrator_still_names_the_old_names() -> None:
    """The module's ``__all__`` was the documented star-import surface before the rename."""
    namespace: dict[str, object] = {}
    exec("from midojo.orchestrator import *", namespace)
    assert namespace["Episode"] is midojo.InjectionSpec
    assert namespace["EpisodeContext"] is midojo.EvaluationContext
    assert namespace["EpisodeRun"] is midojo.orchestrator.EvaluationRun
    assert namespace["EpisodeStep"] is midojo.EvaluationStep
    assert namespace["open_episode"] is midojo.open_evaluation


def test_open_episode_opens_the_evaluation_open_evaluation_opens() -> None:
    spec = midojo.InjectionSpec(turns=["hello"])
    old = midojo.open_episode(spec)
    new = midojo.open_evaluation(spec)

    assert type(old) is type(new) is midojo.EvaluationContext
    assert old.episode is spec and new.episode is spec
    assert old.run_id is new.run_id is None


def test_the_old_schema_module_is_the_new_one() -> None:
    import midojo_sdk.episode_schema
    import midojo_sdk.injection_spec_schema

    assert (
        midojo_sdk.injection_spec_schema.EPISODE_DOCUMENT_V1
        == midojo_sdk.injection_spec_schema.INJECTION_SPEC_DOCUMENT_V1
    )
    assert midojo_sdk.episode_schema.load_schema is midojo_sdk.injection_spec_schema.load_schema
    assert midojo_sdk.episode_schema.SCHEMA_FILE == "schemas/midojo-injection-spec-v1.schema.json"
    assert midojo_sdk.episode_schema.LEGACY_SCHEMA_FILE == "schemas/midojo-episode-v1.schema.json"
