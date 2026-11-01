"""The D23 names are primary, and the D24 aliases still resolve to them.

``InjectionSpec`` and ``Evaluation*`` are the names new code uses; the
``Episode*`` names and ``open_episode`` stay importable, and every one of them
is the same object as the new name it aliases, so a caller on either spelling
behaves the same until the aliases go (M9).
"""

from __future__ import annotations

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
