"""Episode documents: the closed v1 schema, its vectors, the reader and writer, and the v0 wrapper (architecture 2.1.2).

The vectors are golden: the valid ones must load and be written back
unchanged, and the refused ones must be refused both by the reader and, where
the schema can express the rule, by the schema itself. Every bundled suite pair
must compile to a valid document too, so the suite path can never drift from
the contract.
"""

from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from midojo_sdk import episode_schema, injection
from midojo_sdk.episode_schema import EPISODE_DOCUMENT_V1, PLAN_CODEC_VERSION, SCHEMA_FILE, load_schema
from midojo_sdk.injection import InjectionPlan

from midojo.episode import Episode, InterceptionSettings, ToolPolicy, thaw
from midojo.episode_document import (
    LEGACY_DOCUMENT_VERSIONS,
    SUPPORTED_DOCUMENT_VERSIONS,
    EpisodeDocumentError,
    UnsupportedDocumentVersion,
)
from midojo.suites import get_suite

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "episode_document"
V1_VECTORS = sorted(path.name for path in FIXTURES.glob("v1_*.json"))
REFUSED_VECTORS = sorted(path.name for path in FIXTURES.glob("refused_*.json"))
# The one refusal the schema cannot express: a check's inner shape is the verifier registry's.
SCHEMA_REFUSES = [name for name in REFUSED_VECTORS if name != "refused_openshell_predicate_on_unmanaged.json"]

# The pairs each bundled suite exposes: every user task, with and without each injection task.
SUITE_PAIRS = {"document_assistant": 6, "minibank": 40, "weather": 21}

DEFAULT_INTERCEPTION = {
    "advertise_host": None,
    "bind_host": None,
    "mode": "host",
    "observe_requests": False,
    "port": None,
    "upstream": None,
}


def read_vector(name: str) -> dict[str, Any]:
    """The fixture ``name`` decoded, as the document it holds."""
    document: dict[str, Any] = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    return document


def episode_validator() -> jsonschema.Draft202012Validator:
    """A standalone validator over the SDK's schema, as the consumer's tests build one."""
    return jsonschema.Draft202012Validator(load_schema())


def plan_validator() -> jsonschema.Draft202012Validator:
    """A validator for the schema's ``$defs/plan`` alone, so a plan is checked without a document."""
    schema = load_schema()
    return jsonschema.Draft202012Validator(
        {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "$defs": schema["$defs"],
            "allOf": [{"$ref": "#/$defs/plan"}],
        }
    )


def mutated(document: dict[str, Any], path: tuple[object, ...], key: str, value: Any = None) -> dict[str, Any]:
    """A copy of ``document`` with ``key`` added where ``path`` points."""
    copy: dict[str, Any] = json.loads(json.dumps(document))
    target: Any = copy
    for step in path:
        target = target[step]
    target[key] = value
    return copy


# --- VAL-DOC-001: the schema, its package data and the vectors ---


def test_the_schema_is_package_data_the_sdk_loads() -> None:
    schema = load_schema()
    definitions = schema["$defs"]

    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["additionalProperties"] is False
    assert isinstance(definitions, dict)
    assert "plan" in definitions
    assert EPISODE_DOCUMENT_V1 == "midojo-episode-v1"
    assert json.loads(resources.files("midojo_sdk").joinpath(SCHEMA_FILE).read_text(encoding="utf-8")) == schema
    schema["title"] = "changed by a caller"
    assert load_schema()["title"] != "changed by a caller"


@pytest.mark.parametrize("name", V1_VECTORS)
def test_each_valid_vector_validates_loads_and_writes_itself_back(name: str) -> None:
    document = read_vector(name)

    episode_validator().validate(document)
    episode = Episode.from_document(document)
    assert episode.to_document() == document


@pytest.mark.parametrize("name", REFUSED_VECTORS)
def test_each_refused_vector_is_refused_by_the_reader(name: str) -> None:
    with pytest.raises(EpisodeDocumentError):
        Episode.from_document(read_vector(name))


@pytest.mark.parametrize("name", SCHEMA_REFUSES)
def test_each_refused_vector_the_schema_can_express_is_refused_by_the_schema(name: str) -> None:
    assert list(episode_validator().iter_errors(read_vector(name)))


@pytest.mark.parametrize(
    ("name", "named"),
    [
        ("refused_unknown_version.json", "midojo-episode-v9"),
        ("refused_unknown_key.json", "extensions"),
        ("refused_nonempty_verifiers.json", "verifiers"),
        ("refused_bad_disposition.json", "drop"),
        ("refused_openshell_predicate_on_unmanaged.json", "OpenShell"),
        ("refused_plan_unknown_key.json", "bogus"),
        ("refused_unknown_runtime_type.json", "kubernetes"),
    ],
)
def test_each_refusal_names_what_it_refuses(name: str, named: str) -> None:
    with pytest.raises(EpisodeDocumentError, match=named):
        Episode.from_document(read_vector(name))


# --- VAL-DOC-002: the document is closed ---


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("plan",),
        ("plan", "generator_permissions"),
        ("plan", "instructions", 0),
        ("plan", "instructions", 0, "target"),
        ("plan", "instructions", 0, "occurrence"),
        ("interception",),
        ("tools",),
        ("tools", "write_disposition"),
        ("runtime",),
    ],
    ids=[
        "top-level",
        "plan",
        "generator_permissions",
        "instruction",
        "target",
        "occurrence",
        "interception",
        "tools",
        "write_disposition",
        "runtime",
    ],
)
def test_an_unknown_key_anywhere_is_refused_naming_it(path: tuple[object, ...]) -> None:
    document = mutated(read_vector("v1_single.json"), path, "extensions", {})

    with pytest.raises(EpisodeDocumentError, match="extensions"):
        Episode.from_document(document)
    errors = list(episode_validator().iter_errors(document))
    assert errors
    assert any("extensions" in error.message for error in errors)


def test_a_top_level_extensions_object_is_refused() -> None:
    document = read_vector("v1_single.json")
    document["extensions"] = {"asago": {"anything": True}}

    with pytest.raises(EpisodeDocumentError, match="extensions"):
        Episode.from_document(document)
    assert list(episode_validator().iter_errors(document))


def test_an_unknown_key_in_an_openshell_runtime_definition_is_refused() -> None:
    document = mutated(read_vector("v1_openshell.json"), ("runtime",), "extensions", {})

    with pytest.raises(EpisodeDocumentError, match="extensions"):
        Episode.from_document(document)
    assert list(episode_validator().iter_errors(document))


def test_a_runtime_type_that_is_not_a_defined_runtime_is_refused_naming_it() -> None:
    document = read_vector("v1_single.json")
    document["runtime"] = {"type": "kubernetes"}

    with pytest.raises(EpisodeDocumentError, match="kubernetes"):
        Episode.from_document(document)
    assert list(episode_validator().iter_errors(document))


# --- VAL-DOC-003: execution settings ---


def _settings_document() -> dict[str, Any]:
    document = read_vector("v1_single.json")
    document["interception"] = {
        "mode": "host",
        "upstream": "http://interceptor.test:8082/mcp",
        "bind_host": "127.0.0.1",
        "advertise_host": "host.docker.internal",
        "port": 8082,
        "observe_requests": True,
    }
    document["tools"] = {
        "allowed": ["lookup_order", "process_refund"],
        "write_disposition": {"default": "block", "tools": {"lookup_order": "forward"}},
    }
    return document


def test_the_interception_and_tool_settings_round_trip() -> None:
    document = _settings_document()

    episode = Episode.from_document(document)
    assert episode.interception == InterceptionSettings(
        mode="host",
        upstream="http://interceptor.test:8082/mcp",
        bind_host="127.0.0.1",
        advertise_host="host.docker.internal",
        port=8082,
        observe_requests=True,
    )
    assert episode.tools is not None
    assert episode.tools == ToolPolicy(
        allowed=("lookup_order", "process_refund"),
        write_disposition_default="block",
        write_disposition={"lookup_order": "forward"},
    )
    assert episode.to_document()["interception"] == document["interception"]
    assert episode.to_document()["tools"] == document["tools"]
    assert Episode.from_document(episode.to_document()) == episode


def test_the_settings_are_frozen_data() -> None:
    episode = Episode.from_document(_settings_document())

    assert episode.tools is not None
    with pytest.raises(TypeError):
        episode.tools.write_disposition["process_refund"] = "block"  # type: ignore[index]
    with pytest.raises(AttributeError):
        episode.interception.port = 0  # type: ignore[misc]


def test_two_dumps_of_one_episode_are_the_same_bytes(tmp_path: Path) -> None:
    episode = Episode.from_document(_settings_document())
    first, second = tmp_path / "first.json", tmp_path / "second.json"

    episode.dump_document(first)
    episode.dump_document(second)

    assert first.read_bytes() == second.read_bytes()
    assert first.read_text(encoding="utf-8").endswith("}\n")
    assert Episode.load_document(first) == episode


def test_a_document_without_tools_mirrors_every_upstream_tool() -> None:
    document = read_vector("v1_single.json")
    del document["tools"]

    episode = Episode.from_document(document)

    assert episode.tools is None
    assert episode.to_document()["tools"] is None
    # An empty policy exposes no tool, which is not the same as no policy at all.
    empty = read_vector("v1_single.json")
    empty["tools"] = {"allowed": [], "write_disposition": {"default": "forward", "tools": {}}}
    assert Episode.from_document(empty).tools == ToolPolicy(allowed=())


def test_a_document_without_interception_gets_the_defaults() -> None:
    document = read_vector("v1_single.json")
    del document["interception"]

    episode = Episode.from_document(document)

    assert episode.interception == InterceptionSettings()
    assert episode.to_document()["interception"] == DEFAULT_INTERCEPTION


def test_a_minimal_document_holds_only_its_version() -> None:
    document = {"schema_version": "midojo-episode-v1"}

    episode_validator().validate(document)
    episode = Episode.from_document(document)

    assert episode.runtime_definition().type == "unmanaged"
    assert episode.turns is None
    assert episode.plan == InjectionPlan()
    assert episode.tools is None
    assert Episode.from_document(episode.to_document()) == episode


def test_to_document_always_writes_v1() -> None:
    direct = Episode(turns=["hello"])
    with pytest.warns(DeprecationWarning):
        v0_loaded = Episode.from_document(read_vector("v0_cm58_episode.json"))

    assert direct.to_document()["schema_version"] == "midojo-episode-v1"
    assert v0_loaded.to_document()["schema_version"] == "midojo-episode-v1"


def test_a_document_naming_another_environment_model_is_refused() -> None:
    document = read_vector("v1_single.json")
    document["environment_type"] = "WeatherEnvironment"

    with pytest.raises(EpisodeDocumentError, match="WeatherEnvironment"):
        Episode.from_document(document)


# --- VAL-DOC-004: the plan codec version and the plan sub-schema ---


def test_the_plan_codec_version_is_the_document_version() -> None:
    assert PLAN_CODEC_VERSION == "midojo-plan-v1"
    assert injection.PLAN_CODEC_VERSION == PLAN_CODEC_VERSION
    assert episode_schema.PLAN_CODEC_VERSION == PLAN_CODEC_VERSION
    assert SUPPORTED_DOCUMENT_VERSIONS == (EPISODE_DOCUMENT_V1,)
    assert LEGACY_DOCUMENT_VERSIONS == ("midojo-episode-v2",)


def test_the_plan_sub_schema_validates_the_bundled_plan() -> None:
    plan = json.loads((ROOT / "tests" / "fixtures" / "injection_plan_before_call_arguments.json").read_text())

    plan_validator().validate(plan)
    assert InjectionPlan.from_data(plan).to_data() == plan


def test_the_plan_sub_schema_validates_the_cm58_plan() -> None:
    plan = read_vector("v0_cm58_episode.json")["episode"]["plan"]

    plan_validator().validate(plan)


# --- VAL-DOC-005: every bundled suite pair compiles ---


@pytest.mark.parametrize("suite_name", sorted(SUITE_PAIRS))
def test_each_bundled_suite_compiles_every_pair_to_a_valid_document(suite_name: str) -> None:
    suite = get_suite(suite_name)
    pairs = [
        (user_task, injection_task)
        for user_task in suite.user_tasks
        for injection_task in [None, *suite.injection_tasks]
    ]
    assert len(pairs) == SUITE_PAIRS[suite_name]
    validator = episode_validator()

    for user_task, injection_task in pairs:
        probes = suite.get_probes_for_task(injection_task) if injection_task is not None else None
        episode = suite.episode(user_task, injection_task, probes)
        document = episode.to_document()

        validator.validate(document)
        assert document["environment_type"] == suite.environment_type.__name__
        assert document["suite_name"] == suite_name
        again = Episode.from_document(document)
        assert again == episode
        assert again.to_document() == document


# --- VAL-DOC-006: the v0 consumer wrapper ---


def test_the_v0_vector_is_the_cm58_artifact_byte_for_byte() -> None:
    # Copied from the cm.58 delivery run's compiled episode.json.
    data = (FIXTURES / "v0_cm58_episode.json").read_bytes()

    assert hashlib.sha256(data).hexdigest() == "2ed42034d7332094ca1548947da6cc0139181596f9c3c187e9961c5448969e49"


def test_the_cm58_wrapper_reads_as_v0_with_a_deprecation_warning() -> None:
    wrapper = read_vector("v0_cm58_episode.json")
    inner = wrapper["episode"]

    with pytest.warns(DeprecationWarning, match="midojo-episode-v2"):
        episode = Episode.from_document(wrapper)

    assert list(episode.turns or ()) == inner["turns"]
    assert episode.plan.to_data() == inner["plan"]
    assert thaw(episode.security) == inner["security"]
    assert episode.utility is None
    assert thaw(episode.runtime) == inner["runtime"]
    assert thaw(episode.environment) == inner["environment"]
    assert episode.suite_name is None
    assert episode.user_task_id is None
    assert episode.injection_task_id is None
    assert episode.tools is not None
    assert list(episode.tools.allowed) == wrapper["tools"]["allowed"]
    assert episode.tools.write_disposition_default == wrapper["tools"]["write_disposition"]["default"]
    assert dict(episode.tools.write_disposition) == wrapper["tools"]["write_disposition"]["tools"]
    # The late-bound marker names no upstream: the driver supplies the endpoint at open().
    assert episode.interception.mode == "host"
    assert episode.interception.upstream is None

    document = episode.to_document()
    episode_validator().validate(document)
    assert document["schema_version"] == "midojo-episode-v1"
    for asago_only in ("tool", "tool_revision_range", "package_id", "scenario_id", "package_digest", "claim_level"):
        assert asago_only not in document
    assert "episode" not in document
    assert "midojo-episode-v2" not in json.dumps(document)


def test_load_document_reads_a_v0_file_with_the_warning(tmp_path: Path) -> None:
    path = tmp_path / "episode.json"
    path.write_text((FIXTURES / "v0_cm58_episode.json").read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.warns(DeprecationWarning, match="midojo-episode-v2"):
        episode = Episode.load_document(path)

    assert episode.interception.upstream is None
    assert episode.suite_name is None


def test_a_v0_wrapper_with_a_nonempty_verifier_list_is_refused() -> None:
    wrapper = read_vector("v0_cm58_episode.json")
    wrapper["verifiers"] = [{"check": "asago_rubric_judge_episode", "module": "asago_verifiers.asago_episode_judge"}]

    with pytest.warns(DeprecationWarning), pytest.raises(EpisodeDocumentError, match="verifiers"):
        Episode.from_document(wrapper)


def test_a_v0_wrapper_with_another_top_level_key_is_refused() -> None:
    wrapper = read_vector("v0_cm58_episode.json")
    wrapper["metadata"] = {"anything": True}

    with pytest.warns(DeprecationWarning), pytest.raises(EpisodeDocumentError, match="metadata"):
        Episode.from_document(wrapper)


def test_a_v0_wrapper_with_an_unknown_episode_member_is_refused() -> None:
    wrapper = read_vector("v0_cm58_episode.json")
    wrapper["episode"]["environment_type"] = "EpisodeEnvironment"

    with pytest.warns(DeprecationWarning), pytest.raises(EpisodeDocumentError, match="environment_type"):
        Episode.from_document(wrapper)


def test_a_v0_wrapper_with_a_literal_upstream_keeps_it() -> None:
    wrapper = read_vector("v0_cm58_episode.json")
    wrapper["interception"]["upstream"] = "http://localhost:8081/mcp"

    with pytest.warns(DeprecationWarning):
        episode = Episode.from_document(wrapper)

    assert episode.interception.upstream == "http://localhost:8081/mcp"


def test_a_v0_wrapper_without_a_version_reads_as_v0() -> None:
    wrapper = read_vector("v0_cm58_episode.json")
    del wrapper["schema_version"]

    with pytest.warns(DeprecationWarning, match="midojo-episode-v2"):
        episode = Episode.from_document(wrapper)

    assert episode.suite_name is None


# --- VAL-DOC-007: version refusals ---


@pytest.mark.parametrize("version", ["midojo-episode-v9", "midojo-episode-v1.1", "", "2"])
def test_an_unknown_version_is_refused_naming_it_and_the_supported_versions(version: str) -> None:
    document = read_vector("v1_single.json")
    document["schema_version"] = version

    with pytest.raises(UnsupportedDocumentVersion) as raised:
        Episode.from_document(document)

    assert raised.value.version == version
    assert raised.value.supported == (EPISODE_DOCUMENT_V1,)
    assert repr(version) in str(raised.value)
    assert EPISODE_DOCUMENT_V1 in str(raised.value)


def test_a_non_string_version_is_refused() -> None:
    document = read_vector("v1_single.json")
    document["schema_version"] = 1

    with pytest.raises(EpisodeDocumentError, match="schema_version") as raised:
        Episode.from_document(document)

    assert not isinstance(raised.value, UnsupportedDocumentVersion)


def test_a_document_without_a_version_and_without_the_wrapper_shape_is_refused() -> None:
    document = read_vector("v1_single.json")
    del document["schema_version"]
    del document["plan"]

    with pytest.raises(EpisodeDocumentError, match="schema_version"):
        Episode.from_document(document)


def test_the_version_is_read_before_anything_else() -> None:
    document = read_vector("v1_single.json")
    document["schema_version"] = "midojo-episode-v9"
    document["turns"] = 3

    with pytest.raises(UnsupportedDocumentVersion):
        Episode.from_document(document)
