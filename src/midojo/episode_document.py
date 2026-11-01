"""The episode as a document: the closed v1 form, and the v0 wrapper it reads for one release.

An episode document is the same data an :class:`~midojo.episode.Episode` holds,
without the trusted code (the environment model and the verifier
implementations): a suite, a driver or another repository can write one, read
it and run it. The schema is the SDK's
(``midojo_sdk.episode_schema``), so every repository validates with the same
file; this module is the reader and writer MiDojo itself uses, and a test holds
the two in agreement over the golden vectors.

The document is closed. A key MiDojo does not define is refused at every level
(the top level, the plan, an instruction's target, the interception settings,
the tool policy and each runtime definition), so a misspelled field never
silently disappears. ``schema_version`` says which form the document is: this
module writes only :data:`~midojo_sdk.episode_schema.EPISODE_DOCUMENT_V1` and
reads the v0 consumer wrapper (``midojo-episode-v2``) with a
:class:`DeprecationWarning`. Any other version raises
:class:`UnsupportedDocumentVersion`, naming it.

The plan in a document is the SDK's plan encoding; its version is the
document's ``schema_version``, so changing the plan's bytes needs a new
document version (see :data:`midojo_sdk.injection.PLAN_CODEC_VERSION`).

``verifiers`` is always an empty list: a verifier registry is trusted code, not
data, so a document can name none. An episode built with its own registry has
no document form, and :func:`to_document` refuses it rather than dropping the
registry silently.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from midojo_sdk.episode_schema import EPISODE_DOCUMENT_V1
from midojo_sdk.injection import InjectionPlan

from midojo.episode import Episode, InterceptionSettings, RuntimeCheckError, ToolPolicy, thaw

SUPPORTED_DOCUMENT_VERSIONS = (EPISODE_DOCUMENT_V1,)
"""The document versions this module writes and reads."""

LEGACY_DOCUMENT_VERSIONS = ("midojo-episode-v2",)
"""The consumer's v0 wrapper: read for one release, never written."""

_V1_FIELDS = frozenset(
    {
        "schema_version",
        "suite_name",
        "user_task_id",
        "injection_task_id",
        "environment_type",
        "environment",
        "runtime",
        "turns",
        "plan",
        "security",
        "utility",
        "verifiers",
        "interception",
        "tools",
    }
)
_INTERCEPTION_FIELDS = frozenset({"mode", "upstream", "bind_host", "advertise_host", "port", "observe_requests"})
_TOOLS_FIELDS = frozenset({"allowed", "write_disposition"})
_WRITE_DISPOSITION_FIELDS = frozenset({"default", "tools"})
_RUNTIME_TYPES = ("unmanaged", "openshell")

# The v0 wrapper's inner episode: the members MiDojo reads. ``environment_type`` is never in a v0 file; MiDojo
# infers it, as it always has, from the suite name and the environment.
_V0_EPISODE_FIELDS = frozenset(
    {
        "turns",
        "plan",
        "security",
        "utility",
        "environment",
        "runtime",
        "suite_name",
        "user_task_id",
        "injection_task_id",
    }
)
# The consumer's own members: the consumer and orch read them from the file, and MiDojo ignores them.
_V0_ASAGO_ONLY = frozenset(
    {
        "tool",
        "tool_revision_range",
        "package_id",
        "scenario_id",
        "package_digest",
        "claim_level",
        "delivery",
        "placements",
    }
)
_V0_TOP_LEVEL = frozenset({"schema_version", "episode", "tools", "interception", "verifiers"}) | _V0_ASAGO_ONLY


class EpisodeDocumentError(ValueError):
    """A document MiDojo refuses: an unknown key, a wrong type, or a check that does not fit the runtime.

    ``gap`` names why a check does not fit the runtime it declares
    (``check:openshell-predicate-on-unmanaged-runtime``), for a caller that
    reports what is missing; it is None for every other refusal.
    """

    def __init__(self, message: str, *, gap: str | None = None) -> None:
        self.gap = gap
        super().__init__(message)


class UnsupportedDocumentVersion(EpisodeDocumentError):  # noqa: N818 - named by the document contract, not a lint choice.
    """A document whose ``schema_version`` MiDojo neither reads nor writes. ``version`` names it."""

    def __init__(self, version: object) -> None:
        self.version = version
        self.supported = SUPPORTED_DOCUMENT_VERSIONS
        super().__init__(
            f"episode document version {version!r} is not supported; "
            f"supported: {', '.join(SUPPORTED_DOCUMENT_VERSIONS)}; "
            f"read-only legacy: {', '.join(LEGACY_DOCUMENT_VERSIONS)}"
        )


def to_document(episode: Episode) -> dict[str, Any]:
    """``episode`` as a v1 document: always v1, never the v0 wrapper.

    Every member is written, null where the episode has none, so two dumps of
    the same episode are the same bytes. ``environment_type`` names the
    environment model only when the episode has a suite name: a direct
    episode's model is MiDojo's own inference, not provenance.
    """
    if episode.verifiers is not None:
        raise EpisodeDocumentError(
            "verifiers: an episode with its own verifier registry has no document form; a registry is trusted code, "
            "not data, so a document names no verifier"
        )
    environment_type = None
    if episode.suite_name is not None:
        environment_type = episode.environment_type.__name__ if episode.environment_type is not None else None
    tools: dict[str, Any] | None = None
    if episode.tools is not None:
        tools = {
            "allowed": list(episode.tools.allowed),
            "write_disposition": {
                "default": episode.tools.write_disposition_default,
                "tools": dict(episode.tools.write_disposition),
            },
        }
    interception = episode.interception
    return {
        "schema_version": EPISODE_DOCUMENT_V1,
        "suite_name": episode.suite_name,
        "user_task_id": episode.user_task_id,
        "injection_task_id": episode.injection_task_id,
        "environment_type": environment_type,
        "environment": None if episode.environment is None else thaw(episode.environment),
        "runtime": thaw(episode.runtime),
        "turns": None if episode.turns is None else list(episode.turns),
        "plan": episode.plan.to_data(),
        "security": None if episode.security is None else thaw(episode.security),
        "utility": None if episode.utility is None else thaw(episode.utility),
        "verifiers": [],
        "interception": {
            "mode": interception.mode,
            "upstream": interception.upstream,
            "bind_host": interception.bind_host,
            "advertise_host": interception.advertise_host,
            "port": interception.port,
            "observe_requests": interception.observe_requests,
        },
        "tools": tools,
    }


def from_document(cls: type[Episode], document: Mapping[str, object]) -> Episode:
    """An episode from a v1 document, or from the v0 consumer wrapper with a :class:`DeprecationWarning`."""
    data = _object(document, "episode document")
    if "schema_version" not in data:
        if isinstance(data.get("episode"), Mapping):
            return _read_v0(cls, data)
        raise EpisodeDocumentError("episode document: schema_version is required")
    version = data["schema_version"]
    if type(version) is not str:
        raise EpisodeDocumentError(f"episode document: schema_version must be a string, not {type(version).__name__}")
    if version == EPISODE_DOCUMENT_V1:
        return _read_v1(cls, data)
    if version in LEGACY_DOCUMENT_VERSIONS:
        return _read_v0(cls, data)
    raise UnsupportedDocumentVersion(version)


def load_document(cls: type[Episode], path: str | Path) -> Episode:
    """An episode from the JSON document at ``path``: :func:`from_document` of its decoded contents."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EpisodeDocumentError(f"{path}: not JSON: {exc}") from None
    return from_document(cls, document)


def dump_document(episode: Episode, path: str | Path) -> None:
    """Write ``episode``'s v1 document to ``path`` as canonical JSON: sorted keys, two-space indent, one newline."""
    Path(path).write_text(json.dumps(to_document(episode), indent=2, sort_keys=True) + "\n", encoding="utf-8")


# --- reading ---


def _read_v1(cls: type[Episode], data: Mapping[str, Any]) -> Episode:
    _closed(data, _V1_FIELDS, "episode document")
    _verifiers(data.get("verifiers"))
    turns = _turns(data.get("turns"))
    try:
        episode = cls(
            runtime=_runtime(data.get("runtime")),
            turns=turns,
            plan=_plan(data.get("plan")),
            security=_data_object(data.get("security"), "security"),
            utility=_data_object(data.get("utility"), "utility"),
            environment=_data_object(data.get("environment"), "environment"),
            suite_name=_string(data.get("suite_name"), "suite_name"),
            user_task_id=_string(data.get("user_task_id"), "user_task_id"),
            injection_task_id=_string(data.get("injection_task_id"), "injection_task_id"),
            interception=_interception(data.get("interception")),
            tools=_tools(data.get("tools")),
        )
    except RuntimeCheckError as exc:
        raise EpisodeDocumentError(f"episode document: {exc}", gap=exc.gap) from None
    except (TypeError, ValueError) as exc:
        raise EpisodeDocumentError(f"episode document: {exc}") from None
    _environment_type(episode, _string(data.get("environment_type"), "environment_type"))
    return episode


def _read_v0(cls: type[Episode], data: Mapping[str, Any]) -> Episode:
    warnings.warn(
        f"the episode document version {LEGACY_DOCUMENT_VERSIONS[0]!r} is read as v0 and is deprecated; "
        f"write {EPISODE_DOCUMENT_V1!r} instead",
        DeprecationWarning,
        stacklevel=3,
    )
    _closed(data, _V0_TOP_LEVEL, "episode document (v0)")
    if data.get("verifiers", []) != []:
        raise EpisodeDocumentError(
            "verifiers: a v0 wrapper must list no verifier; MiDojo reads only [] and names none in a document"
        )
    inner = _object(data.get("episode"), "episode")
    _closed(inner, _V0_EPISODE_FIELDS, "episode (v0)")
    return _read_v1(
        cls,
        {
            "schema_version": EPISODE_DOCUMENT_V1,
            "suite_name": inner.get("suite_name"),
            "user_task_id": inner.get("user_task_id"),
            "injection_task_id": inner.get("injection_task_id"),
            "environment_type": None,
            "environment": inner.get("environment"),
            "runtime": inner.get("runtime"),
            "turns": inner.get("turns"),
            "plan": inner.get("plan"),
            "security": inner.get("security"),
            "utility": inner.get("utility"),
            "verifiers": [],
            "interception": _v0_interception(data.get("interception")),
            "tools": data.get("tools"),
        },
    )


def _v0_interception(value: object) -> dict[str, Any]:
    """The v0 interception settings, with the late-bound ``endpoint.mcp_url`` marker read as no upstream."""
    data = {} if value is None else dict(_object(value, "interception"))
    if data.get("upstream") == "endpoint.mcp_url":
        data["upstream"] = None
    return data


# --- validating one member ---


def _object(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EpisodeDocumentError(f"{where} must be a JSON object, not {type(value).__name__}")
    return value


def _closed(data: Mapping[str, object], allowed: frozenset[str], where: str) -> None:
    if unknown := [key for key in data if key not in allowed]:
        raise EpisodeDocumentError(f"{where}: unknown key {unknown[0]!r}; the document is closed")


def _string(value: object, where: str) -> str | None:
    if value is None or type(value) is str:
        return value
    raise EpisodeDocumentError(f"{where} must be a string or null, not {type(value).__name__}")


def _turns(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if type(value) is not list:
        raise EpisodeDocumentError(f"turns must be an array of strings or null, not {type(value).__name__}")
    if not value:
        raise EpisodeDocumentError("turns must hold at least one message, or be null when the host drives them")
    for index, message in enumerate(value):
        if type(message) is not str:
            raise EpisodeDocumentError(f"turns[{index}] must be a string, not {type(message).__name__}")
    return tuple(value)


def _plan(value: object) -> InjectionPlan:
    if value is None:
        return InjectionPlan()
    try:
        return InjectionPlan.from_data(_object(value, "plan"))
    except ValueError as exc:
        raise EpisodeDocumentError(f"plan: {exc}") from None


def _runtime(value: object) -> Mapping[str, Any]:
    if value is None:
        return {"type": "unmanaged"}
    data = _object(value, "runtime")
    if data.get("type") not in _RUNTIME_TYPES:
        raise EpisodeDocumentError(
            f"runtime.type: {data.get('type')!r} is not one of the runtime definitions this document defines "
            f"({', '.join(repr(name) for name in _RUNTIME_TYPES)})"
        )
    return data


def _data_object(value: object, where: str) -> Mapping[str, Any] | None:
    return None if value is None else _object(value, where)


def _verifiers(value: object) -> None:
    """Refuse a document that names a verifier: a registry is trusted code, so a document names none."""
    if value is None:
        return
    if type(value) is not list or value:
        raise EpisodeDocumentError(
            "verifiers: a document names no verifier; MiDojo reads only [] because a verifier registry is trusted "
            "code, not data"
        )


def _interception(value: object) -> InterceptionSettings:
    if value is None:
        return InterceptionSettings()
    data = _object(value, "interception")
    _closed(data, _INTERCEPTION_FIELDS, "interception")
    try:
        return InterceptionSettings(
            mode=data.get("mode", "host"),
            upstream=_string(data.get("upstream"), "interception.upstream"),
            bind_host=_string(data.get("bind_host"), "interception.bind_host"),
            advertise_host=_string(data.get("advertise_host"), "interception.advertise_host"),
            port=data.get("port"),
            observe_requests=data.get("observe_requests", False),
        )
    except (TypeError, ValueError) as exc:
        raise EpisodeDocumentError(f"interception: {exc}") from None


def _tools(value: object) -> ToolPolicy | None:
    if value is None:
        return None
    data = _object(value, "tools")
    _closed(data, _TOOLS_FIELDS, "tools")
    if "allowed" not in data:
        raise EpisodeDocumentError("tools.allowed is required")
    allowed = data["allowed"]
    if type(allowed) is not list or any(type(name) is not str for name in allowed):
        raise EpisodeDocumentError("tools.allowed must be an array of strings")
    disposition = data.get("write_disposition", {})
    settings = _object(disposition, "tools.write_disposition")
    _closed(settings, _WRITE_DISPOSITION_FIELDS, "tools.write_disposition")
    per_tool = settings.get("tools", {})
    overrides = _object(per_tool, "tools.write_disposition.tools")
    try:
        return ToolPolicy(
            tuple(allowed),
            write_disposition_default=settings.get("default", "forward"),
            write_disposition=dict(overrides),
        )
    except (TypeError, ValueError) as exc:
        raise EpisodeDocumentError(f"tools: {exc}") from None


def _environment_type(episode: Episode, named: str | None) -> None:
    """Refuse a document that names an environment model MiDojo would not infer for it."""
    if named is None:
        return
    inferred = episode.environment_type.__name__ if episode.environment_type is not None else None
    if named != inferred:
        raise EpisodeDocumentError(
            f"environment_type: the episode's suite_name and environment imply {inferred!r}, not {named!r}"
        )
