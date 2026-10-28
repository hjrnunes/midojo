"""Runtime observation types, keyed by source.

Runtime observations are evidence collected outside the agent, such as
OpenShell's workdir diff and OCSF events. Each source registers the Pydantic
model its observations must match. The control plane validates recorded
observations against that model and rejects unknown sources, so verifiers
receive typed models instead of raw JSON.

A backend registers its source when its module is imported::

    register_observation_type("openshell", OpenShellObservations)

Evidence status:
  A source reports how completely it collected each part of its evidence, as an
  ``evidence_status`` field on its model (e.g. OpenShell's ``ocsf`` and
  ``workdir``). :func:`evidence_status` flattens these into one map keyed by
  ``<source>.<part>`` (``openshell.ocsf``). The map is open to other evidence
  that isn't a runtime observation, such as :data:`FUNCTION_CALLS`: the
  control plane marks it ``incomplete`` when a call record may be missing (an
  interception layer reported a record it couldn't deliver, or closure refused
  one), and a key that's absent has no known gap.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Literal

from pydantic import BaseModel

# How completely a part of the evidence was collected:
#   complete       everything was collected
#   incomplete     some was collected, but not all (e.g. the log sync timed out)
#   failed         collection was attempted and nothing usable came back
#   not_collected  collection never ran (e.g. no observations were recorded)
EvidenceStatus = Literal["complete", "incomplete", "failed", "not_collected"]

FUNCTION_CALLS = "function_calls"
"""The evidence-status key of the recorded tool calls."""

# The evidence the control plane itself records, apart from runtime observations.
RECORD_EVIDENCE = frozenset({FUNCTION_CALLS, "environment", "turns", "agent_output"})

_OBSERVATION_TYPES: dict[str, type[BaseModel]] = {}


def register_observation_type(source: str, model: type[BaseModel]) -> None:
    """Register the model that a source's observations must match."""
    if _OBSERVATION_TYPES.get(source, model) is not model:
        raise ValueError(f"Observation source {source!r} is already registered")
    _OBSERVATION_TYPES[source] = model


def observation_type(source: str) -> type[BaseModel] | None:
    """The model registered for a source, or None if the source is unknown."""
    return _OBSERVATION_TYPES.get(source)


def evidence_status(observations: Mapping[str, BaseModel]) -> dict[str, EvidenceStatus]:
    """The evidence status each observation reports, keyed by ``<source>.<part>``."""
    statuses: dict[str, EvidenceStatus] = {}
    for source, observed in observations.items():
        reported = getattr(observed, "evidence_status", None)
        if isinstance(reported, BaseModel):
            statuses.update({f"{source}.{part}": status for part, status in reported.model_dump().items()})
    return statuses


def digest_content(observed: BaseModel) -> dict[str, Any]:
    """An observation as the snapshot digest encodes it: its dump, a later list field left out while empty.

    A model names the list fields it gained after snapshots were first
    digested in a ``DIGEST_OMITS_WHEN_EMPTY`` class attribute. Each is left
    out only while it is exactly an empty ``list``, its default, so a snapshot
    recorded before the field existed keeps its digest. Any other value is
    encoded as it is (``None``, ``False``, ``0`` or ``{}`` each digest apart
    from ``[]``), and a missing field raises ValueError: neither can pass for
    the default.
    """
    content = observed.model_dump()
    for name in getattr(observed, "DIGEST_OMITS_WHEN_EMPTY", ()):
        if name not in content:
            raise ValueError(f"{type(observed).__name__} has no {name!r} field")
        if type(content[name]) is list and not content[name]:
            del content[name]
    return content


def is_evidence_key(key: str) -> bool:
    """Whether ``key`` names evidence an evidence status can mark.

    That is one of :data:`RECORD_EVIDENCE`, a registered observation source,
    or ``<source>.<part>`` for a part its model's ``evidence_status`` reports.
    """
    if key in RECORD_EVIDENCE or key in _OBSERVATION_TYPES:
        return True
    source, _, part = key.partition(".")
    model = _OBSERVATION_TYPES.get(source)
    status = model.model_fields.get("evidence_status") if model is not None else None
    reported = status.annotation if status is not None else None
    return isinstance(reported, type) and issubclass(reported, BaseModel) and part in reported.model_fields


def require_evidence_keys(keys: Iterable[str]) -> None:
    """Raise ValueError unless every key names evidence an evidence status can mark (:func:`is_evidence_key`)."""
    if unknown := [key for key in keys if not is_evidence_key(key)]:
        raise ValueError(f"Unknown evidence keys: {unknown}")
