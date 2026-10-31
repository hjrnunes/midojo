"""MiDojo's snapshot ID: a sealed snapshot's content digest, and the read view of that content.

MiDojo names a sealed snapshot by its own content (:func:`snapshot_digest`): ``sha256:<hex>``
of a lossless encoding of everything in the snapshot but its ID. The encoding takes only JSON
values: ``str``, ``int``, ``bool`` and ``None`` are themselves; a float is
``["float", <its IEEE-754 bits as big-endian hex>]``, so ``-0.0``, every NaN payload and every
sign encode apart from the values around them; a list is ``["list", [items]]``; an object is
``["mapping", [[key, value], ...]]`` with its items sorted by their encoding, written compactly
(``separators=(",", ":")``) with ASCII escapes. Two snapshots share an ID only when their
content is equal, and a value JSON does not carry (``bytes``, a date, a tuple, a set, a
non-finite float, a mapping with a key that is not text) is refused with :class:`DigestFault`
rather than encoded: nothing JSON holds it back as.

A snapshot is the JSON form ``GET /runs/{run_id}/evaluations/{eval_id}/snapshot`` serves, so a
consumer reads the ID and verifies it (:func:`verified`, :class:`SnapshotView`) without any
MiDojo code. MiDojo's own ``snapshot_id`` is this digest: a snapshot it seals holds only JSON
values, so the ID a client reads is the one this module computes from the bytes it read. A
snapshot holding a value JSON does not carry back (a test-built one, or an environment a suite
authored with a ``bytes`` field) keeps MiDojo's Python-value encoder
(``midojo.control_plane.state``); such a snapshot's ID names content a client that rebuilds it
from the served JSON does not have, so ``midojo.control_plane.client.snapshot_from_response``
refuses it.

Two rules keep a snapshot's ID stable across MiDojo releases:

- An injection outcome's ``payload_digest``, ``placed_digest`` and ``slot`` are encoded only when
  set (:data:`LATER_OUTCOME_FIELDS`): a ``None`` one is left out, so a snapshot sealed before
  the fields existed keeps its ID. The JSON form holds each as null, or not at all.
- An observation is encoded without the list fields its model names in
  ``DIGEST_OMITS_WHEN_EMPTY``, each left out only while it is exactly an empty list. JSON does
  not carry the model, so the names are kept here by the source the observation is recorded
  under, which is its key in ``observations`` (:data:`OBSERVATION_OMITS_WHEN_EMPTY`): OpenShell's
  ``openshell`` names ``http_requests``. Any other value, null included, is encoded as it is, and
  a field that is absent (a snapshot sealed before it existed) stays absent.

``SNAPSHOT_DIGEST_VERSION`` names this encoding. A change to the bytes it hashes is a change to
every ID, so it requires a new version and a migration, not a quiet edit.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

# The encoding's version: a change to the bytes it hashes renames every snapshot.
SNAPSHOT_DIGEST_VERSION: int = 1

# The snapshot's content, in ``snapshot_digest``'s order: every field but ``snapshot_id``.
CONTENT_FIELDS: tuple[str, ...] = (
    "evaluation_id",
    "run_id",
    "user_task_id",
    "injection_task_id",
    "agent_output",
    "pre_environment",
    "environment",
    "function_calls",
    "turns",
    "observations",
    "evidence_status",
    "sealed_at",
    "capture_failures",
)

# The fields an injection outcome gained after snapshots were first digested: each is encoded only when set.
LATER_OUTCOME_FIELDS: tuple[str, ...] = ("payload_digest", "placed_digest", "slot")

# Each observation source's list fields MiDojo encodes only when not empty (its model's
# ``DIGEST_OMITS_WHEN_EMPTY``), by the source the model is recorded under.
OBSERVATION_OMITS_WHEN_EMPTY: Mapping[str, tuple[str, ...]] = {"openshell": ("http_requests",)}


class DigestFault(ValueError):  # noqa: N818  -- the architecture names this fault, and callers catch it by name
    """A snapshot this codec refuses: not JSON data, a field missing, or an ID its content does not have."""


def snapshot_content(snapshot: Mapping[str, object]) -> dict[str, object]:
    """Return the snapshot's content as the digest encodes it: every field of :data:`CONTENT_FIELDS`.

    Each call record is passed through :func:`call_content` and each observation through
    :func:`observation_content`. Raises :class:`DigestFault` when ``snapshot`` is not a mapping or
    is missing one of the fields.
    """

    if not isinstance(snapshot, Mapping):
        raise DigestFault(f"a snapshot is a JSON object, not a {type(snapshot).__name__}")
    missing = [name for name in CONTENT_FIELDS if name not in snapshot]
    if missing:
        raise DigestFault(f"the snapshot is missing {', '.join(missing)}")
    content = {name: snapshot[name] for name in CONTENT_FIELDS}
    if isinstance(content["function_calls"], list):
        content["function_calls"] = [call_content(call) for call in content["function_calls"]]
    if isinstance(content["observations"], dict):
        content["observations"] = {
            source: observation_content(source, observed) for source, observed in content["observations"].items()
        }
    return content


def call_content(call: object) -> object:
    """Return a call record as the digest encodes it: each injection outcome without a later field that is None.

    Only ``payload_digest``, ``placed_digest`` and ``slot`` are encoded when set
    (:data:`LATER_OUTCOME_FIELDS`), so a record sealed without them digests as it did before. ``call``
    itself is left as it is.
    """

    if not isinstance(call, dict) or not isinstance(call.get("injection_outcomes"), list):
        return call
    outcomes = [
        {key: value for key, value in outcome.items() if not (key in LATER_OUTCOME_FIELDS and value is None)}
        if isinstance(outcome, dict)
        else outcome
        for outcome in call["injection_outcomes"]
    ]
    return {**call, "injection_outcomes": outcomes}


def observation_content(source: object, observed: object) -> object:
    """Return an observation as the digest encodes it: without each of its source's empty list fields.

    A field named in :data:`OBSERVATION_OMITS_WHEN_EMPTY` for ``source`` is left out while it is
    exactly an empty list, and kept otherwise. ``observed`` itself is left as it is.
    """

    omitted = OBSERVATION_OMITS_WHEN_EMPTY.get(source, ()) if isinstance(source, str) else ()
    if not (omitted and isinstance(observed, dict)):
        return observed
    return {key: value for key, value in observed.items() if not (key in omitted and type(value) is list and not value)}


def snapshot_digest(snapshot: Mapping[str, object]) -> str:
    """Return ``sha256:<hex>`` of the snapshot's content: the ID MiDojo names a sealed snapshot by.

    Raises :class:`DigestFault` for a field missing, for a value JSON does not carry, or for a
    snapshot that is not a mapping.
    """

    encoded = dumps(canonical(snapshot_content(snapshot)))
    return f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"


def verified(snapshot: Mapping[str, object]) -> Mapping[str, object]:
    """Return ``snapshot`` once its content is what its ``snapshot_id`` names; otherwise raise :class:`DigestFault`.

    A snapshot without an ID, with an ID that is not text, or with content that digests to another
    ID is refused: nothing else holds the content, so a caller that wants it read only once holds
    what this returned.
    """

    if not isinstance(snapshot, Mapping):
        raise DigestFault(f"a snapshot is a JSON object, not a {type(snapshot).__name__}")
    named = snapshot.get("snapshot_id")
    if named is None:
        raise DigestFault("the snapshot has no snapshot_id")
    if not isinstance(named, str):
        raise DigestFault(f"the snapshot's snapshot_id is a {type(named).__name__}, not text")
    computed = snapshot_digest(snapshot)
    if computed != named:
        raise DigestFault(f"the snapshot's content digests to {computed}, not {named}")
    return snapshot


def canonical(value: object) -> object:
    """Return the lossless encoding of a JSON ``value``: unequal values encode differently.

    Exactly ``str``, ``int`` and ``bool``, and ``None``, are themselves; any other value is a
    ``[tag, content]`` pair, so no value is mistaken for another. A float is its IEEE-754 bits,
    so every NaN payload and sign encodes apart too. Raises :class:`DigestFault` for any value
    JSON does not carry (``bytes``, a tuple, a set, a date, a mapping with a key that is not text).
    """

    kind = type(value)
    if value is None or kind in (str, int, bool):
        return value
    if kind is float:
        return ["float", struct.pack(">d", cast("float", value)).hex()]
    if kind is list:
        return ["list", [canonical(item) for item in cast("list[object]", value)]]
    if kind is dict:
        mapping = cast("dict[object, object]", value)
        for key in mapping:
            if type(key) is not str:
                raise DigestFault(f"a {type(key).__name__} is not a JSON object key")
        items = [[canonical(key), canonical(item)] for key, item in mapping.items()]
        return ["mapping", sorted(items, key=dumps)]
    raise DigestFault(f"a {kind.__name__} is not a JSON value")


def dumps(encoded: object) -> str:
    """Return the text the digest hashes, and sorts mapping items by: compact, with ASCII escapes."""

    return json.dumps(encoded, separators=(",", ":"))


@dataclass(frozen=True)
class SnapshotView:
    """A sealed snapshot read as its verified ID and content, holding no MiDojo models.

    ``content`` is :func:`snapshot_content`'s form: the fields the digest encodes, without the ID.
    Its top level is read-only; the JSON containers inside it are the ones that were read.
    """

    snapshot_id: str
    content: Mapping[str, object]

    @classmethod
    def from_json(cls, data: str | bytes | Mapping[str, object]) -> SnapshotView:
        """Read and verify a sealed snapshot, from its JSON text or from the JSON object itself.

        Raises :class:`DigestFault` for text that is not JSON (``NaN`` and the infinities
        included), for JSON that is not an object, and for a snapshot :func:`verified` refuses.
        """

        snapshot = _read(data)
        verified(snapshot)
        return cls(
            snapshot_id=cast("str", snapshot["snapshot_id"]),
            content=MappingProxyType(snapshot_content(snapshot)),
        )


def _read(data: str | bytes | Mapping[str, object]) -> Mapping[str, object]:
    """``data`` as the snapshot object it holds: parsed from JSON text, or the mapping itself."""

    if isinstance(data, Mapping):
        return data
    if isinstance(data, str | bytes | bytearray):
        try:
            decoded: Any = json.loads(data, parse_constant=_no_constant)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DigestFault(f"the snapshot is not JSON: {error}") from None
        if not isinstance(decoded, Mapping):
            raise DigestFault(f"a snapshot is a JSON object, not a {type(decoded).__name__}")
        return decoded
    raise DigestFault(f"a snapshot is JSON text or an object, not a {type(data).__name__}")


def _no_constant(name: str) -> Any:
    """Refuse ``NaN`` and the infinities: a JSON document holds no number a float cannot round-trip as text."""

    raise DigestFault(f"the snapshot holds {name}, which JSON does not")
