"""The snapshot codec: the digest MiDojo names a sealed snapshot by, and the read view of one.

The vectors in ``vectors/snapshot/vectors.json`` were recorded from MiDojo's implementation
before the digest moved out of the runtime (commit 770118e), each with the JSON form
``GET .../snapshot`` served and the id that implementation computed for it. This package must
reproduce every one of them, and refuse the corruptions listed with it.

One recorded value changed afterwards: the run summary inside the ``recorded-openshell-run``
vector names its schema ``agent-run-summary-v1``, and that vector's ``snapshot_id`` is the
digest of the edited snapshot.

The refusals of ``NaN`` and the infinities came later: each holds the id the codec computed for
that value's IEEE-754 bits before it refused them (checked with ``shasum -a 256``), which is the id
MiDojo's own encoder names such a snapshot by, so only the refusal of the value stops it.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from midojo_spec.snapshot import (
    CONTENT_FIELDS,
    LATER_OUTCOME_FIELDS,
    OBSERVATION_OMITS_WHEN_EMPTY,
    SNAPSHOT_DIGEST_VERSION,
    DigestFault,
    SnapshotView,
    canonical,
    dumps,
    snapshot_content,
    snapshot_digest,
    verified,
)

PACKAGE = Path(__file__).resolve().parent.parent
VECTORS_PATH = PACKAGE / "vectors/snapshot/vectors.json"

VECTORS = json.loads(VECTORS_PATH.read_text())
ACCEPTED = [vector for vector in VECTORS if "refusal" not in vector]
REFUSED = [vector for vector in VECTORS if "refusal" in vector]
RECORDED_RUN_ID = "sha256:d0e377a085a5526d0a5fb00e0dcfbd3de43b46951cbcf126a30af8b5f483f8ad"


def vector(name: str) -> dict:
    return next(entry for entry in VECTORS if entry["name"] == name)


def names(entries: list[dict]) -> list[str]:
    return [entry["name"] for entry in entries]


# --- the recorded vectors ---


@pytest.mark.parametrize("entry", ACCEPTED, ids=names(ACCEPTED))
def test_the_sdk_reproduces_the_id_midojo_recorded(entry):
    assert snapshot_digest(entry["snapshot"]) == entry["snapshot_id"]
    assert verified(entry["snapshot"]) is entry["snapshot"]


@pytest.mark.parametrize("entry", ACCEPTED, ids=names(ACCEPTED))
def test_a_snapshot_view_reads_and_verifies_the_same_json_form(entry):
    view = SnapshotView.from_json(json.dumps(entry["snapshot"]))

    assert view.snapshot_id == entry["snapshot_id"]
    assert dict(view.content) == snapshot_content(entry["snapshot"])
    assert "snapshot_id" not in view.content
    assert SnapshotView.from_json(json.dumps(entry["snapshot"]).encode()) == view
    assert SnapshotView.from_json(entry["snapshot"]) == view


def test_the_recorded_run_keeps_the_id_the_consumers_pin():
    assert snapshot_digest(vector("recorded-openshell-run")["snapshot"]) == RECORDED_RUN_ID


def test_the_vectors_cover_every_required_case():
    covered = {entry["name"] for entry in VECTORS}

    assert {
        "recorded-openshell-run",
        "openshell-empty-http-requests",
        "consumer-openshell-http-requests-empty",
        "later-outcomes-set",
        "later-outcomes-null",
        "float-edge",
        "non-ascii",
        "refused-non-json-value",
        "refused-nan",
        "refused-infinity",
        "refused-negative-infinity",
        "refused-missing-field",
        "refused-tampered-id",
    } <= covered
    for entry in VECTORS:
        assert {"name", "snapshot", "snapshot_id"} <= set(entry)
    assert vector("openshell-empty-http-requests")["snapshot"]["observations"]["openshell"]["http_requests"] == []
    assert vector("later-outcomes-set")["snapshot"]["function_calls"][0]["injection_outcomes"][0]["slot"]
    later = vector("later-outcomes-null")["snapshot"]["function_calls"][0]["injection_outcomes"][0]
    assert (later["payload_digest"], later["placed_digest"], later["slot"]) == (None, None, None)


def materialized(entry: dict) -> dict:
    """The snapshot a refusal entry describes, with the non-JSON value JSON can only name put back."""
    value = json.loads(json.dumps(entry["snapshot"]))
    refused = entry["refused_value"]
    assert refused["type"] in ("bytes", "float")
    held = bytes.fromhex(refused["bytes_hex"]) if refused["type"] == "bytes" else float(refused["value"])
    for path in refused["paths"]:
        holder = value
        for step in path[:-1]:
            holder = holder[step]
        holder[path[-1]] = held
    return value


@pytest.mark.parametrize("entry", REFUSED, ids=names(REFUSED))
def test_every_refusal_case_raises_a_digest_fault(entry):
    with pytest.raises(DigestFault):
        SnapshotView.from_json(entry["snapshot"])
    with pytest.raises(DigestFault):
        verified(entry["snapshot"])
    if entry["refusal"] == "id-mismatch":
        # The content still digests: only the name it carries is another snapshot's.
        assert snapshot_digest(entry["snapshot"]) != entry["snapshot_id"]
    elif entry["refusal"] == "non-json":
        # The fixture can only name the value, so the snapshot holding it is rebuilt before asking.
        with pytest.raises(DigestFault):
            snapshot_digest(materialized(entry))
        # The id is the one that value's own bits digest to, so only the refusal of the value itself stops it.
        resealed = {**materialized(entry), "snapshot_id": entry["snapshot_id"]}
        with pytest.raises(DigestFault):
            verified(resealed)
        with pytest.raises(DigestFault):
            SnapshotView.from_json(resealed)
    else:
        with pytest.raises(DigestFault):
            snapshot_digest(entry["snapshot"])


def test_the_version_and_the_encoded_fields_are_this_snapshots():
    assert SNAPSHOT_DIGEST_VERSION == 1
    assert CONTENT_FIELDS == (
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
    assert LATER_OUTCOME_FIELDS == ("payload_digest", "placed_digest", "slot")
    assert dict(OBSERVATION_OMITS_WHEN_EMPTY) == {"openshell": ("http_requests",)}


# --- the recorded OpenShell run: verified once, refused for every change a consumer could make ---


def _mutations(snapshot: dict) -> dict[str, dict]:
    changed_output = json.loads(json.dumps(snapshot))
    changed_output["agent_output"] = changed_output["agent_output"] + "."
    changed_call = json.loads(json.dumps(snapshot))
    changed_call["function_calls"][0]["result"] = str(changed_call["function_calls"][0]["result"]) + " "
    missing_id = json.loads(json.dumps(snapshot))
    del missing_id["snapshot_id"]
    other_id = json.loads(json.dumps(snapshot))
    other_id["snapshot_id"] = "sha256:" + "0" * 64
    return {
        "agent_output": changed_output,
        "function_calls": changed_call,
        "missing-id": missing_id,
        "other-id": other_id,
    }


def test_the_recorded_run_verifies_and_every_change_to_it_is_refused():
    snapshot = vector("recorded-openshell-run")["snapshot"]

    assert verified(snapshot) is snapshot
    assert SnapshotView.from_json(snapshot).snapshot_id == RECORDED_RUN_ID
    for changed in _mutations(snapshot).values():
        with pytest.raises(DigestFault):
            verified(changed)
        with pytest.raises(DigestFault):
            SnapshotView.from_json(changed)


# --- the codec ---


def test_the_encoding_keeps_apart_the_values_json_does_not():
    assert canonical(True) is True and canonical(1) == 1
    assert dumps(canonical(True)) != dumps(canonical(1))
    assert canonical(0.0) == ["float", "0000000000000000"]
    assert canonical(-0.0) == ["float", "8000000000000000"]
    assert canonical([1, "a"]) == ["list", [1, "a"]]
    assert canonical({"b": 1, "a": 2}) == ["mapping", [["a", 2], ["b", 1]]]
    assert canonical(None) is None and canonical("x") == "x"
    # A lone surrogate is text, and the encoding escapes it like any other text.
    assert canonical(["\ud800"]) == ["list", ["\ud800"]]


@pytest.mark.parametrize(
    "value",
    [
        b"x",
        bytearray(b"x"),
        (1,),
        {1},
        frozenset({1}),
        object(),
        {1: "a"},
        float("nan"),
        float("inf"),
        float("-inf"),
        [{"a": [float("nan")]}],
    ],
    ids=[
        "bytes",
        "bytearray",
        "tuple",
        "set",
        "frozenset",
        "object",
        "non-text-key",
        "nan",
        "infinity",
        "negative-infinity",
        "nan-nested",
    ],
)
def test_the_encoding_refuses_a_value_json_does_not_carry_back(value):
    with pytest.raises(DigestFault):
        canonical(value)


def test_the_text_hashed_is_compact_and_ascii_escaped():
    assert dumps({"city": "Zürich"}) == '{"city":"Z\\u00fcrich"}'
    assert dumps([1, 2]) == "[1,2]"


def test_the_snapshot_mappings_key_order_does_not_change_the_id():
    entry = vector("recorded-openshell-run")

    assert snapshot_digest(dict(reversed(list(entry["snapshot"].items())))) == RECORDED_RUN_ID


def test_an_injection_outcomes_later_fields_count_only_when_set():
    entry = vector("later-outcomes-set")["snapshot"]
    outcome = next(one for one in entry["function_calls"][0]["injection_outcomes"] if one["slot"] is not None)
    nulled = json.loads(json.dumps(entry))
    for name in LATER_OUTCOME_FIELDS:
        nulled["function_calls"][0]["injection_outcomes"][0][name] = None
    removed = json.loads(json.dumps(nulled))
    for name in LATER_OUTCOME_FIELDS:
        del removed["function_calls"][0]["injection_outcomes"][0][name]

    # Null and absent encode alike, so a snapshot sealed before the fields keeps its id.
    assert snapshot_digest(nulled) == snapshot_digest(removed)
    assert snapshot_digest(nulled) != entry["snapshot_id"]
    content: Any = snapshot_content(nulled)
    assert set(content["function_calls"][0]["injection_outcomes"][0]) == {
        key for key in outcome if key not in LATER_OUTCOME_FIELDS
    }


@pytest.mark.parametrize(
    ("observed", "omitted"),
    [
        ({"http_requests": []}, True),
        ({"http_requests": [], "files_created": ["/a"]}, True),
        ({"http_requests": [{"raw": "HTTP:GET"}], "files_created": ["/a"]}, False),
        ({"http_requests": None}, False),
        ({"http_requests": ()}, False),
    ],
    ids=["empty", "empty-with-others", "one-request", "null", "tuple"],
)
def test_an_observation_list_a_model_names_is_left_out_only_while_empty(observed, omitted):
    content: Any = snapshot_content({**vector("minimal")["snapshot"], "observations": {"openshell": observed}})
    kept = content["observations"]["openshell"]

    assert ("http_requests" in kept) is not omitted
    assert all(kept[name] == value for name, value in observed.items() if not (name == "http_requests" and omitted))


def test_another_sources_lists_are_left_as_they_are():
    content: Any = snapshot_content({**vector("minimal")["snapshot"], "observations": {"other": {"http_requests": []}}})
    assert content["observations"]["other"] == {"http_requests": []}


def test_a_missing_field_and_a_non_json_value_are_digest_faults_not_other_errors():
    snapshot = vector("minimal")["snapshot"]
    missing = {name: value for name, value in snapshot.items() if name != "turns"}
    with pytest.raises(DigestFault):
        snapshot_digest(missing)
    with pytest.raises(DigestFault):
        snapshot_digest({**snapshot, "sealed_at": ("2026-10-29",)})
    with pytest.raises(DigestFault):
        SnapshotView.from_json("{not json")
    with pytest.raises(DigestFault):
        SnapshotView.from_json('[{"snapshot_id": "sha256:' + "0" * 64 + '"}]')
    with pytest.raises(DigestFault):
        SnapshotView.from_json('{"snapshot_id": "sha256:' + "0" * 64 + '", "evaluation_id": NaN}')


@pytest.mark.parametrize("name", ["refused-infinity", "refused-negative-infinity"])
def test_a_json_number_past_the_range_of_a_double_is_refused_like_the_bare_constants(name):
    # json.loads reads 1e999 as an infinity without asking parse_constant, so this text holds a number
    # whose own bits digest to the id it names.
    entry = vector(name)
    text = json.dumps({**materialized(entry), "snapshot_id": entry["snapshot_id"]}).replace("Infinity", "1e999")

    decoded = json.loads(text)
    assert decoded["environment"]["data"]["reading"] == float(entry["refused_value"]["value"])
    assert decoded["snapshot_id"] == entry["snapshot_id"]
    with pytest.raises(DigestFault, match="not a JSON value"):
        SnapshotView.from_json(text)
    with pytest.raises(DigestFault, match="not a JSON value"):
        SnapshotView.from_json(text.encode())


class _NameThatRaises(type):
    """A metaclass whose classes' ``__name__`` raises while ``armed``, so the test runner can still name them."""

    armed = False

    @property
    def __name__(cls) -> str:  # pyright: ignore[reportIncompatibleVariableOverride]
        if _NameThatRaises.armed:
            raise RuntimeError("the name is not readable")
        return "Hostile"


class _Hostile(metaclass=_NameThatRaises):
    pass


def test_an_object_whose_type_name_cannot_be_read_raises_its_own_error_and_is_never_verified():
    # The module docstring says so: DigestFault covers what JSON can hold, and json.loads builds no such object.
    snapshot = vector("minimal")["snapshot"]
    holders = [
        {**snapshot, "snapshot_id": _Hostile()},
        {**snapshot, "environment": {"data": {"x": _Hostile()}}},
    ]

    _NameThatRaises.armed = True
    try:
        for held in holders:
            with pytest.raises(RuntimeError, match="not readable"):
                verified(held)
            with pytest.raises(RuntimeError, match="not readable"):
                SnapshotView.from_json(held)
    finally:
        _NameThatRaises.armed = False


def test_a_digest_fault_is_a_value_error():
    assert issubclass(DigestFault, ValueError)


# --- the import boundary ---


# Every top-level module that is neither the standard library nor this package is unimportable, so nothing the
# module imports can hide: midojo, httpx, pydantic, fastapi, grpc, google.protobuf and openshell are all blocked.
BLOCK_PREAMBLE = """
import importlib.abc
import json
import sys
from pathlib import Path

ALLOWED = set(sys.stdlib_module_names) | {"midojo_spec", "__main__"}


class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in ALLOWED or top.startswith("_frozen_importlib") or top.startswith("_distutils_hack"):
            return None
        raise ModuleNotFoundError(f"No module named {fullname!r} (blocked by the test)", name=fullname)


for name in [name for name in sys.modules if name.split(".")[0] not in ALLOWED]:
    del sys.modules[name]
sys.meta_path.insert(0, Unavailable())


def outside_the_standard_library():
    return sorted(name for name in sys.modules if name.split(".")[0] not in ALLOWED)
"""

RUN_THE_SNAPSHOT_CODEC = """
from midojo_spec.snapshot import SnapshotView, canonical, dumps, snapshot_digest, verified

vectors = json.loads(Path("vectors/snapshot/vectors.json").read_text())
snapshot = next(entry for entry in vectors if entry["name"] == "recorded-openshell-run")["snapshot"]
assert snapshot_digest(snapshot) == "sha256:d0e377a085a5526d0a5fb00e0dcfbd3de43b46951cbcf126a30af8b5f483f8ad"
assert verified(snapshot) is snapshot
view = SnapshotView.from_json(snapshot)
assert view.snapshot_id == snapshot["snapshot_id"]
assert dumps(canonical(snapshot["function_calls"]))
print(outside_the_standard_library())
"""


def _child(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=PACKAGE,
        env={**os.environ, "PYTHONPATH": ""},
        timeout=120,
    )


def test_the_snapshot_module_imports_and_runs_with_only_the_standard_library():
    result = _child(BLOCK_PREAMBLE + RUN_THE_SNAPSHOT_CODEC)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"loaded {result.stdout.strip()}"
