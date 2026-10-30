"""The SDK's snapshot codec: the digest MiDojo names a sealed snapshot by, and the read view of one.

The vectors in ``tests/fixtures/snapshot_digest/vectors.json`` were recorded from
MiDojo's implementation before the digest moved here (``m90/main`` at 770118e), each with
the JSON form ``GET .../snapshot`` served and the id that implementation computed for it.
The SDK must reproduce every one of them, and refuse the corruptions listed with it.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
from midojo_sdk.snapshot import (
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

ROOT = Path(__file__).resolve().parent.parent
SDK_SOURCE = ROOT / "interception/python-sdk/src/midojo_sdk"
VECTORS_PATH = ROOT / "tests/fixtures/snapshot_digest/vectors.json"

VECTORS = json.loads(VECTORS_PATH.read_text())
ACCEPTED = [vector for vector in VECTORS if "refusal" not in vector]
REFUSED = [vector for vector in VECTORS if "refusal" in vector]
CM58_ID = "sha256:490adb33e785a3862031342b3fee004f08634471bfbbb1fdded08839ce5272d6"


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


def test_the_cm58_snapshot_keeps_the_id_the_consumers_pin():
    assert snapshot_digest(vector("cm58-scn051")["snapshot"]) == CM58_ID


def test_the_vectors_cover_every_required_case():
    covered = {entry["name"] for entry in VECTORS}

    assert {
        "cm58-scn051",
        "openshell-empty-http-requests",
        "consumer-openshell-http-requests-empty",
        "later-outcomes-set",
        "later-outcomes-null",
        "float-edge",
        "non-ascii",
        "refused-non-json-value",
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
    assert refused["type"] == "bytes"
    for path in refused["paths"]:
        holder = value
        for step in path[:-1]:
            holder = holder[step]
        holder[path[-1]] = bytes.fromhex(refused["bytes_hex"])
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
        with pytest.raises(DigestFault):
            verified({**materialized(entry), "snapshot_id": entry["snapshot_id"]})
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


# --- the cm.58 snapshot: verified once, refused for every change a consumer could make ---


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


def test_the_cm58_snapshot_verifies_and_every_change_to_it_is_refused():
    snapshot = vector("cm58-scn051")["snapshot"]

    assert verified(snapshot) is snapshot
    assert SnapshotView.from_json(snapshot).snapshot_id == CM58_ID
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
    [b"x", bytearray(b"x"), (1,), {1}, frozenset({1}), object(), {1: "a"}],
    ids=["bytes", "bytearray", "tuple", "set", "frozenset", "object", "non-text-key"],
)
def test_the_encoding_refuses_a_value_json_does_not_carry_back(value):
    with pytest.raises(DigestFault):
        canonical(value)


def test_the_text_hashed_is_compact_and_ascii_escaped():
    assert dumps({"city": "Zürich"}) == '{"city":"Z\\u00fcrich"}'
    assert dumps([1, 2]) == "[1,2]"


def test_the_snapshot_mappings_key_order_does_not_change_the_id():
    entry = vector("cm58-scn051")

    assert snapshot_digest(dict(reversed(list(entry["snapshot"].items())))) == CM58_ID


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


def test_a_digest_fault_is_a_value_error():
    assert issubclass(DigestFault, ValueError)


# --- the import boundary ---


# Every top-level module that is neither the standard library nor the SDK is unimportable, so nothing the SDK
# imports can hide: midojo, httpx, pydantic, fastapi, grpc, google.protobuf and openshell are all blocked.
BLOCK_PREAMBLE = """
import importlib.abc
import json
import sys
from pathlib import Path

ALLOWED = set(sys.stdlib_module_names) | {"midojo_sdk", "__main__"}


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
from midojo_sdk.snapshot import SnapshotView, canonical, dumps, snapshot_digest, verified

vectors = json.loads(Path("tests/fixtures/snapshot_digest/vectors.json").read_text())
snapshot = next(entry for entry in vectors if entry["name"] == "cm58-scn051")["snapshot"]
assert snapshot_digest(snapshot) == "sha256:490adb33e785a3862031342b3fee004f08634471bfbbb1fdded08839ce5272d6"
assert verified(snapshot) is snapshot
view = SnapshotView.from_json(snapshot)
assert view.snapshot_id == snapshot["snapshot_id"]
assert dumps(canonical(snapshot["function_calls"]))
print(outside_the_standard_library())
"""

IMPORT_THE_SDK = """
import midojo_sdk.injection
import midojo_sdk.snapshot
print(outside_the_standard_library())
"""


def _child(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ""},
        timeout=120,
    )


def test_the_snapshot_module_imports_and_runs_with_only_the_standard_library():
    result = _child(BLOCK_PREAMBLE + RUN_THE_SNAPSHOT_CODEC)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"loaded {result.stdout.strip()}"


def test_the_sdk_never_imports_midojo():
    offenders: list[str] = []
    files = sorted(SDK_SOURCE.rglob("*.py"))
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                offenders += [f"{path.name}: import {alias.name}" for alias in node.names if _is_midojo(alias.name)]
            elif isinstance(node, ast.ImportFrom) and node.module is not None and _is_midojo(node.module):
                offenders.append(f"{path.name}: from {node.module} import")
    assert offenders == [], offenders
    assert len(files) >= 8, files


def _is_midojo(name: str) -> bool:
    return name == "midojo" or name.startswith("midojo.")


def test_the_sdk_modules_import_with_midojo_blocked():
    result = _child(BLOCK_PREAMBLE + IMPORT_THE_SDK)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"loaded {result.stdout.strip()}"
