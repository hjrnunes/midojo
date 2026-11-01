"""MiDojo's snapshot ID is the SDK's digest of the snapshot's JSON form, with a documented fallback.

A snapshot holds only values a JSON document carries back as the types it keeps, so its
``snapshot_id`` is ``midojo_sdk.snapshot.snapshot_digest`` of the JSON form ``GET
.../snapshot`` serves: one implementation names a sealed snapshot. A snapshot holding a
value JSON does not carry back (bytes, a date, a tuple, a set, a non-finite float) keeps
MiDojo's own Python-value encoder; those values name content a snapshot rebuilt from JSON
cannot reproduce anyway (``control_plane.client.snapshot_from_response`` refuses it), and
no consumer can read such a snapshot's id back.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from midojo_sdk import snapshot as sdk
from pydantic import Field

from midojo.control_plane.client import snapshot_from_response
from midojo.control_plane.models import SnapshotResponse
from midojo.control_plane.routers.runs import snapshot_response
from midojo.control_plane.state import Snapshot, canonical_json, snapshot_digest
from midojo.types import Environment, FunctionCallRecord
from tests.snapshot_invariant import assert_named_by_the_sdk_digest

VECTORS = json.loads((Path(__file__).resolve().parent / "fixtures/snapshot_digest/vectors.json").read_text())
ACCEPTED = [vector for vector in VECTORS if "refusal" not in vector]


class Lab(Environment):
    """The environment the recorded vectors were built with: whatever JSON a vector holds."""

    data: dict[str, Any] = Field(default_factory=dict)


def build(*, environment: dict[str, Any] | None = None, agent_output: str = "done") -> Snapshot:
    values = environment if environment is not None else {"city": "Zürich"}
    return Snapshot(
        evaluation_id="e1",
        run_id=None,
        user_task_id=None,
        injection_task_id=None,
        agent_output=agent_output,
        pre_environment=Lab(data=values),
        environment=Lab(data=values),
        function_calls=[
            FunctionCallRecord(
                call_id="c1",
                function="lookup_order",
                args={"order_id": "ORD-101"},
                result='{"item":"tea"}',
                disposition="forwarded",
                provenance="interceptor",
                timestamp="2026-10-29T10:00:00+00:00",
                turn=0,
            )
        ],
        turns=(),
        observations={},
        evidence_status={"function_calls": "complete"},
        sealed_at="2026-10-29T10:02:41+00:00",
    )


def served(snapshot: Snapshot) -> dict[str, Any]:
    return snapshot_response(snapshot).model_dump(mode="json")


def test_a_snapshots_id_is_the_sdk_digest_of_the_json_form_the_control_plane_serves():
    snapshot = build()

    assert snapshot.snapshot_id == sdk.snapshot_digest(served(snapshot))
    assert snapshot.snapshot_id == snapshot_digest(snapshot) == snapshot.snapshot_digest()
    assert sdk.verified(served(snapshot)) == served(snapshot)


@pytest.mark.parametrize("entry", ACCEPTED, ids=[entry["name"] for entry in ACCEPTED])
def test_the_sdk_reproduces_every_recorded_vector_through_midojo(entry):
    # Each vector's environment is the one it was built with: the plain Environment the cm.58 and OpenShell
    # snapshots were recorded with, or the vectors' own Lab.
    environment_type = Environment if entry["snapshot"]["pre_environment"] == {} else Lab

    rebuilt = snapshot_from_response(SnapshotResponse.model_validate(entry["snapshot"]), environment_type)

    assert rebuilt.snapshot_id == entry["snapshot_id"]
    assert rebuilt.snapshot_id == sdk.snapshot_digest(served(rebuilt))
    assert rebuilt.snapshot_digest() == entry["snapshot_id"]


def test_midojos_canonical_encoder_is_the_sdks_for_every_json_value():
    for value in (None, True, False, 0, -1, 2**70, 0.0, -0.0, 0.1, "x", "", [1, [2]], {"a": [None]}, {"b": 1, "a": 2}):
        assert canonical_json(value) == sdk.dumps(sdk.canonical(value))


@pytest.mark.parametrize(
    "value",
    [
        b"x",
        bytearray(b"x"),
        date(2026, 10, 29),
        datetime(2026, 10, 29, 10, 2, tzinfo=UTC),
        (1, 2),
        {1, 2},
        float("nan"),
    ],
    ids=["bytes", "bytearray", "date", "datetime", "tuple", "set", "nan"],
)
def test_a_snapshot_holding_a_value_json_does_not_carry_keeps_midojos_python_encoder(value):
    # The documented fallback: JSON loses the type, so the SDK digest of the JSON form would rename the snapshot.
    snapshot = build(environment={"blob": value})

    assert snapshot.snapshot_id == snapshot_digest(snapshot) == snapshot.snapshot_digest()
    assert snapshot.snapshot_id != sdk.snapshot_digest(served(snapshot))
    # Its content is what its id names: a rebuild from the same values keeps it.
    assert snapshot.verified().snapshot_id == snapshot.snapshot_id


def test_a_changed_snapshot_fails_verification_with_a_digest_fault():
    snapshot = build()
    object.__setattr__(snapshot, "agent_output", "done.")

    with pytest.raises(sdk.DigestFault, match="was changed after it was made"):
        snapshot.verified()


def test_a_changed_call_value_fails_verification_with_a_digest_fault():
    snapshot = build()
    call = snapshot._function_calls[0]
    call.result = '{"item":"coffee"}'

    with pytest.raises(sdk.DigestFault, match="was changed after it was made"):
        snapshot.verified()


def test_a_snapshot_without_its_id_fails_verification_with_a_digest_fault():
    snapshot = build()
    object.__delattr__(snapshot, "snapshot_id")

    with pytest.raises(sdk.DigestFault, match="was changed after it was made"):
        snapshot.verified()


def test_a_snapshot_named_by_another_id_fails_verification_with_a_digest_fault():
    snapshot = build()
    object.__setattr__(snapshot, "snapshot_id", "sha256:" + "0" * 64)

    with pytest.raises(sdk.DigestFault, match="was changed after it was made"):
        snapshot.verified()


def test_an_untouched_snapshot_verifies_to_a_new_snapshot_of_its_content():
    snapshot = build()
    held = snapshot.verified()

    assert held is not snapshot and held == snapshot
    assert held.snapshot_id == snapshot.snapshot_id
    assert [call.call_id for call in held.function_calls] == ["c1"]


def test_the_check_the_suite_installs_passes_an_untouched_snapshot_and_catches_a_renamed_one():
    snapshot = build()
    assert_named_by_the_sdk_digest(snapshot)
    object.__setattr__(snapshot, "snapshot_id", "sha256:" + "1" * 64)

    with pytest.raises(AssertionError, match="is not the SDK digest of the JSON served for it"):
        assert_named_by_the_sdk_digest(snapshot)


def test_the_suite_wide_check_leaves_the_python_value_fallback_alone():
    assert_named_by_the_sdk_digest(build(environment={"blob": b"x"}))
