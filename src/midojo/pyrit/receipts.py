"""Keep a campaign's MiDojo evidence after its process exits: receipts, grades and sealed snapshots.

A target's receipts and its control plane's snapshots live in memory, and
PyRIT's memory holds only the metadata copies. :func:`write_receipts` writes
them to a directory :func:`prepare_output_dir` made for one run:

- ``summary.json``: the run summary the caller gives (see :func:`attack_summary`)
  and the counts of sends, replays and snapshots written;
- ``receipts.json``: one record per target send, in send order, with its
  replay's record when the scorer confirmed a success;
- ``snapshots/<evaluation_id>.json``: each sealed snapshot, as
  ``GET /runs/{run_id}/evaluations/{eval_id}/snapshot`` returns it, so
  ``client.snapshot_from_response`` rebuilds it and checks its digest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pyrit.models.results.attack_result import AttackResult

from midojo.control_plane.routers.runs import snapshot_response
from midojo.control_plane.service import ControlPlane
from midojo.pyrit.target import MidojoEpisodeTarget, MidojoVerdict

SUMMARY_FILE = "summary.json"
RECEIPTS_FILE = "receipts.json"
SNAPSHOTS_DIR = "snapshots"


def prepare_output_dir(path: Path) -> Path:
    """Create ``path`` for one run's evidence; refuse one that exists and isn't an empty directory."""
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"{path} already exists and is not an empty directory: give a new output directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


def attack_summary(
    result: AttackResult | None, *, objective: str, error: BaseException | None = None, **extra: Any
) -> dict[str, Any]:
    """The run summary: PyRIT's outcome, conversation and objective, or the error that stopped the campaign."""
    summary: dict[str, Any] = {"objective": objective}
    if result is not None:
        summary.update(
            objective=result.objective,
            outcome=result.outcome.value,
            outcome_reason=result.outcome_reason,
            conversation_id=result.conversation_id,
            attack_result_id=result.attack_result_id,
            executed_turns=result.executed_turns,
            execution_time_ms=result.execution_time_ms,
            last_response_id=str(result.last_response.id) if result.last_response is not None else None,
        )
    if error is not None:
        summary["error"] = f"{type(error).__name__}: {error}"
    summary.update(extra)
    return summary


def write_receipts(target: MidojoEpisodeTarget, output_dir: Path, *, summary: Mapping[str, Any]) -> Path:
    """Write the target's receipts, their sealed snapshots and ``summary`` to ``output_dir``; return the summary's path.

    A snapshot that can't be read (an evaluation never sealed) is recorded as
    the record's ``snapshot_error`` rather than stopping the others.
    """
    snapshots = output_dir / SNAPSHOTS_DIR
    records = []
    for response_id, verdict in target.receipts():
        record = {"response_id": response_id, "broke": verdict.broke}
        record.update(_record(verdict, "send", target.control, snapshots))
        record["replay"] = (
            None if verdict.replay is None else _record(verdict.replay, "replay", target.control, snapshots)
        )
        records.append(record)
    _write_json(output_dir / RECEIPTS_FILE, records)
    counts = {
        "sends": len(records),
        "replays": sum(record["replay"] is not None for record in records),
        "snapshots": len(list(snapshots.glob("*.json"))) if snapshots.is_dir() else 0,
    }
    path = output_dir / SUMMARY_FILE
    _write_json(path, {**summary, **counts, "receipts_file": RECEIPTS_FILE, "snapshots_dir": SNAPSHOTS_DIR})
    return path


def _record(
    verdict: MidojoVerdict, kind: Literal["send", "replay"], control: ControlPlane, snapshots: Path
) -> dict[str, Any]:
    grade = verdict.grade
    record: dict[str, Any] = {
        "kind": kind,
        "status": verdict.status,
        "evaluation_id": verdict.evaluation_id,
        "run_id": verdict.run_id,
        "snapshot_id": verdict.snapshot_id,
        "reason": verdict.reason,
        "security": verdict.security,
        "utility": grade.utility if grade is not None else None,
        "utility_reason": grade.utility_reason if grade is not None else None,
        "security_reason": grade.security_reason if grade is not None else None,
        "grade_validity": grade.grade_validity.model_dump(mode="json") if grade is not None else None,
        "grade": json.loads(verdict.grade_json) if verdict.grade_json is not None else None,
        "snapshot_file": None,
    }
    if verdict.evaluation_id is not None:
        try:
            snapshot = control.snapshot(verdict.run_id, verdict.evaluation_id)
        except Exception as error:  # an evaluation never sealed has no snapshot; keep the other records
            record["snapshot_error"] = f"{type(error).__name__}: {error}"
        else:
            snapshots.mkdir(exist_ok=True)
            path = snapshots / f"{snapshot.evaluation_id}.json"
            path.write_text(snapshot_response(snapshot).model_dump_json(indent=2) + "\n")
            record["snapshot_file"] = path.relative_to(snapshots.parent).as_posix()
    return record


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
