"""In-memory persistence seam: runs, evaluation values and the session token index.

The store decides nothing and changes no evaluation in place. It keeps one
:class:`~midojo.control_plane.state.Evaluation` value per evaluation, and only
the writer (:class:`midojo.control_plane.service.ControlPlane`) replaces it,
with :meth:`Store.commit_evaluation`, while it holds that evaluation's lock.
Every value it hands out is a deep copy, so a caller can't reach what it keeps.

A persistent store can implement the same contract: a commit replaces one
evaluation's value, and the writer serializes commits per evaluation.
"""

from __future__ import annotations

import copy
import secrets
import threading
import uuid
from typing import Protocol, TypeVar

from midojo.types import SuiteName

from .state import Evaluation, Run

T = TypeVar("T")


def detached(value: T) -> T:
    """A deep copy of ``value``: nothing in it is shared with the original."""
    return copy.deepcopy(value)


class Store(Protocol):
    """Interface for run/evaluation persistence."""

    # --- runs ---
    def create_run(self, suite_name: SuiteName) -> Run: ...
    def get_run(self, run_id: str) -> Run | None: ...
    def list_runs(self) -> list[Run]: ...

    # --- evaluations ---
    def new_evaluation_id(self) -> str:
        """An evaluation ID no evaluation in this store has, reserved for the caller."""
        ...

    def insert_evaluation(self, evaluation: Evaluation) -> None:
        """Store a new evaluation under its run. Only the writer calls this."""
        ...

    def get_evaluation(self, run_id: str, eval_id: str) -> Evaluation | None:
        """A copy of the evaluation's current value, or None if (run_id, eval_id) is unknown."""
        ...

    def evaluation_value(self, run_id: str, eval_id: str) -> Evaluation | None:
        """The stored value itself, for the writer only: it never changes it and never hands it out."""
        ...

    def list_evaluations(self, run_id: str) -> list[Evaluation]: ...
    def commit_evaluation(self, evaluation: Evaluation) -> None:
        """Replace the stored value of ``evaluation``. Only the writer calls this, holding the evaluation's lock."""
        ...

    # --- the session token index ---
    def bind_session(self, token_hash: str, run_id: str, eval_id: str) -> None: ...
    def unbind_session(self, token_hash: str) -> None: ...
    def session_target(self, token_hash: str) -> tuple[str, str] | None: ...


class InMemoryStore:
    """Process-local, in-memory :class:`Store`. State is lost on restart.

    Its own maps are guarded by one lock, so threads can call it; ordering the
    writes to one evaluation is the writer's job.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, Run] = {}
        # run_id -> eval_id -> value, in creation order.
        self._evaluations: dict[str, dict[str, Evaluation]] = {}
        self._evaluation_ids: set[str] = set()
        self._sessions: dict[str, tuple[str, str]] = {}

    # --- runs ---

    def create_run(self, suite_name: SuiteName) -> Run:
        run = Run(id=uuid.uuid4().hex, suite_name=suite_name)
        with self._lock:
            self._runs[run.id] = run
            self._evaluations[run.id] = {}
        return run

    def get_run(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[Run]:
        with self._lock:
            return list(self._runs.values())

    # --- evaluations ---

    def new_evaluation_id(self) -> str:
        with self._lock:
            # Keep evaluation IDs short and unique across every run in this store.
            eval_id = secrets.token_hex(5)
            while eval_id in self._evaluation_ids:
                eval_id = secrets.token_hex(5)
            self._evaluation_ids.add(eval_id)
            return eval_id

    def insert_evaluation(self, evaluation: Evaluation) -> None:
        with self._lock:
            evaluations = self._evaluations[evaluation.run_id]
            if evaluation.id in evaluations:
                raise ValueError(f"Evaluation {evaluation.id!r} already exists")
            self._evaluation_ids.add(evaluation.id)
            evaluations[evaluation.id] = detached(evaluation)

    def get_evaluation(self, run_id: str, eval_id: str) -> Evaluation | None:
        with self._lock:
            evaluation = self._evaluations.get(run_id, {}).get(eval_id)
        return detached(evaluation)

    def list_evaluations(self, run_id: str) -> list[Evaluation]:
        with self._lock:
            evaluations = list(self._evaluations.get(run_id, {}).values())
        return detached(evaluations)

    def evaluation_value(self, run_id: str, eval_id: str) -> Evaluation | None:
        with self._lock:
            return self._evaluations.get(run_id, {}).get(eval_id)

    def commit_evaluation(self, evaluation: Evaluation) -> None:
        with self._lock:
            evaluations = self._evaluations[evaluation.run_id]
            if evaluation.id not in evaluations:
                raise KeyError(f"Unknown evaluation: {evaluation.id}")
            evaluations[evaluation.id] = evaluation

    # --- the session token index ---

    def bind_session(self, token_hash: str, run_id: str, eval_id: str) -> None:
        with self._lock:
            self._sessions[token_hash] = (run_id, eval_id)

    def unbind_session(self, token_hash: str) -> None:
        with self._lock:
            self._sessions.pop(token_hash, None)

    def session_target(self, token_hash: str) -> tuple[str, str] | None:
        with self._lock:
            return self._sessions.get(token_hash)
