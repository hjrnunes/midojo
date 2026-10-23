"""In-memory persistence seam: runs, evaluation values and the session token index.

The store decides nothing and changes no evaluation in place. It keeps one
:class:`~midojo.control_plane.state.Evaluation` value per evaluation. No public
method hands out or keeps a stored value: reads return deep copies, and
:meth:`Store.insert_evaluation` keeps a copy of what it's given.

The only way to replace a stored value is the writer capability,
:meth:`Store.writer`, which a store gives out once: the control plane's writer
(:class:`midojo.control_plane.service.ControlPlane`) takes it. Its
:meth:`EvaluationWriter.apply` runs one step per evaluation at a time, under
that evaluation's lock, on a copy of the stored value, and keeps a copy of
what the step returns.

A persistent store can implement the same contract with a transaction per
``apply``.
"""

from __future__ import annotations

import copy
import secrets
import threading
import uuid
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Protocol, TypeVar

from midojo.types import SuiteName

from .state import Evaluation, Run

T = TypeVar("T")

# A writer step: given a copy of the stored value (None when the evaluation is
# unknown), return the next value to keep, or None to keep the stored one.
Step = Callable[[Evaluation | None], Evaluation | None]


def detached(value: T) -> T:
    """A deep copy of ``value``: nothing in it is shared with the original."""
    return copy.deepcopy(value)


class EvaluationWriter(Protocol):
    """The capability to replace stored evaluations, held by the control plane's writer alone."""

    def apply(self, run_id: str | None, eval_id: str, step: Step) -> None:
        """Run ``step`` on a copy of the evaluation's value, under its lock, and keep a copy of what it returns."""
        ...


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
        """Store a copy of a new evaluation under its run, or outside any run when its ``run_id`` is None."""
        ...

    def get_evaluation(self, run_id: str | None, eval_id: str) -> Evaluation | None:
        """A copy of the evaluation's current value, or None if (run_id, eval_id) is unknown."""
        ...

    def list_evaluations(self, run_id: str | None) -> list[Evaluation]: ...

    def writer(self) -> EvaluationWriter:
        """The writer capability. Given out once: a second call raises RuntimeError."""
        ...

    # --- the session token index ---
    def bind_session(self, token_hash: str, run_id: str | None, eval_id: str) -> None: ...
    def unbind_session(self, token_hash: str) -> None: ...
    def session_target(self, token_hash: str) -> tuple[str | None, str] | None: ...


class InMemoryStore:
    """Process-local, in-memory :class:`Store`. State is lost on restart.

    Its own maps are guarded by one lock, so threads can call it; its writer
    orders the steps on each evaluation with a lock per evaluation.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, Run] = {}
        # run_id -> eval_id -> value, in creation order; None holds the evaluations outside any run.
        self._evaluations: dict[str | None, dict[str, Evaluation]] = {None: {}}
        self._evaluation_ids: set[str] = set()
        self._sessions: dict[str, tuple[str | None, str]] = {}
        self._writer: _InMemoryWriter | None = None

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

    def get_evaluation(self, run_id: str | None, eval_id: str) -> Evaluation | None:
        with self._lock:
            evaluation = self._evaluations.get(run_id, {}).get(eval_id)
            return detached(evaluation)

    def list_evaluations(self, run_id: str | None) -> list[Evaluation]:
        with self._lock:
            return detached(list(self._evaluations.get(run_id, {}).values()))

    def writer(self) -> EvaluationWriter:
        with self._lock:
            if self._writer is not None:
                raise RuntimeError("This store's writer has already been given out")
            self._writer = _InMemoryWriter(self)
            return self._writer

    # --- the session token index ---

    def bind_session(self, token_hash: str, run_id: str | None, eval_id: str) -> None:
        with self._lock:
            self._sessions[token_hash] = (run_id, eval_id)

    def unbind_session(self, token_hash: str) -> None:
        with self._lock:
            self._sessions.pop(token_hash, None)

    def session_target(self, token_hash: str) -> tuple[str | None, str] | None:
        with self._lock:
            return self._sessions.get(token_hash)


class _InMemoryWriter:
    """:class:`EvaluationWriter` for an :class:`InMemoryStore`: one lock per evaluation."""

    def __init__(self, store: InMemoryStore) -> None:
        self._store = store
        self._locks: dict[tuple[str | None, str], threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def _lock_for(self, run_id: str | None, eval_id: str) -> AbstractContextManager[object]:
        with self._locks_guard:
            return self._locks.setdefault((run_id, eval_id), threading.Lock())

    def apply(self, run_id: str | None, eval_id: str, step: Step) -> None:
        store = self._store
        with self._lock_for(run_id, eval_id):
            with store._lock:
                current = detached(store._evaluations.get(run_id, {}).get(eval_id))
            following = step(current)
            if following is None:
                return
            if current is None or (following.run_id, following.id) != (run_id, eval_id):
                raise KeyError(f"Unknown evaluation: {eval_id}")
            following = detached(following)
            with store._lock:
                store._evaluations[run_id][eval_id] = following
