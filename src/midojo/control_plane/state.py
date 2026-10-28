"""Run and evaluation values. Application handles live on FastAPI app.state.

An :class:`Evaluation` is a value: a frozen dataclass whose sequences are
tuples. The lifecycle rules (:mod:`midojo.control_plane.lifecycle`) never change
one; they build the next value, and the writer
(:mod:`midojo.control_plane.service`) replaces the stored value with it. The
pydantic models inside (environments, call records, observations) are mutable
types, so the writer copies them in and out: no caller ever holds an object the
store keeps.
"""

from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import json
import struct
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta, timezone
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel
from pydantic_core import TzInfo

from midojo.episode import MAX_NESTING_DEPTH, Episode
from midojo.observations import FUNCTION_CALLS, EvidenceStatus
from midojo.types import Environment, FunctionCallRecord, InterceptionMode, SuiteName

from .models import CapturePhase, GradeResponse


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Stage(enum.StrEnum):
    """Where an evaluation is in its lifecycle.

    ``open``: everything is admitted. ``closing``: no new call is admitted, and
    calls already admitted may still finish, until they all have or the closure
    wait runs out. ``collecting``: the final observations are recorded; no call
    finishes any more. ``sealed``: the snapshot is taken, and nothing changes it.
    """

    OPEN = "open"
    CLOSING = "closing"
    COLLECTING = "collecting"
    SEALED = "sealed"


@dataclass(frozen=True)
class Turn:
    """One user message in an evaluation's conversation and the agent's reply to it."""

    index: int
    message: str
    reply: str | None = None
    started_at: str = field(default_factory=_now)
    ended_at: str | None = None


@dataclass(frozen=True)
class Session:
    """The evaluation's live session: its token's hash and when it expires (epoch seconds)."""

    token_hash: str
    expires_at: float


@dataclass(frozen=True)
class LateFact:
    """A write that closure refused, kept so the evaluation shows its evidence is incomplete.

    ``evidence`` names the evidence-status keys it marks ``incomplete``
    (``function_calls``, ``environment``, ``turns``, ``agent_output``, or an
    observation source's ``<source>.<part>``). ``payload`` is the write as sent,
    as JSON-compatible data (bytes as UTF-8 text, undecodable bytes escaped).
    """

    event: str
    stage: Stage
    by: str
    at: str
    evidence: tuple[str, ...]
    payload: Mapping[str, Any]
    call_id: str | None = None


@dataclass(frozen=True)
class CaptureFailure:
    """A call record an interception layer couldn't deliver: the recorded calls may lack that call.

    ``phase`` is the record that was lost: ``start`` (the call may be missing
    altogether) or ``finish`` (its record may be unfinished or missing).
    ``reason`` is the last error the layer got. A capture failure marks
    ``function_calls`` ``incomplete``; it never adds, changes or removes a
    record.
    """

    call_id: str
    function: str
    phase: CapturePhase
    reason: str
    stage: Stage
    by: str
    at: str


@dataclass(frozen=True, init=False)
class Snapshot:
    """What the seal fixes: everything grading reads, and how completely it was collected. Immutable.

    ``evidence_status`` is the observations' own status, with every key a late
    fact recorded before the seal marked ``incomplete``, ``function_calls``
    marked ``incomplete`` when a capture failure was reported before the seal,
    and every key the host named as incomplete at the seal.

    ``evaluation_id`` and ``run_id`` name the evaluation it seals (``run_id``
    None outside a run), and ``snapshot_id`` this snapshot: a result graded
    from it names what it graded, never "the latest evaluation".
    ``snapshot_id`` is the snapshot's own content digest (:func:`snapshot_digest`),
    computed here: no caller chooses it, and equal content has an equal ID.

    Nothing a holder does changes a snapshot. Its scalars and its turns and
    capture failures are immutable, ``evidence_status`` is a read-only mapping,
    and the environments, call records and observations (mutable pydantic
    models) are kept as private copies: each read returns a new deep copy, so
    changing what a read returned changes nothing here. Copying a snapshot
    returns the snapshot itself. The private ``_``-prefixed copies are not
    part of that contract: they are shared, and only the snapshot reads them.
    A holder that changes one anyway (or sets a field with
    ``object.__setattr__``) leaves content that ``snapshot_id`` no longer
    names, and grading refuses it (:meth:`verified`).
    """

    evaluation_id: str
    run_id: str | None
    snapshot_id: str
    user_task_id: str | None
    injection_task_id: str | None
    agent_output: str
    turns: tuple[Turn, ...]
    evidence_status: Mapping[str, EvidenceStatus]
    sealed_at: str
    capture_failures: tuple[CaptureFailure, ...]
    _pre_environment: Environment
    _environment: Environment
    _function_calls: tuple[FunctionCallRecord, ...]
    _observations: Mapping[str, BaseModel]

    def __init__(
        self,
        *,
        evaluation_id: str,
        run_id: str | None,
        user_task_id: str | None,
        injection_task_id: str | None,
        agent_output: str,
        pre_environment: Environment,
        environment: Environment,
        function_calls: Iterable[FunctionCallRecord],
        turns: Iterable[Turn],
        observations: Mapping[str, BaseModel],
        evidence_status: Mapping[str, EvidenceStatus],
        sealed_at: str,
        capture_failures: Iterable[CaptureFailure] = (),
    ) -> None:
        values: dict[str, Any] = {
            "evaluation_id": evaluation_id,
            "run_id": run_id,
            "user_task_id": user_task_id,
            "injection_task_id": injection_task_id,
            "agent_output": agent_output,
            "turns": tuple(turns),
            "evidence_status": MappingProxyType(dict(evidence_status)),
            "sealed_at": sealed_at,
            "capture_failures": tuple(capture_failures),
            "_pre_environment": pre_environment.model_copy(deep=True),
            "_environment": environment.model_copy(deep=True),
            "_function_calls": tuple(call.model_copy(deep=True) for call in function_calls),
            "_observations": MappingProxyType(
                {source: observed.model_copy(deep=True) for source, observed in observations.items()}
            ),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "snapshot_id", snapshot_digest(self))

    @property
    def pre_environment(self) -> Environment:
        """The environment when the evaluation was created (a new copy)."""
        return self._pre_environment.model_copy(deep=True)

    @property
    def environment(self) -> Environment:
        """The environment at the seal (a new copy)."""
        return self._environment.model_copy(deep=True)

    @property
    def function_calls(self) -> tuple[FunctionCallRecord, ...]:
        """The call records, in the order they were started (new copies)."""
        return tuple(call.model_copy(deep=True) for call in self._function_calls)

    @property
    def observations(self) -> Mapping[str, BaseModel]:
        """The runtime observations by source (new copies)."""
        return {source: observed.model_copy(deep=True) for source, observed in self._observations.items()}

    def verified(self) -> Snapshot:
        """A new snapshot of this one's content, shared with no one, once that content is what ``snapshot_id`` names.

        The content is read once, through the copying reads, and digested
        again: a snapshot whose content no longer has its ID (one of its
        private copies, or a field, was changed after it was made) raises
        ValueError, and so does one whose content can't be read back at all
        (a field deleted, or set to a value of another type). What is
        returned is checked, and nothing else holds it.
        """
        changed = (
            f"Snapshot {getattr(self, 'snapshot_id', None)} of evaluation {getattr(self, 'evaluation_id', None)} "
            "was changed after it was made"
        )
        try:
            turns, capture_failures = copy.deepcopy((self.turns, self.capture_failures))
            held = Snapshot(
                evaluation_id=self.evaluation_id,
                run_id=self.run_id,
                user_task_id=self.user_task_id,
                injection_task_id=self.injection_task_id,
                agent_output=self.agent_output,
                pre_environment=self.pre_environment,
                environment=self.environment,
                function_calls=self.function_calls,
                turns=turns,
                observations=self.observations,
                evidence_status=self.evidence_status,
                sealed_at=self.sealed_at,
                capture_failures=capture_failures,
            )
        except Exception as error:
            raise ValueError(f"{changed}: its content can't be read back ({type(error).__name__}: {error})") from error
        if held.snapshot_id != self.snapshot_id:
            raise ValueError(f"{changed}: its content differs from its snapshot_id")
        return held

    def __hash__(self) -> int:
        return hash(self.snapshot_id)

    def __copy__(self) -> Snapshot:
        return self

    def __deepcopy__(self, _memo: dict[int, Any]) -> Snapshot:
        return self


def snapshot_digest(snapshot: Snapshot) -> str:
    """``sha256:<hex>`` of the snapshot's content, everything in it but its ID, in a lossless encoding.

    Two snapshots share a digest only when their content is equal. The
    encoding takes only the values :func:`unsupported_value` accepts, which is
    all an evaluation holds: the control plane refuses any other where it
    enters (a call's args, an environment, observations). An injection
    outcome's ``payload_digest``, ``placed_digest`` and ``slot`` are encoded
    only when set (:func:`_call_content`), so a snapshot without them has the
    digest it had before they existed.
    """
    content = {
        "evaluation_id": snapshot.evaluation_id,
        "run_id": snapshot.run_id,
        "user_task_id": snapshot.user_task_id,
        "injection_task_id": snapshot.injection_task_id,
        "agent_output": snapshot.agent_output,
        "pre_environment": snapshot._pre_environment.model_dump(),
        "environment": snapshot._environment.model_dump(),
        "function_calls": [_call_content(call) for call in snapshot._function_calls],
        "turns": [dataclasses.asdict(turn) for turn in snapshot.turns],
        "observations": {source: observed.model_dump() for source, observed in snapshot._observations.items()},
        "evidence_status": dict(snapshot.evidence_status),
        "sealed_at": snapshot.sealed_at,
        "capture_failures": [
            {**dataclasses.asdict(failure), "stage": failure.stage.value} for failure in snapshot.capture_failures
        ],
    }
    return f"sha256:{hashlib.sha256(_dumps(_canonical(content)).encode()).hexdigest()}"


# Fields an injection outcome gained after snapshots were first digested: each is encoded only when set.
_LATER_OUTCOME_FIELDS = ("payload_digest", "placed_digest", "slot")


def _call_content(call: FunctionCallRecord) -> dict[str, Any]:
    """A call record as the digest encodes it: its dump, an outcome's later fields left out while None."""
    content = call.model_dump()
    for outcome in content["injection_outcomes"] or ():
        for name in _LATER_OUTCOME_FIELDS:
            if outcome[name] is None:
                del outcome[name]
    return content


def unsupported_value(value: Any) -> str | None:
    """Where ``value`` holds a value the snapshot digest can't encode, and of which type; None when it has none.

    The digest encodes exactly these: None, ``bool``, ``int``, ``float``,
    ``str``, ``bytes`` and ``bytearray``; ``list``, ``tuple``, ``set`` and
    ``frozenset`` of them, and ``dict`` with keys and values of them; ``date``, ``datetime``, ``time`` and
    ``timedelta``, with no time zone or a fixed offset (``datetime.timezone``,
    or pydantic's parsed ``TzInfo``). Exactly these types, not their
    subclasses: anything else could be unequal and still encode alike, or
    fail to encode at all. A read-only mapping view isn't one either: it
    can't be copied, and the control plane copies what it keeps. Nor is a
    list, tuple, dict or set that contains itself, at any depth: it has no
    finite encoding. Nor are containers nested more than
    :data:`~midojo.episode.MAX_NESTING_DEPTH` deep (``value`` itself, when a
    container, is the first level): the check stops there, so it never hits
    the recursion limit, and what it accepts can be copied and encoded.
    """
    try:
        _canonical(value, limit=MAX_NESTING_DEPTH)
    except _UnsupportedValueError as unsupported:
        return str(unsupported)
    return None


def canonical_json(value: Any) -> str:
    """``value`` in the snapshot digest's lossless encoding, as compact JSON: unequal values encode differently.

    Takes only what :func:`unsupported_value` accepts; anything else raises TypeError.
    """
    return _dumps(_canonical(value))


class _UnsupportedValueError(TypeError):
    def __init__(self, path: str, value: Any, why: str = "") -> None:
        kind = f"{type(value).__module__}.{type(value).__qualname__}"
        super().__init__(f"{path or 'the value'} is a {kind}{f' {why}' if why else ''}")


_FIXED_OFFSETS = (timezone, TzInfo)


_CONTAINERS = (list, tuple, dict, set, frozenset)


def _canonical(value: Any, path: str = "", within: frozenset[int] = frozenset(), limit: int | None = None) -> Any:
    """A JSON value that encodes ``value`` losslessly: unequal values encode differently. Raises ``_UnsupportedValueError``.

    Exactly ``str``, ``int`` and ``bool``, and None, are themselves; any other
    value is a ``[tag, content]`` pair, so no value is mistaken for another:
    a float by its IEEE-754 bits (so every NaN payload and sign too), bytes
    as hex, a list or a tuple by its items, a set or a frozenset by its items
    sorted by their encoding (each tagged with its own type), a mapping by
    its ``[key, value]`` items sorted by their encoding, a date, datetime or
    time in ISO format, and a timedelta as days, seconds and microseconds.
    ``within`` holds the containers ``value`` is inside of: a container that
    contains itself has no finite encoding. With a ``limit``, a container
    inside ``limit`` others is refused too.
    """
    kind = type(value)
    if value is None or kind in (str, int, bool):
        return value
    if kind is float:
        return ["float", struct.pack(">d", value).hex()]
    if kind in (bytes, bytearray):
        return [kind.__name__, value.hex()]
    if kind in _CONTAINERS:
        if id(value) in within:
            raise _UnsupportedValueError(path, value, "that contains itself")
        if limit is not None and len(within) >= limit:
            raise _UnsupportedValueError(path, value, f"nested more than {limit} deep")
        within = within | {id(value)}
    if kind in (list, tuple):
        return [
            kind.__name__,
            [_canonical(item, f"{path}[{index}]", within, limit) for index, item in enumerate(value)],
        ]
    if kind is dict:
        items = []
        for index, (key, item) in enumerate(value.items()):
            encoded = _canonical(key, f"{path}<key {index}>", within, limit)
            items.append([encoded, _canonical(item, f"{path}[{_dumps(encoded)}]", within, limit)])
        return ["mapping", sorted(items, key=_dumps)]
    if kind in (set, frozenset):
        return [kind.__name__, sorted((_canonical(item, f"{path}<item>", within, limit) for item in value), key=_dumps)]
    if kind in (datetime, time) and (value.tzinfo is None or type(value.tzinfo) in _FIXED_OFFSETS):
        return [kind.__name__, value.isoformat()]
    if kind is date:
        return ["date", value.isoformat()]
    if kind is timedelta:
        return ["timedelta", [value.days, value.seconds, value.microseconds]]
    raise _UnsupportedValueError(path, value)


def _dumps(encoded: Any) -> str:
    return json.dumps(encoded, separators=(",", ":"))


@dataclass(frozen=True)
class Evaluation:
    """One run of an episode: its records while it's open, then its sealed snapshot and grade.

    ``run_id``, ``suite_name`` and the task IDs are provenance: set when the
    episode came from a suite pair in a run, None for a direct caller's episode.
    ``episode`` is what was run; grading reads its checks.
    """

    id: str
    run_id: str | None
    user_task_id: str | None
    injection_task_id: str | None
    pre_environment: Environment
    environment: Environment
    function_calls: tuple[FunctionCallRecord, ...] = ()
    # Turns in the order they were started; function calls name the latest one.
    turns: tuple[Turn, ...] = ()
    # Runtime observations keyed by source (e.g. "openshell" -> OpenShellObservations),
    # recorded by the orchestrator and read by verifiers at grade time.
    observations: Mapping[str, BaseModel] = field(default_factory=dict)
    agent_input: str | None = None
    created_at: str = field(default_factory=_now)
    active_injections: Mapping[str, str] = field(default_factory=dict)
    session: Session | None = None
    stage: Stage = Stage.OPEN
    # When the closure wait ends (epoch seconds), once closure has begun.
    closure_deadline: float | None = None
    snapshot: Snapshot | None = None
    late_facts: tuple[LateFact, ...] = ()
    # Call records the interception layer reported it couldn't deliver, in arrival order.
    capture_failures: tuple[CaptureFailure, ...] = ()
    # Set once, from the snapshot, and never replaced.
    grade: GradeResponse | None = None
    # Read-only: copying the evaluation shares it (see Episode.__deepcopy__).
    episode: Episode | None = None
    suite_name: SuiteName | None = None
    # How many times the environment has changed since the evaluation was created.
    environment_revision: int = 0
    # Whether call records carry environment snapshots: the environment observed when each call started and
    # finished, which shows what changed during the call, not what the call caused.
    environment_history: bool = False
    # Where the tool calls are intercepted, chosen by the host at creation: it sets the provenance of the
    # records the session makes, and whether a capture without records is complete (see lifecycle._seal).
    interception: InterceptionMode = "agent"

    @property
    def completed(self) -> bool:
        return self.stage is Stage.SEALED

    @property
    def graded(self) -> bool:
        return self.grade is not None

    @property
    def environment_type(self) -> type[Environment]:
        """The model a replacement environment is validated against: the episode's."""
        if self.episode is not None and self.episode.environment_type is not None:
            return self.episode.environment_type
        return type(self.environment)

    @property
    def agent_output(self) -> str | None:
        return self.snapshot.agent_output if self.snapshot is not None else None

    @property
    def evidence_status(self) -> dict[str, EvidenceStatus]:
        """The grade's evidence status, with every key a late fact or a capture failure marks ``incomplete``.

        Empty before grading unless late facts or capture failures were
        recorded. The grade itself never changes: a fact that arrives after the
        seal shows up here only.
        """
        status: dict[str, EvidenceStatus] = dict(self.grade.evidence_status) if self.grade is not None else {}
        status.update(late_evidence(self.late_facts))
        status.update(capture_evidence(self.capture_failures))
        return status


def late_evidence(facts: tuple[LateFact, ...]) -> dict[str, EvidenceStatus]:
    """The evidence-status keys ``facts`` mark incomplete."""
    return {key: "incomplete" for fact in facts for key in fact.evidence}


def capture_evidence(failures: tuple[CaptureFailure, ...]) -> dict[str, EvidenceStatus]:
    """The evidence-status keys ``failures`` mark incomplete: ``function_calls``, once any was reported."""
    return {FUNCTION_CALLS: "incomplete"} if failures else {}


@dataclass(frozen=True)
class Creation:
    """An evaluation created under an idempotency key: the request it answered, and the evaluation it opened.

    ``request`` is what the request asked for (``user_task_id``,
    ``injection_task_id``, ``injections`` and ``interception``);
    ``evaluation_id`` and ``turns`` are what the response gave. The session
    token isn't kept: a resend gets a fresh one (the store keeps only token
    hashes).
    """

    request: Mapping[str, Any]
    evaluation_id: str
    turns: tuple[str, ...]


@dataclass(frozen=True)
class Run:
    id: str
    suite_name: SuiteName
    created_at: str = field(default_factory=_now)
