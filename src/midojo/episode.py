"""Episodes: the description of what to run, apart from any record of running it.

An :class:`Episode` is a frozen value: the turns to send, the active injections,
the checks that grade it, the initial state of the simulated tools, and the
runtime it needs, as a definition (never a live runtime object). An evaluation
is the record of one run of an episode.

An episode holds only frozen data: read-only mappings, tuples and immutable
scalars. Everything live is built fresh from that data each time it's needed:
the checks when grading, the environment when an evaluation is provisioned,
and the runtime definition and the runtime itself. So nothing a caller gets
from an episode, or from a copy of one, can change what it describes.

A suite compiles one of its user task x injection task pairs into an episode
(:meth:`midojo.yaml_task_suite.YAMLTaskSuite.episode`); a direct caller builds
one from these parts and runs it through the control plane
(:meth:`midojo.control_plane.service.ControlPlane.open_evaluation`) with no
suite or run registered.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

from pydantic import BaseModel, TypeAdapter

from midojo.env_inference import infer_environment_type
from midojo.observations import evidence_status
from midojo.runtimes import AgentRuntime, build_runtime
from midojo.runtimes.openshell import OBSERVATIONS_SOURCE, OpenShellObservations
from midojo.suite_definition import AgentRuntimeDefinition
from midojo.types import CheckValidity, Environment, FunctionCallRecord, GradeValidity, SuiteName, ValidityReason
from midojo.verifier import Check, VerificationContext, VerificationResult, parse_check
from midojo.verifiers import ConversationTurn
from midojo.verifiers.openshell import reads_openshell_observations

# The name a runtime is built under when the episode has no suite.
DEFAULT_EPISODE_NAME = "episode"

_RUNTIME_DEFINITION: TypeAdapter[AgentRuntimeDefinition] = TypeAdapter(AgentRuntimeDefinition)
# Exactly these types, not their subclasses: a subclass can carry mutable state.
_IMMUTABLE_SCALARS = frozenset(
    {str, bytes, int, float, bool, type(None), datetime.date, datetime.datetime, datetime.time, datetime.timedelta}
)


class UnknownTaskError(KeyError):
    """A suite has no task with the requested ID. ``detail`` says which."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def freeze(value: Any) -> Any:
    """A deep, read-only copy of JSON-like data: mappings become read-only views, lists and tuples tuples.

    Mapping keys must be exactly ``str``, and every other value a mapping, a
    list, a tuple or exactly one of the immutable built-in scalars (``str``,
    ``bytes``, ``int``, ``float``, ``bool``, None, ``date``, ``datetime``,
    ``time``, ``timedelta``; a date or time only with no time zone or a
    ``datetime.timezone``). Anything else, a subclass of these included, raises
    TypeError: nothing is converted.
    """
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            frozen[_exact_str(key, "a key in episode data")] = freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(freeze(item) for item in value)
    if type(value) not in _IMMUTABLE_SCALARS:
        raise TypeError(f"episode data must be JSON-like, not {type(value).__name__}")
    tzinfo = getattr(value, "tzinfo", None)
    if tzinfo is not None and type(tzinfo) is not datetime.timezone:
        raise TypeError(f"a date or time in episode data needs a datetime.timezone, not {type(tzinfo).__name__}")
    return value


def _exact_str(value: Any, what: str) -> str:
    """``value`` when it is exactly a ``str``; a subclass could carry mutable state, so it raises TypeError."""
    if type(value) is not str:
        raise TypeError(f"{what} must be a str, not {type(value).__name__}")
    return value


def thaw(value: Any) -> Any:
    """Plain, mutable data (dicts and lists) from :func:`freeze`'s output: a new copy each call."""
    if isinstance(value, Mapping):
        return {key: thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw(item) for item in value]
    return value


def _runtime_data(runtime: AgentRuntimeDefinition | Mapping[str, Any]) -> Mapping[str, Any]:
    """A runtime definition, validated, as frozen data. Fields given as a mapping must be episode data already."""
    given = runtime if isinstance(runtime, BaseModel) else thaw(freeze(runtime))
    definition = _RUNTIME_DEFINITION.validate_python(given)
    return freeze(definition.model_dump())


@dataclass(frozen=True)
class Episode:
    """What to run: immutable once built.

    ``turns`` are the user's messages, sent one at a time in one evaluation, or
    None when the host drives the conversation. ``injections`` maps each active
    probe (``"<task>:<probe>"``) to its payload. ``security`` and ``utility``
    are the checks as authored (a suite's ``security:``/``utility:`` blocks,
    e.g. ``{"output_contains": "pwned"}``); either may be absent. Each grading
    parses a fresh :class:`~midojo.verifier.Check` from them. ``environment``
    is the initial state of the simulated tools, validated with
    ``environment_type`` (inferred from the state when not given) each time an
    evaluation is provisioned. ``runtime`` is the runtime the episode needs: an
    ``agent_runtime`` definition, given as the model or as its fields, kept as
    its fields; :meth:`runtime_definition` and :meth:`new_runtime` build fresh
    objects from it.

    ``suite_name``, ``user_task_id`` and ``injection_task_id`` are provenance:
    set when a suite compiled the episode, and None otherwise.

    Every input is copied into frozen data at construction (see :func:`freeze`),
    so changing it afterwards changes nothing here, and nothing reachable from
    the episode can be changed. Turns, injection probes and payloads, and the
    provenance IDs must be exactly ``str``; anything else raises TypeError.
    ``environment_type``, like the verifier registry the checks resolve
    against, is held by reference and treated as code, not data.
    """

    runtime: AgentRuntimeDefinition | Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: MappingProxyType({"type": "unmanaged"})
    )
    turns: Sequence[str] | None = None
    injections: Mapping[str, str] = dataclasses.field(default_factory=dict)
    security: Mapping[str, Any] | None = None
    utility: Mapping[str, Any] | None = None
    environment: Mapping[str, Any] | None = None
    environment_type: type[Environment] | None = None
    suite_name: SuiteName | None = None
    user_task_id: str | None = None
    injection_task_id: str | None = None

    def __post_init__(self) -> None:
        if self.turns is not None:
            if isinstance(self.turns, str):
                raise TypeError("turns must be a sequence of messages, not a string")
            if not self.turns:
                raise ValueError("turns must hold at least one message, or be None when the host drives them")
            turns = tuple(_exact_str(message, "a turn") for message in self.turns)
            object.__setattr__(self, "turns", turns)
        injections = freeze(dict(self.injections))
        for payload in injections.values():
            _exact_str(payload, "an injection payload")
        object.__setattr__(self, "injections", injections)
        for name in ("suite_name", "user_task_id", "injection_task_id"):
            if (value := getattr(self, name)) is not None:
                _exact_str(value, name)
        object.__setattr__(self, "runtime", _runtime_data(self.runtime))
        for name in ("security", "utility"):
            spec = getattr(self, name)
            if spec is not None:
                object.__setattr__(self, name, freeze(spec))
                self._check(name)  # Refuse a check that doesn't parse, or doesn't fit the runtime, now.
        environment = None if self.environment is None else freeze(self.environment)
        object.__setattr__(self, "environment", environment)
        if self.environment_type is None:
            name = self.suite_name or DEFAULT_EPISODE_NAME
            object.__setattr__(self, "environment_type", infer_environment_type(name, thaw(environment or {})))

    def __deepcopy__(self, _memo: dict[int, Any]) -> Episode:
        # Nothing reachable from an episode can change, so a copy is the episode itself.
        return self

    def runtime_definition(self) -> AgentRuntimeDefinition:
        """A fresh copy of the runtime definition the episode needs."""
        return _RUNTIME_DEFINITION.validate_python(thaw(self.runtime))

    def new_runtime(self) -> AgentRuntime:
        """A fresh runtime for one run of this episode: it shares no state with any other."""
        return build_runtime(self.suite_name or DEFAULT_EPISODE_NAME, self.runtime_definition())

    def provision_environment(self) -> Environment:
        """A fresh environment from the initial state; the episode keeps none of it."""
        assert self.environment_type is not None  # __post_init__ infers one.
        return self.environment_type.model_validate(thaw(self.environment or {}))

    def _on_openshell(self) -> bool:
        return cast(Mapping[str, Any], self.runtime)["type"] == "openshell"

    def _check(self, name: str) -> Check | None:
        """A fresh check parsed from the ``security`` or ``utility`` spec, or None without one."""
        spec = getattr(self, name)
        if spec is None:
            return None
        check = parse_check(thaw(spec))
        if not self._on_openshell() and reads_openshell_observations(check.parsed):
            raise ValueError(f"The {name} check uses OpenShell predicates, which require the openshell agent_runtime")
        return check

    def grade(
        self,
        agent_output: str,
        pre_environment: Environment,
        post_environment: Environment,
        function_calls: list[FunctionCallRecord],
        observations: Mapping[str, BaseModel] | None = None,
        turns: list[ConversationTurn] | None = None,
    ) -> dict[str, Any]:
        """Grade one run of the episode with checks parsed fresh from its specs.

        ``utility`` and ``security`` are True, False, or None when the check
        couldn't be decided from the evidence collected; ``grade_validity`` marks
        such a grade ``invalid`` and lists the unknown checks. Each grade has a
        reason, whatever its outcome. Without a check, the grade is False, its
        reason None, and its validity ``not_applicable``. ``evidence_status`` is
        how completely each evidence source was collected.
        """
        observations = dict(observations or {})
        if self._on_openshell() and OBSERVATIONS_SOURCE not in observations:
            observations[OBSERVATIONS_SOURCE] = OpenShellObservations.not_collected()
        ctx = VerificationContext(
            agent_output=agent_output,
            pre_environment=pre_environment,
            post_environment=post_environment,
            function_calls=function_calls,
            observations=observations,
            turns=turns or [],
        )
        # One traversal per check grades it and names the criterion that decided it.
        utility, utility_reason, utility_validity = _assess(self._check("utility"), ctx)
        security, security_reason, security_validity = _assess(self._check("security"), ctx)
        return {
            "utility": utility,
            "utility_reason": utility_reason,
            "security": security,
            "security_reason": security_reason,
            "grade_validity": GradeValidity(utility=utility_validity, security=security_validity).model_dump(),
            "evidence_status": evidence_status(observations),
        }


def _assess(check: Check | None, ctx: VerificationContext) -> tuple[bool | None, str | None, CheckValidity]:
    if check is None:
        return False, None, CheckValidity(status="not_applicable")
    result = check.assess(ctx)
    return result.passed, result.reason, _validity(result)


def _validity(result: VerificationResult) -> CheckValidity:
    """A grade is invalid exactly when its top-level check resolved to unknown."""
    if result.passed is not None:
        return CheckValidity(status="valid")
    reasons = list(result.unknowns) or [ValidityReason(check=result.reason, reason="the check returned no verdict")]
    return CheckValidity(status="invalid", reasons=reasons)
