"""Episodes: the description of what to run, apart from any record of running it.

An :class:`Episode` is a frozen value: the turns to send, the active injections,
the checks that grade it, the initial state of the simulated tools, and the
runtime it needs, as a definition (never a live runtime object). An evaluation
is the record of one run of an episode.

A suite compiles one of its user task x injection task pairs into an episode
(:meth:`midojo.yaml_task_suite.YAMLTaskSuite.episode`); a direct caller builds
one from these parts and runs it through the control plane
(:meth:`midojo.control_plane.service.ControlPlane.open_evaluation`) with no
suite or run registered.
"""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from midojo.env_inference import infer_environment_type
from midojo.observations import evidence_status
from midojo.runtimes import AgentRuntime, build_runtime
from midojo.runtimes.openshell import OBSERVATIONS_SOURCE, OpenShellObservations
from midojo.suite_definition import AgentRuntimeDefinition, OpenShellRuntimeDefinition, UnmanagedRuntimeDefinition
from midojo.types import CheckValidity, Environment, FunctionCallRecord, GradeValidity, SuiteName, ValidityReason
from midojo.verifier import Check, VerificationContext, VerificationResult
from midojo.verifiers import ConversationTurn

# The name a runtime is built under when the episode has no suite.
DEFAULT_EPISODE_NAME = "episode"


class UnknownTaskError(KeyError):
    """A suite has no task with the requested ID. ``detail`` says which."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


def _freeze(value: Any) -> Any:
    """A deep, read-only copy of JSON-like data: mappings become read-only views, lists tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze(item) for item in value)
    return copy.deepcopy(value)


def _thaw(value: Any) -> Any:
    """Plain, mutable data from :func:`_freeze`'s output."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return copy.deepcopy(value)


@dataclass(frozen=True)
class Episode:
    """What to run: immutable once built.

    ``turns`` are the user's messages, sent one at a time in one evaluation, or
    None when the host drives the conversation. ``injections`` maps each active
    probe (``"<task>:<probe>"``) to its payload. ``security`` and ``utility``
    grade it; either may be absent. ``environment`` is the initial state of the
    simulated tools, as data, validated with ``environment_type`` (inferred from
    the state when not given) each time an evaluation is provisioned.
    ``runtime`` is the runtime the episode needs, as its definition:
    :meth:`new_runtime` builds a fresh one.

    ``suite_name``, ``user_task_id`` and ``injection_task_id`` are provenance:
    set when a suite compiled the episode, and None otherwise.

    Every input is copied at construction, so changing it afterwards changes
    nothing here; mappings are read-only views and sequences tuples.
    """

    runtime: AgentRuntimeDefinition = dataclasses.field(
        default_factory=lambda: UnmanagedRuntimeDefinition(type="unmanaged")
    )
    turns: Sequence[str] | None = None
    injections: Mapping[str, str] = dataclasses.field(default_factory=dict)
    security: Check | None = None
    utility: Check | None = None
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
            object.__setattr__(self, "turns", tuple(self.turns))
        object.__setattr__(self, "injections", MappingProxyType(dict(self.injections)))
        object.__setattr__(self, "runtime", self.runtime.model_copy(deep=True))
        environment = None if self.environment is None else _freeze(self.environment)
        object.__setattr__(self, "environment", environment)
        if self.environment_type is None:
            name = self.suite_name or DEFAULT_EPISODE_NAME
            object.__setattr__(self, "environment_type", infer_environment_type(name, _thaw(environment or {})))

    def __deepcopy__(self, _memo: dict[int, Any]) -> Episode:
        # Everything but the runtime definition is read-only (the checks are never changed, by
        # convention), so a copy shares it; the definition's settings get their own copy.
        clone = copy.copy(self)
        object.__setattr__(clone, "runtime", self.runtime.model_copy(deep=True))
        return clone

    def new_runtime(self) -> AgentRuntime:
        """A fresh runtime for one run of this episode: it shares no state with any other."""
        return build_runtime(self.suite_name or DEFAULT_EPISODE_NAME, self.runtime)

    def provision_environment(self) -> Environment:
        """A fresh environment from the initial state; the episode keeps none of it."""
        assert self.environment_type is not None  # __post_init__ infers one.
        return self.environment_type.model_validate(_thaw(self.environment or {}))

    def grade(
        self,
        agent_output: str,
        pre_environment: Environment,
        post_environment: Environment,
        function_calls: list[FunctionCallRecord],
        observations: Mapping[str, BaseModel] | None = None,
        turns: list[ConversationTurn] | None = None,
    ) -> dict[str, Any]:
        """Grade one run of the episode with its checks.

        ``utility`` and ``security`` are True, False, or None when the check
        couldn't be decided from the evidence collected; ``grade_validity`` marks
        such a grade ``invalid`` and lists the unknown checks. Each grade has a
        reason, whatever its outcome. Without a check, the grade is False, its
        reason None, and its validity ``not_applicable``. ``evidence_status`` is
        how completely each evidence source was collected.
        """
        observations = dict(observations or {})
        if isinstance(self.runtime, OpenShellRuntimeDefinition) and OBSERVATIONS_SOURCE not in observations:
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
        utility, utility_reason, utility_validity = _assess(self.utility, ctx)
        security, security_reason, security_validity = _assess(self.security, ctx)
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
