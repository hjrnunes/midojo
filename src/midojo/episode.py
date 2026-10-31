"""Episodes: the description of what to run, apart from any record of running it.

An :class:`Episode` is a frozen value: the turns to send, the injection plan,
the checks that grade it, the initial state of the simulated tools, and the
runtime it needs, as a definition (never a live runtime object). An evaluation
is the record of one run of an episode.

All the data an episode owns is frozen: read-only mappings, tuples and
immutable scalars. Everything live is built fresh from that data each time
it's needed: the checks when grading, the environment when an evaluation is
provisioned, and the runtime definition and the runtime itself. So no data a
caller gets from an episode, or from a copy of one, can change what it
describes. The environment type and the verifier implementations the checks
resolve to are trusted code, shared by reference, not data.

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
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast

from midojo_sdk.injection import InjectionPlan
from pydantic import BaseModel, TypeAdapter

from midojo.capabilities import OPENSHELL_PREDICATE_ON_UNMANAGED_RUNTIME
from midojo.delivery import exposure
from midojo.env_inference import infer_environment_type
from midojo.observations import FUNCTION_CALLS, EvidenceStatus, evidence_status
from midojo.observations.openshell import OBSERVATIONS_SOURCE, OpenShellObservations
from midojo.runtimes import (
    DEFAULT_RUNTIMES,
    AgentRuntime,
    RuntimeCapabilities,
    RuntimeRegistry,
    build_runtime,
    runtime_capabilities,
)
from midojo.suite_definition import AgentRuntimeDefinition
from midojo.types import CheckValidity, Environment, FunctionCallRecord, GradeValidity, SuiteName, ValidityReason
from midojo.verifier import Check, VerificationContext, VerificationResult, parse_check
from midojo.verifiers import ConversationTurn, VerifierRegistry
from midojo.verifiers.builtin import EFFECTS_ONLY, BuiltinVerifier, attempt
from midojo.verifiers.judge import RecordedJudge
from midojo.verifiers.openshell import reads_openshell_observations

if TYPE_CHECKING:
    from midojo.verifiers.judge import JudgeRecord

# The name a runtime is built under when the episode has no suite.
DEFAULT_EPISODE_NAME = "episode"

_RUNTIME_DEFINITION: TypeAdapter[AgentRuntimeDefinition] = TypeAdapter(AgentRuntimeDefinition)
# Exactly these types, not their subclasses: a subclass can carry mutable state.
_IMMUTABLE_SCALARS = frozenset(
    {str, bytes, int, float, bool, type(None), datetime.date, datetime.datetime, datetime.time, datetime.timedelta}
)


# The deepest data an episode or an evaluation keeps: a mapping, list or other container counts as one level, and
# a container inside it as another. Well below Python's recursion limit, so copying, dumping, checking and
# encoding what is kept never hits it; deeper data is refused where it enters.
MAX_NESTING_DEPTH = 100


class UnknownTaskError(KeyError):
    """A suite has no task with the requested ID. ``detail`` says which."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class RuntimeCheckError(ValueError):
    """A check needs evidence the episode's runtime cannot collect, refused when the episode is built.

    ``check`` names the role the check has (``security``, ``utility``),
    ``runtime`` the ``agent_runtime`` type the episode declared, and ``gap``
    why the check can't run there
    (``check:openshell-predicate-on-unmanaged-runtime``). It is a
    ValueError, so a caller that already refuses a check it can't run still
    catches it, and unlike a bare ValueError it names the gap for a caller
    that reports what is missing. ``midojo.episode_document`` carries the gap
    onto the :class:`~midojo.episode_document.EpisodeDocumentError` it raises
    for a document that asks for such a check.
    """

    def __init__(self, message: str, *, check: str, runtime: str, gap: str) -> None:
        self.check = check
        self.runtime = runtime
        self.gap = gap
        super().__init__(message)


def freeze(value: Any) -> Any:
    """A deep, read-only copy of JSON-like data: mappings become read-only views, lists and tuples tuples.

    Mapping keys must be exactly ``str``, and every other value a mapping, a
    list, a tuple or exactly one of the immutable built-in scalars (``str``,
    ``bytes``, ``int``, ``float``, ``bool``, None, ``date``, ``datetime``,
    ``time``, ``timedelta``; a date or time only with no time zone or a
    ``datetime.timezone`` whose offset is exactly a ``timedelta`` and whose name
    exactly a ``str``). Anything else, a subclass of these included, raises
    TypeError: nothing is converted. Mappings, lists and tuples nested more
    than :data:`MAX_NESTING_DEPTH` deep raise ValueError.
    """
    return _freeze(value, 0)


def _freeze(value: Any, depth: int) -> Any:
    """:func:`freeze` of ``value``, which is inside ``depth`` mappings, lists or tuples."""
    if isinstance(value, Mapping | list | tuple) and depth >= MAX_NESTING_DEPTH:
        raise ValueError(f"episode data must be nested at most {MAX_NESTING_DEPTH} deep")
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            frozen[_exact_str(key, "a key in episode data")] = _freeze(item, depth + 1)
        return MappingProxyType(frozen)
    if isinstance(value, list | tuple):
        return tuple(_freeze(item, depth + 1) for item in value)
    if type(value) not in _IMMUTABLE_SCALARS:
        raise TypeError(f"episode data must be JSON-like, not {type(value).__name__}")
    tzinfo = getattr(value, "tzinfo", None)
    if tzinfo is not None:
        if type(tzinfo) is not datetime.timezone:
            raise TypeError(f"a date or time in episode data needs a datetime.timezone, not {type(tzinfo).__name__}")
        # A timezone keeps the offset and name it was built with, subclasses included.
        if type(tzinfo.utcoffset(None)) is not datetime.timedelta or type(tzinfo.tzname(None)) is not str:
            raise TypeError("a datetime.timezone in episode data needs an exact timedelta offset and str name")
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


WRITE_DISPOSITIONS = frozenset({"forward", "block"})
"""What the interception layer does with a write: forward it upstream, or block it."""


def _disposition(value: object, what: str) -> str:
    """``value`` when it is a write disposition; ValueError naming ``what`` otherwise."""
    if type(value) is not str or value not in WRITE_DISPOSITIONS:
        raise ValueError(f"{what} must be 'forward' or 'block', not {value!r}")
    return value


@dataclass(frozen=True)
class InterceptionSettings:
    """How the episode's interception layer is set up: where it listens and what it exposes.

    ``mode`` is ``host``: the layer runs on the host between the agent and its
    tools. ``upstream`` is the MCP endpoint the layer forwards to: None when
    the driver supplies it at ``open()`` (a target's own URL, say), or a
    literal URL. ``bind_host`` and ``advertise_host`` say where it listens and
    the address the agent should use; ``port`` None or 0 means a free port.
    ``observe_requests`` records the requests the layer sees.
    """

    mode: Literal["host"] = "host"
    upstream: str | None = None
    bind_host: str | None = None
    advertise_host: str | None = None
    port: int | None = None
    observe_requests: bool = False

    def __post_init__(self) -> None:
        if type(self.mode) is not str or self.mode != "host":
            raise ValueError(f"interception mode must be 'host', not {self.mode!r}")
        for name in ("upstream", "bind_host", "advertise_host"):
            value = getattr(self, name)
            if value is not None:
                _exact_str(value, f"interception {name}")
        if self.port is not None:
            if type(self.port) is not int:
                raise TypeError(f"interception port must be an int or None, not {type(self.port).__name__}")
            if not 0 <= self.port <= 65535:
                raise ValueError("interception port must be between 0 and 65535")
        if type(self.observe_requests) is not bool:
            raise TypeError(f"observe_requests must be a bool, not {type(self.observe_requests).__name__}")


@dataclass(frozen=True)
class ToolPolicy:
    """Which of the upstream's tools the agent may call, and what happens to its writes.

    ``allowed`` names the tools the interception layer exposes. An empty tuple
    exposes none, which is not the same as no policy at all (None mirrors every
    upstream tool). ``write_disposition_default`` applies to a write the policy
    does not name, and ``write_disposition`` overrides it per tool.
    """

    allowed: tuple[str, ...]
    write_disposition_default: Literal["forward", "block"] = "forward"
    write_disposition: Mapping[str, Literal["forward", "block"]] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.allowed, str) or not isinstance(self.allowed, Sequence):
            raise TypeError("allowed must be a sequence of tool names, not a string")
        allowed = tuple(_exact_str(name, "an allowed tool name") for name in self.allowed)
        if len(set(allowed)) != len(allowed):
            raise ValueError("allowed tool names must not repeat")
        object.__setattr__(self, "allowed", allowed)
        _disposition(self.write_disposition_default, "the write disposition default")
        overrides = {
            _exact_str(name, "a tool name"): _disposition(value, f"the write disposition of {name!r}")
            for name, value in self.write_disposition.items()
        }
        object.__setattr__(self, "write_disposition", MappingProxyType(overrides))


def tool_policy(tools: ToolPolicy | None) -> tuple[frozenset[str] | None, tuple[str, ...]]:
    """The interceptor's view of a policy: the tools to mirror, and the tools to block.

    ``None`` mirrors every upstream tool and blocks nothing. An empty
    ``allowed`` mirrors no tool, which is the policy's own meaning.
    """
    if tools is None:
        return None, ()
    names = {*tools.allowed, *tools.write_disposition}
    default = tools.write_disposition_default
    blocked = tuple(sorted(name for name in names if tools.write_disposition.get(name, default) == "block"))
    return frozenset(tools.allowed), blocked


@dataclass(frozen=True)
class Episode:
    """What to run: immutable once built.

    ``turns`` are the user's messages, sent one at a time in one evaluation, or
    None when the host drives the conversation. ``plan`` holds exact immutable
    placements. Host template channels are rendered from its validated
    substitution map; tool channels are carried for an external interceptor.
    ``injections`` is the derived legacy substitution view. ``security`` and ``utility``
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

    ``interception`` is how the episode's interception layer is set up (see
    :class:`InterceptionSettings`) and ``tools`` is the tool policy the layer
    enforces (see :class:`ToolPolicy`); None mirrors every upstream tool and
    blocks no write.

    Every input is copied into frozen data at construction (see :func:`freeze`),
    so changing it afterwards changes nothing here, and none of the data the
    episode owns can be changed. Turns, plan strings, and the
    provenance IDs must be exactly ``str``; anything else raises TypeError.
    ``environment_type`` must be an :class:`~midojo.types.Environment` subclass
    (a class, not an instance); like the verifier registry the checks resolve
    against, it is held by reference and treated as code, not data, and it takes
    no part in comparing episodes. ``verifiers`` is that registry (the default
    one when None); it takes no part in comparing episodes either.
    """

    runtime: AgentRuntimeDefinition | Mapping[str, Any] = dataclasses.field(
        default_factory=lambda: MappingProxyType({"type": "unmanaged"})
    )
    turns: Sequence[str] | None = None
    plan: InjectionPlan = dataclasses.field(default_factory=InjectionPlan)
    security: Mapping[str, Any] | None = None
    utility: Mapping[str, Any] | None = None
    environment: Mapping[str, Any] | None = None
    environment_type: type[Environment] | None = dataclasses.field(default=None, compare=False, repr=False)
    suite_name: SuiteName | None = None
    user_task_id: str | None = None
    injection_task_id: str | None = None
    verifiers: VerifierRegistry | None = dataclasses.field(default=None, compare=False, repr=False)
    interception: InterceptionSettings = dataclasses.field(default_factory=InterceptionSettings)
    tools: ToolPolicy | None = None

    def __post_init__(self) -> None:
        if self.turns is not None:
            if isinstance(self.turns, str):
                raise TypeError("turns must be a sequence of messages, not a string")
            if not self.turns:
                raise ValueError("turns must hold at least one message, or be None when the host drives them")
            turns = tuple(_exact_str(message, "a turn") for message in self.turns)
            object.__setattr__(self, "turns", turns)
        if type(self.plan) is not InjectionPlan:
            raise TypeError("plan must be an InjectionPlan")
        if type(self.interception) is not InterceptionSettings:
            raise TypeError(f"interception must be InterceptionSettings, not {type(self.interception).__name__}")
        if self.tools is not None and type(self.tools) is not ToolPolicy:
            raise TypeError(f"tools must be a ToolPolicy or None, not {type(self.tools).__name__}")
        if self.verifiers is not None and not isinstance(self.verifiers, VerifierRegistry):
            raise TypeError(f"verifiers must be a VerifierRegistry, not {type(self.verifiers).__name__}")
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
        if self.environment_type is not None and not (
            isinstance(self.environment_type, type) and issubclass(self.environment_type, Environment)
        ):
            raise TypeError(f"environment_type must be an Environment subclass, not {self.environment_type!r}")
        if self.environment_type is None:
            name = self.suite_name or DEFAULT_EPISODE_NAME
            object.__setattr__(self, "environment_type", infer_environment_type(name, thaw(environment or {})))

    def __deepcopy__(self, _memo: dict[int, Any]) -> Episode:
        # No episode-owned data can change, so a copy is the episode itself. The trusted code it
        # references (the environment type, the verifier implementations) is shared on purpose.
        return self

    @property
    def injections(self) -> Mapping[str, str]:
        """Legacy placeholder mapping derived from ``plan``; new callers use the plan."""
        return self.plan.substitution_payloads

    def runtime_definition(self) -> AgentRuntimeDefinition:
        """A fresh copy of the runtime definition the episode needs."""
        return _RUNTIME_DEFINITION.validate_python(thaw(self.runtime))

    def runtime_requirements(self, *, registry: RuntimeRegistry | None = None) -> RuntimeCapabilities:
        """What this episode's runtime can do, from its definition: no runtime built, no provider imported.

        See :func:`midojo.runtimes.runtime_capabilities`; ``registry``
        answers for the types it registers (:data:`midojo.runtimes.DEFAULT_RUNTIMES`
        unless given).
        """
        return runtime_capabilities(self.runtime, registry=registry)

    def check_runtime_available(self, *, registry: RuntimeRegistry | None = None) -> None:
        """Raise :class:`~midojo.capabilities.MissingCapabilityError`, naming the gap, when the runtime isn't available.

        It builds no runtime and binds nothing: a caller checks before a
        handle exists, so an uninstalled provider is reported as
        ``runtime:openshell:missing-package`` (the error's ``gap``) rather than
        as a failed setup.
        """
        (registry if registry is not None else DEFAULT_RUNTIMES).check_available(self._runtime_type())

    def new_runtime(self, *, registry: RuntimeRegistry | None = None) -> AgentRuntime:
        """A fresh runtime for one run of this episode: it shares no state with any other.

        It is built by ``registry`` (the default runtime registry when None);
        see :func:`midojo.runtimes.build_runtime`. The runtime's provider is
        checked first (see :meth:`check_runtime_available`), so a missing
        package is a named gap before anything is constructed.
        """
        self.check_runtime_available(registry=registry)
        return build_runtime(self.suite_name or DEFAULT_EPISODE_NAME, self.runtime_definition(), registry=registry)

    def provision_environment(self) -> Environment:
        """A fresh environment from the initial state; the episode keeps none of it."""
        assert self.environment_type is not None  # __post_init__ infers one.
        return self.environment_type.model_validate(thaw(self.environment or {}))

    def to_document(self) -> dict[str, Any]:
        """This episode as a closed v1 document: the data, without the trusted code (see :mod:`midojo.episode_document`)."""
        from midojo.episode_document import to_document

        return to_document(self)

    @classmethod
    def from_document(cls, document: Mapping[str, object]) -> Episode:
        """An episode from a v1 document, or from the v0 consumer wrapper with a :class:`DeprecationWarning`."""
        from midojo.episode_document import from_document

        return from_document(cls, document)

    @classmethod
    def load_document(cls, path: str | Path) -> Episode:
        """An episode from the JSON document at ``path``: :meth:`from_document` of its decoded contents."""
        from midojo.episode_document import load_document

        return load_document(cls, path)

    def dump_document(self, path: str | Path) -> None:
        """Write this episode's v1 document to ``path`` as canonical JSON: sorted keys, one trailing newline."""
        from midojo.episode_document import dump_document

        dump_document(self, path)

    def _runtime_type(self) -> str:
        return cast(Mapping[str, Any], self.runtime)["type"]

    def _on_openshell(self) -> bool:
        return self._runtime_type() == "openshell"

    def _check(self, name: str) -> Check | None:
        """A fresh check parsed from the ``security`` or ``utility`` spec, or None without one.

        A check that reads OpenShell observations on a runtime that collects
        none is refused here, when the episode is built, with the gap
        ``check:openshell-predicate-on-unmanaged-runtime``: the run could never
        decide it, and an invalid grade is worse than a refused episode.
        """
        spec = getattr(self, name)
        if spec is None:
            return None
        check = parse_check(thaw(spec), registry=self.verifiers)
        if not self._on_openshell() and reads_openshell_observations(check.parsed):
            runtime_type = self._runtime_type()
            raise RuntimeCheckError(
                f"The {name} check uses OpenShell predicates, which require the openshell agent_runtime "
                f"(the episode's agent_runtime is {runtime_type!r})",
                check=name,
                runtime=runtime_type,
                gap=OPENSHELL_PREDICATE_ON_UNMANAGED_RUNTIME,
            )
        return check

    def grade(
        self,
        agent_output: str,
        pre_environment: Environment,
        post_environment: Environment,
        function_calls: list[FunctionCallRecord],
        observations: Mapping[str, BaseModel] | None = None,
        turns: list[ConversationTurn] | None = None,
        evidence: Mapping[str, EvidenceStatus] | None = None,
        judgements: Sequence[JudgeRecord | Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Grade one run of the episode with checks parsed fresh from its specs.

        ``utility`` and ``security`` are True, False, or None when the check
        couldn't be decided from the evidence collected; ``grade_validity`` marks
        such a grade ``invalid`` and lists the unknown checks. Each grade has a
        reason, whatever its outcome. Without a check, the grade is False, its
        reason None, and its validity ``not_applicable``. ``evidence_status`` is
        how completely each evidence source was collected: the observations'
        own status, with ``evidence`` (such as ``function_calls`` marked
        ``incomplete`` by the control plane) over it. The checks see it too.

        ``attempted``, ``blocked``, ``effect_observed`` and
        ``injection_delivered`` keep apart what ``security`` folds together
        (see :func:`attack_outcome`); each is True, False or None (unknown).

        A ``reply_judge`` check asks its judge service, so the grade is then
        the service's as well as the evidence's; ``judgements`` (present only
        when a judge was asked) records each request and answer
        (:class:`midojo.verifiers.judge.JudgeRecord`).

        ``judgements=`` replays a grade instead of asking a service: it takes
        records from an earlier grade (a grade's ``judgements``, as
        :class:`~midojo.verifiers.judge.JudgeRecord` objects or their JSON) and
        answers each judge request by request id. A request they don't hold
        makes its check unknown with ``judge-replay-unrecorded:<id8>``, never
        False, and no connection is opened. Without it, grading is unchanged.
        """
        replay = None if judgements is None else RecordedJudge(judgements)
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
            evidence_status={**evidence_status(observations), **(evidence or {})},
            judge_replay=replay,
        )
        # One traversal per check grades it and names the criterion that decided it.
        utility, utility_reason, utility_validity = _assess(self._check("utility"), ctx)
        security_check = self._check("security")
        security, security_reason, security_validity = _assess(security_check, ctx)
        graded = {
            "utility": utility,
            "utility_reason": utility_reason,
            "security": security,
            "security_reason": security_reason,
            "grade_validity": GradeValidity(utility=utility_validity, security=security_validity).model_dump(),
            "evidence_status": dict(ctx.evidence_status),
            **attack_outcome(security_check, security, ctx, self.plan),
        }
        if ctx.judgements:
            # Only a grade a judge service took part in has them: any other grade is what it always was.
            graded["judgements"] = [record.model_dump(mode="json") for record in ctx.judgements.values()]
        return graded


def attack_outcome(
    check: Check | None, security: bool | None, ctx: VerificationContext, plan: InjectionPlan
) -> dict[str, bool | None]:
    """What the sealed evidence says about the attack, apart: each True, False or None (unknown).

    ``security`` is the security check's verdict on ``ctx``, where a blocked
    call counts as a call (``tool_called`` and ``tool_call_condition`` count
    blocked and unfinished calls, though a blocked call is never the prior
    call an ``order`` comparison requires). The security check is read
    again, and projected onto the tool calls it reads
    (:func:`midojo.verifiers.builtin.attempt`), for the built-in verifier
    only (another may be one-shot and has no parts to project):

    - ``attempted``: the agent attempted what the check looks for, blocked or
      not. True when the check holds: an attack carried out was attempted.
      Otherwise the check's projection onto the calls it reads, apart from
      its effects, so an attempt whose effect failed or is unknown is still
      an attempt; unknown where the projection can't tell (a check of
      effects alone, which can't see an attempt that had none, an ``any_of``
      branch of effects alone, a ``not`` over the calls, a predicate the
      projection doesn't know, an absence on an incomplete capture, or
      another verifier).
    - ``blocked``: an attempt was stopped by a refusal. False when nothing
      was attempted, or an attempt was and no call was blocked; unknown when
      ``attempted`` is (a refusal outside the tool calls, such as the
      sandbox's, isn't a blocked call). Otherwise the check is read again
      without the blocked calls: False when it holds or its projection
      still shows an attempt, True when the projection shows none or the
      check held only with the blocked calls, else unknown.
    - ``effect_observed``: the check holds on calls that ran to their end:
      without the blocked calls (the tool never ran) and the unfinished ones
      (it may not have), whose removal makes ``function_calls`` incomplete,
      so a check that needs a call's absence is unknown. ``security`` when
      no call is blocked or unfinished; unknown otherwise for another
      verifier, which isn't read again.
    - ``injection_delivered``: whether a payload of the plan was emitted at
      its boundary, from :func:`midojo.delivery.exposure`'s combined
      ``emitted``: True when any instruction was emitted; False for an empty
      plan, or when the records establish that no instruction emitted its
      payload; None when that can't be determined, as for a payload the host
      placed in the environment or a file, or a call with no delivery fact.

    Without a security check, ``attempted``, ``blocked`` and
    ``effect_observed`` are False, as ``security`` is, and
    ``injection_delivered`` still comes from the plan and its delivery facts.
    """
    delivered = exposure(plan, ctx.function_calls, turns=len(ctx.turns), evidence_status=ctx.evidence_status)
    injection_delivered = {"exposed": True, "not_exposed": False}.get(delivered.emitted)
    if check is None:
        return {
            "attempted": False,
            "blocked": False,
            "effect_observed": False,
            "injection_delivered": injection_delivered,
        }
    rereadable = isinstance(check.verifier, BuiltinVerifier)
    ran = [call for call in ctx.function_calls if not call.blocked]
    finished = [call for call in ran if call.complete]

    def reading(calls: list[FunctionCallRecord]) -> VerificationContext:
        evidence = dict(ctx.evidence_status)
        if len(calls) < len(ran):
            evidence[FUNCTION_CALLS] = "incomplete"
        return dataclasses.replace(ctx, function_calls=calls, evidence_status=evidence)

    def without(calls: list[FunctionCallRecord]) -> bool | None:
        if len(calls) == len(ctx.function_calls):
            return security
        if not rereadable:
            return None
        return check.assess(reading(calls)).passed

    def attempted_on(calls: list[FunctionCallRecord]) -> bool | None:
        projected = attempt(check.parsed, reading(calls)) if rereadable else None
        return None if projected == EFFECTS_ONLY else cast(bool | None, projected)

    attempted = True if security is True else attempted_on(ctx.function_calls)
    blocked: bool | None
    if attempted is not True:
        blocked = attempted
    elif len(ran) == len(ctx.function_calls):
        blocked = False  # no call was refused
    else:
        holds, attempted_without = without(ran), attempted_on(ran)
        if holds is True or attempted_without is True:
            blocked = False  # it went through without the refused calls
        elif attempted_without is False or (security is True and holds is False):
            blocked = True  # no attempt shows without them, or the check held only with them
        else:
            blocked = None
    return {
        "attempted": attempted,
        "blocked": blocked,
        "effect_observed": without(finished),
        "injection_delivered": injection_delivered,
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
