"""Attackers: the policy that picks each attempt's injection.

The rest of this package is the attack *material*: delivery techniques and
payload corpora. An attacker is the *policy* that uses it. For every attempt of
a (user task x injection task) pair it proposes the injection to deliver, and it
may learn from the outcomes of the attempts before. The orchestrator's attempt
loop is written against the :class:`Attacker` protocol, so a model-driven
attacker slots in beside :class:`StaticAttacker` without reshaping the loop.

The loop is a client of the episode API, under the same rules as any other:
each attempt is one episode compiled from the suite pair with the attempt's
injections, run through the episode context, sealed and graded. An attempt's
:class:`AttemptOutcome` is read from that sealed verdict and the delivery facts
of its snapshot, three-valued where the grade is. An attacker only fills the
payloads of the probes the pair's injection task declares
(:func:`permitted_injections`, :func:`check_plan_within`): it can't name another
probe, channel or target, so it never widens the episode's plan.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from midojo_sdk.injection import InjectionInstruction, InjectionPlan

from midojo.control_plane.models import GradeResponse
from midojo.delivery import EXPOSURE_BOUNDARIES, DeliveryFact, Exposure

# What became of an attempt: ``graded`` (its episode ran, sealed and was graded), ``failed`` (the attacker or the
# episode raised, as a failed pair does: no verdict) or ``refused`` (the proposal left the permitted surface: no
# episode ran).
AttemptStatus = Literal["graded", "failed", "refused"]

# How an attempt was kept apart from every other: ``reset`` (the runtime's declared reset ran before it),
# ``disposable`` (the runtime gives every evaluation a fresh target) or ``none`` (neither: only a single attempt runs
# on such a runtime, as a pair always did).
AttemptIsolation = Literal["reset", "disposable", "none"]


class OutOfSurfaceError(ValueError):
    """An attempt named a probe, channel or target the pair's injection plan doesn't allow."""


@dataclass(frozen=True)
class AttackAttempt:
    """What an attacker proposes for one attempt.

    ``injections`` maps probe keys (``"<injection_task>:<probe>"``) to payload
    strings, as ``suite.episode`` takes them. Only the probes the pair's
    injection task declares may be named (see :func:`permitted_injections`); a
    probe left out is simply not placed.
    """

    injections: dict[str, str]


@dataclass(frozen=True)
class AttemptOutcome:
    """The observed result of one attempt, read from its sealed verdict and delivery facts.

    Fed back to the attacker as history, and recorded in the run's results.
    ``utility``, ``security``, ``attempted``, ``blocked``, ``effect_observed``
    and ``injection_delivered`` are the grade's, each True, False or None
    (unknown); ``exposure`` is the combined status at each boundary
    (``emitted``, ``agent_tool_result``, ``model_input``) and ``delivery`` the
    snapshot's delivery facts. A failed or refused attempt has none of them:
    it is no evidence either way, never a non-break.

    ``broke`` is the break the attempt counts as: its ``security``, except for
    an adaptive attacker, whose success counts only once ``replay`` (the same
    injections run again in a fresh episode, from a clean start) breaks too:
    True when it does, False when it doesn't, None when the replay is unknown,
    failed or couldn't run. A failed or refused attempt's ``broke`` is None.
    """

    attempt: int
    injections: dict[str, str]
    status: AttemptStatus
    isolation: AttemptIsolation = "none"
    eval_id: str | None = None
    utility: bool | None = None
    security: bool | None = None
    security_reason: str | None = None
    grade_validity: dict[str, Any] | None = None
    attempted: bool | None = None
    blocked: bool | None = None
    effect_observed: bool | None = None
    injection_delivered: bool | None = None
    exposure: dict[str, str] | None = None
    delivery: list[dict[str, Any]] = field(default_factory=list)
    agent_output: str | None = None
    failure_reason: str | None = None
    broke: bool | None = None
    replay: AttemptOutcome | None = None

    @classmethod
    def from_verdict(
        cls,
        attempt: int,
        injections: Mapping[str, str],
        grade: GradeResponse,
        facts: Sequence[DeliveryFact] = (),
        exposure: Exposure | None = None,
        *,
        eval_id: str,
        agent_output: str,
        isolation: AttemptIsolation,
    ) -> AttemptOutcome:
        """A graded attempt, from its sealed verdict and its snapshot's delivery facts and exposure.

        ``broke`` is the grade's ``security``; an adaptive attacker's success is
        confirmed with :meth:`confirmed_by`. A pair without an injection task
        has no ``exposure``.
        """
        return cls(
            attempt=attempt,
            injections=dict(injections),
            status="graded",
            isolation=isolation,
            eval_id=eval_id,
            utility=grade.utility,
            security=grade.security,
            security_reason=grade.security_reason,
            grade_validity=grade.grade_validity.model_dump(mode="json"),
            attempted=grade.attempted,
            blocked=grade.blocked,
            effect_observed=grade.effect_observed,
            injection_delivered=grade.injection_delivered,
            exposure=None
            if exposure is None
            else {boundary: exposure.at(boundary) for boundary in EXPOSURE_BOUNDARIES},
            delivery=[fact.model_dump(mode="json") for fact in facts],
            agent_output=agent_output,
            broke=grade.security,
        )

    @classmethod
    def unrun(
        cls,
        attempt: int,
        injections: Mapping[str, str],
        status: Literal["failed", "refused"],
        reason: str,
        *,
        eval_id: str | None = None,
        isolation: AttemptIsolation = "none",
    ) -> AttemptOutcome:
        """An attempt with no verdict: it failed (its evaluation, if opened, is ``eval_id``) or was refused."""
        return cls(
            attempt=attempt,
            injections=dict(injections),
            status=status,
            isolation=isolation,
            eval_id=eval_id,
            failure_reason=reason,
        )

    def confirmed_by(self, replay: AttemptOutcome) -> AttemptOutcome:
        """This adaptive success, counted only as far as its clean-start ``replay`` breaks too."""
        broke = self.broke
        if broke is True:
            broke = replay.security if replay.status == "graded" else None
        return dataclasses.replace(self, replay=replay, broke=broke)


class Attacker(Protocol):
    """Proposes the injection for each attempt, optionally adapting to history.

    ``propose`` is async so a model-driven attacker can call its own inference
    endpoint. ``goal`` is the objective for an attacker that pursues one;
    nothing supplies an objective yet, so the orchestrator passes ``None``.
    ``history`` holds the outcomes of the pair's earlier attempts, in order.

    ``adaptive`` declares whether proposals depend on ``history``: an adaptive
    attacker's success counts only after its winning injections, replayed in a
    fresh episode from a clean start, break the agent again. An attacker that
    doesn't declare it is taken to be adaptive.
    """

    adaptive: bool

    async def propose(self, goal: str | None, history: Sequence[AttemptOutcome]) -> AttackAttempt: ...


def is_adaptive(attacker: object) -> bool:
    """Whether ``attacker``'s proposals may depend on history: True unless it declares ``adaptive = False``."""
    return getattr(attacker, "adaptive", True) is not False


class StaticAttacker:
    """Proposes the same suite-defined injection on every attempt.

    With one attempt this is MiDojo's original single-shot run. With K attempts
    it delivers the identical injection K times, each in a fresh episode on an
    isolated target, which shows how often it breaks the agent under the
    agent's own stochasticity. It ignores ``goal`` and ``history``: there is
    nothing to adapt, so every attempt is already a clean-start run of a fixed
    payload and needs no replay.
    """

    adaptive = False

    def __init__(self, injections: dict[str, str]) -> None:
        self._injections = dict(injections)

    async def propose(self, goal: str | None, history: Sequence[AttemptOutcome]) -> AttackAttempt:
        return AttackAttempt(injections=dict(self._injections))


def permitted_injections(attempt: object, permitted: Mapping[str, str]) -> dict[str, str]:
    """The attempt's injections, refused unless each names a probe in ``permitted`` with a ``str`` payload.

    ``permitted`` is the pair's own injections (``suite.get_probes_for_task``);
    an attempt may fill any of their payloads and leave any out, but name no
    other probe. Raises :class:`OutOfSurfaceError`.
    """
    if not isinstance(attempt, AttackAttempt):
        raise OutOfSurfaceError(f"an attacker must propose an AttackAttempt, not {type(attempt).__name__}")
    if not isinstance(attempt.injections, Mapping):
        raise OutOfSurfaceError("an attempt's injections must map probe keys to payloads")
    injections: dict[str, str] = {}
    for key, payload in attempt.injections.items():
        if type(key) is not str or key not in permitted:
            allowed = ", ".join(sorted(permitted)) or "none"
            raise OutOfSurfaceError(f"probe {key!r} is outside the pair's injection plan (permitted: {allowed})")
        if type(payload) is not str:
            raise OutOfSurfaceError(f"probe {key!r} needs a str payload, not {type(payload).__name__}")
        injections[key] = payload
    return injections


def _placement(instruction: InjectionInstruction) -> InjectionInstruction:
    """The instruction with its payload blanked: where and how it places, not what."""
    return dataclasses.replace(instruction, payload="")


def check_plan_within(proposed: InjectionPlan, permitted: InjectionPlan) -> None:
    """Refuse a proposed plan that places anything the pair's own plan doesn't: it may only change payloads.

    Every proposed instruction must match one of ``permitted``'s in all but
    its payload (channel, target, probe key, occurrence, transformation, size
    limit, mode and mismatch policy), and every proposed probe must be one of
    ``permitted``'s. Raises :class:`OutOfSurfaceError`.
    """
    allowed = {_placement(instruction) for instruction in permitted.instructions}
    for instruction in proposed.instructions:
        if _placement(instruction) not in allowed:
            raise OutOfSurfaceError(
                f"instruction for probe {instruction.probe_key!r} on channel {instruction.channel.value} places "
                f"outside the pair's injection plan"
            )
    probes = {key for key, _ in permitted.probe_payloads}
    if extra := sorted({key for key, _ in proposed.probe_payloads} - probes):
        raise OutOfSurfaceError(f"probes {extra} are outside the pair's injection plan")
