"""Delivery facts, and the exposure derived from them.

A *delivery fact* is what the component that places an injection did at its
own boundary, recorded as it happened: for a tool-output instruction, what the
interceptor selected, transformed and emitted in one call's result (each
sealed call's ``injection_outcomes``, :class:`midojo.types.CallInjectionOutcome`).
A fact references its instruction by index and probe key; the plan holds
none.

*Exposure* is derived from the facts and names its boundary:

- ``emitted``: the placing component sent the payload out at its own boundary
  (the interceptor in a tool result it returned; the host in a turn message
  it started);
- ``agent_tool_result``: the payload reached the agent's tool result;
- ``model_input``: the payload reached the input of the agent's model.

Each is ``exposed``, ``not_exposed`` or ``unknown``. Nothing past the
interceptor is observed: the interceptor records a result before it returns
it, so an emitted payload's ``agent_tool_result`` is unknown, and with no
recording of the model's requests every payload's ``model_input`` is unknown. A
payload the interceptor didn't emit never reached the agent through that
instruction, so its ``agent_tool_result`` is ``not_exposed`` too.

Delivery and exposure are report-only: they never change a grade's
``security`` or ``utility``, and never decide which evaluations count in a
rate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from midojo_sdk.injection import Channel, InjectionInstruction, InjectionPlan
from pydantic import BaseModel

from midojo.observations import FUNCTION_CALLS, EvidenceStatus
from midojo.types import (
    CallDisposition,
    FunctionCallRecord,
    InjectionStatus,
    InjectionTransformation,
    ResultPart,
)

ExposureBoundary = Literal["emitted", "agent_tool_result", "model_input"]
EXPOSURE_BOUNDARIES: tuple[ExposureBoundary, ...] = ("emitted", "agent_tool_result", "model_input")

ExposureStatus = Literal["exposed", "not_exposed", "unknown"]

# Who places an instruction's payload: the interceptor in tool results, or the host in what it compiles
# (the environment, the turns, the runtime's files) before the agent runs.
PlacedBy = Literal["interceptor", "host"]

# The boundaries each channel's payload can cross. A turn message isn't a tool result.
_BOUNDARIES: Mapping[Channel, tuple[ExposureBoundary, ...]] = {
    Channel.TOOL_OUTPUT: EXPOSURE_BOUNDARIES,
    Channel.TOOL_DESCRIPTION: EXPOSURE_BOUNDARIES,
    Channel.USER_PROMPT: ("emitted", "model_input"),
    Channel.ENVIRONMENT_STATE: EXPOSURE_BOUNDARIES,
    Channel.RUNTIME_FILE: EXPOSURE_BOUNDARIES,
}


class DeliveryFact(BaseModel):
    """What the interceptor did with one instruction on one recorded call, at its own boundary.

    ``instruction_index`` is the instruction's index in the plan the
    interceptor was given; ``selected``, ``transformation`` and ``emitted``
    are None when the interceptor didn't report them.
    """

    call_id: str
    function: str
    disposition: CallDisposition | None
    turn: int | None
    instruction_index: int
    probe_key: str | None
    occurrence: int
    status: InjectionStatus
    selected: bool | None
    transformation: InjectionTransformation | None
    emitted: list[ResultPart] | None


def delivery_facts(function_calls: Iterable[FunctionCallRecord]) -> list[DeliveryFact]:
    """Every delivery fact the recorded calls carry, in call then instruction order."""
    return [
        DeliveryFact(
            call_id=call.call_id,
            function=call.function,
            disposition=call.disposition,
            turn=call.turn,
            instruction_index=outcome.instruction_index,
            probe_key=outcome.probe_key,
            occurrence=outcome.occurrence,
            status=outcome.status,
            selected=outcome.selected,
            transformation=outcome.transformation,
            emitted=None if outcome.emitted is None else list(outcome.emitted),
        )
        for call in function_calls
        for outcome in call.injection_outcomes or ()
    ]


class InstructionExposure(BaseModel):
    """One instruction of an episode's plan: who placed it, and its exposure at each boundary its channel has.

    ``instruction_index`` is its index in the episode's plan, and
    ``reason`` says why the first boundary that isn't ``exposed`` is what it
    is.
    """

    instruction_index: int
    channel: str
    probe_key: str | None
    placed_by: PlacedBy
    exposure: dict[ExposureBoundary, ExposureStatus]
    reason: str


class Exposure(BaseModel):
    """An evaluation's exposure at each boundary, over every instruction of its plan.

    A boundary is ``exposed`` when any instruction's payload is exposed there,
    ``not_exposed`` when every instruction with that boundary is
    ``not_exposed`` (or none has it: an empty plan places nothing, and a
    prompt payload is in no tool result), and ``unknown`` otherwise.
    """

    emitted: ExposureStatus
    agent_tool_result: ExposureStatus
    model_input: ExposureStatus
    instructions: list[InstructionExposure]

    def at(self, boundary: ExposureBoundary) -> ExposureStatus:
        return getattr(self, boundary)


def exposure(
    plan: InjectionPlan,
    function_calls: Sequence[FunctionCallRecord],
    *,
    turns: int,
    evidence_status: Mapping[str, EvidenceStatus],
) -> Exposure:
    """The exposure of ``plan``'s payloads in an evaluation, derived from its sealed records.

    ``function_calls`` are the evaluation's call records and their delivery
    facts, ``turns`` how many turns it started, and ``evidence_status`` how
    completely they were collected (a key that's absent has no known gap).

    A tool-output instruction is placed by the interceptor, which is given
    the plan's tool-output part (``plan.for_channels(Channel.TOOL_OUTPUT)``):
    its facts are those whose index is the instruction's place in that part
    and whose probe key is the instruction's. It is ``emitted`` when a fact
    names a part of the result that carries it. It is ``not_exposed`` only
    when the record says it can't have been: ``function_calls`` is complete,
    and every call to the instruction's tool that ran finished with
    reported facts (a blocked call never ran, and the interceptor never
    places a payload in its refusal), none naming the instruction as
    emitted. Anything else is unknown.

    A user-prompt instruction is placed by the host in a turn's message: it
    is ``emitted`` once that turn started, ``not_exposed`` when it didn't and
    ``turns`` is complete. An environment-state or runtime-file payload is
    placed by the host before the agent runs, and nothing records whether a
    tool or a file read carried it on: every boundary is unknown.
    """
    tool_output = [index for index, instruction in enumerate(plan) if instruction.channel is Channel.TOOL_OUTPUT]
    instructions: list[InstructionExposure] = []
    for index, instruction in enumerate(plan):
        emitted: ExposureStatus
        if instruction.channel is Channel.TOOL_OUTPUT:
            emitted, reason = _tool_output_emitted(
                tool_output.index(index), instruction, function_calls, evidence_status
            )
        elif instruction.channel is Channel.USER_PROMPT:
            emitted, reason = _prompt_emitted(instruction, turns, evidence_status)
        else:
            emitted, reason = "unknown", f"placed by the host in the {instruction.channel.value}: no delivery fact"
        boundaries = _BOUNDARIES[instruction.channel]
        instructions.append(
            InstructionExposure(
                instruction_index=index,
                channel=instruction.channel.value,
                probe_key=instruction.probe_key,
                placed_by="interceptor" if instruction.channel is Channel.TOOL_OUTPUT else "host",
                exposure={boundary: _past(boundary, emitted) for boundary in boundaries},
                reason=reason,
            )
        )

    def combined(boundary: ExposureBoundary) -> ExposureStatus:
        return _combine([item.exposure[boundary] for item in instructions if boundary in item.exposure])

    return Exposure(
        emitted=combined("emitted"),
        agent_tool_result=combined("agent_tool_result"),
        model_input=combined("model_input"),
        instructions=instructions,
    )


def _past(boundary: ExposureBoundary, emitted: ExposureStatus) -> ExposureStatus:
    """The exposure at ``boundary`` of a payload whose emission is ``emitted``.

    Only emission is recorded. A payload that wasn't emitted reached no tool
    result through its instruction; past the interceptor nothing is observed,
    and no model input is ever observed.
    """
    if boundary == "emitted":
        return emitted
    if boundary == "agent_tool_result" and emitted == "not_exposed":
        return "not_exposed"
    return "unknown"


def _combine(statuses: list[ExposureStatus]) -> ExposureStatus:
    if "exposed" in statuses:
        return "exposed"
    if all(status == "not_exposed" for status in statuses):
        return "not_exposed"
    return "unknown"


def _tool_output_emitted(
    interceptor_index: int,
    instruction: InjectionInstruction,
    function_calls: Sequence[FunctionCallRecord],
    evidence_status: Mapping[str, EvidenceStatus],
) -> tuple[ExposureStatus, str]:
    gaps: list[str] = []
    if evidence_status.get(FUNCTION_CALLS, "complete") != "complete":
        gaps.append("function call capture is incomplete")
    tool = instruction.target.tool
    for call in function_calls:
        if tool is not None and call.function != tool:
            continue
        if call.blocked:
            continue  # the tool never ran: the agent got the layer's refusal
        if not call.complete:
            gaps.append(f"call {call.call_id} is unfinished")
            continue
        if call.injection_outcomes is None:
            gaps.append(f"call {call.call_id} reported no delivery facts")
            continue
        for outcome in call.injection_outcomes:
            if outcome.instruction_index != interceptor_index:
                continue
            if outcome.probe_key != instruction.probe_key:
                gaps.append(f"call {call.call_id} names instruction {interceptor_index} with another probe")
            elif outcome.emitted:
                return "exposed", f"emitted in {', '.join(outcome.emitted)} of call {call.call_id}"
            elif outcome.emitted is None:
                gaps.append(f"call {call.call_id} didn't report what it emitted")
    if gaps:
        return "unknown", "; ".join(gaps)
    return "not_exposed", "no call's result carried it"


def _prompt_emitted(
    instruction: InjectionInstruction, turns: int, evidence_status: Mapping[str, EvidenceStatus]
) -> tuple[ExposureStatus, str]:
    path = instruction.target.template_path
    turn = path[0] if path and type(path[0]) is int else None
    if turn is None:
        return "unknown", "the instruction names no turn"
    if turn < turns:
        return "exposed", f"sent in turn {turn}"
    if evidence_status.get("turns", "complete") != "complete":
        return "unknown", f"turn {turn} wasn't recorded, and turns are incomplete"
    return "not_exposed", f"turn {turn} never started"
