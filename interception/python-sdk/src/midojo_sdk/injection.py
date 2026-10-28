"""Immutable injection plans shared by hosts and interception layers.

``append`` is the one default mode: for an unqualified content selector it
preserves upstream data and has an unambiguous meaning. Specific placements
never guess another field.
"""

from __future__ import annotations

import enum
import hashlib
import json
import math
import re
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class RecordSelector:
    """Select the sole list record whose ``field`` exactly equals ``value``.

    The field defaults to ``id``. Values are exact ``str`` or ``int`` values:
    booleans are refused, and an integer never matches its string form.
    """

    value: str | int
    field: str = "id"

    def __post_init__(self) -> None:
        if type(self.field) is not str:
            raise TypeError("record selector field must be a str")
        if not self.field:
            raise ValueError("record selector field must not be empty")
        if type(self.value) not in {str, int}:
            raise TypeError("record selector value must be an exact str or int")


type PathPart = str | int | RecordSelector
type FieldPath = tuple[PathPart, ...]
type ArgumentValue = str | int | float | bool | None
type CallArguments = tuple[tuple[str, ArgumentValue], ...]
type CallArgumentsInput = CallArguments | Mapping[str, ArgumentValue]


class Channel(enum.StrEnum):
    TOOL_OUTPUT = "tool_output"
    TOOL_DESCRIPTION = "tool_description"
    USER_PROMPT = "user_prompt"
    ENVIRONMENT_STATE = "environment_state"
    RUNTIME_FILE = "runtime_file"


class SelectorSurface(enum.StrEnum):
    CONTENT_BLOCK = "content_block"
    STRUCTURED_CONTENT = "structured_content"
    DESCRIPTION = "description"
    TEMPLATE = "template"


class InjectionMode(enum.StrEnum):
    APPEND = "append"
    REPLACE = "replace"


class Occurrence(enum.StrEnum):
    FIRST = "first"
    NTH = "nth"
    EVERY = "every"


class Transformation(enum.StrEnum):
    IDENTITY = "identity"
    JSON_STRING = "json_string"


class MismatchPolicy(enum.StrEnum):
    STRICT = "strict"
    FALLBACK_APPEND = "fallback_append"


class OutcomeStatus(enum.StrEnum):
    """How one instruction fared on one call.

    ``partial`` is a structured-content placement the agent's text content
    doesn't carry: the result's text isn't a JSON mirror of its structured
    content, so only ``structuredContent`` has the payload. ``unmatched`` is
    also a call to the target's tool whose arguments don't match its
    ``call_arguments``, or whose result doesn't match its ``result_record``,
    whatever the mismatch policy; its occurrence is 0.
    """

    APPLIED = "applied"
    PARTIAL = "partial"
    FALLBACK = "fallback"
    UNMATCHED = "unmatched"
    NO_RESULT = "no_result"
    TOO_LARGE = "too_large"
    SKIPPED_OCCURRENCE = "skipped_occurrence"


PLACED: frozenset[OutcomeStatus] = frozenset({OutcomeStatus.APPLIED, OutcomeStatus.PARTIAL, OutcomeStatus.FALLBACK})
"""The outcomes that wrote the payload into the result: only these record what was placed and where."""


@dataclass(frozen=True, init=False)
class InjectionTarget:
    """An exact record selector within a channel.

    ``call_arguments`` restricts a tool-output target to the calls of
    ``tool`` whose arguments match: each named top-level argument must be
    present and equal, type for type (``"1"`` isn't ``1``, ``True`` isn't
    ``1``, ``1.0`` isn't ``1``), to its JSON scalar value. It is given as
    ``(name, value)`` pairs or a mapping (``{"order_id": "ORD-102"}``), and
    always read as the pairs sorted by name. A call to ``tool`` whose
    arguments don't match isn't an occurrence of the instruction: it isn't
    counted, and its result is left unchanged with an ``unmatched`` outcome
    that names the argument.

    ``result_record`` restricts it the same way by the result instead: the
    call's structured content must be an object whose ``field`` exactly
    equals ``value``, by the list record selector's rules. It is for a tool
    that returns one record whose id is in the result only
    (``get_referral(patient_id)`` answering ``{"referral_id": "REF-1", ...}``);
    when the id is an argument of the call, use ``call_arguments``. Both may
    be given, and both must match. A result with no structured content, or
    structured content that isn't an object, never matches, whatever the
    target's surface: a content-block target is decided by the structured
    content too.
    """

    tool: str | None
    surface: SelectorSurface
    content_block_index: int | None
    content_block_type: str | None
    structured_content_path: FieldPath
    template_path: FieldPath
    template_key: str | None
    call_arguments: CallArguments
    result_record: RecordSelector | None

    def __init__(
        self,
        tool: str | None = None,
        surface: SelectorSurface = SelectorSurface.CONTENT_BLOCK,
        content_block_index: int | None = None,
        content_block_type: str | None = None,
        structured_content_path: FieldPath = (),
        template_path: FieldPath = (),
        template_key: str | None = None,
        call_arguments: CallArgumentsInput = (),
        result_record: RecordSelector | None = None,
    ) -> None:
        # Written out, not generated, so that it takes either form of call_arguments while the attribute is typed
        # as the pairs it always holds.
        object.__setattr__(self, "tool", tool)
        object.__setattr__(self, "surface", surface)
        object.__setattr__(self, "content_block_index", content_block_index)
        object.__setattr__(self, "content_block_type", content_block_type)
        object.__setattr__(self, "structured_content_path", structured_content_path)
        object.__setattr__(self, "template_path", template_path)
        object.__setattr__(self, "template_key", template_key)
        if type(self.surface) is not SelectorSurface:
            raise TypeError("surface must be a SelectorSurface")
        _optional_str(self.tool, "target tool")
        object.__setattr__(self, "call_arguments", _call_arguments(call_arguments))
        if self.call_arguments:
            if self.surface not in {SelectorSurface.CONTENT_BLOCK, SelectorSurface.STRUCTURED_CONTENT}:
                raise ValueError("call_arguments only select tool-output calls")
            if self.tool is None:
                raise ValueError("call_arguments need a target tool")
        object.__setattr__(self, "result_record", result_record)
        if self.result_record is not None:
            if type(self.result_record) is not RecordSelector:
                raise TypeError("result_record must be a RecordSelector or None")
            if self.surface not in {SelectorSurface.CONTENT_BLOCK, SelectorSurface.STRUCTURED_CONTENT}:
                raise ValueError("result_record only selects tool-output results")
            if self.tool is None:
                raise ValueError("result_record needs a target tool")
        _optional_str(self.content_block_type, "content block type")
        _optional_str(self.template_key, "template key")
        if self.content_block_index is not None and (
            type(self.content_block_index) is not int or self.content_block_index < 0
        ):
            raise ValueError("content_block_index must be a non-negative int")
        object.__setattr__(self, "structured_content_path", _path(self.structured_content_path, "field path"))
        object.__setattr__(self, "template_path", _path(self.template_path, "template path", record_selectors=False))
        content_selector = self.content_block_index is not None or self.content_block_type is not None
        template_selector = bool(self.template_path) or self.template_key is not None
        if self.surface is SelectorSurface.CONTENT_BLOCK and (self.structured_content_path or template_selector):
            raise ValueError("content_block selectors cannot include structured-content or template fields")
        if self.surface is SelectorSurface.STRUCTURED_CONTENT and (content_selector or template_selector):
            raise ValueError("structured_content selectors cannot include content-block or template fields")
        if self.surface is SelectorSurface.DESCRIPTION and (
            content_selector or self.structured_content_path or template_selector
        ):
            raise ValueError(
                "description selectors cannot include content-block, structured-content, or template fields"
            )
        if self.surface is SelectorSurface.TEMPLATE and (content_selector or self.structured_content_path):
            raise ValueError("template selectors cannot include content-block or structured-content fields")

    def call_argument_mismatch(self, arguments: Mapping[str, Any]) -> str | None:
        """Why a call with ``arguments`` doesn't match ``call_arguments``, or None when it does."""
        problems = []
        for name, expected in self.call_arguments:
            if name not in arguments:
                problems.append(f"call argument {name!r} is missing")
            elif not _same_scalar(arguments[name], expected):
                problems.append(f"call argument {name!r} is {arguments[name]!r}, not {expected!r}")
        return "; ".join(problems) or None

    def result_record_mismatch(self, structured_content: Any) -> str | None:
        """Why a result with ``structured_content`` doesn't match ``result_record``, or None when it does."""
        if self.result_record is None:
            return None
        name, expected = self.result_record.field, self.result_record.value
        if structured_content is None:
            return f"result has no structured content to match field {name!r}"
        if not isinstance(structured_content, dict):
            kind = type(structured_content).__name__
            return f"result structured content is {kind}, not an object with field {name!r}"
        if name not in structured_content:
            return f"result field {name!r} is missing"
        if not _same_scalar(structured_content[name], expected):
            return f"result field {name!r} is {structured_content[name]!r}, not {expected!r}"
        return None


@dataclass(frozen=True)
class OccurrenceRule:
    """Which matching calls receive an instruction; ``nth`` is one-based."""

    kind: Occurrence = Occurrence.FIRST
    nth: int | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not Occurrence:
            raise TypeError("kind must be an Occurrence")
        if self.kind is Occurrence.NTH:
            if type(self.nth) is not int or self.nth < 1:
                raise ValueError("an nth occurrence needs a positive one-based nth")
        elif self.nth is not None:
            raise ValueError("nth is only valid for the nth occurrence rule")

    def selects(self, occurrence: int) -> bool:
        if self.kind is Occurrence.EVERY:
            return True
        return occurrence == (1 if self.kind is Occurrence.FIRST else self.nth)


@dataclass(frozen=True)
class InjectionInstruction:
    """One immutable placement instruction.

    ``probe_key`` is optional provenance only. ``size_limit`` limits the
    transformed payload's UTF-8 bytes; ``None`` explicitly means no limit.
    """

    channel: Channel
    payload: str
    target: InjectionTarget = field(default_factory=InjectionTarget)
    occurrence: OccurrenceRule = field(default_factory=OccurrenceRule)
    transformation: Transformation = Transformation.IDENTITY
    size_limit: int | None = None
    mismatch_policy: MismatchPolicy = MismatchPolicy.STRICT
    mode: InjectionMode = InjectionMode.APPEND
    probe_key: str | None = None

    def __post_init__(self) -> None:
        exact = (
            (self.channel, Channel),
            (self.target, InjectionTarget),
            (self.occurrence, OccurrenceRule),
            (self.transformation, Transformation),
            (self.mismatch_policy, MismatchPolicy),
            (self.mode, InjectionMode),
        )
        if invalid := [expected.__name__ for value, expected in exact if type(value) is not expected]:
            raise TypeError(f"instruction fields need exact plan types: {invalid}")
        _str(self.payload, "payload")
        _optional_str(self.probe_key, "probe_key")
        if self.size_limit is not None and (type(self.size_limit) is not int or self.size_limit < 0):
            raise ValueError("size_limit must be a non-negative int or None")
        if self.channel is Channel.TOOL_OUTPUT:
            if self.target.surface not in {SelectorSurface.CONTENT_BLOCK, SelectorSurface.STRUCTURED_CONTENT}:
                raise ValueError("tool_output needs a content_block or structured_content selector")
            if self.target.surface is SelectorSurface.STRUCTURED_CONTENT and not self.target.structured_content_path:
                raise ValueError("structured_content needs a non-empty field path")
        elif self.channel is Channel.TOOL_DESCRIPTION:
            if self.target.surface is not SelectorSurface.DESCRIPTION:
                raise ValueError("tool_description needs a description selector")
        elif (
            self.target.surface is not SelectorSurface.TEMPLATE
            or self.target.template_key is None
            or self.target.tool is not None
        ):
            raise ValueError(f"{self.channel.value} needs a template selector and template_key, without a tool")


DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
"""The form of a payload digest: ``sha256:`` and the SHA-256 in lowercase hex."""


def payload_digest(text: str) -> str:
    """``sha256:<hex>`` of ``text``'s UTF-8 bytes: how an outcome names a payload it placed."""
    return f"sha256:{hashlib.sha256(_str(text, 'payload').encode()).hexdigest()}"


@dataclass(frozen=True)
class InjectionSlot:
    """Where a tool-output instruction places its payload: its target and its mode.

    A call's outcome records the slot of the instruction it placed, so the
    sealed record says where a payload went without the plan (a generated
    payload's included). :meth:`to_data` gives ``{"id", "target", "mode"}``:
    ``target`` in the plan codec's form (the instruction's ``target`` in
    :meth:`InjectionPlan.to_data`, record selectors as
    ``{"field": ..., "equals": ...}``), ``mode`` the mode's value, and ``id``
    the target's :attr:`id`. The same instruction of a fixed plan gives
    the same data, so a consumer compares a recorded slot with a plan's.
    """

    target: InjectionTarget
    mode: InjectionMode = InjectionMode.APPEND

    def __post_init__(self) -> None:
        # A tool-output instruction with this target and mode validates the slot.
        InjectionInstruction(Channel.TOOL_OUTPUT, "", self.target, mode=self.mode)

    @classmethod
    def of(cls, instruction: InjectionInstruction) -> InjectionSlot:
        """The slot of a tool-output ``instruction``."""
        return cls(instruction.target, instruction.mode)

    @property
    def id(self) -> str:
        """``sha256:<hex>`` of the target's plan-codec JSON (sorted keys, compact, UTF-8): one ID per place, any mode."""
        encoded = json.dumps(_target_to_data(self.target), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return f"sha256:{hashlib.sha256(encoded.encode()).hexdigest()}"

    def to_data(self) -> dict[str, Any]:
        """The slot as fresh JSON data: ``{"id": ..., "target": ..., "mode": ...}``."""
        return {"id": self.id, "target": _target_to_data(self.target), "mode": self.mode.value}

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> InjectionSlot:
        """Decode :meth:`to_data`'s form; ``id`` may be left out, but one given must be the target's."""
        try:
            value = _object(data, "slot")
            _only_fields(value, {"id", "target", "mode"}, "slot")
            if "target" not in value or "mode" not in value:
                raise ValueError("slot must contain target and mode")
            slot = cls(_target_from_data(value["target"]), InjectionMode(value["mode"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid injection slot data: {exc}") from None
        if "id" in value and value["id"] != slot.id:
            raise ValueError(f"invalid injection slot data: id {value['id']!r} is not its target's {slot.id!r}")
        return slot


@dataclass(frozen=True)
class GeneratorPermissions:
    """Tools, structured fields, and modes generator output may name."""

    tools: frozenset[str] = frozenset()
    fields: frozenset[FieldPath] = frozenset()
    modes: frozenset[InjectionMode] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", frozenset(_str(item, "permitted tool") for item in self.tools))
        object.__setattr__(self, "fields", frozenset(_path(item, "permitted field") for item in self.fields))
        modes = frozenset(self.modes)
        if any(type(mode) is not InjectionMode for mode in modes):
            raise TypeError("permitted modes must be InjectionMode values")
        object.__setattr__(self, "modes", modes)


@dataclass(frozen=True)
class InjectionPlan:
    """An immutable ordered collection of instructions and generator bounds.

    ``probe_payloads`` is the host's substitution map. It may contain active
    probes not referenced by a template in this episode, but every template
    instruction must name an entry with the same payload. Tool-output and
    tool-description instructions are carried for an external interceptor;
    select that interceptor's part with :meth:`for_channels`, which will then
    validate the selected instructions against its capability.
    """

    instructions: tuple[InjectionInstruction, ...] = ()
    generator_permissions: GeneratorPermissions = field(default_factory=GeneratorPermissions)
    probe_payloads: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "instructions", tuple(self.instructions))
        if not all(type(item) is InjectionInstruction for item in self.instructions):
            raise TypeError("every plan item must be an InjectionInstruction")
        if type(self.generator_permissions) is not GeneratorPermissions:
            raise TypeError("generator_permissions must be exact GeneratorPermissions")
        probes = tuple((_str(key, "probe key"), _str(payload, "probe payload")) for key, payload in self.probe_payloads)
        if len(dict(probes)) != len(probes):
            raise ValueError("probe_payloads contains a duplicate key")
        probes = tuple(sorted(probes))
        object.__setattr__(self, "probe_payloads", probes)
        substitutions = dict(probes)
        for instruction in self.instructions:
            if instruction.target.surface is not SelectorSurface.TEMPLATE:
                continue
            key = instruction.target.template_key
            assert key is not None  # InjectionInstruction validates template selectors.
            if key not in substitutions:
                raise ValueError(f"template instruction {key!r} is missing from probe_payloads")
            if substitutions[key] != instruction.payload:
                raise ValueError(f"template instruction {key!r} disagrees with probe_payloads")
            if instruction.probe_key is not None and instruction.probe_key != key:
                raise ValueError(f"template instruction {key!r} must use the same probe_key")

    def __iter__(self) -> Iterator[InjectionInstruction]:
        return iter(self.instructions)

    def __len__(self) -> int:
        return len(self.instructions)

    @property
    def substitution_payloads(self) -> Mapping[str, str]:
        """The immutable host substitution map, validated against template instructions."""
        return MappingProxyType(dict(self.probe_payloads))

    def for_channels(self, *channels: Channel) -> InjectionPlan:
        """Return the part delivered by an interceptor responsible for ``channels``."""
        if any(type(channel) is not Channel for channel in channels):
            raise TypeError("channels must be exact Channel values")
        selected = frozenset(channels)
        return InjectionPlan(
            tuple(instruction for instruction in self.instructions if instruction.channel in selected),
            generator_permissions=self.generator_permissions,
            probe_payloads=self.probe_payloads,
        )

    def to_json(self) -> str:
        """Encode this plan as stable JSON made only from plain JSON values.

        Key and index path steps remain JSON strings and integers. A record
        selector is ``{"field": "id", "equals": "NOTE-1"}``; ``from_json``
        also accepts an omitted ``field``, which defaults to ``id``. A
        target's ``call_arguments`` is a JSON object of argument names to
        scalar values, ``{"order_id": "ORD-102"}``, left out when empty, and
        its ``result_record`` a record selector,
        ``{"field": "referral_id", "equals": "REF-1"}``, left out when None.
        """
        return json.dumps(self.to_data(), sort_keys=True, separators=(",", ":"))

    def to_data(self) -> dict[str, Any]:
        """Encode this plan as fresh dict/list JSON data."""
        return _plan_to_data(self)

    @classmethod
    def from_json(cls, value: str | bytes | bytearray) -> InjectionPlan:
        """Decode the plain-data JSON representation produced by :meth:`to_json`."""
        try:
            data = json.loads(value)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid injection plan JSON: {exc}") from None
        return cls.from_data(data)

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> InjectionPlan:
        """Decode a plan from plain dict/list JSON data."""
        try:
            return _plan_from_data(data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid injection plan data: {exc}") from None


@dataclass(frozen=True)
class InterceptorCapability:
    """What an interceptor places exactly.

    ``call_arguments`` says it matches a target's call arguments, and
    ``result_record`` that it matches a target's result record.
    """

    channels: frozenset[Channel]
    selector_surfaces: frozenset[SelectorSurface]
    content_block_types: frozenset[str]
    modes: frozenset[InjectionMode]
    transformations: frozenset[Transformation]
    mismatch_policies: frozenset[MismatchPolicy]
    call_arguments: bool = False
    result_record: bool = False


@dataclass(frozen=True)
class CapabilityGap:
    instruction_index: int
    aspect: str
    value: str


class CapabilityError(ValueError):
    """A plan cannot be executed exactly by an interceptor."""

    def __init__(self, gaps: Sequence[CapabilityGap]) -> None:
        self.gaps = tuple(gaps)
        detail = ", ".join(f"instruction {g.instruction_index} {g.aspect}={g.value!r}" for g in self.gaps)
        super().__init__(f"injection plan exceeds interceptor capability: {detail}")


@dataclass(frozen=True)
class InjectionOutcome:
    """How one instruction fared on one call, with the delivery facts of the interceptor's own boundary.

    ``selected`` is whether the instruction's exact selector matched a place
    in the call's result. ``transformation`` is the transformation applied to
    the payload, None when the payload was never transformed (the occurrence
    rule skipped the call, the call's arguments didn't match the target's
    ``call_arguments`` or its result the target's ``result_record``, or there
    was no result). ``emitted`` names the
    parts of the result the interceptor returned that carry the transformed
    payload where it placed it (``content_block``, ``structured_content``),
    checked on the result as returned, after every instruction for the call:
    empty when none does. These are facts about what the interceptor sent,
    not about what the agent or its model saw.

    Occurrence 0 is only for a call whose arguments didn't match the target's
    ``call_arguments``, or whose result didn't match its ``result_record``
    (an upstream that produced no result included): an ``unmatched`` outcome
    that placed nothing, so
    building any other at occurrence 0 raises ValueError. ``unmatched`` is
    also counted (a strict selector that didn't match is occurrence 1 or more).

    What was placed, and where, is set only for a placement (``applied``,
    ``partial``, ``fallback``; building any other with one raises
    ValueError): ``payload_digest`` is :func:`payload_digest` of the
    instruction's payload before its transformation, ``placed_digest`` of the
    transformed text the instruction wrote (the same for ``identity``; the
    text a JSON mirror re-serialises around it isn't digested), and ``slot``
    the instruction's :class:`InjectionSlot`, its target and mode as planned
    (a ``fallback`` wrote a new text block instead, as ``detail`` says).
    """

    instruction_index: int
    probe_key: str | None
    scope: str
    call_id: str
    tool: str
    occurrence: int
    admission_order: int
    status: OutcomeStatus
    detail: str
    selected: bool = False
    transformation: Transformation | None = None
    emitted: tuple[SelectorSurface, ...] = ()
    payload_digest: str | None = None
    placed_digest: str | None = None
    slot: InjectionSlot | None = None

    def __post_init__(self) -> None:
        for digest in (self.payload_digest, self.placed_digest):
            if digest is not None and (type(digest) is not str or not DIGEST_PATTERN.fullmatch(digest)):
                raise ValueError(f"a payload digest is 'sha256:' and 64 lowercase hex digits, not {digest!r}")
        if self.slot is not None and type(self.slot) is not InjectionSlot:
            raise TypeError("slot must be an InjectionSlot or None")
        placement = (self.payload_digest, self.placed_digest, self.slot)
        if self.occurrence == 0 and (
            self.status != OutcomeStatus.UNMATCHED
            or self.selected
            or self.transformation is not None
            or self.emitted
            or placement != (None, None, None)
        ):
            raise ValueError("only an unmatched outcome that placed nothing has occurrence 0")
        if self.status not in PLACED and placement != (None, None, None):
            raise ValueError(f"a {self.status.value} outcome placed nothing: it has no payload digest or slot")


def capability_gaps(
    plan: InjectionPlan, capability: InterceptorCapability, *, tools: Collection[str] | None = None
) -> tuple[CapabilityGap, ...]:
    """Return every unsupported part of ``plan``."""
    gaps: list[CapabilityGap] = []
    for index, instruction in enumerate(plan):
        checks = (
            ("channel", instruction.channel, capability.channels),
            ("selector", instruction.target.surface, capability.selector_surfaces),
            ("mode", instruction.mode, capability.modes),
            ("transformation", instruction.transformation, capability.transformations),
            ("mismatch_policy", instruction.mismatch_policy, capability.mismatch_policies),
        )
        for aspect, value, supported in checks:
            if value not in supported:
                gaps.append(CapabilityGap(index, aspect, value.value))
        block_type = instruction.target.content_block_type
        if block_type is not None and block_type not in capability.content_block_types:
            gaps.append(CapabilityGap(index, "content_block_type", block_type))
        if tools is not None and instruction.target.tool is not None and instruction.target.tool not in tools:
            gaps.append(CapabilityGap(index, "tool", instruction.target.tool))
        if instruction.target.call_arguments and not capability.call_arguments:
            names = ",".join(name for name, _ in instruction.target.call_arguments)
            gaps.append(CapabilityGap(index, "call_arguments", names))
        if instruction.target.result_record is not None and not capability.result_record:
            gaps.append(CapabilityGap(index, "result_record", instruction.target.result_record.field))
    return tuple(gaps)


class GeneratorInstructionError(ValueError):
    """Generator output names something outside an episode's permissions."""


def instruction_from_generator(
    output: Mapping[str, Any], permissions: GeneratorPermissions, *, probe_key: str | None = None
) -> InjectionInstruction:
    """Validate constrained generator output and build an instruction."""
    unknown = set(output) - {"payload", "tool", "field", "mode"}
    if unknown:
        raise GeneratorInstructionError(f"unknown generator fields: {sorted(unknown)}")
    try:
        payload, tool, raw_field, raw_mode = (output[key] for key in ("payload", "tool", "field", "mode"))
        _str(payload, "generated payload")
        _str(tool, "generated tool")
        selected_field = _path(raw_field, "generated field")
        mode = InjectionMode(raw_mode)
    except KeyError as exc:
        raise GeneratorInstructionError(f"missing generator field: {exc.args[0]}") from None
    except (TypeError, ValueError) as exc:
        raise GeneratorInstructionError(str(exc)) from None
    if tool not in permissions.tools:
        raise GeneratorInstructionError(f"tool {tool!r} is not permitted")
    if selected_field not in permissions.fields:
        raise GeneratorInstructionError(f"field {selected_field!r} is not permitted")
    if mode not in permissions.modes:
        raise GeneratorInstructionError(f"mode {mode.value!r} is not permitted")
    return InjectionInstruction(
        channel=Channel.TOOL_OUTPUT,
        payload=payload,
        target=InjectionTarget(
            tool=tool, surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=selected_field
        ),
        mode=mode,
        probe_key=probe_key,
    )


def instruction_from_generator_for_plan(
    output: Mapping[str, Any], plan: InjectionPlan, *, probe_key: str | None = None
) -> InjectionInstruction:
    """Build from generator output constrained by an episode's plan permissions."""
    if type(plan) is not InjectionPlan:
        raise TypeError("plan must be an InjectionPlan")
    return instruction_from_generator(output, plan.generator_permissions, probe_key=probe_key)


def _path(value: Any, what: str, *, record_selectors: bool = True) -> FieldPath:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{what} must be a sequence")
    parts: list[PathPart] = []
    for part in value:
        if type(part) is dict:
            if not record_selectors:
                raise TypeError(f"{what} cannot contain a record selector")
            parts.append(_selector_from_data(part))
        else:
            parts.append(part)
    allowed = {str, int, RecordSelector} if record_selectors else {str, int}
    if any(type(part) not in allowed or (type(part) is int and part < 0) for part in parts):
        suffix = ", or RecordSelector" if record_selectors else ""
        raise TypeError(f"{what} parts must be exact str, non-negative int{suffix} values")
    return tuple(parts)


def _str(value: Any, what: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{what} must be a str")
    return value


def _optional_str(value: Any, what: str) -> None:
    if value is not None:
        _str(value, what)


_ARGUMENT_TYPES = (str, int, float, bool, type(None))


def _call_arguments(value: Any) -> CallArguments:
    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("call_arguments must be a mapping or a sequence of (name, value) pairs")
    else:
        items = []
        for item in value:
            if not isinstance(item, Sequence) or isinstance(item, str) or len(item) != 2:
                raise TypeError("each call argument must be a (name, value) pair")
            items.append((item[0], item[1]))
    for name, argument in items:
        if type(name) is not str or not name:
            raise TypeError("call argument names must be non-empty str values")
        if type(argument) not in _ARGUMENT_TYPES:
            raise TypeError(f"call argument {name!r} must be an exact str, int, float, bool or None")
        if type(argument) is float and not math.isfinite(argument):
            raise ValueError(f"call argument {name!r} must be a finite float")
    if len({name for name, _ in items}) != len(items):
        raise ValueError("call_arguments names an argument twice")
    return tuple(sorted(items, key=lambda item: item[0]))


def _same_scalar(actual: Any, expected: ArgumentValue) -> bool:
    return type(actual) is type(expected) and actual == expected


def _path_to_data(path: FieldPath) -> list[str | int | dict[str, str | int]]:
    encoded: list[str | int | dict[str, str | int]] = []
    for part in path:
        if isinstance(part, RecordSelector):
            encoded.append(_selector_to_data(part))
        else:
            encoded.append(part)
    return encoded


def _selector_to_data(selector: RecordSelector) -> dict[str, str | int]:
    return {"field": selector.field, "equals": selector.value}


def _path_from_data(value: Any, what: str, *, record_selectors: bool = True) -> FieldPath:
    if type(value) is not list:
        raise TypeError(f"{what} must be a JSON array")
    return _path(value, what, record_selectors=record_selectors)


def _selector_from_data(value: dict[str, Any]) -> RecordSelector:
    _only_fields(value, {"field", "equals"}, "record selector")
    if "equals" not in value:
        raise ValueError("record selector must contain equals")
    return RecordSelector(value["equals"], value.get("field", "id"))


def _plan_to_data(plan: InjectionPlan) -> dict[str, Any]:
    instructions = []
    for item in plan.instructions:
        instructions.append(
            {
                "channel": item.channel.value,
                "payload": item.payload,
                "target": _target_to_data(item.target),
                "occurrence": {"kind": item.occurrence.kind.value, "nth": item.occurrence.nth},
                "transformation": item.transformation.value,
                "size_limit": item.size_limit,
                "mismatch_policy": item.mismatch_policy.value,
                "mode": item.mode.value,
                "probe_key": item.probe_key,
            }
        )
    permission_fields = [_path_to_data(path) for path in plan.generator_permissions.fields]
    permission_fields.sort(key=lambda path: json.dumps(path, sort_keys=True, separators=(",", ":")))
    return {
        "instructions": instructions,
        "generator_permissions": {
            "tools": sorted(plan.generator_permissions.tools),
            "fields": permission_fields,
            "modes": sorted(mode.value for mode in plan.generator_permissions.modes),
        },
        "probe_payloads": [list(item) for item in plan.probe_payloads],
    }


def _plan_from_data(value: Any) -> InjectionPlan:
    data = _object(value, "injection plan")
    _only_fields(data, {"instructions", "generator_permissions", "probe_payloads"}, "injection plan")
    if "instructions" not in data:
        raise ValueError("injection plan must contain instructions")
    raw_instructions = data["instructions"]
    if type(raw_instructions) is not list:
        raise TypeError("instructions must be a JSON array")
    permissions_data = _object(data.get("generator_permissions", {}), "generator_permissions")
    _only_fields(permissions_data, {"tools", "fields", "modes"}, "generator_permissions")
    tools = permissions_data.get("tools", [])
    fields = permissions_data.get("fields", [])
    modes = permissions_data.get("modes", [])
    if type(tools) is not list or type(fields) is not list or type(modes) is not list:
        raise TypeError("generator permission tools, fields, and modes must be JSON arrays")
    permissions = GeneratorPermissions(
        tools=frozenset(tools),
        fields=frozenset(_path_from_data(path, "permitted field") for path in fields),
        modes=frozenset(InjectionMode(mode) for mode in modes),
    )
    raw_probes = data.get("probe_payloads", [])
    if type(raw_probes) is not list:
        raise TypeError("probe_payloads must be a JSON array")
    probes = []
    for item in raw_probes:
        if type(item) is not list or len(item) != 2:
            raise TypeError("each probe_payloads entry must be a two-item JSON array")
        probes.append((item[0], item[1]))
    return InjectionPlan(
        tuple(_instruction_from_data(item) for item in raw_instructions),
        generator_permissions=permissions,
        probe_payloads=tuple(probes),
    )


def _instruction_from_data(value: Any) -> InjectionInstruction:
    data = _object(value, "instruction")
    _only_fields(
        data,
        {
            "channel",
            "payload",
            "target",
            "occurrence",
            "transformation",
            "size_limit",
            "mismatch_policy",
            "mode",
            "probe_key",
        },
        "instruction",
    )
    if "channel" not in data or "payload" not in data:
        raise ValueError("instruction must contain channel and payload")
    target = _target_from_data(data.get("target", {}))
    occurrence_data = _object(data.get("occurrence", {}), "occurrence")
    _only_fields(occurrence_data, {"kind", "nth"}, "occurrence")
    occurrence = OccurrenceRule(
        Occurrence(occurrence_data.get("kind", Occurrence.FIRST.value)), occurrence_data.get("nth")
    )
    return InjectionInstruction(
        channel=Channel(data["channel"]),
        payload=data["payload"],
        target=target,
        occurrence=occurrence,
        transformation=Transformation(data.get("transformation", Transformation.IDENTITY.value)),
        size_limit=data.get("size_limit"),
        mismatch_policy=MismatchPolicy(data.get("mismatch_policy", MismatchPolicy.STRICT.value)),
        mode=InjectionMode(data.get("mode", InjectionMode.APPEND.value)),
        probe_key=data.get("probe_key"),
    )


def _target_to_data(target: InjectionTarget) -> dict[str, Any]:
    data: dict[str, Any] = {
        "tool": target.tool,
        "surface": target.surface.value,
        "content_block_index": target.content_block_index,
        "content_block_type": target.content_block_type,
        "structured_content_path": _path_to_data(target.structured_content_path),
        "template_path": _path_to_data(target.template_path),
        "template_key": target.template_key,
    }
    # Each left out when empty, so a target without them encodes as before.
    if target.call_arguments:
        data["call_arguments"] = dict(target.call_arguments)
    if target.result_record is not None:
        data["result_record"] = _selector_to_data(target.result_record)
    return data


def _target_from_data(value: Any) -> InjectionTarget:
    target_data = _object(value, "target")
    _only_fields(
        target_data,
        {
            "tool",
            "surface",
            "content_block_index",
            "content_block_type",
            "structured_content_path",
            "template_path",
            "template_key",
            "call_arguments",
            "result_record",
        },
        "target",
    )
    return InjectionTarget(
        tool=target_data.get("tool"),
        surface=SelectorSurface(target_data.get("surface", SelectorSurface.CONTENT_BLOCK.value)),
        content_block_index=target_data.get("content_block_index"),
        content_block_type=target_data.get("content_block_type"),
        structured_content_path=_path_from_data(
            target_data.get("structured_content_path", []), "structured content path"
        ),
        template_path=_path_from_data(target_data.get("template_path", []), "template path", record_selectors=False),
        template_key=target_data.get("template_key"),
        call_arguments=_object(target_data.get("call_arguments", {}), "call_arguments"),
        result_record=(
            _selector_from_data(_object(target_data["result_record"], "result_record"))
            if "result_record" in target_data
            else None
        ),
    )


def _object(value: Any, what: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise TypeError(f"{what} must be a JSON object")
    return value


def _only_fields(value: Mapping[str, Any], allowed: set[str], what: str) -> None:
    if unknown := set(value) - allowed:
        raise ValueError(f"{what} has unknown fields: {sorted(unknown)}")
