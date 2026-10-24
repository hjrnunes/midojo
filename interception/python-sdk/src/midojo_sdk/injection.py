"""Immutable injection plans shared by hosts and interception layers.

``append`` is the one default mode: for an unqualified content selector it
preserves upstream data and has an unambiguous meaning. Specific placements
never guess another field.
"""

from __future__ import annotations

import enum
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

type PathPart = str | int
type FieldPath = tuple[PathPart, ...]


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
    APPLIED = "applied"
    FALLBACK = "fallback"
    UNMATCHED = "unmatched"
    TOO_LARGE = "too_large"
    SKIPPED_OCCURRENCE = "skipped_occurrence"


@dataclass(frozen=True)
class InjectionTarget:
    """An exact record selector within a channel."""

    tool: str | None = None
    surface: SelectorSurface = SelectorSurface.CONTENT_BLOCK
    content_block_index: int | None = None
    content_block_type: str | None = None
    structured_content_path: FieldPath = ()
    template_path: FieldPath = ()
    template_key: str | None = None

    def __post_init__(self) -> None:
        if type(self.surface) is not SelectorSurface:
            raise TypeError("surface must be a SelectorSurface")
        _optional_str(self.tool, "target tool")
        _optional_str(self.content_block_type, "content block type")
        _optional_str(self.template_key, "template key")
        if self.content_block_index is not None and (
            type(self.content_block_index) is not int or self.content_block_index < 0
        ):
            raise ValueError("content_block_index must be a non-negative int")
        object.__setattr__(self, "structured_content_path", _path(self.structured_content_path, "field path"))
        object.__setattr__(self, "template_path", _path(self.template_path, "template path"))
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
    """An immutable ordered collection of instructions and generator bounds."""

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
        object.__setattr__(self, "probe_payloads", probes)

    def __iter__(self) -> Iterator[InjectionInstruction]:
        return iter(self.instructions)

    def __len__(self) -> int:
        return len(self.instructions)


@dataclass(frozen=True)
class InterceptorCapability:
    channels: frozenset[Channel]
    selector_surfaces: frozenset[SelectorSurface]
    content_block_types: frozenset[str]
    modes: frozenset[InjectionMode]
    transformations: frozenset[Transformation]
    mismatch_policies: frozenset[MismatchPolicy]


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
    instruction_index: int
    probe_key: str | None
    call_id: str
    tool: str
    occurrence: int
    admission_order: int
    status: OutcomeStatus
    detail: str


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


def _path(value: Any, what: str) -> FieldPath:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{what} must be a sequence")
    parts = tuple(value)
    if any(type(part) is not str and not (type(part) is int and part >= 0) for part in parts):
        raise TypeError(f"{what} parts must be exact str or non-negative int values")
    return parts


def _str(value: Any, what: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{what} must be a str")
    return value


def _optional_str(value: Any, what: str) -> None:
    if value is not None:
        _str(value, what)
