"""Placement: what an injection plan does to a tool call's result, decided apart from the wire.

An interceptor forwards a call and records it; an engine decides what the
plan places. The engine sees a call as :class:`CallView` and the upstream's
result as :class:`ResultView` (None until the result is in, and before any
instruction changed it), and answers a :class:`PlacementDecision`: leave the
result as the upstream sent it, or replace it with a finished one. The
interceptor puts the finished result on the wire, tells the engine what it
actually emitted (:meth:`PlacementEngine.settle`, with an
:class:`EmittedView`), and records the outcomes the engine returns, so every
outcome a call's record carries comes from the engine.

The default engine, :class:`PlanPlacementEngine`, is the injection plan's
own: it matches a call's arguments and the result's record, counts
occurrences, applies the payload's transformation and size limit, splices the
text into the structured content (re-serialising a JSON mirror in the
upstream's layout) or into a content block, and applies the mismatch policy.

This module is stdlib only, plus :mod:`midojo_sdk.injection`: it holds no MCP
SDK, pydantic or ``midojo`` type, so a plan's placements can be decided and
checked without the interceptor's dependencies. The views below mirror the
fields of the MCP SDK's result and content blocks, and the result's copy API,
so the splice code reads here as it did on the wire types.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from midojo_sdk.injection import (
    PLACED,
    Channel,
    InjectionInstruction,
    InjectionMode,
    InjectionOutcome,
    InjectionPlan,
    InjectionSlot,
    InjectionTarget,
    MismatchPolicy,
    OutcomeStatus,
    RecordSelector,
    SelectorSurface,
    Transformation,
    payload_digest,
)

__all__ = [
    "LEAVE",
    "CallView",
    "ContentBlock",
    "EmittedView",
    "PlacementDecision",
    "PlacementEngine",
    "PlanPlacementEngine",
    "Planned",
    "ResultView",
    "TextContent",
    "leave",
    "place_field",
    "replace",
]


class _BlockMeta(type):
    """Recognises a text block by its shape rather than by its class.

    A view crosses a module boundary: the interceptor builds the views it
    hands an engine, and an engine can come from a reloaded copy of this
    module (a mutation harness re-executes it), whose ``TextContent`` is a
    different class with the same fields. A text block is therefore any block
    whose ``type`` is ``"text"`` and whose ``text`` is a string.
    """

    def __instancecheck__(cls, instance: object) -> bool:
        return getattr(instance, "type", None) == "text" and isinstance(getattr(instance, "text", None), str)


@dataclasses.dataclass(frozen=True)
class TextContent(metaclass=_BlockMeta):
    """A text content block: its text, and the wire block it came from, if any.

    The module's stand-in for the MCP SDK's ``TextContent``: placement reads
    and writes a result through these views, so it needs neither the SDK nor
    pydantic. A block placement created (an appended one, a fallback) has no
    ``wire`` block, and the interceptor builds it.
    """

    type: str = "text"
    text: str = ""
    wire: Any = None


@dataclasses.dataclass(frozen=True)
class ContentBlock:
    """A content block that isn't text: placement keeps it exactly as the wire gave it."""

    type: str
    wire: Any = None


@dataclasses.dataclass(frozen=True)
class ResultView:
    """A tool result as placement reads and rewrites it: its content blocks and its structured content.

    ``model_copy`` mirrors the MCP SDK result's copy method, which the splice
    code was written against: ``update`` names the fields to replace.
    """

    content: Sequence[TextContent | ContentBlock] = ()
    structured_content: Any = None
    is_error: bool = False

    def model_copy(self, *, update: Mapping[str, Any] | None = None) -> ResultView:
        """A copy with the fields ``update`` names replaced."""
        fields: dict[str, Any] = {
            "content": self.content,
            "structured_content": self.structured_content,
            "is_error": self.is_error,
        }
        fields.update(update or {})
        return ResultView(**fields)


@dataclasses.dataclass(frozen=True)
class CallView:
    """The call a decision is about: its tool, its arguments, its ID, and the counter scope it belongs to."""

    tool: str
    arguments: Mapping[str, Any]
    call_id: str
    scope: str = ""


@dataclasses.dataclass(frozen=True)
class EmittedView:
    """What the interceptor actually emitted for a call: the call, and the result it returned.

    ``result`` is None when the call produced none (the upstream failed), so
    nothing reached the agent.
    """

    call: CallView | None = None
    result: ResultView | None = None


@dataclasses.dataclass(frozen=True)
class PlacementDecision:
    """What an engine decided for one phase of one call.

    ``kind`` is ``leave`` (the result stays as the upstream sent it) or
    ``replace`` (``result`` is the finished result to put on the wire; a
    ``replace`` decision without one changes nothing). ``outcomes`` are what
    the call's record carries for this decision.
    """

    kind: Literal["leave", "replace"] = "leave"
    result: ResultView | None = None
    outcomes: tuple[InjectionOutcome, ...] = ()


LEAVE = PlacementDecision()
"""The decision that changes nothing."""


def leave(*outcomes: InjectionOutcome) -> PlacementDecision:
    """The decision that leaves the result as it is, reporting ``outcomes`` for the call."""
    return PlacementDecision(outcomes=tuple(outcomes)) if outcomes else LEAVE


def replace(result: ResultView, outcomes: Sequence[InjectionOutcome] = ()) -> PlacementDecision:
    """The decision that replaces the result with ``result``, reporting ``outcomes`` for the call."""
    return PlacementDecision(kind="replace", result=result, outcomes=tuple(outcomes))


@runtime_checkable
class PlacementEngine(Protocol):
    """What an interceptor asks about a call: its placements, and what it emitted.

    ``decide`` is called twice for a call that reaches the upstream: with no
    result, once the call's start is recorded, and then with the result the
    upstream returned, before any instruction changed it. What the second
    decision replaces is what the agent gets. ``settle`` is called once for
    the call, with what the interceptor emitted, and returns the call's
    outcomes in the order its record carries them; ``outcomes`` is every
    outcome the engine recorded.
    """

    def decide(self, call: CallView, result: ResultView | None) -> PlacementDecision: ...

    def settle(self, decision: PlacementDecision, emitted: EmittedView) -> tuple[InjectionOutcome, ...]: ...

    def outcomes(self) -> tuple[InjectionOutcome, ...]: ...


@runtime_checkable
class Planned(Protocol):
    """An engine built from a plan: an interceptor validates its instructions against what it can place."""

    plan: InjectionPlan


def place_field(
    result: ResultView,
    path: tuple[str | int | RecordSelector, ...],
    text: str,
    *,
    mode: InjectionMode = InjectionMode.APPEND,
) -> ResultView:
    """``result`` with ``text`` written at ``path`` in its structured content, its JSON mirror re-serialised.

    The pure part of a structured-content placement, for an engine of your
    own: it rewrites the structured content and the text that mirrors it, and
    leaves the text as sent when it doesn't mirror the structured content.
    Raises ValueError when ``path`` selects no string to write into.
    """
    target = InjectionTarget(surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=path)
    instruction = InjectionInstruction(channel=Channel.TOOL_OUTPUT, payload=text, target=target, mode=mode)
    updated, problem = _inject_structured(result, instruction, text)
    if updated is None:
        raise ValueError(problem)
    return updated


def _transform(instruction: InjectionInstruction) -> str:
    if instruction.transformation is Transformation.JSON_STRING:
        return json.dumps(instruction.payload)
    return instruction.payload


def _inject_path(
    value: Any, path: tuple[str | int | RecordSelector, ...], payload: str, mode: InjectionMode
) -> str | None:
    current = value
    for part in path[:-1]:
        current, mismatch = _resolve_path_part(current, part)
        if mismatch is not None:
            return mismatch
    return _inject_leaf(current, path[-1], payload, mode)


def _resolve_path_part(current: Any, part: str | int | RecordSelector) -> tuple[Any, str | None]:
    if type(part) is str:
        if not isinstance(current, dict):
            return None, f"field {part!r} requires an object, got {type(current).__name__}"
        if part not in current:
            return None, f"field {part!r} is missing"
        return current[part], None
    if type(part) is int:
        if not isinstance(current, list):
            return None, f"index {part} requires a list, got {type(current).__name__}"
        if part >= len(current):
            return None, f"index {part} is out of range"
        return current[part], None
    assert type(part) is RecordSelector
    label = f"[{part.field}={part.value!r}]"
    if not isinstance(current, list):
        return None, f"record selector {label} requires a list, got {type(current).__name__}"
    matches = [
        item
        for item in current
        if isinstance(item, dict)
        and part.field in item
        and type(item[part.field]) is type(part.value)
        and item[part.field] == part.value
    ]
    if not matches:
        return None, f"record selector {label} matched no records"
    if len(matches) != 1:
        return None, f"record selector {label} matched {len(matches)} records"
    return matches[0], None


def _inject_leaf(current: Any, part: str | int | RecordSelector, payload: str, mode: InjectionMode) -> str | None:
    if type(part) is str:
        if not isinstance(current, dict):
            return f"field {part!r} requires an object, got {type(current).__name__}"
        if part not in current:
            return f"field {part!r} is missing"
        if not isinstance(current[part], str):
            return f"field {part!r} does not contain a string"
        old = current[part]
        current[part] = payload if mode is InjectionMode.REPLACE else old + payload
        return None
    if type(part) is int:
        if not isinstance(current, list):
            return f"index {part} requires a list, got {type(current).__name__}"
        if part >= len(current):
            return f"index {part} is out of range"
        if not isinstance(current[part], str):
            return f"index {part} does not contain a string"
        old = current[part]
        current[part] = payload if mode is InjectionMode.REPLACE else old + payload
        return None
    assert isinstance(part, RecordSelector)
    _, mismatch = _resolve_path_part(current, part)
    return mismatch or f"record selector [{part.field}={part.value!r}] selects a record, not a string field"


def _inject_structured(
    result: ResultView, instruction: InjectionInstruction, payload: str
) -> tuple[ResultView | None, str | None]:
    """Place ``payload`` in the structured content, and in the text that mirrors it.

    Returns the updated result, or None with the mismatch. A result whose
    text content is a JSON mirror of its structured content gets that text
    re-serialised from the updated structured content; any other keeps its
    text, and the second value then says why the placement is only partial.
    """
    structured = copy.deepcopy(result.structured_content)
    mismatch = _inject_path(structured, instruction.target.structured_content_path, payload, instruction.mode)
    if mismatch is not None:
        return None, mismatch
    mirror = _json_mirror(result)
    if isinstance(mirror, str):
        return result.model_copy(update={"structured_content": structured}), mirror
    block, options = mirror
    content = [dataclasses.replace(block, text=json.dumps(structured, **options))]
    return result.model_copy(update={"structured_content": structured, "content": content}), None


def _json_mirror(result: ResultView) -> tuple[TextContent, dict[str, Any]] | str:
    """The text block that is the JSON serialisation of the structured content, and how to serialise it again.

    The mirror is the common MCP shape: exactly one content block, a text
    block whose text parses as JSON equal, type for type, to the structured
    content. A result without one gets the reason instead.
    """
    content = result.content
    if len(content) != 1 or not isinstance(content[0], TextContent):
        return f"text content is not a single text block ({len(content)} content blocks)"
    block = content[0]
    try:
        parsed = json.loads(block.text)
    except ValueError:
        return "text content is not JSON"
    if not _same_json(parsed, result.structured_content):
        return "text content is JSON that differs from the structured content"
    return block, _json_style(block.text, result.structured_content)


def _same_json(a: Any, b: Any) -> bool:
    """Whether two JSON values are equal type for type: ``1`` isn't ``1.0`` or ``true``."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_json(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same_json(x, y) for x, y in zip(a, b, strict=True))
    return a == b


_JSON_LAYOUTS: tuple[tuple[int | None, tuple[str, str]], ...] = (
    (None, (",", ":")),
    (None, (", ", ": ")),
    (2, (",", ": ")),
    (4, (",", ": ")),
)


def _json_style(text: str, value: Any) -> dict[str, Any]:
    """``json.dumps`` options for the injected value: the predefined style that reproduces ``text`` exactly, if any.

    The styles are compact, spaced, and two- and four-space indented layouts,
    each with and without ASCII escapes and with keys in the structured
    content's order or sorted; the first that gives exactly ``text`` from
    ``value`` is used. Nothing else is reproduced: not a key order of the
    text's own (other than sorted), trailing whitespace or a final newline,
    or another serialiser's number format. For such text it falls back to
    compact for one line, or indented by the width of the second line's
    leading spaces (two when it has none) otherwise, with keys in the
    structured content's order. Unescaped non-ASCII is tried first, so an
    all-ASCII original, which can't show which it used, and any fallback get
    a non-ASCII payload unescaped.
    """
    for indent, separators in _JSON_LAYOUTS:
        for ensure_ascii in (False, True):
            for sort_keys in (False, True):
                options = {
                    "indent": indent,
                    "separators": separators,
                    "ensure_ascii": ensure_ascii,
                    "sort_keys": sort_keys,
                }
                if json.dumps(value, **options) == text:
                    return options
    lines = text.splitlines()
    if len(lines) > 1:
        indent = len(lines[1]) - len(lines[1].lstrip(" ")) or 2
        return {"indent": indent, "separators": (",", ": "), "ensure_ascii": False}
    return {"separators": (",", ":"), "ensure_ascii": False}


def _inject_content(
    result: ResultView, instruction: InjectionInstruction, payload: str
) -> tuple[ResultView, _Edit] | None:
    target = instruction.target
    content = list(result.content)
    if target.content_block_index is None and target.content_block_type is None:
        block = TextContent(type="text", text=payload)
        if instruction.mode is InjectionMode.REPLACE:
            return result.model_copy(update={"content": [block]}), _Edit("replace_all", 0, 0, payload)
        edit = _Edit("append", len(content), 0, payload)
        return result.model_copy(update={"content": [*content, block]}), edit
    return _inject_selected_block(result, instruction, payload, content)


def _inject_selected_block(
    result: ResultView, instruction: InjectionInstruction, payload: str, content: list[TextContent | ContentBlock]
) -> tuple[ResultView, _Edit] | None:
    target = instruction.target
    indexes = range(len(content)) if target.content_block_index is None else (target.content_block_index,)
    for index in indexes:
        if index >= len(content):
            continue
        block = content[index]
        if target.content_block_type is not None and block.type != target.content_block_type:
            continue
        if not isinstance(block, TextContent):
            continue
        if instruction.mode is InjectionMode.REPLACE:
            text, edit = payload, _Edit("rewrite", index, 0, payload)
        else:
            text, edit = block.text + payload, _Edit("extend", index, len(block.text), payload)
        content[index] = dataclasses.replace(block, text=text)
        return result.model_copy(update={"content": content}), edit
    return None


@dataclasses.dataclass(frozen=True)
class _Edit:
    """How one instruction changed a result's content blocks, and the text it wrote where.

    ``kind`` is ``replace_all`` (new content, all of it), ``append`` (a block
    added at the end), ``rewrite`` (one block's text replaced) or ``extend``
    (text appended to one block's text, which stays its prefix). ``text`` was
    written at ``start`` in the text of the block at index ``block`` of the
    content after the edit.
    """

    kind: Literal["replace_all", "append", "rewrite", "extend"]
    block: int
    start: int
    text: str


# Where an instruction's text is: the key of the block it wrote into (see _track), the offset and the text.
_Written = tuple[object, int, str]

# A structured field as the path resolved it: object keys and list indexes, a record selector's included.
_Location = tuple[str | int, ...]


@dataclasses.dataclass(frozen=True)
class _FieldEdit:
    """How one instruction wrote a structured field: where, whether it replaced the field, and what at which offset."""

    location: _Location
    replaced: bool
    start: int
    text: str


# Where an instruction's payload is in the structured content: the field's key (see _track_field), its location,
# the offset and the text.
_FieldWritten = tuple[object, _Location, int, str]


@dataclasses.dataclass(frozen=True)
class _Placement:
    """What one instruction did to a call's result: its outcome, the transformed payload and where it put it.

    ``placed`` names where: ``structured_content``, ``mirror`` (the text
    re-serialised from the structured content) and ``content_block`` (a text
    block the instruction wrote or appended). ``edit`` is how it changed the
    content blocks, None when it left them as they were; ``field`` how it
    wrote the structured content, None when it didn't.
    """

    status: OutcomeStatus
    detail: str
    payload: str
    placed: tuple[str, ...] = ()
    edit: _Edit | None = None
    field: _FieldEdit | None = None


# The outcomes whose exact selector matched a place in the result.
_SELECTED = frozenset({OutcomeStatus.APPLIED, OutcomeStatus.PARTIAL})


def _inject_result(result: ResultView, instruction: InjectionInstruction) -> tuple[ResultView, _Placement]:
    payload = _transform(instruction)
    if instruction.size_limit is not None and len(payload.encode()) > instruction.size_limit:
        return result, _Placement(OutcomeStatus.TOO_LARGE, "transformed payload exceeds size_limit", payload)
    if instruction.target.surface is SelectorSurface.STRUCTURED_CONTENT:
        # Located before the write, on the structured content the write resolves its path in.
        field = _field_edit(result.structured_content, instruction, payload)
        updated, problem = _inject_structured(result, instruction, payload)
        if updated is not None and problem is not None:
            detail = f"exact selector matched structured content only; {problem}"
            return updated, _Placement(OutcomeStatus.PARTIAL, detail, payload, ("structured_content",), field=field)
        if updated is not None:
            detail = "exact selector matched; mirrored text re-serialised"
            [mirror] = updated.content
            assert isinstance(mirror, TextContent)  # _inject_structured re-serialised it.
            edit = _Edit("replace_all", 0, 0, mirror.text)
            placed = ("structured_content", "mirror")
            return updated, _Placement(OutcomeStatus.APPLIED, detail, payload, placed, edit, field)
        mismatch = problem
    else:
        injected = _inject_content(result, instruction, payload)
        if injected is not None:
            updated, edit = injected
            placement = _Placement(OutcomeStatus.APPLIED, "exact selector matched", payload, ("content_block",), edit)
            return updated, placement
        mismatch = "content-block selector did not match"
    if instruction.mismatch_policy is MismatchPolicy.FALLBACK_APPEND:
        content = [*result.content, TextContent(type="text", text=payload)]
        updated = result.model_copy(update={"content": content})
        detail = f"{mismatch}; appended a text block"
        edit = _Edit("append", len(content) - 1, 0, payload)
        return updated, _Placement(OutcomeStatus.FALLBACK, detail, payload, ("content_block",), edit)
    return result, _Placement(OutcomeStatus.UNMATCHED, f"{mismatch}; result unchanged", payload)


def _track(keys: list[object], edit: _Edit | None, size: int) -> tuple[list[object], _Written | None]:
    """The block keys after ``edit``, and where it wrote its text.

    A block keeps its key while its text is unchanged or only extended, so
    text written into it stays where it was written; a rewritten block, an
    appended one and all new content get new keys, so text written into a
    block that was since replaced is found nowhere.
    """
    if edit is None:
        return keys, None
    if edit.kind == "replace_all":
        keys = [object() for _ in range(size)]
    elif edit.kind == "append":
        keys = [*keys, object()]
    elif edit.kind == "rewrite":
        keys = [object() if index == edit.block else key for index, key in enumerate(keys)]
    assert len(keys) == size
    return keys, (keys[edit.block], edit.start, edit.text)


def _still_written(result: ResultView, keys: list[object], written: _Written | None) -> bool:
    """Whether the block ``written`` names is still in ``result`` and holds its text where it was written."""
    if written is None:
        return False
    key, start, text = written
    index = next((index for index, block_key in enumerate(keys) if block_key is key), None)
    if index is None:
        return False
    block = result.content[index]
    return isinstance(block, TextContent) and block.text[start : start + len(text)] == text


def _field_edit(structured: Any, instruction: InjectionInstruction, payload: str) -> _FieldEdit | None:
    """How writing ``payload`` at the instruction's path will change ``structured``; None when the path selects no string."""
    location: list[str | int] = []
    current = structured
    for part in instruction.target.structured_content_path:
        parent = current
        current, mismatch = _resolve_path_part(current, part)
        if mismatch is not None:
            return None
        if type(part) is RecordSelector:
            location.append(next(index for index, item in enumerate(parent) if item is current))
        else:
            assert type(part) is str or type(part) is int
            location.append(part)
    if not isinstance(current, str):
        return None
    if instruction.mode is InjectionMode.REPLACE:
        return _FieldEdit(tuple(location), True, 0, payload)
    return _FieldEdit(tuple(location), False, len(current), payload)


def _track_field(
    fields: dict[_Location, object], edit: _FieldEdit | None
) -> tuple[dict[_Location, object], _FieldWritten | None]:
    """The structured fields' keys after ``edit``, and where it wrote its payload.

    A field keeps its key while its text is only appended to, so a payload
    written into it stays where it was written; a replaced field gets a new
    key, so a payload written into it before is found nowhere, even when the
    replacement is the same text.
    """
    if edit is None:
        return fields, None
    if edit.replaced or edit.location not in fields:
        fields = {**fields, edit.location: object()}
    return fields, (fields[edit.location], edit.location, edit.start, edit.text)


def _field_still_written(result: ResultView, fields: dict[_Location, object], written: _FieldWritten | None) -> bool:
    """Whether the field ``written`` names still has its key and holds its payload where it was written."""
    if written is None:
        return False
    key, location, start, text = written
    if fields.get(location) is not key:
        return False
    current: Any = result.structured_content
    for part in location:
        if type(part) is str and isinstance(current, dict) and part in current:
            current = current[part]
        elif type(part) is int and isinstance(current, list) and part < len(current):
            current = current[part]
        else:
            return False
    return isinstance(current, str) and current[start : start + len(text)] == text


def _emitted(
    result: ResultView,
    keys: list[object],
    fields: dict[_Location, object],
    placement: _Placement,
    written: _Written | None,
    field_written: _FieldWritten | None,
) -> tuple[SelectorSurface, ...]:
    """The parts of the returned ``result`` that still carry the placement's payload where it was placed.

    Structured content carries it when the field the instruction wrote still
    holds the payload where it wrote it, with no later replace of that field
    in between. The text carries a mirrored placement when it is still a
    JSON mirror of structured content that carries it, or the block the
    instruction re-serialised still holds that text; and a text-block
    placement when the block it wrote into or appended still holds the
    payload where it wrote it. Only that place counts: the same text
    elsewhere in the result, or in a block or field a later instruction
    replaced, isn't this instruction's placement.
    """
    carried: list[SelectorSurface] = []
    structured = "structured_content" in placement.placed and _field_still_written(result, fields, field_written)
    if structured:
        carried.append(SelectorSurface.STRUCTURED_CONTENT)
    if "mirror" in placement.placed:
        mirrored = structured and not isinstance(_json_mirror(result), str)
        if mirrored or _still_written(result, keys, written):
            carried.append(SelectorSurface.CONTENT_BLOCK)
    elif "content_block" in placement.placed and _still_written(result, keys, written):
        carried.append(SelectorSurface.CONTENT_BLOCK)
    return tuple(carried)


def _placed(instruction: InjectionInstruction, placement: _Placement) -> dict[str, Any]:
    """What a placement wrote and where: the digests of the payload and of its transformed text, and its slot."""
    return {
        "payload_digest": payload_digest(instruction.payload),
        "placed_digest": payload_digest(placement.payload),
        "slot": InjectionSlot.of(instruction),
    }


def _outcome(
    scope: str,
    index: int,
    instruction: InjectionInstruction,
    call_id: str,
    tool: str,
    occurrence: int,
    admission_order: int,
    status: OutcomeStatus,
    detail: str,
    *,
    selected: bool = False,
    transformation: Transformation | None = None,
    emitted: tuple[SelectorSurface, ...] = (),
    payload_digest: str | None = None,
    placed_digest: str | None = None,
    slot: InjectionSlot | None = None,
) -> InjectionOutcome:
    return InjectionOutcome(
        instruction_index=index,
        probe_key=instruction.probe_key,
        scope=scope,
        call_id=call_id,
        tool=tool,
        occurrence=occurrence,
        admission_order=admission_order,
        status=status,
        detail=detail,
        selected=selected,
        transformation=transformation,
        emitted=emitted,
        payload_digest=payload_digest,
        placed_digest=placed_digest,
        slot=slot,
    )


type _AdmittedInstruction = tuple[str, int, InjectionInstruction, int, int]
# Scope, index, instruction and admission order of an instruction whose target's result_record decides, once the
# result is in, whether the call is an occurrence of it.
type _PendingInstruction = tuple[str, int, InjectionInstruction, int]

# One instruction's splice: the admission it belongs to, what it placed, and where its text is (see _track and
# _track_field).
type _PlacementRecord = tuple[_AdmittedInstruction, _Placement, _Written | None, _FieldWritten | None]


@dataclasses.dataclass
class _InFlight:
    """One call's admission, from its call-phase decision to its settlement."""

    admitted: list[_AdmittedInstruction]
    skipped: list[InjectionOutcome]
    pending: list[_PendingInstruction]
    # Set once the result phase applied its instructions: what a placement check reads afterwards.
    trail: _Trail | None = None


@dataclasses.dataclass(frozen=True)
class _Trail:
    """The result an instruction set finished, and where each instruction wrote into it."""

    result: ResultView
    keys: list[object]
    fields: dict[_Location, object]
    placements: list[_PlacementRecord]


class PlanPlacementEngine:
    """The default engine: one injection plan's placements, its occurrence counters and its outcomes.

    ``plan`` is the tool-output part of an episode's plan. ``evaluation_id``
    names the evaluation its occurrences belong to, for a call view that
    doesn't name its own scope; a new scope counts from zero, so a server
    that serves several evaluations keeps their occurrence rules apart.
    ``outcomes`` are the ones it recorded, ordered by call admission and then
    by instruction.
    """

    def __init__(self, plan: InjectionPlan = InjectionPlan(), *, evaluation_id: str | None = None) -> None:
        if type(plan) is not InjectionPlan:
            raise TypeError("plan must be an InjectionPlan")
        if evaluation_id is not None and (type(evaluation_id) is not str or not evaluation_id):
            raise ValueError("evaluation_id must be a non-empty str or None.")
        self._plan = plan
        self._evaluation_id = evaluation_id
        self._occurrences: dict[object, list[int]] = {}
        self._admissions: dict[object, int] = {}
        self._outcomes: list[InjectionOutcome] = []
        self._in_flight: dict[str, _InFlight] = {}
        # Counters and outcomes are shared by the tasks that serve the calls; each critical section is pure
        # Python with no await, so a thread lock is enough and the event loop is never blocked for long.
        self._lock = threading.Lock()

    @property
    def plan(self) -> InjectionPlan:
        """The instructions this engine places."""
        return self._plan

    def outcomes(self) -> tuple[InjectionOutcome, ...]:
        """Recorded outcomes, ordered by call admission then instruction."""
        with self._lock:
            return tuple(sorted(self._outcomes, key=lambda item: (item.admission_order, item.instruction_index)))

    def decide(self, call: CallView, result: ResultView | None) -> PlacementDecision:
        """The call's placements: which instructions apply, and the result with their text written into it."""
        scope = call.scope or self._evaluation_id or ""
        if result is None:
            admitted, skipped, pending = self._admit_injections(scope, call)
            self._in_flight[call.call_id] = _InFlight(admitted=admitted, skipped=skipped, pending=pending)
            return leave(*skipped)
        flight = self._in_flight.get(call.call_id)
        if flight is None:
            return LEAVE
        more, unmatched = self._admit_on_result(call, result.structured_content, flight.pending)
        admitted = sorted([*flight.admitted, *more], key=lambda admission: admission[1])
        return self._place(call, result, admitted, [*flight.skipped, *unmatched], flight)

    def settle(self, decision: PlacementDecision, emitted: EmittedView) -> tuple[InjectionOutcome, ...]:
        """Record the call's outcomes, with what was actually emitted, and return them in record order.

        A call that produced no result (:attr:`EmittedView.result` is None)
        placed nothing: its pending instructions never matched a result
        record and its admitted ones have no result to write into.
        """
        call = emitted.call
        if call is None:
            return decision.outcomes
        with self._lock:
            flight = self._in_flight.pop(call.call_id, None)
            if flight is None:
                outcomes = decision.outcomes
            elif flight.trail is None:
                outcomes = (
                    *flight.skipped,
                    *self._unread(flight.pending, call),
                    *self._no_result(flight.admitted, call),
                )
            else:
                outcomes = self._settled(decision, flight.trail, emitted.result)
            self._outcomes.extend(outcomes)
        return tuple(outcomes)

    def _admit_injections(
        self, scope: str, call: CallView
    ) -> tuple[list[_AdmittedInstruction], list[InjectionOutcome], list[_PendingInstruction]]:
        """Number matches under a lock before forwarding; that is admission order.

        Returns the instructions admitted for the call, the outcomes of those
        whose call arguments didn't match it (``unmatched``, occurrence 0: the
        call isn't counted) or whose occurrence rule skipped it, and those
        whose ``result_record`` the result decides (see ``_admit_on_result``).
        """
        admitted: list[_AdmittedInstruction] = []
        skipped: list[InjectionOutcome] = []
        pending: list[_PendingInstruction] = []
        with self._lock:
            self._occurrences.setdefault(scope, [0] * len(self._plan))
            order = self._admissions.get(scope, 0) + 1
            self._admissions[scope] = order
            for index, instruction in enumerate(self._plan):
                if instruction.target.tool not in (None, call.tool):
                    continue
                if (mismatch := instruction.target.call_argument_mismatch(call.arguments)) is not None:
                    skipped.append(
                        _outcome(
                            scope,
                            index,
                            instruction,
                            call.call_id,
                            call.tool,
                            0,
                            order,
                            OutcomeStatus.UNMATCHED,
                            f"{mismatch}; result unchanged",
                        )
                    )
                    continue
                if instruction.target.result_record is not None:
                    pending.append((scope, index, instruction, order))
                    continue
                self._count(scope, index, instruction, call.call_id, call.tool, order, admitted, skipped)
        return admitted, skipped, pending

    def _admit_on_result(
        self, call: CallView, structured_content: Any, pending: Sequence[_PendingInstruction]
    ) -> tuple[list[_AdmittedInstruction], list[InjectionOutcome]]:
        """Number the pending instructions whose ``result_record`` the result matches, as they arrive.

        The result is the upstream's, before any instruction changed it. One
        that doesn't match is ``unmatched`` at occurrence 0, like a call
        argument miss: the call isn't counted, so occurrence rules count only
        results that match, in the order they come back.
        """
        admitted: list[_AdmittedInstruction] = []
        skipped: list[InjectionOutcome] = []
        with self._lock:
            for scope, index, instruction, order in pending:
                if (mismatch := instruction.target.result_record_mismatch(structured_content)) is not None:
                    skipped.append(
                        _outcome(
                            scope,
                            index,
                            instruction,
                            call.call_id,
                            call.tool,
                            0,
                            order,
                            OutcomeStatus.UNMATCHED,
                            f"{mismatch}; result unchanged",
                        )
                    )
                    continue
                self._count(scope, index, instruction, call.call_id, call.tool, order, admitted, skipped)
        return admitted, skipped

    def _count(
        self,
        scope: str,
        index: int,
        instruction: InjectionInstruction,
        call_id: str,
        tool: str,
        order: int,
        admitted: list[_AdmittedInstruction],
        skipped: list[InjectionOutcome],
    ) -> None:
        """Count a matching call as the instruction's next occurrence; the caller holds the plan lock."""
        counts = self._occurrences[scope]
        counts[index] += 1
        occurrence = counts[index]
        if instruction.occurrence.selects(occurrence):
            admitted.append((scope, index, instruction, occurrence, order))
        else:
            skipped.append(
                _outcome(
                    scope,
                    index,
                    instruction,
                    call_id,
                    tool,
                    occurrence,
                    order,
                    OutcomeStatus.SKIPPED_OCCURRENCE,
                    "occurrence rule did not select this call",
                )
            )

    def _unread(self, pending: Sequence[_PendingInstruction], call: CallView) -> list[InjectionOutcome]:
        """With no result, no ``result_record`` matched: the call isn't counted for those instructions."""
        return [
            _outcome(
                scope,
                index,
                instruction,
                call.call_id,
                call.tool,
                0,
                order,
                OutcomeStatus.UNMATCHED,
                "upstream produced no tool result to match a result record",
            )
            for scope, index, instruction, order in pending
        ]

    def _no_result(self, admitted: Sequence[_AdmittedInstruction], call: CallView) -> list[InjectionOutcome]:
        return [
            _outcome(
                scope,
                index,
                instruction,
                call.call_id,
                call.tool,
                occurrence,
                order,
                OutcomeStatus.NO_RESULT,
                "upstream produced no tool result",
            )
            for scope, index, instruction, occurrence, order in admitted
        ]

    def _place(
        self,
        call: CallView,
        result: ResultView,
        admitted: Sequence[_AdmittedInstruction],
        skipped: Sequence[InjectionOutcome],
        flight: _InFlight,
    ) -> PlacementDecision:
        """Apply the admitted instructions, in admission order, to the result the upstream returned."""
        placements: list[_PlacementRecord] = []
        # One key per content block and per structured field written, tracking each through the instructions
        # (see _track and _track_field).
        keys: list[object] = [object() for _ in result.content]
        fields: dict[_Location, object] = {}
        for admission in admitted:
            result, placement = _inject_result(result, admission[2])
            keys, written = _track(keys, placement.edit, len(result.content))
            fields, field_written = _track_field(fields, placement.field)
            placements.append((admission, placement, written, field_written))
        # Emission is checked on the result as returned (see settle), so a later instruction that replaced a
        # placement shows.
        outcomes = list(skipped)
        for (scope, index, instruction, occurrence, order), placement, _written, _field_written in placements:
            outcomes.append(
                _outcome(
                    scope,
                    index,
                    instruction,
                    call.call_id,
                    call.tool,
                    occurrence,
                    order,
                    placement.status,
                    placement.detail,
                    selected=placement.status in _SELECTED,
                    transformation=instruction.transformation,
                    **(_placed(instruction, placement) if placement.status in PLACED else {}),
                )
            )
        flight.trail = _Trail(result, keys, fields, placements)
        written = any(placement.edit is not None or placement.field is not None for _, placement, _, _ in placements)
        return replace(result, outcomes) if written else leave(*outcomes)

    def _settled(
        self, decision: PlacementDecision, trail: _Trail, emitted: ResultView | None
    ) -> list[InjectionOutcome]:
        """The call's outcomes with each placement's emitted parts, read from the result the interceptor returned."""
        outcomes = list(decision.outcomes)
        if not trail.placements:
            return outcomes
        settled = outcomes[: len(outcomes) - len(trail.placements)]
        for outcome, (_, placement, written, field_written) in zip(
            outcomes[len(outcomes) - len(trail.placements) :], trail.placements, strict=True
        ):
            carried = (
                _emitted(emitted, trail.keys, trail.fields, placement, written, field_written)
                if emitted is not None
                else ()
            )
            settled.append(dataclasses.replace(outcome, emitted=carried))
        return settled
