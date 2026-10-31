"""Hand-written cases for MidojoMCP's injection splice, and the mutants they kill.

Each case forwards calls through a ``MidojoMCP`` whose upstream answers with
a fixed result, and compares what comes back with a literal written by hand:
the text of every content block, the structured content, and each
instruction's status, occurrence and emitted parts. Nothing expected is
computed by the code under test (or by ``json.dumps``): the JSON texts are
typed out. The cases cover structured placement and the JSON mirror's
re-serialisation in each layout, record selectors and indexes, content-block
placement, size limits, transformations, mismatch policies, occurrence
counting, call-argument and result-record matching, and a later instruction
replacing an earlier one's placement. A placement (applied, partial,
fallback) must name its payload's digest, its written text's and its slot,
and any other outcome none of them; where a case gives ``placed``, their
values are written from the literal payload and target
(``tests.slot_literals``). Each mutant below is one plausible single edit to
the splice code (``tests.mutation_harness``), and the cases must catch every
one of them; the same copies with no edit must fail no case.
"""

from __future__ import annotations

import asyncio
import types
from dataclasses import dataclass, field
from typing import Any

import midojo_sdk.injection as injection
import midojo_sdk.mcp as interceptor
import midojo_sdk.placement as splice
import pytest
from mcp.types import CallToolResult, ImageContent, TextContent
from midojo_sdk.injection import (
    Channel,
    InjectionInstruction,
    InjectionMode,
    InjectionPlan,
    InjectionTarget,
    MismatchPolicy,
    Occurrence,
    OccurrenceRule,
    RecordSelector,
    SelectorSurface,
    Transformation,
)

from tests.mutation_harness import Mutant, mutated, mutated_function, unedited, unedited_function
from tests.slot_literals import codec_target, placed

SC, CB = "structured_content", "content_block"
TOOL = "lookup"
PLACEMENTS = {"applied", "partial", "fallback"}


class Recorder:
    """Takes every start and finish, details included, and keeps nothing: the cases read the results."""

    async def start_call(self, **_: Any) -> None:
        pass

    async def finish_call(self, **_: Any) -> None:
        pass

    async def finish_call_detailed(self, **_: Any) -> None:
        pass


class FixedUpstream:
    def __init__(self, result: CallToolResult) -> None:
        self.result = result

    async def call_tool_mcp(self, _name: str, _args: dict[str, Any]) -> CallToolResult:
        return self.result


def result(*texts: str, structured: dict[str, Any] | None = None) -> CallToolResult:
    content: list[Any] = [
        ImageContent(type="image", data="iVBORw0KGgo=", mimeType="image/png")
        if text == "<image>"
        else TextContent(type="text", text=text)
        for text in texts
    ]
    return CallToolResult(content=content, structuredContent=structured)


def at(path: tuple, payload: str = "!", target: dict[str, Any] | None = None, **fields: Any) -> InjectionInstruction:
    """A structured-content instruction."""
    selector = InjectionTarget(
        surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=path, **(target or {})
    )
    return InjectionInstruction(channel=Channel.TOOL_OUTPUT, payload=payload, target=selector, **fields)


def block(payload: str = "!", target: dict[str, Any] | None = None, **fields: Any) -> InjectionInstruction:
    """A content-block instruction."""
    return InjectionInstruction(
        channel=Channel.TOOL_OUTPUT, payload=payload, target=InjectionTarget(**(target or {})), **fields
    )


@dataclass(frozen=True)
class Case:
    """Calls with ``calls``' arguments; after each, ``texts`` and ``structured`` are what it returned.

    ``outcomes`` are ``(call number, instruction index, status, occurrence, emitted parts)``. ``placed`` gives,
    by instruction index, ``(payload, written text, target in the plan codec's form, mode)`` for its placements.
    """

    id: str
    upstream: CallToolResult
    instructions: tuple[InjectionInstruction, ...]
    returned: list[tuple[list[str], dict[str, Any] | None]]
    outcomes: list[tuple[int, int, str, int, tuple[str, ...]]]
    calls: list[dict[str, Any]] = field(default_factory=lambda: [{}])
    details: dict[int, str] = field(default_factory=dict)
    placed: dict[int, tuple[str, str, dict[str, Any], str]] = field(default_factory=dict)

    async def observed(self, module: types.ModuleType) -> tuple[Any, Any, Any, Any, Any]:
        engine = module.PlanPlacementEngine(InjectionPlan(self.instructions), evaluation_id="e1")
        mcp: Any = interceptor.MidojoMCP(
            "splice",
            recorder=Recorder(),
            upstream_transport=object(),
            engine=engine,
            evaluation_id="e1",
        )
        mcp._upstream = FixedUpstream(self.upstream)
        returned = []
        for arguments in self.calls:
            sent = (await mcp._forward_call(TOOL, arguments)).to_mcp_result()
            texts = [part.text if isinstance(part, TextContent) else f"<{part.type}>" for part in sent.content]
            returned.append((texts, sent.structuredContent))
        recorded = engine.outcomes()
        outcomes = sorted(
            (o.admission_order, o.instruction_index, o.status.value, o.occurrence, tuple(p.value for p in o.emitted))
            for o in recorded
        )
        details = {o.instruction_index: o.detail for o in recorded}
        named = sorted(
            (o.admission_order, o.instruction_index, _named(o.payload_digest, o.placed_digest, o.slot))
            for o in recorded
        )
        fields = {
            index: [
                {
                    "payload_digest": o.payload_digest,
                    "placed_digest": o.placed_digest,
                    "slot": None if o.slot is None else o.slot.to_data(),
                }
                for o in sorted(recorded, key=lambda o: o.admission_order)
                if o.instruction_index == index and o.status.value in PLACEMENTS
            ]
            for index in self.placed
        }
        return returned, outcomes, {index: details[index] for index in self.details}, named, fields

    def expected(self) -> tuple[Any, Any, Any, Any, Any]:
        outcomes = sorted(self.outcomes)
        named = [(call, index, "all" if status in PLACEMENTS else "none") for call, index, status, _, _ in outcomes]
        fields = {
            index: [
                placed(*self.placed[index]) for _, i, status, _, _ in outcomes if i == index and status in PLACEMENTS
            ]
            for index in self.placed
        }
        return self.returned, outcomes, self.details, named, fields


def _named(*values: Any) -> str:
    """Whether an outcome names all of what a placement placed and where, none of it, or only some."""
    present = [value is not None for value in values]
    return "all" if all(present) else "none" if not any(present) else "some"


NOTE = {"a": {"note": "hi"}}
NOTED = {"a": {"note": "hi!"}}
NOTES = {"notes": [{"id": "N1", "body": "a"}, {"id": "N2", "body": "b"}]}
BOTH = (SC, CB)

CASES = [
    # --- a JSON mirror re-serialised in the upstream's own layout ---
    Case(
        "mirror.compact",
        result('{"a":{"note":"hi"}}', structured=NOTE),
        (at(("a", "note")),),
        [(['{"a":{"note":"hi!"}}'], NOTED)],
        [(1, 0, "applied", 1, BOTH)],
        details={0: "exact selector matched; mirrored text re-serialised"},
        placed={
            0: ("!", "!", codec_target(surface="structured_content", structured_content_path=["a", "note"]), "append")
        },
    ),
    Case(
        "mirror.spaced",
        result('{"a": {"note": "hi"}}', structured=NOTE),
        (at(("a", "note")),),
        [(['{"a": {"note": "hi!"}}'], NOTED)],
        [(1, 0, "applied", 1, BOTH)],
    ),
    Case(
        "mirror.indent-2",
        result('{\n  "a": {\n    "note": "hi"\n  }\n}', structured=NOTE),
        (at(("a", "note")),),
        [(['{\n  "a": {\n    "note": "hi!"\n  }\n}'], NOTED)],
        [(1, 0, "applied", 1, BOTH)],
    ),
    Case(
        "mirror.indent-4",
        result('{\n    "a": {\n        "note": "hi"\n    }\n}', structured=NOTE),
        (at(("a", "note")),),
        [(['{\n    "a": {\n        "note": "hi!"\n    }\n}'], NOTED)],
        [(1, 0, "applied", 1, BOTH)],
    ),
    Case(
        "mirror.sorted-keys",
        result('{"a":"x","b":"y"}', structured={"b": "y", "a": "x"}),
        (at(("a",)),),
        [(['{"a":"x!","b":"y"}'], {"b": "y", "a": "x!"})],
        [(1, 0, "applied", 1, BOTH)],
    ),
    Case(
        "mirror.indent-4-sorted-keys",
        result('{\n    "a": "x",\n    "b": "y"\n}', structured={"b": "y", "a": "x"}),
        (at(("a",)),),
        [(['{\n    "a": "x!",\n    "b": "y"\n}'], {"b": "y", "a": "x!"})],
        [(1, 0, "applied", 1, BOTH)],
    ),
    Case(
        "mirror.ascii-escapes-kept",
        result('{"a":"caf\\u00e9"}', structured={"a": "café"}),
        (at(("a",)),),
        [(['{"a":"caf\\u00e9!"}'], {"a": "café!"})],
        [(1, 0, "applied", 1, BOTH)],
    ),
    Case(
        "mirror.ascii-original-non-ascii-payload",
        result('{"a":"x"}', structured={"a": "x"}),
        (at(("a",), payload="é"),),
        [(['{"a":"xé"}'], {"a": "xé"})],
        [(1, 0, "applied", 1, BOTH)],
    ),
    Case(
        "mirror.unreproduced-one-line",
        result('{"a": "x"}\n', structured={"a": "x"}),
        (at(("a",)),),
        [(['{"a":"x!"}'], {"a": "x!"})],
        [(1, 0, "applied", 1, BOTH)],
    ),
    Case(
        "mirror.unreproduced-indent-3",
        result('{\n   "a": "x"\n}\n', structured={"a": "x"}),
        (at(("a",)),),
        [(['{\n   "a": "x!"\n}'], {"a": "x!"})],
        [(1, 0, "applied", 1, BOTH)],
    ),
    # --- text that isn't a mirror is left as sent ---
    Case(
        "partial.two-blocks",
        result('{"a":"x"}', "one note", structured={"a": "x"}),
        (at(("a",)),),
        [(['{"a":"x"}', "one note"], {"a": "x!"})],
        [(1, 0, "partial", 1, (SC,))],
        placed={0: ("!", "!", codec_target(surface="structured_content", structured_content_path=["a"]), "append")},
        details={
            0: "exact selector matched structured content only; text content is not a single text block (2 content blocks)"
        },
    ),
    Case(
        "partial.float-for-int",
        result('{"n":1.0,"a":"x"}', structured={"n": 1, "a": "x"}),
        (at(("a",)),),
        [(['{"n":1.0,"a":"x"}'], {"n": 1, "a": "x!"})],
        [(1, 0, "partial", 1, (SC,))],
        details={
            0: "exact selector matched structured content only; text content is JSON that differs from the structured content"
        },
    ),
    Case(
        "partial.not-json",
        result("a note", structured={"a": "x"}),
        (at(("a",)),),
        [(["a note"], {"a": "x!"})],
        [(1, 0, "partial", 1, (SC,))],
    ),
    # --- record selectors and indexes ---
    Case(
        "record.by-id",
        result('{"notes":[{"id":"N1","body":"a"},{"id":"N2","body":"b"}]}', structured=NOTES),
        (at(("notes", RecordSelector("N2"), "body")),),
        [
            (
                ['{"notes":[{"id":"N1","body":"a"},{"id":"N2","body":"b!"}]}'],
                {"notes": [{"id": "N1", "body": "a"}, {"id": "N2", "body": "b!"}]},
            )
        ],
        [(1, 0, "applied", 1, BOTH)],
    ),
    Case(
        "record.by-field",
        result(
            '{"rows":[{"sku":"S1","body":"a"},{"sku":"S2","body":"b"}]}',
            structured={"rows": [{"sku": "S1", "body": "a"}, {"sku": "S2", "body": "b"}]},
        ),
        (at(("rows", RecordSelector("S1", field="sku"), "body")),),
        [
            (
                ['{"rows":[{"sku":"S1","body":"a!"},{"sku":"S2","body":"b"}]}'],
                {"rows": [{"sku": "S1", "body": "a!"}, {"sku": "S2", "body": "b"}]},
            )
        ],
        [(1, 0, "applied", 1, BOTH)],
        placed={
            0: (
                "!",
                "!",
                codec_target(
                    surface="structured_content",
                    structured_content_path=["rows", {"field": "sku", "equals": "S1"}, "body"],
                ),
                "append",
            )
        },
    ),
    Case(
        "record.int-is-not-true",
        result(
            '{"notes":[{"id":true,"body":"a"},{"id":1,"body":"b"}]}',
            structured={"notes": [{"id": True, "body": "a"}, {"id": 1, "body": "b"}]},
        ),
        (at(("notes", RecordSelector(1), "body")),),
        [
            (
                ['{"notes":[{"id":true,"body":"a"},{"id":1,"body":"b!"}]}'],
                {"notes": [{"id": True, "body": "a"}, {"id": 1, "body": "b!"}]},
            )
        ],
        [(1, 0, "applied", 1, BOTH)],
        placed={
            0: (
                "!",
                "!",
                codec_target(
                    surface="structured_content",
                    structured_content_path=["notes", {"field": "id", "equals": 1}, "body"],
                ),
                "append",
            )
        },
    ),
    Case(
        "record.two-matches",
        result(
            '{"notes":[{"id":"N1","body":"a"},{"id":"N1","body":"b"}]}',
            structured={"notes": [{"id": "N1", "body": "a"}, {"id": "N1", "body": "b"}]},
        ),
        (at(("notes", RecordSelector("N1"), "body")),),
        [
            (
                ['{"notes":[{"id":"N1","body":"a"},{"id":"N1","body":"b"}]}'],
                {"notes": [{"id": "N1", "body": "a"}, {"id": "N1", "body": "b"}]},
            )
        ],
        [(1, 0, "unmatched", 1, ())],
        details={0: "record selector [id='N1'] matched 2 records; result unchanged"},
    ),
    Case(
        "record.plain-key-on-a-list",
        result('{"notes":[{"id":"N1","body":"a"},{"id":"N2","body":"b"}]}', structured=NOTES),
        (at(("notes", "N2", "body")),),
        [(['{"notes":[{"id":"N1","body":"a"},{"id":"N2","body":"b"}]}'], NOTES)],
        [(1, 0, "unmatched", 1, ())],
        details={0: "field 'N2' requires an object, got list; result unchanged"},
    ),
    Case(
        "index.in-range",
        result('{"tags":["a","b"]}', structured={"tags": ["a", "b"]}),
        (at(("tags", 1)),),
        [(['{"tags":["a","b!"]}'], {"tags": ["a", "b!"]})],
        [(1, 0, "applied", 1, BOTH)],
    ),
    Case(
        "index.out-of-range",
        result('{"tags":["a","b"]}', structured={"tags": ["a", "b"]}),
        (at(("tags", 5)),),
        [(['{"tags":["a","b"]}'], {"tags": ["a", "b"]})],
        [(1, 0, "unmatched", 1, ())],
        details={0: "index 5 is out of range; result unchanged"},
    ),
    Case(
        "field.replace",
        result('{"a":{"note":"hi"}}', structured=NOTE),
        (at(("a", "note"), payload="NEW", mode=InjectionMode.REPLACE),),
        [(['{"a":{"note":"NEW"}}'], {"a": {"note": "NEW"}})],
        [(1, 0, "applied", 1, BOTH)],
    ),
    Case(
        "field.not-a-string",
        result('{"n":1}', structured={"n": 1}),
        (at(("n",)),),
        [(['{"n":1}'], {"n": 1})],
        [(1, 0, "unmatched", 1, ())],
        details={0: "field 'n' does not contain a string; result unchanged"},
    ),
    Case(
        "field.not-a-string-fallback",
        result('{"n":1}', structured={"n": 1}),
        (at(("n",), mismatch_policy=MismatchPolicy.FALLBACK_APPEND),),
        [(['{"n":1}', "!"], {"n": 1})],
        [(1, 0, "fallback", 1, (CB,))],
        # A fallback's slot is the planned one.
        placed={0: ("!", "!", codec_target(surface="structured_content", structured_content_path=["n"]), "append")},
        details={0: "field 'n' does not contain a string; appended a text block"},
    ),
    # --- content blocks ---
    Case(
        "block.append",
        result("alpha", "beta"),
        (block(),),
        [(["alpha", "beta", "!"], None)],
        [(1, 0, "applied", 1, (CB,))],
    ),
    Case(
        "block.replace-all",
        result("alpha", "beta"),
        (block(mode=InjectionMode.REPLACE),),
        [(["!"], None)],
        [(1, 0, "applied", 1, (CB,))],
    ),
    Case(
        "block.index-extend",
        result("alpha", "beta"),
        (block(target={"content_block_index": 1}),),
        [(["alpha", "beta!"], None)],
        [(1, 0, "applied", 1, (CB,))],
    ),
    Case(
        "block.index-rewrite",
        result("alpha", "beta"),
        (block(target={"content_block_index": 1}, mode=InjectionMode.REPLACE),),
        [(["alpha", "!"], None)],
        [(1, 0, "applied", 1, (CB,))],
        placed={0: ("!", "!", codec_target(content_block_index=1), "replace")},
    ),
    Case(
        "block.index-out-of-range",
        result("alpha", "beta"),
        (block(target={"content_block_index": 5}),),
        [(["alpha", "beta"], None)],
        [(1, 0, "unmatched", 1, ())],
        details={0: "content-block selector did not match; result unchanged"},
    ),
    Case(
        "block.first-text-block",
        result("<image>", "beta"),
        (block(target={"content_block_type": "text"}),),
        [(["<image>", "beta!"], None)],
        [(1, 0, "applied", 1, (CB,))],
        placed={0: ("!", "!", codec_target(content_block_type="text"), "append")},
    ),
    # --- size limits and transformations ---
    Case(
        "size.bytes-over",
        result("alpha"),
        (block(payload="é", size_limit=1),),
        [(["alpha"], None)],
        [(1, 0, "too_large", 1, ())],
    ),
    Case(
        "size.bytes-at",
        result("alpha"),
        (block(payload="é", size_limit=2),),
        [(["alpha", "é"], None)],
        [(1, 0, "applied", 1, (CB,))],
    ),
    Case(
        "transform.json-string",
        result("alpha"),
        (block(payload='a"b', transformation=Transformation.JSON_STRING),),
        [(["alpha", '"a\\"b"'], None)],
        [(1, 0, "applied", 1, (CB,))],
        # The payload's digest is of the payload as planned, the placed digest of the transformed text.
        placed={0: ('a"b', '"a\\"b"', codec_target(), "append")},
    ),
    # --- occurrences ---
    Case(
        "occurrence.first",
        result("alpha"),
        (block(),),
        [(["alpha", "!"], None), (["alpha"], None), (["alpha"], None)],
        [(1, 0, "applied", 1, (CB,)), (2, 0, "skipped_occurrence", 2, ()), (3, 0, "skipped_occurrence", 3, ())],
        calls=[{}, {}, {}],
    ),
    Case(
        "occurrence.nth",
        result("alpha"),
        (block(occurrence=OccurrenceRule(Occurrence.NTH, 2)),),
        [(["alpha"], None), (["alpha", "!"], None), (["alpha"], None)],
        [(1, 0, "skipped_occurrence", 1, ()), (2, 0, "applied", 2, (CB,)), (3, 0, "skipped_occurrence", 3, ())],
        calls=[{}, {}, {}],
    ),
    Case(
        "occurrence.every",
        result("alpha"),
        (block(occurrence=OccurrenceRule(Occurrence.EVERY)),),
        [(["alpha", "!"], None), (["alpha", "!"], None)],
        [(1, 0, "applied", 1, (CB,)), (2, 0, "applied", 2, (CB,))],
        calls=[{}, {}],
        placed={0: ("!", "!", codec_target(), "append")},
    ),
    # --- call arguments and result records, type for type ---
    Case(
        "call-arguments.type-for-type",
        result("alpha"),
        (block(target={"tool": TOOL, "call_arguments": {"n": 1}}),),
        [(["alpha"], None), (["alpha"], None), (["alpha", "!"], None), (["alpha"], None)],
        [
            (1, 0, "unmatched", 0, ()),
            (2, 0, "unmatched", 0, ()),
            (3, 0, "applied", 1, (CB,)),
            (4, 0, "skipped_occurrence", 2, ()),
        ],
        calls=[{"n": True}, {"n": 1.0}, {"n": 1}, {"n": 1}],
        placed={0: ("!", "!", codec_target(tool=TOOL, call_arguments={"n": 1}), "append")},
    ),
    Case(
        "result-record.true-is-not-1",
        result('{"rid":true,"a":"x"}', structured={"rid": True, "a": "x"}),
        (at(("a",), target={"tool": TOOL, "result_record": RecordSelector(1, field="rid")}),),
        [(['{"rid":true,"a":"x"}'], {"rid": True, "a": "x"})],
        [(1, 0, "unmatched", 0, ())],
        details={0: "result field 'rid' is True, not 1; result unchanged"},
    ),
    Case(
        "result-record.matches",
        result('{"rid":1,"a":"x"}', structured={"rid": 1, "a": "x"}),
        (at(("a",), target={"tool": TOOL, "result_record": RecordSelector(1, field="rid")}),),
        [(['{"rid":1,"a":"x!"}'], {"rid": 1, "a": "x!"})],
        [(1, 0, "applied", 1, BOTH)],
        placed={
            0: (
                "!",
                "!",
                codec_target(
                    tool=TOOL,
                    surface="structured_content",
                    structured_content_path=["a"],
                    result_record={"field": "rid", "equals": 1},
                ),
                "append",
            )
        },
    ),
    # --- a slot whose target isn't ASCII: its id digests the text as UTF-8 ---
    Case(
        "slot.non-ascii-target",
        result('{"notes":[{"id":"NOTÉ","body":"a"}]}', structured={"notes": [{"id": "NOTÉ", "body": "a"}]}),
        (at(("notes", RecordSelector("NOTÉ"), "body")),),
        [(['{"notes":[{"id":"NOTÉ","body":"a!"}]}'], {"notes": [{"id": "NOTÉ", "body": "a!"}]})],
        [(1, 0, "applied", 1, BOTH)],
        placed={
            0: (
                "!",
                "!",
                codec_target(
                    surface="structured_content",
                    structured_content_path=["notes", {"field": "id", "equals": "NOTÉ"}, "body"],
                ),
                "append",
            )
        },
    ),
    # --- a later instruction replacing an earlier one's placement ---
    Case(
        "later.block-replace",
        result("alpha", "beta"),
        (block(payload="A"), block(payload="B", mode=InjectionMode.REPLACE)),
        [(["B"], None)],
        [(1, 0, "applied", 1, ()), (1, 1, "applied", 1, (CB,))],
        # One place, so one slot id, in two modes.
        placed={0: ("A", "A", codec_target(), "append"), 1: ("B", "B", codec_target(), "replace")},
    ),
    Case(
        "later.block-rewrite-with-the-same-text",
        result("alpha"),
        (
            block(payload="A", target={"content_block_index": 0}),
            block(payload="alphaA", target={"content_block_index": 0}, mode=InjectionMode.REPLACE),
        ),
        [(["alphaA"], None)],
        [(1, 0, "applied", 1, ()), (1, 1, "applied", 1, (CB,))],
    ),
    Case(
        "later.field-replace-with-the-same-text",
        result('{"a":{"note":"hi"}}', structured=NOTE),
        (at(("a", "note")), at(("a", "note"), payload="hi!", mode=InjectionMode.REPLACE)),
        [(['{"a":{"note":"hi!"}}'], NOTED)],
        [(1, 0, "applied", 1, ()), (1, 1, "applied", 1, BOTH)],
    ),
    Case(
        "later.field-append",
        result('{"a":{"note":"hi"}}', structured=NOTE),
        (at(("a", "note")), at(("a", "note"), payload="?")),
        [(['{"a":{"note":"hi!?"}}'], {"a": {"note": "hi!?"}})],
        [(1, 0, "applied", 1, BOTH), (1, 1, "applied", 1, BOTH)],
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
async def test_the_splice_gives_the_hand_written_result(case: Case):
    assert await case.observed(splice) == case.expected()


# --- the mutants ---


def failing(module: types.ModuleType) -> list[str]:
    """The ids of the cases whose result differs from the one written by hand, or that raise."""

    async def check(case: Case) -> bool:
        try:
            return await case.observed(module) == case.expected()
        except Exception:
            return False

    async def every() -> list[str]:
        return [case.id for case in CASES if not await check(case)]

    return asyncio.run(every())


SPLICE_MUTANTS = [
    Mutant("keys-never-sorted", "for sort_keys in (False, True):", "for sort_keys in (False,):"),
    Mutant("ascii-escapes-first", "for ensure_ascii in (False, True):", "for ensure_ascii in (True, False):"),
    Mutant("no-four-space-layout", '    (4, (",", ": ")),\n', ""),
    Mutant(
        "fallback-indent-fixed",
        'indent = len(lines[1]) - len(lines[1].lstrip(" ")) or 2',
        "indent = 2",
    ),
    Mutant(
        "mirror-loose-types",
        "    if type(a) is not type(b):\n        return False",
        "    if False:\n        return False",
    ),
    Mutant(
        "mirror-of-the-first-block",
        "if len(content) != 1 or not isinstance(content[0], TextContent):",
        "if not content or not isinstance(content[0], TextContent):",
    ),
    Mutant("selector-loose-types", "        and type(item[part.field]) is type(part.value)\n", ""),
    Mutant("selector-first-of-many", "if len(matches) != 1:", "if len(matches) < 1:"),
    Mutant(
        "field-prepends",
        "current[part] = payload if mode is InjectionMode.REPLACE else old + payload\n        return None\n    if type(part) is int:",
        "current[part] = payload if mode is InjectionMode.REPLACE else payload + old\n        return None\n    if type(part) is int:",
    ),
    Mutant(
        "field-replace-appends",
        "current[part] = payload if mode is InjectionMode.REPLACE else old + payload\n        return None\n    if type(part) is int:",
        "current[part] = old + payload\n        return None\n    if type(part) is int:",
    ),
    Mutant(
        "index-prepends",
        "current[part] = payload if mode is InjectionMode.REPLACE else old + payload\n        return None\n    assert",
        "current[part] = payload if mode is InjectionMode.REPLACE else payload + old\n        return None\n    assert",
    ),
    Mutant(
        "size-in-characters", "len(payload.encode()) > instruction.size_limit", "len(payload) > instruction.size_limit"
    ),
    Mutant(
        "size-limit-exclusive",
        "len(payload.encode()) > instruction.size_limit",
        "len(payload.encode()) >= instruction.size_limit",
    ),
    Mutant("no-json-string", "return json.dumps(instruction.payload)", "return instruction.payload"),
    Mutant("every-call-is-first", "counts[index] += 1", "counts[index] = 1"),
    Mutant("occurrence-rule-ignored", "if instruction.occurrence.selects(occurrence):", "if True:"),
    Mutant(
        "call-argument-miss-counted",
        "                    continue\n                if instruction.target.result_record is not None:",
        "                if instruction.target.result_record is not None:",
    ),
    Mutant(
        "block-index-ignored",
        "indexes = range(len(content)) if target.content_block_index is None else (target.content_block_index,)",
        "indexes = range(len(content))",
    ),
    Mutant(
        "block-rewrite-extends",
        'text, edit = payload, _Edit("rewrite", index, 0, payload)',
        'text, edit = block.text + payload, _Edit("rewrite", index, 0, payload)',
    ),
    Mutant(
        "replace-all-appends",
        'return result.model_copy(update={"content": [block]}), _Edit("replace_all", 0, 0, payload)',
        'return result.model_copy(update={"content": [*content, block]}), _Edit("append", len(content), 0, payload)',
    ),
    Mutant("fallback-never", "if instruction.mismatch_policy is MismatchPolicy.FALLBACK_APPEND:", "if False:"),
    Mutant("partial-never", "if updated is not None and problem is not None:", "if False:"),
    Mutant(
        "replaced-field-keeps-its-key",
        "if edit.replaced or edit.location not in fields:",
        "if edit.location not in fields:",
    ),
    Mutant(
        "replaced-block-keeps-its-key",
        "        keys = [object() if index == edit.block else key for index, key in enumerate(keys)]",
        "        pass",
    ),
    Mutant(
        "payload-digest-after-transformation",
        '"payload_digest": payload_digest(instruction.payload),',
        '"payload_digest": payload_digest(placement.payload),',
    ),
    Mutant(
        "placed-digest-before-transformation",
        '"placed_digest": payload_digest(placement.payload),',
        '"placed_digest": payload_digest(instruction.payload),',
    ),
    Mutant(
        "fallback-names-nothing",
        "if placement.status in PLACED else {}",
        "if placement.status in _SELECTED else {}",
    ),
    Mutant(
        "slot-without-its-mode", '"slot": InjectionSlot.of(instruction),', '"slot": InjectionSlot(instruction.target),'
    ),
    Mutant(
        "emitted-ignores-the-result",
        'structured = "structured_content" in placement.placed and _field_still_written(result, fields, field_written)',
        'structured = "structured_content" in placement.placed',
    ),
]


@pytest.mark.parametrize("mutant", SPLICE_MUTANTS, ids=[mutant.name for mutant in SPLICE_MUTANTS])
def test_every_splice_mutant_is_caught(mutant: Mutant):
    assert failing(mutated(splice, mutant)), f"no case catches {mutant.name}"


SLOT_ID = injection.InjectionSlot.__dict__["id"].fget
"""The slot id's getter, whose source (decorator included) a mutant edits."""

SLOT_ID_MUTANTS = [
    Mutant("keys-unsorted", "sort_keys=True,", "sort_keys=False,"),
    Mutant("spaced", 'separators=(",", ":"),', 'separators=(", ", ": "),'),
    Mutant("ascii-escaped", "ensure_ascii=False", "ensure_ascii=True"),
]


@pytest.mark.parametrize("mutant", SLOT_ID_MUTANTS, ids=[mutant.name for mutant in SLOT_ID_MUTANTS])
def test_every_slot_id_mutant_is_caught(mutant: Mutant, monkeypatch):
    slot_id = mutated_function(SLOT_ID, mutant)
    assert isinstance(slot_id, property)
    monkeypatch.setattr(injection.InjectionSlot, "id", slot_id)
    assert failing(splice), f"no case catches {mutant.name}"


# --- the control: the same copies, with no edit, fail no case ---


def test_an_unedited_splice_copy_fails_no_case():
    assert failing(unedited(splice)) == []


def test_unedited_matcher_and_slot_id_copies_fail_no_case(monkeypatch):
    monkeypatch.setattr(injection, "_same_scalar", unedited_function(injection._same_scalar))
    monkeypatch.setattr(injection.InjectionSlot, "id", unedited_function(SLOT_ID))
    assert failing(splice) == []


def test_a_call_argument_and_result_record_mutant_is_caught(monkeypatch):
    loose = mutated_function(
        injection._same_scalar,
        Mutant(
            "scalar-loose-types",
            "return type(actual) is type(expected) and actual == expected",
            "return actual == expected",
        ),
    )
    monkeypatch.setattr(injection, "_same_scalar", loose)
    assert set(failing(splice)) >= {"call-arguments.type-for-type", "result-record.true-is-not-1"}
