"""The frozen injection plan and MCP executor."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import Sequence
from typing import Any

import pytest
from fastmcp.exceptions import ToolError
from mcp.types import CallToolResult, TextContent
from midojo_sdk.injection import (
    CapabilityError,
    Channel,
    GeneratorInstructionError,
    GeneratorPermissions,
    InjectionInstruction,
    InjectionMode,
    InjectionOutcome,
    InjectionPlan,
    InjectionSlot,
    InjectionTarget,
    MismatchPolicy,
    Occurrence,
    OccurrenceRule,
    OutcomeStatus,
    RecordSelector,
    SelectorSurface,
    Transformation,
    capability_gaps,
    instruction_from_generator,
    instruction_from_generator_for_plan,
    payload_digest,
)
from midojo_sdk.mcp import MIDOJO_MCP_CAPABILITY, MidojoMCP, ToolContext
from midojo_sdk.recorder import Disposition, InjectionRecord

from midojo.episode import Episode
from midojo.suites import get_suite
from midojo.types import CallInjectionOutcome
from tests.scripted_mcp import ScriptedMCP
from tests.test_mcp_fidelity import CapturingRecorder

MIRRORED = "exact selector matched; mirrored text re-serialised"


class DetailedCapturingRecorder(CapturingRecorder):
    """A capturing recorder that records a finish's details too (a ``DetailedRecorder``)."""

    async def finish_call_detailed(
        self,
        *,
        call_id: str,
        function: str,
        args: dict[str, Any],
        result: str,
        disposition: Disposition,
        error: str | None = None,
        structured_content: dict[str, Any] | None,
        injection_outcomes: Sequence[InjectionRecord],
    ) -> None:
        await self.finish_call(
            call_id=call_id, function=function, args=args, result=result, disposition=disposition, error=error
        )
        self.finishes[-1] |= {"structured_content": structured_content, "injection_outcomes": list(injection_outcomes)}


def _instruction(**changes: Any) -> InjectionInstruction:
    fields = {
        "channel": Channel.TOOL_OUTPUT,
        "payload": " INJECTED",
        "target": InjectionTarget(content_block_index=0, content_block_type="text"),
    }
    return InjectionInstruction(**(fields | changes))


def _placed(instruction: InjectionInstruction) -> dict[str, Any]:
    """What an identity placement of ``instruction`` records it placed, and where."""
    digest = payload_digest(instruction.payload)
    return {"payload_digest": digest, "placed_digest": digest, "slot": InjectionSlot.of(instruction).to_data()}


class ConcurrentUpstream:
    def __init__(self) -> None:
        self.arrived = {1: asyncio.Event(), 2: asyncio.Event()}
        self.release = {1: asyncio.Event(), 2: asyncio.Event()}
        self.results: dict[int, CallToolResult] = {}

    async def call_tool_mcp(self, name: str, args: dict[str, Any]) -> CallToolResult:
        number = args["number"]
        self.arrived[number].set()
        await self.release[number].wait()
        result = CallToolResult(
            content=[TextContent(type="text", text=f"result {number}")],
            structuredContent={"items": [{"summary": f"result {number}"}]},
        )
        self.results[number] = result
        return result


def _mcp(
    plan: InjectionPlan, *, evaluation_id: str = "evaluation-1"
) -> tuple[MidojoMCP, ConcurrentUpstream, DetailedCapturingRecorder]:
    recorder = DetailedCapturingRecorder()
    mcp = MidojoMCP("proxy", recorder=recorder, upstream_transport=object(), plan=plan, evaluation_id=evaluation_id)
    upstream = ConcurrentUpstream()
    mcp._upstream = upstream  # type: ignore[assignment]
    return mcp, upstream, recorder


def _wire_result(result: Any) -> CallToolResult:
    return result.to_mcp_result()


def _texts(result: CallToolResult) -> list[str]:
    assert all(isinstance(block, TextContent) for block in result.content)
    return [block.text for block in result.content if isinstance(block, TextContent)]


def _structured(result: CallToolResult) -> dict[str, Any]:
    assert isinstance(result.structuredContent, dict)
    return result.structuredContent


async def _inject_structured(structured_content: Any, path: tuple, **changes: Any) -> tuple[CallToolResult, Any]:
    """Inject into a result whose text mirrors its structured content compactly, as FastMCP sends it."""
    text = "unchanged" if structured_content is None else json.dumps(structured_content, separators=(",", ":"))

    class StructuredUpstream:
        async def call_tool_mcp(self, _name: str, _args: dict[str, Any]) -> CallToolResult:
            return CallToolResult(
                content=[TextContent(type="text", text=text)],
                structuredContent=structured_content,
            )

    target = InjectionTarget(surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=path)
    mcp, _, _ = _mcp(InjectionPlan((_instruction(target=target, **changes),)))
    mcp._upstream = StructuredUpstream()  # type: ignore[assignment]
    result = _wire_result(await mcp._forward_call("lookup", {}))
    return result, mcp.injection_outcomes[0]


@pytest.mark.asyncio
async def test_empty_plan_returns_the_upstreams_result_object():
    mcp, upstream, _ = _mcp(InjectionPlan())
    call = asyncio.create_task(mcp._forward_call("lookup", {"number": 1}))
    await upstream.arrived[1].wait()
    upstream.release[1].set()
    assert _wire_result(await call) is upstream.results[1]
    assert mcp.injection_outcomes == ()


@pytest.mark.asyncio
async def test_size_limit_reports_without_relocating_payload():
    mcp, upstream, recorder = _mcp(InjectionPlan((_instruction(size_limit=1),)))
    call = asyncio.create_task(mcp._forward_call("lookup", {"number": 1}))
    await upstream.arrived[1].wait()
    upstream.release[1].set()
    result = _wire_result(await call)
    assert _texts(result) == ["result 1"]
    [outcome] = mcp.injection_outcomes
    assert outcome.status is OutcomeStatus.TOO_LARGE
    assert outcome.scope == "evaluation-1"
    assert outcome.call_id == recorder.starts[0]["call_id"] == recorder.finishes[0]["call_id"]


@pytest.mark.asyncio
async def test_structured_field_selector_changes_only_that_field():
    target = InjectionTarget(
        surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=("items", 0, "summary")
    )
    mcp, upstream, recorder = _mcp(InjectionPlan((_instruction(target=target),)))
    call = asyncio.create_task(mcp._forward_call("lookup", {"number": 1}))
    await upstream.arrived[1].wait()
    upstream.release[1].set()
    result = _wire_result(await call)
    assert _texts(result) == ["result 1"]
    assert result.structuredContent == {"items": [{"summary": "result 1 INJECTED"}]}
    # "result 1" isn't the JSON of the structured content, so only the structured content carries the payload.
    [outcome] = mcp.injection_outcomes
    assert outcome.status is OutcomeStatus.PARTIAL
    assert recorder.finishes[0]["result"] == "result 1"
    assert recorder.finishes[0]["structured_content"] == result.structuredContent


@pytest.mark.asyncio
async def test_a_content_injection_before_a_structured_one_leaves_no_mirror_so_that_one_is_partial():
    class StructuredUpstream:
        async def call_tool_mcp(self, _name: str, _args: dict[str, Any]) -> CallToolResult:
            return CallToolResult(
                content=[TextContent(type="text", text="result")],
                structuredContent={"notes": [{"id": "NOTE-1", "body": "original"}]},
            )

    structured_target = InjectionTarget(
        surface=SelectorSurface.STRUCTURED_CONTENT,
        structured_content_path=("notes", RecordSelector("NOTE-1"), "body"),
    )
    plan = InjectionPlan(
        (
            _instruction(payload=" CONTENT"),
            _instruction(payload=" STRUCTURED", target=structured_target),
        )
    )
    mcp, _, recorder = _mcp(plan)
    mcp._upstream = StructuredUpstream()  # type: ignore[assignment]

    result = _wire_result(await mcp._forward_call("lookup", {}))

    assert _texts(result) == ["result CONTENT"]
    assert _structured(result)["notes"][0]["body"] == "original STRUCTURED"
    assert [outcome.status for outcome in mcp.injection_outcomes] == [OutcomeStatus.APPLIED, OutcomeStatus.PARTIAL]
    [finish] = recorder.finishes
    assert (finish["result"], finish["structured_content"]) == ("result CONTENT", result.structuredContent)
    assert [record["status"] for record in finish["injection_outcomes"]] == ["applied", "partial"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [(InjectionMode.APPEND, "original INJECTED"), (InjectionMode.REPLACE, " INJECTED")],
)
async def test_record_selector_injects_one_matching_record_in_both_modes(mode, expected):
    source = {"notes": [{"id": "NOTE-1", "body": "original"}, {"id": "NOTE-2", "body": "other"}]}
    result, outcome = await _inject_structured(source, ("notes", RecordSelector("NOTE-1"), "body"), mode=mode)

    assert result.structuredContent == {
        "notes": [{"id": "NOTE-1", "body": expected}, {"id": "NOTE-2", "body": "other"}]
    }
    assert source["notes"][0]["body"] == "original"
    assert json.loads(_texts(result)[0]) == result.structuredContent
    assert (outcome.status, outcome.detail) == (OutcomeStatus.APPLIED, MIRRORED)


@pytest.mark.asyncio
async def test_record_selector_honors_size_limit_and_json_string_transformation():
    source = {"notes": [{"id": "NOTE-1", "body": "original"}]}
    path = ("notes", RecordSelector("NOTE-1"), "body")
    transformed, applied = await _inject_structured(
        source,
        path,
        payload="line\n",
        transformation=Transformation.JSON_STRING,
        size_limit=len(json.dumps("line\n").encode()),
    )
    unchanged, too_large = await _inject_structured(source, path, size_limit=1)

    assert _structured(transformed)["notes"][0]["body"] == 'original"line\\n"'
    assert applied.status is OutcomeStatus.APPLIED
    assert unchanged.structuredContent == source
    assert (too_large.status, too_large.detail) == (
        OutcomeStatus.TOO_LARGE,
        "transformed payload exceeds size_limit",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("structured", "path", "reason"),
    [
        ({"notes": [{"id": "NOTE-2", "body": "x"}]}, ("notes", RecordSelector("NOTE-1"), "body"), "matched no records"),
        (
            {"notes": [{"id": "NOTE-1", "body": "x"}, {"id": "NOTE-1", "body": "y"}]},
            ("notes", RecordSelector("NOTE-1"), "body"),
            "matched 2 records",
        ),
        ({"notes": {}}, ("notes", RecordSelector("NOTE-1"), "body"), "requires a list"),
        ({"notes": [{"id": "NOTE-1"}]}, ("notes", RecordSelector("NOTE-1"), "body"), "field 'body' is missing"),
        (
            {"notes": [{"id": "NOTE-1", "body": "x"}]},
            ("notes", RecordSelector("NOTE-1")),
            "selects a record, not a string field",
        ),
        (None, ("notes", RecordSelector("NOTE-1"), "body"), "requires an object, got NoneType"),
    ],
    ids=["no-match", "duplicates", "non-list", "missing-field", "selector-leaf", "text-only-result"],
)
async def test_record_selector_mismatches_are_strict_and_explain_the_cause(structured, path, reason):
    result, outcome = await _inject_structured(structured, path)

    assert result.structuredContent == structured
    assert outcome.status is OutcomeStatus.UNMATCHED
    assert reason in outcome.detail


@pytest.mark.asyncio
async def test_nested_record_selectors_mix_custom_fields_keys_and_indexes():
    source = {
        "orders": [
            {"id": "O-1", "items": [{"sku": "X", "notes": ["first", "target"]}]},
            {"id": "O-2", "items": [{"sku": "X", "notes": ["other"]}]},
        ]
    }
    path = (
        "orders",
        RecordSelector("O-1"),
        "items",
        RecordSelector("X", field="sku"),
        "notes",
        1,
    )
    result, outcome = await _inject_structured(source, path, mode=InjectionMode.REPLACE)

    assert _structured(result)["orders"][0]["items"][0]["notes"] == ["first", " INJECTED"]
    assert _structured(result)["orders"][1]["items"][0]["notes"] == ["other"]
    assert outcome.status is OutcomeStatus.APPLIED


@pytest.mark.asyncio
async def test_record_selector_matching_is_type_exact_and_fallback_stays_explicit():
    source = {"notes": [{"id": "1", "body": "string"}, {"id": 1, "body": "integer"}]}
    integer, _ = await _inject_structured(source, ("notes", RecordSelector(1), "body"))
    fallback, outcome = await _inject_structured(
        {"notes": []},
        ("notes", RecordSelector("missing"), "body"),
        mismatch_policy=MismatchPolicy.FALLBACK_APPEND,
    )

    assert _structured(integer)["notes"] == [
        {"id": "1", "body": "string"},
        {"id": 1, "body": "integer INJECTED"},
    ]
    assert _texts(fallback) == ['{"notes":[]}', " INJECTED"]
    assert outcome.status is OutcomeStatus.FALLBACK
    assert "matched no records" in outcome.detail


def test_record_selector_validation_and_hashability():
    assert len({RecordSelector("NOTE-1"), RecordSelector("NOTE-1")}) == 1
    with pytest.raises(ValueError, match="must not be empty"):
        RecordSelector("NOTE-1", field="")
    for invalid in (True, 1.0, None):
        with pytest.raises(TypeError, match="exact str or int"):
            RecordSelector(invalid)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="template path"):
        InjectionTarget(surface=SelectorSurface.TEMPLATE, template_path=(RecordSelector("x"),))


def test_plan_json_round_trip_carries_record_selectors_through_an_episode():
    path = ("notes", RecordSelector("NOTE-1"), "body")
    plan = InjectionPlan(
        (
            _instruction(
                target=InjectionTarget(surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=path)
            ),
        ),
        generator_permissions=GeneratorPermissions(
            tools=frozenset({"lookup"}), fields=frozenset({path}), modes=frozenset({InjectionMode.REPLACE})
        ),
    )
    encoded = plan.to_json()
    decoded = InjectionPlan.from_json(encoded)
    data = plan.to_data()
    episode = Episode(plan=decoded)

    assert InjectionPlan.from_data(data) == decoded == plan == episode.plan
    assert plan.to_data() == data and plan.to_data() is not data
    assert json.loads(encoded)["instructions"][0]["target"]["structured_content_path"] == [
        "notes",
        {"field": "id", "equals": "NOTE-1"},
        "body",
    ]


def test_plain_paths_keep_their_json_shape_and_default_selector_field_decodes():
    target = InjectionTarget(surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=("items", 0, "body"))
    plan = InjectionPlan((_instruction(target=target),))
    encoded = plan.to_json()
    data = json.loads(encoded)
    assert data["instructions"][0]["target"]["structured_content_path"] == ["items", 0, "body"]
    assert plan.to_json() == encoded

    data["instructions"][0]["target"]["structured_content_path"] = ["notes", {"equals": "NOTE-1"}, "body"]
    [selector] = [
        step
        for step in InjectionPlan.from_json(json.dumps(data)).instructions[0].target.structured_content_path
        if isinstance(step, RecordSelector)
    ]
    assert selector == RecordSelector("NOTE-1")


def test_plan_data_and_json_decoders_refuse_unknown_or_missing_fields():
    valid = InjectionPlan((_instruction(),)).to_data()
    unknown_instruction = json.loads(json.dumps(valid))
    unknown_instruction["instructions"][0]["paylod"] = "typo"
    invalid = [
        {"instruction": valid["instructions"]},
        valid | {"unknown": True},
        unknown_instruction,
    ]

    for data in invalid:
        with pytest.raises(ValueError):
            InjectionPlan.from_data(data)
        with pytest.raises(ValueError):
            InjectionPlan.from_json(json.dumps(data))
    with pytest.raises(ValueError, match="invalid injection plan JSON"):
        InjectionPlan.from_json("{")


def test_plan_is_frozen_and_generator_output_is_bounded():
    instruction = _instruction()
    plan = InjectionPlan((instruction,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        instruction.payload = "changed"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.instructions = ()  # type: ignore[misc]
    permissions = GeneratorPermissions(
        tools=frozenset({"search"}),
        fields=frozenset({("items", 0, "summary")}),
        modes=frozenset({InjectionMode.REPLACE}),
    )
    made = instruction_from_generator(
        {"payload": "x", "tool": "search", "field": ["items", 0, "summary"], "mode": "replace"}, permissions
    )
    assert made.target.structured_content_path == ("items", 0, "summary")
    for changed in (
        {"tool": "other"},
        {"field": ["secret"]},
        {"mode": "append"},
    ):
        output = {"payload": "x", "tool": "search", "field": ["items", 0, "summary"], "mode": "replace"} | changed
        with pytest.raises(GeneratorInstructionError):
            instruction_from_generator(output, permissions)


def test_generator_output_accepts_a_json_record_selector_allowed_by_the_plan():
    selected = ("notes", RecordSelector("NOTE-1"), "body")
    custom = ("orders", RecordSelector("O-1", field="order_id"), "body")
    permissions = GeneratorPermissions(
        tools=frozenset({"search"}),
        fields=frozenset({selected, custom}),
        modes=frozenset({InjectionMode.REPLACE}),
    )

    made = instruction_from_generator(
        {
            "payload": "x",
            "tool": "search",
            "field": ["notes", {"equals": "NOTE-1"}, "body"],
            "mode": "replace",
        },
        permissions,
    )

    assert made.target.structured_content_path == selected

    made = instruction_from_generator(
        {
            "payload": "x",
            "tool": "search",
            "field": ["orders", {"field": "order_id", "equals": "O-1"}, "body"],
            "mode": "replace",
        },
        permissions,
    )

    assert made.target.structured_content_path == custom


def test_template_instructions_must_agree_with_the_substitution_map():
    instruction = InjectionInstruction(
        channel=Channel.USER_PROMPT,
        payload="payload",
        target=InjectionTarget(surface=SelectorSurface.TEMPLATE, template_key="task:probe"),
        mode=InjectionMode.REPLACE,
        probe_key="task:probe",
    )
    with pytest.raises(ValueError, match="missing from probe_payloads"):
        InjectionPlan((instruction,))
    with pytest.raises(ValueError, match="disagrees"):
        InjectionPlan((instruction,), probe_payloads=(("task:probe", "different"),))
    with pytest.raises(ValueError, match="same probe_key"):
        InjectionPlan(
            (dataclasses.replace(instruction, probe_key="other:probe"),),
            probe_payloads=(("task:probe", "payload"),),
        )

    plan = InjectionPlan(
        (instruction,),
        probe_payloads=(("unused:probe", "unused"), ("task:probe", "payload")),
    )
    assert dict(plan.substitution_payloads) == {"task:probe": "payload", "unused:probe": "unused"}
    with pytest.raises(TypeError):
        plan.substitution_payloads["task:probe"] = "changed"  # type: ignore[index]


def test_a_plan_selects_the_channels_an_external_interceptor_delivers():
    tool = _instruction()
    template = InjectionInstruction(
        channel=Channel.RUNTIME_FILE,
        payload="payload",
        target=InjectionTarget(surface=SelectorSurface.TEMPLATE, template_key="task:probe"),
        mode=InjectionMode.REPLACE,
        probe_key="task:probe",
    )
    plan = InjectionPlan((template, tool), probe_payloads=(("task:probe", "payload"),))

    selected = plan.for_channels(Channel.TOOL_OUTPUT)

    assert selected.instructions == (tool,)
    assert selected.generator_permissions == plan.generator_permissions
    assert selected.probe_payloads == plan.probe_payloads


def test_generator_output_is_constrained_by_the_episode_plan_permissions():
    permissions = GeneratorPermissions(
        tools=frozenset({"search"}),
        fields=frozenset({("items", 0, "summary")}),
        modes=frozenset({InjectionMode.REPLACE}),
    )
    episode = Episode(plan=InjectionPlan(generator_permissions=permissions))
    valid = {"payload": "x", "tool": "search", "field": ["items", 0, "summary"], "mode": "replace"}
    made = instruction_from_generator_for_plan(valid, episode.plan, probe_key="task:probe")
    assert (made.target.tool, made.target.structured_content_path, made.probe_key) == (
        "search",
        ("items", 0, "summary"),
        "task:probe",
    )
    with pytest.raises(GeneratorInstructionError, match="not permitted"):
        instruction_from_generator_for_plan(valid | {"tool": "other"}, episode.plan)


def test_selector_fields_must_belong_to_the_selected_surface():
    with pytest.raises(ValueError, match="structured_content selectors"):
        InjectionTarget(
            surface=SelectorSurface.STRUCTURED_CONTENT,
            content_block_index=0,
            structured_content_path=("summary",),
        )
    with pytest.raises(ValueError, match="without a tool"):
        InjectionInstruction(
            channel=Channel.USER_PROMPT,
            payload="x",
            target=InjectionTarget(surface=SelectorSurface.TEMPLATE, template_key="task:probe", tool="search"),
        )


def test_capability_gaps_are_typed_and_listed_before_calls():
    target = InjectionTarget(surface=SelectorSurface.DESCRIPTION)
    plan = InjectionPlan((InjectionInstruction(Channel.TOOL_DESCRIPTION, "x", target),))
    with pytest.raises(CapabilityError) as raised:
        _mcp(plan)
    assert [(gap.aspect, gap.value) for gap in raised.value.gaps] == [
        ("channel", "tool_description"),
        ("selector", "description"),
    ]


def test_a_nonempty_plan_with_an_injected_recorder_requires_an_evaluation_binding():
    with pytest.raises(ValueError, match="needs an evaluation_id"):
        MidojoMCP(
            "proxy",
            recorder=CapturingRecorder(),
            upstream_transport=object(),
            plan=InjectionPlan((_instruction(),)),
        )


def test_midojo_mcp_requires_an_exact_plan_and_an_upstream_to_deliver_it():
    with pytest.raises(TypeError, match="plan must be an InjectionPlan"):
        MidojoMCP(
            "proxy",
            recorder=CapturingRecorder(),
            upstream_transport=object(),
            plan=(_instruction(),),  # type: ignore[arg-type]
        )
    with pytest.raises(CapabilityError) as raised:
        MidojoMCP(
            "proxy",
            recorder=CapturingRecorder(),
            plan=InjectionPlan((_instruction(),)),
            evaluation_id="evaluation-1",
        )
    assert [(gap.aspect, gap.value) for gap in raised.value.gaps] == [("delivery", "mirrored_upstream")]


@pytest.mark.parametrize("target_tool", ["simulated", None], ids=["named", "all-tools"])
def test_simulated_tools_reject_instructions_that_would_match_them(target_tool):
    instruction = _instruction(target=InjectionTarget(tool=target_tool))
    mcp = MidojoMCP(
        "proxy",
        recorder=CapturingRecorder(),
        upstream_transport=object(),
        plan=InjectionPlan((instruction,)),
        evaluation_id="evaluation-1",
    )

    async def simulated(ctx: ToolContext) -> str:
        return "result"

    with pytest.raises(CapabilityError) as raised:
        mcp.tool()(simulated)
    assert [(gap.aspect, gap.value) for gap in raised.value.gaps] == [("delivery", "simulated_tool:simulated")]


@pytest.mark.asyncio
async def test_a_compiled_suite_plan_can_select_the_part_midojo_mcp_delivers():
    suite = get_suite("weather")
    probes = suite.get_probes_for_task("tornado_alert_via_prompt")
    episode = suite.episode("weather_new_york", "tornado_alert_via_prompt", probes)
    plan = episode.plan.for_channels(Channel.TOOL_OUTPUT)
    mcp = MidojoMCP("proxy", recorder=CapturingRecorder(), upstream_transport=ScriptedMCP().transport, plan=plan)

    assert plan.instructions == ()
    assert await mcp.mirror_upstream()


@pytest.mark.asyncio
async def test_targeted_tool_is_validated_against_discovery():
    plan = InjectionPlan((_instruction(target=InjectionTarget(tool="absent")),))
    mcp = MidojoMCP(
        "proxy",
        recorder=CapturingRecorder(),
        upstream_transport=ScriptedMCP().transport,
        plan=plan,
        evaluation_id="evaluation-1",
    )
    with pytest.raises(CapabilityError) as raised:
        await mcp.mirror_upstream()
    assert [(gap.aspect, gap.value) for gap in raised.value.gaps] == [("tool", "absent")]


@pytest.mark.asyncio
async def test_concurrent_matches_are_counted_in_admission_order():
    plan = InjectionPlan((_instruction(occurrence=OccurrenceRule(Occurrence.FIRST)),))
    mcp, upstream, _ = _mcp(plan)
    first = asyncio.create_task(mcp._forward_call("lookup", {"number": 1}))
    await upstream.arrived[1].wait()
    second = asyncio.create_task(mcp._forward_call("lookup", {"number": 2}))
    await upstream.arrived[2].wait()
    upstream.release[2].set()
    second_result = _wire_result(await second)
    upstream.release[1].set()
    first_result = _wire_result(await first)
    assert _texts(first_result) == ["result 1 INJECTED"]
    assert _texts(second_result) == ["result 2"]
    assert [(item.admission_order, item.occurrence, item.status) for item in mcp.injection_outcomes] == [
        (1, 1, OutcomeStatus.APPLIED),
        (2, 2, OutcomeStatus.SKIPPED_OCCURRENCE),
    ]


@pytest.mark.asyncio
async def test_admission_order_follows_start_completion_not_start_arrival():
    class ReorderingRecorder(CapturingRecorder):
        def __init__(self) -> None:
            super().__init__()
            self.first_arrived = asyncio.Event()
            self.release_first = asyncio.Event()

        async def start_call(
            self, *, call_id: str, function: str, args: dict[str, Any], disposition: Disposition
        ) -> None:
            await super().start_call(call_id=call_id, function=function, args=args, disposition=disposition)
            if args["number"] == 1:
                self.first_arrived.set()
                await self.release_first.wait()

    plan = InjectionPlan((_instruction(occurrence=OccurrenceRule(Occurrence.FIRST)),))
    mcp, upstream, _ = _mcp(plan)
    recorder = ReorderingRecorder()
    mcp._recorder = recorder
    first = asyncio.create_task(mcp._forward_call("lookup", {"number": 1}))
    await recorder.first_arrived.wait()
    second = asyncio.create_task(mcp._forward_call("lookup", {"number": 2}))
    await upstream.arrived[2].wait()
    recorder.release_first.set()
    await upstream.arrived[1].wait()
    upstream.release[1].set()
    upstream.release[2].set()

    first_result, second_result = await asyncio.gather(first, second)

    assert [start["args"]["number"] for start in recorder.starts] == [1, 2]
    assert _texts(_wire_result(first_result)) == ["result 1"]
    assert _texts(_wire_result(second_result)) == ["result 2 INJECTED"]
    assert [(item.tool, item.admission_order, item.occurrence) for item in mcp.injection_outcomes] == [
        ("lookup", 1, 1),
        ("lookup", 2, 2),
    ]


@pytest.mark.asyncio
async def test_injected_recorders_use_one_counter_scope_per_bound_evaluation():
    plan = InjectionPlan((_instruction(occurrence=OccurrenceRule(Occurrence.FIRST)),))
    first_mcp, first_upstream, _ = _mcp(plan, evaluation_id="evaluation-1")
    second_mcp, second_upstream, _ = _mcp(plan, evaluation_id="evaluation-2")
    first = asyncio.create_task(first_mcp._forward_call("lookup", {"number": 1}))
    second = asyncio.create_task(second_mcp._forward_call("lookup", {"number": 1}))
    await first_upstream.arrived[1].wait()
    await second_upstream.arrived[1].wait()
    first_upstream.release[1].set()
    second_upstream.release[1].set()
    assert _texts(_wire_result(await first)) == ["result 1 INJECTED"]
    assert _texts(_wire_result(await second)) == ["result 1 INJECTED"]
    assert [(item.admission_order, item.occurrence) for item in first_mcp.injection_outcomes] == [(1, 1)]
    assert [(item.admission_order, item.occurrence) for item in second_mcp.injection_outcomes] == [(1, 1)]


@pytest.mark.asyncio
async def test_blocked_calls_are_not_admitted_for_injection():
    plan = InjectionPlan((_instruction(occurrence=OccurrenceRule(Occurrence.FIRST)),))
    mcp, upstream, recorder = _mcp(plan)
    mcp.block("lookup")
    with pytest.raises(ToolError):
        await mcp._forward_call("lookup", {"number": 1})
    assert not upstream.arrived[1].is_set()
    assert mcp.injection_outcomes == ()
    assert recorder.starts == []
    assert [finish["disposition"] for finish in recorder.finishes] == ["blocked"]


@pytest.mark.asyncio
async def test_denied_calls_are_not_admitted_for_injection():
    plan = InjectionPlan((_instruction(occurrence=OccurrenceRule(Occurrence.FIRST)),))
    mcp, upstream, recorder = _mcp(plan)
    mcp.deny("lookup")(lambda _args: "not allowed")
    with pytest.raises(ToolError, match="not allowed"):
        await mcp._forward_call("lookup", {"number": 1})
    assert not upstream.arrived[1].is_set()
    assert mcp.injection_outcomes == ()
    assert recorder.starts == []
    assert [finish["disposition"] for finish in recorder.finishes] == ["denied"]


@pytest.mark.asyncio
async def test_upstream_errors_consume_the_occurrence_and_name_the_recorded_call():
    class FailingUpstream:
        async def call_tool_mcp(self, name: str, args: dict[str, Any]) -> CallToolResult:
            raise RuntimeError("upstream failed")

    plan = InjectionPlan((_instruction(occurrence=OccurrenceRule(Occurrence.FIRST)),))
    mcp, _, recorder = _mcp(plan)
    mcp._upstream = FailingUpstream()  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="upstream failed"):
        await mcp._forward_call("lookup", {"number": 1})
    [outcome] = mcp.injection_outcomes
    assert (outcome.status, outcome.occurrence, outcome.admission_order) == (OutcomeStatus.NO_RESULT, 1, 1)
    assert outcome.call_id == recorder.starts[0]["call_id"] == recorder.finishes[0]["call_id"]
    assert recorder.finishes[0]["disposition"] == "forwarded"


@pytest.mark.asyncio
async def test_cancelled_calls_consume_the_occurrence_and_leave_the_record_unfinished():
    plan = InjectionPlan((_instruction(occurrence=OccurrenceRule(Occurrence.FIRST)),))
    mcp, upstream, recorder = _mcp(plan)
    call = asyncio.create_task(mcp._forward_call("lookup", {"number": 1}))
    await upstream.arrived[1].wait()
    call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await call
    [outcome] = mcp.injection_outcomes
    assert (outcome.status, outcome.occurrence, outcome.admission_order) == (OutcomeStatus.NO_RESULT, 1, 1)
    assert outcome.call_id == recorder.starts[0]["call_id"]
    assert recorder.finishes == []


@pytest.mark.asyncio
async def test_strict_mismatch_is_reported_and_fallback_is_explicit():
    target = InjectionTarget(surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=("missing",))
    strict, strict_upstream, _ = _mcp(InjectionPlan((_instruction(target=target),)))
    strict_call = asyncio.create_task(strict._forward_call("lookup", {"number": 1}))
    await strict_upstream.arrived[1].wait()
    strict_upstream.release[1].set()
    strict_result = _wire_result(await strict_call)
    assert _texts(strict_result) == ["result 1"]
    assert strict.injection_outcomes[0].status is OutcomeStatus.UNMATCHED
    fallback_instruction = _instruction(target=target, mismatch_policy=MismatchPolicy.FALLBACK_APPEND)
    fallback, fallback_upstream, _ = _mcp(InjectionPlan((fallback_instruction,)))
    fallback_call = asyncio.create_task(fallback._forward_call("lookup", {"number": 1}))
    await fallback_upstream.arrived[1].wait()
    fallback_upstream.release[1].set()
    fallback_result = _wire_result(await fallback_call)
    assert _texts(fallback_result) == ["result 1", " INJECTED"]
    assert fallback.injection_outcomes[0].status is OutcomeStatus.FALLBACK


# --- the text that mirrors structured content ---

NOTES = {"notes": [{"id": "NOTE-1", "body": "first"}, {"id": "NOTE-2", "body": "second ü"}]}
NOTE_BODY = ("notes", RecordSelector("NOTE-1"), "body")


class FixedUpstream:
    def __init__(self, result: CallToolResult) -> None:
        self.result = result

    async def call_tool_mcp(self, _name: str, _args: dict[str, Any]) -> CallToolResult:
        return self.result


def _structured_instruction(**changes: Any) -> InjectionInstruction:
    target = InjectionTarget(surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=NOTE_BODY)
    return _instruction(target=target, **changes)


def _noted(body: str, structured: dict[str, Any] = NOTES) -> dict[str, Any]:
    changed = json.loads(json.dumps(structured))
    changed["notes"][0]["body"] = body
    return changed


async def _forward(
    result: CallToolResult, *instructions: InjectionInstruction
) -> tuple[CallToolResult, MidojoMCP, DetailedCapturingRecorder]:
    mcp, _, recorder = _mcp(InjectionPlan(instructions))
    mcp._upstream = FixedUpstream(result)  # type: ignore[assignment]
    return _wire_result(await mcp._forward_call("lookup", {})), mcp, recorder


def _text_result(*texts: str, structured: Any = None, is_error: bool = False) -> CallToolResult:
    content = [TextContent(type="text", text=text) for text in texts]
    return CallToolResult(content=list(content), structuredContent=structured, isError=is_error)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "style",
    [
        {"separators": (",", ":"), "ensure_ascii": False},
        {},
        {"indent": 2, "ensure_ascii": False},
        {"indent": 4, "sort_keys": True},
        {"separators": (",", ":"), "ensure_ascii": True},
    ],
    ids=["compact", "spaced", "indent-2", "indent-4-sorted", "compact-ascii"],
)
@pytest.mark.parametrize(
    ("mode", "body"), [(InjectionMode.APPEND, "first INJECTED"), (InjectionMode.REPLACE, " INJECTED")]
)
async def test_a_mirrored_text_block_is_re_serialised_in_its_own_style(style, mode, body):
    text = json.dumps(NOTES, **style)
    instruction = _structured_instruction(mode=mode)

    result, mcp, recorder = await _forward(_text_result(text, structured=NOTES), instruction)

    expected = _noted(body)
    assert result.structuredContent == expected
    assert _texts(result) == [json.dumps(expected, **style)]
    [outcome] = mcp.injection_outcomes
    assert (outcome.status, outcome.detail) == (OutcomeStatus.APPLIED, MIRRORED)
    # The record is what the interceptor returned: the re-serialised text and the injected structured content.
    [finish] = recorder.finishes
    assert (finish["result"], finish["structured_content"], finish["error"]) == (_texts(result)[0], expected, None)
    assert finish["injection_outcomes"] == [
        {
            "instruction_index": 0,
            "probe_key": None,
            "occurrence": 1,
            "status": "applied",
            "detail": MIRRORED,
            "selected": True,
            "transformation": "identity",
            "emitted": ["structured_content", "content_block"],
            **_placed(instruction),
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "layout"),
    [
        ('{"n":1e20,' + json.dumps(NOTES, separators=(",", ":"))[1:], {"separators": (",", ":")}),
        (json.dumps({"n": 1e20, **NOTES}, indent=3).replace("1e+20", "1e20"), {"indent": 3, "separators": (",", ": ")}),
    ],
    ids=["compact", "indented"],
)
async def test_a_mirror_no_layout_reproduces_keeps_its_compact_or_indented_layout(text, layout):
    structured = {"n": 1e20, **NOTES}

    result, mcp, _ = await _forward(_text_result(text, structured=structured), _structured_instruction())

    expected = _noted("first INJECTED", structured)
    assert _texts(result) == [json.dumps(expected, ensure_ascii=False, **layout)]
    assert mcp.injection_outcomes[0].status is OutcomeStatus.APPLIED


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "structured", "reason"),
    [
        (["Notes: first, second"], NOTES, "text content is not JSON"),
        (['{"notes": []}'], NOTES, "differs from the structured content"),
        (['{"count": true, "notes": []}'], {"count": 1, "notes": NOTES["notes"]}, "differs"),
        ([json.dumps(NOTES), "more"], NOTES, "not a single text block (2 content blocks)"),
        ([], NOTES, "not a single text block (0 content blocks)"),
    ],
    ids=["not-json", "other-json", "type-differs", "two-blocks", "no-content"],
)
@pytest.mark.parametrize("policy", list(MismatchPolicy))
async def test_text_that_is_not_a_mirror_is_left_alone_and_the_placement_is_partial(
    content, structured, reason, policy
):
    upstream = _text_result(*content, structured=structured)

    result, mcp, recorder = await _forward(upstream, _structured_instruction(mismatch_policy=policy))

    # Never a guess: the text is what the upstream sent, even under fallback_append.
    assert _texts(result) == content
    assert result.structuredContent == _noted("first INJECTED", structured)
    [outcome] = mcp.injection_outcomes
    assert outcome.status is OutcomeStatus.PARTIAL
    assert outcome.detail.startswith("exact selector matched structured content only; ")
    assert reason in outcome.detail
    [finish] = recorder.finishes
    assert (finish["result"], finish["structured_content"]) == ("\n".join(content), result.structuredContent)
    [record] = finish["injection_outcomes"]
    assert (record["status"], record["detail"]) == ("partial", outcome.detail)


@pytest.mark.asyncio
async def test_size_limit_is_checked_before_the_mirror_is_touched():
    upstream = _text_result(json.dumps(NOTES, separators=(",", ":")), structured=NOTES)

    fits, _, _ = await _forward(upstream, _structured_instruction(size_limit=len(b" INJECTED")))
    too_large, mcp, recorder = await _forward(upstream, _structured_instruction(size_limit=len(b" INJECTED") - 1))

    assert json.loads(_texts(fits)[0]) == fits.structuredContent == _noted("first INJECTED")
    assert (too_large.content, too_large.structuredContent) == (upstream.content, NOTES)
    assert mcp.injection_outcomes[0].status is OutcomeStatus.TOO_LARGE
    [finish] = recorder.finishes
    assert (finish["structured_content"], finish["injection_outcomes"][0]["status"]) == (NOTES, "too_large")


@pytest.mark.asyncio
@pytest.mark.parametrize("mirrored", [True, False], ids=["mirror", "not-a-mirror"])
async def test_an_error_result_is_placed_and_recorded_like_any_other(mirrored):
    text = json.dumps(NOTES) if mirrored else "Error: see structured content"
    upstream = _text_result(text, structured=NOTES, is_error=True)

    result, mcp, recorder = await _forward(upstream, _structured_instruction())

    assert result.isError is True
    assert result.structuredContent == _noted("first INJECTED")
    assert _texts(result) == ([json.dumps(result.structuredContent)] if mirrored else [text])
    assert mcp.injection_outcomes[0].status is (OutcomeStatus.APPLIED if mirrored else OutcomeStatus.PARTIAL)
    [finish] = recorder.finishes
    assert finish["result"] == finish["error"] == _texts(result)[0]
    assert finish["structured_content"] == result.structuredContent


@pytest.mark.asyncio
async def test_a_text_only_result_is_placed_and_recorded_as_before():
    result, mcp, recorder = await _forward(_text_result("plain"), _instruction())

    assert (_texts(result), result.structuredContent) == (["plain INJECTED"], None)
    [outcome] = mcp.injection_outcomes
    assert (outcome.status, outcome.detail) == (OutcomeStatus.APPLIED, "exact selector matched")
    [finish] = recorder.finishes
    assert (finish["result"], finish["structured_content"]) == ("plain INJECTED", None)
    assert [record["status"] for record in finish["injection_outcomes"]] == ["applied"]


@pytest.mark.asyncio
async def test_a_call_without_admitted_instructions_records_no_outcomes():
    target = InjectionTarget(tool="lookup", content_block_index=0)
    plan = InjectionPlan((_instruction(target=target, occurrence=OccurrenceRule(Occurrence.NTH, 2)),))
    mcp, _, recorder = _mcp(plan)
    mcp._upstream = FixedUpstream(_text_result("first"))  # type: ignore[assignment]

    await mcp._forward_call("lookup", {})
    await mcp._forward_call("other", {})

    skipped, unmatched_tool = recorder.finishes
    assert [record["status"] for record in skipped["injection_outcomes"]] == ["skipped_occurrence"]
    assert unmatched_tool["injection_outcomes"] == []


@pytest.mark.asyncio
async def test_a_mirror_in_its_own_key_order_is_re_serialised_in_the_structured_contents_order():
    structured = {"b": "x", "notes": [{"id": "NOTE-1", "body": "first"}], "a": 1}
    text = '{"a":1,"notes":[{"body":"first","id":"NOTE-1"}],"b":"x"}'

    result, mcp, _ = await _forward(_text_result(text, structured=structured), _structured_instruction())

    # The text's own key order can't be reproduced: the payload reaches it, in the structured content's order.
    expected = _noted("first INJECTED", structured)
    assert _texts(result) == [json.dumps(expected, separators=(",", ":"), ensure_ascii=False)]
    assert mcp.injection_outcomes[0].status is OutcomeStatus.APPLIED


@pytest.mark.asyncio
async def test_a_mirror_with_a_trailing_newline_loses_it_but_keeps_its_indentation():
    text = json.dumps(NOTES, indent=2, ensure_ascii=False) + "\n"

    result, _, _ = await _forward(_text_result(text, structured=NOTES), _structured_instruction())

    assert _texts(result) == [json.dumps(_noted("first INJECTED"), indent=2, ensure_ascii=False)]


@pytest.mark.asyncio
async def test_an_all_ascii_mirror_gets_a_non_ascii_payload_unescaped():
    structured = {"notes": [{"id": "NOTE-1", "body": "first"}]}
    text = json.dumps(structured, separators=(",", ":"))

    result, _, _ = await _forward(_text_result(text, structured=structured), _structured_instruction(payload=" ü"))

    [sent] = _texts(result)
    assert "first ü" in sent
    assert json.loads(sent) == result.structuredContent == _noted("first ü", structured)


# --- the call-argument predicate ---

ITEM = ("item",)
ORD_102 = (("order_id", "ORD-102"),)


class OrdersUpstream:
    """A single-record lookup: the order the call names, with its text mirroring the structured content compactly."""

    async def call_tool_mcp(self, _name: str, args: dict[str, Any]) -> CallToolResult:
        order = {"order_id": args.get("order_id"), "item": f"item of {args.get('order_id')}"}
        text = json.dumps(order, separators=(",", ":"))
        return CallToolResult(content=[TextContent(type="text", text=text)], structuredContent=order)


def _order_instruction(call_arguments: Any = ORD_102, **changes: Any) -> InjectionInstruction:
    target = InjectionTarget(
        tool="lookup_order",
        surface=SelectorSurface.STRUCTURED_CONTENT,
        structured_content_path=ITEM,
        call_arguments=call_arguments,
    )
    return _instruction(target=target, **changes)


async def _lookups(
    *calls: dict[str, Any], instruction: InjectionInstruction
) -> tuple[list[CallToolResult], MidojoMCP, DetailedCapturingRecorder]:
    mcp, _, recorder = _mcp(InjectionPlan((instruction,)))
    mcp._upstream = OrdersUpstream()  # type: ignore[assignment]
    results = [_wire_result(await mcp._forward_call("lookup_order", args)) for args in calls]
    return results, mcp, recorder


def _item(result: CallToolResult) -> str:
    return _structured(result)["item"]


def test_call_arguments_are_sorted_hashable_pairs_of_exact_json_scalars():
    target = InjectionTarget(tool="lookup", call_arguments=(("b", 1), ("a", True), ("c", None), ("d", 1.5)))
    assert target.call_arguments == (("a", True), ("b", 1), ("c", None), ("d", 1.5))
    assert InjectionTarget(tool="lookup", call_arguments={"b": 1, "a": True, "c": None, "d": 1.5}) == target
    assert hash(target) == hash(dataclasses.replace(target))
    assert InjectionTarget().call_arguments == ()
    invalid = [
        ({"tool": "lookup", "call_arguments": (("order", ["ORD-1"]),)}, TypeError, "exact str, int, float"),
        ({"tool": "lookup", "call_arguments": (("order", {"id": 1}),)}, TypeError, "exact str, int, float"),
        ({"tool": "lookup", "call_arguments": (("", "x"),)}, TypeError, "non-empty str"),
        ({"tool": "lookup", "call_arguments": ((1, "x"),)}, TypeError, "non-empty str"),
        ({"tool": "lookup", "call_arguments": (("n", float("inf")),)}, ValueError, "finite"),
        ({"tool": "lookup", "call_arguments": (("n", 1), ("n", 2))}, ValueError, "twice"),
        ({"tool": "lookup", "call_arguments": "order_id"}, TypeError, "mapping or a sequence"),
        ({"tool": "lookup", "call_arguments": (("order_id",),)}, TypeError, "pair"),
        ({"call_arguments": ORD_102}, ValueError, "need a target tool"),
        (
            {"surface": SelectorSurface.DESCRIPTION, "tool": "lookup", "call_arguments": ORD_102},
            ValueError,
            "only select tool-output calls",
        ),
    ]
    for fields, error, message in invalid:
        with pytest.raises(error, match=message):
            InjectionTarget(**fields)


@pytest.mark.parametrize(
    ("arguments", "mismatch"),
    [
        ({"order_id": "ORD-102", "verbose": True}, None),
        ({"order_id": "ORD-101"}, "call argument 'order_id' is 'ORD-101', not 'ORD-102'"),
        ({}, "call argument 'order_id' is missing"),
        ({"order": "ORD-102"}, "call argument 'order_id' is missing"),
        ({"order_id": None}, "call argument 'order_id' is None, not 'ORD-102'"),
    ],
    ids=["match", "different", "missing", "other-name", "null"],
)
def test_call_argument_mismatch_names_the_argument_and_why(arguments, mismatch):
    target = InjectionTarget(tool="lookup_order", call_arguments=ORD_102)
    assert target.call_argument_mismatch(arguments) == mismatch


@pytest.mark.parametrize(
    ("expected", "actual"),
    [("1", 1), (1, "1"), (1, True), (True, 1), (1, 1.0), (1.0, 1), (0, False), (None, "")],
)
def test_call_argument_matching_is_type_exact(expected, actual):
    target = InjectionTarget(tool="lookup", call_arguments=(("n", expected),))
    assert target.call_argument_mismatch({"n": expected}) is None
    assert target.call_argument_mismatch({"n": actual}) == f"call argument 'n' is {actual!r}, not {expected!r}"


def test_every_named_argument_must_match_and_each_failure_is_named():
    target = InjectionTarget(tool="lookup", call_arguments={"a": 1, "b": "x", "c": False})
    assert target.call_argument_mismatch({"a": 1, "b": "x", "c": False}) is None
    assert target.call_argument_mismatch({"a": 2, "c": False}) == (
        "call argument 'a' is 2, not 1; call argument 'b' is missing"
    )


def test_plan_json_round_trip_carries_call_arguments_through_an_episode():
    instruction = _order_instruction({"order_id": "ORD-102", "count": 2, "gift": False, "note": None})
    plan = InjectionPlan((instruction,))
    encoded = plan.to_json()
    data = plan.to_data()

    assert data["instructions"][0]["target"]["call_arguments"] == {
        "count": 2,
        "gift": False,
        "note": None,
        "order_id": "ORD-102",
    }
    assert InjectionPlan.from_json(encoded) == InjectionPlan.from_data(data) == plan == Episode(plan=plan).plan
    assert InjectionPlan.from_json(encoded).to_json() == encoded


def test_plan_data_without_call_arguments_decodes_and_encodes_as_before():
    plan = InjectionPlan((_instruction(), _order_instruction(())))
    data = plan.to_data()

    # An empty predicate leaves the key out: plans that never named one encode exactly as before.
    assert all("call_arguments" not in item["target"] for item in data["instructions"])
    assert list(data["instructions"][0]["target"]) == [
        "tool",
        "surface",
        "content_block_index",
        "content_block_type",
        "structured_content_path",
        "template_path",
        "template_key",
    ]
    decoded = InjectionPlan.from_data(data)
    assert decoded == plan
    assert [item.target.call_arguments for item in decoded] == [(), ()]
    assert decoded.to_json() == plan.to_json()


@pytest.mark.parametrize(
    "call_arguments",
    [["order_id", "ORD-102"], {"order_id": ["ORD-102"]}, {"order_id": {"id": "ORD-102"}}, "ORD-102"],
    ids=["array", "array-value", "object-value", "string"],
)
def test_plan_data_refuses_call_arguments_that_are_not_an_object_of_scalars(call_arguments):
    data = InjectionPlan((_order_instruction(),)).to_data()
    data["instructions"][0]["target"]["call_arguments"] = call_arguments
    with pytest.raises(ValueError, match="invalid injection plan data"):
        InjectionPlan.from_data(data)


def test_an_interceptor_without_call_argument_matching_reports_a_capability_gap():
    plan = InjectionPlan((_order_instruction({"order_id": "ORD-102", "count": 2}),))
    unaware = dataclasses.replace(MIDOJO_MCP_CAPABILITY, call_arguments=False)

    assert [(gap.aspect, gap.value) for gap in capability_gaps(plan, unaware)] == [("call_arguments", "count,order_id")]
    assert capability_gaps(plan, MIDOJO_MCP_CAPABILITY) == ()
    assert capability_gaps(InjectionPlan((_instruction(),)), unaware) == ()


@pytest.mark.asyncio
async def test_only_the_call_with_matching_arguments_is_changed_and_the_others_say_why():
    instruction = _order_instruction(occurrence=OccurrenceRule(Occurrence.EVERY), mode=InjectionMode.REPLACE)

    results, mcp, recorder = await _lookups(
        {"order_id": "ORD-101"}, {}, {"order_id": "ORD-102"}, {"order_id": "ORD-103"}, instruction=instruction
    )

    assert [_item(result) for result in results] == ["item of ORD-101", "item of None", " INJECTED", "item of ORD-103"]
    assert [(o.status, o.occurrence, o.admission_order, o.detail) for o in mcp.injection_outcomes] == [
        (OutcomeStatus.UNMATCHED, 0, 1, "call argument 'order_id' is 'ORD-101', not 'ORD-102'; result unchanged"),
        (OutcomeStatus.UNMATCHED, 0, 2, "call argument 'order_id' is missing; result unchanged"),
        (OutcomeStatus.APPLIED, 1, 3, MIRRORED),
        (OutcomeStatus.UNMATCHED, 0, 4, "call argument 'order_id' is 'ORD-103', not 'ORD-102'; result unchanged"),
    ]
    # Each call's record carries its own outcome, with what the agent got. A call whose arguments don't match
    # placed nothing: nothing selected, transformed or emitted.
    placed = (True, "identity", ["structured_content", "content_block"])
    facts = [(False, None, []), (False, None, []), placed, (False, None, [])]
    assert [finish["injection_outcomes"] for finish in recorder.finishes] == [
        [
            {
                "instruction_index": 0,
                "probe_key": None,
                "occurrence": o.occurrence,
                "status": o.status,
                "detail": o.detail,
                "selected": selected,
                "transformation": transformation,
                "emitted": emitted,
                **(_placed(instruction) if selected else {}),
            }
        ]
        for o, (selected, transformation, emitted) in zip(mcp.injection_outcomes, facts, strict=True)
    ]
    assert [finish["structured_content"] for finish in recorder.finishes] == [r.structuredContent for r in results]


@pytest.mark.asyncio
async def test_a_call_to_another_tool_records_no_argument_outcome():
    mcp, _, recorder = _mcp(InjectionPlan((_order_instruction(),)))
    mcp._upstream = OrdersUpstream()  # type: ignore[assignment]

    await mcp._forward_call("lookup_customer", {"customer_id": "C-1"})

    assert mcp.injection_outcomes == ()
    assert recorder.finishes[0]["injection_outcomes"] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rule", "injected"),
    [
        (OccurrenceRule(Occurrence.FIRST), [False, False, True, False, False]),
        (OccurrenceRule(Occurrence.NTH, 2), [False, False, False, False, True]),
        (OccurrenceRule(Occurrence.EVERY), [False, False, True, False, True]),
    ],
    ids=["first", "nth-2", "every"],
)
async def test_occurrences_count_only_calls_whose_arguments_match(rule, injected):
    calls = ("ORD-101", "ORD-103", "ORD-102", "ORD-101", "ORD-102")

    results, mcp, _ = await _lookups(
        *({"order_id": order} for order in calls), instruction=_order_instruction(occurrence=rule)
    )

    assert [_item(result).endswith(" INJECTED") for result in results] == injected
    # The non-matching calls are occurrence 0; the matching ones are numbered 1 and 2 whatever came between.
    assert [outcome.occurrence for outcome in mcp.injection_outcomes] == [0, 0, 1, 0, 2]
    statuses = [outcome.status for outcome in mcp.injection_outcomes]
    assert [status is OutcomeStatus.UNMATCHED for status in statuses] == [True, True, False, True, False]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "item"), [(InjectionMode.APPEND, "item of ORD-102 INJECTED"), (InjectionMode.REPLACE, " INJECTED")]
)
async def test_a_matched_calls_mirrored_text_is_rewritten_and_an_unmatched_ones_is_untouched(mode, item):
    instruction = _order_instruction(occurrence=OccurrenceRule(Occurrence.EVERY), mode=mode)

    matched, unmatched = (await _lookups({"order_id": "ORD-102"}, {"order_id": "ORD-101"}, instruction=instruction))[0]

    assert _structured(matched) == {"order_id": "ORD-102", "item": item}
    assert _texts(matched) == [json.dumps({"order_id": "ORD-102", "item": item}, separators=(",", ":"))]
    assert _structured(unmatched) == {"order_id": "ORD-101", "item": "item of ORD-101"}
    assert _texts(unmatched) == ['{"order_id":"ORD-101","item":"item of ORD-101"}']


@pytest.mark.asyncio
async def test_a_content_block_target_honours_call_arguments_too():
    target = InjectionTarget(tool="lookup_order", content_block_index=0, call_arguments=ORD_102)
    instruction = _instruction(target=target, occurrence=OccurrenceRule(Occurrence.EVERY))

    matched, unmatched = (await _lookups({"order_id": "ORD-102"}, {"order_id": "ORD-101"}, instruction=instruction))[0]

    assert _texts(matched) == ['{"order_id":"ORD-102","item":"item of ORD-102"} INJECTED']
    assert _texts(unmatched) == ['{"order_id":"ORD-101","item":"item of ORD-101"}']


@pytest.mark.asyncio
async def test_size_limit_applies_to_the_matched_call_only():
    instruction = _order_instruction(occurrence=OccurrenceRule(Occurrence.EVERY), size_limit=len(b" INJECTED") - 1)

    results, mcp, _ = await _lookups({"order_id": "ORD-101"}, {"order_id": "ORD-102"}, instruction=instruction)

    assert [_item(result) for result in results] == ["item of ORD-101", "item of ORD-102"]
    assert [(o.status, o.occurrence) for o in mcp.injection_outcomes] == [
        (OutcomeStatus.UNMATCHED, 0),
        (OutcomeStatus.TOO_LARGE, 1),
    ]

    fits, _, _ = await _lookups({"order_id": "ORD-102"}, instruction=_order_instruction(size_limit=len(b" INJECTED")))
    assert _item(fits[0]) == "item of ORD-102 INJECTED"


@pytest.mark.asyncio
async def test_fallback_append_never_places_a_payload_on_a_call_whose_arguments_dont_match():
    instruction = _order_instruction(mismatch_policy=MismatchPolicy.FALLBACK_APPEND)

    [result], mcp, _ = await _lookups({"order_id": "ORD-101"}, instruction=instruction)

    assert _texts(result) == ['{"order_id":"ORD-101","item":"item of ORD-101"}']
    [outcome] = mcp.injection_outcomes
    assert (outcome.status, outcome.occurrence) == (OutcomeStatus.UNMATCHED, 0)


def _outcome_at(occurrence: int, status: OutcomeStatus, **facts: Any) -> InjectionOutcome:
    return InjectionOutcome(0, None, "scope", "c1", "lookup_order", occurrence, 1, status, "d", **facts)


@pytest.mark.parametrize(
    ("status", "facts"),
    [
        (OutcomeStatus.APPLIED, {"selected": True, "transformation": Transformation.IDENTITY}),
        (OutcomeStatus.SKIPPED_OCCURRENCE, {}),
        (OutcomeStatus.NO_RESULT, {}),
        (OutcomeStatus.UNMATCHED, {"selected": True}),
        (OutcomeStatus.UNMATCHED, {"transformation": Transformation.IDENTITY}),
        (OutcomeStatus.UNMATCHED, {"emitted": (SelectorSurface.CONTENT_BLOCK,)}),
    ],
    ids=["applied", "skipped", "no-result", "unmatched-selected", "unmatched-transformed", "unmatched-emitted"],
)
def test_only_an_unmatched_outcome_that_placed_nothing_has_occurrence_0(status, facts):
    with pytest.raises(ValueError, match="occurrence 0"):
        _outcome_at(0, status, **facts)
    # Counted, the same outcome is valid: an unmatched one too (a strict selector that didn't match).
    assert _outcome_at(1, status, **facts).occurrence == 1
    assert _outcome_at(0, OutcomeStatus.UNMATCHED).occurrence == 0


@pytest.mark.asyncio
async def test_midojo_mcp_reports_occurrence_0_only_for_a_call_argument_miss():
    # A strict selector that misses the matching calls (no "missing" field): unmatched, but counted.
    target = InjectionTarget(
        tool="lookup_order",
        surface=SelectorSurface.STRUCTURED_CONTENT,
        structured_content_path=("missing",),
        call_arguments={"order_id": "ORD-102"},
    )
    instruction = _instruction(target=target, occurrence=OccurrenceRule(Occurrence.NTH, 2))
    calls = ("ORD-101", "ORD-102", "ORD-102", "ORD-102")

    _, mcp, recorder = await _lookups(*({"order_id": order} for order in calls), instruction=instruction)

    assert [(o.occurrence, o.status, o.selected, o.transformation, o.emitted) for o in mcp.injection_outcomes] == [
        (0, OutcomeStatus.UNMATCHED, False, None, ()),
        (1, OutcomeStatus.SKIPPED_OCCURRENCE, False, None, ()),
        (2, OutcomeStatus.UNMATCHED, False, Transformation.IDENTITY, ()),
        (3, OutcomeStatus.SKIPPED_OCCURRENCE, False, None, ()),
    ]
    # And the control plane takes every record it sent.
    records = [record for finish in recorder.finishes for record in finish["injection_outcomes"]]
    assert [CallInjectionOutcome.model_validate(record).occurrence for record in records] == [0, 1, 2, 3]


# --- delivery facts: what the interceptor selected, transformed and emitted ---


def _facts(record: InjectionRecord) -> tuple[str, bool | None, str | None, list[str] | None]:
    return (record["status"], record.get("selected"), record.get("transformation"), record.get("emitted"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "instruction", "facts"),
    [
        (_text_result("plain"), _instruction(), ("applied", True, "identity", ["content_block"])),
        (
            _text_result("plain"),
            _instruction(transformation=Transformation.JSON_STRING),
            ("applied", True, "json_string", ["content_block"]),
        ),
        (
            _text_result(json.dumps(NOTES), structured=NOTES),
            _structured_instruction(),
            ("applied", True, "identity", ["structured_content", "content_block"]),
        ),
        (
            _text_result("Notes: first", structured=NOTES),
            _structured_instruction(),
            ("partial", True, "identity", ["structured_content"]),
        ),
        (_text_result(), _instruction(), ("unmatched", False, "identity", [])),
        (
            _text_result(),
            _instruction(mismatch_policy=MismatchPolicy.FALLBACK_APPEND),
            ("fallback", False, "identity", ["content_block"]),
        ),
        (_text_result("plain"), _instruction(size_limit=1), ("too_large", False, "identity", [])),
    ],
    ids=["content", "json-string", "mirror", "partial", "unmatched", "fallback", "too-large"],
)
async def test_each_outcome_records_what_was_selected_transformed_and_emitted(result, instruction, facts):
    _, mcp, recorder = await _forward(result, instruction)

    [finish] = recorder.finishes
    [record] = finish["injection_outcomes"]
    assert _facts(record) == facts
    [outcome] = mcp.injection_outcomes
    assert (outcome.status.value, outcome.selected, outcome.transformation, [s.value for s in outcome.emitted]) == (
        facts[0],
        facts[1],
        Transformation(facts[2]),
        facts[3],
    )


@pytest.mark.asyncio
async def test_a_payload_in_a_json_mirror_is_emitted_in_both_parts_though_the_text_escapes_it():
    payload = ' "quoted"\nnext line ü'

    result, _, recorder = await _forward(
        _text_result(json.dumps(NOTES, ensure_ascii=True), structured=NOTES), _structured_instruction(payload=payload)
    )

    [text] = _texts(result)
    # The text carries the payload JSON-encoded, never as its raw characters: emission doesn't search for it.
    assert payload not in text
    assert json.loads(text)["notes"][0]["body"] == "first" + payload
    [record] = recorder.finishes[0]["injection_outcomes"]
    assert _facts(record) == ("applied", True, "identity", ["structured_content", "content_block"])


@pytest.mark.asyncio
async def test_a_skipped_occurrence_or_a_missing_result_was_never_transformed():
    target = InjectionTarget(tool="lookup", content_block_index=0)
    plan = InjectionPlan((_instruction(target=target, occurrence=OccurrenceRule(Occurrence.NTH, 2)),))
    mcp, _, recorder = _mcp(plan)
    mcp._upstream = FixedUpstream(_text_result("first"))  # type: ignore[assignment]
    await mcp._forward_call("lookup", {})

    class FailingUpstream:
        async def call_tool_mcp(self, name: str, args: dict[str, Any]) -> CallToolResult:
            raise RuntimeError("upstream failed")

    mcp._upstream = FailingUpstream()  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        await mcp._forward_call("lookup", {})

    skipped, failed = (finish["injection_outcomes"] for finish in recorder.finishes)
    assert [_facts(record) for record in skipped] == [("skipped_occurrence", False, None, [])]
    assert [_facts(record) for record in failed] == [("no_result", False, None, [])]


@pytest.mark.asyncio
async def test_a_later_instruction_that_replaces_the_text_leaves_the_earlier_one_unemitted():
    replace_all = _instruction(target=InjectionTarget(), payload="REPLACED", mode=InjectionMode.REPLACE)

    result, _, recorder = await _forward(_text_result("plain"), _instruction(), replace_all)

    assert _texts(result) == ["REPLACED"]
    first, second = recorder.finishes[0]["injection_outcomes"]
    # The first placement was applied, but the result returned doesn't carry it.
    assert _facts(first) == ("applied", True, "identity", [])
    assert _facts(second) == ("applied", True, "identity", ["content_block"])


@pytest.mark.asyncio
async def test_text_appended_after_a_mirror_keeps_the_mirrored_placement_emitted():
    appended = _instruction(target=InjectionTarget(content_block_index=0), payload=" TAIL")

    result, _, recorder = await _forward(
        _text_result(json.dumps(NOTES), structured=NOTES), _structured_instruction(), appended
    )

    [text] = _texts(result)
    assert text.endswith(" TAIL")
    mirrored, tail = recorder.finishes[0]["injection_outcomes"]
    # The text is no longer a mirror, but it still holds the mirror text the first instruction wrote.
    assert _facts(mirrored) == ("applied", True, "identity", ["structured_content", "content_block"])
    assert _facts(tail) == ("applied", True, "identity", ["content_block"])


@pytest.mark.asyncio
async def test_the_same_text_in_another_block_doesnt_make_a_replaced_placement_emitted():
    first = _instruction(target=InjectionTarget(content_block_index=0), payload="X", mode=InjectionMode.REPLACE)
    second = _instruction(target=InjectionTarget(content_block_index=0), payload="Y", mode=InjectionMode.REPLACE)

    # The upstream's second block already holds the first payload's text, where no instruction put it.
    result, _, recorder = await _forward(_text_result("plain", "X"), first, second)

    assert _texts(result) == ["Y", "X"]
    replaced, kept = recorder.finishes[0]["injection_outcomes"]
    assert _facts(replaced) == ("applied", True, "identity", [])
    assert _facts(kept) == ("applied", True, "identity", ["content_block"])


@pytest.mark.asyncio
async def test_the_same_text_appended_by_a_later_instruction_doesnt_make_a_replaced_placement_emitted():
    first = _instruction(target=InjectionTarget(content_block_index=0), payload="X", mode=InjectionMode.REPLACE)
    second = _instruction(target=InjectionTarget(content_block_index=0), payload="Y", mode=InjectionMode.REPLACE)
    third = _instruction(target=InjectionTarget(), payload="X")

    result, _, recorder = await _forward(_text_result("plain"), first, second, third)

    assert _texts(result) == ["Y", "X"]
    replaced, rewritten, appended = recorder.finishes[0]["injection_outcomes"]
    assert _facts(replaced) == ("applied", True, "identity", [])
    assert _facts(rewritten) == ("applied", True, "identity", ["content_block"])
    assert _facts(appended) == ("applied", True, "identity", ["content_block"])


@pytest.mark.asyncio
async def test_a_mirror_rewritten_while_another_block_holds_its_text_is_not_emitted_in_the_text():
    mirror_text = json.dumps(_noted("first INJECTED"))
    # A later instruction appends a block holding exactly the mirror's text, and another rewrites the mirror.
    appended = _instruction(target=InjectionTarget(), payload=mirror_text)
    rewrite = _instruction(target=InjectionTarget(content_block_index=0), payload="Y", mode=InjectionMode.REPLACE)

    result, _, recorder = await _forward(
        _text_result(json.dumps(NOTES), structured=NOTES), _structured_instruction(), appended, rewrite
    )

    assert _texts(result) == ["Y", mirror_text]
    mirrored, copied, rewritten = recorder.finishes[0]["injection_outcomes"]
    # The structured field still carries the first payload; the text it was mirrored into doesn't.
    assert _facts(mirrored) == ("applied", True, "identity", ["structured_content"])
    assert _facts(copied) == ("applied", True, "identity", ["content_block"])
    assert _facts(rewritten) == ("applied", True, "identity", ["content_block"])


# A structured placement counts only in the field it wrote, while no later instruction replaced that field.


def _replace_body(path: tuple = NOTE_BODY, payload: str = " INJECTED") -> InjectionInstruction:
    target = InjectionTarget(surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=path)
    return _instruction(target=target, payload=payload, mode=InjectionMode.REPLACE)


@pytest.mark.asyncio
async def test_a_field_replaced_with_the_same_payload_is_emitted_only_by_the_later_instruction():
    result, _, recorder = await _forward(
        _text_result("Notes: first", structured=NOTES), _replace_body(), _replace_body()
    )

    assert _structured(result)["notes"][0]["body"] == " INJECTED"
    overwritten, kept = recorder.finishes[0]["injection_outcomes"]
    assert _facts(overwritten) == ("partial", True, "identity", [])
    assert _facts(kept) == ("partial", True, "identity", ["structured_content"])


@pytest.mark.asyncio
async def test_a_mirrored_field_replaced_with_the_same_payload_is_emitted_only_by_the_later_instruction():
    result, _, recorder = await _forward(
        _text_result(json.dumps(NOTES), structured=NOTES), _replace_body(), _replace_body()
    )

    [text] = _texts(result)
    assert json.loads(text) == _structured(result) == _noted(" INJECTED")
    overwritten, kept = recorder.finishes[0]["injection_outcomes"]
    # The mirror text follows the field: the earlier instruction's write was replaced in both parts.
    assert _facts(overwritten) == ("applied", True, "identity", [])
    assert _facts(kept) == ("applied", True, "identity", ["structured_content", "content_block"])


@pytest.mark.asyncio
async def test_the_same_field_selected_by_record_and_by_index_is_one_field():
    by_index = _replace_body(path=("notes", 0, "body"))

    _, _, recorder = await _forward(_text_result(json.dumps(NOTES), structured=NOTES), _replace_body(), by_index)

    by_record, later = recorder.finishes[0]["injection_outcomes"]
    assert _facts(by_record) == ("applied", True, "identity", [])
    assert _facts(later) == ("applied", True, "identity", ["structured_content", "content_block"])


@pytest.mark.asyncio
async def test_two_appends_to_one_field_are_both_emitted_while_both_survive():
    result, _, recorder = await _forward(
        _text_result(json.dumps(NOTES), structured=NOTES), _structured_instruction(), _structured_instruction()
    )

    assert _structured(result)["notes"][0]["body"] == "first INJECTED INJECTED"
    first, second = recorder.finishes[0]["injection_outcomes"]
    assert _facts(first) == _facts(second) == ("applied", True, "identity", ["structured_content", "content_block"])


@pytest.mark.asyncio
async def test_appends_a_later_replace_overwrote_are_not_emitted_though_it_wrote_the_same_text():
    # The replace writes exactly what the two appends had made of the field.
    replace = _replace_body(payload="first INJECTED INJECTED")

    result, _, recorder = await _forward(
        _text_result(json.dumps(NOTES), structured=NOTES), _structured_instruction(), _structured_instruction(), replace
    )

    assert _structured(result)["notes"][0]["body"] == "first INJECTED INJECTED"
    first, second, last = recorder.finishes[0]["injection_outcomes"]
    assert _facts(first) == _facts(second) == ("applied", True, "identity", [])
    assert _facts(last) == ("applied", True, "identity", ["structured_content", "content_block"])


# --- the result-record predicate: a single-object result chosen by a field of the result ---

REF_1 = RecordSelector("REF-1", field="referral_id")
REFERRALS = {"P-1": "REF-1", "P-2": "REF-2", "P-3": "REF-1"}


class ReferralsUpstream:
    """``get_referral(patient_id)``: one referral, whose id the call doesn't name, mirrored compactly in its text."""

    async def call_tool_mcp(self, _name: str, args: dict[str, Any]) -> CallToolResult:
        patient = args.get("patient_id")
        referral = {"patient_id": patient, "reason": f"reason for {patient}"}
        if patient in REFERRALS:
            referral = {"referral_id": REFERRALS[patient]} | referral
        text = json.dumps(referral, separators=(",", ":"))
        return CallToolResult(content=[TextContent(type="text", text=text)], structuredContent=referral)


class QueuedUpstream:
    """Answers each call with the next of its results."""

    def __init__(self, *results: CallToolResult) -> None:
        self.results = list(results)

    async def call_tool_mcp(self, _name: str, _args: dict[str, Any]) -> CallToolResult:
        return self.results.pop(0)


def _referral_instruction(result_record: Any = REF_1, **changes: Any) -> InjectionInstruction:
    target_fields = {
        "tool": "get_referral",
        "surface": SelectorSurface.STRUCTURED_CONTENT,
        "structured_content_path": ("reason",),
        "result_record": result_record,
    } | changes.pop("target", {})
    return _instruction(target=InjectionTarget(**target_fields), **changes)


async def _referrals(
    *patients: str | None, instruction: InjectionInstruction, upstream: Any = None
) -> tuple[list[CallToolResult], MidojoMCP, DetailedCapturingRecorder]:
    mcp, _, recorder = _mcp(InjectionPlan((instruction,)))
    mcp._upstream = upstream or ReferralsUpstream()  # type: ignore[assignment]
    results = []
    for patient in patients:
        arguments = {} if patient is None else {"patient_id": patient}
        results.append(_wire_result(await mcp._forward_call("get_referral", arguments)))
    return results, mcp, recorder


def _reason(result: CallToolResult) -> str:
    return _structured(result)["reason"]


def test_result_record_is_a_record_selector_on_a_tool_output_target_with_a_tool():
    target = InjectionTarget(tool="get_referral", result_record=REF_1)
    assert target.result_record == RecordSelector("REF-1", "referral_id")
    assert hash(target) == hash(dataclasses.replace(target))
    assert InjectionTarget().result_record is None
    # It combines with call_arguments.
    both = InjectionTarget(tool="get_referral", call_arguments={"patient_id": "P-1"}, result_record=REF_1)
    assert (both.call_arguments, both.result_record) == ((("patient_id", "P-1"),), REF_1)
    invalid = [
        ({"tool": "get_referral", "result_record": {"field": "referral_id", "equals": "REF-1"}}, TypeError, "Record"),
        ({"tool": "get_referral", "result_record": "REF-1"}, TypeError, "RecordSelector or None"),
        ({"result_record": REF_1}, ValueError, "needs a target tool"),
        (
            {"surface": SelectorSurface.DESCRIPTION, "tool": "get_referral", "result_record": REF_1},
            ValueError,
            "only selects tool-output results",
        ),
        (
            {"surface": SelectorSurface.TEMPLATE, "template_key": "t:main", "result_record": REF_1},
            ValueError,
            "only selects tool-output results",
        ),
    ]
    for fields, error, message in invalid:
        with pytest.raises(error, match=message):
            InjectionTarget(**fields)


@pytest.mark.parametrize(
    ("selector", "structured", "mismatch"),
    [
        (REF_1, {"referral_id": "REF-1", "reason": "x"}, None),
        (REF_1, {"referral_id": "REF-2"}, "result field 'referral_id' is 'REF-2', not 'REF-1'"),
        (REF_1, {"template_id": "REF-1"}, "result field 'referral_id' is missing"),
        (REF_1, {"referral_id": None}, "result field 'referral_id' is None, not 'REF-1'"),
        (RecordSelector(1, "referral_id"), {"referral_id": "1"}, "result field 'referral_id' is '1', not 1"),
        (RecordSelector("1", "referral_id"), {"referral_id": 1}, "result field 'referral_id' is 1, not '1'"),
        (RecordSelector(1, "referral_id"), {"referral_id": True}, "result field 'referral_id' is True, not 1"),
        (RecordSelector(7), {"id": 7}, None),
        (REF_1, None, "result has no structured content to match field 'referral_id'"),
        (
            REF_1,
            [{"referral_id": "REF-1"}],
            "result structured content is list, not an object with field 'referral_id'",
        ),
    ],
    ids=["match", "different", "missing", "null", "int-vs-str", "str-vs-int", "bool-vs-int", "id", "none", "list"],
)
def test_result_record_mismatch_names_the_field_and_why(selector, structured, mismatch):
    target = InjectionTarget(tool="get_referral", result_record=selector)
    assert target.result_record_mismatch(structured) == mismatch


def test_a_target_without_result_record_matches_any_result():
    assert InjectionTarget(tool="get_referral").result_record_mismatch(None) is None


def test_plan_json_round_trip_carries_result_record_through_an_episode():
    instruction = _referral_instruction(target={"call_arguments": {"patient_id": "P-1"}})
    plan = InjectionPlan((instruction, _referral_instruction(RecordSelector(7))))
    encoded = plan.to_json()
    data = plan.to_data()

    targets = [item["target"] for item in data["instructions"]]
    assert [target["result_record"] for target in targets] == [
        {"field": "referral_id", "equals": "REF-1"},
        {"field": "id", "equals": 7},
    ]
    assert targets[0]["call_arguments"] == {"patient_id": "P-1"}
    assert InjectionPlan.from_json(encoded) == InjectionPlan.from_data(data) == plan == Episode(plan=plan).plan
    assert InjectionPlan.from_json(encoded).to_json() == encoded


def test_plan_data_with_result_record_and_no_field_selects_id():
    data = InjectionPlan((_referral_instruction(),)).to_data()
    data["instructions"][0]["target"]["result_record"] = {"equals": "REF-1"}

    [instruction] = InjectionPlan.from_data(data)

    assert instruction.target.result_record == RecordSelector("REF-1")


def test_plan_data_without_result_record_decodes_and_encodes_as_before():
    plan = InjectionPlan((_instruction(), _order_instruction(), _referral_instruction(None)))
    data = plan.to_data()

    assert all("result_record" not in item["target"] for item in data["instructions"])
    decoded = InjectionPlan.from_data(data)
    assert [item.target.result_record for item in decoded] == [None, None, None]
    assert decoded.to_json() == plan.to_json()


@pytest.mark.parametrize(
    "result_record",
    [
        None,
        "REF-1",
        ["referral_id", "REF-1"],
        {"field": "referral_id"},
        {"field": "referral_id", "equals": True},
        {"field": "referral_id", "equals": 1.5},
        {"field": "referral_id", "equals": None},
        {"field": "", "equals": "REF-1"},
        {"field": 1, "equals": "REF-1"},
        {"field": "referral_id", "equals": "REF-1", "nested": True},
    ],
    ids=["null", "string", "array", "no-equals", "bool", "float", "null-value", "empty-field", "int-field", "extra"],
)
def test_plan_data_refuses_a_result_record_that_is_not_a_record_selector(result_record):
    data = InjectionPlan((_referral_instruction(),)).to_data()
    data["instructions"][0]["target"]["result_record"] = result_record
    with pytest.raises(ValueError, match="invalid injection plan data"):
        InjectionPlan.from_data(data)


def test_plan_data_refuses_a_result_record_without_a_tool():
    data = InjectionPlan((_referral_instruction(),)).to_data()
    data["instructions"][0]["target"]["tool"] = None
    with pytest.raises(ValueError, match="result_record needs a target tool"):
        InjectionPlan.from_data(data)


def test_an_interceptor_without_result_record_matching_reports_a_capability_gap():
    plan = InjectionPlan((_referral_instruction(),))
    unaware = dataclasses.replace(MIDOJO_MCP_CAPABILITY, result_record=False)

    assert MIDOJO_MCP_CAPABILITY.result_record is True
    assert [(gap.aspect, gap.value) for gap in capability_gaps(plan, unaware)] == [("result_record", "referral_id")]
    assert capability_gaps(plan, MIDOJO_MCP_CAPABILITY) == ()
    assert capability_gaps(InjectionPlan((_order_instruction(),)), unaware) == ()
    assert "instruction 0 result_record='referral_id'" in str(CapabilityError(capability_gaps(plan, unaware)))


@pytest.mark.asyncio
async def test_only_the_result_whose_field_matches_is_changed_and_the_others_say_why():
    instruction = _referral_instruction(occurrence=OccurrenceRule(Occurrence.EVERY), mode=InjectionMode.REPLACE)

    # P-2 answers REF-2, P-4 a referral with no referral_id, P-1 REF-1.
    results, mcp, recorder = await _referrals("P-2", "P-4", "P-1", instruction=instruction)

    assert [_reason(result) for result in results] == ["reason for P-2", "reason for P-4", " INJECTED"]
    assert [_texts(result) for result in results][2] == [
        json.dumps({"referral_id": "REF-1", "patient_id": "P-1", "reason": " INJECTED"}, separators=(",", ":"))
    ]
    assert [(o.status, o.occurrence, o.admission_order, o.detail) for o in mcp.injection_outcomes] == [
        (OutcomeStatus.UNMATCHED, 0, 1, "result field 'referral_id' is 'REF-2', not 'REF-1'; result unchanged"),
        (OutcomeStatus.UNMATCHED, 0, 2, "result field 'referral_id' is missing; result unchanged"),
        (OutcomeStatus.APPLIED, 1, 3, MIRRORED),
    ]
    # A result that doesn't match placed nothing: nothing selected, transformed or emitted.
    assert [_facts(finish["injection_outcomes"][0]) for finish in recorder.finishes] == [
        ("unmatched", False, None, []),
        ("unmatched", False, None, []),
        ("applied", True, "identity", ["structured_content", "content_block"]),
    ]
    assert [finish["structured_content"] for finish in recorder.finishes] == [r.structuredContent for r in results]
    # And the control plane takes every record it sent.
    records = [record for finish in recorder.finishes for record in finish["injection_outcomes"]]
    assert [CallInjectionOutcome.model_validate(record).occurrence for record in records] == [0, 0, 1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rule", "injected"),
    [
        (OccurrenceRule(Occurrence.FIRST), [False, True, False, False, False]),
        (OccurrenceRule(Occurrence.NTH, 2), [False, False, False, True, False]),
        (OccurrenceRule(Occurrence.EVERY), [False, True, False, True, True]),
    ],
    ids=["first", "nth-2", "every"],
)
async def test_occurrences_count_only_results_whose_field_matches(rule, injected):
    # P-1 and P-3 both answer REF-1.
    results, mcp, _ = await _referrals(
        "P-2", "P-1", "P-2", "P-3", "P-1", instruction=_referral_instruction(occurrence=rule)
    )

    assert [_reason(result).endswith(" INJECTED") for result in results] == injected
    assert [outcome.occurrence for outcome in mcp.injection_outcomes] == [0, 1, 0, 2, 3]
    statuses = [outcome.status for outcome in mcp.injection_outcomes]
    assert [status is OutcomeStatus.UNMATCHED for status in statuses] == [True, False, True, False, False]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selector", "result", "detail"),
    [
        (
            RecordSelector(1, "referral_id"),
            _text_result('{"referral_id":"1"}', structured={"referral_id": "1"}),
            "result field 'referral_id' is '1', not 1; result unchanged",
        ),
        (
            RecordSelector("1", "referral_id"),
            _text_result('{"referral_id":1}', structured={"referral_id": 1}),
            "result field 'referral_id' is 1, not '1'; result unchanged",
        ),
        (
            REF_1,
            _text_result("Referral REF-1: reason"),
            "result has no structured content to match field 'referral_id'; result unchanged",
        ),
    ],
    ids=["int-vs-str", "str-vs-int", "no-structured-content"],
)
@pytest.mark.parametrize("policy", list(MismatchPolicy))
@pytest.mark.parametrize("surface", ["structured", "content"])
async def test_a_result_that_cant_match_is_left_unchanged_whatever_the_policy(
    selector, result, detail, policy, surface
):
    target = {"result_record": selector}
    if surface == "content":
        target |= {"surface": SelectorSurface.CONTENT_BLOCK, "structured_content_path": (), "content_block_index": 0}
    instruction = _referral_instruction(target=target, mismatch_policy=policy)
    texts = _texts(result)

    [returned], mcp, recorder = await _referrals("P-1", instruction=instruction, upstream=FixedUpstream(result))

    assert _texts(returned) == texts
    assert returned.structuredContent == result.structuredContent
    [outcome] = mcp.injection_outcomes
    assert (outcome.status, outcome.occurrence, outcome.detail) == (OutcomeStatus.UNMATCHED, 0, detail)
    assert [_facts(record) for record in recorder.finishes[0]["injection_outcomes"]] == [("unmatched", False, None, [])]


@pytest.mark.asyncio
async def test_structured_content_that_isnt_an_object_never_matches():
    listed = CallToolResult.model_construct(
        content=[TextContent(type="text", text="[]")], structuredContent=[{"referral_id": "REF-1"}]
    )
    mcp, _, recorder = _mcp(InjectionPlan((_referral_instruction(),)))
    mcp._upstream = FixedUpstream(listed)  # type: ignore[assignment]

    # FastMCP itself refuses to return structured content that isn't an object; the call's outcome is decided first.
    with pytest.raises(ValueError, match="structured_content must be a dict"):
        await mcp._forward_call("get_referral", {"patient_id": "P-1"})

    [outcome] = mcp.injection_outcomes
    detail = "result structured content is list, not an object with field 'referral_id'; result unchanged"
    assert (outcome.status, outcome.occurrence, outcome.detail) == (OutcomeStatus.UNMATCHED, 0, detail)
    assert [_facts(record) for record in recorder.finishes[0]["injection_outcomes"]] == [("unmatched", False, None, [])]


@pytest.mark.asyncio
async def test_a_content_block_target_is_decided_by_the_structured_content_too():
    target = {"surface": SelectorSurface.CONTENT_BLOCK, "structured_content_path": (), "content_block_index": 0}
    instruction = _referral_instruction(target=target, occurrence=OccurrenceRule(Occurrence.EVERY))

    unmatched, matched = (await _referrals("P-2", "P-1", instruction=instruction))[0]

    assert _texts(unmatched) == ['{"referral_id":"REF-2","patient_id":"P-2","reason":"reason for P-2"}']
    assert _texts(matched) == ['{"referral_id":"REF-1","patient_id":"P-1","reason":"reason for P-1"} INJECTED']


@pytest.mark.asyncio
async def test_result_record_and_call_arguments_must_both_match():
    instruction = _referral_instruction(
        target={"call_arguments": {"patient_id": "P-3"}}, occurrence=OccurrenceRule(Occurrence.EVERY)
    )

    # P-1 answers REF-1 but isn't P-3; P-3 answers REF-1; P-2 is neither.
    results, mcp, _ = await _referrals("P-1", "P-3", "P-2", instruction=instruction)

    assert [_reason(result).endswith(" INJECTED") for result in results] == [False, True, False]
    # A call argument miss is decided before the call, so it's the one named.
    assert [(o.status, o.occurrence, o.detail) for o in mcp.injection_outcomes] == [
        (OutcomeStatus.UNMATCHED, 0, "call argument 'patient_id' is 'P-1', not 'P-3'; result unchanged"),
        (OutcomeStatus.APPLIED, 1, MIRRORED),
        (OutcomeStatus.UNMATCHED, 0, "call argument 'patient_id' is 'P-2', not 'P-3'; result unchanged"),
    ]

    results, mcp, _ = await _referrals(
        "P-2", instruction=_referral_instruction(target={"call_arguments": {"patient_id": "P-2"}})
    )
    [outcome] = mcp.injection_outcomes
    assert (outcome.status, outcome.occurrence, outcome.detail) == (
        OutcomeStatus.UNMATCHED,
        0,
        "result field 'referral_id' is 'REF-2', not 'REF-1'; result unchanged",
    )


@pytest.mark.asyncio
async def test_a_result_record_is_read_from_the_upstream_result_before_any_instruction_changes_it():
    # The first instruction rewrites the field the second matches on; the second still sees the upstream's value.
    rewrite = _referral_instruction(
        None, target={"structured_content_path": ("referral_id",)}, payload="REF-9", mode=InjectionMode.REPLACE
    )
    matching = _referral_instruction()

    mcp, _, _ = _mcp(InjectionPlan((rewrite, matching)))
    mcp._upstream = ReferralsUpstream()  # type: ignore[assignment]
    result = _wire_result(await mcp._forward_call("get_referral", {"patient_id": "P-1"}))

    assert _structured(result) == {"referral_id": "REF-9", "patient_id": "P-1", "reason": "reason for P-1 INJECTED"}
    assert [(o.instruction_index, o.status) for o in mcp.injection_outcomes] == [
        (0, OutcomeStatus.APPLIED),
        (1, OutcomeStatus.APPLIED),
    ]


@pytest.mark.asyncio
async def test_a_call_with_no_result_is_no_occurrence_of_a_result_record():
    class FailingUpstream:
        async def call_tool_mcp(self, name: str, args: dict[str, Any]) -> CallToolResult:
            raise RuntimeError("upstream failed")

    instruction = _referral_instruction(occurrence=OccurrenceRule(Occurrence.FIRST))
    mcp, _, recorder = _mcp(InjectionPlan((instruction,)))
    mcp._upstream = FailingUpstream()  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        await mcp._forward_call("get_referral", {"patient_id": "P-1"})
    mcp._upstream = ReferralsUpstream()  # type: ignore[assignment]
    result = _wire_result(await mcp._forward_call("get_referral", {"patient_id": "P-1"}))

    # The failed call wasn't counted, so the first result that matches is still the first occurrence.
    assert _reason(result) == "reason for P-1 INJECTED"
    assert [(o.status, o.occurrence, o.detail) for o in mcp.injection_outcomes] == [
        (OutcomeStatus.UNMATCHED, 0, "upstream produced no tool result to match a result record"),
        (OutcomeStatus.APPLIED, 1, MIRRORED),
    ]
    assert _facts(recorder.finishes[0]["injection_outcomes"][0]) == ("unmatched", False, None, [])


@pytest.mark.asyncio
async def test_size_limit_applies_to_the_matched_result_only():
    instruction = _referral_instruction(occurrence=OccurrenceRule(Occurrence.EVERY), size_limit=1)

    results, mcp, _ = await _referrals("P-2", "P-1", instruction=instruction)

    assert [_reason(result) for result in results] == ["reason for P-2", "reason for P-1"]
    assert [(o.status, o.occurrence) for o in mcp.injection_outcomes] == [
        (OutcomeStatus.UNMATCHED, 0),
        (OutcomeStatus.TOO_LARGE, 1),
    ]
