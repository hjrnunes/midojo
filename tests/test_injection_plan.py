"""The frozen injection plan and MCP executor."""

from __future__ import annotations

import asyncio
import dataclasses
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
    InjectionPlan,
    InjectionTarget,
    MismatchPolicy,
    Occurrence,
    OccurrenceRule,
    OutcomeStatus,
    SelectorSurface,
    instruction_from_generator,
    instruction_from_generator_for_plan,
)
from midojo_sdk.mcp import MidojoMCP, ToolContext
from midojo_sdk.recorder import Disposition

from midojo.episode import Episode
from midojo.suites import get_suite
from tests.scripted_mcp import ScriptedMCP
from tests.test_mcp_fidelity import CapturingRecorder


def _instruction(**changes: Any) -> InjectionInstruction:
    fields = {
        "channel": Channel.TOOL_OUTPUT,
        "payload": " INJECTED",
        "target": InjectionTarget(content_block_index=0, content_block_type="text"),
    }
    return InjectionInstruction(**(fields | changes))


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
            structuredContent={"summary": f"result {number}"},
        )
        self.results[number] = result
        return result


def _mcp(
    plan: InjectionPlan, *, evaluation_id: str = "evaluation-1"
) -> tuple[MidojoMCP, ConcurrentUpstream, CapturingRecorder]:
    recorder = CapturingRecorder()
    mcp = MidojoMCP("proxy", recorder=recorder, upstream_transport=object(), plan=plan, evaluation_id=evaluation_id)
    upstream = ConcurrentUpstream()
    mcp._upstream = upstream  # type: ignore[assignment]
    return mcp, upstream, recorder


def _wire_result(result: Any) -> CallToolResult:
    return result.to_mcp_result()


def _texts(result: CallToolResult) -> list[str]:
    assert all(isinstance(block, TextContent) for block in result.content)
    return [block.text for block in result.content if isinstance(block, TextContent)]


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
    target = InjectionTarget(surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=("summary",))
    mcp, upstream, _ = _mcp(InjectionPlan((_instruction(target=target),)))
    call = asyncio.create_task(mcp._forward_call("lookup", {"number": 1}))
    await upstream.arrived[1].wait()
    upstream.release[1].set()
    result = _wire_result(await call)
    assert _texts(result) == ["result 1"]
    assert result.structuredContent == {"summary": "result 1 INJECTED"}


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
