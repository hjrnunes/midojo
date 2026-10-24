"""The frozen injection plan and MCP executor."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Any

import pytest
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
)
from midojo_sdk.mcp import MidojoMCP

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


def _mcp(plan: InjectionPlan) -> tuple[MidojoMCP, ConcurrentUpstream]:
    mcp = MidojoMCP("proxy", recorder=CapturingRecorder(), upstream_transport=object(), plan=plan)
    upstream = ConcurrentUpstream()
    mcp._upstream = upstream  # type: ignore[assignment]
    return mcp, upstream


def _wire_result(result: Any) -> CallToolResult:
    return result.to_mcp_result()


def _texts(result: CallToolResult) -> list[str]:
    assert all(isinstance(block, TextContent) for block in result.content)
    return [block.text for block in result.content if isinstance(block, TextContent)]


@pytest.mark.asyncio
async def test_empty_plan_returns_the_upstreams_result_object():
    mcp, upstream = _mcp(InjectionPlan())
    call = asyncio.create_task(mcp._forward_call("lookup", {"number": 1}))
    await upstream.arrived[1].wait()
    upstream.release[1].set()
    assert _wire_result(await call) is upstream.results[1]
    assert mcp.injection_outcomes == ()


@pytest.mark.asyncio
async def test_size_limit_reports_without_relocating_payload():
    mcp, upstream = _mcp(InjectionPlan((_instruction(size_limit=1),)))
    call = asyncio.create_task(mcp._forward_call("lookup", {"number": 1}))
    await upstream.arrived[1].wait()
    upstream.release[1].set()
    result = _wire_result(await call)
    assert _texts(result) == ["result 1"]
    assert mcp.injection_outcomes[0].status is OutcomeStatus.TOO_LARGE


@pytest.mark.asyncio
async def test_structured_field_selector_changes_only_that_field():
    target = InjectionTarget(surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=("summary",))
    mcp, upstream = _mcp(InjectionPlan((_instruction(target=target),)))
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


def test_capability_gaps_are_typed_and_listed_before_calls():
    target = InjectionTarget(surface=SelectorSurface.DESCRIPTION)
    plan = InjectionPlan((InjectionInstruction(Channel.TOOL_DESCRIPTION, "x", target),))
    with pytest.raises(CapabilityError) as raised:
        _mcp(plan)
    assert [(gap.aspect, gap.value) for gap in raised.value.gaps] == [
        ("channel", "tool_description"),
        ("selector", "description"),
    ]


@pytest.mark.asyncio
async def test_targeted_tool_is_validated_against_discovery():
    plan = InjectionPlan((_instruction(target=InjectionTarget(tool="absent")),))
    mcp = MidojoMCP("proxy", recorder=CapturingRecorder(), upstream_transport=ScriptedMCP().transport, plan=plan)
    with pytest.raises(CapabilityError) as raised:
        await mcp.mirror_upstream()
    assert [(gap.aspect, gap.value) for gap in raised.value.gaps] == [("tool", "absent")]


@pytest.mark.asyncio
async def test_concurrent_matches_are_counted_in_admission_order():
    plan = InjectionPlan((_instruction(occurrence=OccurrenceRule(Occurrence.FIRST)),))
    mcp, upstream = _mcp(plan)
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
async def test_strict_mismatch_is_reported_and_fallback_is_explicit():
    target = InjectionTarget(surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=("missing",))
    strict, strict_upstream = _mcp(InjectionPlan((_instruction(target=target),)))
    strict_call = asyncio.create_task(strict._forward_call("lookup", {"number": 1}))
    await strict_upstream.arrived[1].wait()
    strict_upstream.release[1].set()
    strict_result = _wire_result(await strict_call)
    assert _texts(strict_result) == ["result 1"]
    assert strict.injection_outcomes[0].status is OutcomeStatus.UNMATCHED
    fallback_instruction = _instruction(target=target, mismatch_policy=MismatchPolicy.FALLBACK_APPEND)
    fallback, fallback_upstream = _mcp(InjectionPlan((fallback_instruction,)))
    fallback_call = asyncio.create_task(fallback._forward_call("lookup", {"number": 1}))
    await fallback_upstream.arrived[1].wait()
    fallback_upstream.release[1].set()
    fallback_result = _wire_result(await fallback_call)
    assert _texts(fallback_result) == ["result 1", " INJECTED"]
    assert fallback.injection_outcomes[0].status is OutcomeStatus.FALLBACK
