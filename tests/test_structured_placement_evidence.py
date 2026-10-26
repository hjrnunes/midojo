"""What a structured-content placement leaves in the evidence, recorded through MidojoMCP in process.

The sealed call record must be what the agent received, text and structured
parts, with the call's injection outcomes. A recorder written before the
detailed finish still gets every call recorded (its text only), and the agent
still gets its result.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent
from midojo_sdk.client import transient
from midojo_sdk.injection import (
    Channel,
    InjectionInstruction,
    InjectionMode,
    InjectionPlan,
    InjectionTarget,
    OutcomeStatus,
    RecordSelector,
    SelectorSurface,
)
from midojo_sdk.mcp import MidojoMCP
from midojo_sdk.recorder import (
    Disposition,
    ReportingRecorder,
    RetryingRecorder,
    finish_detailed,
    records_details,
)

from midojo.control_plane.config import AppConfig
from midojo.control_plane.inprocess import InProcessAgentClient
from midojo.control_plane.service import ControlPlane

MIRRORED = "exact selector matched; mirrored text re-serialised"
NOTES = {"notes": [{"id": "NOTE-1", "body": "first"}, {"id": "NOTE-2", "body": "second"}]}


def compact(value: Any) -> str:
    """JSON as FastMCP's text block mirrors structured content: compact."""
    return json.dumps(value, separators=(",", ":"))


class LegacyRecorder:
    """A recorder with the finish_call signature from before the detailed finish: no extra keywords."""

    def __init__(self, inner: Any = None) -> None:
        self.inner = inner
        self.finishes: list[dict[str, Any]] = []

    async def start_call(self, *, call_id: str, function: str, args: dict[str, Any], disposition: Disposition) -> None:
        if self.inner is not None:
            await self.inner.start_call(call_id=call_id, function=function, args=args, disposition=disposition)

    async def finish_call(
        self,
        *,
        call_id: str,
        function: str,
        args: dict[str, Any],
        result: str,
        disposition: Disposition,
        error: str | None = None,
    ) -> None:
        self.finishes.append({"call_id": call_id, "result": result, "error": error, "disposition": disposition})
        if self.inner is not None:
            await self.inner.finish_call(
                call_id=call_id, function=function, args=args, result=result, disposition=disposition, error=error
            )


class FixedUpstream:
    def __init__(self, result: CallToolResult) -> None:
        self.result = result

    async def call_tool_mcp(self, _name: str, _args: dict[str, Any]) -> CallToolResult:
        return self.result


def placement(path: tuple = ("notes", RecordSelector("NOTE-1"), "body"), **changes: Any) -> InjectionInstruction:
    target = InjectionTarget(surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=path)
    fields: dict[str, Any] = {"channel": Channel.TOOL_OUTPUT, "payload": " INJECTED", "target": target}
    return InjectionInstruction(**(fields | changes))


def notes_result(text: str | None = None, *, structured: dict[str, Any] = NOTES, error: bool = False) -> CallToolResult:
    sent = compact(structured) if text is None else text
    return CallToolResult(content=[TextContent(type="text", text=sent)], structuredContent=structured, isError=error)


def noted(body: str, structured: dict[str, Any] = NOTES) -> dict[str, Any]:
    changed = json.loads(json.dumps(structured))
    changed["notes"][0]["body"] = body
    return changed


async def forward(recorder: Any, upstream: CallToolResult, *instructions: InjectionInstruction, evaluation_id="e1"):
    mcp = MidojoMCP(
        "proxy",
        recorder=recorder,
        upstream_transport=object(),
        plan=InjectionPlan(instructions),
        evaluation_id=evaluation_id,
    )
    mcp._upstream = FixedUpstream(upstream)  # type: ignore[assignment]
    result = (await mcp._forward_call("list_notes", {})).to_mcp_result()
    assert isinstance(result, CallToolResult)
    return result, mcp


def agent_text(result: CallToolResult) -> str:
    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


# --- the sealed record, through InProcessAgentClient on a real control plane ---


@pytest.fixture
def control(suite) -> ControlPlane:
    return ControlPlane({"weather": suite}, config=AppConfig(closure_wait_seconds=0))


async def sealed(control: ControlPlane, upstream: CallToolResult, *instructions: InjectionInstruction, legacy=False):
    """Forward one call through MidojoMCP recording in process, seal the evaluation, and return the result and record."""
    run = control.create_run("weather")
    evaluation = control.create_evaluation(run.id, "weather_new_york")
    recorder: Any = InProcessAgentClient(control, evaluation.session_token)
    if legacy:
        recorder = LegacyRecorder(recorder)
    result, mcp = await forward(recorder, upstream, *instructions, evaluation_id=evaluation.id)
    snapshot = await control.complete(run.id, evaluation.id, "done")
    [record] = snapshot.function_calls
    return result, mcp.injection_outcomes, record


def outcome_rows(record) -> list[tuple[int, str | None, int, str, str]]:
    assert record.injection_outcomes is not None
    return [(o.instruction_index, o.probe_key, o.occurrence, o.status, o.detail) for o in record.injection_outcomes]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "summary"), [(InjectionMode.APPEND, "sunny INJECTED"), (InjectionMode.REPLACE, " INJECTED")]
)
async def test_a_mirror_seals_the_re_serialised_text_and_structured_content(control, mode, summary):
    structured = {"summary": "sunny", "days": [1, 2]}
    upstream = CallToolResult(
        content=[TextContent(type="text", text=json.dumps(structured))], structuredContent=structured
    )

    result, outcomes, record = await sealed(control, upstream, placement(("summary",), mode=mode, probe_key="p:main"))

    expected = {"summary": summary, "days": [1, 2]}
    assert (result.structuredContent, agent_text(result)) == (expected, json.dumps(expected))
    assert (record.result, record.structured_content, record.error) == (agent_text(result), expected, None)
    assert outcome_rows(record) == [(0, "p:main", 1, "applied", MIRRORED)]
    assert [o.status for o in outcomes] == [OutcomeStatus.APPLIED]


@pytest.mark.asyncio
async def test_a_record_selector_placement_seals_what_the_agent_got(control):
    result, _, record = await sealed(control, notes_result(), placement(mode=InjectionMode.REPLACE))

    expected = noted(" INJECTED")
    assert (result.structuredContent, agent_text(result)) == (expected, compact(expected))
    assert (record.result, record.structured_content) == (agent_text(result), expected)
    assert outcome_rows(record) == [(0, None, 1, "applied", MIRRORED)]


@pytest.mark.asyncio
async def test_a_text_that_is_not_a_mirror_seals_a_partial_placement(control):
    result, outcomes, record = await sealed(control, notes_result("NOTE-1: first; NOTE-2: second"), placement())

    assert agent_text(result) == "NOTE-1: first; NOTE-2: second"
    assert result.structuredContent == noted("first INJECTED")
    assert (record.result, record.structured_content) == (agent_text(result), result.structuredContent)
    [outcome] = outcomes
    assert outcome.status is OutcomeStatus.PARTIAL
    assert outcome_rows(record) == [(0, None, 1, "partial", outcome.detail)]
    assert "text content is not JSON" in outcome.detail


@pytest.mark.asyncio
@pytest.mark.parametrize("text", [None, "Error: quota exceeded"], ids=["mirror", "not-a-mirror"])
async def test_an_error_result_seals_its_text_as_result_and_error(control, text):
    result, _, record = await sealed(control, notes_result(text, error=True), placement())

    assert result.isError is True
    expected_text = compact(noted("first INJECTED")) if text is None else text
    assert agent_text(result) == expected_text
    assert (record.result, record.error, record.structured_content) == (
        expected_text,
        expected_text,
        noted("first INJECTED"),
    )
    assert [row[3] for row in outcome_rows(record)] == ["applied" if text is None else "partial"]


@pytest.mark.asyncio
async def test_a_too_large_payload_seals_both_parts_unchanged(control):
    upstream = notes_result()

    result, _, record = await sealed(control, upstream, placement(size_limit=3))

    assert (agent_text(result), result.structuredContent) == (compact(NOTES), NOTES)
    assert (record.result, record.structured_content) == (agent_text(result), NOTES)
    assert outcome_rows(record) == [(0, None, 1, "too_large", "transformed payload exceeds size_limit")]


@pytest.mark.asyncio
async def test_a_recorder_without_details_seals_the_text_and_says_the_details_are_missing(control, caplog):
    with caplog.at_level(logging.WARNING, logger="midojo_sdk.mcp"):
        result, outcomes, record = await sealed(control, notes_result(), placement(), legacy=True)

    # The agent still gets the injected result; the record keeps the text and doesn't claim the rest.
    assert result.structuredContent == noted("first INJECTED")
    assert [o.status for o in outcomes] == [OutcomeStatus.APPLIED]
    assert record.result == agent_text(result)
    assert (record.structured_content, record.injection_outcomes) == (None, None)
    assert "recorded with its text only" in caplog.text


# --- a recorder written before the detailed finish ---


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrap",
    [
        lambda legacy: legacy,
        lambda legacy: RetryingRecorder(legacy, retryable=transient),
        lambda legacy: ReportingRecorder(legacy),
        lambda legacy: ReportingRecorder(RetryingRecorder(legacy, retryable=transient)),
    ],
    ids=["direct", "retrying", "reporting", "reporting-retrying"],
)
async def test_an_old_signature_recorder_still_records_and_the_agent_gets_the_injected_result(wrap):
    legacy = LegacyRecorder()
    recorder = wrap(legacy)

    result, mcp = await forward(recorder, notes_result(), placement())

    assert not records_details(recorder)
    assert result.structuredContent == noted("first INJECTED")
    assert agent_text(result) == compact(noted("first INJECTED"))
    assert [o.status for o in mcp.injection_outcomes] == [OutcomeStatus.APPLIED]
    [finish] = legacy.finishes
    assert (finish["result"], finish["error"], finish["disposition"]) == (agent_text(result), None, "forwarded")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrap",
    [
        lambda legacy: legacy,
        lambda legacy: RetryingRecorder(legacy, retryable=transient),
        lambda legacy: ReportingRecorder(legacy),
    ],
    ids=["direct", "retrying", "reporting"],
)
async def test_a_detailed_finish_falls_back_to_the_plain_one(wrap):
    legacy = LegacyRecorder()
    recorder = wrap(legacy)
    details = {"structured_content": {"n": 1}, "injection_outcomes": []}

    if isinstance(recorder, LegacyRecorder):
        assert not await finish_detailed(
            recorder, call_id="c1", function="f", args={}, result="r", disposition="forwarded", **details
        )
    else:
        await recorder.finish_call_detailed(
            call_id="c1", function="f", args={}, result="r", disposition="forwarded", **details
        )

    assert legacy.finishes == [{"call_id": "c1", "result": "r", "error": None, "disposition": "forwarded"}]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "wrap",
    [
        lambda legacy: legacy,
        lambda legacy: RetryingRecorder(legacy, retryable=transient),
        lambda legacy: ReportingRecorder(legacy),
    ],
    ids=["direct", "retrying", "reporting"],
)
@pytest.mark.parametrize("planned", [False, True], ids=["no-plan", "other-tool"])
async def test_a_text_only_fallback_with_nothing_matched_still_warns(wrap, planned, caplog):
    legacy = LegacyRecorder()
    upstream = CallToolResult(content=[TextContent(type="text", text="plain")])
    other_tool = InjectionTarget(
        tool="other_tool", surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=("summary",)
    )
    instructions = (placement(target=other_tool),) if planned else ()

    with caplog.at_level(logging.WARNING, logger="midojo_sdk.mcp"):
        result, mcp = await forward(wrap(legacy), upstream, *instructions)

    assert (agent_text(result), result.structuredContent) == ("plain", None)
    assert mcp.injection_outcomes == ()
    assert legacy.finishes == [
        {"call_id": legacy.finishes[0]["call_id"], "result": "plain", "error": None, "disposition": "forwarded"}
    ]
    assert "recorded with its text only" in caplog.text


@pytest.mark.asyncio
async def test_a_sealed_text_only_fallback_reports_no_details(control, caplog):
    upstream = CallToolResult(content=[TextContent(type="text", text="plain")])

    with caplog.at_level(logging.WARNING, logger="midojo_sdk.mcp"):
        result, _, record = await sealed(control, upstream, legacy=True)

    assert record.result == agent_text(result) == "plain"
    # Not an empty list: the record doesn't claim that nothing was placed.
    assert (record.structured_content, record.injection_outcomes) == (None, None)
    assert "recorded with its text only" in caplog.text
