"""The placement engine: what the interceptor asks, what the plan's own engine answers, and where each lives.

An interceptor forwards a call and records it; an engine decides what the
plan places. These tests drive a real ``MidojoMCP`` in front of a stub
upstream: the plan's own engine still gives the agent the injected result
from ``plan=`` and from an injected ``engine=``, an engine of your own has
both of its decisions honoured (leave the result as it is, replace it with a
finished one), and the call's recorded outcomes are the engine's. They also
pin the split: the interceptor holds no plan state and none of the splice
code, and the placement module loads without the interceptor's dependencies.
"""

from __future__ import annotations

import copy
import inspect
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
from mcp.types import CallToolResult, TextContent
from midojo_sdk import placement
from midojo_sdk.injection import (
    Channel,
    InjectionInstruction,
    InjectionMode,
    InjectionOutcome,
    InjectionPlan,
    InjectionTarget,
    OutcomeStatus,
    SelectorSurface,
    payload_digest,
)
from midojo_sdk.mcp import MidojoMCP

ROOT = Path(__file__).resolve().parent.parent

CONFORMS: placement.PlacementEngine = placement.PlanPlacementEngine()
"""The plan's own engine satisfies the protocol: pyright checks this assignment, and a test checks it at run time."""


class Recorder:
    """Keeps every call completed through the detailed path."""

    def __init__(self) -> None:
        self.finished: list[dict[str, Any]] = []

    async def start_call(self, **_: Any) -> None:
        pass

    async def finish_call(self, **kwargs: Any) -> None:
        self.finished.append(kwargs)

    async def finish_call_detailed(self, **kwargs: Any) -> None:
        self.finished.append(kwargs)


class StubUpstream:
    """A mirrored tool that answers every call with the same klarna-shaped result, as a real upstream would."""

    def __init__(self, structured: dict[str, Any]) -> None:
        self.structured = structured
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool_mcp(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        self.calls.append((name, arguments))
        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(self.structured))],
            structuredContent=copy.deepcopy(self.structured),
        )


ORDER = {"order": {"id": "ORD-101", "note": "shipped", "total": 10}}


def outcome(call: placement.CallView, status: OutcomeStatus, detail: str, **fields: Any) -> InjectionOutcome:
    """One instruction's outcome on ``call``, as an engine of your own reports it."""
    return InjectionOutcome(
        instruction_index=0,
        probe_key=None,
        scope=call.scope,
        call_id=call.call_id,
        tool=call.tool,
        occurrence=1,
        admission_order=1,
        status=status,
        detail=detail,
        **fields,
    )


class Leaving:
    """An engine of your own that leaves every result as the upstream sent it."""

    def __init__(self) -> None:
        self.seen: list[placement.ResultView | None] = []
        self._outcomes: list[InjectionOutcome] = []
        self.decisions: list[placement.PlacementDecision] = []

    def decide(self, call: placement.CallView, result: placement.ResultView | None) -> placement.PlacementDecision:
        self.seen.append(result)
        decision = placement.leave(outcome(call, OutcomeStatus.UNMATCHED, "the engine left the result alone"))
        self.decisions.append(decision)
        return decision

    def settle(
        self, decision: placement.PlacementDecision, emitted: placement.EmittedView
    ) -> tuple[InjectionOutcome, ...]:
        self._outcomes.extend(decision.outcomes)
        return decision.outcomes

    def outcomes(self) -> tuple[InjectionOutcome, ...]:
        return tuple(self._outcomes)


class Replacing:
    """An engine of your own that rewrites one structured field, using placement's own splice helper."""

    def __init__(self, path: tuple[str, ...], text: str) -> None:
        self.path = path
        self.text = text
        self._outcomes: list[InjectionOutcome] = []

    def decide(self, call: placement.CallView, result: placement.ResultView | None) -> placement.PlacementDecision:
        if result is None:
            return placement.leave()
        written = outcome(
            call,
            OutcomeStatus.APPLIED,
            "the engine replaced the field",
            payload_digest=payload_digest(self.text),
            placed_digest=payload_digest(self.text),
        )
        self._outcomes.append(written)
        return placement.replace(
            placement.place_field(result, self.path, self.text, mode=InjectionMode.REPLACE), (written,)
        )

    def settle(
        self, decision: placement.PlacementDecision, emitted: placement.EmittedView
    ) -> tuple[InjectionOutcome, ...]:
        return decision.outcomes

    def outcomes(self) -> tuple[InjectionOutcome, ...]:
        return tuple(self._outcomes)


def texts(result: CallToolResult) -> list[str]:
    """The text the agent is sent, block by block."""
    return [block.text for block in result.content if isinstance(block, TextContent)]


def forwarded(engine: Any, structured: dict[str, Any]) -> tuple[MidojoMCP, StubUpstream, Recorder]:
    """A real interceptor in front of a stub upstream, honouring ``engine``."""
    recorder = Recorder()
    mcp = MidojoMCP("stub", recorder=recorder, upstream_transport=object(), engine=engine)
    upstream = StubUpstream(structured)
    mcp._upstream = upstream  # type: ignore[assignment]
    return mcp, upstream, recorder


async def sent(mcp: MidojoMCP, arguments: dict[str, Any] | None = None) -> CallToolResult:
    """The result the agent gets for one mirrored call."""
    forwarded_result = await mcp._forward_call(
        "lookup_order", {"order_id": "ORD-101"} if arguments is None else arguments
    )
    return cast(CallToolResult, forwarded_result.to_mcp_result())


def test_the_default_engine_satisfies_the_protocol():
    decide = inspect.signature(placement.PlacementEngine.decide)
    assert list(decide.parameters) == ["self", "call", "result"]
    assert str(decide.parameters["result"].annotation).endswith("| None")
    assert list(inspect.signature(placement.PlacementEngine.outcomes).parameters) == ["self"]
    assert list(inspect.signature(placement.PlanPlacementEngine.decide).parameters) == ["self", "call", "result"]
    assert isinstance(placement.PlanPlacementEngine(), placement.PlacementEngine)
    assert isinstance(CONFORMS, placement.PlanPlacementEngine)


@pytest.mark.asyncio
async def test_an_engine_that_leaves_the_result_is_honoured_and_its_outcomes_are_recorded():
    engine = Leaving()
    mcp, upstream, recorder = forwarded(engine, ORDER)

    result = await sent(mcp)

    # The agent sees the upstream's result exactly as it sent it.
    assert texts(result) == [json.dumps(ORDER)]
    assert result.structuredContent == ORDER
    # The engine decided twice: with no result when the call started, then with the upstream's.
    assert engine.seen[0] is None
    assert engine.seen[1] is not None and engine.seen[1].structured_content == ORDER
    assert [decision.kind for decision in engine.decisions] == ["leave", "leave"]
    # The call's outcomes are the engine's, in its order.
    assert mcp.injection_engine is engine
    assert mcp.injection_outcomes == engine.outcomes()
    assert [o.detail for o in mcp.injection_outcomes] == ["the engine left the result alone"]
    assert recorder.finished[-1]["injection_outcomes"][0]["detail"] == "the engine left the result alone"
    assert upstream.calls == [("lookup_order", {"order_id": "ORD-101"})]


@pytest.mark.asyncio
async def test_an_engine_that_replaces_a_field_is_honoured_and_its_outcomes_are_recorded():
    engine = Replacing(("order", "note"), "THIS TEXT")
    mcp, _, recorder = forwarded(engine, ORDER)

    result = await sent(mcp)

    # The agent sees the field the engine replaced, and the text that mirrors it, re-serialised as the upstream wrote it.
    replaced = {"order": {"id": "ORD-101", "note": "THIS TEXT", "total": 10}}
    assert result.structuredContent == replaced
    assert texts(result) == [json.dumps(replaced)]
    assert mcp.injection_outcomes == engine.outcomes()
    assert [o.placed_digest for o in mcp.injection_outcomes] == [payload_digest("THIS TEXT")]
    assert recorder.finished[-1]["injection_outcomes"][0]["placed_digest"] == payload_digest("THIS TEXT")


@pytest.mark.asyncio
async def test_a_plan_still_builds_the_default_engine():
    instruction = InjectionInstruction(
        channel=Channel.TOOL_OUTPUT,
        payload="!",
        target=InjectionTarget(surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=("order", "note")),
    )
    plan = InjectionPlan((instruction,))
    mcp = MidojoMCP("stub", recorder=Recorder(), upstream_transport=object(), plan=plan, evaluation_id="e1")
    mcp._upstream = StubUpstream(ORDER)  # type: ignore[assignment]

    result = await sent(mcp)

    injected = {"order": {"id": "ORD-101", "note": "shipped!", "total": 10}}
    assert result.structuredContent == injected
    assert texts(result) == [json.dumps(injected)]
    assert mcp.injection_plan == plan
    assert [o.status.value for o in mcp.injection_outcomes] == ["applied"]


def test_a_server_takes_a_plan_or_an_engine_not_both():
    with pytest.raises(ValueError, match="a plan or a placement engine"):
        MidojoMCP(
            "stub",
            recorder=Recorder(),
            upstream_transport=object(),
            plan=InjectionPlan(
                (
                    InjectionInstruction(
                        channel=Channel.TOOL_OUTPUT,
                        payload="!",
                        target=InjectionTarget(
                            surface=SelectorSurface.STRUCTURED_CONTENT,
                            structured_content_path=("order", "note"),
                        ),
                    ),
                )
            ),
            engine=Leaving(),
            evaluation_id="e1",
        )


def test_the_interceptor_holds_no_plan_state():
    mcp = MidojoMCP("stub", recorder=Recorder(), upstream_transport=object(), engine=Leaving())
    for name in ("_plan", "_plan_lock", "_occurrences", "_admissions", "_outcomes"):
        assert name not in vars(mcp)
    assert mcp.injection_engine is not None


PLACEMENT_SYMBOLS = (
    "_admit_injections",
    "_admit_on_result",
    "_count",
    "_emitted",
    "_field_edit",
    "_inject_content",
    "_inject_leaf",
    "_inject_path",
    "_inject_result",
    "_inject_selected_block",
    "_inject_structured",
    "_json_mirror",
    "_json_style",
    "_outcome",
    "_placed",
    "_resolve_path_part",
    "_same_json",
    "_track",
    "_track_field",
    "_transform",
)
"""The splice helpers and the plan's bookkeeping: each is defined in the placement module, none in the interceptor."""

PLACEMENT_LOGIC = (
    "counts[index] += 1",
    "Transformation.JSON_STRING",
    "instruction.size_limit",
    "MismatchPolicy.FALLBACK_APPEND",
    "call_argument_mismatch",
    "result_record_mismatch",
    "json.dumps(structured, **options)",
)
"""Occurrence counting, transformations, size limits, the mismatch policy, matching, and the JSON mirror's text."""


def test_the_placement_code_lives_in_the_placement_module():
    import midojo_sdk.mcp as interceptor

    placed = inspect.getsource(placement)
    forwards = inspect.getsource(interceptor)
    for symbol in PLACEMENT_SYMBOLS:
        assert f"def {symbol}(" in placed or f"{symbol} =" in placed or f"{symbol}:" in placed, symbol
        assert f"def {symbol}(" not in forwards, symbol
    for text in PLACEMENT_LOGIC:
        assert text in placed, text
        assert text not in forwards, text


def test_the_interceptor_keeps_forwarding_applying_recording_and_dispositions():
    import midojo_sdk.mcp as interceptor

    forwards = inspect.getsource(interceptor)
    for kept in (
        "await self._upstream.call_tool_mcp(name, arguments)",
        "result = _applied(result, decision)",
        "await _finish_forwarded(recorder, call_id, name, arguments",
        "async def mirror_upstream",
        "session_token(",
        "def block(",
        "def deny(",
    ):
        assert kept in forwards, kept


BLOCKED_MODULES = ("anyio", "fastmcp", "httpx", "mcp", "midojo", "pydantic")
"""What the placement module must load without: the interceptor's dependencies, and everything under ``midojo``."""

BLOCKER = f"""
import importlib.abc
import sys

BLOCKED = {BLOCKED_MODULES!r}


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{{name}} is blocked")
        return None


sys.meta_path.insert(0, Blocker())
import midojo_sdk.placement

assert "midojo_sdk.placement" in sys.modules
assert "fastmcp" not in sys.modules
assert [name for name in sys.modules if name.split(".")[0] in BLOCKED] == []
print("ok")
"""


def test_the_placement_module_loads_without_the_interceptors_dependencies():
    completed = subprocess.run([sys.executable, "-c", BLOCKER], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"


def test_the_placement_module_is_light():
    from tests.test_import_boundaries import LIGHT_MODULES

    assert placement.__name__ in LIGHT_MODULES
