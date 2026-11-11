"""The MCP face: simulated tools, refusals and the seams it records through, with no service and no harness.

``MidojoMCP`` gets its recorder and world from the caller (the interceptor service gives it ones whose every
call crosses the link), or, with no recorder, from ``sessions``. These tests give it in-memory ones and call it
through FastMCP's in-memory transport, so what they check is the face itself: what it records and in what order,
what it refuses, and which recorder a request reaches.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from midojo_spec.context import ToolContext
from midojo_spec.injection import Channel, InjectionInstruction, InjectionPlan, InjectionTarget
from midojo_spec.recorder import EnvironmentRevision

from midojo_mim.mcp import BLOCKED_REASON, MidojoMCP


class MemoryRecorder:
    """Every record the face made, in the order it made them."""

    def __init__(self, *, fail_start: bool = False) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []
        self._fail_start = fail_start

    async def start_call(self, **fields: Any) -> None:
        if self._fail_start:
            raise ConnectionError("the record could not be delivered")
        self.events.append(("start", fields))

    async def finish_call(self, **fields: Any) -> None:
        self.events.append(("finish", fields))

    async def finish_call_detailed(self, **fields: Any) -> None:
        self.events.append(("finish", fields))

    async def report_capture_failure(self, **fields: Any) -> None:
        self.events.append(("capture_failure", fields))

    def kinds(self) -> list[str]:
        return [kind for kind, _ in self.events]


class MemoryWorld:
    """An environment held in a dict, with the revision a change must name to apply."""

    def __init__(self, **environment: Any) -> None:
        self.revision = 0
        self.environment = dict(environment)

    async def read(self) -> EnvironmentRevision:
        return EnvironmentRevision(self.revision, dict(self.environment))

    async def change(
        self, values: dict[str, Any], *, expected_revision: int | None = None
    ) -> EnvironmentRevision | None:
        if expected_revision is not None and expected_revision != self.revision:
            return None
        self.environment.update(values)
        self.revision += 1
        return await self.read()


class FixedSessions:
    """A ``Sessions`` that names the recorder the test last picked: one evaluation per request."""

    def __init__(self) -> None:
        self.current = MemoryRecorder()
        self.world_of = MemoryWorld(note="from the session")
        self.counter_scope = "session-a"

    def recorder(self) -> MemoryRecorder:
        return self.current

    def world(self) -> MemoryWorld:
        return self.world_of

    def scope(self) -> str:
        return self.counter_scope


async def call(mcp: MidojoMCP, tool: str, **arguments: Any) -> str:
    async with Client(mcp._fastmcp) as client:
        result = await client.call_tool(tool, arguments)
    return "\n".join(getattr(block, "text", "") for block in result.content)


def simulated(mcp: MidojoMCP) -> list[str]:
    """Register the two tools these tests call: one reads the environment, one writes it."""

    @mcp.tool()
    async def read_note(ctx: ToolContext) -> str:
        """Read the note."""
        return str(await ctx.env("note"))

    @mcp.tool()
    async def write_note(ctx: ToolContext, text: str) -> str:
        """Replace the note."""
        return "written" if await ctx.env_update("note", text) else "stale"

    return ["read_note", "write_note"]


def test_a_face_needs_a_recorder_or_sessions():
    with pytest.raises(ValueError, match="needs a recorder or sessions"):
        MidojoMCP("face")


def test_a_tool_takes_a_tool_context_that_the_schema_hides():
    mcp = MidojoMCP("face", recorder=MemoryRecorder())
    simulated(mcp)
    tools = {tool.name: tool for tool in asyncio.run(mcp._fastmcp.list_tools())}
    assert set(tools) == {"read_note", "write_note"}
    assert set(tools["write_note"].parameters["properties"]) == {"text"}

    with pytest.raises(TypeError, match="ToolContext"):

        @mcp.tool()
        async def no_context(text: str) -> str:
            return text


@pytest.mark.asyncio
async def test_a_simulated_tool_is_recorded_before_and_after_it_runs_and_reads_and_writes_the_world():
    recorder, world = MemoryRecorder(), MemoryWorld(note="first")
    mcp = MidojoMCP("face", recorder=recorder, world=world)
    simulated(mcp)

    assert await call(mcp, "read_note") == "first"
    assert await call(mcp, "write_note", text="second") == "written"
    assert world.environment["note"] == "second"
    assert await call(mcp, "read_note") == "second"

    assert recorder.kinds() == ["start", "finish"] * 3
    start, finish = recorder.events[0][1], recorder.events[1][1]
    assert start["call_id"] == finish["call_id"] and start["call_id"]
    assert (start["function"], start["disposition"], finish["disposition"]) == ("read_note", "simulated", "simulated")
    assert len({fields["call_id"] for _, fields in recorder.events}) == 3, "each call has a call id of its own"


@pytest.mark.asyncio
async def test_a_stale_environment_change_changes_nothing():
    world = MemoryWorld(note="first")
    mcp = MidojoMCP("face", recorder=MemoryRecorder(), world=world)

    @mcp.tool()
    async def overwrite_after_a_read(ctx: ToolContext) -> str:
        """Read the environment, let another writer in, then change it at the revision read."""
        seen = await ctx.read()
        await world.change({"note": "someone else"})
        applied = await ctx.env_update("note", "mine", expected_revision=seen.revision)
        return "applied" if applied else "stale"

    assert await call(mcp, "overwrite_after_a_read") == "stale"
    assert world.environment["note"] == "someone else"


@pytest.mark.asyncio
async def test_a_tool_has_no_environment_when_the_face_has_no_world():
    recorder = MemoryRecorder()
    mcp = MidojoMCP("face", recorder=recorder)
    simulated(mcp)

    with pytest.raises(ToolError, match="no environment"):
        await call(mcp, "read_note")
    # The failed call is a finished record carrying the error, not a start with no end.
    assert recorder.kinds() == ["start", "finish"]
    assert "no environment" in recorder.events[1][1]["error"]


@pytest.mark.asyncio
async def test_a_blocked_call_is_recorded_once_and_never_runs():
    recorder = MemoryRecorder()
    ran: list[str] = []
    mcp = MidojoMCP("face", recorder=recorder)

    @mcp.tool()
    async def send_email(ctx: ToolContext, to: str) -> str:
        """Send an email."""
        ran.append(to)
        return "sent"

    mcp.block("send_email")
    with pytest.raises(ToolError, match=BLOCKED_REASON):
        await call(mcp, "send_email", to="a@example.test")

    assert ran == []
    assert recorder.kinds() == ["finish"]
    assert recorder.events[0][1]["disposition"] == "blocked"
    assert recorder.events[0][1]["result"] == BLOCKED_REASON


@pytest.mark.asyncio
async def test_a_denied_call_is_recorded_with_the_reason_and_never_runs():
    recorder = MemoryRecorder()
    ran: list[str] = []
    mcp = MidojoMCP("face", recorder=recorder)

    @mcp.tool()
    async def transfer(ctx: ToolContext, amount: int) -> str:
        """Move money."""
        ran.append(str(amount))
        return "moved"

    @mcp.deny("transfer")
    def too_much(args: dict[str, Any]) -> str | None:
        return "over the limit" if args["amount"] > 100 else None

    with pytest.raises(ToolError, match="over the limit"):
        await call(mcp, "transfer", amount=500)
    assert ran == []
    assert (recorder.events[0][1]["disposition"], recorder.events[0][1]["result"]) == ("denied", "over the limit")

    assert await call(mcp, "transfer", amount=5) == "moved"
    assert ran == ["5"]


@pytest.mark.asyncio
async def test_a_tool_that_could_not_be_recorded_does_not_run():
    ran: list[str] = []
    recorder = MemoryRecorder(fail_start=True)
    mcp = MidojoMCP("face", recorder=recorder)

    @mcp.tool()
    async def lookup(ctx: ToolContext) -> str:
        """Look something up."""
        ran.append("lookup")
        return "found"

    with pytest.raises(ToolError):
        await call(mcp, "lookup")
    assert ran == [], "a call the driver never heard of is not run"
    # The loss of the record is reported, so the evaluation's capture is marked incomplete.
    assert "capture_failure" in recorder.kinds()


@pytest.mark.asyncio
async def test_with_no_recorder_each_request_reaches_the_recorder_and_world_its_session_names():
    sessions = FixedSessions()
    mcp = MidojoMCP("face", sessions=sessions)
    simulated(mcp)

    first, second = MemoryRecorder(), MemoryRecorder()
    sessions.current = first
    assert await call(mcp, "read_note") == "from the session"
    sessions.current = second
    assert await call(mcp, "read_note") == "from the session"

    assert first.kinds() == ["start", "finish"] and second.kinds() == ["start", "finish"]
    assert first.events[0][1]["call_id"] != second.events[0][1]["call_id"]


def test_a_recorder_of_its_own_wins_over_sessions_and_sessions_still_gives_the_world():
    recorder, sessions = MemoryRecorder(), FixedSessions()
    mcp = MidojoMCP("face", recorder=recorder, sessions=sessions)
    simulated(mcp)

    async def run() -> str:
        return await call(mcp, "read_note")

    assert asyncio.run(run()) == "from the session"
    assert recorder.kinds() == ["start", "finish"]
    assert sessions.current.kinds() == []


def test_a_plan_with_a_recorder_needs_an_evaluation_id_and_one_with_sessions_takes_none():
    plan = InjectionPlan(
        (
            InjectionInstruction(
                channel=Channel.TOOL_OUTPUT,
                payload="IMPORTANT: ...",
                target=InjectionTarget(tool="get_weather"),
                probe_key="w:main",
            ),
        )
    )
    with pytest.raises(ValueError, match="evaluation_id"):
        MidojoMCP("face", recorder=MemoryRecorder(), plan=plan, upstream_transport=object())
    with pytest.raises(ValueError, match="only used with an injected recorder"):
        MidojoMCP("face", sessions=FixedSessions(), evaluation_id="e1")
    MidojoMCP("face", recorder=MemoryRecorder(), plan=plan, evaluation_id="e1", upstream_transport=object())
    MidojoMCP("face", sessions=FixedSessions(), plan=plan, upstream_transport=object())
