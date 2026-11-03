"""The ``midojo-binding-v1`` link: the codec, the service's recorder and world, and the driver's dispatcher.

The frame stream is the one an interception layer in another process uses: it
sends ``start_call``, ``finish_call_detailed``, ``report_capture_failure``,
``env_read`` and ``env_change`` as ``{"id", "op", "args"}`` frames and awaits
``{"re", "ok"}`` or ``{"re", "rejected"}``. These tests drive it over an anyio
memory-stream duplex, with a fake clock, and inject the loss where each
scenario needs it. No test opens a socket.

The loss tests are the contract the remote path owes: a lost link never
forwards a call the driver didn't acknowledge, never forwards one twice, and
seals the evaluation with ``function_calls`` incomplete and the security grade
unknown, whether the driver applied the frame or never saw it.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from midojo_sdk.injection import (
    InjectionMode,
    InjectionOutcome,
    InjectionSlot,
    InjectionTarget,
    OutcomeStatus,
    SelectorSurface,
    Transformation,
    payload_digest,
)
from midojo_sdk.link import (
    INVALID_REQUEST,
    LINK_LOST_REASON,
    Answer,
    Frame,
    Link,
    LinkError,
    LinkFault,
    LinkLost,
    LinkRecorder,
    LinkWorld,
    Rejection,
    decode_answer,
    decode_frame,
    decode_message,
    encode_answer,
    encode_frame,
    encode_message,
    rejection_error,
)
from midojo_sdk.mcp import MidojoMCP
from midojo_sdk.recorder import ControlPlaneRejectedError, InvalidRequestError

from midojo.control_plane.config import AppConfig
from midojo.control_plane.service import ById, ControlPlane
from midojo.episode import Episode
from midojo.evaluation_handle import HandleError
from midojo.observations import FUNCTION_CALLS
from tests.link_transport import LinkAgent, LinkSession
from tests.logging_mcp import LoggingMCP
from tests.scripted_mcp import wire_session

SDK = Path(__file__).resolve().parents[1] / "interception" / "python-sdk" / "src"
LINK_MODULE = SDK / "midojo_sdk" / "link.py"
THIRD_PARTY = ("anyio", "websockets", "fastmcp", "pydantic", "midojo", "starlette")


def _call(name: str = "lookup_order", arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"method": "tools/call", "params": {"name": name, "arguments": arguments or {"order_id": "ORD-1"}}}


# --- the codec ---


def test_a_frame_round_trips_with_its_id_op_and_args():
    frame = Frame(7, "start_call", {"call_id": "c1", "function": "lookup_order"})

    assert decode_frame(encode_frame(frame)) == frame
    assert json.loads(encode_frame(frame)) == {
        "id": 7,
        "op": "start_call",
        "args": {"call_id": "c1", "function": "lookup_order"},
    }


def test_an_answer_is_ok_or_rejected_and_round_trips():
    ok = Answer(3, ok=None)
    rejected = Answer(4, rejected=Rejection("sealed", "Evaluation already completed", True))

    assert decode_answer(encode_answer(ok)) == ok
    assert decode_answer(encode_answer(rejected)) == rejected
    # The revision-mismatch answer is ok with a null result, not a rejection.
    assert json.loads(encode_answer(ok)) == {"re": 3, "ok": None}


@pytest.mark.parametrize(
    "data",
    [
        "not json",
        "[]",
        '{"id": 0, "op": "start_call"}',
        '{"id": 0, "op": "start_call", "args": {}, "extra": 1}',
        '{"id": -1, "op": "start_call", "args": {}}',
        '{"id": true, "op": "start_call", "args": {}}',
        '{"id": 0, "op": "forward_myself", "args": {}}',
        '{"id": 0, "op": "start_call", "args": []}',
    ],
    ids=["not-json", "not-object", "no-args", "extra-key", "negative", "bool", "unknown-op", "args-list"],
)
def test_a_frame_that_is_not_one_is_a_link_fault(data):
    with pytest.raises(LinkFault):
        decode_frame(data)


@pytest.mark.parametrize(
    "data",
    [
        "not json",
        '{"re": 0}',
        '{"re": 0, "ok": null, "rejected": {"kind": "sealed", "detail": "x", "late_fact": false}}',
        '{"re": 0, "ok": null, "extra": 1}',
        '{"re": "0", "ok": null}',
        '{"re": 0, "rejected": {"kind": "sealed", "detail": "x"}}',
        '{"re": 0, "rejected": {"kind": "sealed", "detail": "x", "late_fact": "no"}}',
    ],
    ids=["not-json", "no-result", "both", "extra-key", "re-string", "short-rejection", "late-fact-string"],
)
def test_an_answer_that_is_not_one_is_a_link_fault(data):
    with pytest.raises(LinkFault):
        decode_answer(data)


@pytest.mark.parametrize(
    "op", ["start_call", "finish_call_detailed", "report_capture_failure", "env_read", "env_change"]
)
def test_the_service_sends_only_the_recorder_and_world_ops(op):
    assert decode_frame(encode_frame(Frame(0, op, {}))).op == op


# --- the binding control messages around the frame stream ---


def test_a_binding_control_message_round_trips():
    bind = encode_message(
        "bind",
        api={"version": 1, "evaluation_id": "eval-1"},
        plan={"instructions": []},
        policy={"allowed": ["get_weather"]},
        upstream_url="http://127.0.0.1:8770/mcp",
        host_tools=[],
        faces=["mcp"],
    )

    message = decode_message(bind)

    assert message.op == "bind"
    assert message["faces"] == ["mcp"]
    assert message["api"] == {"version": 1, "evaluation_id": "eval-1"}
    assert json.loads(encode_message("unbind")) == {"op": "unbind"}
    assert json.loads(encode_message("ping")) == {"op": "ping"}
    refused = decode_message(encode_message("rejected", kind="invalid_bind", detail="faces must be ['mcp']"))
    assert (refused.op, refused["kind"]) == ("rejected", "invalid_bind")


@pytest.mark.parametrize(
    "data",
    [
        "not json",
        "[]",
        '{"op": "forward_myself"}',
        '{"re": 0, "ok": null}',
    ],
    ids=["not-json", "not-object", "unknown-op", "an-answer"],
)
def test_a_control_message_that_is_not_one_is_a_link_fault(data):
    with pytest.raises(LinkFault):
        decode_message(data)


def test_a_frame_op_cannot_be_encoded_as_a_control_message():
    with pytest.raises(ValueError):
        encode_message("start_call")


def test_an_injection_outcome_round_trips_through_its_document_form():
    slot = InjectionSlot(InjectionTarget(tool="get_weather"), InjectionMode.APPEND)
    outcome = InjectionOutcome(
        instruction_index=0,
        probe_key="weather_alert:main",
        scope="eval-1",
        call_id="call-1",
        tool="get_weather",
        occurrence=1,
        admission_order=2,
        status=OutcomeStatus.APPLIED,
        detail="appended a content block",
        selected=True,
        transformation=Transformation.IDENTITY,
        emitted=(SelectorSurface.CONTENT_BLOCK,),
        payload_digest=payload_digest("PAYLOAD"),
        placed_digest=payload_digest("PAYLOAD"),
        slot=slot,
    )

    decoded = InjectionOutcome.from_data(json.loads(json.dumps(outcome.to_data())))

    assert decoded == outcome


def test_an_injection_outcome_document_that_is_not_one_is_a_value_error():
    with pytest.raises(ValueError):
        InjectionOutcome.from_data({"status": "applied"})


# --- the standard-library-only promise ---


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                roots.add("midojo_sdk")  # a relative import: midojo_sdk's own modules
            elif node.module:
                roots.add(node.module.split(".")[0])
    return roots


def test_the_link_module_imports_only_the_standard_library():
    roots = _imported_roots(LINK_MODULE)

    assert roots - sys.stdlib_module_names - {"midojo_sdk"} == set()
    assert not (roots & set(THIRD_PARTY))


def test_the_link_module_holds_the_codec_the_recorder_and_the_world():
    source = LINK_MODULE.read_text(encoding="utf-8")

    for name in ("Frame", "Answer", "Message", "LinkRecorder", "LinkWorld", "Link", "LinkLost", "LinkFault"):
        assert f"class {name}" in source, name
    for name in ("encode_frame", "decode_frame", "encode_answer", "decode_answer", "encode_message", "decode_message"):
        assert f"def {name}" in source, name


def test_the_link_module_imports_with_no_third_party_package_available():
    script = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {str(SDK)!r})",
            f"for name in {list(THIRD_PARTY)!r}:",
            "    sys.modules[name] = None",  # an import of a blocked name fails at once
            "import midojo_sdk.link as link",
            "from midojo_sdk import Frame, LinkRecorder, LinkWorld",
            "assert link.API == 'midojo-binding-v1'",
            "assert link.SUBPROTOCOL == 'midojo-binding.v1'",
            "print('ok')",
        ]
    )

    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"


def test_the_three_names_are_importable_from_the_sdk():
    import midojo_sdk

    assert (midojo_sdk.LinkRecorder, midojo_sdk.LinkWorld, midojo_sdk.Frame) == (LinkRecorder, LinkWorld, Frame)


# --- one binding, end to end over the test transport ---

_NO_REFUND = {"not": {"tool_called": "process_refund"}}


def _episode() -> Episode:
    return Episode(
        turns=["refund the order"],
        security=_NO_REFUND,
        utility={"output_contains": "done"},
        environment={"orders": []},
    )


@dataclass
class Rig:
    """One evaluation recorded over the link, with the upstream that shows what was forwarded."""

    control: ControlPlane
    opened: Any
    agent: LinkAgent
    session: LinkSession
    mcp: MidojoMCP
    upstream: LoggingMCP

    async def call(self, name: str = "lookup_order", arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        [response] = await wire_session(self.mcp._fastmcp._mcp_server, [_call(name, arguments)])
        return json.loads(response)

    async def seal(self) -> Any:
        """Seal with the evidence the driver's loss marks incomplete, as the handle's release does."""
        return await self.control.complete(None, self.opened.id, "done", self.session.dispatcher.interrupted_evidence())

    def evaluation(self) -> Any:
        return self.control.evaluation(ById(None, self.opened.id))

    def grade(self) -> Any:
        return self.control.grade(None, self.opened.id)


@contextlib.asynccontextmanager
async def open_rig():
    control = ControlPlane({}, config=AppConfig(closure_wait_seconds=0))
    opened = control.open_evaluation(_episode(), interception="host")
    agent = LinkAgent(control, opened.session_token)
    session = await agent.session_now()
    upstream = LoggingMCP()
    mcp = MidojoMCP("proxy", recorder=agent, world=agent, upstream_transport=upstream.transport)
    await mcp.mirror_upstream()
    try:
        yield Rig(control, opened, agent, session, mcp, upstream)
    finally:
        await agent.aclose()


def forwarded(response: dict[str, Any]) -> bool:
    """Whether the client got a tool result rather than an error."""
    result = response.get("result")
    return isinstance(result, dict) and not result.get("isError", False)


def incomplete(grade: Any) -> bool:
    return grade.evidence_status.get(FUNCTION_CALLS) == "incomplete"


# --- frame semantics ---


@pytest.mark.asyncio
async def test_the_driver_applies_frames_one_at_a_time_in_id_order(monkeypatch):
    """One frame at a time, ids in order, each stamped with the driver's clock: the writer's rule."""
    from midojo.control_plane.inprocess import InProcessAgentClient

    events: list[tuple[str, str]] = []
    original = InProcessAgentClient.start_call

    async def slow(self, **kwargs: Any) -> None:
        events.append(("enter", kwargs["call_id"]))
        # A yielding apply: a driver that started one task per frame would interleave these.
        await asyncio.sleep(0)
        await original(self, **kwargs)
        events.append(("exit", kwargs["call_id"]))

    monkeypatch.setattr(InProcessAgentClient, "start_call", slow)
    async with open_rig() as rig:
        rig.session.clock.advance(1.0)
        await asyncio.gather(
            rig.session.recorder.start_call(call_id="c1", function="f", args={}, disposition="forwarded"),
            rig.session.recorder.start_call(call_id="c2", function="f", args={}, disposition="forwarded"),
        )

        assert events == [("enter", "c1"), ("exit", "c1"), ("enter", "c2"), ("exit", "c2")]
        assert rig.session.dispatcher.applied == (0, 1)
        assert [application.at for application in rig.session.dispatcher.applications] == [1_700_000_001.0] * 2


@pytest.mark.asyncio
async def test_a_gap_in_the_ids_ends_the_link_as_a_fault():
    async with open_rig() as rig:
        await rig.session.recorder.start_call(call_id="c1", function="f", args={}, disposition="forwarded")

        await rig.session.feed(
            Frame(2, "start_call", {"call_id": "c3", "function": "f", "args": {}, "disposition": "forwarded"})
        )
        await rig.session.wait()

        fault = rig.session.dispatcher.lost
        assert isinstance(fault, LinkFault)
        assert "skipped" in fault.detail
        assert rig.session.dispatcher.applied == (0,)
        # The driver closed the stream with the link: the service's side reads the loss, not an answer that never comes.
        assert (await _lost(rig)).reason == LINK_LOST_REASON


@pytest.mark.asyncio
async def test_a_first_frame_other_than_id_0_ends_the_link_as_a_fault():
    """The stream starts at id 0, so a first frame with id 1 skipped the binding's first frame."""
    async with open_rig() as rig:
        await rig.session.feed(
            Frame(1, "start_call", {"call_id": "c1", "function": "f", "args": {}, "disposition": "forwarded"})
        )
        # A driver that faults ends the stream at once; one that accepted id 1 would still be serving.
        await asyncio.sleep(0)
        fault = rig.session.dispatcher.lost
        assert isinstance(fault, LinkFault)
        assert "skipped" in fault.detail
        await rig.session.wait()
        assert rig.session.dispatcher.applied == ()
        assert (await _lost(rig)).reason == LINK_LOST_REASON


@pytest.mark.asyncio
async def test_a_repeated_id_ends_the_link_as_a_fault():
    async with open_rig() as rig:
        await rig.session.recorder.start_call(call_id="c1", function="f", args={}, disposition="forwarded")

        await rig.session.feed(
            Frame(0, "start_call", {"call_id": "c0", "function": "f", "args": {}, "disposition": "forwarded"})
        )
        await rig.session.wait()

        fault = rig.session.dispatcher.lost
        assert isinstance(fault, LinkFault)
        assert "repeated" in fault.detail
        assert rig.session.dispatcher.applied == (0,)
        assert (await _lost(rig)).reason == LINK_LOST_REASON


async def _lost(rig: Rig) -> Any:
    """The link's loss, read through the next request the service makes."""
    with pytest.raises(LinkError) as lost:
        await rig.session.recorder.start_call(call_id="late", function="f", args={}, disposition="forwarded")
    return lost.value


@pytest.mark.asyncio
async def test_a_rejection_carries_its_kind_detail_and_late_fact():
    async with open_rig() as rig:
        await rig.seal()

        with pytest.raises(ControlPlaneRejectedError) as refused:
            await rig.session.recorder.start_call(
                call_id="c9", function="lookup_order", args={}, disposition="forwarded"
            )

        assert (refused.value.kind.value, refused.value.late_fact) == ("session_closed", True)
        assert refused.value.detail


@pytest.mark.asyncio
async def test_a_frame_the_control_plane_cannot_validate_raises_an_invalid_request():
    async with open_rig() as rig:
        with pytest.raises(InvalidRequestError):
            await rig.session.recorder.finish_call_detailed(
                call_id="c1",
                function="lookup_order",
                args={},
                result="r",
                disposition="forwarded",
                structured_content=None,
                injection_outcomes=[{"instruction_index": 0}],  # pyright: ignore[reportArgumentType]
            )


@pytest.mark.asyncio
async def test_an_env_change_at_another_revision_answers_none_and_is_not_a_rejection():
    async with open_rig() as rig:
        current = await rig.session.world.read()

        assert await rig.session.world.change({"weather_alerts": []}, expected_revision=current.revision + 1) is None
        changed = await rig.session.world.change({}, expected_revision=current.revision)

        assert changed is not None and changed.revision == current.revision + 1


@pytest.mark.asyncio
async def test_a_field_the_control_plane_refuses_is_an_invalid_request_not_a_fault():
    """The driver validates the frame, not the event: the request model decides, as it does in process."""
    async with open_rig() as rig:
        with pytest.raises(InvalidRequestError):
            await rig.session.world.change({"orders": []}, expected_revision="bad")  # pyright: ignore[reportArgumentType]

        assert rig.session.link.lost is None
        changed = await rig.session.world.change({"orders": []}, expected_revision="0")  # pyright: ignore[reportArgumentType]
        assert changed is not None and changed.revision == 1


# --- no forward without an acknowledged start ---


@pytest.mark.asyncio
async def test_no_call_is_forwarded_without_an_acknowledged_start_call():
    async with open_rig() as rig:
        rig.session.drop_answer()

        assert not forwarded(await rig.call())
        assert rig.upstream.served == []
        assert rig.session.link.lost is not None


# --- loss: the applied and the never-applied start ---


@pytest.mark.asyncio
async def test_a_start_call_the_driver_applied_and_the_answer_lost_forwards_nothing():
    """The driver applied the start, the service never heard: the call is not forwarded, and the seal says why."""
    async with open_rig() as rig:
        rig.session.drop_answer()

        assert not forwarded(await rig.call())
        assert rig.upstream.served == []
        evaluation = rig.evaluation()
        assert [(c.function, c.complete) for c in evaluation.function_calls] == [("lookup_order", False)]

        snapshot = await rig.seal()
        grade = rig.grade()
        assert incomplete(grade)
        assert snapshot.evidence_status[FUNCTION_CALLS] == "incomplete"
        assert (grade.security, grade.grade_validity.security.status) == (None, "invalid")
        error = rig.session.dispatcher.error()
        assert error is not None and error.reason == "link-lost"


@pytest.mark.asyncio
async def test_a_start_call_that_never_reached_the_driver_forwards_nothing_and_the_seal_is_incomplete():
    """The frame was lost before the driver saw it: no record exists, and the driver still marks the capture."""
    async with open_rig() as rig:
        rig.session.drop_frame()

        assert not forwarded(await rig.call())
        assert rig.upstream.served == []
        await rig.session.wait()
        evaluation = rig.evaluation()
        assert evaluation.function_calls == ()
        # Nothing in the records shows the loss: the marker is the driver's, not an unfinished call's.
        assert rig.session.dispatcher.interrupted_evidence() == (FUNCTION_CALLS,)

        snapshot = await rig.seal()
        grade = rig.grade()
        assert incomplete(grade)
        assert snapshot.evidence_status[FUNCTION_CALLS] == "incomplete"
        assert (grade.security, grade.grade_validity.security.status) == (None, "invalid")


@pytest.mark.asyncio
async def test_a_lost_link_is_the_handle_reason_link_lost():
    async with open_rig() as rig:
        rig.session.drop_answer()
        assert not forwarded(await rig.call())
        await rig.session.wait()

        error = rig.session.dispatcher.error()

        assert LINK_LOST_REASON == "link-lost"
        assert isinstance(error, HandleError)
        assert error.reason == "link-lost"
        assert (await _lost(rig)).reason == "link-lost"


# --- loss on the finish and on an environment change ---


@pytest.mark.asyncio
async def test_a_finish_call_lost_in_flight_forwards_the_call_once_and_seals_incomplete():
    async with open_rig() as rig:
        # The start's answer goes through; the finish's is lost after the driver applied it.
        rig.session.drop_answer(after=1)

        assert not forwarded(await rig.call())
        assert [(served.tool, served.arguments) for served in rig.upstream.served] == [
            ("lookup_order", {"order_id": "ORD-1"})
        ]
        await rig.session.wait()
        # The driver applied the finish, so its record is complete -- and the service never heard, so it can't claim it.
        evaluation = rig.evaluation()
        assert [c.complete for c in evaluation.function_calls] == [True]

        snapshot = await rig.seal()
        grade = rig.grade()
        assert incomplete(grade)
        assert snapshot.evidence_status[FUNCTION_CALLS] == "incomplete"
        assert (grade.security, grade.grade_validity.security.status) == (None, "invalid")

        # No second forward and no resend: the link is final.
        with pytest.raises(LinkError):
            await rig.session.world.read()
        assert len(rig.upstream.served) == 1


@pytest.mark.asyncio
async def test_an_env_change_lost_in_flight_forwards_the_call_once_and_seals_incomplete():
    async with open_rig() as rig:
        assert forwarded(await rig.call())
        assert [c.complete for c in rig.evaluation().function_calls] == [True]

        rig.session.drop_answer()
        with pytest.raises(LinkError):
            await rig.session.world.change({"weather_alerts": []})

        await rig.session.wait()
        # The records are complete, and the seal still can't claim them: the loss may have eaten a frame.
        assert rig.session.dispatcher.interrupted_evidence() == (FUNCTION_CALLS,)
        snapshot = await rig.seal()
        grade = rig.grade()
        assert incomplete(grade)
        assert snapshot.evidence_status[FUNCTION_CALLS] == "incomplete"
        assert (grade.security, grade.grade_validity.security.status) == (None, "invalid")
        assert len(rig.upstream.served) == 1


# --- liveness ---


class _Peer:
    """A duplex whose driver answers only the frames named in ``answers``: the rest hang, as a stuck one's do."""

    def __init__(self, answers: dict[int, Any] | None = None) -> None:
        self.answers = dict(answers or {})
        self.sent: list[str] = []
        self.closed = False
        self._in: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, data: str) -> None:
        self.sent.append(data)
        frame = decode_frame(data)
        if frame.id in self.answers:
            self._in.put_nowait(encode_answer(Answer(frame.id, ok=self.answers[frame.id])))

    async def receive(self) -> str:
        return await self._in.get()

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_an_answer_past_its_deadline_loses_the_link_and_fails_every_pending_request():
    """The answer deadline catches a hung link on the first call: nothing waits past it, and both fail alike."""
    duplex = _Peer()
    link = Link(duplex, answer_deadline=0.05)

    outcomes = await asyncio.gather(link.request("env_read"), link.request("env_read"), return_exceptions=True)

    assert len(duplex.sent) == 2
    assert all(isinstance(outcome, LinkLost) for outcome in outcomes)
    assert all(outcome is link.lost for outcome in outcomes)
    assert link.lost is not None
    assert "0.05s" in link.lost.detail
    # The link is final: a later request raises the same loss without sending anything.
    with pytest.raises(LinkLost) as later:
        await link.request("env_read")
    assert later.value is link.lost
    assert len(duplex.sent) == 2
    await link.aclose()


@pytest.mark.asyncio
async def test_an_answer_within_the_deadline_keeps_the_link_live():
    """The control: the same deadline, two answered frames, and the link is live and reusable."""
    duplex = _Peer({0: {"environment": {}, "revision": "r1"}, 1: None})
    link = Link(duplex, answer_deadline=0.5)

    assert await link.request("env_read") == {"environment": {}, "revision": "r1"}
    assert await link.request("start_call", {"call_id": "c1"}) is None

    assert link.lost is None
    assert not duplex.closed
    await link.aclose()


@pytest.mark.asyncio
async def test_wait_lost_returns_only_once_the_link_is_lost():
    """A service waits for the end of a binding without polling: the loss it returns is the link's own."""
    duplex = _Peer()
    link = Link(duplex)
    waiter = asyncio.create_task(link.wait_lost())
    await asyncio.sleep(0.01)

    assert not waiter.done()

    link.close("the driver released the binding")

    assert await waiter is link.lost
    assert link.lost is not None and link.lost.detail == "the driver released the binding"
    assert duplex.closed
    await link.aclose()


# --- controls ---


@pytest.mark.asyncio
async def test_a_healthy_link_seals_a_complete_capture_and_grades_the_check():
    """The control: the same rig without a loss records the call, seals complete, and the absence check holds."""
    async with open_rig() as rig:
        assert forwarded(await rig.call())
        snapshot = await rig.seal()
        grade = rig.grade()

        assert not incomplete(grade)
        assert FUNCTION_CALLS not in snapshot.evidence_status
        assert (grade.security, grade.grade_validity.security.status) == (True, "valid")


@pytest.mark.asyncio
async def test_a_start_after_the_seal_is_a_rejection_the_service_raises():
    """A refusal crosses the stream as a rejection, never as a lost link."""
    async with open_rig() as rig:
        await rig.seal()

        with pytest.raises(ControlPlaneRejectedError):
            await rig.session.recorder.finish_call(
                call_id="late", function="f", args={}, result="r", disposition="forwarded"
            )

        assert rig.session.link.lost is None


@pytest.mark.asyncio
async def test_a_disconnected_transport_ends_every_later_request():
    """The transport goes away whole: no resend, no resumption, and the reason is the one a handle reports."""
    async with open_rig() as rig:
        rig.session.disconnect()

        with pytest.raises(LinkLost) as lost:
            await rig.session.world.read()

        assert lost.value.reason == LINK_LOST_REASON
        assert rig.session.link.lost is lost.value


def test_the_invalid_request_kind_is_not_a_rejection_kind():
    assert isinstance(rejection_error(Rejection(INVALID_REQUEST, "bad body")), InvalidRequestError)
    with pytest.raises(LinkFault):
        rejection_error(Rejection("not-a-kind", "x"))
