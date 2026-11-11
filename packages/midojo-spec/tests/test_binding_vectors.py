"""``vectors/binding/codec.json``: the ``midojo-binding-v1`` frames, answers and control messages.

The texts were written by hand (compact JSON, sorted keys, ASCII), so what a writer must produce is
checked against an oracle that is not the encoder. A service or driver in another language runs the
same file: encode each ``canonical`` text from its value, decode every text to its value, refuse
each refused text, and map each rejection kind as ``rejection_error`` does.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest

from midojo_spec import link
from midojo_spec.link import Answer, Frame, LinkFault, Message, Rejection
from midojo_spec.recorder import ControlPlaneRejectedError, InvalidRequestError, RejectionKind

DOC = json.loads((Path(__file__).parents[1] / "vectors" / "binding" / "codec.json").read_text(encoding="utf-8"))


def names(cases: list[dict[str, Any]]) -> list[str]:
    return [case["name"] for case in cases]


def refusal(error: str) -> str:
    return re.escape(error)


def test_the_constants_are_the_ones_the_vectors_name() -> None:
    constants = DOC["constants"]

    assert link.API == constants["api"]
    assert link.SUBPROTOCOL == constants["subprotocol"]
    assert link.HEARTBEAT_INTERVAL_SECONDS == constants["heartbeat_interval_seconds"]
    assert link.HEARTBEAT_EXPIRY_SECONDS == constants["heartbeat_expiry_seconds"]
    assert link.ANSWER_DEADLINE_SECONDS == constants["answer_deadline_seconds"]
    assert link.BIND_PATH == constants["bind_path"]
    assert link.URL_ENV == constants["url_env"]
    assert link.CREDENTIAL_ENV == constants["credential_env"]
    assert link.LINK_LOST_REASON == constants["link_lost_reason"]
    assert link.INVALID_REQUEST == constants["invalid_request"]
    assert list(link.OPS) == constants["ops"]
    assert list(link.MESSAGE_OPS) == constants["message_ops"]
    assert list(link.BIND_FIELDS) == constants["bind_fields"]
    assert list(link.BOUND_FIELDS) == constants["bound_fields"]
    assert list(link.UNBOUND_FIELDS) == constants["unbound_fields"]


def test_the_subprotocol_and_the_api_are_different_strings() -> None:
    assert link.SUBPROTOCOL != link.API


@pytest.mark.parametrize("case", DOC["frames"], ids=names(DOC["frames"]))
def test_a_frame_text_decodes_to_its_frame(case: dict[str, Any]) -> None:
    assert link.decode_frame(case["text"]) == Frame(**case["frame"])


@pytest.mark.parametrize("case", DOC["frames"], ids=names(DOC["frames"]))
def test_a_frame_encodes_to_its_canonical_text(case: dict[str, Any]) -> None:
    if case["canonical"]:
        assert link.encode_frame(Frame(**case["frame"])) == case["text"]
    else:
        assert link.decode_frame(link.encode_frame(Frame(**case["frame"]))) == Frame(**case["frame"])


@pytest.mark.parametrize("case", DOC["frames_refused"], ids=names(DOC["frames_refused"]))
def test_a_frame_text_is_refused_as_a_fault(case: dict[str, Any]) -> None:
    with pytest.raises(LinkFault, match=refusal(case["error"])) as fault:
        link.decode_frame(case["text"])

    assert str(fault.value).startswith(f"{link.LINK_LOST_REASON}: ")


def answer(case: dict[str, Any]) -> Answer:
    rejected = case["answer"]["rejected"]
    return Answer(
        case["answer"]["re"], ok=case["answer"]["ok"], rejected=None if rejected is None else Rejection(**rejected)
    )


@pytest.mark.parametrize("case", DOC["answers"], ids=names(DOC["answers"]))
def test_an_answer_text_decodes_to_its_answer(case: dict[str, Any]) -> None:
    assert link.decode_answer(case["text"]) == answer(case)


@pytest.mark.parametrize("case", DOC["answers"], ids=names(DOC["answers"]))
def test_an_answer_encodes_to_its_canonical_text(case: dict[str, Any]) -> None:
    if case["canonical"]:
        assert link.encode_answer(answer(case)) == case["text"]
    else:
        assert link.decode_answer(link.encode_answer(answer(case))) == answer(case)


@pytest.mark.parametrize("case", DOC["answers_refused"], ids=names(DOC["answers_refused"]))
def test_an_answer_text_is_refused_as_a_fault(case: dict[str, Any]) -> None:
    with pytest.raises(LinkFault, match=refusal(case["error"])):
        link.decode_answer(case["text"])


def test_the_rejection_kinds_are_the_ones_the_control_plane_gives() -> None:
    assert set(DOC["rejection_kinds"]) == {kind.value for kind in RejectionKind}
    assert len(DOC["rejection_kinds"]) == len(set(DOC["rejection_kinds"]))


@pytest.mark.parametrize("kind", DOC["rejection_kinds"])
@pytest.mark.parametrize("late_fact", [False, True])
def test_a_rejection_kind_is_the_control_planes_refusal(kind: str, late_fact: bool) -> None:
    error = link.rejection_error(Rejection(kind, "the detail", late_fact))

    assert isinstance(error, ControlPlaneRejectedError)
    assert (error.kind, error.detail, error.late_fact) == (RejectionKind(kind), "the detail", late_fact)


def test_an_invalid_request_is_not_a_refusal_of_the_evaluation() -> None:
    error = link.rejection_error(Rejection(link.INVALID_REQUEST, "a field is missing"))

    assert isinstance(error, InvalidRequestError)
    assert error.detail == "a field is missing"


@pytest.mark.parametrize("kind", DOC["unknown_rejection_kinds"])
def test_a_kind_the_control_plane_does_not_give_is_a_fault_of_the_peer(kind: str) -> None:
    with pytest.raises(LinkFault, match="unknown rejection kind"):
        link.rejection_error(Rejection(kind, "d"))


@pytest.mark.parametrize("case", DOC["messages"], ids=names(DOC["messages"]))
def test_a_message_text_decodes_to_its_message(case: dict[str, Any]) -> None:
    assert link.decode_message(case["text"]) == Message(case["op"], case["fields"])


@pytest.mark.parametrize("case", DOC["messages"], ids=names(DOC["messages"]))
def test_a_message_encodes_to_its_canonical_text(case: dict[str, Any]) -> None:
    if case["canonical"]:
        assert link.encode_message(case["op"], **case["fields"]) == case["text"]
    else:
        message = link.decode_message(link.encode_message(case["op"], **case["fields"]))
        assert message == Message(case["op"], case["fields"])


@pytest.mark.parametrize("case", DOC["messages_refused"], ids=names(DOC["messages_refused"]))
def test_a_message_text_is_refused_as_a_fault(case: dict[str, Any]) -> None:
    with pytest.raises(LinkFault, match=refusal(case["error"])):
        link.decode_message(case["text"])


@pytest.mark.parametrize("op", DOC["encode_refused_ops"])
def test_a_message_op_the_stream_does_not_have_is_not_encoded(op: str) -> None:
    with pytest.raises(ValueError, match="unknown control message op"):
        link.encode_message(op)


@pytest.mark.parametrize("closing", [False, True], ids=["opened only", "opened and closed"])
def test_a_text_nested_too_deep_to_parse_is_a_fault_not_a_crash(closing: bool) -> None:
    nested = DOC["nested_too_deep"]
    text = "[" * nested["depth"] + ("]" * nested["depth"] if closing else "")

    for decode, name in (
        (link.decode_frame, "frame"),
        (link.decode_answer, "answer"),
        (link.decode_message, "message"),
    ):
        with pytest.raises(LinkFault, match=refusal(nested[name])):
            decode(text)
