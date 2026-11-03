"""The judge reproduces Garak's requests and records for multi-turn conversations.

``fixtures/rubric/turns.json`` holds two- and three-turn conversations as MiDojo
records them and every judge request and record Garak's ``SequentialInjection``
and ``RubricJudge`` made for them (the consumer's ``generate_turns.py``, Garak at
968cc22). ``judge_turns`` ``final`` grades the last turn and ``all`` every turn;
the consumer's judge service splits the conversation that way, so this test
checks the judge on the requests the split produced: each turn's request, the
messages it sends, the HTTP body, the record a canned reply gives, and the count.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from midojo_sdk import rubric_judge as port

TURNS = json.loads((Path(__file__).parent / "fixtures/rubric/turns.json").read_text(encoding="utf-8"))
MODES = ("final", "all")
RUBRIC_KEYS = ("question", "criteria", "facts", "runtime_facts")


def cases() -> list[Any]:
    return [pytest.param(case, id=case["id"]) for case in TURNS["cases"]]


def rubric_of(request: dict[str, Any]) -> dict[str, Any]:
    return {key: request[key] for key in RUBRIC_KEYS if key in request}


def judged(block: dict[str, Any]) -> list[dict[str, Any]]:
    """The records that asked the judge, in turn order."""

    return [record for turn in block["records"] for record in turn if record["request"] is not None]


def test_the_fixture_names_the_garak_revision_and_its_detector() -> None:
    assert TURNS["implementation"]["revision"] == "968cc22418160415f1d712f5c7cb54fbb9592098"
    assert TURNS["implementation"]["repository"] == "garak"
    assert TURNS["implementation"]["detector"].endswith("RubricJudge.detect")


@pytest.mark.parametrize("case", cases())
@pytest.mark.parametrize("mode", MODES)
def test_each_judged_turn_rebuilds_garaks_request(case: dict[str, Any], mode: str) -> None:
    block = case["garak"][mode]

    for request in block["requests"]:
        assert port.build_request(rubric_of(request), request["evidence"]) == request


@pytest.mark.parametrize("case", cases())
@pytest.mark.parametrize("mode", MODES)
def test_each_request_sends_garaks_messages_in_garaks_body(case: dict[str, Any], mode: str) -> None:
    block = case["garak"][mode]

    for request, body in zip(block["requests"], block["http_bodies"], strict=True):
        assert port.judge_messages(request) == body["messages"]
        assert (
            port.request_body(body["model"], body["messages"])
            == {key: value for key, value in body.items() if key != "extra_body"} | body["extra_body"]
        )


@pytest.mark.parametrize("case", cases())
@pytest.mark.parametrize("mode", MODES)
def test_a_canned_reply_gives_garaks_record(case: dict[str, Any], mode: str) -> None:
    block = case["garak"][mode]
    records = judged(block)

    assert len(records) == len(block["requests"]) == block["request_count"]
    for request, record in zip(block["requests"], records, strict=True):
        assert port.judge_output(rubric_of(request), request["evidence"], lambda _: record["raw_response"]) == record


@pytest.mark.parametrize("case", cases())
@pytest.mark.parametrize("mode", MODES)
def test_the_request_count_adds_up_over_the_turns(case: dict[str, Any], mode: str) -> None:
    block = case["garak"][mode]
    records = judged(block)

    assert block["request_count"] == sum(port.request_count(record) for record in records)
    assert block["final"] == records[-1]
