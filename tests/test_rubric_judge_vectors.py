"""The rubric judge gives Garak's outputs on every vector computed from Garak's functions.

``fixtures/rubric/vectors.json`` holds each case's inputs and the output of the
Garak function (``generate_vectors.py``, run in Garak's interpreter at 968cc22).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from midojo_sdk import rubric_judge as port

VECTORS = json.loads((Path(__file__).parent / "fixtures/rubric/vectors.json").read_text(encoding="utf-8"))
EVIDENCES = VECTORS["evidences"]
GROUPS = VECTORS["vectors"]


def ids(group: str) -> list[Any]:
    return [pytest.param(vector, id=vector["id"]) for vector in GROUPS[group]]


def test_the_vectors_name_the_garak_revision_and_the_sink_rule() -> None:
    assert VECTORS["implementation"]["revision"] == "968cc22418160415f1d712f5c7cb54fbb9592098"
    assert VECTORS["sink"]["function"] == "call_body"
    assert {name: len(group) for name, group in GROUPS.items()} == {
        "assess": 820,
        "evidence": 10,
        "judge": 20,
        "request": 19,
    }


@pytest.mark.parametrize("vector", ids("evidence"))
def test_evidence_matches_garaks_build_evidence(vector: dict[str, Any]) -> None:
    assert port.build_evidence(**vector["midojo"]) == vector["evidence"]


def test_evidence_without_agent_text_has_no_messages() -> None:
    assert port.build_evidence("u", [], None, [])["messages"] == []


@pytest.mark.parametrize("vector", ids("request"))
def test_request_matches_garaks_build_request(vector: dict[str, Any]) -> None:
    request = port.build_request(vector["rubric"], EVIDENCES[vector["evidence"]])

    assert request == vector["request"]
    assert port.request_text(request) == vector["request_text"]


@pytest.mark.parametrize("vector", ids("assess"))
def test_assess_matches_garaks_assess_response(vector: dict[str, Any]) -> None:
    result = port.assess_response(vector["parsed"], EVIDENCES[vector["evidence"]])

    assert result == (vector["verdict"], vector["evidence_refs"], vector["reason"])


class Canned:
    """Stands in for the judge call: records the messages and answers once."""

    def __init__(self, canned: dict[str, Any]) -> None:
        self.canned = canned
        self.messages: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> Any:
        self.messages.append(messages)
        if "raise" in self.canned:
            raise port.JudgeCallError(self.canned["raise"])
        return self.canned["content"]


@pytest.mark.parametrize("vector", ids("judge"))
def test_the_judged_record_matches_garaks_judge_output(vector: dict[str, Any]) -> None:
    call = Canned(vector["canned"])
    evidence = port.build_evidence(**vector["midojo"])

    record = port.judge_output(vector["rubric"], evidence, call)

    assert record == vector["record"]
    assert len(call.messages) == 1
    assert call.messages[0] == vector["http_body"]["messages"]


@pytest.mark.parametrize("vector", ids("judge"))
def test_the_request_body_matches_what_garaks_client_sent(vector: dict[str, Any]) -> None:
    body = vector["http_body"]

    built = port.request_body("judge-model", body["messages"])

    assert built == {key: value for key, value in body.items() if key != "extra_body"} | body["extra_body"]


def test_a_request_is_counted_only_when_one_was_built() -> None:
    evidence = port.build_evidence("Q", [], "A", [])
    record = port.judge_output({"question": "Q?"}, evidence, Canned({"content": "{}"}))

    assert record["request"] is not None
    assert port.request_count(record) == 1
    assert port.request_count({**record, "request": None}) == 0
