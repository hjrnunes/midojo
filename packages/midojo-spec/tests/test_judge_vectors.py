"""``vectors/judge`` holds what the ``midojo.judge.v1`` rules answer for each case they must keep.

The request ids, the request problems and the response problems were computed by the
implementations the rules came from (the MiDojo runtime's reply judge and the consumer's
judge service); the ids were also recomputed with ``jq -cS`` and ``shasum -a 256``. A
judge service in another language runs the same files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from midojo_spec import judge

VECTORS = Path(__file__).parents[1] / "vectors" / "judge"


def load(name: str) -> dict[str, Any]:
    return json.loads((VECTORS / name).read_text(encoding="utf-8"))


REQUEST_IDS = load("request_id.json")["cases"]
REQUESTS = load("request.json")["cases"]
RESPONSES = load("response.json")


@pytest.mark.parametrize("case", REQUEST_IDS, ids=[case["name"] for case in REQUEST_IDS])
def test_a_body_has_its_request_id(case: dict[str, Any]) -> None:
    assert judge.request_id(case["body"]) == case["id"]


@pytest.mark.parametrize("case", REQUEST_IDS, ids=[case["name"] for case in REQUEST_IDS])
def test_a_request_holds_its_own_id(case: dict[str, Any]) -> None:
    request = {"schema": judge.SCHEMA, "id": case["id"], **case["body"]}

    assert judge.id_holds(request)
    assert not judge.id_holds({**request, "id": "sha256:" + "0" * 64})


def test_an_id_for_another_body_does_not_hold() -> None:
    first, second = REQUEST_IDS[0], REQUEST_IDS[1]

    assert not judge.id_holds({"schema": judge.SCHEMA, "id": first["id"], **second["body"]})


def test_a_request_without_an_id_does_not_hold_one() -> None:
    assert not judge.id_holds({"schema": judge.SCHEMA, **REQUEST_IDS[0]["body"]})


def test_a_body_that_is_not_json_data_has_no_id_that_holds() -> None:
    assert not judge.id_holds({"schema": judge.SCHEMA, "id": "sha256:x", "claim": object()})


@pytest.mark.parametrize("case", REQUESTS, ids=[case["name"] for case in REQUESTS])
def test_a_request_has_its_problem(case: dict[str, Any]) -> None:
    assert judge.request_problem(case["request"]) == case["problem"]


@pytest.mark.parametrize("case", RESPONSES["cases"], ids=[case["name"] for case in RESPONSES["cases"]])
def test_an_answer_has_its_problem(case: dict[str, Any]) -> None:
    assert judge.response_problem(case["answer"], RESPONSES["expected_id"]) == case["problem"]


def test_a_problem_never_holds_what_the_other_side_sent() -> None:
    secret = "sk-live-never-echoed"

    for answer in (
        {"schema": judge.SCHEMA, secret: 1},
        {"schema": secret},
        {"schema": judge.SCHEMA, "id": secret},
        {"schema": judge.SCHEMA, "verdict": secret},
        {"schema": judge.SCHEMA, "verdict": "supported", "rationale": [secret]},
        [secret],
    ):
        assert secret not in (judge.response_problem(answer, "sha256:" + "0" * 64) or "")
    for request in ({"schema": judge.SCHEMA, secret: 1}, {"schema": secret}, {"schema": judge.SCHEMA, "id": [secret]}):
        assert secret not in (judge.request_problem(request) or "")


def test_the_verdicts_are_the_three_the_format_names() -> None:
    assert judge.VERDICTS == ("supported", "contradicted", "unresolved")
    assert judge.SCHEMA == "midojo.judge.v1"
