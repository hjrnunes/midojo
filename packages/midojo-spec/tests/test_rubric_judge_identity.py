"""The judge's model and prompt are in its exchange row, where Garak's record holds them.

Garak's record holds the judge ``model`` and the ``prompt`` it sent (since 3097e41b). The
port's record holds neither: the identity a strict replay reads (``model``,
``prompt_version``) and the ``messages`` sent are in the judge exchange row. The vectors of
``vectors/rubric/fences.json`` (Garak at 3e601af3) pin that the two agree: the messages the
judge sends are Garak's ``prompt``, the model the request body names is Garak's ``model``,
and the row built from them holds both.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from midojo_spec import rubric_judge as port
from midojo_spec.judge_exchange import Exchange, Observed

VECTORS = json.loads((Path(__file__).parents[1] / "vectors" / "rubric" / "fences.json").read_text(encoding="utf-8"))[
    "vectors"
]
RECORD_KEYS = {"verdict", "evidence_refs", "reason", "prompt_version", "request", "raw_response"}
REQUEST_ID = "sha256:" + "00" * 32


def vectors() -> list[Any]:
    return [pytest.param(vector, id=vector["id"]) for vector in VECTORS]


def drive(vector: dict[str, Any]) -> tuple[dict[str, Any], list[list[dict[str, str]]]]:
    """The port's record for a canned reply and the messages its judge call was given."""

    sent: list[list[dict[str, str]]] = []
    canned = vector["canned"]

    def call(messages: list[dict[str, str]]) -> Any:
        sent.append(messages)
        if "raise" in canned:
            raise port.JudgeCallError(canned["raise"])
        return canned["content"]

    record = port.judge_output(vector["rubric"], port.build_evidence(**vector["midojo"]), call)
    return record, sent


def observed_of(vector: dict[str, Any]) -> Observed:
    canned = vector["canned"]
    if "raise" in canned:
        return Observed(started_at="2026-11-11T00:00:00.000+00:00", duration_ms=0, error=canned["raise"])
    return Observed(
        started_at="2026-11-11T00:00:00.000+00:00", duration_ms=0, http_status=200, content=canned["content"]
    )


@pytest.mark.parametrize("vector", vectors())
def test_the_record_keeps_the_keys_it_had_before_garak_added_the_identity(vector: dict[str, Any]) -> None:
    record, _ = drive(vector)

    assert set(record) == RECORD_KEYS
    assert set(vector["record"]) == RECORD_KEYS | {"model", "prompt"}


@pytest.mark.parametrize("vector", vectors())
def test_the_messages_the_judge_sends_are_garaks_prompt(vector: dict[str, Any]) -> None:
    record, sent = drive(vector)
    garak = vector["record"]

    assert sent == [garak["prompt"]]
    assert port.judge_messages(record["request"]) == garak["prompt"]


@pytest.mark.parametrize("vector", vectors())
def test_the_request_body_names_garaks_model_and_prompt(vector: dict[str, Any]) -> None:
    _, sent = drive(vector)
    garak = vector["record"]

    body = port.request_body(vector["http_body"]["model"], sent[0])

    assert body["model"] == garak["model"]
    assert body["messages"] == garak["prompt"]


@pytest.mark.parametrize("vector", vectors())
def test_the_exchange_row_holds_garaks_model_prompt_and_prompt_version(vector: dict[str, Any]) -> None:
    record, sent = drive(vector)
    garak = vector["record"]

    row = Exchange.observed(
        observed_of(vector),
        request_id=REQUEST_ID,
        ask=0,
        model=vector["http_body"]["model"],
        prompt_version=record["prompt_version"],
        messages=sent[0],
    )

    assert row.model == garak["model"]
    assert [{"role": role, "content": content} for role, content in row.messages] == garak["prompt"]
    assert row.prompt_version == garak["prompt_version"]
    assert row.reply == garak["raw_response"]
    assert row.error == (
        garak["reason"].removeprefix("judge_failed:") if garak["reason"].startswith("judge_failed:") else None
    )
