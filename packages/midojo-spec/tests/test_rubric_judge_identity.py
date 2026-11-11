"""The judge's model and prompt are in its exchange row, where Garak's record holds them.

Garak's record holds the judge ``model`` and the ``prompt`` it sent (since 3097e41b). The
port's record holds neither: the identity a strict replay reads (``model``,
``prompt_version``) and the ``messages`` sent are in the judge exchange row. The vectors of
``vectors/rubric/fences.json`` (Garak at 3e601af3) pin that the two agree: the messages the
judge sends are Garak's ``prompt``, the model the request body names is Garak's ``model``,
and the row built from them holds both.

The judge service, not this package, names the model: it gives one name to ``http_judge``
(which puts it in the request body) and to the row. ``MODEL`` is the name Garak's judge is
configured with in ``generate_fences.py``, so these tests check that a judge given that name
sends and records what Garak records.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from midojo_spec import rubric_judge as port
from midojo_spec.judge_exchange import Exchange, Observed

VECTORS = json.loads((Path(__file__).parents[1] / "vectors" / "rubric" / "fences.json").read_text(encoding="utf-8"))[
    "vectors"
]
MODEL = "judge-model"
RECORD_KEYS = {"verdict", "evidence_refs", "reason", "prompt_version", "request", "raw_response"}
REQUEST_ID = "sha256:" + "00" * 32
OVER_HTTP = {
    "judge.reply-only.unfenced",
    "judge.reply-only.fenced-json",
    "judge.reply-only.prose-before-fence",
    "judge.one-json-call.fenced-contradicted-tool",
}


def vectors(only: set[str] | None = None) -> list[Any]:
    return [pytest.param(vector, id=vector["id"]) for vector in VECTORS if only is None or vector["id"] in only]


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


def garaks_wire_body(vector: dict[str, Any]) -> dict[str, Any]:
    """The JSON body Garak's client sent: its keyword arguments with ``extra_body`` merged in."""

    body = vector["http_body"]
    return {key: value for key, value in body.items() if key != "extra_body"} | body["extra_body"]


@contextmanager
def serving(content: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """A local chat-completions server that answers every request with ``content``.

    Yields its base URL and the request bodies it received; nothing leaves 127.0.0.1.
    """

    bodies: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - the name http.server dispatches on
            bodies.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            answer = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(answer)))
            self.end_headers()
            self.wfile.write(answer)

        def log_message(self, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", bodies
    finally:
        server.shutdown()
        server.server_close()


def test_the_fixture_configures_garaks_judge_with_the_model_these_tests_name() -> None:
    assert {vector["record"]["model"] for vector in VECTORS} == {MODEL}
    assert {vector["http_body"]["model"] for vector in VECTORS} == {MODEL}


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
def test_the_request_body_is_the_one_garaks_client_sent(vector: dict[str, Any]) -> None:
    _, sent = drive(vector)

    body = port.request_body(MODEL, sent[0])

    assert body == garaks_wire_body(vector)
    assert body["model"] == vector["record"]["model"]
    assert body["messages"] == vector["record"]["prompt"]


@pytest.mark.parametrize("vector", vectors())
def test_the_exchange_row_holds_garaks_model_prompt_and_prompt_version(vector: dict[str, Any]) -> None:
    record, sent = drive(vector)
    garak = vector["record"]

    row = Exchange.observed(
        observed_of(vector),
        request_id=REQUEST_ID,
        ask=0,
        model=MODEL,
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


@pytest.mark.parametrize("vector", vectors(OVER_HTTP))
def test_over_http_the_wire_body_and_the_row_hold_what_garak_records(
    vector: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAICOMPATIBLE_API_KEY", "sk-identity-test")
    garak = vector["record"]
    seen: list[tuple[list[dict[str, str]], Observed]] = []

    with serving(vector["canned"]["content"]) as (url, bodies):
        call = port.http_judge(url, MODEL, observer=lambda messages, observed: seen.append((messages, observed)))
        record = port.judge_output(vector["rubric"], port.build_evidence(**vector["midojo"]), call)

    assert bodies == [garaks_wire_body(vector)]
    ((messages, observed),) = seen
    row = Exchange.observed(
        observed,
        request_id=REQUEST_ID,
        ask=0,
        model=MODEL,
        prompt_version=record["prompt_version"],
        messages=messages,
    )
    assert row.model == garak["model"]
    assert [{"role": role, "content": content} for role, content in row.messages] == garak["prompt"]
    assert row.prompt_version == garak["prompt_version"]
    assert row.reply == garak["raw_response"] == record["raw_response"]
    assert row.error is None
