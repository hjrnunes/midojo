"""Offline replay vectors; no test in this module may open a socket."""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest

from midojo_spec.judge_exchange import Exchange, Observed
from midojo_spec.judge_replay import (
    AmbiguousExchange,
    IdentityRefused,
    ReplayRefusal,
    UnrecordedExchange,
    identity_problem,
    replay,
)

ID = "sha256:" + "a" * 64
VECTORS = json.loads((Path(__file__).parents[1] / "vectors/judge/replay.json").read_text())
REFUSALS = {
    UnrecordedExchange.why: UnrecordedExchange,
    IdentityRefused.why: IdentityRefused,
    AmbiguousExchange.why: AmbiguousExchange,
}


@pytest.fixture(autouse=True)
def forbid_sockets(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("replay attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def row(data: dict[str, Any]) -> Exchange:
    return Exchange.observed(
        Observed(
            started_at="2026-01-01T00:00:00+00:00",
            duration_ms=1,
            content=data.get("reply"),
            error=data.get("error"),
        ),
        request_id=ID,
        ask=data.get("ask", 0),
        model=data["model"],
        prompt_version=data["prompt_version"],
        messages=[{"role": "user", "content": "secret prompt"}],
        secrets=[data["reply"]] if data.get("redact_reply") else (),
    )


@pytest.mark.parametrize("case", VECTORS, ids=lambda case: case["name"])
def test_replay_vectors(case: dict[str, Any]) -> None:
    rows = [row(data) for data in case["rows"]]
    if "refusal" in case:
        with pytest.raises(REFUSALS[case["refusal"]]) as caught:
            replay(rows, ID, 0, model=case["model"], prompt_version=case["prompt_version"])
        error = caught.value
        assert error.why == case["refusal"]
        assert (error.request_id, error.ask) == (ID, 0)
        if isinstance(error, IdentityRefused):
            assert error.reason == case["reason"]
    else:
        assert replay(rows, ID, 0, model=case["model"], prompt_version=case["prompt_version"]) is rows[case["selected"]]


@pytest.mark.parametrize(
    ("model", "version", "expected"),
    [
        (None, "v1", "missing"),
        ("", "v1", "missing"),
        ("A", None, "missing"),
        ("B", "v2", "model"),
        ("A", "v2", "prompt_version"),
        ("A", "v1", None),
    ],
)
def test_identity_problem(model: str | None, version: str | None, expected: str | None) -> None:
    assert identity_problem(model, version, expected_model="A", expected_prompt_version="v1") == expected


@pytest.mark.parametrize("field", ["model", "prompt_version"])
@pytest.mark.parametrize("bad", [None, "", 0])
def test_missing_expected_identity_is_rejected_before_search(field: str, bad: Any) -> None:
    identity = {"model": "A", "prompt_version": "v1"}
    identity[field] = bad
    with pytest.raises(ValueError, match=field):
        replay([], ID, 0, **identity)
    with pytest.raises(ValueError, match="expected_"):
        identity_problem(
            "A", "v1", expected_model=identity["model"], expected_prompt_version=identity["prompt_version"]
        )


def test_refusals_never_quote_recorded_text_or_the_requested_model() -> None:
    secrets = ("secret prompt", "secret reply", "secret model")
    found = row({"model": "secret model", "prompt_version": "v1", "reply": "secret reply"})
    for rows, model in (
        ([], "secret model"),
        ([found], "other model"),
        ([found, row({"model": "secret model", "prompt_version": "v1", "reply": "different reply"})], "secret model"),
    ):
        with pytest.raises(ReplayRefusal) as caught:
            replay(rows, ID, 0, model=model, prompt_version="v1")
        assert not any(secret in str(caught.value) for secret in secrets)


def test_distinct_redacted_replies_remain_ambiguous() -> None:
    def exchange(reply: str) -> Exchange:
        return Exchange.observed(
            Observed(started_at="2026-01-01T00:00:00+00:00", duration_ms=1, content=reply),
            request_id="sha256:" + "a" * 64,
            ask=0,
            model="A",
            prompt_version="v1",
            messages=[{"role": "user", "content": "the shared question"}],
            secrets=[reply],
        )

    first, second = exchange("private-first"), exchange("private-second")
    assert first.reply == second.reply == "[redacted]"
    assert first.response_sha256 != second.response_sha256
    with pytest.raises(AmbiguousExchange):
        replay([first, second], first.request_id, 0, model="A", prompt_version="v1")


def test_identical_redacted_replies_and_digests_replay_first() -> None:
    data = {"model": "A", "prompt_version": "v1", "reply": "private-same", "redact_reply": True}
    first, second = row(data), row(data)
    assert first.reply == second.reply == "[redacted]"
    assert first.response_sha256 == second.response_sha256
    assert replay([first, second], ID, 0, model="A", prompt_version="v1") is first
