"""``http_judge`` reports each model request it sends to an observer, answered or failed.

A local server stands in for the model: nothing here reaches the network beyond 127.0.0.1.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from midojo_spec import rubric_judge
from midojo_spec.judge_exchange import Exchange, Observed, digests_hold
from midojo_spec.rubric_judge import JudgeCallError, http_judge

MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
KEY = "sk-live-observer-test"


class Model:
    """A canned chat-completions server: what it answers next, and what it was sent."""

    def __init__(self) -> None:
        self.status = 200
        self.body: bytes = b""
        self.delay = 0.0
        self.seen: list[dict[str, Any]] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - the name http.server dispatches on
                length = int(self.headers["Content-Length"])
                owner.seen.append(
                    {
                        "path": self.path,
                        "authorization": self.headers.get("Authorization"),
                        "body": json.loads(self.rfile.read(length)),
                    }
                )
                time.sleep(owner.delay)
                self.send_response(owner.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(owner.body)))
                self.end_headers()
                self.wfile.write(owner.body)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def answer(self, status: int = 200, **document: Any) -> None:
        self.status = status
        self.body = json.dumps(document).encode()


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Iterator[Model]:
    monkeypatch.setenv("OPENAICOMPATIBLE_API_KEY", KEY)
    served = Model()
    yield served
    served.server.shutdown()
    served.server.server_close()


def run(model: Model, **kwargs: Any) -> tuple[Any, list[tuple[list[dict[str, str]], Observed]], Exception | None]:
    """The call's result, what its observer saw, and the exception it raised, if any."""
    seen: list[tuple[list[dict[str, str]], Observed]] = []
    call = http_judge(
        model.url, "judge-model", observer=lambda messages, observed: seen.append((messages, observed)), **kwargs
    )
    result, failure = None, None
    try:
        result = call(MESSAGES)
    except Exception as exc:
        failure = exc
    return result, seen, failure


def test_an_answer_is_observed_with_its_text_usage_and_timing(model: Model) -> None:
    model.answer(
        model="judge-model-2026",
        choices=[{"message": {"content": '{"verdict": "supported"}'}, "finish_reason": "stop"}],
        usage={"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
    )

    result, seen, failure = run(model)

    assert failure is None
    assert result == '{"verdict": "supported"}'
    ((messages, observed),) = seen
    assert messages == MESSAGES
    assert observed.http_status == 200
    assert observed.content == '{"verdict": "supported"}'
    assert observed.finish_reason == "stop"
    assert observed.response_model == "judge-model-2026"
    assert observed.usage == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
    assert observed.error is None
    assert observed.duration_ms >= 0
    assert observed.started_at.endswith("+00:00")


def test_the_model_is_sent_the_request_and_the_key_stays_out_of_what_is_observed(model: Model) -> None:
    model.answer(choices=[{"message": {"content": "x"}}])

    _, seen, _ = run(model)

    assert model.seen[0]["path"] == "/v1/chat/completions"
    assert model.seen[0]["authorization"] == f"Bearer {KEY}"
    assert model.seen[0]["body"] == rubric_judge.request_body("judge-model", MESSAGES)
    assert KEY not in repr(seen)


@pytest.mark.parametrize(
    ("status", "label"),
    [(401, "AuthenticationError"), (404, "NotFoundError"), (429, "RateLimitError"), (500, "InternalServerError")],
)
def test_a_failed_call_is_observed_with_its_label_and_status(model: Model, status: int, label: str) -> None:
    model.answer(status, error="no")

    result, seen, failure = run(model)

    assert result is None
    assert isinstance(failure, JudgeCallError)
    assert failure.label == label
    assert failure.status == status
    ((_, observed),) = seen
    assert (observed.error, observed.http_status, observed.content) == (label, status, None)


def test_a_400_gives_no_content_and_is_observed_as_a_status_alone(model: Model) -> None:
    model.answer(400, error="bad request")

    result, seen, failure = run(model)

    assert (result, failure) == (None, None)
    ((_, observed),) = seen
    assert (observed.http_status, observed.error, observed.content) == (400, None, None)


def test_a_body_that_is_not_json_is_a_failed_call_with_its_status(model: Model) -> None:
    model.status, model.body = 200, b"<html>"

    _, seen, failure = run(model)

    assert isinstance(failure, JudgeCallError)
    assert failure.label == "JSONDecodeError"
    ((_, observed),) = seen
    assert (observed.error, observed.http_status) == ("JSONDecodeError", 200)


def test_a_body_too_deeply_nested_to_parse_is_observed_and_raised_as_it_was(model: Model) -> None:
    model.status, model.body = 200, b"[" * 100_000 + b"]" * 100_000

    result, seen, failure = run(model)

    assert result is None
    assert isinstance(failure, RecursionError)
    ((_, observed),) = seen
    assert (observed.error, observed.http_status, observed.content) == ("RecursionError", 200, None)


def test_an_answer_with_no_text_is_observed_without_content_and_returns_what_it_returned_before(model: Model) -> None:
    model.answer(choices=[{"message": {"content": ["part"]}}], usage={"total_tokens": 1})

    result, seen, failure = run(model)

    assert failure is None
    assert result == ["part"]
    ((_, observed),) = seen
    assert observed.content is None
    assert observed.usage == {"total_tokens": 1}


def test_an_answer_with_no_choices_returns_none_and_is_observed(model: Model) -> None:
    model.answer(usage={"total_tokens": 1})

    result, seen, _ = run(model)

    assert result is None
    assert seen[0][1].content is None
    assert seen[0][1].usage == {"total_tokens": 1}


def test_a_usage_that_is_not_an_object_is_not_kept(model: Model) -> None:
    model.answer(choices=[{"message": {"content": "x"}}], usage=[1, 2])

    _, seen, _ = run(model)

    assert seen[0][1].usage is None


def test_a_deadline_that_passes_is_a_timeout_and_is_observed(model: Model) -> None:
    model.answer(choices=[{"message": {"content": "late"}}])
    model.delay = 2.0

    started = time.monotonic()
    _, seen, failure = run(model, timeout=5.0, deadline=time.monotonic() + 0.3)

    assert isinstance(failure, JudgeCallError)
    assert failure.label == "APITimeoutError"
    assert time.monotonic() - started < 1.9
    ((_, observed),) = seen
    assert (observed.error, observed.content) == ("APITimeoutError", None)


def test_a_connection_that_fails_is_observed(model: Model) -> None:
    model.server.shutdown()
    model.server.server_close()

    _, seen, failure = run(model)

    assert isinstance(failure, JudgeCallError)
    assert failure.label == "APIConnectionError"
    assert seen[0][1].error == "APIConnectionError"
    assert seen[0][1].http_status is None


def test_each_request_is_observed_once(model: Model) -> None:
    model.answer(choices=[{"message": {"content": "x"}}])
    seen: list[Observed] = []
    call = http_judge(model.url, "judge-model", observer=lambda _messages, observed: seen.append(observed))

    call(MESSAGES)
    call(MESSAGES)

    assert len(seen) == 2
    assert len(model.seen) == 2


def test_an_observer_that_raises_stops_the_call(model: Model) -> None:
    model.answer(choices=[{"message": {"content": "x"}}])

    def refuse(_messages: Any, _observed: Observed) -> None:
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        http_judge(model.url, "judge-model", observer=refuse)(MESSAGES)


def test_without_an_observer_the_call_returns_the_text_as_before(model: Model) -> None:
    model.answer(choices=[{"message": {"content": "plain"}}])

    assert http_judge(model.url, "judge-model")(MESSAGES) == "plain"


def test_an_observed_answer_becomes_a_row_whose_digests_hold(model: Model) -> None:
    model.answer(choices=[{"message": {"content": '{"verdict": "supported"}'}}], usage={"total_tokens": 2})
    _, seen, _ = run(model)
    messages, observed = seen[0]

    exchange = Exchange.observed(
        observed,
        request_id="sha256:" + "ab" * 32,
        ask=0,
        model="judge-model",
        prompt_version=rubric_judge.PROMPT_VERSION,
        messages=messages,
        secrets=[KEY],
    )

    assert digests_hold(exchange) is True
    assert exchange.request_sha256 == hashlib.sha256(b"u").hexdigest()
