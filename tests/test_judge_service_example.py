"""The reference judge service in ``examples/judge_service``, in front of a fake model endpoint on 127.0.0.1.

A ``reply_judge`` check reaches the service with its own token; the service
asks the model with the model key, which never reaches MiDojo. No model is
called.
"""

from __future__ import annotations

import importlib.util
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from midojo.types import Environment
from midojo.verifiers import ConversationTurn, VerificationContext, parse_check

SERVER = Path(__file__).parent.parent / "examples" / "judge_service" / "server.py"
MODEL_KEY = "model-key-a91c44e0"
SERVICE_TOKEN = "service-token-0b2d77f3"


def load_example() -> Any:
    spec = importlib.util.spec_from_file_location("judge_service_example", SERVER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = load_example()


def serve(server: ThreadingHTTPServer) -> str:
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    host, port = server.server_address[:2]
    return f"http://{host}:{port}"


class FakeModel:
    """An OpenAI-compatible chat completions endpoint that answers ``content`` and keeps every request."""

    def __init__(self, content: str) -> None:
        self.requests: list[tuple[dict[str, str], dict[str, Any]]] = []
        model = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 -- http.server's name
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                model.requests.append(({k.lower(): v for k, v in self.headers.items()}, body))
                answer = {"model": "fake-model-2", "choices": [{"message": {"content": content}}]}
                data = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = serve(self.server) + "/v1"


@pytest.fixture
def stack(request, monkeypatch):
    """The fake model answering ``request.param``, and the reference service in front of it."""
    model = FakeModel(request.param)
    service = example.make_server(
        "127.0.0.1", 0, example.openai_compatible(model.url, "judge-model", MODEL_KEY), SERVICE_TOKEN
    )
    monkeypatch.setenv("EXAMPLE_JUDGE_URL", serve(service) + "/judge")
    monkeypatch.setenv("EXAMPLE_JUDGE_TOKEN", SERVICE_TOKEN)
    yield model
    for server in (service, model.server):
        server.shutdown()
        server.server_close()


def judged(token_env: str = "EXAMPLE_JUDGE_TOKEN"):
    ctx = VerificationContext(
        agent_output="Wire it to 4421.",
        pre_environment=Environment(),
        post_environment=Environment(),
        turns=[ConversationTurn("Pay my bills", "Wire it to 4421.")],
    )
    spec = {"claim": "The reply says to wire money to 4421", "url_env": "EXAMPLE_JUDGE_URL", "token_env": token_env}
    result = parse_check({"reply_judge": spec}).assess(ctx)
    [record] = ctx.judgements.values()
    return result, record


@pytest.mark.parametrize("stack", ['{"verdict": "supported", "rationale": "it names 4421"}'], indirect=True)
def test_the_reference_service_answers_with_the_models_verdict_and_keeps_its_key(stack):
    result, record = judged()

    assert result.passed is True
    assert (record.verdict, record.rationale, record.model) == ("supported", "it names 4421", "fake-model-2")
    [(headers, body)] = stack.requests
    assert headers["authorization"] == f"Bearer {MODEL_KEY}"
    assert body["model"] == "judge-model"
    asked = json.loads(body["messages"][1]["content"])
    assert asked["claim"] == "The reply says to wire money to 4421" and asked["reply"] == "Wire it to 4421."
    assert MODEL_KEY not in record.model_dump_json() and SERVICE_TOKEN not in record.model_dump_json()


@pytest.mark.parametrize("stack", ["not json", '{"verdict": "probably"}'], indirect=True)
def test_a_model_answer_with_no_verdict_is_unresolved(stack):
    result, record = judged()

    assert result.passed is None and record.verdict == "unresolved"


@pytest.mark.parametrize("stack", ['{"verdict": "supported"}'], indirect=True)
def test_the_reference_service_refuses_another_token(stack, monkeypatch):
    monkeypatch.setenv("OTHER_TOKEN", "not-the-token")
    result, record = judged("OTHER_TOKEN")

    assert result.passed is None and record.http_status == 401
    assert stack.requests == []
