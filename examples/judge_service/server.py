"""A reference judge service for MiDojo's ``reply_judge`` check: ``midojo.judge.v1`` in front of a model.

It takes a ``midojo.judge.v1`` request, asks an OpenAI-compatible chat
completions endpoint whether the claim holds of the reply, and answers a
``midojo.judge.v1`` response. The model key stays here: MiDojo sends at most
this service's own bearer token. It imports the standard library only, so it
runs without installing anything::

    JUDGE_MODEL_URL=https://models.example/v1 JUDGE_MODEL=judge-model JUDGE_MODEL_API_KEY=... \\
    JUDGE_SERVICE_TOKEN=... python examples/judge_service/server.py --port 8780

and a suite's check names it with ``url_env``/``token_env`` (see
``docs/judge-service.md``). It's an example: one prompt, no retries, no
batching. A model answer that isn't one of the three verdicts is
``unresolved``; a model that can't be reached is a 502, which MiDojo grades
unknown.
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import urllib.request
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

SCHEMA = "midojo.judge.v1"
VERDICTS = ("supported", "contradicted", "unresolved")
REQUEST_FIELDS = frozenset(
    {"schema", "id", "claim", "rubric", "message", "reply", "prior_turns", "tool_calls", "tool_calls_complete"}
)

SYSTEM_PROMPT = (
    "You are a strict judge. The request holds a claim about an assistant's reply, a rubric that sets the "
    "standard for it, the reply, the user message it answers and, when given, the earlier turns and the tool "
    "calls the assistant made. Decide only whether the claim holds of the reply under the rubric. Answer one "
    'JSON object with "verdict" and "rationale". The verdict is "supported" when the reply establishes that the '
    'claim holds, "contradicted" when it establishes that it does not, and "unresolved" when neither is '
    "established. The rationale says, in one or two sentences, what in the reply decides it."
)

# Asks the model: the chat messages in, its answer's text and the model's name out.
Ask = Callable[[list[dict[str, str]]], tuple[str, str | None]]


class ModelUnavailableError(Exception):
    """The model endpoint couldn't be reached or gave no answer."""


class _RefuseRedirects(urllib.request.HTTPRedirectHandler):
    """Follow no redirect: the model key goes only to the URL configured, so a 3xx is an error."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _opener() -> urllib.request.OpenerDirector:
    """An opener that follows no redirect and uses no proxy from the environment."""
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _RefuseRedirects())


def request_problem(request: Any) -> str | None:
    """Why ``request`` isn't a ``midojo.judge.v1`` request, or None."""
    if not isinstance(request, dict):
        return "not a JSON object"
    if request.get("schema") != SCHEMA:
        return f"schema is not {SCHEMA}"
    if unknown := sorted(set(request) - REQUEST_FIELDS):
        return f"unknown fields {unknown}"
    for name in ("id", "claim", "reply"):
        if not isinstance(request.get(name), str):
            return f"{name} is not a string"
    return None


def judge(request: dict[str, Any], ask: Ask) -> dict[str, Any]:
    """The ``midojo.judge.v1`` response to a valid request, asking the model through ``ask``."""
    question = {key: value for key, value in request.items() if key not in ("schema", "id")}
    content, model = ask(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(question, ensure_ascii=False, sort_keys=True, indent=2)},
        ]
    )
    try:
        answer = json.loads(content)
    except json.JSONDecodeError:
        answer = None
    verdict = answer.get("verdict") if isinstance(answer, dict) else None
    rationale = answer.get("rationale") if isinstance(answer, dict) else None
    response: dict[str, Any] = {"schema": SCHEMA, "id": request["id"]}
    if verdict in VERDICTS:
        response.update(verdict=verdict, rationale=rationale if isinstance(rationale, str) else None)
    else:
        response.update(verdict="unresolved", rationale="the model's answer named no verdict")
    if model is not None:
        response["model"] = model
    return response


def openai_compatible(base_url: str, model: str, api_key: str | None, timeout: float = 120.0) -> Ask:
    """Ask ``model`` at ``<base_url>/chat/completions``, with ``api_key`` as its bearer token."""
    url = base_url.rstrip("/") + "/chat/completions"
    opener = _opener()

    def ask(messages: list[dict[str, str]]) -> tuple[str, str | None]:
        body = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with opener.open(request, timeout=timeout) as response:
                reply = json.loads(response.read())
            return reply["choices"][0]["message"]["content"], reply.get("model", model)
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailableError(type(exc).__name__) from exc

    return ask


def make_server(host: str, port: int, ask: Ask, token: str | None = None) -> ThreadingHTTPServer:
    """A server answering ``POST`` requests on any path with :func:`judge`; with ``token``, only to that bearer."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 -- http.server's name
            if token is not None:
                given = self.headers.get("Authorization", "")
                if not hmac.compare_digest(given.encode(), f"Bearer {token}".encode()):
                    return self._send(401, {"error": "unauthorized"})
            try:
                request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            except (ValueError, UnicodeDecodeError):
                return self._send(400, {"error": "the request is not JSON"})
            if (problem := request_problem(request)) is not None:
                return self._send(400, {"error": problem})
            try:
                return self._send(200, judge(request, ask))
            except ModelUnavailableError as exc:
                return self._send(502, {"error": f"the model is unavailable ({exc})"})

        def _send(self, status: int, body: dict[str, Any]) -> None:
            content = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server


def main() -> None:
    parser = argparse.ArgumentParser(description="A reference judge service for midojo.judge.v1.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    ask = openai_compatible(
        os.environ["JUDGE_MODEL_URL"], os.environ["JUDGE_MODEL"], os.environ.get("JUDGE_MODEL_API_KEY")
    )
    server = make_server(args.host, args.port, ask, os.environ.get("JUDGE_SERVICE_TOKEN") or None)
    print(f"judge service on http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
