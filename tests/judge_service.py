"""A fake judge service for tests: a real HTTP server on 127.0.0.1, answering by the request's path.

``/supported``, ``/contradicted`` and ``/unresolved`` answer a valid
``midojo.judge.v1`` response with that verdict. The other paths each answer
one way a service can fail (see ``ANSWERS``). Every request is kept, with its
headers and body, in ``requests``. No model is called.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

MODEL = "fake-judge-1"


@dataclass(frozen=True)
class Received:
    path: str
    headers: dict[str, str]
    body: dict[str, Any]


def _valid(verdict: str):
    def answer(request: dict[str, Any], headers: dict[str, str]) -> tuple[int, Any]:
        return 200, {
            "schema": "midojo.judge.v1",
            "id": request["id"],
            "verdict": verdict,
            "rationale": f"the reply makes the claim {verdict}",
            "model": MODEL,
            "details": {"evidence_refs": ["reply"]},
        }

    return answer


def _with(**changes: Any):
    def answer(request: dict[str, Any], headers: dict[str, str]) -> tuple[int, Any]:
        _, body = _valid("supported")(request, headers)
        body = {**body, **changes}
        return 200, {key: value for key, value in body.items() if value is not _DROP}

    return answer


_DROP = object()


def _nested(depth: int) -> dict:
    value: dict = {}
    for _ in range(depth - 1):
        value = {"deeper": value}
    return value


ANSWERS = {
    "/supported": _valid("supported"),
    "/contradicted": _valid("contradicted"),
    "/unresolved": _valid("unresolved"),
    "/minimal": lambda request, headers: (200, {"schema": "midojo.judge.v1", "verdict": "contradicted"}),
    "/500": lambda request, headers: (500, {"error": "boom"}),
    "/404": lambda request, headers: (404, {"error": "no such judge"}),
    "/redirect": lambda request, headers: (302, None),
    "/not-json": lambda request, headers: (200, b"<html>supported</html>"),
    "/array": lambda request, headers: (200, ["supported"]),
    "/no-verdict": _with(verdict=_DROP),
    "/bad-verdict": _with(verdict="maybe"),
    "/verdict-type": _with(verdict=True),
    "/no-schema": _with(schema=_DROP),
    "/wrong-schema": _with(schema="midojo.judge.v0"),
    "/rationale-type": _with(rationale=["a", "list"]),
    "/model-type": _with(model=7),
    "/details-type": _with(details="text"),
    "/unknown-field": _with(confidence=0.9),
    "/other-id": _with(id="sha256:" + "0" * 64),
    "/echo-token": lambda request, headers: (
        200,
        {"schema": "midojo.judge.v1", "verdict": "supported", "rationale": headers.get("authorization", "")},
    ),
    # A valid answer whose response header and reason phrase echo the Authorization header (see the handler).
    "/echo-in-headers": _valid("supported"),
    "/verdict-list": _with(verdict=[]),
    "/verdict-object": _with(verdict={}),
    "/deep-details": _with(details=_nested(100)),
}

# A valid answer but for a number too large for a float, which Python's json module would decode as inf.
ANSWERS["/huge-number"] = lambda request, headers: (
    200,
    json.dumps(_valid("supported")(request, headers)[1]).replace('["reply"]', "1e400").encode(),
)

# A valid answer whose rationale is JSON's escape for a lone surrogate, which decodes to one.
ANSWERS["/surrogate-rationale"] = lambda request, headers: (
    200,
    json.dumps(_valid("supported")(request, headers)[1])
    .replace('"the reply makes the claim supported"', '"half \\ud800 a pair"')
    .encode(),
)

# Answers whose details hold a constant JSON doesn't have, as Python's json module writes it.
for _constant in ("NaN", "Infinity", "-Infinity"):
    ANSWERS[f"/details-{_constant}"] = lambda request, headers, constant=_constant: (
        200,
        json.dumps(_valid("supported")(request, headers)[1]).replace('["reply"]', constant).encode(),
    )


class FakeJudge:
    """A judge service listening on a free loopback port until ``close``."""

    def __init__(self, delay: float = 0.0) -> None:
        self.requests: list[Received] = []
        self.delay = delay
        judge = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 -- http.server's name
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length))
                headers = {key.lower(): value for key, value in self.headers.items()}
                self.path = self.path.split("?")[0]
                judge.requests.append(Received(self.path, headers, body))
                if self.path == "/slow":
                    time.sleep(2)
                    self.path = "/supported"
                time.sleep(judge.delay)
                status, answer = ANSWERS[self.path](body, headers)
                content = answer if isinstance(answer, bytes) else json.dumps(answer).encode()
                if self.path == "/echo-in-headers":
                    # The reason phrase is what httpx logs at INFO; the header is what httpcore logs at DEBUG.
                    self.send_response(status, headers.get("authorization", ""))
                    self.send_header("X-Echo", headers.get("authorization", ""))
                else:
                    self.send_response(status)
                if status == 302:
                    self.send_header("Location", "/supported")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, format: str, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self._thread.start()

    def url(self, path: str) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}{path}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
