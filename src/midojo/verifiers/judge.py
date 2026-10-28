"""The reply judge: a predicate that asks an external judge service whether a claim holds of the agent's reply.

``reply_judge`` sends the judged reply (the agent's output), the user message
it answers, the claim and the rubric (and, when the check asks for them, the
turns before it and the recorded tool calls) to a judge service URL as one
``midojo.judge.v1`` request, and maps the service's verdict to the check's:
``supported`` is True, ``contradicted`` False and ``unresolved`` unknown
(None). The service holds whatever model key it needs: MiDojo sends at most a
bearer token for the service itself, read from an environment variable the
check names, and never keeps it. See ``docs/judge-service.md`` for the schema.

Every failure is unknown, never False, with its reason: no URL or token in the
environment, a connection that fails or times out, a status other than 2xx, a
body that isn't JSON or doesn't follow the response schema. What was asked and
answered is kept in a :class:`JudgeRecord` on the verification context
(``VerificationContext.judgements``, keyed by request), which the grade
reports as ``judgements``. A check asked again in the same grading with the
same request (the attack outcome reads the security check again) gets the
answer it got the first time, without a second request.

Grading with a judge is not a pure function of the snapshot: the verdict is
the service's. The control plane grades an evaluation once and never replaces
the grade, so its recorded grade, with the judgements it holds, is the one
that counts; :func:`midojo.grade` asks the service again each time it's called.
The request is sent synchronously (``httpx.Client``); the control plane grades
in a worker thread, so a slow judge never blocks its event loop. While it is
sent, the ``httpx`` and ``httpcore`` loggers drop what that thread logs (a
response header or reason phrase may echo the token); other threads log as
before.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict
from pydantic_core import to_jsonable_python

from midojo.verifiers import VerificationContext, VerificationResult

NAME = "reply_judge"
SCHEMA = "midojo.judge.v1"
JudgeVerdict = Literal["supported", "contradicted", "unresolved"]
PASSED: dict[str, bool | None] = {"supported": True, "contradicted": False, "unresolved": None}
DEFAULT_TIMEOUT_SECONDS = 60.0

_SPEC_KEYS = frozenset(
    {"claim", "rubric", "url", "url_env", "token_env", "timeout_seconds", "prior_turns", "tool_calls"}
)
_RESPONSE_KEYS = frozenset({"schema", "id", "verdict", "rationale", "model", "details"})
# How deep ``details`` may nest: deeper data can't be dumped into a grade.
MAX_DETAILS_DEPTH = 64
# The record's fields that hold what the service sent (or a failure quoting it): the ones redaction reads.
_SERVICE_TEXT = ("rationale", "model", "details", "failure")
REDACTED = "[redacted]"


class JudgeRecord(BaseModel):
    """What one reply judge check asked its service and what it was answered, for the grade's audit.

    ``request`` is the body sent (or, when ``sent`` is False, the body that
    would have been), never the bearer token; ``endpoint`` is the URL without
    its user info, query or fragment, None when no URL was configured. An
    answered request has its ``http_status``; a valid answer has its
    ``verdict``, and the ``rationale``, ``model`` and ``details`` the service
    gave. ``failure`` says why the check is unknown when no valid answer came.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    endpoint: str | None
    request: dict[str, Any]
    sent: bool
    http_status: int | None = None
    verdict: JudgeVerdict | None = None
    rationale: str | None = None
    model: str | None = None
    details: dict[str, Any] | None = None
    failure: str | None = None


@dataclass(frozen=True)
class ReplyJudge:
    """Passes when the judge service finds ``claim`` supported by the reply, fails when it finds it contradicted.

    ``url`` is the service's URL, or ``url_env`` names the environment
    variable that holds it; ``token_env`` names the one holding its bearer
    token, if it takes one (a literal token is refused, so none lands in a
    suite file). Both are read when the check is assessed. ``prior_turns``
    and ``tool_calls`` add the earlier turns and the recorded tool calls to
    the request.
    """

    claim: str
    rubric: str | dict[str, Any] | None = None
    url: str | None = None
    url_env: str | None = None
    token_env: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    prior_turns: bool = False
    tool_calls: bool = False

    def assess(self, ctx: VerificationContext) -> VerificationResult:
        try:
            request = self.request(ctx)
        except Exception as exc:
            # Never a failed grading: unknown, with what could be told of the request.
            request = self._unbuilt_request()
            failure = f"the judge request could not be built ({type(exc).__name__})"
            unbuilt = JudgeRecord(request_id=request["id"], endpoint=None, request=request, sent=False, failure=failure)
            key = json.dumps(["unbuilt", request["id"], self.url, self.url_env, self.token_env])
            return self.result(ctx.judgements.setdefault(key, unbuilt))
        key = json.dumps([request["id"], self.url, self.url_env, self.token_env, self.timeout_seconds])
        record = ctx.judgements.get(key)
        if record is None:
            record = ctx.judgements[key] = self.ask(request)
        return self.result(record)

    def _unbuilt_request(self) -> dict[str, Any]:
        """What a request that couldn't be built is known to hold: the check's own claim and rubric."""
        body = {"claim": self.claim, "rubric": self.rubric}
        return {"schema": SCHEMA, "id": request_id(body), **body}

    def evaluate(self, ctx: VerificationContext) -> bool | None:
        return self.assess(ctx).passed

    @property
    def label(self) -> str:
        return f'reply judge "{self.claim}"'

    def request(self, ctx: VerificationContext) -> dict[str, Any]:
        """The ``midojo.judge.v1`` request body for ``ctx``: its ``id`` is the digest of the rest."""
        turns = ctx.turns
        body: dict[str, Any] = {
            "claim": self.claim,
            "rubric": self.rubric,
            "message": turns[-1].message if turns else None,
            "reply": ctx.agent_output,
        }
        if self.prior_turns:
            body["prior_turns"] = [{"message": turn.message, "reply": turn.reply} for turn in turns[:-1]]
        if self.tool_calls:
            body["tool_calls"] = [
                {
                    "function": call.function,
                    "args": _json_data(call.args),
                    "result": call.result,
                    "error": call.error,
                    "complete": call.complete,
                    "blocked": call.blocked,
                    "turn": call.turn,
                }
                for call in ctx.function_calls
            ]
            body["tool_calls_complete"] = ctx.function_calls_complete and all(c.complete for c in ctx.function_calls)
        return {"schema": SCHEMA, "id": request_id(body), **body}

    def ask(self, request: dict[str, Any]) -> JudgeRecord:
        """Send ``request`` to the service and keep what it answered, or why there is no answer."""
        fields: dict[str, Any] = {"request_id": request["id"], "request": request, "endpoint": None, "sent": False}
        url = self.url if self.url is not None else os.environ.get(self.url_env or "")
        if not url:
            return JudgeRecord(**fields, failure=f"no judge service URL: {self.url_env} is not set")
        try:
            endpoint = httpx.URL(url)
        except httpx.InvalidURL:
            return JudgeRecord(**fields, failure="the judge service URL is not a valid URL")
        fields["endpoint"] = str(endpoint.copy_with(username=None, password=None, query=None, fragment=None))
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        token = None
        if self.token_env is not None:
            token = os.environ.get(self.token_env)
            if not token:
                return JudgeRecord(**fields, failure=f"no judge service token: {self.token_env} is not set")
            headers["Authorization"] = f"Bearer {token}"
        try:
            record = _redacted(self._exchange(endpoint, headers, fields), token)
            record.model_dump_json()  # the grade dumps it: one that can't be dumped is a failure here
            return record
        except Exception as exc:
            # A judge's failure is an unknown check, never a failed grading. The request may have gone out.
            return JudgeRecord(**{**fields, "sent": True}, failure=f"the judge check failed ({type(exc).__name__})")

    def _exchange(self, endpoint: httpx.URL, headers: dict[str, str], fields: dict[str, Any]) -> JudgeRecord:
        content = json.dumps(fields["request"], ensure_ascii=False).encode("utf-8")
        fields = {**fields, "sent": True}
        try:
            # No proxy or .netrc from the environment, and no redirect: the token goes only to the URL configured.
            with (
                _transport_quiet(),
                httpx.Client(timeout=self.timeout_seconds, trust_env=False, follow_redirects=False) as http,
            ):
                response = http.post(endpoint, content=content, headers=headers)
        except httpx.TimeoutException as exc:
            failure = f"the judge service timed out after {self.timeout_seconds:g}s ({type(exc).__name__})"
            return JudgeRecord(**fields, failure=failure)
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            # The exception's message may hold the URL; its type says enough.
            return JudgeRecord(**fields, failure=f"the judge service request failed ({type(exc).__name__})")
        sent = {**fields, "http_status": response.status_code}
        if not response.is_success:
            return JudgeRecord(**sent, failure=f"the judge service answered HTTP {response.status_code}")
        try:
            answer = json.loads(response.content, parse_constant=_refuse_constant, parse_float=_finite_float)
        except _NonFiniteError as exc:
            return JudgeRecord(**sent, failure=f"the judge service's answer {exc}")
        except (ValueError, RecursionError):  # JSONDecodeError and UnicodeDecodeError are ValueErrors
            return JudgeRecord(**sent, failure="the judge service's answer is not JSON")
        problem = response_problem(answer, fields["request_id"])
        if problem is not None:
            return JudgeRecord(**sent, failure=f"the judge service's answer is not a {SCHEMA} response: {problem}")
        return JudgeRecord(
            **sent,
            verdict=answer["verdict"],
            rationale=answer.get("rationale"),
            model=answer.get("model"),
            details=answer.get("details"),
        )

    def result(self, record: JudgeRecord) -> VerificationResult:
        """The check's verdict and reason from what the service answered."""
        request = f"request {record.request_id}"
        if record.failure is not None:
            return VerificationResult.unknown(self.label, f"{record.failure}; {request}")
        about = f"model {record.model}; {request}" if record.model else request
        passed = PASSED[record.verdict or "unresolved"]
        if passed is None:
            return VerificationResult.unknown(self.label, f"the judge service left it unresolved ({about})")
        return VerificationResult(passed, f"{self.label}: {record.verdict} ({about})")


def request_id(body: dict[str, Any]) -> str:
    """``sha256:<hex>`` of ``body`` as sorted-key compact UTF-8 JSON: the same question has the same id."""
    text = json.dumps({"schema": SCHEMA, **body}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def response_problem(answer: Any, expected_id: str) -> str | None:
    """Why ``answer`` isn't a valid ``midojo.judge.v1`` response to the request ``expected_id``, or None."""
    if not isinstance(answer, dict):
        return "not a JSON object"
    if unknown := sorted(set(answer) - _RESPONSE_KEYS):
        return f"unknown fields {unknown}"
    if "schema" not in answer:
        return "no schema"
    if answer["schema"] != SCHEMA:
        return f"schema is {answer['schema']!r}"
    if "id" in answer and answer["id"] != expected_id:
        return "it answers another request id"
    verdict = answer.get("verdict")
    if type(verdict) is not str or verdict not in PASSED:  # the type first: a list or an object isn't hashable
        return f"verdict {verdict!r} is not one of {list(PASSED)}"
    for name in ("rationale", "model"):
        if answer.get(name) is not None and type(answer[name]) is not str:
            return f"{name} is not a string"
    if answer.get("details") is not None and type(answer["details"]) is not dict:
        return "details is not an object"
    if _depth(answer.get("details")) > MAX_DETAILS_DEPTH:
        return f"details nests deeper than {MAX_DETAILS_DEPTH}"
    return None


def _depth(value: Any) -> int:
    """How many objects and arrays deep ``value`` nests, counted level by level rather than by recursion."""
    depth, level = 0, [value]
    while containers := [item for item in level if type(item) in (dict, list)]:
        depth += 1
        level = [child for item in containers for child in (item.values() if type(item) is dict else item)]
    return depth


def parse_reply_judge(value: Any) -> ReplyJudge:
    """A :class:`ReplyJudge` from its YAML spec, or ValueError naming what's wrong with it."""
    if not isinstance(value, dict):
        raise ValueError(f"{NAME}: expected a mapping, got {type(value).__name__}")
    if "token" in value:
        raise ValueError(f"{NAME}: a literal token is not accepted; name the variable that holds it with token_env")
    if unknown := sorted(set(value) - _SPEC_KEYS):
        raise ValueError(f"{NAME}: unknown keys {unknown}")
    claim = value.get("claim")
    if type(claim) is not str or not claim.strip():
        raise ValueError(f"{NAME}: claim must be a non-empty string")
    rubric = value.get("rubric")
    if rubric is not None and type(rubric) not in (str, dict):
        raise ValueError(f"{NAME}: rubric must be a string or a mapping")
    if isinstance(rubric, dict):
        try:
            json.dumps(rubric, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{NAME}: rubric must be JSON data") from exc
    if ("url" in value) == ("url_env" in value):
        raise ValueError(f"{NAME}: give exactly one of url and url_env")
    for name in ("url", "url_env", "token_env"):
        if name in value and (type(value[name]) is not str or not value[name]):
            raise ValueError(f"{NAME}: {name} must be a non-empty string")
    timeout = value.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError(f"{NAME}: timeout_seconds must be a positive number")
    for name in ("prior_turns", "tool_calls"):
        if type(value.get(name, False)) is not bool:
            raise ValueError(f"{NAME}: {name} must be true or false")
    return ReplyJudge(
        claim=claim,
        rubric=rubric,
        url=value.get("url"),
        url_env=value.get("url_env"),
        token_env=value.get("token_env"),
        timeout_seconds=float(timeout),
        prior_turns=value.get("prior_turns", False),
        tool_calls=value.get("tool_calls", False),
    )


def _json_data(value: Any) -> Any:
    """``value``, of any type a sealed snapshot holds, as JSON data, the same every time; TypeError for another type.

    Text, integers, booleans and None are themselves; a finite float is
    itself and a non-finite one null (as ``lifecycle.json_data`` gives it);
    bytes are UTF-8 text, undecodable bytes escaped (``\\xff``); a list or
    tuple is a list, and a set or frozenset a list sorted by each item's
    sorted-key JSON; a dict whose keys are all text is an object, and any
    other dict a list of ``[key, value]`` pairs sorted by the key's JSON,
    so no key is lost or merged; a date, datetime, time or timedelta is its
    ISO 8601 text.
    """
    kind = type(value)
    if value is None or kind in (bool, int, str):
        return value
    if kind is float:
        return value if math.isfinite(value) else None
    if kind in (bytes, bytearray):
        return value.decode("utf-8", "backslashreplace")
    if kind in (list, tuple):
        return [_json_data(item) for item in value]
    if kind in (set, frozenset):
        return sorted((_json_data(item) for item in value), key=_sort_key)
    if kind is dict:
        if all(type(key) is str for key in value):
            return {key: _json_data(item) for key, item in value.items()}
        pairs = [[_json_data(key), _json_data(item)] for key, item in value.items()]
        return sorted(pairs, key=lambda pair: _sort_key(pair[0]))
    if kind in (date, datetime, time, timedelta):
        return to_jsonable_python(value)
    raise TypeError(f"a {kind.__module__}.{kind.__qualname__} is not JSON data")


def _sort_key(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class _NonFiniteError(ValueError):
    """The answer has ``NaN``, ``Infinity`` or ``-Infinity``, which JSON doesn't, or a number no float holds."""


def _refuse_constant(name: str) -> Any:
    raise _NonFiniteError(f"is not JSON: it has {name}")


def _finite_float(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        shown = text if len(text) <= 40 else text[:40] + "..."
        raise _NonFiniteError(f"has a number no float holds: {shown}")
    return number


def _redacted(record: JudgeRecord, token: str | None) -> JudgeRecord:
    """``record`` with ``token`` replaced wherever the service's text holds it (one it echoed back, say).

    Only the values of the fields the service filled in are read (keys and
    values of ``details`` included), never a field name or the verdict, so the
    record keeps its schema whatever the token is. When it can't be redacted,
    the record keeps none of the service's text and says so.
    """
    if not token:
        return record
    try:
        return record.model_copy(update={name: _without(getattr(record, name), token) for name in _SERVICE_TEXT})
    except Exception as exc:  # keep nothing the service sent rather than fail the grading
        failure = f"the judge service's answer could not be redacted ({type(exc).__name__})"
        kept = record.model_dump(include={"request_id", "endpoint", "request", "sent", "http_status"})
        return JudgeRecord(**kept, failure=failure)


def _without(value: Any, token: str) -> Any:
    """``value`` with ``token`` replaced in every string it holds."""
    if type(value) is str:
        return value.replace(token, REDACTED)
    if type(value) is dict:
        return {_without(key, token): _without(item, token) for key, item in value.items()}
    if type(value) is list:
        return [_without(item, token) for item in value]
    return value


# Set while this context sends a judge request: the transport loggers drop what it logs then.
_SENDING: ContextVar[bool] = ContextVar("midojo_judge_sending", default=False)


class _DropWhileSending(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not _SENDING.get()


_DROP_WHILE_SENDING = _DropWhileSending()


# Every logger httpx and httpcore log through: their modules' ``logging.getLogger`` calls, as a test reads them
# from the installed sources. httpx imports httpcore lazily, with its first client, and httpcore imports its HTTP/2
# and SOCKS modules only when used, so most of these don't exist yet when this module is imported.
TRANSPORT_LOGGERS = (
    "httpx",
    "httpcore",
    "httpcore.connection",
    "httpcore.http11",
    "httpcore.http2",
    "httpcore.proxy",
    "httpcore.socks",
)


def _filter_transport_loggers() -> None:
    """Give every transport logger the filter, creating the ones that don't exist yet.

    A logger's filter only sees the records made on that logger, not a
    child's, so each gets it. Creating a logger here, before httpcore's
    module asks for it, is what makes this hold from the first request:
    ``getLogger`` then hands that module this same logger, filter and all.
    Loggers already made under another ``httpx`` or ``httpcore`` name get
    it too.
    """
    existing = [name for name in list(logging.root.manager.loggerDict) if name.startswith(("httpx", "httpcore"))]
    for name in {*TRANSPORT_LOGGERS, *existing}:
        logger = logging.getLogger(name)
        if _DROP_WHILE_SENDING not in logger.filters:
            logger.addFilter(_DROP_WHILE_SENDING)


_filter_transport_loggers()


@contextmanager
def _transport_quiet() -> Iterator[None]:
    """Drop the ``httpx`` and ``httpcore`` log records this thread makes until it leaves.

    The filter reads a context variable, so other threads keep logging.
    """
    _filter_transport_loggers()
    sending = _SENDING.set(True)
    try:
        yield
    finally:
        _SENDING.reset(sending)
