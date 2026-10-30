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
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict
from pydantic_core import to_jsonable_python

from midojo.verifiers import ValidityReason, VerificationContext, VerificationResult

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


class UnrecordedJudgeRequest(LookupError):  # noqa: N818 - named by the replay contract, not a lint choice.
    """A replay was asked for a request its records don't hold: the check is unknown, never False.

    ``request_id`` is the id that was asked for and ``id8`` its first eight
    hex digits, which the check's reason names (``judge-replay-unrecorded:<id8>``).
    """

    def __init__(self, request: Mapping[str, Any]) -> None:
        self.request_id = str(request["id"])
        self.id8 = self.request_id.removeprefix("sha256:")[:8]
        super().__init__(f"judge-replay-unrecorded:{self.id8}")


class RecordedJudge:
    """Answers ``reply_judge`` requests from records kept by an earlier grading, and never opens a connection.

    ``records`` are :class:`JudgeRecord` instances, or their JSON mappings (as
    a grade's ``judgements`` holds them), keyed by request id. Asking for a
    request no record names raises :class:`UnrecordedJudgeRequest`, which the
    check turns into unknown with ``judge-replay-unrecorded:<id8>``: a replay
    never guesses a verdict. When the same request id was recorded more than
    once (asked of more than one endpoint), the last record answers.

    Pass one to :func:`midojo.grade` (or :meth:`midojo.Episode.grade`) as
    ``judgements=``: the grading then reads its judgements from it instead of
    asking a service. Records whose ``failure`` is set replay as unknown, as
    they graded the first time.
    """

    def __init__(self, records: Iterable[JudgeRecord | Mapping[str, Any]]) -> None:
        self._records: dict[str, JudgeRecord] = {}
        for record in records:
            kept = record if isinstance(record, JudgeRecord) else JudgeRecord.model_validate(record)
            self._records[kept.request_id] = kept

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, request_id: object) -> bool:
        return request_id in self._records

    def ask(self, request: Mapping[str, Any]) -> JudgeRecord:
        """The record for ``request``'s id, or :class:`UnrecordedJudgeRequest`."""
        try:
            return self._records[str(request["id"])]
        except KeyError:
            raise UnrecordedJudgeRequest(request) from None


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
        """Ask the service once per request, and give its verdict, or unknown with why: never an exception.

        The known failures have their own reasons. Any other exception on the
        way (resolving the configuration, building, sending, decoding,
        validating or recording) is caught here, as a backstop: unknown, with
        the exception's type only (its message may hold the token or a
        header), and a record of how far the request got.
        """
        progress = _Progress()
        try:
            record = self._judgement(ctx, progress)
        except UnrecordedJudgeRequest as missing:
            return self._unrecorded(ctx, missing, progress)
        except Exception as exc:
            record = progress.failed(self, exc)
            ctx.judgements.setdefault(progress.key or json.dumps(["failed", record.request_id]), record)
        return self.result(record)

    def _unrecorded(
        self, ctx: VerificationContext, missing: UnrecordedJudgeRequest, progress: _Progress
    ) -> VerificationResult:
        """A replay with no record for this request: unknown, naming the id, never a verdict.

        The answer is kept on the context as a failed :class:`JudgeRecord`, as
        every other unanswered request is, so the grade shows which request had
        no record. The reason is exactly ``judge-replay-unrecorded:<id8>``.
        """
        why = str(missing)
        request = progress.request or {"schema": SCHEMA, "id": missing.request_id}
        record = JudgeRecord(
            request_id=missing.request_id, endpoint=progress.endpoint, request=request, sent=False, failure=why
        )
        ctx.judgements.setdefault(progress.key or json.dumps(["replay", missing.request_id]), record)
        return VerificationResult(
            None, f"{self.label} (unknown: {why})", (ValidityReason(check=self.label, reason=why),)
        )

    def _judgement(self, ctx: VerificationContext, progress: _Progress) -> JudgeRecord:
        try:
            request = self.request(ctx)
        except Exception as exc:
            # Unknown, with what could be told of the request.
            request = self._unbuilt_request()
            why = str(exc) if isinstance(exc, _UnbuildableError) else type(exc).__name__
            failure = f"the judge request could not be built ({why})"
            unbuilt = JudgeRecord(request_id=request["id"], endpoint=None, request=request, sent=False, failure=failure)
            key = json.dumps(["unbuilt", request["id"], self.url, self.url_env, self.token_env])
            return ctx.judgements.setdefault(key, unbuilt)
        progress.request = request
        progress.key = json.dumps([request["id"], self.url, self.url_env, self.token_env, self.timeout_seconds])
        record = ctx.judgements.get(progress.key)
        if record is None and ctx.judge_replay is not None:
            # A re-grade answers from its records, by request id; one they don't hold raises here, unsent.
            record = ctx.judgements[progress.key] = ctx.judge_replay.ask(request)
        if record is None:
            record = ctx.judgements[progress.key] = self.ask(request, progress)
        return record

    def _unbuilt_request(self) -> dict[str, Any]:
        """What a request that couldn't be built is known to hold: the check's own claim and rubric, escaped.

        It doesn't depend on what failed: the claim and rubric go through the
        same escaping as every request's text, and when even that fails the
        request holds neither.
        """
        try:
            body = _json_data({"claim": self.claim, "rubric": self.rubric})
        except Exception:
            body = {}
        return {"schema": SCHEMA, "id": request_id(body), **body}

    def evaluate(self, ctx: VerificationContext) -> bool | None:
        return self.assess(ctx).passed

    @property
    def label(self) -> str:
        return f'reply judge "{_text(str(self.claim))}"'

    def request(self, ctx: VerificationContext) -> dict[str, Any]:
        """The ``midojo.judge.v1`` request body for ``ctx``: its ``id`` is the digest of the rest.

        The claim and rubric must be UTF-8 text, as parsing requires (a check
        built some other way may not be): else :class:`_UnbuildableError`.
        Everything else in it, what the agent and its tools produced, is
        written as JSON data by :func:`_json_data`, a lone surrogate escaped.
        """
        for name in ("claim", "rubric"):
            if _not_utf8(getattr(self, name)):
                raise _UnbuildableError(f"the check's {name} is not UTF-8 text: it has a lone surrogate")
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
                    "args": call.args,
                    "result": call.result,
                    "error": call.error,
                    "complete": call.complete,
                    "blocked": call.blocked,
                    "turn": call.turn,
                }
                for call in ctx.function_calls
            ]
            body["tool_calls_complete"] = ctx.function_calls_complete and all(c.complete for c in ctx.function_calls)
        body = _json_data(body)
        return {"schema": SCHEMA, "id": request_id(body), **body}

    def ask(self, request: dict[str, Any], progress: _Progress | None = None) -> JudgeRecord:
        """Send ``request`` to the service and keep what it answered, or why there is no answer.

        A URL or token that can't be used is a failure before anything is
        sent: its reason names where it came from, never the token, and
        shows no URL text. ``progress`` notes how far it got, for the
        backstop in :meth:`assess`.
        """
        progress = progress or _Progress()
        progress.request = request
        fields: dict[str, Any] = {"request_id": request["id"], "request": request, "endpoint": None, "sent": False}
        source = "the check's url" if self.url is not None else _text(str(self.url_env))
        url = self.url if self.url is not None else os.environ.get(self.url_env or "")
        if not url:
            return JudgeRecord(**fields, failure=f"no judge service URL: {source} is not set")
        if _not_utf8(url):  # an environment value with bytes that aren't UTF-8 decodes to lone surrogates
            return JudgeRecord(**fields, failure=f"the judge service URL in {source} is not UTF-8 text")
        try:
            endpoint = httpx.URL(url)
            cleaned = endpoint.copy_with(username=None, password=None, query=None, fragment=None)
            progress.endpoint = fields["endpoint"] = _text(str(cleaned))
        except Exception:
            return JudgeRecord(**fields, failure="the judge service URL is not a valid URL")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        token = None
        if self.token_env is not None:
            name = _text(str(self.token_env))
            token = os.environ.get(self.token_env)
            if not token:
                return JudgeRecord(**fields, failure=f"no judge service token: {name} is not set")
            if not _header_safe(token):
                failure = f"the judge service token in {name} can't be sent: it isn't printable ASCII"
                return JudgeRecord(**fields, failure=failure)
            headers["Authorization"] = f"Bearer {token}"
        record = _redacted(self._exchange(endpoint, headers, fields, progress), token)
        record.model_dump_json()  # the grade dumps it: one that can't be dumped is a failure (the backstop's)
        return record

    def _exchange(
        self, endpoint: httpx.URL, headers: dict[str, str], fields: dict[str, Any], progress: _Progress
    ) -> JudgeRecord:
        content = json.dumps(fields["request"], ensure_ascii=False).encode("utf-8")
        fields = {**fields, "sent": True}
        try:
            # No proxy or .netrc from the environment, and no redirect: the token goes only to the URL configured.
            with (
                _transport_quiet(),
                httpx.Client(timeout=self.timeout_seconds, trust_env=False, follow_redirects=False) as http,
            ):
                progress.sent = True  # from here on, the request may have gone out
                response = http.post(endpoint, content=content, headers=headers)
        except httpx.TimeoutException as exc:
            failure = f"the judge service timed out after {self.timeout_seconds:g}s ({type(exc).__name__})"
            return JudgeRecord(**fields, failure=failure)
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            # The exception's message may hold the URL; its type says enough.
            return JudgeRecord(**fields, failure=f"the judge service request failed ({type(exc).__name__})")
        progress.http_status = response.status_code
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
            # JSON's \ud800 escape decodes to a lone surrogate: escaped, as in a request, so the record dumps.
            rationale=_json_data(answer.get("rationale")),
            model=_json_data(answer.get("model")),
            details=_json_data(answer.get("details")),
        )

    def result(self, record: JudgeRecord) -> VerificationResult:
        """The check's verdict and reason from what the service answered."""
        # Nothing the service sent goes into a reason: not its model or rationale (the record keeps those), and a
        # verdict only as the schema's own word for it. A failure's text is MiDojo's own.
        request = f"request {record.request_id}"
        if record.failure is not None:
            return VerificationResult.unknown(self.label, f"{record.failure}; {request}")
        verdict = next((name for name in PASSED if name == record.verdict), "unresolved")
        if PASSED[verdict] is None:
            return VerificationResult.unknown(self.label, f"the judge service left it unresolved ({request})")
        return VerificationResult(PASSED[verdict], f"{self.label}: {verdict} ({request})")


def request_id(body: dict[str, Any]) -> str:
    """``sha256:<hex>`` of ``body`` as sorted-key compact UTF-8 JSON: the same question has the same id."""
    text = json.dumps({"schema": SCHEMA, **body}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    # A body from _json_data holds no lone surrogate; one that does still hashes, as the same bytes every time.
    return "sha256:" + hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def response_problem(answer: Any, expected_id: str) -> str | None:
    """Why ``answer`` isn't a valid ``midojo.judge.v1`` response to the request ``expected_id``, or None.

    The reason never holds anything the service sent, in any encoding: it
    names only the schema's own fields, what they must be and the JSON type
    found, and counts the fields outside the schema rather than naming
    them. So a token the service echoes back can't reach a reason, however
    the redaction of the record's own fields would have to spell it.
    """
    if type(answer) is not dict:
        return f"the answer is {_json_type(answer)}, not an object"
    if outside := len(set(answer) - _RESPONSE_KEYS):
        return f"it has {outside} field{'s' if outside > 1 else ''} outside the schema"
    if "schema" not in answer:
        return "it has no schema"
    if answer["schema"] != SCHEMA:
        return f'schema is not "{SCHEMA}"'
    if "id" in answer and answer["id"] != expected_id:
        return "id is not the request's id"
    if "verdict" not in answer:
        return "it has no verdict"
    verdict = answer["verdict"]
    if type(verdict) is not str:  # the type first: a list or an object isn't hashable
        return f"verdict is {_json_type(verdict)}, not a string"
    if verdict not in PASSED:
        return "verdict is not one of " + ", ".join(f'"{name}"' for name in PASSED)
    for name in ("rationale", "model"):
        if answer.get(name) is not None and type(answer[name]) is not str:
            return f"{name} is {_json_type(answer[name])}, not a string"
    if answer.get("details") is not None and type(answer["details"]) is not dict:
        return f"details is {_json_type(answer['details'])}, not an object"
    if _depth(answer.get("details")) > MAX_DETAILS_DEPTH:
        return f"details nests deeper than {MAX_DETAILS_DEPTH}"
    return None


def _json_type(value: Any) -> str:
    """The JSON type of a decoded value, as a reason names it: never the value."""
    if value is None:
        return "null"
    kinds = {dict: "an object", list: "an array", str: "a string", bool: "a boolean", int: "a number"}
    return kinds.get(type(value), "a number")


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
    for name in ("claim", "rubric", "url", "url_env", "token_env"):
        if _not_utf8(value.get(name)):
            raise ValueError(f"{NAME}: {name} is not UTF-8 text: it has a lone surrogate")
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
    bytes are UTF-8 text, undecodable bytes escaped (``\\xff``); text is itself
    with each lone surrogate escaped (``\\ud800``), as UTF-8 can't hold
    one; a list or tuple is a list, and a set or frozenset a list sorted by
    each item's sorted-key JSON; a dict whose keys are all text, still
    distinct once escaped, is an object, and any other dict a list of
    ``[key, value]`` pairs sorted by the whole pair's JSON, so no key is
    lost or merged and equal dicts write alike; a date, datetime, time or
    timedelta is its ISO 8601 text.
    """
    kind = type(value)
    if kind is str:
        return _text(value)
    if value is None or kind in (bool, int):
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
        pairs = [[_json_data(key), _json_data(item)] for key, item in value.items()]
        keys = [key for key, _ in pairs]
        if all(type(key) is str for key in value) and len(set(keys)) == len(keys):
            return dict(pairs)
        # Sorted by the whole pair, so keys that write alike (b"x" and "x") still sort the same every time.
        return sorted(pairs, key=_sort_key)
    if kind in (date, datetime, time, timedelta):
        return to_jsonable_python(value)
    raise TypeError(f"a {kind.__module__}.{kind.__qualname__} is not JSON data")


def _text(value: str) -> str:
    """``value`` with each lone surrogate escaped (``\\ud800``): text UTF-8 can hold, the same every time."""
    return value.encode("utf-8", "backslashreplace").decode("utf-8")


def _not_utf8(value: Any) -> bool:
    """Whether text in ``value`` (a string, or the keys and values of a mapping or list) has a lone surrogate."""
    if type(value) is str:
        return value != _text(value)
    if type(value) is dict:
        return any(_not_utf8(key) or _not_utf8(item) for key, item in value.items())
    if type(value) in (list, tuple):
        return any(_not_utf8(item) for item in value)
    return False


def _header_safe(token: str) -> bool:
    """Whether ``token`` can go in an Authorization header as it is: printable ASCII, no control character."""
    return all(" " <= character <= "~" for character in token)


@dataclass
class _Progress:
    """How far one judge request got: what the backstop's record says when a step fails unexpectedly."""

    request: dict[str, Any] | None = None
    key: str | None = None
    endpoint: str | None = None
    sent: bool = False
    http_status: int | None = None

    def failed(self, check: ReplyJudge, exc: Exception) -> JudgeRecord:
        """The record of a request stopped by ``exc``: its type only, nothing the service answered, no token."""
        failure = f"the judge check failed ({type(exc).__name__})"
        try:
            request = self.request if self.request is not None else check._unbuilt_request()
            fields = {"request_id": request["id"], "request": request, "endpoint": self.endpoint, "sent": self.sent}
            record = JudgeRecord(**fields, http_status=self.http_status, failure=failure)
            record.model_dump_json()
            return record
        except Exception:
            return JudgeRecord(request_id="", endpoint=None, request={}, sent=self.sent, failure=failure)


class _UnbuildableError(ValueError):
    """A check whose own settings can't make a request; its message is safe to show."""


def _sort_key(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class _NonFiniteError(ValueError):
    """The answer has ``NaN``, ``Infinity`` or ``-Infinity``, which JSON doesn't, or a number no float holds."""


def _refuse_constant(name: str) -> Any:
    # The parser only hands over these three; the reason names them in MiDojo's own words all the same.
    known = {"NaN": "NaN", "Infinity": "Infinity", "-Infinity": "-Infinity"}
    raise _NonFiniteError(f"is not JSON: it has {known.get(name, 'a constant JSON has no word for')}")


def _finite_float(text: str) -> float:
    number = float(text)
    if not math.isfinite(number):
        raise _NonFiniteError("has a number no float holds")  # not the number itself: the service wrote it
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
