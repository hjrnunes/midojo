"""The judge exchange record: one JSONL row for each model request a judge makes.

A judge service answers a ``midojo.judge.v1`` request by asking a model, once for each
judged turn. Each of those model requests leaves one row in an append-only file, whether
the model answered or the call failed, so a later reading (a replay, an audit, a re-judge
with another model) never needs the model again.

A row (:class:`Exchange`, schema ``midojo.judge-exchange.v1``) holds:

- ``sequence``: the row's place in its file, assigned by :class:`ExchangeLog`;
- ``request_id`` and ``ask``: the key. ``request_id`` is the ``midojo.judge.v1`` request the
  model request answers, and ``ask`` counts that answer's model requests from 0, in the
  order the service sent them. A service asked the same request twice writes the key twice,
  and a reader treats the rows as candidates; it never takes the last as the truth;
- ``model`` and ``prompt_version``: the identity of the judge that asked. A replay reuses a
  row only when both are the ones it asks for;
- ``messages``, ``reply``, ``error``, ``http_status``, ``finish_reason``, ``response_model``
  and ``usage``: what was sent and what came back. ``error`` is the failed call's label
  (``APITimeoutError``) and excludes a ``reply``;
- ``request_sha256`` and ``response_sha256``: the SHA-256 (lowercase hex) of the last
  message's text, the judge request the model read, and of the reply's text. They are the
  digests the sealed grade's judgements hold for the same turn;
- ``started_at`` (UTC, ISO 8601) and ``duration_ms``;
- ``redacted``: true when a secret was replaced by ``[redacted]`` in a text field.

The digests are taken from the text as it was sent and received, before any redaction, so a
row's digests equal the grade's whatever was redacted. :func:`digests_hold` therefore
recomputes them only for a row that was not redacted, and says so (``None``) otherwise.

The file is ASCII JSON, one object on each line, so a lone surrogate in a model's reply
survives a round trip and its digest still holds. :class:`ExchangeLog` appends under a lock,
with each row in one ``write`` of a descriptor opened for append. A reader
(:func:`read_exchanges`) is strict: a line that is not a row, or a file whose sequences do not
rise, is an :class:`ExchangeFormatError` naming the line and the rule, never its content.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "midojo.judge-exchange.v1"
# A suggested name: the file's owner chooses where it lives.
FILENAME = "judge-record.jsonl"
REDACTED = "[redacted]"
# ``usage`` is the model service's own object; one nested deeper than this is dropped, not stored.
MAX_USAGE_DEPTH = 8

FIELDS = (
    "schema",
    "sequence",
    "request_id",
    "ask",
    "model",
    "prompt_version",
    "messages",
    "request_sha256",
    "reply",
    "response_sha256",
    "error",
    "http_status",
    "finish_reason",
    "response_model",
    "usage",
    "started_at",
    "duration_ms",
    "redacted",
)
_REQUEST_ID = re.compile(r"sha256:[0-9a-f]{64}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


class ExchangeFormatError(ValueError):
    """A row or a file that is not in the exchange format; the message names the rule, never the data."""


def sha256_text(text: str) -> str:
    """The SHA-256 (lowercase hex) of ``text``'s UTF-8 bytes, a lone surrogate kept as it is."""
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def now_iso() -> str:
    """The current UTC time as ISO 8601 text to the millisecond: ``2026-11-11T09:30:00.123+00:00``."""
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class Observed:
    """What one model request showed, before it is tied to a judge request.

    ``content`` is the reply's text when the model answered with text; ``error`` is the
    failed call's label; ``http_status`` is None when no status line came back.
    """

    started_at: str
    duration_ms: int
    http_status: int | None = None
    content: str | None = None
    finish_reason: str | None = None
    response_model: str | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class Exchange:
    """One model request of a judge and its outcome; see the module docstring for each field."""

    request_id: str
    ask: int
    model: str | None
    prompt_version: str
    messages: tuple[tuple[str, str], ...]
    request_sha256: str
    reply: str | None
    response_sha256: str | None
    error: str | None
    http_status: int | None
    finish_reason: str | None
    response_model: str | None
    usage: dict[str, Any] | None
    started_at: str
    duration_ms: int
    redacted: bool = False
    sequence: int | None = None

    @classmethod
    def observed(
        cls,
        observed: Observed,
        *,
        request_id: str,
        ask: int,
        model: str | None,
        prompt_version: str,
        messages: Sequence[Mapping[str, str]],
        secrets: Iterable[str] = (),
    ) -> Exchange:
        """The row for a model request ``observed`` answered, with ``secrets`` replaced by ``[redacted]``.

        The digests are of the text as sent and received. ``messages`` are the chat messages
        sent (``role`` and ``content``); the last is the judge request the model read.
        """
        if not messages:
            raise ExchangeFormatError("an exchange needs the messages it sent")
        sent = tuple((str(message["role"]), str(message["content"])) for message in messages)
        usage = observed.usage if _usage_fits(observed.usage) else None
        row = cls(
            request_id=request_id,
            ask=ask,
            model=model,
            prompt_version=prompt_version,
            messages=sent,
            request_sha256=sha256_text(sent[-1][1]),
            reply=observed.content,
            response_sha256=None if observed.content is None else sha256_text(observed.content),
            error=observed.error,
            http_status=observed.http_status,
            finish_reason=observed.finish_reason,
            response_model=observed.response_model,
            usage=usage,
            started_at=observed.started_at,
            duration_ms=observed.duration_ms,
        )
        return redact(row, secrets)

    def to_json(self) -> dict[str, Any]:
        """The row as the JSON object the file holds, its fields in the schema's order."""
        return {
            "schema": SCHEMA,
            "sequence": self.sequence,
            "request_id": self.request_id,
            "ask": self.ask,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "messages": [{"role": role, "content": content} for role, content in self.messages],
            "request_sha256": self.request_sha256,
            "reply": self.reply,
            "response_sha256": self.response_sha256,
            "error": self.error,
            "http_status": self.http_status,
            "finish_reason": self.finish_reason,
            "response_model": self.response_model,
            "usage": self.usage,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "redacted": self.redacted,
        }

    @classmethod
    def from_json(cls, row: Any) -> Exchange:
        """The exchange a row holds, or :class:`ExchangeFormatError` naming the first rule it breaks."""
        if (problem := exchange_problem(row)) is not None:
            raise ExchangeFormatError(problem)
        return cls(
            request_id=row["request_id"],
            ask=row["ask"],
            model=row["model"],
            prompt_version=row["prompt_version"],
            messages=tuple((message["role"], message["content"]) for message in row["messages"]),
            request_sha256=row["request_sha256"],
            reply=row["reply"],
            response_sha256=row["response_sha256"],
            error=row["error"],
            http_status=row["http_status"],
            finish_reason=row["finish_reason"],
            response_model=row["response_model"],
            usage=row["usage"],
            started_at=row["started_at"],
            duration_ms=row["duration_ms"],
            redacted=row["redacted"],
            sequence=row["sequence"],
        )


def exchange_problem(row: Any) -> str | None:
    """Why ``row`` is not a ``midojo.judge-exchange.v1`` row, or None.

    The reason names fields, their types and the rule; it never holds what the row says.
    """
    if type(row) is not dict:
        return "the row is not an object"
    if row.get("schema") != SCHEMA:
        return f'schema is not "{SCHEMA}"'
    if outside := len(set(row) - set(FIELDS)):
        return f"the row has {outside} field{'s' if outside > 1 else ''} outside the schema"
    if missing := [name for name in FIELDS if name not in row]:
        return f"the row has no {', '.join(missing)}"
    checks = (
        ("sequence", _index(row["sequence"])),
        ("request_id", _matches(_REQUEST_ID, row["request_id"])),
        ("ask", _index(row["ask"])),
        ("model", _optional_text(row["model"])),
        ("prompt_version", type(row["prompt_version"]) is str and bool(row["prompt_version"])),
        ("messages", _messages(row["messages"])),
        ("request_sha256", _matches(_DIGEST, row["request_sha256"])),
        ("reply", _optional_text(row["reply"])),
        ("response_sha256", row["response_sha256"] is None or _matches(_DIGEST, row["response_sha256"])),
        ("error", _optional_text(row["error"])),
        ("http_status", row["http_status"] is None or _index(row["http_status"])),
        ("finish_reason", _optional_text(row["finish_reason"])),
        ("response_model", _optional_text(row["response_model"])),
        ("usage", row["usage"] is None or (type(row["usage"]) is dict and _usage_fits(row["usage"]))),
        ("started_at", _timestamp(row["started_at"])),
        ("duration_ms", _index(row["duration_ms"])),
        ("redacted", type(row["redacted"]) is bool),
    )
    for name, holds in checks:
        if not holds:
            return f"{name} is not valid"
    if (row["reply"] is None) != (row["response_sha256"] is None):
        return "reply and response_sha256 are both present or both null"
    if row["error"] is not None and row["reply"] is not None:
        return "a failed call has no reply"
    return None


def digests_hold(exchange: Exchange) -> bool | None:
    """Whether the row's digests are those of its text: None for a redacted row, which cannot say."""
    if exchange.redacted:
        return None
    if exchange.request_sha256 != sha256_text(exchange.messages[-1][1]):
        return False
    return (exchange.reply is None and exchange.response_sha256 is None) or (
        exchange.reply is not None and exchange.response_sha256 == sha256_text(exchange.reply)
    )


def redact(exchange: Exchange, secrets: Iterable[str]) -> Exchange:
    """``exchange`` with each of ``secrets`` replaced by ``[redacted]`` in every text it holds.

    Digests, field names and the numbers stay as they are. The longest secret goes first, so
    one that contains another leaves no tail. ``redacted`` is set when a replacement happened.
    """
    found = sorted({secret for secret in secrets if secret}, key=len, reverse=True)
    if not found:
        return exchange
    scrubbed = dataclasses.replace(
        exchange,
        model=_scrub(exchange.model, found),
        messages=tuple((_scrub(role, found), _scrub(content, found)) for role, content in exchange.messages),
        reply=_scrub(exchange.reply, found),
        error=_scrub(exchange.error, found),
        finish_reason=_scrub(exchange.finish_reason, found),
        response_model=_scrub(exchange.response_model, found),
        usage=_scrub(exchange.usage, found),
    )
    return dataclasses.replace(scrubbed, redacted=exchange.redacted or scrubbed != exchange)


class ExchangeLog:
    """An append-only JSONL file of exchanges: thread-safe, one row in one ``write``.

    One process writes a file: ``sequence`` continues from the rows already there, and the
    lock that orders it is this object's. The file is created with mode 0600, as it holds
    prompts and replies.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._next = 0
        if self.path.exists():
            with self.path.open("rb") as existing:
                self._next = sum(1 for line in existing if line.strip())

    def append(self, exchange: Exchange) -> Exchange:
        """Write ``exchange`` as the next row and return it with its ``sequence``.

        A row that is not valid is refused before anything is written.
        """
        with self._lock:
            row = dataclasses.replace(exchange, sequence=self._next)
            document = row.to_json()
            if (problem := exchange_problem(document)) is not None:
                raise ExchangeFormatError(problem)
            line = json.dumps(document, ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n"
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                view = memoryview(line)
                while view:
                    view = view[os.write(descriptor, view) :]
            finally:
                os.close(descriptor)
            self._next += 1
            return row


def read_exchanges(path: str | os.PathLike[str]) -> list[Exchange]:
    """Every row of the file at ``path``, in order; :class:`ExchangeFormatError` for any line that is not one.

    A blank line, a repeated JSON key, ``NaN`` and a sequence that does not rise are all
    refused, with the line's number and the rule.
    """
    exchanges: list[Exchange] = []
    last = -1
    with Path(path).open("rb") as lines:
        for number, line in enumerate(lines, start=1):
            try:
                row = json.loads(line.decode("utf-8"), object_pairs_hook=_no_repeats, parse_constant=_no_constants)
                exchange = Exchange.from_json(row)
            except (ValueError, RecursionError) as problem:
                reason = problem if isinstance(problem, ExchangeFormatError) else "it is not JSON"
                raise ExchangeFormatError(f"line {number}: {reason}") from None
            assert exchange.sequence is not None
            if exchange.sequence <= last:
                raise ExchangeFormatError(f"line {number}: sequence does not rise")
            last = exchange.sequence
            exchanges.append(exchange)
    return exchanges


def by_key(exchanges: Iterable[Exchange]) -> dict[tuple[str, int], list[Exchange]]:
    """The rows grouped by ``(request_id, ask)``, each group in file order: the candidates for that key."""
    groups: dict[tuple[str, int], list[Exchange]] = {}
    for exchange in exchanges:
        groups.setdefault((exchange.request_id, exchange.ask), []).append(exchange)
    return groups


def _no_repeats(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ExchangeFormatError("a JSON key repeats")
    return dict(pairs)


def _no_constants(name: str) -> Any:
    raise ExchangeFormatError("it has NaN or Infinity, which JSON does not")


def _scrub(value: Any, secrets: Sequence[str]) -> Any:
    if type(value) is str:
        for secret in secrets:
            value = value.replace(secret, REDACTED)
        return value
    if type(value) is dict:
        return {_scrub(key, secrets): _scrub(item, secrets) for key, item in value.items()}
    if type(value) is list:
        return [_scrub(item, secrets) for item in value]
    return value


def _usage_fits(usage: Any) -> bool:
    if usage is None:
        return True
    if type(usage) is not dict:
        return False
    levels, level = 0, [usage]
    while containers := [item for item in level if type(item) in (dict, list)]:
        levels += 1
        level = [child for item in containers for child in (item.values() if type(item) is dict else item)]
    return levels <= MAX_USAGE_DEPTH


def _index(value: Any) -> bool:
    return type(value) is int and value >= 0


def _matches(pattern: re.Pattern[str], value: Any) -> bool:
    return type(value) is str and pattern.fullmatch(value) is not None


def _optional_text(value: Any) -> bool:
    return value is None or type(value) is str


def _messages(value: Any) -> bool:
    return (
        type(value) is list
        and bool(value)
        and all(
            type(message) is dict
            and set(message) == {"role", "content"}
            and type(message["role"]) is str
            and type(message["content"]) is str
            for message in value
        )
    )


def _timestamp(value: Any) -> bool:
    if type(value) is not str:
        return False
    try:
        return datetime.fromisoformat(value).tzinfo is not None
    except ValueError:
        return False
