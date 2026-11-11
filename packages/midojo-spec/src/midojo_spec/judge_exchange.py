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
row's digests equal the grade's whatever was redacted. A row that was not redacted must hold
the digests of its own text, and :func:`exchange_problem` refuses one that does not; a
redacted row cannot be checked, so :func:`digests_hold` says ``None`` for it.

Redaction replaces each occurrence of a secret in a text field with ``[redacted]``, in one pass
over each text, so overlapping secrets leave no tail and a marker is never rescanned. It does not
reassemble a secret that two fields hold in halves. ``model`` and ``prompt_version`` are the
judge's identity and are never rewritten: a secret that occurs in either, or in the marker
itself, is refused with :class:`ExchangeFormatError`, because a row that leaked it or lost the
identity a replay checks would be worse than no row.

The file is ASCII JSON, one object on each line, so a lone surrogate in a model's reply
survives a round trip and its digest still holds. :class:`ExchangeLog` appends under a lock that
every log on one file in one process shares, with each row in one ``write`` of a descriptor
opened for append; a write that fails is cut back, so no partial row stays. One process appends
to a file at a time. A reader (:func:`read_exchanges`) is strict: a line that is not a row, or a
file whose sequences do not rise, is an :class:`ExchangeFormatError` naming the line and the
rule, never its content.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import threading
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = "midojo.judge-exchange.v1"
# A suggested name: the file's owner chooses where it lives.
FILENAME = "judge-record.jsonl"
REDACTED = "[redacted]"
# ``usage`` is the model service's own object; one nested deeper than this is dropped, not stored.
MAX_USAGE_DEPTH = 8
# Every integer in a row fits a signed 64-bit integer: one that does not is dropped from ``usage`` and
# refused elsewhere, because Python writes only integers of up to 4300 digits as JSON and another
# language's reader may hold no more than 64 bits.
MAX_INTEGER = 2**63 - 1
MIN_INTEGER = -(2**63)

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
    if not row["redacted"]:
        # Only a row that kept its text can vouch for its digests; a redacted one cannot be checked.
        if row["request_sha256"] != sha256_text(row["messages"][-1]["content"]):
            return "request_sha256 is not the digest of the last message"
        if row["reply"] is not None and row["response_sha256"] != sha256_text(row["reply"]):
            return "response_sha256 is not the digest of the reply"
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

    Digests, field names and the numbers stay as they are. Each text is scanned once for every
    occurrence of every secret, and each run of overlapping or touching occurrences becomes one
    marker, so no tail of a secret that contains or overlaps another stays, and a marker is never
    scanned again. ``redacted`` is set when a replacement happened.

    Raises :class:`ExchangeFormatError` when a secret occurs in the ``model`` or the
    ``prompt_version`` (the identity a replay checks, which is never rewritten), in the marker
    itself, or would be spelled by the marker and the text beside it. The message is fixed text
    and quotes no secret (a secret short enough to be a word of it, such as ``redacted``, is
    still a word of it). A secret that two messages hold in halves is in neither and is not found.
    """
    found = sorted({secret for secret in secrets if secret}, key=len, reverse=True)
    if not found:
        return exchange
    if any(secret in REDACTED for secret in found):
        raise ExchangeFormatError(f"a secret occurs in the redaction marker {REDACTED}")
    if any(secret in (exchange.model or "") or secret in exchange.prompt_version for secret in found):
        raise ExchangeFormatError(
            "a secret occurs in the model or the prompt version, which identify the judge and are never redacted"
        )
    scrubbed = dataclasses.replace(
        exchange,
        messages=tuple((_scrub(role, found), _scrub(content, found)) for role, content in exchange.messages),
        reply=_scrub(exchange.reply, found),
        error=_scrub(exchange.error, found),
        finish_reason=_scrub(exchange.finish_reason, found),
        response_model=_scrub(exchange.response_model, found),
        usage=_scrub(exchange.usage, found),
    )
    texts = [
        *(text for message in scrubbed.messages for text in message),
        *_strings(scrubbed.reply),
        *_strings(scrubbed.error),
        *_strings(scrubbed.finish_reason),
        *_strings(scrubbed.response_model),
        *_strings(scrubbed.usage),
    ]
    if any(secret in text for secret in found for text in texts):
        # Only the marker joined to its neighbours can still spell a secret.
        raise ExchangeFormatError(f"a secret is spelled by the redaction marker {REDACTED} and the text beside it")
    return dataclasses.replace(scrubbed, redacted=exchange.redacted or scrubbed != exchange)


def _strings(value: Any) -> Iterable[str]:
    if type(value) is str:
        yield value
    elif type(value) is dict:
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)
    elif type(value) is list:
        for item in value:
            yield from _strings(item)


class _Shared:
    """What every log on one file in this process shares: the lock, and the file as this process left it."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # The file's (size, modification time) after this process's last append, and the sequence its
        # next row takes. A file that is not as this process left it was written by someone else, and
        # it is read again before the next row.
        self.stamp: tuple[int, int] | None = None
        self.next = 0


# Keyed by the file itself (device and inode), never by a path: a relative path, a symlink or a
# renamed directory names the same file or another one, and only the open file knows which.
_SHARED: dict[tuple[int, int], _Shared] = {}
_SHARED_LOCK = threading.Lock()


class ExchangeLog:
    """An append-only JSONL file of exchanges: thread-safe, one row in one ``write``.

    One process appends to a file at a time. Every ``ExchangeLog`` on the same file in a process
    shares one lock and one count, so two of them number the rows as one, whatever path each names.
    The path is made absolute when the log is made, so a later change of directory does not move it.

    ``sequence`` continues from the rows already in the file, which are read strictly the first time
    and again whenever the file's size or modification time is not what this process left: a file
    that does not read as a log is refused, and so is one that ends in a partial row. An edit that
    keeps both the size and the modification time is not noticed until the file is read. A write that
    fails is cut back to where it began, so no partial row stays. The file is created with mode 0600,
    as it holds prompts and replies.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(os.path.abspath(path))

    def append(self, exchange: Exchange) -> Exchange:
        """Write ``exchange`` as the next row and return it with its ``sequence``.

        A row that is not valid is refused before anything is written, and so is a file that does
        not read as a log, or whose last row already holds the largest sequence (``MAX_INTEGER``).
        """
        if (problem := exchange_problem(dataclasses.replace(exchange, sequence=0).to_json())) is not None:
            raise ExchangeFormatError(problem)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            identity = os.fstat(descriptor)
            with _SHARED_LOCK:
                shared = _SHARED.setdefault((identity.st_dev, identity.st_ino), _Shared())
            with shared.lock:
                return self._append(descriptor, shared, exchange)
        finally:
            os.close(descriptor)

    @staticmethod
    def _append(descriptor: int, shared: _Shared, exchange: Exchange) -> Exchange:
        status = os.fstat(descriptor)
        size = status.st_size
        if size == 0 or shared.stamp != (size, status.st_mtime_ns):
            shared.next = _continued(descriptor, size)
        if shared.next > MAX_INTEGER:
            # The writer makes this one number itself, so it is checked here: the row's own check ran with 0.
            raise ExchangeFormatError("sequence is not valid: the log holds the last row a 64-bit sequence can number")
        row = dataclasses.replace(exchange, sequence=shared.next)
        line = json.dumps(row.to_json(), ensure_ascii=True, separators=(",", ":")).encode("ascii") + b"\n"
        try:
            view = memoryview(line)
            while view:
                view = view[os.write(descriptor, view) :]
        except BaseException:
            # Cut the write back: a torn row left here would be glued to the next one. Whether or not
            # the cut works, the file is read again before the next row.
            shared.stamp = None
            with suppress(OSError):
                os.ftruncate(descriptor, size)
            raise
        after = os.fstat(descriptor)
        shared.stamp = (after.st_size, after.st_mtime_ns)
        shared.next += 1
        return row


def _continued(descriptor: int, size: int) -> int:
    """The sequence the next row takes in the open file of ``size`` bytes: after the last row, read strictly."""
    if size == 0:
        return 0
    with os.fdopen(os.dup(descriptor), "rb") as existing:
        existing.seek(-1, os.SEEK_END)
        if existing.read(1) != b"\n":
            raise ExchangeFormatError("the file ends in a partial row; nothing was written")
        existing.seek(0)
        rows = _read(existing)
    last = rows[-1].sequence if rows else -1
    assert last is not None  # a row read from a file has its sequence
    return last + 1


def read_exchanges(path: str | os.PathLike[str]) -> list[Exchange]:
    """Every row of the file at ``path``, in order; :class:`ExchangeFormatError` for any line that is not one.

    A blank line, a repeated JSON key, ``NaN`` and a sequence that does not rise are all
    refused, with the line's number and the rule.
    """
    with Path(path).open("rb") as lines:
        return _read(lines)


def _read(lines: Iterable[bytes]) -> list[Exchange]:
    exchanges: list[Exchange] = []
    last = -1
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
        return _scrub_text(value, secrets)
    if type(value) is dict:
        return {_scrub(key, secrets): _scrub(item, secrets) for key, item in value.items()}
    if type(value) is list:
        return [_scrub(item, secrets) for item in value]
    return value


def _scrub_text(text: str, secrets: Sequence[str]) -> str:
    """``text`` with every run of overlapping or touching secret occurrences replaced by one marker."""
    spans: list[tuple[int, int]] = []
    for secret in secrets:
        start = text.find(secret)
        while start != -1:
            spans.append((start, start + len(secret)))
            start = text.find(secret, start + 1)
    if not spans:
        return text
    spans.sort()
    pieces: list[str] = []
    cursor = 0
    run_start, run_end = spans[0]
    for start, end in [*spans[1:], (len(text) + 1, len(text) + 1)]:
        if start <= run_end:
            run_end = max(run_end, end)
            continue
        pieces.append(text[cursor:run_start])
        pieces.append(REDACTED)
        cursor = run_end
        run_start, run_end = start, end
    pieces.append(text[cursor:])
    return "".join(pieces)


def _usage_fits(usage: Any) -> bool:
    """Whether ``usage`` is plain JSON the file can hold: text keys, finite numbers, at most ``MAX_USAGE_DEPTH`` deep."""
    if usage is None:
        return True
    if type(usage) is not dict:
        return False
    stack: list[tuple[Any, int]] = [(usage, 1)]
    while stack:
        value, level = stack.pop()
        kind = type(value)
        if kind is dict or kind is list:
            if level > MAX_USAGE_DEPTH:
                return False
            if kind is dict:
                if any(type(key) is not str for key in value):
                    return False
                stack.extend((item, level + 1) for item in value.values())
            else:
                stack.extend((item, level + 1) for item in value)
        elif kind is float:
            if not math.isfinite(value):
                return False
        elif kind is int:
            if not MIN_INTEGER <= value <= MAX_INTEGER:
                return False
        elif kind not in (str, bool, type(None)):
            return False
    return True


def _index(value: Any) -> bool:
    return type(value) is int and 0 <= value <= MAX_INTEGER


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


_UTC_SUFFIXES = ("+00:00", "Z")
"""The two spellings of UTC a ``started_at`` ends with: the one the writer emits, and the ISO 8601 designator."""


def _timestamp(value: Any) -> bool:
    if type(value) is not str:
        return False
    # UTC is part of the format, and so is how it is spelled: a reader that took any zone would accept a time its
    # writers never produce, and ``fromisoformat`` reads ``+0000``, ``-00:00`` and ``+00:00:00.1`` as a zero offset too.
    if not value.endswith(_UTC_SUFFIXES):
        return False
    try:
        return datetime.fromisoformat(value).utcoffset() == timedelta(0)
    except ValueError:
        return False
