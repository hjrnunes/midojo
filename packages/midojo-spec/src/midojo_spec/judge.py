"""``midojo.judge.v1``: the request a judge service is sent and the answer it gives back.

A check that asks a judge sends one request, a JSON object, to a judge service and reads
one response. This module holds the format's rules and nothing else: it names no HTTP
client and builds no request from a snapshot, so the MiDojo runtime, a judge service and a
replay all read the same rules.

Request fields: ``schema``, ``id``, ``claim``, ``rubric``, ``message``, ``reply`` and,
when the check asks for them, ``prior_turns`` and ``tool_calls`` with
``tool_calls_complete``. ``id`` is ``sha256:`` and the digest of every other field
(:func:`request_id`), so the same question always has the same id.

Response fields: ``schema``, ``id`` (optional), ``verdict`` (``supported``,
``contradicted`` or ``unresolved``) and the optional ``rationale``, ``model`` and
``details``. :func:`response_problem` says why an answer is not one.

The problems these functions return never hold anything the other side sent: they name the
schema's own fields, what they must be and the JSON type found, so a token a service echoes
back cannot reach a log line through a reason.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

SCHEMA = "midojo.judge.v1"
VERDICTS = ("supported", "contradicted", "unresolved")

REQUEST_FIELDS = frozenset(
    {"schema", "id", "claim", "rubric", "message", "reply", "prior_turns", "tool_calls", "tool_calls_complete"}
)
REQUIRED_FIELDS = frozenset({"schema", "id", "claim", "rubric", "message", "reply"})
TURN_FIELDS = frozenset({"message", "reply"})
CALL_FIELDS = frozenset({"function", "args", "result", "error", "complete", "blocked", "turn"})
RESPONSE_FIELDS = frozenset({"schema", "id", "verdict", "rationale", "model", "details"})
# How deep ``details`` may nest: deeper data can't be dumped into a grade.
MAX_DETAILS_DEPTH = 64


def request_id(body: dict[str, Any]) -> str:
    """``sha256:<hex>`` of ``body`` as sorted-key compact UTF-8 JSON: the same question has the same id.

    ``body`` is every request field but ``schema`` and ``id``; the schema is part of what
    is hashed. A text with a lone surrogate still hashes, as the same bytes every time.
    """
    text = json.dumps({"schema": SCHEMA, **body}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def id_holds(request: dict[str, Any]) -> bool:
    """Whether ``request``'s ``id`` is the digest of its other fields."""
    body = {name: value for name, value in request.items() if name not in ("schema", "id")}
    try:
        return request.get("id") == request_id(body)
    except (TypeError, ValueError):
        return False


def request_problem(request: Any) -> str | None:
    """Why ``request`` is not a ``midojo.judge.v1`` request, or None.

    The shape only: it does not check that ``id`` is the digest (:func:`id_holds` does).
    """
    if not isinstance(request, dict):
        return "the request is not a JSON object"
    if request.get("schema") != SCHEMA:
        return f"schema is not {SCHEMA}"
    if outside := len(set(request) - REQUEST_FIELDS):
        return f"the request has {outside} fields outside the schema"
    if missing := sorted(REQUIRED_FIELDS - set(request)):
        return f"the request has no {', '.join(missing)}"
    if not isinstance(request["id"], str):
        return "id is not a string"
    if not isinstance(request["claim"], str) or not request["claim"].strip():
        return "claim is not a non-empty string"
    if request["rubric"] is not None and not isinstance(request["rubric"], str | dict):
        return "rubric is not a string, an object or null"
    for name in ("message", "reply"):
        if not _optional_text(request[name]):
            return f"{name} is not a string or null"
    if "prior_turns" in request and not _list_of(request["prior_turns"], _turn_shape):
        return "prior_turns is not a list of turns"
    if ("tool_calls" in request) != ("tool_calls_complete" in request):
        return "tool_calls and tool_calls_complete come together"
    if "tool_calls" in request and not _list_of(request["tool_calls"], _call_shape):
        return "tool_calls is not a list of calls"
    if "tool_calls_complete" in request and type(request["tool_calls_complete"]) is not bool:
        return "tool_calls_complete is not a boolean"
    return None


def response_problem(answer: Any, expected_id: str) -> str | None:
    """Why ``answer`` isn't a valid ``midojo.judge.v1`` response to the request ``expected_id``, or None.

    The reason never holds anything the service sent, in any encoding: it names only the
    schema's own fields, what they must be and the JSON type found, and counts the fields
    outside the schema rather than naming them.
    """
    if type(answer) is not dict:
        return f"the answer is {json_type(answer)}, not an object"
    if outside := len(set(answer) - RESPONSE_FIELDS):
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
        return f"verdict is {json_type(verdict)}, not a string"
    if verdict not in VERDICTS:
        return "verdict is not one of " + ", ".join(f'"{name}"' for name in VERDICTS)
    for name in ("rationale", "model"):
        if answer.get(name) is not None and type(answer[name]) is not str:
            return f"{name} is {json_type(answer[name])}, not a string"
    if answer.get("details") is not None and type(answer["details"]) is not dict:
        return f"details is {json_type(answer['details'])}, not an object"
    if depth(answer.get("details")) > MAX_DETAILS_DEPTH:
        return f"details nests deeper than {MAX_DETAILS_DEPTH}"
    return None


def json_type(value: Any) -> str:
    """The JSON type of a decoded value, as a reason names it: never the value."""
    if value is None:
        return "null"
    kinds = {dict: "an object", list: "an array", str: "a string", bool: "a boolean", int: "a number"}
    return kinds.get(type(value), "a number")


def depth(value: Any) -> int:
    """How many objects and arrays deep ``value`` nests, counted level by level rather than by recursion."""
    levels, level = 0, [value]
    while containers := [item for item in level if type(item) in (dict, list)]:
        levels += 1
        level = [child for item in containers for child in (item.values() if type(item) is dict else item)]
    return levels


def _optional_text(value: Any) -> bool:
    return value is None or isinstance(value, str)


def _list_of(value: Any, shape: Callable[[Any], bool]) -> bool:
    return isinstance(value, list) and all(shape(item) for item in value)


def _turn_shape(turn: Any) -> bool:
    return (
        isinstance(turn, dict) and set(turn) == TURN_FIELDS and all(_optional_text(turn[name]) for name in TURN_FIELDS)
    )


def _call_shape(call: Any) -> bool:
    return (
        isinstance(call, dict)
        and set(call) == CALL_FIELDS
        and isinstance(call["function"], str)
        and _optional_text(call["error"])
        and type(call["complete"]) is bool
        and type(call["blocked"]) is bool
        and (call["turn"] is None or type(call["turn"]) is int)
    )
