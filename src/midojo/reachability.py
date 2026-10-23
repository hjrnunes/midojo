"""Whether an injection payload can be found in what an evaluation recorded.

Reachability is report-only. It never changes or invalidates a grade, and it
never decides which evaluations count in the attack-success rate: a payload
the search misses (for example one a tool re-encoded) says nothing about
whether the attack worked.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from typing import Any, Literal

from pydantic import BaseModel

from midojo.observations import EvidenceStatus

ReachabilityStatus = Literal["matched", "not_matched", "unknown"]

# How the payload was found:
#   raw_text              in the text as recorded
#   decoded_json          in a string (or key) of the text parsed as JSON
#   json_encoded_payload  as the payload's JSON string encoding (escaped quotes,
#                         ``\n``, ``\uXXXX``) in text that doesn't parse as JSON
ReachabilityMethod = Literal["raw_text", "decoded_json", "json_encoded_payload"]

# Where the payload was found:
#   agent_input           the prompt the orchestrator sent the agent
#   recorded_tool_result  a tool result as the interceptor recorded it. The SDKs
#                         record a result before returning it to the agent, so
#                         this does not prove the agent saw it
ReachabilityBoundary = Literal["agent_input", "recorded_tool_result"]


class Reachability(BaseModel):
    status: ReachabilityStatus
    method: ReachabilityMethod | None = None
    boundary: ReachabilityBoundary | None = None


UNKNOWN = Reachability(status="unknown")


def _normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def _json_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _json_strings(item)


def _encoded(payload: str) -> set[str]:
    # The payload's JSON string encoding, without the surrounding quotes, with
    # and without non-ASCII characters escaped.
    return {json.dumps(payload, ensure_ascii=ascii)[1:-1] for ascii in (True, False)}


def _match_tool_result(result: str, payloads: list[str]) -> ReachabilityMethod | None:
    normalized = [_normalize(p) for p in payloads]
    if any(p in _normalize(result) for p in normalized):
        return "raw_text"
    try:
        decoded = json.loads(result)
    except ValueError:
        encoded = [_normalize(e) for p in payloads for e in _encoded(p)]
        if any(e in _normalize(result) for e in encoded):
            return "json_encoded_payload"
        return None
    strings = [_normalize(s) for s in _json_strings(decoded)]
    if any(p in s for p in normalized for s in strings):
        return "decoded_json"
    return None


def reachability(
    payloads: Iterable[str],
    agent_input: str | None,
    tool_results: Iterable[str],
    function_calls_status: EvidenceStatus | None = None,
) -> Reachability:
    """Where, if anywhere, one of the payloads appears in an evaluation's records.

    Text is compared with whitespace collapsed and case folded. The agent input
    is searched as raw text. Each recorded tool result is searched as raw text;
    if it parses as JSON, its decoded strings are searched too, and if it
    doesn't, the payload's JSON string encoding is searched instead. The first
    match wins, the agent input before the tool results.

    ``status`` is:

    - ``matched``, with the ``method`` and ``boundary`` of the first match;
    - ``not_matched`` when no payload was found and the recorded function calls
      are complete, or report no status (``function_calls_status`` None);
    - ``unknown`` when there's no non-empty payload to search for, or nothing
      matched while the function-call evidence is reported but not
      ``complete``. Callers also report ``unknown`` when the evaluation record
      couldn't be read.

    ``method`` and ``boundary`` are None unless the status is ``matched``.
    """
    payloads = [p for p in payloads if p and p.strip()]
    if not payloads:
        return UNKNOWN
    if agent_input and any(_normalize(p) in _normalize(agent_input) for p in payloads):
        return Reachability(status="matched", method="raw_text", boundary="agent_input")
    for result in tool_results:
        method = _match_tool_result(result, payloads)
        if method:
            return Reachability(status="matched", method=method, boundary="recorded_tool_result")
    if function_calls_status not in (None, "complete"):
        return UNKNOWN
    return Reachability(status="not_matched")


def count(reports: Iterable[Reachability]) -> dict[ReachabilityStatus, int]:
    """How many reports have each status."""
    counts: dict[ReachabilityStatus, int] = {"matched": 0, "not_matched": 0, "unknown": 0}
    for report in reports:
        counts[report.status] += 1
    return counts
