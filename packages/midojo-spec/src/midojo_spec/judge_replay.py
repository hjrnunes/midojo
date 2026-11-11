"""Strict offline replay of judge exchanges under the requested judge identity.

Only rows with the same request key, model and prompt version can answer. A failed
call remains a failed call; replay never asks a service or reads configuration.
Refusal messages name rules and fields, not recorded text or judge identities.
"""

from __future__ import annotations

from collections.abc import Iterable

from .judge_exchange import Exchange


class ReplayRefusal(LookupError):  # noqa: N818 - the public replay contract names this base.
    """No unambiguous exchange answers the key; ``why`` is a stable reason code."""

    why = "judge-replay"

    def __init__(self, request_id: str, ask: int) -> None:
        self.request_id = request_id
        self.ask = ask
        super().__init__(f"{self.why}: request_id and ask cannot be replayed")


class UnrecordedExchange(ReplayRefusal):
    """No row has this request_id and ask."""

    why = "judge-replay-unrecorded"


class IdentityRefused(ReplayRefusal):
    """Rows have the key, but none has the requested judge identity."""

    why = "judge-replay-identity"

    def __init__(self, request_id: str, ask: int, reason: str) -> None:
        if reason not in ("missing", "model", "prompt_version"):
            raise ValueError("identity refusal reason must name missing, model or prompt_version")
        self.reason = reason
        super().__init__(request_id, ask)
        self.args = (f"{self.why}: {reason} in model or prompt_version for request_id and ask",)


class AmbiguousExchange(ReplayRefusal):
    """Matching rows disagree on reply, error or the original reply's digest."""

    why = "judge-replay-ambiguous"


def identity_problem(
    model: str | None,
    prompt_version: str | None,
    *,
    expected_model: str,
    expected_prompt_version: str,
) -> str | None:
    """The first identity mismatch: missing, model, prompt_version, or None.

    The requested identity must be present even if no recorded row exists.
    """
    if type(expected_model) is not str or not expected_model:
        raise ValueError("expected_model must be a non-empty str")
    if type(expected_prompt_version) is not str or not expected_prompt_version:
        raise ValueError("expected_prompt_version must be a non-empty str")
    if not model or not prompt_version:
        return "missing"
    if model != expected_model:
        return "model"
    if prompt_version != expected_prompt_version:
        return "prompt_version"
    return None


def replay(
    exchanges: Iterable[Exchange],
    request_id: str,
    ask: int,
    *,
    model: str,
    prompt_version: str,
) -> Exchange:
    """Return the first matching row if reply, error and response digest agree; otherwise refuse.

    Candidate order is file order. Other models' rows do not participate in the
    decision, including when a deliberate re-judge appended their rows later.
    The response digest distinguishes different replies hidden by the same
    redaction marker. Failed calls have no response digest or separate error digest.
    """
    identity_problem(model, prompt_version, expected_model=model, expected_prompt_version=prompt_version)
    found = False
    first_problem: str | None = None
    matching: list[Exchange] = []
    for row in exchanges:
        if (row.request_id, row.ask) != (request_id, ask):
            continue
        problem = identity_problem(
            row.model, row.prompt_version, expected_model=model, expected_prompt_version=prompt_version
        )
        if not found:
            found = True
            first_problem = problem
        if problem is None:
            matching.append(row)
    if not found:
        raise UnrecordedExchange(request_id, ask)
    if not matching:
        raise IdentityRefused(request_id, ask, first_problem or "missing")
    first = matching[0]
    if any(
        (row.reply, row.error, row.response_sha256) != (first.reply, first.error, first.response_sha256)
        for row in matching[1:]
    ):
        raise AmbiguousExchange(request_id, ask)
    return first
