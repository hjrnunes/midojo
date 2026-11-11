"""The recorder contract's logic, on the standard library alone: resend, report a lost record, detail fallback.

A fake recorder stands in for the control plane's client; no test here needs MiDojo or the SDK.
The root suite runs the same contract through the real clients.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pytest

from midojo_spec import recorder
from midojo_spec.recorder import (
    CAPTURE_REASON_MAX_LENGTH,
    RECORD_ATTEMPTS,
    ControlPlaneRejectedError,
    InvalidRequestError,
    RejectionKind,
    ReportingRecorder,
    RetryingRecorder,
    finish_detailed,
    records_details,
    reporting,
    retry,
)

CALL = {"call_id": "c1", "function": "get_weather", "args": {"city": "Oslo"}, "disposition": "forwarded"}
FINISH = {**CALL, "result": "sunny"}
DETAILS = {"structured_content": None, "injection_outcomes": []}


class FlakyError(Exception):
    pass


class Fake:
    """Records what it is sent; each call raises the next of ``failures`` before it succeeds."""

    def __init__(
        self, failures: list[BaseException] | None = None, *, detailed: bool = False, reports: bool = True
    ) -> None:
        self.failures = list(failures or [])
        self.sent: list[tuple[str, dict[str, Any]]] = []
        self.reported: list[dict[str, Any]] = []
        if detailed:
            self.finish_call_detailed = self._finish_call_detailed
        if reports:
            self.report_capture_failure = self._report_capture_failure

    def _next(self) -> None:
        if self.failures:
            raise self.failures.pop(0)

    async def start_call(self, **kwargs: Any) -> None:
        self.sent.append(("start_call", kwargs))
        self._next()

    async def finish_call(self, **kwargs: Any) -> None:
        self.sent.append(("finish_call", kwargs))
        self._next()

    async def _finish_call_detailed(self, **kwargs: Any) -> None:
        self.sent.append(("finish_call_detailed", kwargs))
        self._next()

    async def _report_capture_failure(self, **kwargs: Any) -> None:
        self.reported.append(kwargs)
        self._next()


@pytest.fixture
def waits(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    seen: list[float] = []

    async def sleep(seconds: float) -> None:
        seen.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    return seen


def transient(error: Exception) -> bool:
    return isinstance(error, FlakyError)


def test_a_transient_error_is_resent_with_a_doubling_wait(waits: list[float]) -> None:
    calls = 0

    async def send() -> str:
        nonlocal calls
        calls += 1
        if calls < 4:
            raise FlakyError
        return "done"

    assert asyncio.run(retry(send, retryable=transient, attempts=4, backoff_seconds=0.5)) == "done"
    assert calls == 4
    assert waits == [0.5, 1.0, 2.0]


def test_the_last_attempts_error_is_raised(waits: list[float]) -> None:
    async def send() -> None:
        raise FlakyError("still down")

    with pytest.raises(FlakyError, match="still down"):
        asyncio.run(retry(send, retryable=transient, attempts=2, backoff_seconds=1.0))
    assert waits == [1.0]


def test_an_error_that_is_not_transient_is_raised_at_once(waits: list[float]) -> None:
    calls = 0

    async def send() -> None:
        nonlocal calls
        calls += 1
        raise ValueError("bad")

    with pytest.raises(ValueError):
        asyncio.run(retry(send, retryable=transient))
    assert (calls, waits) == (1, [])


@pytest.mark.parametrize(
    "final",
    [ControlPlaneRejectedError(RejectionKind.SEALED, "closed"), InvalidRequestError("a field is missing")],
    ids=["a refusal", "an invalid request"],
)
def test_a_refusal_is_never_resent_whatever_the_caller_says(final: Exception, waits: list[float]) -> None:
    calls = 0

    async def send() -> None:
        nonlocal calls
        calls += 1
        raise final

    with pytest.raises(type(final)):
        asyncio.run(retry(send, retryable=lambda _error: True))
    assert (calls, waits) == (1, [])


def test_the_defaults_are_three_attempts_and_a_tenth_of_a_second(waits: list[float]) -> None:
    async def send() -> None:
        raise FlakyError

    with pytest.raises(FlakyError):
        asyncio.run(retry(send, retryable=transient))
    assert RECORD_ATTEMPTS == 3
    assert waits == [recorder.RECORD_BACKOFF_SECONDS, recorder.RECORD_BACKOFF_SECONDS * 2]


def test_a_resent_record_keeps_its_call_id(waits: list[float]) -> None:
    fake = Fake([FlakyError(), FlakyError()])

    asyncio.run(RetryingRecorder(fake, retryable=transient).start_call(**CALL))

    assert [name for name, _ in fake.sent] == ["start_call"] * 3
    assert {sent["call_id"] for _, sent in fake.sent} == {"c1"}


def test_a_finish_with_details_falls_back_to_the_text_when_the_recorder_records_none(waits: list[float]) -> None:
    fake = Fake()

    asyncio.run(RetryingRecorder(fake, retryable=transient).finish_call_detailed(**FINISH, **DETAILS))

    assert [name for name, _ in fake.sent] == ["finish_call"]
    assert set(fake.sent[0][1]) == {"call_id", "function", "args", "result", "disposition", "error"}


def test_a_finish_with_details_reaches_a_recorder_that_records_them(waits: list[float]) -> None:
    fake = Fake(detailed=True)

    asyncio.run(RetryingRecorder(fake, retryable=transient).finish_call_detailed(**FINISH, **DETAILS))

    assert [name for name, _ in fake.sent] == ["finish_call_detailed"]
    assert fake.sent[0][1]["injection_outcomes"] == []


def test_a_wrapper_says_what_the_recorder_it_wraps_records() -> None:
    assert records_details(Fake(detailed=True)) is True
    assert records_details(Fake()) is False
    assert RetryingRecorder(Fake(detailed=True), retryable=transient).records_details is True
    assert ReportingRecorder(Fake()).records_details is False
    assert records_details(RetryingRecorder(Fake(), retryable=transient)) is False


def test_finish_detailed_says_whether_the_details_were_recorded() -> None:
    assert asyncio.run(finish_detailed(Fake(detailed=True), **FINISH, **DETAILS)) is True
    assert asyncio.run(finish_detailed(Fake(), **FINISH, **DETAILS)) is False


def test_a_lost_record_is_reported_and_its_own_error_is_raised() -> None:
    lost = FlakyError("down")
    fake = Fake([lost])

    with pytest.raises(FlakyError):
        asyncio.run(ReportingRecorder(fake).start_call(**CALL))

    (report,) = fake.reported
    assert (report["call_id"], report["function"], report["phase"]) == ("c1", "get_weather", "start")
    assert report["reason"] == "FlakyError: down"


def test_a_finish_that_is_lost_is_reported_as_a_finish() -> None:
    fake = Fake([FlakyError()])

    with pytest.raises(FlakyError):
        asyncio.run(ReportingRecorder(fake).finish_call(**FINISH))

    assert fake.reported[0]["phase"] == "finish"


def test_a_refusal_kept_as_a_late_fact_is_not_reported_and_another_refusal_is() -> None:
    late = Fake([ControlPlaneRejectedError(RejectionKind.SEALED, "closed", late_fact=True)])
    other = Fake([ControlPlaneRejectedError(RejectionKind.CALL_CONFLICT, "twice")])

    with pytest.raises(ControlPlaneRejectedError):
        asyncio.run(ReportingRecorder(late).start_call(**CALL))
    with pytest.raises(ControlPlaneRejectedError):
        asyncio.run(ReportingRecorder(other).start_call(**CALL))

    assert late.reported == []
    assert len(other.reported) == 1


def test_a_cancelled_record_is_not_reported() -> None:
    fake = Fake([asyncio.CancelledError()])

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(ReportingRecorder(fake).start_call(**CALL))

    assert fake.reported == []


def test_a_report_that_fails_too_is_logged_and_never_replaces_the_records_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    fake = Fake([FlakyError("first")])
    fake.failures.append(FlakyError("second"))  # the report's own failure

    with caplog.at_level(logging.WARNING, logger="midojo_spec.recorder"), pytest.raises(FlakyError, match="first"):
        asyncio.run(ReportingRecorder(fake).start_call(**CALL))

    assert any("wasn't delivered" in message for message in caplog.messages)


def test_a_recorder_that_cannot_report_still_raises_the_records_error(caplog: pytest.LogCaptureFixture) -> None:
    fake = Fake([FlakyError("lost")], reports=False)

    with caplog.at_level(logging.WARNING, logger="midojo_spec.recorder"), pytest.raises(FlakyError, match="lost"):
        asyncio.run(ReportingRecorder(fake).start_call(**CALL))

    assert any("wasn't delivered" in message for message in caplog.messages)


def test_a_long_reason_is_cut_to_what_the_control_plane_accepts() -> None:
    fake = Fake([FlakyError("x" * 5000)])

    with pytest.raises(FlakyError):
        asyncio.run(ReportingRecorder(fake).start_call(**CALL))

    assert len(fake.reported[0]["reason"]) == CAPTURE_REASON_MAX_LENGTH == 2048


def test_reporting_does_not_wrap_a_reporting_recorder_twice() -> None:
    wrapped = ReportingRecorder(Fake())

    assert reporting(wrapped) is wrapped
    assert isinstance(reporting(Fake()), ReportingRecorder)


def test_the_errors_say_their_kind_and_detail() -> None:
    refused = ControlPlaneRejectedError(RejectionKind.SEALED, "closed", late_fact=True)
    invalid = InvalidRequestError("a field is missing")

    assert str(refused) == "sealed: closed"
    assert (refused.kind, refused.detail, refused.late_fact) == (RejectionKind.SEALED, "closed", True)
    assert ControlPlaneRejectedError(RejectionKind.SEALED, "d").late_fact is False
    assert str(invalid) == "invalid request: a field is missing"
    assert isinstance(invalid, ValueError)
    assert recorder.FINAL_ERRORS == (ControlPlaneRejectedError, InvalidRequestError)


def test_the_refused_dispositions_are_the_two_that_never_ran_the_tool() -> None:
    assert recorder.REFUSED == frozenset({"denied", "blocked"})
