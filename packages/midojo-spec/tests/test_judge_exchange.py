"""The judge exchange record: rows, redaction, the append-only file and its strict reader."""

from __future__ import annotations

import copy
import json
import os
import stat
import threading
from pathlib import Path
from typing import Any

import pytest

from midojo_spec import judge_exchange as record
from midojo_spec.judge_exchange import Exchange, ExchangeFormatError, ExchangeLog, Observed

VECTORS = json.loads((Path(__file__).parents[1] / "vectors" / "judge" / "exchange.json").read_text(encoding="utf-8"))
VALID = VECTORS["valid"]
INVALID = VECTORS["invalid"]
REDACTION = VECTORS["redaction"]


def row(**changes: Any) -> Exchange:
    return Exchange.from_json({**copy.deepcopy(VALID[0]["row"]), **changes})


def observed_exchange(**changes: Any) -> Exchange:
    fields: dict[str, Any] = {
        "observed": Observed(started_at="2026-11-11T09:30:00.123+00:00", duration_ms=5, content="reply"),
        "request_id": "sha256:" + "cd" * 32,
        "ask": 0,
        "model": "m",
        "prompt_version": "v",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
    }
    fields.update(changes)
    return Exchange.observed(**fields)


@pytest.mark.parametrize("case", VALID, ids=[case["name"] for case in VALID])
def test_a_valid_row_round_trips(case: dict[str, Any]) -> None:
    assert record.exchange_problem(case["row"]) is None

    exchange = Exchange.from_json(case["row"])

    assert exchange.to_json() == case["row"]
    assert list(exchange.to_json()) == list(record.FIELDS)


@pytest.mark.parametrize("case", VALID, ids=[case["name"] for case in VALID])
def test_a_row_says_whether_its_digests_hold(case: dict[str, Any]) -> None:
    assert record.digests_hold(Exchange.from_json(case["row"])) is case["digests_hold"]


@pytest.mark.parametrize("case", INVALID, ids=[case["name"] for case in INVALID])
def test_an_invalid_row_has_its_problem(case: dict[str, Any]) -> None:
    assert record.exchange_problem(case["row"]) == case["problem"]
    with pytest.raises(ExchangeFormatError, match=case["problem"]):
        Exchange.from_json(case["row"])


@pytest.mark.parametrize("case", REDACTION, ids=[case["name"] for case in REDACTION])
def test_a_model_request_becomes_its_row(case: dict[str, Any]) -> None:
    exchange = Exchange.observed(Observed(**case["observed"]), secrets=case["secrets"], **case["identity"])

    assert exchange.to_json() == case["expected"]


def test_the_vector_digests_are_sha256_of_utf8() -> None:
    for case in VECTORS["digests"]:
        assert record.sha256_text(case["text"]) == case["sha256"]


def test_a_problem_never_holds_what_the_row_says() -> None:
    secret = "sk-live-never-echoed"
    bad = copy.deepcopy(VALID[0]["row"])
    bad.update(request_id=secret, ask=secret, messages=[secret], started_at=secret, reply=[secret])

    problem = record.exchange_problem(bad)

    assert problem is not None
    assert secret not in problem


def test_the_digests_come_from_the_last_message() -> None:
    exchange = observed_exchange()

    assert exchange.request_sha256 == record.sha256_text("u")
    assert exchange.response_sha256 == record.sha256_text("reply")


def test_a_model_request_with_no_messages_is_refused() -> None:
    with pytest.raises(ExchangeFormatError):
        observed_exchange(messages=[])


def test_a_redacted_row_cannot_vouch_for_its_digests_and_an_unredacted_one_can() -> None:
    assert record.digests_hold(observed_exchange()) is True
    assert record.digests_hold(observed_exchange(secrets=["reply"])) is None


def test_a_row_with_a_reply_and_an_error_is_refused_at_the_append(tmp_path: Path) -> None:
    exchange = observed_exchange(
        observed=Observed(started_at="2026-11-11T09:30:00+00:00", duration_ms=1, content="x", error="E")
    )

    with pytest.raises(ExchangeFormatError):
        ExchangeLog(tmp_path / "log.jsonl").append(exchange)
    assert not (tmp_path / "log.jsonl").exists()


def test_the_log_numbers_rows_from_zero_and_reads_them_back(tmp_path: Path) -> None:
    log = ExchangeLog(tmp_path / "nested" / "log.jsonl")

    written = [log.append(observed_exchange(ask=ask)) for ask in range(3)]

    assert [exchange.sequence for exchange in written] == [0, 1, 2]
    assert record.read_exchanges(log.path) == written
    assert len(log.path.read_text(encoding="ascii").splitlines()) == 3


def test_the_log_continues_the_sequence_of_a_file_it_finds(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    ExchangeLog(path).append(observed_exchange())
    ExchangeLog(path).append(observed_exchange())

    assert [exchange.sequence for exchange in record.read_exchanges(path)] == [0, 1]


def test_the_log_file_is_private(tmp_path: Path) -> None:
    log = ExchangeLog(tmp_path / "log.jsonl")
    log.append(observed_exchange())

    assert stat.S_IMODE(os.stat(log.path).st_mode) == 0o600


def test_a_lone_surrogate_in_a_reply_survives_the_file_and_its_digest_holds(tmp_path: Path) -> None:
    log = ExchangeLog(tmp_path / "log.jsonl")
    log.append(
        observed_exchange(observed=Observed(started_at="2026-11-11T09:30:00+00:00", duration_ms=1, content="a\ud800b"))
    )

    (read,) = record.read_exchanges(log.path)

    assert read.reply == "a\ud800b"
    assert record.digests_hold(read) is True
    assert log.path.read_bytes().isascii()


def test_non_ascii_text_is_written_escaped_and_read_back_whole(tmp_path: Path) -> None:
    log = ExchangeLog(tmp_path / "log.jsonl")
    log.append(observed_exchange(messages=[{"role": "user", "content": "caf\u00e9 \U0001f600"}]))

    assert b"\\u00e9" in log.path.read_bytes()
    assert record.read_exchanges(log.path)[0].messages == (("user", "caf\u00e9 \U0001f600"),)


def test_appends_from_many_threads_keep_every_row_whole_and_numbered(tmp_path: Path) -> None:
    log = ExchangeLog(tmp_path / "log.jsonl")
    big = "x" * 20000

    def work(worker: int) -> None:
        for ask in range(25):
            log.append(observed_exchange(ask=ask, messages=[{"role": "user", "content": f"{worker}:{ask}:{big}"}]))

    threads = [threading.Thread(target=work, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    rows = record.read_exchanges(log.path)

    assert [exchange.sequence for exchange in rows] == list(range(200))
    assert {exchange.messages[0][1].split(":")[0] for exchange in rows} == {str(worker) for worker in range(8)}
    assert all(record.digests_hold(exchange) for exchange in rows)


def test_rows_group_by_request_and_ask_in_file_order(tmp_path: Path) -> None:
    log = ExchangeLog(tmp_path / "log.jsonl")
    other = "sha256:" + "ef" * 32
    first = log.append(observed_exchange())
    log.append(observed_exchange(request_id=other))
    again = log.append(observed_exchange())

    groups = record.by_key(record.read_exchanges(log.path))

    assert groups[(first.request_id, 0)] == [first, again]
    assert len(groups[(other, 0)]) == 1


def write_lines(path: Path, *lines: str) -> Path:
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def valid_line(sequence: int) -> str:
    return json.dumps({**VALID[0]["row"], "sequence": sequence})


def test_the_reader_names_the_line_and_the_rule(tmp_path: Path) -> None:
    bad = json.dumps({**VALID[0]["row"], "ask": -1})
    path = write_lines(tmp_path / "log.jsonl", valid_line(0), bad)

    with pytest.raises(ExchangeFormatError, match=r"line 2: ask is not valid"):
        record.read_exchanges(path)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("", r"line 1: it is not JSON"),
        ("not json", r"line 1: it is not JSON"),
        ('{"schema": "x"', r"line 1: it is not JSON"),
        ('{"a": NaN}', r"line 1: it has NaN"),
        ('{"a": 1, "a": 2}', r"line 1: a JSON key repeats"),
        ("[]", r"line 1: the row is not an object"),
    ],
)
def test_the_reader_refuses_a_line_that_is_not_a_row(tmp_path: Path, text: str, problem: str) -> None:
    path = write_lines(tmp_path / "log.jsonl", text)

    with pytest.raises(ExchangeFormatError, match=problem):
        record.read_exchanges(path)


def test_the_reader_refuses_a_torn_last_line(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text(valid_line(0) + "\n" + valid_line(1)[:40], encoding="utf-8")

    with pytest.raises(ExchangeFormatError, match="line 2"):
        record.read_exchanges(path)


@pytest.mark.parametrize("second", [0, 1])
def test_the_reader_refuses_sequences_that_do_not_rise(tmp_path: Path, second: int) -> None:
    path = write_lines(tmp_path / "log.jsonl", valid_line(1), valid_line(second))

    with pytest.raises(ExchangeFormatError, match="line 2: sequence does not rise"):
        record.read_exchanges(path)


def test_the_reader_keeps_a_gap_in_the_sequence(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "log.jsonl", valid_line(3), valid_line(9))

    assert [exchange.sequence for exchange in record.read_exchanges(path)] == [3, 9]


def test_an_empty_file_has_no_rows(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_bytes(b"")

    assert record.read_exchanges(path) == []


def test_a_row_deeper_than_the_parser_allows_is_a_format_error(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "log.jsonl", "[" * 100000)

    with pytest.raises(ExchangeFormatError, match="line 1"):
        record.read_exchanges(path)
