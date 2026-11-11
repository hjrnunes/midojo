"""The judge exchange record: rows, redaction, the append-only file and its strict reader."""

from __future__ import annotations

import copy
import dataclasses
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


REDACTION_REFUSED = VECTORS["redaction_refused"]


@pytest.mark.parametrize("case", REDACTION_REFUSED, ids=[case["name"] for case in REDACTION_REFUSED])
def test_a_redaction_that_would_lose_the_identity_or_spell_a_secret_is_refused(case: dict[str, Any]) -> None:
    with pytest.raises(ExchangeFormatError) as refused:
        Exchange.observed(Observed(**case["observed"]), secrets=case["secrets"], **case["identity"])

    assert str(refused.value) == case["problem"]
    for secret in case["secrets"]:
        assert secret in record.REDACTED or secret not in str(refused.value)


def test_a_secret_in_the_judge_identity_is_refused_and_never_rewritten() -> None:
    for fields in ({"model": "m-tok"}, {"prompt_version": "v-tok"}):
        with pytest.raises(ExchangeFormatError, match="identify the judge"):
            observed_exchange(secrets=["tok"], **fields)

    redacted = observed_exchange(secrets=["u"])

    assert (redacted.model, redacted.prompt_version, redacted.redacted) == ("m", "v", True)


def test_redaction_scrubs_every_text_and_leaves_no_listed_secret() -> None:
    secrets = ["sk-live-1", "live-12", "e-123x", "abc"]
    text = "sk-live-123x abc live-12 xabcx sk-live-1sk-live-1"
    exchange = observed_exchange(
        observed=Observed(
            started_at="2026-11-11T09:30:00.123+00:00",
            duration_ms=1,
            content=text,
            finish_reason="stop abc",
            response_model="x abc",
            usage={"abc": ["sk-live-1", {"deep": "abc live-12"}]},
        ),
        messages=[{"role": "system", "content": text}, {"role": "user", "content": text}],
    )
    scrubbed_text = "[redacted] [redacted] [redacted] x[redacted]x [redacted]"

    scrubbed = record.redact(exchange, secrets)

    assert scrubbed.messages == (("system", scrubbed_text), ("user", scrubbed_text))
    assert scrubbed.reply == scrubbed_text
    assert scrubbed.finish_reason == "stop [redacted]"
    assert scrubbed.response_model == "x [redacted]"
    assert scrubbed.usage == {"[redacted]": ["[redacted]", {"deep": "[redacted] [redacted]"}]}
    assert scrubbed.redacted is True
    assert scrubbed.request_sha256 == exchange.request_sha256
    assert scrubbed.response_sha256 == exchange.response_sha256
    with pytest.raises(ExchangeFormatError):
        record.redact(exchange, [*secrets, "[red"])


def test_overlapping_and_touching_secrets_become_one_marker_each_run() -> None:
    exchange = observed_exchange(
        observed=Observed(started_at="2026-11-11T09:30:00.123+00:00", duration_ms=1, content="..xabz..cd.ab.ab")
    )

    assert record.redact(exchange, ["ab", "xa", "cd"]).reply == "..[redacted]z..[redacted].[redacted].[redacted]"
    assert record.redact(exchange, ["abz.", "xab"]).reply == "..[redacted].cd.ab.ab"


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
    # The two digests are `printf u | shasum -a 256` and `printf reply | shasum -a 256`; the
    # first message, "s", hashes to 043a7187..., which must not be the request digest.
    exchange = observed_exchange()

    assert exchange.request_sha256 == "0bfe935e70c321c7ca3afc75ce0d0ca2f98b5422e008bb31c00c6d7f1f1c0ad6"
    assert exchange.response_sha256 == "5782b18687e6cf8a482fc32d2db5b196d8821c458a0c069c6acf3953446e7bb5"
    assert not exchange.request_sha256.startswith("043a7187")


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


NOT_JSON_USAGE = [
    {"a": float("nan")},
    {"a": float("inf")},
    {"a": -float("inf")},
    {"a": {1: 2}},
    {"a": object()},
    {"a": (1, 2)},
    {"a": {1, 2}},
    {"a": b"x"},
    {"a": [{"b": float("nan")}]},
]


@pytest.mark.parametrize("usage", NOT_JSON_USAGE, ids=[repr(usage) for usage in NOT_JSON_USAGE])
def test_a_usage_that_is_not_plain_json_is_dropped_from_an_observation_and_refused_in_a_row(
    usage: dict[str, Any], tmp_path: Path
) -> None:
    observation = Observed(started_at="2026-11-11T09:30:00.123+00:00", duration_ms=1, content="x", usage=usage)
    exchange = observed_exchange(observed=observation)
    forced = dataclasses.replace(exchange, usage=usage)
    log = ExchangeLog(tmp_path / "log.jsonl")

    assert exchange.usage is None
    assert record.exchange_problem({**forced.to_json(), "sequence": 0}) == "usage is not valid"
    with pytest.raises(ExchangeFormatError, match="usage is not valid"):
        log.append(forced)
    assert not log.path.exists()


def test_usage_in_a_row_that_is_plain_json_is_kept() -> None:
    usage = {"total_tokens": 3, "ratio": 0.5, "ok": True, "none": None, "list": [1, {"a": "b"}], "text": "t"}

    assert (
        observed_exchange(
            observed=Observed(started_at="2026-11-11T09:30:00.123+00:00", duration_ms=1, content="x", usage=usage)
        ).usage
        == usage
    )


def test_a_row_whose_digests_are_of_other_text_is_refused_by_the_append_and_the_reader(tmp_path: Path) -> None:
    exchange = observed_exchange()
    tampered = dataclasses.replace(exchange, response_sha256=record.sha256_text("something else"))
    log = ExchangeLog(tmp_path / "log.jsonl")

    with pytest.raises(ExchangeFormatError, match="response_sha256 is not the digest of the reply"):
        log.append(tampered)
    assert not log.path.exists()
    path = write_lines(tmp_path / "read.jsonl", json.dumps({**tampered.to_json(), "sequence": 0}))
    with pytest.raises(ExchangeFormatError, match=r"line 1: response_sha256 is not the digest of the reply"):
        record.read_exchanges(path)


def test_two_logs_on_one_file_number_the_rows_as_one(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    first, second = ExchangeLog(path), ExchangeLog(path)

    written = [log.append(observed_exchange()) for log in (first, second, first, second)]

    assert [exchange.sequence for exchange in written] == [0, 1, 2, 3]
    assert [exchange.sequence for exchange in record.read_exchanges(path)] == [0, 1, 2, 3]


def test_logs_named_by_different_paths_to_one_file_share_it(tmp_path: Path) -> None:
    (tmp_path / "dir").mkdir()
    path = tmp_path / "dir" / "log.jsonl"
    link = tmp_path / "link.jsonl"
    link.symlink_to(path)
    first, second = ExchangeLog(path), ExchangeLog(tmp_path / "dir" / ".." / "dir" / "log.jsonl")
    third = ExchangeLog(link)

    for log in (first, second, third, first):
        log.append(observed_exchange())

    assert [exchange.sequence for exchange in record.read_exchanges(path)] == [0, 1, 2, 3]


def test_many_logs_on_one_file_from_many_threads_keep_the_file_readable(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"

    def work(worker: int) -> None:
        log = ExchangeLog(path)
        for ask in range(20):
            log.append(observed_exchange(ask=ask, messages=[{"role": "user", "content": f"{worker}:{ask}"}]))

    threads = [threading.Thread(target=work, args=(worker,)) for worker in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [exchange.sequence for exchange in record.read_exchanges(path)] == list(range(160))


def test_a_row_another_writer_added_is_counted_before_the_next_one(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    log = ExchangeLog(path)
    log.append(observed_exchange())
    with path.open("a", encoding="ascii") as other:
        other.write(valid_line(1) + "\n")

    assert log.append(observed_exchange()).sequence == 2
    assert [exchange.sequence for exchange in record.read_exchanges(path)] == [0, 1, 2]


class Stop(BaseException):
    """What a signal handler or a cancelled task raises: not an ``Exception``."""


def failing_row_write(monkeypatch: pytest.MonkeyPatch, fails: BaseException) -> None:
    """Make the write of a row put half of it on disk and then fail with ``fails``."""
    real = os.write

    def write(descriptor: int, data: Any) -> int:
        if bytes(data).startswith(b'{"schema"'):
            real(descriptor, bytes(data)[: len(data) // 2])
            raise fails
        return real(descriptor, data)

    monkeypatch.setattr(os, "write", write)


@pytest.mark.parametrize("fails", [OSError("disk full"), Stop()], ids=["an error", "a stop"])
def test_a_write_that_fails_leaves_no_partial_row_and_the_next_append_is_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: BaseException
) -> None:
    log = ExchangeLog(tmp_path / "log.jsonl")
    log.append(observed_exchange())
    whole = log.path.read_bytes()
    with monkeypatch.context() as patched:
        failing_row_write(patched, fails)
        with pytest.raises(type(fails)):
            log.append(observed_exchange())

    assert log.path.read_bytes() == whole
    assert log.append(observed_exchange()).sequence == 1
    assert [exchange.sequence for exchange in record.read_exchanges(log.path)] == [0, 1]


def test_a_write_that_cannot_be_cut_back_is_not_followed_by_a_row_glued_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log = ExchangeLog(tmp_path / "log.jsonl")
    log.append(observed_exchange())
    with monkeypatch.context() as patched:
        failing_row_write(patched, OSError("disk full"))

        def refuse(_descriptor: int, _length: int) -> None:
            raise OSError("read-only")

        patched.setattr(os, "ftruncate", refuse)
        with pytest.raises(OSError, match="disk full"):
            log.append(observed_exchange())
    torn = log.path.read_bytes()

    with pytest.raises(ExchangeFormatError, match="partial row"):
        log.append(observed_exchange())
    assert log.path.read_bytes() == torn


def test_a_file_that_ends_in_a_partial_row_is_not_appended_to(tmp_path: Path) -> None:
    path = tmp_path / "log.jsonl"
    path.write_text(valid_line(0) + "\n" + valid_line(1)[:40], encoding="utf-8")
    before = path.read_bytes()

    with pytest.raises(ExchangeFormatError, match="partial row"):
        ExchangeLog(path).append(observed_exchange())
    assert path.read_bytes() == before


def test_a_file_that_does_not_read_as_a_log_is_not_appended_to(tmp_path: Path) -> None:
    path = write_lines(tmp_path / "log.jsonl", valid_line(0), "not json")
    before = path.read_bytes()

    with pytest.raises(ExchangeFormatError, match="line 2"):
        ExchangeLog(path).append(observed_exchange())
    assert path.read_bytes() == before


def test_a_log_made_with_a_relative_path_stays_on_that_file_when_the_directory_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    here, there = tmp_path / "here", tmp_path / "there"
    here.mkdir()
    there.mkdir()
    monkeypatch.chdir(here)
    first = ExchangeLog("log.jsonl")
    assert first.path.is_absolute()
    first.append(observed_exchange())
    monkeypatch.chdir(there)
    second = ExchangeLog(here / "log.jsonl")

    first.append(observed_exchange())
    second.append(observed_exchange())

    assert [exchange.sequence for exchange in record.read_exchanges(here / "log.jsonl")] == [0, 1, 2]
    assert list(there.iterdir()) == []


def test_a_log_whose_symlink_is_pointed_elsewhere_numbers_each_file_by_its_own_rows(tmp_path: Path) -> None:
    def row(text: str) -> Exchange:
        return observed_exchange(messages=[{"role": "user", "content": text}])

    first, second, probe = tmp_path / "first.jsonl", tmp_path / "second.jsonl", tmp_path / "probe.jsonl"
    for _ in range(2):
        ExchangeLog(second).append(row("b"))
    ExchangeLog(probe).append(row("a"))
    # One long row in the first file, as many bytes as the two short rows of the second: the sizes
    # match, so a count kept by size alone would be taken for the second file's.
    padding = "a" * (1 + second.stat().st_size - probe.stat().st_size)
    link = tmp_path / "log.jsonl"
    link.symlink_to(first)
    through_link = ExchangeLog(link)
    through_link.append(row(padding))
    assert first.stat().st_size == second.stat().st_size
    link.unlink()
    link.symlink_to(second)

    appended = through_link.append(row("c"))

    assert appended.sequence == 2
    assert [exchange.sequence for exchange in record.read_exchanges(first)] == [0]
    assert [exchange.sequence for exchange in record.read_exchanges(second)] == [0, 1, 2]


def test_a_file_emptied_between_appends_is_numbered_from_zero(tmp_path: Path) -> None:
    log = ExchangeLog(tmp_path / "log.jsonl")
    log.append(observed_exchange())
    log.append(observed_exchange())
    log.path.write_bytes(b"")

    assert log.append(observed_exchange()).sequence == 0


def edit_in_place(path: Path, old: bytes, new: bytes) -> None:
    assert len(old) == len(new)
    path.write_bytes(path.read_bytes().replace(old, new, 1))


def test_an_edit_that_keeps_the_size_but_not_the_time_makes_the_next_append_read_the_file(tmp_path: Path) -> None:
    log = ExchangeLog(tmp_path / "log.jsonl")
    log.append(observed_exchange())
    before = log.path.stat()
    edit_in_place(log.path, b"midojo.judge-exchange.v1", b"midojo.judge-exchange.v2")
    os.utime(log.path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    edited = log.path.read_bytes()

    with pytest.raises(ExchangeFormatError, match=r'line 1: schema is not "midojo.judge-exchange.v1"'):
        log.append(observed_exchange())
    assert log.path.read_bytes() == edited


def test_an_edit_that_keeps_the_size_and_the_time_is_not_noticed_until_the_file_is_read(tmp_path: Path) -> None:
    log = ExchangeLog(tmp_path / "log.jsonl")
    log.append(observed_exchange())
    before = log.path.stat()
    edit_in_place(log.path, b"midojo.judge-exchange.v1", b"midojo.judge-exchange.v2")
    os.utime(log.path, ns=(before.st_atime_ns, before.st_mtime_ns))

    assert log.append(observed_exchange()).sequence == 1
    with pytest.raises(ExchangeFormatError, match="line 1"):
        record.read_exchanges(log.path)


def test_an_integer_the_file_cannot_hold_is_dropped_from_usage_and_refused_in_a_row(tmp_path: Path) -> None:
    for value in (10**5000, record.MAX_INTEGER + 1, record.MIN_INTEGER - 1):
        usage = {"a": value}
        exchange = observed_exchange(
            observed=Observed(started_at="2026-11-11T09:30:00.123+00:00", duration_ms=1, content="x", usage=usage)
        )
        forced = dataclasses.replace(exchange, usage=usage)
        log = ExchangeLog(tmp_path / f"{abs(value) % 1000}.jsonl")

        assert exchange.usage is None
        assert record.exchange_problem({**forced.to_json(), "sequence": 0}) == "usage is not valid"
        with pytest.raises(ExchangeFormatError, match="usage is not valid"):
            log.append(forced)
        assert not log.path.exists()


def test_integers_at_the_edge_of_sixty_four_bits_are_kept_and_read_back(tmp_path: Path) -> None:
    usage = {"high": record.MAX_INTEGER, "low": record.MIN_INTEGER}
    log = ExchangeLog(tmp_path / "log.jsonl")
    log.append(
        observed_exchange(
            observed=Observed(started_at="2026-11-11T09:30:00.123+00:00", duration_ms=1, content="x", usage=usage)
        )
    )

    assert record.read_exchanges(log.path)[0].usage == usage


@pytest.mark.parametrize("field", ["sequence", "ask", "duration_ms", "http_status"])
def test_an_index_the_file_cannot_hold_is_refused_whole(field: str, tmp_path: Path) -> None:
    row = {**copy.deepcopy(VALID[0]["row"]), field: 10**5000}

    assert record.exchange_problem(row) == f"{field} is not valid"
    with pytest.raises(ExchangeFormatError, match=f"{field} is not valid"):
        Exchange.from_json(row)
    path = tmp_path / "log.jsonl"
    path.write_text(json.dumps({**VALID[0]["row"], field: 2**63}) + "\n", encoding="utf-8")
    with pytest.raises(ExchangeFormatError, match=rf"line 1: {field} is not valid"):
        record.read_exchanges(path)


def test_a_log_whose_last_row_holds_the_largest_sequence_refuses_the_next_row_and_stays_readable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "log.jsonl"
    seed = dataclasses.replace(observed_exchange(), sequence=record.MAX_INTEGER - 1)
    path.write_text(json.dumps(seed.to_json(), ensure_ascii=True, separators=(",", ":")) + "\n")
    log = ExchangeLog(path)

    last = log.append(observed_exchange())
    full = path.read_bytes()

    assert last.sequence == record.MAX_INTEGER
    for _ in range(2):
        with pytest.raises(ExchangeFormatError, match="sequence is not valid"):
            log.append(observed_exchange())
    assert path.read_bytes() == full
    assert [row.sequence for row in record.read_exchanges(path)] == [record.MAX_INTEGER - 1, record.MAX_INTEGER]
