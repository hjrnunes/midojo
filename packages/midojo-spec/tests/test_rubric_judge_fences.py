"""The judge reads a verdict wrapped in one markdown code block, as Garak's judge does.

``vectors/rubric/fences.json`` holds canned judge replies (bare, in one code block, malformed,
plain prose), the record Garak's ``RubricJudge._judge_output`` gives for each, and what
Garak's ``_strip_code_fence`` returns for a set of texts (``generate_fences.py``, run in
Garak's interpreter at 3e601af3).
"""

from __future__ import annotations

import itertools
import json
import random
import re
import time
from pathlib import Path
from typing import Any

import pytest

from midojo_spec import rubric_judge as port

FENCES = json.loads((Path(__file__).parents[1] / "vectors" / "rubric" / "fences.json").read_text(encoding="utf-8"))
VECTORS = FENCES["vectors"]
IDENTITY = ("model", "prompt")
SUPPORTED = {
    "unfenced",
    "fenced-json",
    "fenced-bare",
    "fenced-one-line",
    "fenced-padded",
    "fenced-crlf",
}
FENCED_OTHER = {"fenced-contradicted-tool", "fenced-unresolved", "fenced-unsupported-ref"}
NOT_JSON = {
    "prose-before-fence",
    "prose-after-fence",
    "unclosed-fence",
    "two-fences",
    "fenced-prose",
    "fenced-empty",
    "plain-prose",
}
# Garak's pattern, copied from ``garak/detectors/rubric_judge.py``: the oracle the port is held to.
GARAK_PATTERN = re.compile(r"\A\s*```[A-Za-z0-9_+-]*[ \t]*\n?(.*?)\s*```\s*\Z", re.DOTALL)


def label(vector: dict[str, Any]) -> str:
    return vector["id"].rsplit(".", 1)[1]


def having(labels: set[str]) -> list[Any]:
    return [pytest.param(vector, id=vector["id"]) for vector in VECTORS if label(vector) in labels]


class Canned:
    """Stands in for the judge call: records the messages and answers once."""

    def __init__(self, canned: dict[str, Any]) -> None:
        self.canned = canned
        self.messages: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> Any:
        self.messages.append(messages)
        if "raise" in self.canned:
            raise port.JudgeCallError(self.canned["raise"])
        return self.canned["content"]


def judged(vector: dict[str, Any], content: Any = None) -> dict[str, Any]:
    canned = vector["canned"] if content is None else {"content": content}
    return port.judge_output(vector["rubric"], port.build_evidence(**vector["midojo"]), Canned(canned))


def test_the_fixture_names_the_garak_revision_and_what_it_holds() -> None:
    assert FENCES["implementation"]["revision"] == "3e601af3c58985da72bd3049931a30572e8fa3ae"
    assert FENCES["implementation"]["detector"] == "RubricJudge._judge_output"
    assert len(VECTORS) == 34
    assert len(FENCES["strips"]) == 34
    assert {label(vector) for vector in VECTORS} == SUPPORTED | FENCED_OTHER | NOT_JSON | {"no-content", "api-error"}


@pytest.mark.parametrize("vector", having(SUPPORTED | FENCED_OTHER | NOT_JSON | {"no-content", "api-error"}))
def test_the_record_matches_garaks_but_for_the_identity_garak_adds(vector: dict[str, Any]) -> None:
    call = Canned(vector["canned"])

    record = port.judge_output(vector["rubric"], port.build_evidence(**vector["midojo"]), call)

    assert record == {key: value for key, value in vector["record"].items() if key not in IDENTITY}
    assert call.messages == [vector["http_body"]["messages"]]


@pytest.mark.parametrize("vector", having(SUPPORTED))
def test_a_bare_or_fenced_verdict_is_supported(vector: dict[str, Any]) -> None:
    record = judged(vector)

    assert (record["verdict"], record["evidence_refs"], record["reason"]) == (
        "supported",
        ["messages[0].content"],
        "judge_completed",
    )
    assert record["raw_response"] == vector["canned"]["content"]


@pytest.mark.parametrize("vector", having(SUPPORTED | FENCED_OTHER))
def test_a_fenced_reply_gives_the_record_its_body_gives(vector: dict[str, Any]) -> None:
    content = vector["canned"]["content"]
    body = port.strip_code_fence(content)

    assert judged(vector) == {**judged(vector, body), "raw_response": content}
    assert judged(vector)["reason"] != "judge_parse_failed"


def test_a_fenced_verdict_that_cites_a_recorded_call_is_read_with_its_evidence() -> None:
    (vector,) = (v for v in VECTORS if v["id"] == "judge.one-json-call.fenced-contradicted-tool")

    record = judged(vector)

    assert (record["verdict"], record["evidence_refs"], record["reason"]) == (
        "contradicted",
        ["tool_calls[0].result.referral"],
        "judge_completed",
    )


@pytest.mark.parametrize("vector", having(FENCED_OTHER))
def test_a_fenced_verdict_is_assessed_like_a_bare_one(vector: dict[str, Any]) -> None:
    body = port.strip_code_fence(vector["canned"]["content"])
    evidence = port.build_evidence(**vector["midojo"])

    assert port.assess_response(json.loads(body), evidence) == (
        vector["record"]["verdict"],
        vector["record"]["evidence_refs"],
        vector["record"]["reason"],
    )


@pytest.mark.parametrize("vector", having(NOT_JSON))
def test_prose_around_a_fence_an_unclosed_fence_and_plain_prose_stay_unparsed(vector: dict[str, Any]) -> None:
    record = judged(vector)

    assert (record["verdict"], record["evidence_refs"], record["reason"]) == ("unresolved", [], "judge_parse_failed")
    assert record["raw_response"] == vector["canned"]["content"]
    assert record["request"] is not None


@pytest.mark.parametrize("strip", [pytest.param(item, id=repr(item["content"])[:40]) for item in FENCES["strips"]])
def test_strip_code_fence_matches_garaks(strip: dict[str, Any]) -> None:
    assert port.strip_code_fence(strip["content"]) == strip["stripped"]


def garaks(text: str) -> str:
    match = GARAK_PATTERN.match(text)
    return match.group(1) if match else text


def test_strip_code_fence_matches_garaks_pattern_on_every_short_text() -> None:
    for length in range(8):
        for symbols in itertools.product("`a -\n\t{", repeat=length):
            text = "".join(symbols)
            assert port.strip_code_fence(text) == garaks(text), repr(text)


# the parts of a reply, in order: whitespace, opening fence, tag, whitespace, body, whitespace,
# closing fence, whitespace, trailing text
PARTS = (
    ["", " ", "\n", "\t", "\r\n", "\u00a0", "\u2003", "\x1c", "  \n "],
    ["```", "``", "````", "~~~"],
    ["", "json", "JSON", "c++", "a-b_c9", "\u00e9", "j s", "a`b"],
    ["", " ", "\n", "\t", "\r\n", "\u00a0"],
    ["", "x", "{}", " x ", "\n{}\n", "`", "``", "```", "a```b", "x\n```\ny", "\u00e9", "  ", "\n"],
    ["", " ", "\n", "\u2003"],
    ["```", "``", "````", ""],
    ["", " ", "\n", "\x1c"],
    ["", "x", "```"],
)


def test_strip_code_fence_matches_garaks_pattern_on_assembled_replies() -> None:
    chooser = random.Random(11)
    texts = ["".join(chooser.choice(part) for part in PARTS) for _ in range(30_000)]

    assert all(port.strip_code_fence(text) == garaks(text) for text in texts)
    # the draw holds many replies the pattern reads as one block, so the check is not vacuous
    assert sum(GARAK_PATTERN.match(text) is not None for text in texts) > 3_000


@pytest.mark.parametrize(
    "text",
    [
        "```" + " " * 200_000,
        "```json\n" + "\n" * 200_000,
        "```\nx" + " " * 200_000,
        "```\nx" + " " * 200_000 + "```",
        " " * 200_000 + "x",
        "```" + "a" * 200_000,
        "```" + "a" * 200_000 + "```",
    ],
    ids=[
        "open-spaces",
        "open-newlines",
        "open-body-spaces",
        "closed-body-spaces",
        "leading-spaces",
        "open-tag-run",
        "closed-tag-run",
    ],
)
def test_strip_code_fence_takes_linear_time(text: str) -> None:
    started = time.perf_counter()

    port.strip_code_fence(text)

    # Garak's pattern takes seconds on 2,000 spaces after an opening fence
    assert time.perf_counter() - started < 1.0


def test_a_long_run_of_spaces_after_an_opening_fence_is_a_parse_failure_not_a_hang() -> None:
    record = port.judge_output(
        {"question": "Q?"}, port.build_evidence("Q", [], "A", []), lambda _: "```" + " " * 50_000
    )

    assert (record["verdict"], record["reason"]) == ("unresolved", "judge_parse_failed")


def test_the_port_has_one_fence_reader() -> None:
    source = Path(port.__file__).parent
    readers = [path.name for path in source.glob("*.py") if "def strip_code_fence" in path.read_text(encoding="utf-8")]

    assert readers == ["rubric_judge.py"]
