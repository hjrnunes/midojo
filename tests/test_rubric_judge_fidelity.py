"""The judge's evidence holds the tool results Garak's evidence would hold.

MiDojo's control plane stores the string the orchestrator's sink makes of each MCP tool
result (``call_body``: the text blocks joined by a newline, the error text for a failed
call). Garak's evidence holds ``normalise_tool_call(call)["result"]``, the JSON the model's
captured ``output`` decodes to. ``fixtures/rubric/fidelity.json`` pairs the two, with the
sink and Garak functions run on each case by the consumer's ``generate_fidelity.py``.

Where the sink string and Garak's evidence can differ, from the cases below:

- A failed call (``isError``): the sink string is the error text, and Garak captures no
  ``output``, so its result is null. When the error text is itself JSON, decoding the
  sink string would yield a value Garak never shows; the port reads a call with an error
  as a null result.
- A result of several text blocks, a non-text block, no text block, or a non-mapping value:
  the sink joins text blocks with a newline and dumps other blocks as JSON, and the saved
  runs hold no Garak capture of such a result, so its ``output`` is unknown. The port
  decodes the sink string as is; the cases below pin what that yields.
- Text that is not JSON: both sides give null, so the judge cannot cite the result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from midojo_sdk import rubric_judge as port

FIDELITY = json.loads((Path(__file__).parent / "fixtures/rubric/fidelity.json").read_text(encoding="utf-8"))


def ids(group: str) -> list[Any]:
    return [pytest.param(case, id=case["id"]) for case in FIDELITY[group]]


def evidence_value(case: dict[str, Any]) -> dict[str, Any]:
    """The tool call the port puts in the evidence for the call the sink recorded."""

    return port.tool_call_of(case["sink"])


def test_the_fixture_names_the_sink_rule_and_the_garak_function() -> None:
    assert FIDELITY["sink"]["function"] == "call_body"
    assert FIDELITY["garak_function"].endswith("normalise_tool_call")
    assert len(FIDELITY["real"]) == 75
    assert len(FIDELITY["synthetic"]) == 24


@pytest.mark.parametrize("case", ids("real"))
def test_a_saved_exchange_gives_garaks_evidence_value(case: dict[str, Any]) -> None:
    value = evidence_value(case)

    assert value["result"] == case["garak"]["result"]
    assert value["name"] == case["garak"]["name"]


@pytest.mark.parametrize("case", ids("real"))
def test_a_saved_exchange_gives_garaks_arguments(case: dict[str, Any]) -> None:
    assert evidence_value(case)["arguments"] == case["garak"]["arguments"]


def test_the_saved_exchanges_include_a_failed_call_and_a_json_result() -> None:
    errors = [case for case in FIDELITY["real"] if case["sink"]["error"]]
    decoded = [case for case in FIDELITY["real"] if isinstance(case["garak"]["result"], dict)]

    assert errors
    assert decoded


MODELLED = [case for case in FIDELITY["synthetic"] if case["garak"] is not None]


@pytest.mark.parametrize("case", [pytest.param(c, id=c["id"]) for c in MODELLED])
def test_a_modelled_capture_gives_garaks_evidence_value(case: dict[str, Any]) -> None:
    assert evidence_value(case)["result"] == case["garak"]["result"]
    assert evidence_value(case)["arguments"] == case["garak"]["arguments"]


def test_a_failed_call_whose_text_is_json_is_the_case_the_port_corrects() -> None:
    by_id = {case["id"]: case for case in FIDELITY["synthetic"]}

    for name in ("synthetic-error-json-object", "synthetic-error-json-scalar"):
        case = by_id[name]
        assert json.loads(case["sink"]["result"]) is not None
        assert case["garak"]["result"] is None
        assert evidence_value(case)["result"] is None


# What the port yields where no Garak capture is modelled.
UNMODELLED = {
    "synthetic-two-text-blocks-json": None,
    "synthetic-two-text-blocks-split-json": {"a": 1},
    "synthetic-image-block": {"data": "AAAA", "mimeType": "image/png", "type": "image"},
    "synthetic-text-and-image": None,
    "synthetic-no-content": None,
    "synthetic-structured-only": None,
    "synthetic-not-a-mapping": "raw text result",
    "synthetic-null-result": None,
}


def test_the_unmodelled_cases_are_exactly_the_ones_pinned_here() -> None:
    unmodelled = {case["id"] for case in FIDELITY["synthetic"] if case["garak"] is None}

    assert unmodelled == set(UNMODELLED)


@pytest.mark.parametrize("case", [pytest.param(c, id=c["id"]) for c in FIDELITY["synthetic"] if c["garak"] is None])
def test_an_unmodelled_result_decodes_as_the_sink_string_does(case: dict[str, Any]) -> None:
    assert evidence_value(case)["result"] == UNMODELLED[case["id"]]
