"""``vectors/plan/plans.json``: the injection plan codec's documents and refusals.

The documents were written by hand, field by field, so they do not depend on the codec's own
output. ``accepted`` documents decode and encode to themselves; ``defaults`` documents leave out
what the codec fills in and decode to the full form beside them; ``refused`` documents raise
``ValueError`` with the text each names. A codec in another language runs the same file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from midojo_spec.injection import PLAN_CODEC_VERSION, InjectionPlan

DOC = json.loads((Path(__file__).parents[1] / "vectors" / "plan" / "plans.json").read_text(encoding="utf-8"))


def names(cases: list[dict[str, Any]]) -> list[str]:
    return [case["name"] for case in cases]


def test_the_codec_version_is_the_one_the_vectors_name() -> None:
    assert PLAN_CODEC_VERSION == DOC["codec_version"]


@pytest.mark.parametrize("case", DOC["accepted"], ids=names(DOC["accepted"]))
def test_an_accepted_document_encodes_to_itself(case: dict[str, Any]) -> None:
    plan = InjectionPlan.from_data(case["data"])

    assert plan.to_data() == case["data"]
    assert InjectionPlan.from_data(plan.to_data()) == plan


@pytest.mark.parametrize("case", DOC["accepted"], ids=names(DOC["accepted"]))
def test_an_accepted_document_has_one_json_text(case: dict[str, Any]) -> None:
    text = json.dumps(case["data"], sort_keys=True, separators=(",", ":"))

    assert InjectionPlan.from_json(text).to_json() == text
    assert InjectionPlan.from_json(json.dumps(case["data"], indent=4)).to_json() == text


@pytest.mark.parametrize("case", DOC["defaults"], ids=names(DOC["defaults"]))
def test_a_sparse_document_decodes_to_the_full_form(case: dict[str, Any]) -> None:
    assert InjectionPlan.from_data(case["data"]).to_data() == case["canonical"]
    assert InjectionPlan.from_data(case["canonical"]).to_data() == case["canonical"]


@pytest.mark.parametrize("case", DOC["refused"], ids=names(DOC["refused"]))
def test_a_refused_document_is_a_value_error_that_says_why(case: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="invalid injection plan data") as refused:
        InjectionPlan.from_data(case["data"])

    assert case["error"] in str(refused.value)


def test_a_text_that_is_not_json_is_a_value_error() -> None:
    for text in ("", "{", "[]", "NaN"):
        with pytest.raises(ValueError):
            InjectionPlan.from_json(text)
