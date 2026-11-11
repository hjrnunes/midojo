"""``vectors/policy/policy.json``: the tool policy's document form and the interceptor's view of it.

The cases were written by hand, so what a decoder must accept and refuse is checked against an
oracle that is not the decoder. A reader in another language runs the same file.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any, Literal

import pytest

from midojo_spec.policy import WRITE_DISPOSITIONS, ToolPolicy, tool_policy

DOC = json.loads((Path(__file__).parents[1] / "vectors" / "policy" / "policy.json").read_text(encoding="utf-8"))


def names(cases: list[dict[str, Any]]) -> list[str]:
    return [case["name"] for case in cases]


@pytest.mark.parametrize("case", DOC["accepted"], ids=names(DOC["accepted"]))
def test_data_decodes_to_its_policy(case: dict[str, Any]) -> None:
    policy = ToolPolicy.from_data(case["data"])

    assert policy.allowed == tuple(case["allowed"])
    assert policy.write_disposition_default == case["write_disposition_default"]
    assert dict(policy.write_disposition) == case["write_disposition"]


@pytest.mark.parametrize("case", DOC["accepted"], ids=names(DOC["accepted"]))
def test_a_policy_encodes_to_its_canonical_data(case: dict[str, Any]) -> None:
    assert ToolPolicy.from_data(case["data"]).to_data() == case["canonical"]


@pytest.mark.parametrize("case", DOC["accepted"], ids=names(DOC["accepted"]))
def test_the_canonical_data_decodes_to_the_same_policy(case: dict[str, Any]) -> None:
    policy = ToolPolicy.from_data(case["data"])

    assert ToolPolicy.from_data(case["canonical"]) == policy
    assert ToolPolicy.from_data(policy.to_data()) == policy


@pytest.mark.parametrize("case", DOC["accepted"], ids=names(DOC["accepted"]))
def test_the_interceptor_mirrors_and_blocks_what_the_vector_names(case: dict[str, Any]) -> None:
    mirrored, blocked = tool_policy(ToolPolicy.from_data(case["data"]))

    assert mirrored == frozenset(case["mirrored"])
    assert blocked == tuple(case["blocked"])


def test_no_policy_mirrors_every_tool_and_blocks_none() -> None:
    mirrored, blocked = tool_policy(None)

    assert (mirrored, list(blocked)) == (DOC["no_policy"]["mirrored"], DOC["no_policy"]["blocked"])


@pytest.mark.parametrize("case", DOC["refused"], ids=names(DOC["refused"]))
def test_refused_data_raises_value_error_with_its_reason(case: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(case['message'])}$"):
        ToolPolicy.from_data(case["data"])


def test_the_write_dispositions_are_forward_and_block() -> None:
    assert WRITE_DISPOSITIONS == {"forward", "block"}


class _Name(str):
    pass


@pytest.mark.parametrize("allowed", ["get_weather", 3, None, {"a"}])
def test_the_constructor_refuses_allowed_that_is_not_a_sequence_of_names(allowed: Any) -> None:
    with pytest.raises(TypeError, match="^allowed must be a sequence of tool names, not a string$"):
        ToolPolicy(allowed)


def test_the_constructor_refuses_a_str_subclass_as_a_tool_name() -> None:
    with pytest.raises(TypeError, match="^an allowed tool name must be a str, not _Name$"):
        ToolPolicy((_Name("a"),))
    with pytest.raises(TypeError, match="^a tool name must be a str, not _Name$"):
        ToolPolicy(("a",), write_disposition={_Name("a"): "block"})


def test_a_policy_is_immutable_and_copies_what_it_is_given() -> None:
    overrides: dict[str, Literal["forward", "block"]] = {"a": "block"}
    policy = ToolPolicy(("a",), write_disposition=overrides)
    overrides["a"] = "forward"

    assert policy.write_disposition["a"] == "block"
    with pytest.raises(TypeError):
        policy.write_disposition["a"] = "forward"  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        policy.allowed = ()  # type: ignore[misc]
