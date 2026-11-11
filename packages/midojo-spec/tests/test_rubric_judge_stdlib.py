"""The rubric judge is a port of Garak's ``detectors/rubric_judge.py`` and the code the judge service calls.

It takes the rubric and the evidence as plain data and imports nothing from Asago, so a
fork checkout runs it without an Asago environment. ``test_package_boundary.py`` holds the
import rule for the whole package.
"""

from __future__ import annotations

from midojo_spec import rubric_judge


def test_the_record_function_is_public() -> None:
    assert callable(rubric_judge.record)
    assert not hasattr(rubric_judge, "_record")


def test_the_prompt_version_is_the_one_the_consumer_had_when_the_judge_moved() -> None:
    assert rubric_judge.PROMPT_VERSION == "garak-rubric-judge-v1"
