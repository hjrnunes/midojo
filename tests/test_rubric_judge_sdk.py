"""The rubric judge lives in ``midojo_sdk`` and imports only the standard library.

The judge is a port of Garak's ``detectors/rubric_judge.py`` and the code the
orchestrator and the consumer's judge service call. It takes the rubric and the
evidence as plain data and imports nothing from Asago, so a fork checkout runs it
without an Asago environment.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

MODULE = Path(__file__).parents[1] / "interception" / "python-sdk" / "src" / "midojo_sdk" / "rubric_judge.py"
# The only third-party import the judge may declare.
ALLOWED = {"httpx", "midojo_sdk"}


def imported_modules(tree: ast.Module) -> set[str]:
    """The top-level names every import in ``tree`` reaches."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                names.add("midojo_sdk")
            elif node.module:
                names.add(node.module.split(".")[0])
    return names


def test_the_module_imports_only_the_standard_library_and_httpx() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))

    outside = imported_modules(tree) - sys.stdlib_module_names - ALLOWED

    assert outside == set()
    assert not any(name.startswith("asago_") for name in imported_modules(tree))


def test_the_record_function_is_public() -> None:
    from midojo_sdk import rubric_judge

    assert callable(rubric_judge.record)
    assert not hasattr(rubric_judge, "_record")


def test_the_prompt_version_is_the_one_the_consumer_had_when_the_judge_moved() -> None:
    from midojo_sdk import rubric_judge

    assert rubric_judge.PROMPT_VERSION == "garak-rubric-judge-v1"
