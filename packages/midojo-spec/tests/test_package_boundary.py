"""``midojo_spec`` runs on Python 3.11 with the standard library alone.

A Garak or consumer environment installs it without MiDojo's runtime, the SDK, FastMCP,
pydantic or httpx. These tests read the source, so they hold on whatever interpreter runs
them: the import rule, and the syntax rule (nothing newer than 3.11 parses).
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import sys
from pathlib import Path

import midojo_spec

PACKAGE = Path(midojo_spec.__file__).parent
MODULES = sorted(PACKAGE.rglob("*.py"))


def imported_modules(tree: ast.Module) -> set[str]:
    """The top-level names every import in ``tree`` reaches; a relative import is ``midojo_spec``."""

    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add("midojo_spec" if node.level else (node.module or "").split(".")[0])
    return names


def test_the_package_has_modules() -> None:
    assert MODULES


def test_every_module_imports_only_the_standard_library_and_the_package() -> None:
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        outside = imported_modules(tree) - sys.stdlib_module_names - {"midojo_spec", ""}

        assert outside == set(), f"{path.name} imports {sorted(outside)}"


def test_every_module_parses_as_python_3_11() -> None:
    for path in MODULES:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 11))


def test_no_module_uses_a_type_statement() -> None:
    # ``feature_version`` does not reject PEP 695 on every interpreter, and 3.11 can't parse it.
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert not any(type(node).__name__ == "TypeAlias" for node in ast.walk(tree)), path.name


def test_every_module_imports() -> None:
    for module in pkgutil.walk_packages(midojo_spec.__path__, "midojo_spec."):
        importlib.import_module(module.name)
