"""``midojo_mim`` is the interceptor, not the harness: it imports ``midojo_spec`` and its own declared dependencies.

The interceptor runs beside the agent, in an image that holds no verifier, runtime, control plane or injection-spec
model. These tests read the source, so they hold on whatever interpreter runs them, and run the package in a child
process in which ``midojo`` and ``midojo_sdk`` cannot be imported at all: the rule is that nothing needs them, not
that nothing happens to import them.
"""

from __future__ import annotations

import ast
import importlib
import os
import pkgutil
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import midojo_mim

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(midojo_mim.__file__).parent
MODULES = sorted(SOURCE.rglob("*.py"))
HARNESS = {"midojo", "midojo_sdk", "suites", "tests"}
"""The names the interceptor never imports: the harness, the SDK that drives it, and this repository's own trees."""


def distribution_names() -> set[str]:
    """The import names of the dependencies ``pyproject.toml`` declares, as ``midojo-spec`` becomes ``midojo_spec``."""
    project = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = (re.split(r"[\s<>=!~;\[]", requirement, maxsplit=1)[0] for requirement in project["dependencies"])
    return {name.replace("-", "_").lower() for name in names}


def imported_modules(tree: ast.Module) -> set[str]:
    """The top-level names every import in ``tree`` reaches; a relative import is ``midojo_mim``."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add("midojo_mim" if node.level else (node.module or "").split(".")[0])
    return names


def test_the_package_has_modules():
    assert {path.name for path in MODULES} >= {"service.py", "mcp.py", "placement.py", "link.py"}


def test_no_module_imports_the_harness_or_the_sdk():
    offenders = {
        path.name: sorted(imported_modules(ast.parse(path.read_text(encoding="utf-8"))) & HARNESS) for path in MODULES
    }
    assert {name: found for name, found in offenders.items() if found} == {}


def test_every_import_is_the_standard_library_the_package_or_a_declared_dependency():
    allowed = set(sys.stdlib_module_names) | {"midojo_mim", ""} | distribution_names()
    for path in MODULES:
        outside = imported_modules(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))) - allowed
        assert outside == set(), f"{path.name} imports {sorted(outside)}, which pyproject.toml does not declare"


def test_the_distribution_depends_on_the_spec_and_never_on_the_harness_or_the_sdk():
    declared = distribution_names()
    assert "midojo_spec" in declared
    assert declared & HARNESS == set()


def test_the_package_targets_python_3_11():
    project = tomllib.loads((PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["requires-python"].replace(" ", "") == ">=3.11"
    for path in MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 11))
        # ``feature_version`` does not reject PEP 695 on every interpreter, and 3.11 can't parse it.
        assert not any(type(node).__name__ == "TypeAlias" for node in ast.walk(tree)), path.name


def test_the_image_copies_and_installs_the_two_packages_and_nothing_of_the_harness():
    lines = [
        line.strip()
        for line in (PACKAGE_ROOT / "Containerfile").read_text(encoding="utf-8").replace("\\\n", " ").splitlines()
    ]
    copied = [line.split()[1:-1] for line in lines if line.startswith("COPY ")]
    assert copied == [["packages/midojo-spec"], ["packages/midojo-mim"]]
    installs = [line for line in lines if "pip install" in line]
    assert len(installs) == 1
    assert re.findall(r"\./[\w./\[\]-]+", installs[0]) == ["./midojo-spec", "./midojo-mim"]
    assert not any(line.startswith("ADD ") for line in lines)
    assert [line for line in lines if line.startswith("ENTRYPOINT")] == ['ENTRYPOINT ["midojo-intercept-serve"]']


BLOCK_AND_IMPORT = """
import importlib
import importlib.abc
import pkgutil
import sys

BLOCKED = {blocked!r}


class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"No module named {{fullname!r}} (blocked by the test)", name=fullname)
        return None


sys.meta_path.insert(0, Unavailable())

import midojo_mim

for module in pkgutil.walk_packages(midojo_mim.__path__, "midojo_mim."):
    importlib.import_module(module.name)

loaded = sorted(name for name in sys.modules if name.split(".")[0] in BLOCKED)
print(repr(loaded))
print(repr(sorted(name for name in sys.modules if name.startswith("midojo_spec"))))
"""


def test_every_module_imports_with_the_harness_and_the_sdk_unavailable():
    code = BLOCK_AND_IMPORT.format(blocked=sorted(HARNESS))
    completed = subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=PACKAGE_ROOT,
        env={**os.environ, "PYTHONPATH": ""},
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    blocked_loaded, spec_loaded = (ast.literal_eval(line) for line in completed.stdout.strip().splitlines())
    assert blocked_loaded == [], f"loaded {blocked_loaded}"
    # The contracts it speaks are the spec's; nothing of the harness's injection-spec model is among them.
    assert "midojo_spec.link" in spec_loaded and "midojo_spec.injection" in spec_loaded
    assert not any("injection_spec" in name for name in spec_loaded if name != "midojo_spec.injection_spec_schema")


def test_every_module_imports_in_this_process():
    for module in pkgutil.walk_packages(midojo_mim.__path__, "midojo_mim."):
        importlib.import_module(module.name)
