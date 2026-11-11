"""The SDK is a client of MiDojo's wire contract, never of MiDojo's runtime.

The SDK imports ``midojo_spec`` (the contract's codecs, rules and records), ``midojo_mim`` (the MCP face its agent
mode extends) and nothing from ``midojo``. The package itself needs no third-party package: it imports with
``midojo`` and every other package unavailable.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SDK_SOURCE = ROOT / "interception/python-sdk/src/midojo_sdk"

# Every top-level module that is neither the standard library, the SDK nor the spec package is unimportable, so
# nothing the SDK imports can hide: midojo, httpx, pydantic, fastapi, grpc, google.protobuf and openshell are blocked.
BLOCK_PREAMBLE = """
import importlib.abc
import sys

ALLOWED = set(sys.stdlib_module_names) | {"midojo_sdk", "midojo_spec", "__main__"}


class Unavailable(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        top = fullname.split(".")[0]
        if top in ALLOWED or top.startswith("_frozen_importlib") or top.startswith("_distutils_hack"):
            return None
        raise ModuleNotFoundError(f"No module named {fullname!r} (blocked by the test)", name=fullname)


for name in [name for name in sys.modules if name.split(".")[0] not in ALLOWED]:
    del sys.modules[name]
sys.meta_path.insert(0, Unavailable())


def outside_the_standard_library():
    return sorted(name for name in sys.modules if name.split(".")[0] not in ALLOWED)
"""

IMPORT_THE_SDK = """
import midojo_sdk
import midojo_sdk.session
print(outside_the_standard_library())
"""


def _child(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-I", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ""},
        timeout=120,
    )


def test_the_sdk_never_imports_midojo():
    offenders: list[str] = []
    files = sorted(SDK_SOURCE.rglob("*.py"))
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                offenders += [f"{path.name}: import {alias.name}" for alias in node.names if _is_midojo(alias.name)]
            elif isinstance(node, ast.ImportFrom) and node.module is not None and _is_midojo(node.module):
                offenders.append(f"{path.name}: from {node.module} import")
    assert offenders == [], offenders
    assert len(files) >= 6, files


def _is_midojo(name: str) -> bool:
    return name == "midojo" or name.startswith("midojo.")


def test_the_sdk_modules_import_with_midojo_blocked():
    result = _child(BLOCK_PREAMBLE + IMPORT_THE_SDK)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]", f"loaded {result.stdout.strip()}"
