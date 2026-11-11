"""The agent seams: what the SDKs keep, what a driver is refused, and that nothing here leans on Asago's old driver.

Agent mode (the agent gets the evaluation's session and runs its own interception layer, built with an SDK) is
parked, not removed: the SDKs, their wire tests and the control plane's ``/agent`` routes stay, and a roadmap item
may use them. Three things hold meanwhile.

- The seams the SDKs use are still the contracts' and the face's: ``Recorder`` and ``WorldState`` in ``midojo_spec``,
  ``Sessions`` on the face in ``midojo_mim``, the SDK's face as that face with a control plane added.
- Intake a driver cannot use is refused by name. The interceptor service serves the MCP face over a real upstream and
  nothing else (``packages/midojo-mim/tests/test_service.py`` holds those refusals), the evaluation handle accepts only
  a host or a remote interception, and ``midojo-run`` refuses a suite whose tools live on the host without
  ``--interception host``.
- No Python or TypeScript import in this repository's own source, and no dependency in its manifests or lock, names
  a package of Asago's driver: the modules that drove MiDojo from the orchestrator, from the consumer's
  ``bundle-midojo`` and from ``--tool midojo`` live in those repositories and are removed there. The scan reads
  literal import names only. A name built at run time, a tests tree (a fixture may name an Asago package as data) and
  a file that is not Python or TypeScript source (shell, YAML, a Containerfile, documents) are outside it.
"""

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path
from typing import Any

import midojo_mim.mcp
import midojo_sdk.mcp
import pytest
from midojo_spec import recorder

from midojo import evaluation_handle
from tests.test_episode_handle import scripted_upstream, weather_episode

ROOT = Path(__file__).resolve().parent.parent

# --- the seams the SDKs keep ---


def test_the_sdks_agent_mode_is_the_face_with_a_control_plane_added():
    assert issubclass(midojo_sdk.mcp.MidojoMCP, midojo_mim.mcp.MidojoMCP)
    assert midojo_sdk.mcp.BLOCKED_REASON == midojo_mim.mcp.BLOCKED_REASON
    # The seams a face records and simulates through are the contracts', for the SDK as for the service.
    assert isinstance(recorder.Recorder, type) and isinstance(recorder.WorldState, type)
    assert isinstance(midojo_mim.mcp.Sessions, type)


def test_the_sdks_face_with_neither_a_control_plane_nor_a_recorder_is_refused_by_name():
    with pytest.raises(ValueError, match="needs a control_plane_url or a recorder"):
        midojo_sdk.mcp.MidojoMCP("agent-side")


def test_the_interceptors_face_takes_no_control_plane():
    """Agent mode is the SDK's addition: the interceptor's face has no ``control_plane_url`` to take."""
    agent_mode: dict[str, Any] = {"control_plane_url": "http://127.0.0.1:8090"}
    with pytest.raises(TypeError, match="control_plane_url"):
        midojo_mim.mcp.MidojoMCP("service-side", **agent_mode)


# --- intake a driver cannot use ---


@pytest.mark.parametrize("interception", ["agent", "host", {"upstream": "http://127.0.0.1:1/mcp"}, object(), True])
def test_a_handle_given_an_interception_that_is_neither_host_nor_remote_is_refused_by_name(interception: Any):
    """Agent mode is not an interception the handle can open: it takes a host or a remote one, or runs in this process."""
    with pytest.raises(TypeError, match="HostInterception or a RemoteInterception"):
        evaluation_handle.open(weather_episode(), upstream=scripted_upstream(), interception=interception)


# --- nothing here leans on Asago's driver ---

SCANNED_CODE = [
    "src",
    "suites",
    "scripts",
    "examples",
    "interception/python-sdk/src",
    "packages/midojo-spec/src",
    "packages/midojo-mim/src",
]
"""The trees of this repository's own Python source. Tests are not scanned: a fixture may name an Asago package as data."""

SCANNED_TYPESCRIPT = ["interception/pi-sdk/src"]
"""The TypeScript SDK's own source, read for its import specifiers."""

DYNAMIC_IMPORTS = {"import_module", "__import__"}


def is_asago(name: str) -> bool:
    """A package of Asago's: any name that starts with ``asago``, with or without an npm scope (``@asago/...``)."""
    return name.strip().lower().lstrip("@").startswith("asago")


def imported_names(tree: ast.Module) -> set[str]:
    """Every module a file imports, by statement or by a call that names one as a string."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            func = node.func
            called = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else None
            if called in DYNAMIC_IMPORTS and isinstance(node.args[0].value, str):
                names.add(node.args[0].value)
    return names


def python_files() -> list[Path]:
    return sorted(path for tree in SCANNED_CODE for path in (ROOT / tree).rglob("*.py"))


def test_the_scan_reads_the_repositorys_code():
    files = python_files()
    assert len(files) >= 100, len(files)
    assert {tree for tree in SCANNED_CODE if not any(ROOT / tree in path.parents for path in files)} == set()


def test_no_module_imports_a_package_of_asagos_driver():
    offenders = {
        str(path.relative_to(ROOT)): sorted(
            name for name in imported_names(ast.parse(path.read_text(encoding="utf-8"))) if is_asago(name)
        )
        for path in python_files()
    }
    assert {path: names for path, names in offenders.items() if names} == {}


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ("import asago_orch.midojo_driver", {"asago_orch.midojo_driver"}),
        ("from asago_bundle_midojo import run", {"asago_bundle_midojo"}),
        ("import importlib\nimportlib.import_module('asago_orch')", {"importlib", "asago_orch"}),
        ("__import__('asago_package')", {"asago_package"}),
        ("from . import link", set()),
    ],
)
def test_the_scan_names_what_a_file_imports(source: str, found: set[str]):
    assert imported_names(ast.parse(source)) == found


@pytest.mark.parametrize(
    ("name", "asago"),
    [
        ("asago_orch", True),
        ("asago-bundle-midojo", True),
        ("asago.orch", True),
        ("Asago_Package", True),
        ("asagobundle", True),
        ("@asago/driver", True),
        ("midojo_spec", False),
        ("fastmcp", False),
        ("midojo_asago", False),
        ("", False),
    ],
)
def test_the_name_check_flags_every_name_that_starts_with_asago(name: str, asago: bool):
    assert is_asago(name) is asago


def requirement_name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[@(]", requirement.strip(), maxsplit=1)[0]


def declared_dependencies(pyproject: dict[str, Any]) -> set[str]:
    """Every distribution a pyproject names as a dependency, an extra, a group or a source."""
    project = pyproject.get("project", {})
    requirements: list[str] = list(project.get("dependencies", []))
    for extra in project.get("optional-dependencies", {}).values():
        requirements += extra
    for group in pyproject.get("dependency-groups", {}).values():
        requirements += [item for item in group if isinstance(item, str)]
    names = {requirement_name(requirement) for requirement in requirements}
    return names | set(pyproject.get("tool", {}).get("uv", {}).get("sources", {}))


PYPROJECTS = [
    "pyproject.toml",
    "packages/midojo-spec/pyproject.toml",
    "packages/midojo-mim/pyproject.toml",
    "interception/python-sdk/pyproject.toml",
]


@pytest.mark.parametrize("path", PYPROJECTS)
def test_no_distribution_of_this_repository_depends_on_a_package_of_asagos_driver(path: str):
    pyproject = tomllib.loads((ROOT / path).read_text(encoding="utf-8"))
    assert {name for name in declared_dependencies(pyproject) if is_asago(name)} == set()


def test_the_dependency_reader_reads_every_place_a_pyproject_names_one():
    sample = {
        "project": {
            "dependencies": ["asago-orch>=1", "fastmcp[x]>=3; python_version >= '3.11'"],
            "optional-dependencies": {"suites": ["asago_bundle_midojo @ file:///x"]},
        },
        "dependency-groups": {"dev": ["pytest", {"include-group": "other"}, "asago-package"]},
        "tool": {"uv": {"sources": {"asago-injection-spec": {"workspace": True}}}},
    }
    assert declared_dependencies(sample) == {
        "asago-orch",
        "fastmcp",
        "asago_bundle_midojo",
        "pytest",
        "asago-package",
        "asago-injection-spec",
    }


def test_the_repositorys_pyprojects_name_the_dependencies_the_scan_expects():
    names = set().union(
        *(declared_dependencies(tomllib.loads((ROOT / path).read_text(encoding="utf-8"))) for path in PYPROJECTS)
    )
    assert {"midojo-spec", "midojo-mim", "fastmcp", "midojo-sdk"} <= names


def test_the_lock_holds_no_package_of_asagos_driver():
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    names = {package["name"] for package in lock["package"]}
    assert {"midojo", "midojo-spec", "midojo-mim", "midojo-sdk"} <= names
    assert {name for name in names if is_asago(name)} == set()


def test_the_typescript_sdk_depends_on_no_package_of_asagos_driver():
    package = json.loads((ROOT / "interception/pi-sdk/package.json").read_text(encoding="utf-8"))
    names = {
        name
        for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies")
        for name in package.get(section, {})
    }
    assert names, "the scan found no dependency to read"
    assert {name for name in names if is_asago(name)} == set()


TYPESCRIPT_IMPORT = re.compile(r"""(?:\bfrom|\bimport|\brequire\s*\()\s*\(?\s*["']([^"']+)["']""")


def typescript_imports(source: str) -> set[str]:
    """Every module specifier a TypeScript or JavaScript file names in an import, an export-from, a require or an import()."""
    return set(TYPESCRIPT_IMPORT.findall(source))


def typescript_files() -> list[Path]:
    return sorted(
        path
        for tree in SCANNED_TYPESCRIPT
        for pattern in ("*.ts", "*.mts", "*.js", "*.mjs")
        for path in (ROOT / tree).rglob(pattern)
    )


def test_the_typescript_sdks_source_imports_no_package_of_asagos_driver():
    files = typescript_files()
    imported = {name for path in files for name in typescript_imports(path.read_text(encoding="utf-8"))}
    assert "@earendil-works/pi-coding-agent" in imported, "the scan did not read the SDK's source"
    assert {name for name in imported if is_asago(name)} == set()


@pytest.mark.parametrize(
    ("source", "found"),
    [
        ('import { a } from "@asago/driver";', {"@asago/driver"}),
        ("import type { T } from 'asago-driver';", {"asago-driver"}),
        ('import "asago-side-effect";', {"asago-side-effect"}),
        ('export * from "asago-reexport";', {"asago-reexport"}),
        ('const x = require("asago-cjs");', {"asago-cjs"}),
        ('const x = await import("asago-dynamic");', {"asago-dynamic"}),
        ('import { createHash } from "node:crypto";', {"node:crypto"}),
        ("// no import here", set()),
    ],
)
def test_the_typescript_scan_names_what_a_file_imports(source: str, found: set[str]):
    assert typescript_imports(source) == found
