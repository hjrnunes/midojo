"""A suite's HOST_TOOLS and SYSTEM_MESSAGE are read without loading its tools, and without hiding its import errors."""

import sys

import pytest

from midojo.orchestrator import _resolve_system_message
from midojo.suite_tools import host_tools_reference
from suites import weather


@pytest.mark.parametrize("spec", ["weather", "suites.weather"])
def test_a_bundled_declaration_by_name_or_dotted_path(spec):
    assert host_tools_reference(spec) == weather.HOST_TOOLS


@pytest.mark.parametrize("spec", ["missing_suite_declaration", "missing_suite_package.suite", "suites.missing_suite"])
def test_a_missing_suite_module_or_parent_package_has_no_declaration(spec):
    assert host_tools_reference(spec) is None


@pytest.fixture
def declaration_module(tmp_path, monkeypatch, request):
    """``declaration_probe.suite``, whose source is the test's parameter, beside a tools module that must not load."""
    package = tmp_path / "declaration_probe"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "suite.py").write_text(request.param)
    (package / "tools.py").write_text('raise AssertionError("tools must not be imported")\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        yield "declaration_probe.suite"
    finally:
        for name in ("declaration_probe", "declaration_probe.suite", "declaration_probe.tools"):
            sys.modules.pop(name, None)


@pytest.mark.parametrize(
    "declaration_module,expected",
    [('HOST_TOOLS = "declaration_probe.tools:register"\n', "declaration_probe.tools:register"), ("", None)],
    indirect=["declaration_module"],
)
def test_an_external_declaration_by_dotted_path(declaration_module, expected):
    assert host_tools_reference(declaration_module) == expected
    assert "declaration_probe.tools" not in sys.modules


@pytest.mark.parametrize(
    "declaration_module,missing",
    [
        ("import missing_suite_dependency\n", "missing_suite_dependency"),
        ("import declaration_probe.missing_dependency\n", "declaration_probe.missing_dependency"),
        ("import declaration_probe.suite_helpers\n", "declaration_probe.suite_helpers"),
    ],
    indirect=["declaration_module"],
)
def test_a_missing_dependency_of_the_suite_module_propagates(declaration_module, missing):
    with pytest.raises(ModuleNotFoundError) as raised:
        host_tools_reference(declaration_module)
    assert raised.value.name == missing


@pytest.mark.parametrize("declaration_module", ['raise ImportError("internal sentinel")\n'], indirect=True)
def test_an_import_error_the_suite_module_raises_propagates(declaration_module):
    with pytest.raises(ImportError, match="internal sentinel") as raised:
        host_tools_reference(declaration_module)
    assert type(raised.value) is ImportError


@pytest.mark.parametrize("spec", ["missing_suite_declaration", "missing_suite_package.suite", "suites.missing_suite"])
def test_a_missing_suite_module_or_parent_package_has_no_system_message(spec):
    assert _resolve_system_message(spec) == ""


@pytest.mark.parametrize(
    "declaration_module,expected",
    [('SYSTEM_MESSAGE = "You are the probe."\n', "You are the probe."), ("", "")],
    indirect=["declaration_module"],
)
def test_an_external_system_message_by_dotted_path(declaration_module, expected):
    assert _resolve_system_message(declaration_module) == expected
    assert "declaration_probe.tools" not in sys.modules


@pytest.mark.parametrize(
    "declaration_module,missing",
    [
        ("import missing_suite_dependency\n", "missing_suite_dependency"),
        ("import declaration_probe.missing_dependency\n", "declaration_probe.missing_dependency"),
    ],
    indirect=["declaration_module"],
)
def test_a_missing_dependency_of_the_suite_module_propagates_from_the_system_message(declaration_module, missing):
    with pytest.raises(ModuleNotFoundError) as raised:
        _resolve_system_message(declaration_module)
    assert raised.value.name == missing


@pytest.mark.parametrize("declaration_module", ['raise ImportError("internal sentinel")\n'], indirect=True)
def test_an_import_error_the_suite_module_raises_propagates_from_the_system_message(declaration_module):
    with pytest.raises(ImportError, match="internal sentinel") as raised:
        _resolve_system_message(declaration_module)
    assert type(raised.value) is ImportError
