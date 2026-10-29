"""Read suite host-tool declarations without loading their implementations."""

from __future__ import annotations

import importlib
from types import ModuleType


def host_tools_reference(suite_module: str | ModuleType) -> str | None:
    """Read a suite module's ``HOST_TOOLS`` as ``module:function``, or None.

    A string is a dotted module path or a bundled suite's bare name. An
    already imported module is read directly, without guessing from its
    suite's name. Only the suite module is imported, never anything its
    declaration names. A suite module that doesn't exist (it, or a package
    it would be in, isn't found) has no declaration; any other import error,
    one raised while the suite module runs its own imports included, is
    raised as it was.

    Raise ValueError for a malformed declaration; callers such as the CLI
    can translate it into their own error type.
    """
    if isinstance(suite_module, str):
        imported = import_suite_module(suite_module)
        if imported is None:
            return None
        suite_module = imported
    reference = getattr(suite_module, "HOST_TOOLS", None)
    if reference is None:
        return None
    if not isinstance(reference, str) or reference.count(":") != 1 or not all(reference.split(":")):
        raise ValueError(f"{suite_module.__name__}.HOST_TOOLS must be a 'module:function' string, got {reference!r}.")
    return reference


def import_suite_module(suite: str) -> ModuleType | None:
    """Import a suite module by dotted path or bundled suite name, or None when it doesn't exist.

    Only the suite module itself, or a package it would be in, not being
    found reads as None; any other import error, one raised while the suite
    module runs its own imports included, is raised as it was.
    """
    module_path = suite if "." in suite else f"suites.{suite}"
    try:
        return importlib.import_module(module_path)
    except ModuleNotFoundError as error:
        if not _names_module_or_parent(error, module_path):
            raise
        return None


def _names_module_or_parent(error: ModuleNotFoundError, module_path: str) -> bool:
    """Whether ``error`` is ``module_path`` itself not being found, or a package it would be in."""
    return error.name is not None and (module_path == error.name or module_path.startswith(error.name + "."))
