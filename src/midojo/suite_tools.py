"""Read suite host-tool declarations without loading their implementations."""

from __future__ import annotations

import importlib
from types import ModuleType


def host_tools_reference(suite_module: str | ModuleType) -> str | None:
    """Read a suite module's ``HOST_TOOLS`` as ``module:function``, or None.

    A string is a dotted module path or a bundled suite's bare name. An
    already imported module is read directly, without guessing from its
    suite's name. Only the suite module is imported, never anything its
    declaration names. An unimportable suite module has no declaration.

    Raise ValueError for a malformed declaration; callers such as the CLI
    can translate it into their own error type.
    """
    if isinstance(suite_module, str):
        module_path = suite_module if "." in suite_module else f"suites.{suite_module}"
        try:
            suite_module = importlib.import_module(module_path)
        except ImportError:
            return None
    reference = getattr(suite_module, "HOST_TOOLS", None)
    if reference is None:
        return None
    if not isinstance(reference, str) or reference.count(":") != 1 or not all(reference.split(":")):
        raise ValueError(f"{suite_module.__name__}.HOST_TOOLS must be a 'module:function' string, got {reference!r}.")
    return reference
