"""A small test-only mutation harness: compile a module with one source edit, apart from the real one.

A mutant is a module's source with one exact snippet replaced. The snippet
must occur exactly once, so a mutant whose code has moved fails loudly instead
of silently testing the original. The mutated module is a new module object
under another name: the real one, and everything that imported from it, are
left as they were. A test then hands the mutant's functions to the cases it
runs, or patches them in for the length of a test.
"""

from __future__ import annotations
import __future__

import inspect
import sys
import textwrap
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Mutant:
    """One single edit to a module's source: ``old`` becomes ``new``."""

    name: str
    old: str
    new: str


def _edited(source: str, mutant: Mutant, where: str) -> str:
    count = source.count(mutant.old)
    if count != 1:
        raise AssertionError(f"mutant {mutant.name!r}: its snippet occurs {count} times in {where}")
    return source.replace(mutant.old, mutant.new)


def mutated(module: types.ModuleType, mutant: Mutant) -> types.ModuleType:
    """Return a fresh copy of ``module`` compiled from its source with ``mutant``'s edit applied.

    Only for a module whose import has no side effects beyond defining names.
    """
    source = _edited(inspect.getsource(module), mutant, module.__name__)
    name = f"{module.__name__}__mutant"
    copy = types.ModuleType(name)
    copy.__file__ = module.__file__
    # Dataclasses look their module up while they are created.
    sys.modules[name] = copy
    try:
        exec(compile(source, f"<{mutant.name}>", "exec"), copy.__dict__)
    finally:
        del sys.modules[name]
    return copy


def mutated_function(function: Callable[..., Any], mutant: Mutant) -> Callable[..., Any]:
    """Return ``function`` compiled from its source with ``mutant``'s edit applied, over its module's globals.

    For a function whose module registers something on import, so the module
    can't be compiled again; a method comes back as a plain function, to set
    on its class.
    """
    source = _edited(textwrap.dedent(inspect.getsource(function)), mutant, function.__qualname__)
    namespace: dict[str, Any] = {}
    flags = __future__.annotations.compiler_flag
    exec(
        compile(source, f"<{mutant.name}>", "exec", flags=flags, dont_inherit=True),
        dict(function.__globals__),
        namespace,
    )
    return namespace[function.__name__]
