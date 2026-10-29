"""Optional capabilities: providers imported only when they are used.

Reading a suite, building an episode, grading it and running it on a runtime
that needs nothing more import no provider's dependencies. A provider whose
dependencies are heavy, such as the ``openshell`` runtime (gRPC, protobuf and
the OpenShell SDK), is imported when it is first selected, through
:func:`import_provider`. If one of its dependencies is missing, that raises
:class:`MissingCapabilityError`, which names the capability and the packages
that provide it.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from types import ModuleType


class MissingCapabilityError(ImportError):
    """A capability was selected whose provider can't be imported, as a package it needs isn't installed.

    ``capability`` names what was asked for (for example ``the 'openshell'
    runtime``), ``packages`` the distributions to install for it, and ``name``
    (as for any ImportError) the module that failed to import. It is an
    ImportError, so code that already handles a failed import handles it too;
    the failed import is its ``__cause__``.
    """

    def __init__(self, capability: str, packages: Sequence[str], *, missing: str | None = None) -> None:
        self.capability = capability
        self.packages = tuple(packages)
        install = " ".join(self.packages)
        detail = f" (module {missing!r} is missing)" if missing else ""
        super().__init__(
            f"{capability} is not available{detail}: install {', '.join(map(repr, self.packages))} "
            f"(for example `pip install {install}`)",
            name=missing,
        )


def _provides(dependency: str, missing: str) -> bool:
    """Whether the missing module ``missing`` is ``dependency``, inside it, or a package that contains it."""
    return missing == dependency or missing.startswith(f"{dependency}.") or dependency.startswith(f"{missing}.")


def import_provider(
    module: str, *, capability: str, packages: Sequence[str], dependencies: Sequence[str]
) -> ModuleType:
    """Import a provider's ``module``, raising :class:`MissingCapabilityError` if one of its ``dependencies`` is missing.

    ``dependencies`` are the top-level modules the provider needs from
    ``packages`` (``grpc``, ``google.protobuf``, ``openshell``). Any other
    import error, a missing module that isn't one of them included, is
    raised as it is: it is a fault in the provider, not a missing capability.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        missing = exc.name
        if missing is None or not any(_provides(dependency, missing) for dependency in dependencies):
            raise
        raise MissingCapabilityError(capability, packages, missing=missing) from exc
