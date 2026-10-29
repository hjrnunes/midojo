"""The control plane: evaluations, their lifecycle and their sealed snapshots, in process or served over HTTP.

``midojo.control_plane.config`` stays reachable as an attribute of this
package however it was imported, as when ``import midojo`` loaded it: it is
imported on first access, and listed by ``dir()``.
"""

from types import ModuleType

# The child modules ``import midojo`` used to load as a side effect, imported on first access.
_SUBMODULES = frozenset({"config"})


def __getattr__(name: str) -> ModuleType:
    if name in _SUBMODULES:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_SUBMODULES})
