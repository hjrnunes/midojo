"""MiDojo: man-in-the-middle red teaming for AI agents.

The library API: an :class:`Episode` is what to run, :func:`open_episode` runs
it once and seals its evaluation (see :mod:`midojo.episode_context`), and
:func:`grade` grades the sealed :class:`Snapshot`, a pure function. The names
are imported when first used, so ``import midojo`` stays light.

The child modules a bare ``import midojo`` used to load as a side effect
(``midojo.runtimes``, ``midojo.control_plane``, ...) are still reachable as
attributes of the package: each is imported on first access, and listed by
``dir()``, without being loaded by ``import midojo`` itself.
"""

from typing import TYPE_CHECKING, Any

from midojo import verifiers as _verifiers  # noqa: F401  -- registers built-in verifiers

if TYPE_CHECKING:
    from midojo.control_plane.grading import Verdict, grade
    from midojo.control_plane.state import Snapshot
    from midojo.episode import Episode
    from midojo.episode_context import AgentBinding, EpisodeContext, EpisodeResult, open_episode

__all__ = ["AgentBinding", "Episode", "EpisodeContext", "EpisodeResult", "Snapshot", "Verdict", "grade", "open_episode"]

_EXPORTS = {
    "AgentBinding": "midojo.episode_context",
    "Episode": "midojo.episode",
    "EpisodeContext": "midojo.episode_context",
    "EpisodeResult": "midojo.episode_context",
    "Snapshot": "midojo.control_plane.state",
    "Verdict": "midojo.control_plane.grading",
    "grade": "midojo.control_plane.grading",
    "open_episode": "midojo.episode_context",
}


# The child modules ``import midojo`` loaded when it imported the OpenShell runtime up front, each with the
# descendants it brought along that don't load that runtime (``midojo.runtimes.openshell`` is lazy in its parent).
# Each stays reachable from the package, imported on first access.
_SUBMODULES: dict[str, tuple[str, ...]] = {
    "control_plane": ("config",),
    "observations": (),
    "probes": (),
    "runtimes": (),
    "suite_definition": (),
    "tool_endpoint": (),
    "types": (),
    "verifiers": ("builtin", "combinators", "judge", "openshell", "tool_call_condition"),
}


def __getattr__(name: str) -> Any:
    import importlib

    module = _EXPORTS.get(name)
    if module is not None:
        return getattr(importlib.import_module(module), name)
    if name in _SUBMODULES:
        child = importlib.import_module(f"{__name__}.{name}")
        for descendant in _SUBMODULES[name]:
            importlib.import_module(f"{__name__}.{name}.{descendant}")
        return child
    raise AttributeError(f"module 'midojo' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS, *_SUBMODULES})
