"""MiDojo: man-in-the-middle red teaming for AI agents.

The library API: an :class:`Episode` is what to run, :func:`open_episode` runs
it once and seals its evaluation (see :mod:`midojo.episode_context`), and
:func:`grade` grades the sealed :class:`Snapshot`, a pure function. The names
are imported when first used, so ``import midojo`` stays light.
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


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'midojo' has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)
