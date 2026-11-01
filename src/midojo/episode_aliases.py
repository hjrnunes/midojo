"""The deprecated names the episode -> injection spec rename replaced (D23, D24).

Before the rename, the modules below exported these names because they imported
them from the old modules. The names stay importable from the same module paths
until the aliases go (M9), and each one is the same object as the name it
aliases. The mapping lives here because MiDojo keeps the old spellings out of
every module that is not an alias shim (VAL-RENAME-003): each module resolves a
name through a module-level ``__getattr__`` that delegates to this module.
"""

from __future__ import annotations

import importlib
from typing import Any

# Old name -> the module and name of the new object it aliases.
_TARGETS: dict[str, tuple[str, str]] = {
    "Episode": ("midojo.injection_spec", "InjectionSpec"),
    "EpisodeContext": ("midojo.evaluation_context", "EvaluationContext"),
    "EpisodeRun": ("midojo.orchestrator", "EvaluationRun"),
    "EpisodeStep": ("midojo.evaluation_context", "EvaluationStep"),
    "open_episode": ("midojo.evaluation_context", "open_evaluation"),
}

# Module path -> the old names that module exported before the rename.
_EXPORTED: dict[str, tuple[str, ...]] = {
    "midojo.control_plane.grading": ("Episode",),
    "midojo.control_plane.inprocess_control": ("Episode",),
    "midojo.control_plane.service": ("Episode",),
    "midojo.control_plane.state": ("Episode",),
    "midojo.episode_context": ("Episode",),
    "midojo.episode_document": ("Episode",),
    "midojo.episode_handle": ("Episode",),
    "midojo.orchestrator": ("Episode", "EpisodeContext", "EpisodeRun", "EpisodeStep", "open_episode"),
    "midojo.pyrit.target": ("Episode", "EpisodeContext"),
    "midojo.yaml_task_suite": ("Episode",),
}


def names(module_name: str) -> tuple[str, ...]:
    """The deprecated names ``module_name`` exported before the rename."""
    return _EXPORTED.get(module_name, ())


def resolve(module_name: str, name: str) -> Any:
    """The new object aliased by an old ``name`` that ``module_name`` exported."""
    if name not in names(module_name):
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")
    module, attribute = _TARGETS[name]
    return getattr(importlib.import_module(module), attribute)
