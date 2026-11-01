"""Deprecated alias for :mod:`midojo.injection_spec`, kept until the aliases are removed (D24).

``from midojo.episode import Episode`` gives you
:class:`~midojo.injection_spec.InjectionSpec`; every other name here is the same
object as the new module's. New code imports :mod:`midojo.injection_spec`.
"""

from __future__ import annotations

from midojo.injection_spec import (
    DEFAULT_EPISODE_NAME,
    MAX_NESTING_DEPTH,
    WRITE_DISPOSITIONS,
    InterceptionSettings,
    RuntimeCheckError,
    ToolPolicy,
    UnknownTaskError,
    attack_outcome,
    freeze,
    thaw,
    tool_policy,
)
from midojo.injection_spec import (
    InjectionSpec as Episode,
)

__all__ = [
    "DEFAULT_EPISODE_NAME",
    "MAX_NESTING_DEPTH",
    "WRITE_DISPOSITIONS",
    "Episode",
    "InterceptionSettings",
    "RuntimeCheckError",
    "ToolPolicy",
    "UnknownTaskError",
    "attack_outcome",
    "freeze",
    "thaw",
    "tool_policy",
]
