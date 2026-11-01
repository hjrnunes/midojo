"""Deprecated alias for :mod:`midojo.evaluation_context`, kept until the aliases are removed (D24).

``from midojo.episode_context import EpisodeContext, open_episode`` gives you
:class:`~midojo.evaluation_context.EvaluationContext` and
:func:`~midojo.evaluation_context.open_evaluation`; every other name here is
the same object as the new module's. New code imports
:mod:`midojo.evaluation_context`.
"""

from __future__ import annotations

from midojo.evaluation_context import (
    INTERRUPTED_EVIDENCE,
    AgentBinding,
    ResetOutcome,
)
from midojo.evaluation_context import (
    EvaluationContext as EpisodeContext,
)
from midojo.evaluation_context import (
    EvaluationResult as EpisodeResult,
)
from midojo.evaluation_context import (
    EvaluationStep as EpisodeStep,
)
from midojo.evaluation_context import (
    _refused as _refused,
)
from midojo.evaluation_context import (
    open_evaluation as open_episode,
)
from midojo.injection_spec import InjectionSpec as Episode

__all__ = [
    "INTERRUPTED_EVIDENCE",
    "AgentBinding",
    "Episode",
    "EpisodeContext",
    "EpisodeResult",
    "EpisodeStep",
    "ResetOutcome",
    "open_episode",
]
