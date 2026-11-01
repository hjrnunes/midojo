"""Deprecated alias for :mod:`midojo.evaluation_handle`, kept until the aliases are removed (D24).

``from midojo.episode_handle import EpisodeHandle, open`` gives you
:class:`~midojo.evaluation_handle.EvaluationHandle` and
:func:`~midojo.evaluation_handle.open`; every other name here is the same
object as the new module's. New code imports :mod:`midojo.evaluation_handle`.
"""

from __future__ import annotations

from midojo.evaluation_handle import (
    DEFAULT_MAX_LIVE_EPISODES,
    HANDLE_API_VERSION,
    HANDLE_METHODS,
    MAX_LIVE_ENV,
    HandleClosedError,
    HandleError,
    HandleOwnerError,
    LiveEpisodeLimitError,
    RepeatWithoutResetError,
    SendTurnUnsupported,
    open,
)
from midojo.evaluation_handle import (
    EvaluationContext as EpisodeContext,
)
from midojo.evaluation_handle import (
    EvaluationHandle as EpisodeHandle,
)
from midojo.evaluation_handle import (
    EvaluationResult as EpisodeResult,
)
from midojo.evaluation_handle import (
    EvaluationStep as EpisodeStep,
)
from midojo.injection_spec_document import InjectionSpecDocumentError as EpisodeDocumentError

__all__ = [
    "DEFAULT_MAX_LIVE_EPISODES",
    "HANDLE_API_VERSION",
    "HANDLE_METHODS",
    "MAX_LIVE_ENV",
    "EpisodeContext",
    "EpisodeDocumentError",
    "EpisodeHandle",
    "EpisodeResult",
    "EpisodeStep",
    "HandleClosedError",
    "HandleError",
    "HandleOwnerError",
    "LiveEpisodeLimitError",
    "RepeatWithoutResetError",
    "SendTurnUnsupported",
    "open",
]
