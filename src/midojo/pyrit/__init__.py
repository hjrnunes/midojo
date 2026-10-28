"""The PyRIT route: PyRIT owns the campaign, MiDojo is its target and its scorer.

:class:`MidojoEpisodeTarget` is a PyRIT ``PromptTarget`` whose every send runs
one MiDojo episode (:func:`midojo.open_episode`) with the candidate text as the
payload of one declared surface, and answers with the agent's reply and the
episode's sealed identity and verdict in the response's ``prompt_metadata``.
:class:`MidojoVerdictScorer` (true/false) and :class:`MidojoVerdictFloatScorer`
(float scale, for PAIR and TAP through ``FloatScaleThresholdScorer``) score that
sealed verdict; evidence that can't decide it is undetermined, never false.

It needs PyRIT, MiDojo's ``pyrit`` extra (``pip install 'midojo[pyrit]'``).
"""

try:
    import pyrit as _pyrit  # noqa: F401
except ImportError as error:  # pragma: no cover - only without the extra
    raise ImportError(
        "midojo.pyrit needs PyRIT, which MiDojo's optional 'pyrit' extra installs: "
        "pip install 'midojo[pyrit]' (or uv sync --extra pyrit)"
    ) from error

from midojo.pyrit.scorers import MidojoVerdictFloatScorer, MidojoVerdictScorer, verdict_of
from midojo.pyrit.target import METADATA_PREFIX, MidojoEpisodeTarget, MidojoVerdict

__all__ = [
    "METADATA_PREFIX",
    "MidojoEpisodeTarget",
    "MidojoVerdict",
    "MidojoVerdictFloatScorer",
    "MidojoVerdictScorer",
    "verdict_of",
]
