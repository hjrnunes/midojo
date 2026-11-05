"""MiDojo: man-in-the-middle red teaming for AI agents.

The library API: an :class:`InjectionSpec` is what to run, the module-level
:func:`open_evaluation` runs it once and seals its evaluation (see
:mod:`midojo.evaluation_context`), and :func:`grade` grades the sealed
:class:`Snapshot`, a pure function. The names are imported when first used, so
``import midojo`` stays light.

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
    from midojo.evaluation_context import (
        AgentBinding,
        EvaluationContext,
        EvaluationResult,
        EvaluationStep,
        open_evaluation,
    )
    from midojo.evaluation_handle import EvaluationHandle
    from midojo.injection_spec import InjectionSpec
    from midojo.orchestrator import EvaluationRun

__all__ = [
    "AgentBinding",
    "EvaluationContext",
    "EvaluationHandle",
    "EvaluationResult",
    "EvaluationRun",
    "EvaluationStep",
    "InjectionSpec",
    "Snapshot",
    "Verdict",
    "grade",
    "open_evaluation",
]

_EXPORTS = {
    "AgentBinding": "midojo.evaluation_context",
    "EvaluationContext": "midojo.evaluation_context",
    "EvaluationHandle": "midojo.evaluation_handle",
    "EvaluationResult": "midojo.evaluation_context",
    "EvaluationRun": "midojo.orchestrator",
    "EvaluationStep": "midojo.evaluation_context",
    "InjectionSpec": "midojo.injection_spec",
    "Snapshot": "midojo.control_plane.state",
    "Verdict": "midojo.control_plane.grading",
    "grade": "midojo.control_plane.grading",
    "open_evaluation": "midojo.evaluation_context",
}


# The child modules ``import midojo`` loaded when it imported the OpenShell runtime up front, each imported on first
# access. Their own descendants it loaded are reachable from each child however it was imported: a child that doesn't
# import them itself makes them lazy attributes of its own (``control_plane.config``, ``runtimes.openshell``).
_SUBMODULES = frozenset(
    {
        "control_plane",
        "evaluation_context",
        "evaluation_handle",
        "injection_spec",
        "injection_spec_document",
        "observations",
        "probes",
        "runtimes",
        "suite_definition",
        "tool_endpoint",
        "types",
        "verifiers",
    }
)


def __getattr__(name: str) -> Any:
    import importlib

    target = _EXPORTS.get(name)
    if target is not None:
        module, attribute = (target, name) if isinstance(target, str) else target
        return getattr(importlib.import_module(module), attribute)
    if name in _SUBMODULES:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module 'midojo' has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS, *_SUBMODULES})
