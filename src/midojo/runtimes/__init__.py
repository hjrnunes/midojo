"""Agent runtimes: where an episode's agent runs and how MiDojo observes it.

A suite declares its runtime under ``agent_runtime`` in ``suite.yaml``:

  * ``openshell`` (the default) — :class:`~midojo.runtimes.openshell.OpenShellRuntime`
    creates an OpenShell sandbox for each evaluation, seeds its files, and
    collects runtime observations from outside the agent.
  * ``unmanaged`` (experimental) — :class:`UnmanagedRuntime`. The agent runs
    somewhere MiDojo doesn't control and nothing is observed outside it.

A suite or an episode holds only the runtime's definition: :func:`build_runtime`
is the factory that builds a fresh runtime from it, so concurrent runs never
share one. It looks the definition's ``type`` up in a :class:`RuntimeRegistry`
(:data:`DEFAULT_RUNTIMES` unless the caller passes one;
:func:`register_runtime` adds to the default). The OpenShell runtime is an
optional provider: its module, which needs gRPC, protobuf and the OpenShell
SDK, is imported only when an ``openshell`` runtime is built (or
``midojo.runtimes.OpenShellRuntime`` is first read), and a missing dependency
raises :class:`~midojo.capabilities.MissingCapabilityError` naming the runtime
and the packages to install. Reading a suite or an episode imports none of it. The orchestrator drives every runtime through the :class:`AgentRuntime`
lifecycle. The world state the agent's tools read and write is the suite's
``environment``, which the control plane keeps whatever the runtime.

Each evaluation's ``setup`` gets what the agent may hold: the evaluation's
session token when an interception layer in the agent records its calls, or,
in host mode, only the tool endpoint of the interceptor the host runs
(:class:`~midojo.tool_endpoint.ToolEndpoint`), never both.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from midojo.capabilities import import_provider
from midojo.suite_definition import AgentRuntimeDefinition, OpenShellRuntimeDefinition
from midojo.tool_endpoint import ToolEndpoint

if TYPE_CHECKING:
    from midojo.runtimes import openshell
    from midojo.runtimes.openshell import OpenShellRuntime


class AgentRuntime(Protocol):
    """The lifecycle the orchestrator drives around a benchmark run and each evaluation."""

    name: str

    def start_run(self, run_id: str, *, suite_name: str) -> str | None:
        """Open the run's resources. Returns a description of them for the run log, or None."""
        ...

    def setup(
        self,
        injections: dict[str, str],
        *,
        session_token: str | None = None,
        tool_endpoint: ToolEndpoint | None = None,
        eval_id: str,
        user_task_id: str | None,
        injection_task_id: str | None,
    ) -> None:
        """Prepare the agent for one evaluation, with the active injections in place.

        The agent is given ``session_token`` (in-agent interception) or
        ``tool_endpoint`` (host mode), whichever is set. ``user_task_id`` is
        None for an episode no suite compiled.
        """
        ...

    def observe(self) -> dict[str, BaseModel]:
        """The evaluation's runtime observations, keyed by source."""
        ...

    def teardown(self) -> None:
        """Release the evaluation's resources, including after a failed ``setup``."""
        ...

    def end_run(self) -> None:
        """Release the run's resources."""
        ...


@runtime_checkable
class ResettableRuntime(Protocol):
    """A runtime that can return its target to the initial state: a capability it declares by implementing ``reset``.

    Nothing resets a target implicitly: the episode context calls ``reset``
    only when its caller asks (``open_episode(..., reset=True)``) and the
    runtime declares it, before the evaluation's ``setup``. Neither built-in
    runtime declares it: an OpenShell sandbox is new for each evaluation, but
    what the agent changed outside it (a remote account, say) isn't undone.
    """

    def reset(self) -> None:
        """Return the target to its initial state, before an episode."""
        ...


@runtime_checkable
class DisposableRuntime(Protocol):
    """A runtime whose every evaluation gets a fresh target: a capability it declares with ``disposable = True``.

    :class:`~midojo.runtimes.openshell.OpenShellRuntime` declares it: each
    evaluation's sandbox is created in ``setup`` and deleted in ``teardown``,
    and the world state the agent's tools read is the control plane's, fresh
    per evaluation. What the agent changed outside both (a remote account, say)
    isn't undone; a runtime whose target keeps state between evaluations (an
    unmanaged agent) doesn't declare it.
    """

    disposable: bool


def attempt_isolation(runtime: object) -> Literal["reset", "disposable"] | None:
    """How evaluations on ``runtime`` can be kept from sharing state: its declared reset, a fresh target each, or None.

    A declared reset (:class:`ResettableRuntime`) comes first: it returns the
    whole target to its initial state, beyond the evaluation's own sandbox.
    """
    if isinstance(runtime, ResettableRuntime):
        return "reset"
    if is_disposable(runtime):
        return "disposable"
    return None


def is_disposable(runtime: object) -> bool:
    """Whether ``runtime`` declares a fresh target for every evaluation (:class:`DisposableRuntime`)."""
    return isinstance(runtime, DisposableRuntime) and runtime.disposable is True


class UnmanagedRuntime:
    """An agent that MiDojo neither starts nor observes.

    ``midojo-run`` reaches it at ``--agent-uri`` over ``--protocol``, and each task
    carries the evaluation session, or in host mode the tool endpoint (see
    :mod:`midojo.agent_client`). Every lifecycle step is a no-op.
    """

    name = "unmanaged"

    def start_run(self, run_id: str, *, suite_name: str) -> None:
        return None

    def setup(
        self,
        injections: dict[str, str],
        *,
        session_token: str | None = None,
        tool_endpoint: ToolEndpoint | None = None,
        eval_id: str,
        user_task_id: str | None,
        injection_task_id: str | None,
    ) -> None:
        pass

    def observe(self) -> dict[str, BaseModel]:
        return {}

    def teardown(self) -> None:
        pass

    def end_run(self) -> None:
        pass


RuntimeBuilder = Callable[[str, AgentRuntimeDefinition], "AgentRuntime"]
"""Builds a runtime for a suite (or episode) name from its definition, a private copy the runtime may keep."""

# The packages the OpenShell runtime needs, and the modules they provide.
OPENSHELL_PACKAGES = ("openshell", "grpcio", "protobuf")
_OPENSHELL_DEPENDENCIES = ("openshell", "grpc", "google.protobuf")


def _openshell_module() -> ModuleType:
    """The OpenShell runtime's module, imported on first use."""
    return import_provider(
        "midojo.runtimes.openshell",
        capability="the 'openshell' runtime",
        packages=OPENSHELL_PACKAGES,
        dependencies=_OPENSHELL_DEPENDENCIES,
    )


def _openshell_runtime_type() -> type[OpenShellRuntime]:
    """The OpenShell runtime's class, its module imported on first use."""
    return _openshell_module().OpenShellRuntime


def _build_openshell(suite_name: str, definition: AgentRuntimeDefinition) -> AgentRuntime:
    if not isinstance(definition, OpenShellRuntimeDefinition):
        raise TypeError(f"the 'openshell' runtime needs an OpenShellRuntimeDefinition, not {type(definition).__name__}")
    runtime_type = _openshell_runtime_type()
    return runtime_type(
        suite_name,
        image=definition.image,
        policy=definition.policy,
        providers=definition.providers,
        env_vars=definition.env_vars,
        agent_command=definition.agent_command,
        files=definition.files,
        agent_timeout_seconds=definition.timeout_seconds,
        inject_session=definition.inject_session,
        require_policy_allows=definition.require_policy_allows,
    )


def _build_unmanaged(suite_name: str, definition: AgentRuntimeDefinition) -> AgentRuntime:
    return UnmanagedRuntime()


class RuntimeRegistry:
    """The builder for each ``agent_runtime`` type (``openshell``, ``unmanaged``).

    :func:`build_runtime` (and ``new_runtime`` on a suite or an episode)
    builds through one: :data:`DEFAULT_RUNTIMES` unless the caller passes
    its own, for instance a :meth:`copy` of the default with one builder
    replaced. A builder may import its provider lazily; see
    :func:`~midojo.capabilities.import_provider`.
    """

    def __init__(self, builders: Mapping[str, RuntimeBuilder] | None = None) -> None:
        self._builders: dict[str, RuntimeBuilder] = dict(builders or {})

    def register(self, runtime_type: str, builder: RuntimeBuilder, *, replace: bool = False) -> None:
        """Build runtimes of ``runtime_type`` with ``builder``; ValueError if it has one, unless ``replace``."""
        if runtime_type in self._builders and not replace:
            raise ValueError(f"Runtime type {runtime_type!r} is already registered")
        self._builders[runtime_type] = builder

    def __contains__(self, runtime_type: object) -> bool:
        return runtime_type in self._builders

    def copy(self) -> RuntimeRegistry:
        """A registry with the same builders, changed apart from this one."""
        return RuntimeRegistry(self._builders)

    def build(self, suite_name: str, definition: AgentRuntimeDefinition) -> AgentRuntime:
        """A fresh runtime for ``definition`` (see :func:`build_runtime`)."""
        builder = self._builders.get(definition.type)
        if builder is None:
            raise ValueError(f"No runtime is registered for agent_runtime type {definition.type!r}")
        return builder(suite_name, definition.model_copy(deep=True))


DEFAULT_RUNTIMES = RuntimeRegistry({"openshell": _build_openshell, "unmanaged": _build_unmanaged})
"""The registry :func:`build_runtime` uses when it is given none."""


def register_runtime(runtime_type: str, builder: RuntimeBuilder, *, replace: bool = False) -> None:
    """Register ``builder`` for ``runtime_type`` in :data:`DEFAULT_RUNTIMES`."""
    DEFAULT_RUNTIMES.register(runtime_type, builder, replace=replace)


def build_runtime(
    suite_name: str, definition: AgentRuntimeDefinition, *, registry: RuntimeRegistry | None = None
) -> AgentRuntime:
    """A fresh runtime for ``definition``: each call builds a new one that shares no state with another.

    The runtime gets its own copy of the definition's settings, so nothing it
    keeps is reachable from the definition or from another runtime. It is
    built by ``registry``'s builder for the definition's type, or
    :data:`DEFAULT_RUNTIMES`'s. Building an ``openshell`` runtime imports the
    OpenShell runtime, and raises
    :class:`~midojo.capabilities.MissingCapabilityError` when a package it
    needs isn't installed.
    """
    return (registry if registry is not None else DEFAULT_RUNTIMES).build(suite_name, definition)


# Loaded on first use: the OpenShell runtime's class, and its module.
_LAZY = {"OpenShellRuntime": _openshell_runtime_type, "openshell": _openshell_module}


def __getattr__(name: str) -> Any:
    # ``midojo.runtimes.OpenShellRuntime`` (and ``.openshell``) stay importable, without importing the OpenShell
    # runtime up front: a named import or an attribute read loads it, and so does a star import, which lists both.
    load = _LAZY.get(name)
    if load is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return load()


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


# What ``from midojo.runtimes import *`` exports: every public name it exported when this module imported the
# OpenShell runtime up front (those it imports included, so a star import still binds them), and the registry.
__all__ = [
    "DEFAULT_RUNTIMES",
    "OPENSHELL_PACKAGES",
    "AgentRuntime",
    "AgentRuntimeDefinition",
    "BaseModel",
    "DisposableRuntime",
    "Literal",
    "OpenShellRuntime",
    "OpenShellRuntimeDefinition",
    "Protocol",
    "ResettableRuntime",
    "RuntimeBuilder",
    "RuntimeRegistry",
    "ToolEndpoint",
    "UnmanagedRuntime",
    "annotations",
    "attempt_isolation",
    "build_runtime",
    "is_disposable",
    "openshell",
    "register_runtime",
    "runtime_checkable",
]
