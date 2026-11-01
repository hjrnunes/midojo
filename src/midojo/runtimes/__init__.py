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

A runtime is also a declared need: :func:`runtime_capabilities` answers what a
definition's runtime can do (whether it resets, whether each evaluation gets a
fresh target, what it observes) without building it or importing its provider,
and a registry's ``check_available`` names an uninstalled provider package as a
gap (``runtime:openshell:missing-package``) before anything binds.

Each evaluation's ``setup`` gets what the agent may hold: the evaluation's
session token when an interception layer in the agent records its calls, or,
in host mode, only the tool endpoint of the interceptor the host runs
(:class:`~midojo.tool_endpoint.ToolEndpoint`), never both.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from midojo.capabilities import MissingCapabilityError, import_provider, runtime_missing_package_gap
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


@dataclass(frozen=True)
class RuntimeCapabilities:
    """What a runtime can do, answered from its definition: nothing built, no sandbox, no provider imported.

    ``requires_packages`` are the distributions whose absence blocks the
    runtime (the provider names every package to install in its error);
    ``observations`` the evidence parts the runtime collects, by source;
    ``resettable`` and ``disposable`` say how it keeps one evaluation's target
    from carrying into the next, as :func:`attempt_isolation` decides it.
    ``isolates_attempts`` is whether it isolates at all: resettable or
    disposable.
    """

    name: str
    requires_packages: tuple[str, ...] = ()
    resettable: bool = False
    disposable: bool = False
    observations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a runtime capability set names its runtime")
        for field in ("requires_packages", "observations"):
            values = tuple(getattr(self, field))
            if not all(isinstance(value, str) and value for value in values):
                raise TypeError(f"a runtime's {field} are non-empty names")
            object.__setattr__(self, field, values)

    @property
    def isolates_attempts(self) -> bool:
        """Whether evaluations on this runtime are isolated: it resets its target, or each gets a fresh one."""
        return self.resettable or self.disposable


# What each built-in ``agent_runtime`` type declares, read without calling its builder: building ``openshell``
# imports its provider, and answering what it needs must not.
_BUILTIN_CAPABILITIES: dict[str, RuntimeCapabilities] = {
    "unmanaged": RuntimeCapabilities(name="unmanaged"),
    "openshell": RuntimeCapabilities(
        name="openshell",
        requires_packages=("openshell",),
        disposable=True,
        observations=("openshell",),
    ),
}


def _capabilities_of(runtime: object, *, default_name: str) -> RuntimeCapabilities:
    """What ``runtime`` (a built runtime or its class) declares about itself, ``default_name`` when it names none."""
    declared = getattr(runtime, "capabilities", None)
    if isinstance(declared, RuntimeCapabilities):
        return declared
    name = getattr(runtime, "name", None)
    if not isinstance(name, str) or not name:
        name = default_name
    builtin = _BUILTIN_CAPABILITIES.get(name)
    requires = getattr(runtime, "requires_packages", None)
    observations = getattr(runtime, "observations", None)
    return RuntimeCapabilities(
        name=name,
        requires_packages=tuple(requires) if requires is not None else (builtin.requires_packages if builtin else ()),
        resettable=isinstance(runtime, ResettableRuntime),
        disposable=is_disposable(runtime),
        observations=tuple(observations) if observations is not None else (builtin.observations if builtin else ()),
    )


def _builder_runtime(builder: object) -> object | None:
    """The runtime a builder will produce, when it names one: the class it builds, or a class it captures."""
    if isinstance(builder, type):
        return builder
    if isinstance(builder, functools.partial):
        for candidate in (builder.func, *builder.args, *(builder.keywords or {}).values()):
            found = _runtime_class(candidate)
            if found is not None:
                return found
    candidates: list[object] = list(getattr(builder, "__defaults__", None) or ())
    for cell in getattr(builder, "__closure__", None) or ():
        try:
            candidates.append(cell.cell_contents)
        except ValueError:  # an empty cell: the name it will hold is not bound yet.
            continue
    for candidate in candidates:
        found = _runtime_class(candidate)
        if found is not None:
            return found
    return None


def _runtime_class(candidate: object) -> type | None:
    """``candidate`` when it is a class the runtime protocol could describe, else None."""
    if isinstance(candidate, type) and (hasattr(candidate, "setup") or hasattr(candidate, "name")):
        return candidate
    return None


def runtime_capabilities(runtime: object, *, registry: RuntimeRegistry | None = None) -> RuntimeCapabilities:
    """What ``runtime`` can do, from its definition: no runtime built, no sandbox created, no provider imported.

    ``runtime`` is an ``agent_runtime`` definition (a mapping such as
    ``{"type": "openshell"}`` or a parsed
    :data:`~midojo.suite_definition.AgentRuntimeDefinition`), or a runtime
    itself. A definition answers from what its type declares, looked up in
    ``registry`` (:data:`DEFAULT_RUNTIMES` unless given), which calls no
    builder: a type registered with a runtime class answers from that class,
    one whose builder declares ``capabilities`` from them, one whose builder
    captures the class it builds from that class, and the built-in types from
    :data:`_BUILTIN_CAPABILITIES`. A registered type none of those describe
    answers as a runtime that needs nothing, resets nothing, is not
    disposable and observes nothing.
    """
    runtime_type = _declared_type(runtime)
    if runtime_type is not None:
        return (registry if registry is not None else DEFAULT_RUNTIMES).capabilities_for(runtime_type)
    return _capabilities_of(runtime, default_name=type(runtime).__name__)


def _declared_type(runtime: object) -> str | None:
    """The ``type`` a definition names, or None when ``runtime`` is not a definition."""
    if isinstance(runtime, Mapping):
        value = runtime.get("type", "openshell")  # the same default as midojo.suite_definition._runtime_type
    elif isinstance(runtime, BaseModel):
        value = getattr(runtime, "type", None)
    else:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"an agent_runtime type is a non-empty string, not {value!r}")
    return value


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

    def __init__(
        self,
        builders: Mapping[str, RuntimeBuilder] | None = None,
        *,
        capabilities: Mapping[str, RuntimeCapabilities] | None = None,
        providers: Mapping[str, Callable[[], object]] | None = None,
    ) -> None:
        self._builders: dict[str, RuntimeBuilder] = dict(builders or {})
        self._capabilities: dict[str, RuntimeCapabilities] = dict(capabilities or {})
        self._providers: dict[str, Callable[[], object]] = dict(providers or {})

    def register(
        self,
        runtime_type: str,
        builder: RuntimeBuilder,
        *,
        replace: bool = False,
        capabilities: RuntimeCapabilities | None = None,
        provider: Callable[[], object] | None = None,
    ) -> None:
        """Build runtimes of ``runtime_type`` with ``builder``; ValueError if it has one, unless ``replace``.

        ``capabilities`` are what the type declares, when the builder can't
        say (see :func:`runtime_capabilities`); ``provider`` imports what
        :meth:`check_available` needs to be sure the type can be built here.
        """
        if runtime_type in self._builders and not replace:
            raise ValueError(f"Runtime type {runtime_type!r} is already registered")
        self._builders[runtime_type] = builder
        if capabilities is not None:
            self._capabilities[runtime_type] = capabilities
        if provider is not None:
            self._providers[runtime_type] = provider

    def __contains__(self, runtime_type: object) -> bool:
        return runtime_type in self._builders

    def copy(self) -> RuntimeRegistry:
        """A registry with the same builders, changed apart from this one."""
        return RuntimeRegistry(self._builders, capabilities=self._capabilities, providers=self._providers)

    def capabilities_for(self, runtime_type: str) -> RuntimeCapabilities:
        """What runtimes of ``runtime_type`` can do, answered without building one (see :func:`runtime_capabilities`)."""
        declared = self._capabilities.get(runtime_type)
        if declared is not None:
            return declared
        builder = self._builders.get(runtime_type)
        if builder is None:
            raise ValueError(f"No runtime is registered for agent_runtime type {runtime_type!r}")
        on_builder = getattr(builder, "capabilities", None)
        if isinstance(on_builder, RuntimeCapabilities):
            return on_builder
        runtime = _builder_runtime(builder)
        if runtime is not None:
            return _capabilities_of(runtime, default_name=runtime_type)
        builtin = _BUILTIN_CAPABILITIES.get(runtime_type)
        return builtin if builtin is not None else RuntimeCapabilities(name=runtime_type)

    def check_available(self, runtime_type: str) -> None:
        """Raise :class:`~midojo.capabilities.MissingCapabilityError` when this machine can't build ``runtime_type``.

        The error names the runtime, the packages to install, and the gap
        (``runtime:openshell:missing-package``). It imports the type's
        provider but builds no runtime: nothing is started, bound or held.
        """
        capabilities = self.capabilities_for(runtime_type)
        provider = self._providers.get(runtime_type, _PROVIDERS.get(runtime_type))
        if provider is None or not capabilities.requires_packages:
            return
        try:
            provider()
        except MissingCapabilityError as error:
            raise MissingCapabilityError(
                error.capability,
                error.packages,
                gap=runtime_missing_package_gap(runtime_type),
                missing=error.name,
            ) from error

    def build(self, suite_name: str, definition: AgentRuntimeDefinition) -> AgentRuntime:
        """A fresh runtime for ``definition`` (see :func:`build_runtime`)."""
        builder = self._builders.get(definition.type)
        if builder is None:
            raise ValueError(f"No runtime is registered for agent_runtime type {definition.type!r}")
        return builder(suite_name, definition.model_copy(deep=True))


DEFAULT_RUNTIMES = RuntimeRegistry({"openshell": _build_openshell, "unmanaged": _build_unmanaged})
"""The registry :func:`build_runtime` uses when it is given none."""

# How a registered type's provider is imported for ``check_available``: the built-in types only; a custom type
# passes ``provider`` to ``register``.
_PROVIDERS: dict[str, Callable[[], object]] = {"openshell": _openshell_module}


def register_runtime(
    runtime_type: str,
    builder: RuntimeBuilder,
    *,
    replace: bool = False,
    capabilities: RuntimeCapabilities | None = None,
    provider: Callable[[], object] | None = None,
) -> None:
    """Register ``builder`` for ``runtime_type`` in :data:`DEFAULT_RUNTIMES` (see :meth:`RuntimeRegistry.register`)."""
    DEFAULT_RUNTIMES.register(runtime_type, builder, replace=replace, capabilities=capabilities, provider=provider)


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
    "RuntimeCapabilities",
    "RuntimeRegistry",
    "ToolEndpoint",
    "UnmanagedRuntime",
    "annotations",
    "attempt_isolation",
    "build_runtime",
    "is_disposable",
    "openshell",
    "register_runtime",
    "runtime_capabilities",
    "runtime_checkable",
]
