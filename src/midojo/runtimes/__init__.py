"""Agent runtimes: where an episode's agent runs and how MiDojo observes it.

A suite declares its runtime under ``agent_runtime`` in ``suite.yaml``:

  * ``openshell`` (the default) — :class:`~midojo.runtimes.openshell.OpenShellRuntime`
    creates an OpenShell sandbox for each evaluation, seeds its files, and
    collects runtime observations from outside the agent.
  * ``unmanaged`` (experimental) — :class:`UnmanagedRuntime`. The agent runs
    somewhere MiDojo doesn't control and nothing is observed outside it.

A suite or an episode holds only the runtime's definition: :func:`build_runtime`
is the factory that builds a fresh runtime from it, so concurrent runs never
share one. The orchestrator drives every runtime through the :class:`AgentRuntime`
lifecycle. The world state the agent's tools read and write is the suite's
``environment``, which the control plane keeps whatever the runtime.

Each evaluation's ``setup`` gets what the agent may hold: the evaluation's
session token when an interception layer in the agent records its calls, or,
in host mode, only the tool endpoint of the interceptor the host runs
(:class:`~midojo.tool_endpoint.ToolEndpoint`), never both.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from midojo.runtimes.openshell import OpenShellRuntime
from midojo.suite_definition import AgentRuntimeDefinition, OpenShellRuntimeDefinition
from midojo.tool_endpoint import ToolEndpoint


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


def build_runtime(suite_name: str, definition: AgentRuntimeDefinition) -> AgentRuntime:
    """A fresh runtime for ``definition``: each call builds a new one that shares no state with another.

    The runtime gets its own copy of the definition's settings, so nothing it
    keeps is reachable from the definition or from another runtime.
    """
    definition = definition.model_copy(deep=True)
    if isinstance(definition, OpenShellRuntimeDefinition):
        return OpenShellRuntime(
            suite_name,
            image=definition.image,
            policy=definition.policy,
            providers=definition.providers,
            env_vars=definition.env_vars,
            agent_command=definition.agent_command,
            files=definition.files,
            agent_timeout_seconds=definition.timeout_seconds,
        )
    return UnmanagedRuntime()
