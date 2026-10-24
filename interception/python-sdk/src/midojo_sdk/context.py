"""The context an intercepted tool receives."""

from __future__ import annotations

import warnings
from collections.abc import Awaitable, Callable
from typing import Any

from midojo_sdk.recorder import EnvironmentRevision, WorldState

Forward = Callable[[str, dict[str, Any]], Awaitable[str]]


class ToolContext:
    """Async access to the evaluation's simulated environment, and to the real tool.

    ``world`` is optional: a context for a tool that only forwards or records
    has none, and reading or changing the environment then raises. Changing
    the environment is atomic on its own, never together with the tool's
    record or with what a forwarded call did to the real tool's system.

    ``client`` is the deprecated name of ``world``.
    """

    def __init__(
        self, world: WorldState | None = None, forward: Forward | None = None, *, client: WorldState | None = None
    ) -> None:
        if client is not None:
            if world is not None:
                raise TypeError("ToolContext takes world or client, not both: client is the deprecated name of world.")
            warnings.warn(
                "ToolContext(client=...) is deprecated: pass the WorldState as world.", DeprecationWarning, stacklevel=2
            )
            world = client
        self._world = world
        self._forward = forward

    async def env(self, field: str) -> Any:
        """The current value of one field of the environment."""
        return (await self.read()).environment[field]

    async def read(self) -> EnvironmentRevision:
        """The whole environment and its revision, for a later ``env_update(..., expected_revision=)``."""
        return await self._require_world().read()

    async def env_update(self, field: str, value: Any, *, expected_revision: int | None = None) -> bool:
        """Set one field in one atomic step: no concurrent change to another field is lost.

        Without ``expected_revision`` the change always applies, so a value
        computed from an earlier read can overwrite a concurrent change to the
        same field. With the revision that read returned, the change applies
        only if nothing changed the environment since, and returns False
        otherwise (read again and recompute).
        """
        return await self.change({field: value}, expected_revision=expected_revision)

    async def change(self, values: dict[str, Any], *, expected_revision: int | None = None) -> bool:
        """Set several fields in one atomic step: all of them change, or none.

        ``expected_revision`` works as in :meth:`env_update`: with the revision
        a read returned, the change applies only if nothing changed the
        environment since, and returns False otherwise.
        """
        changed = await self._require_world().change(values, expected_revision=expected_revision)
        return changed is not None

    async def forward(self, tool_name: str, args: dict[str, Any]) -> str:
        """Forward a tool call to the real tool."""
        if self._forward is None:
            raise RuntimeError(f"Cannot forward {tool_name}: no upstream is configured.")
        return await self._forward(tool_name, args)

    def _require_world(self) -> WorldState:
        if self._world is None:
            raise RuntimeError("This tool has no environment: no WorldState is configured.")
        return self._world
