"""Where an interception layer records each call, and what it did with the call."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from midojo_sdk.client import AgentControlPlaneClient

Disposition = Literal["forwarded", "denied", "simulated", "blocked"]
"""What the interception layer did with a call.

- ``forwarded``: sent to the real tool; the agent got the real tool's answer
- ``simulated``: answered by a tool the layer defines, instead of the real tool
- ``denied``: refused by a check on this call's arguments; the tool never ran
- ``blocked``: refused because the tool is blocked for the evaluation; the tool never ran

A denied or blocked call gives the agent an error result, never a success.
"""

REFUSED: frozenset[Disposition] = frozenset({"denied", "blocked"})
"""The dispositions of a call whose tool never ran: the record's ``blocked`` flag."""


class CallRecorder(Protocol):
    """Records one evaluation's tool calls: each call's start, then its finish with the same ``call_id``."""

    async def start_call(self, *, call_id: str, function: str, args: dict[str, Any]) -> None:
        """Record that a call started: an unfinished record, before the tool runs."""
        ...

    async def finish_call(
        self,
        *,
        call_id: str,
        function: str,
        args: dict[str, Any],
        result: str,
        disposition: Disposition,
        error: str | None = None,
    ) -> None:
        """Finish the call started with ``call_id``, or record it whole when it wasn't started."""
        ...


class ControlPlaneRecorder:
    """Records calls on the control plane's ``/agent`` API.

    The API has no field for the disposition, so the record carries only what
    the disposition implies: ``blocked`` for a denied or blocked call.
    """

    def __init__(self, client: AgentControlPlaneClient) -> None:
        self._client = client

    async def start_call(self, *, call_id: str, function: str, args: dict[str, Any]) -> None:
        await self._client.start_function_call(function=function, args=args, call_id=call_id)

    async def finish_call(
        self,
        *,
        call_id: str,
        function: str,
        args: dict[str, Any],
        result: str,
        disposition: Disposition,
        error: str | None = None,
    ) -> None:
        await self._client.record_function_call(
            function=function,
            args=args,
            result=result,
            error=error,
            blocked=disposition in REFUSED,
            call_id=call_id,
        )
