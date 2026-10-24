"""LangChain SDK — intercepts the tool calls of a LangChain agent.

Lets suite authors hook, block and report the tools of a LangChain agent, with
access to the midojo control plane, the same way ``@midojo/pi-sdk`` does for a
PI agent. Calls are recorded through a ``Recorder``, and hooks read and change
the environment through a ``WorldState`` (by default, both on the midojo
control plane).
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from langchain.agents.middleware import AgentMiddleware, ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.types import Command

from midojo_sdk.client import AgentControlPlaneClient, CallIds, transient
from midojo_sdk.context import ToolContext
from midojo_sdk.recorder import ControlPlaneRejectedError, Disposition, Recorder, RetryingRecorder, WorldState
from midojo_sdk.session import session_token

__all__ = ["BLOCKED_REASON", "MidojoMiddleware", "ToolContext"]

logger = logging.getLogger(__name__)

BLOCKED_REASON = "Tool execution was blocked"
"""What the agent is told when it calls a blocked tool, the same as with the PI SDK."""

Hook = Callable[[ToolContext, dict[str, Any], str], Awaitable[str]]
Handler = Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]]
_Intercept = Callable[["_Call", Handler], Awaitable[ToolMessage | Command]]


class _Call:
    """One intercepted tool call: its request, ``call_id`` and disposition, and the seams it's recorded through."""

    def __init__(
        self,
        request: ToolCallRequest,
        call_id: str,
        disposition: Disposition,
        recorder: Recorder,
        world: WorldState | None,
    ) -> None:
        self.request = request
        self.call_id = call_id
        self.disposition: Disposition = disposition
        self.recorder = recorder
        self.world = world

    @property
    def function(self) -> str:
        return self.request.tool_call["name"]

    @property
    def args(self) -> dict[str, Any]:
        return self.request.tool_call["args"]

    async def start(self) -> None:
        await self.recorder.start_call(
            call_id=self.call_id, function=self.function, args=self.args, disposition=self.disposition
        )

    async def finish(self, result: str, error: str | None = None) -> None:
        await self.recorder.finish_call(
            call_id=self.call_id,
            function=self.function,
            args=self.args,
            result=result,
            error=error,
            disposition=self.disposition,
        )


class _Connection:
    """The HTTP client of one event loop, and the seams of each session it has served."""

    def __init__(self, http: httpx.AsyncClient) -> None:
        self.http = http
        self.sessions: dict[str, tuple[Recorder, AgentControlPlaneClient]] = {}


class MidojoMiddleware(AgentMiddleware):
    """Intercepts an agent's tool calls by tool name, with control plane wiring.

    Usage::

        midojo = MidojoMiddleware(control_plane_url=...)

        @midojo.hook("get_weather")
        async def inject_note(ctx: ToolContext, args: dict[str, Any], real_result: str) -> str:
            return real_result + await ctx.env("injected_note")

        midojo.block("send_email")
        midojo.report("read_file")

        agent = create_agent(model, tools=[get_weather, send_email, read_file], middleware=[midojo])

    The agent keeps its real tools, so it sees the same tool names,
    descriptions and arguments as without midojo. Tools that aren't hooked,
    blocked or reported run as usual and aren't recorded. For a LangGraph
    ``ToolNode``, pass ``awrap_tool_call=midojo.awrap_tool_call`` instead.
    Hooked and reported tools run the real tool, so their calls are recorded
    as ``forwarded``; a blocked tool's are ``blocked``.

    ``recorder`` records every intercepted call, and ``world`` is the
    environment hooks get through their ``ToolContext``: seams bound to one
    evaluation by whoever built them, such as the control plane's in-process
    client. Without ``recorder``, calls are recorded on the control plane at
    ``control_plane_url``, for the evaluation session of the agent run
    (``session_context()`` or ``MIDOJO_SESSION_TOKEN``), resending after a
    transient failure (see ``RetryingRecorder``); without ``world``, hooks get
    that session's environment on the same control plane, or none when there's
    no ``control_plane_url``. The HTTP client is ``http``, or one the
    middleware opens for each event loop it runs on, keeping no idle
    connection (so one middleware works across event loops, such as one
    ``asyncio.run`` per evaluation), and closes in :meth:`aclose`. Each
    session's recorder is built once and reused.

    Interception is asynchronous, so run the agent with ``ainvoke`` or
    ``astream``.

    The SDK attempts to record each intercepted call, with a ``call_id`` made
    from the tool call's ID (see ``midojo_sdk.client.CallIds``). For a hooked
    or reported tool, it records the call's start before the tool runs and its
    finish after; when the start fails the tool doesn't run, and when the
    finish never reaches the control plane the call stays there as unfinished.
    A blocked call never runs, so it's recorded once. Resends after a lost
    response create at most one record per ``call_id``; a record that never
    reaches the control plane is still missing, and the agent run stops with
    the last error.

    A refusal by the control plane (a ``ControlPlaneRejectedError``, such as
    an evaluation that's closing) is never resent and doesn't stop the agent
    run: the agent gets the refusal as an error result instead of the tool's.
    A refused start means the tool never runs.
    """

    def __init__(
        self,
        *,
        control_plane_url: str | None = None,
        http: httpx.AsyncClient | None = None,
        recorder: Recorder | None = None,
        world: WorldState | None = None,
    ) -> None:
        super().__init__()
        if control_plane_url is None and recorder is None:
            raise ValueError("MidojoMiddleware needs a control_plane_url or a recorder.")
        self._control_plane_url = control_plane_url
        self._recorder = recorder
        self._world = world
        self._shared: _Connection | None = _Connection(http) if http is not None else None
        self._per_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _Connection] = weakref.WeakKeyDictionary()
        self._intercepts: dict[str, tuple[_Intercept, Disposition]] = {}
        self._call_ids = CallIds("langchain")

    def hook(self, tool_name: str) -> Callable[[Hook], Hook]:
        """Rewrite what the tool named ``tool_name`` returns to the agent.

        The decorated function takes ``(ctx, args, real_result)``: the real tool
        runs first, and the function's return value is what the agent gets. The
        call is recorded with the rewritten result, and with the real tool's
        error when it failed, in which case the agent still gets an error
        result. When the function raises, the call is recorded with its error
        and the agent gets the error message as an error result, so the agent
        run goes on.

        Only a tool that returns a message can be hooked. When the tool
        returns a ``Command`` instead, the call is recorded and the agent run
        stops with a ``TypeError``.
        """

        def decorator(fn: Hook) -> Hook:
            async def intercept(call: _Call, handler: Handler) -> ToolMessage | Command:
                output = await _run(call, handler)
                if not isinstance(output, ToolMessage):
                    await call.finish(str(output))
                    raise TypeError(f"Cannot hook {tool_name}: it returned a {type(output).__name__}, not a message.")
                try:
                    result = await fn(ToolContext(call.world), call.args, output.text)
                except Exception as e:
                    await call.finish(str(e), str(e))
                    return _error(call.request, str(e))
                await call.finish(result, output.text if output.status == "error" else None)
                return output.model_copy(update={"content": result})

            self._add(tool_name, intercept, "forwarded")
            return fn

        return decorator

    def block(self, *tool_names: str) -> None:
        """Block the tools named ``tool_names``: the agent still sees them, but they never run.

        When the agent calls a blocked tool, the call is recorded as blocked and
        the agent gets ``BLOCKED_REASON`` as an error result. Use this for a
        tool whose effect must not happen during an evaluation, such as sending
        email, while still recording that the agent attempted it.
        """

        async def intercept(call: _Call, handler: Handler) -> ToolMessage:
            await call.finish(BLOCKED_REASON)
            return _error(call.request, BLOCKED_REASON)

        for tool_name in tool_names:
            self._add(tool_name, intercept, "blocked")

    def report(self, *tool_names: str) -> None:
        """Record every call to the tools named ``tool_names`` without changing what the agent sees.

        Each tool runs as before and the agent gets its real output, while the
        call and its result are recorded on the control plane. Use this to make
        a tool's result, such as a file read, visible to midojo without
        perturbing the agent.
        """

        async def intercept(call: _Call, handler: Handler) -> ToolMessage | Command:
            output = await _run(call, handler)
            result = output.text if isinstance(output, ToolMessage) else str(output)
            error = result if isinstance(output, ToolMessage) and output.status == "error" else None
            await call.finish(result, error)
            return output

        for tool_name in tool_names:
            self._add(tool_name, intercept, "forwarded")

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Handler) -> ToolMessage | Command:
        intercepted = self._intercepts.get(request.tool_call["name"])
        if intercepted is None:
            return await handler(request)
        intercept, disposition = intercepted
        recorder, world = self._bindings()
        call = _Call(request, self._call_ids.for_call(request.tool_call.get("id")), disposition, recorder, world)
        try:
            return await intercept(call, handler)
        except ControlPlaneRejectedError as e:
            logger.warning("The control plane refused the record of %s (%s): %s", call.function, call.call_id, e)
            return _error(request, str(e))

    async def aclose(self) -> None:
        """Close the HTTP client the middleware opened for the running event loop. A given ``http`` stays open."""
        connection = self._per_loop.pop(asyncio.get_running_loop(), None)
        if connection is not None:
            await connection.http.aclose()

    def _add(self, tool_name: str, intercept: _Intercept, disposition: Disposition) -> None:
        if tool_name in self._intercepts:
            raise ValueError(f"{tool_name} is already intercepted.")
        self._intercepts[tool_name] = (intercept, disposition)

    def _bindings(self) -> tuple[Recorder, WorldState | None]:
        """The recorder and the environment of the call being intercepted."""
        if self._recorder is not None and (self._world is not None or self._control_plane_url is None):
            return self._recorder, self._world
        recorder, client = self._session()
        return self._recorder or recorder, self._world or client

    def _session(self) -> tuple[Recorder, AgentControlPlaneClient]:
        """The resending recorder and the client of the current session, on the control plane, built once."""
        assert self._control_plane_url is not None
        token = session_token()
        connection = self._connection()
        if token not in connection.sessions:
            client = AgentControlPlaneClient(self._control_plane_url, token, http=connection.http)
            connection.sessions[token] = (RetryingRecorder(client, retryable=transient), client)
        return connection.sessions[token]

    def _connection(self) -> _Connection:
        if self._shared is not None:
            return self._shared
        loop = asyncio.get_running_loop()
        if loop not in self._per_loop:
            # No idle connection is kept: one would outlive its event loop, such as an asyncio.run per evaluation.
            http = httpx.AsyncClient(timeout=300.0, limits=httpx.Limits(max_keepalive_connections=0))
            self._per_loop[loop] = _Connection(http)
        return self._per_loop[loop]


async def _run(call: _Call, handler: Handler) -> ToolMessage | Command:
    """Run the real tool after recording the call's start, and finish the call if the tool raises.

    When the tool raises, its error is what propagates, even if the control
    plane refuses the finish.
    """
    await call.start()
    try:
        return await handler(call.request)
    except GraphBubbleUp:
        raise
    except Exception as e:
        try:
            await call.finish(str(e), str(e))
        except ControlPlaneRejectedError as refused:
            logger.warning("The control plane refused the record of %s (%s): %s", call.function, call.call_id, refused)
        raise


def _error(request: ToolCallRequest, message: str) -> ToolMessage:
    return ToolMessage(
        content=message, name=request.tool_call["name"], tool_call_id=request.tool_call["id"], status="error"
    )
