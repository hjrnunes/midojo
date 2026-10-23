"""MCP SDK — Python equivalent of the PI SDK.

Lets suite authors write standalone fake MCP servers whose calls are recorded
through a ``Recorder`` and whose simulated tools read and change the
environment through a ``WorldState`` (by default, both on the midojo control
plane), and mirror a real MCP server's tools through a forwarding layer.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextvars import ContextVar
from typing import Any

import anyio
import httpx
from anyio.abc import TaskGroup, TaskStatus
from fastmcp import Client, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_http_headers
from fastmcp.tools import Tool, ToolResult
from mcp.shared.exceptions import McpError
from mcp.types import CallToolRequest, CallToolResult, ContentBlock, TextContent
from mcp.types import Tool as MCPTool
from pydantic import PrivateAttr

from midojo_sdk.client import AgentControlPlaneClient, transient
from midojo_sdk.context import ToolContext
from midojo_sdk.recorder import Disposition, Recorder, RetryingRecorder, WorldState
from midojo_sdk.session import SESSION_HEADER, session_token

__all__ = ["BLOCKED_REASON", "MidojoMCP", "ToolContext", "UpstreamClient"]

logger = logging.getLogger(__name__)

BLOCKED_REASON = "Tool execution was blocked"
"""What the agent is told when it calls a blocked tool, the same as with the PI and LangChain SDKs."""

Denial = Callable[[dict[str, Any]], str | None]
"""Checks a call's arguments: why the call is refused, or None to let it run."""

# The upstream protocol errors raised by the tools/call request being handled.
# FastMCP turns any exception a tool raises into an error result, so the
# forwarding layer hands an upstream JSON-RPC error to its tools/call handler
# here, and the handler raises it as the same JSON-RPC error.
_protocol_errors: ContextVar[list[McpError] | None] = ContextVar("midojo_protocol_errors", default=None)


class UpstreamClient:
    """Forwards tool calls to an upstream MCP server.

    The upstream is ``upstream_url``, or ``transport``: anything else a
    fastmcp ``Client`` connects to, such as a ``FastMCP`` server in the same
    process or a client transport. Within a downstream session that
    ``MidojoMCP`` serves, every call goes over that session's one upstream
    connection; elsewhere each call opens its own.
    """

    def __init__(self, upstream_url: str | None = None, *, transport: Any = None) -> None:
        if (upstream_url is None) == (transport is None):
            raise ValueError("UpstreamClient needs exactly one of upstream_url and transport.")
        self.upstream_url = upstream_url
        self.transport = transport

    @property
    def target(self) -> Any:
        """What a fastmcp ``Client`` connects to."""
        return self.upstream_url if self.upstream_url is not None else self.transport

    @contextlib.asynccontextmanager
    async def _client(self) -> AsyncIterator[Client]:
        connection = _upstream_connection.get()
        if connection is not None and connection.upstream is self:
            yield await connection.client()
            return
        async with Client(self.target) as client:
            yield client

    async def list_tools(self) -> list[MCPTool]:
        async with self._client() as client:
            return await client.list_tools()

    async def call_tool_mcp(self, name: str, args: dict[str, Any]) -> CallToolResult:
        """The upstream's result as it sent it, error flag included. Raises ``McpError`` on a JSON-RPC error."""
        async with self._client() as client:
            return await client.call_tool_mcp(name, args)

    async def call_tool(self, name: str, args: dict) -> str:
        async with self._client() as client:
            result = await client.call_tool(name, args)

        parts = []
        for content in result.content:
            if isinstance(content, TextContent):
                parts.append(content.text)
            else:
                parts.append(str(content))
        return "\n".join(parts)


class _UpstreamConnection:
    """One downstream session's connection to the upstream: opened on first use, closed when the session ends.

    The connection lives in its own task in the session's task group, so the
    task that opens the fastmcp ``Client`` is the one that closes it.
    """

    def __init__(self, upstream: UpstreamClient, tasks: TaskGroup) -> None:
        self.upstream = upstream
        self._tasks = tasks
        self._lock = anyio.Lock()
        self._closed = anyio.Event()
        self._client: Client | None = None

    async def client(self) -> Client:
        async with self._lock:
            client = self._client
            if client is None:
                client = self._client = await self._tasks.start(self._hold)
            return client

    def close(self) -> None:
        self._closed.set()

    async def _hold(self, *, task_status: TaskStatus[Client] = anyio.TASK_STATUS_IGNORED) -> None:
        client = Client(self.upstream.target)
        await client.__aenter__()
        try:
            task_status.started(client)
            await self._closed.wait()
        finally:
            # Close the connection even when the session's task group is
            # cancelled, and never fail the downstream session over it.
            with anyio.CancelScope(shield=True):
                try:
                    await client.__aexit__(None, None, None)
                except Exception:
                    logger.exception("Closing the upstream connection failed")


# The upstream connection of the downstream session a request belongs to.
_upstream_connection: ContextVar[_UpstreamConnection | None] = ContextVar("midojo_upstream_connection", default=None)


class _ForwardingTool(Tool):
    """A mirrored upstream tool, listed as the upstream lists it and forwarded as called."""

    _definition: MCPTool = PrivateAttr()
    _forward: Callable[[str, dict[str, Any]], Awaitable[ToolResult]] = PrivateAttr()

    async def run(self, arguments: dict[str, Any]) -> ToolResult:
        return await self._forward(self.name, arguments)

    def to_mcp_tool(self, **overrides: Any) -> MCPTool:
        # The upstream's own definition rather than one rebuilt from this
        # tool's fields, which would add FastMCP's _meta. Only a rename applies.
        return self._definition.model_copy(update={"name": overrides.get("name", self.name)}, deep=True)


class _UpstreamResult(ToolResult):
    """A result the agent gets exactly as the upstream sent it, error flag included."""

    _result: CallToolResult = PrivateAttr()

    def to_mcp_result(self) -> CallToolResult:
        return self._result


class MidojoMCP:
    """Wrapper around FastMCP that adds control plane wiring.

    Usage::

        mcp = MidojoMCP("weather", control_plane_url=...)

        @mcp.tool()
        async def get_weather(ctx: ToolContext, city: str) -> str:
            cities = await ctx.env("cities")
            ...

    The ``ctx: ToolContext`` first parameter is injected by the SDK and
    stripped from the MCP tool schema exposed to agents. Such a tool simulates
    the real one: its calls are recorded as ``simulated``. ``mirror_upstream``
    exposes the upstream's other tools, whose calls are ``forwarded``;
    ``block`` and ``deny`` refuse calls, recorded as ``blocked`` or ``denied``.

    The SDK records each call's start before the tool runs and its finish
    after, with a fresh UUID ``call_id``: an MCP request ID only numbers the
    requests of one connection, so it isn't unique per evaluation. When the
    start fails the tool doesn't run; when the finish never reaches the
    control plane, the call stays there as unfinished. A refused call never
    runs, so it's recorded once. A record, a simulated tool's environment
    changes and a forwarded call's effect on the real tool's system are
    separate: none of them waits for or undoes another.

    ``recorder`` records every call, with its disposition, and ``world`` is the
    environment simulated tools get through their ``ToolContext``. Without
    ``recorder``, calls are recorded on the control plane at
    ``control_plane_url``, for the evaluation session the agent sends (the
    ``X-Midojo-Session`` header or ``MIDOJO_SESSION_TOKEN``), resending after
    a transient failure (see ``RetryingRecorder``); without ``world``, the
    environment is that session's on the same control plane, or none when
    there's no ``control_plane_url``. Forwarding (``mirror_upstream``) needs
    only a recorder: a server with ``recorder`` and no ``control_plane_url``
    mirrors and records with no environment at all.
    """

    def __init__(
        self,
        name: str,
        *,
        control_plane_url: str | None = None,
        upstream_url: str | None = None,
        http: httpx.AsyncClient | None = None,
        upstream_transport: Any = None,
        recorder: Recorder | None = None,
        world: WorldState | None = None,
    ) -> None:
        if control_plane_url is None and recorder is None:
            raise ValueError("MidojoMCP needs a control_plane_url or a recorder.")
        self._fastmcp = FastMCP(name)
        self._control_plane_url = control_plane_url
        self._http = http
        self._upstream = (
            UpstreamClient(upstream_url, transport=upstream_transport)
            if upstream_url is not None or upstream_transport is not None
            else None
        )
        self._recorder = recorder
        self._world = world
        self._blocked: set[str] = set()
        self._denials: dict[str, Denial] = {}
        self._pass_protocol_errors()
        self._connect_upstream_per_session()

    def _pass_protocol_errors(self) -> None:
        """Answer a tools/call with the JSON-RPC error a forwarded call got from the upstream."""
        handlers = self._fastmcp._mcp_server.request_handlers
        handle = handlers[CallToolRequest]

        async def call_tool(request: Any) -> Any:
            errors: list[McpError] = []
            token = _protocol_errors.set(errors)
            try:
                response = await handle(request)
            finally:
                _protocol_errors.reset(token)
            if errors:
                raise errors[0]
            return response

        handlers[CallToolRequest] = call_tool

    def _connect_upstream_per_session(self) -> None:
        """Give each downstream session its own upstream connection, as long as the session lasts.

        The low-level server runs once per downstream session (an stdio or
        in-memory connection, a stateful streamable HTTP session), so the
        connection is bound there and every request of the session reuses it.
        """
        server = self._fastmcp._mcp_server
        run = server.run

        async def run_with_upstream(*args: Any, **kwargs: Any) -> None:
            if self._upstream is None:
                await run(*args, **kwargs)
                return
            async with anyio.create_task_group() as tasks:
                connection = _UpstreamConnection(self._upstream, tasks)
                token = _upstream_connection.set(connection)
                try:
                    await run(*args, **kwargs)
                finally:
                    _upstream_connection.reset(token)
                    connection.close()

        setattr(server, "run", run_with_upstream)

    def _session_client(self) -> AgentControlPlaneClient:
        """The control plane client for the session of the request being handled."""
        assert self._control_plane_url is not None
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=300.0)
        incoming = get_http_headers().get(SESSION_HEADER.lower())
        return AgentControlPlaneClient(self._control_plane_url, session_token(incoming), http=self._http)

    def _bindings(self, *, world: bool) -> tuple[Recorder, WorldState | None]:
        """The recorder for the request being handled, and its environment when ``world`` asks for one."""
        client = None
        if self._recorder is None or (world and self._world is None and self._control_plane_url is not None):
            client = self._session_client()
        recorder = self._recorder
        if recorder is None:
            assert client is not None
            recorder = RetryingRecorder(client, retryable=transient)
        if not world:
            return recorder, None
        return recorder, self._world or client

    def tool(self):
        def decorator(fn):
            sig = inspect.signature(fn, eval_str=True)
            params = list(sig.parameters.values())
            if not params or params[0].annotation is not ToolContext:
                raise TypeError(f"First parameter of {fn.__name__} must be annotated as ToolContext")
            user_params = params[1:]
            user_sig = sig.replace(parameters=user_params)

            @functools.wraps(fn)
            async def wrapper(**kwargs):
                recorder, world = self._bindings(world=True)
                await self._refuse(recorder, fn.__name__, kwargs)
                ctx = ToolContext(world, self._upstream.call_tool if self._upstream else None)
                call_id = str(uuid.uuid4())
                await recorder.start_call(call_id=call_id, function=fn.__name__, args=kwargs, disposition="simulated")
                # Finish the call when the tool returns or raises. A cancelled
                # tool (or any other BaseException) gave the agent no result,
                # so the call stays unfinished.
                try:
                    result = await fn(ctx, **kwargs)
                except Exception as e:
                    await recorder.finish_call(
                        call_id=call_id,
                        function=fn.__name__,
                        args=kwargs,
                        result=str(e),
                        error=str(e),
                        disposition="simulated",
                    )
                    raise
                await recorder.finish_call(
                    call_id=call_id, function=fn.__name__, args=kwargs, result=result, disposition="simulated"
                )
                return result

            setattr(wrapper, "__signature__", user_sig)
            wrapper.__annotations__ = {
                p.name: p.annotation for p in user_params if p.annotation is not inspect.Parameter.empty
            }

            self._fastmcp.tool(wrapper, name=fn.__name__, description=fn.__doc__)
            return fn

        return decorator

    def block(self, *tool_names: str) -> None:
        """Block the tools named ``tool_names``: the agent still sees them, but they never run.

        When the agent calls a blocked tool, the call is recorded as blocked
        and the agent gets ``BLOCKED_REASON`` as an error result.
        """
        self._blocked.update(tool_names)

    def deny(self, tool_name: str) -> Callable[[Denial], Denial]:
        """Check each call to ``tool_name`` before it runs.

        The decorated function takes the call's arguments and returns why the
        call is refused, or None to let it run. A refused call never runs: it's
        recorded as denied and the agent gets the reason as an error result.
        """

        def decorator(check: Denial) -> Denial:
            self._denials[tool_name] = check
            return check

        return decorator

    async def _refuse(self, recorder: Recorder, name: str, args: dict[str, Any]) -> None:
        """Record the call and raise its error result when the tool is blocked or a denial check refuses it."""
        disposition: Disposition
        if name in self._blocked:
            disposition, reason = "blocked", BLOCKED_REASON
        elif (check := self._denials.get(name)) is not None and (denial := check(args)) is not None:
            disposition, reason = "denied", denial
        else:
            return
        await recorder.finish_call(
            call_id=str(uuid.uuid4()), function=name, args=args, result=reason, disposition=disposition
        )
        raise ToolError(reason)

    async def mirror_upstream(self) -> list[str]:
        """Expose every upstream tool not already defined here, forwarded and recorded; return their names.

        Each mirrored tool is listed exactly as the upstream lists it: name,
        title, description, input and output schemas, annotations and _meta.
        A call is forwarded with its arguments as sent, and the agent gets the
        upstream's result as it was sent: content blocks, structured content,
        _meta and error flag. A JSON-RPC error from the upstream reaches the
        agent as the same JSON-RPC error. Raises when there's no upstream or it
        can't be listed: a mirror never starts empty by accident.
        """
        if self._upstream is None:
            raise RuntimeError("Cannot mirror: no upstream is configured.")
        defined = {tool.name for tool in await self._fastmcp.list_tools()}
        mirrored = []
        for definition in await self._upstream.list_tools():
            if definition.name in defined:
                continue
            tool = _ForwardingTool(
                name=definition.name,
                title=definition.title,
                description=definition.description,
                parameters=definition.inputSchema,
                output_schema=definition.outputSchema,
                annotations=definition.annotations,
            )
            tool._definition = definition
            tool._forward = self._forward_call
            self._fastmcp.add_tool(tool)
            mirrored.append(definition.name)
        return mirrored

    async def _forward_call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        assert self._upstream is not None
        recorder, _ = self._bindings(world=False)
        await self._refuse(recorder, name, arguments)
        call_id = str(uuid.uuid4())
        await recorder.start_call(call_id=call_id, function=name, args=arguments, disposition="forwarded")
        try:
            result = await self._upstream.call_tool_mcp(name, arguments)
        except McpError as e:
            message = e.error.message
            await recorder.finish_call(
                call_id=call_id, function=name, args=arguments, result=message, error=message, disposition="forwarded"
            )
            errors = _protocol_errors.get()
            if errors is not None:
                errors.append(e)
            raise
        except Exception as e:
            await recorder.finish_call(
                call_id=call_id, function=name, args=arguments, result=str(e), error=str(e), disposition="forwarded"
            )
            raise
        seen = _text(result.content)
        await recorder.finish_call(
            call_id=call_id,
            function=name,
            args=arguments,
            result=seen,
            error=seen if result.isError else None,
            disposition="forwarded",
        )
        forwarded = _UpstreamResult(
            content=result.content, structured_content=result.structuredContent, meta=result.meta
        )
        forwarded._result = result
        return forwarded

    def http_app(self, path: str = "/") -> Any:
        return self._fastmcp.http_app(path=path)

    def run(self, **kwargs) -> None:
        self._fastmcp.run(**kwargs)


def _text(content: Sequence[ContentBlock]) -> str:
    """What the agent saw, as one string: text blocks as sent, other blocks as their wire JSON."""
    return "\n".join(
        block.text if isinstance(block, TextContent) else block.model_dump_json(by_alias=True, exclude_none=True)
        for block in content
    )
