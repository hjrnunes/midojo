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
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Sequence
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

from midojo_sdk import placement
from midojo_sdk.client import AgentControlPlaneClient, transient
from midojo_sdk.context import ToolContext
from midojo_sdk.injection import (
    CapabilityError,
    CapabilityGap,
    Channel,
    InjectionMode,
    InjectionOutcome,
    InjectionPlan,
    InterceptorCapability,
    MismatchPolicy,
    SelectorSurface,
    Transformation,
    capability_gaps,
)
from midojo_sdk.recorder import (
    Disposition,
    InjectionRecord,
    Recorder,
    RetryingRecorder,
    WorldState,
    finish_detailed,
    reporting,
)
from midojo_sdk.session import SESSION_HEADER, session_token

__all__ = ["BLOCKED_REASON", "MIDOJO_MCP_CAPABILITY", "MidojoMCP", "ToolContext", "UpstreamClient"]

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

MIDOJO_MCP_CAPABILITY = InterceptorCapability(
    channels=frozenset({Channel.TOOL_OUTPUT}),
    selector_surfaces=frozenset({SelectorSurface.CONTENT_BLOCK, SelectorSurface.STRUCTURED_CONTENT}),
    content_block_types=frozenset({"text"}),
    modes=frozenset(InjectionMode),
    transformations=frozenset(Transformation),
    mismatch_policies=frozenset(MismatchPolicy),
    call_arguments=True,
    result_record=True,
)
"""Placements the MCP forwarding interceptor implements for mirrored tool results without degradation."""


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
    runs, so it's recorded once. A start or finish that fails for good
    (after its resends, or refused other than as a late fact) is reported as
    a capture failure, which marks the evaluation's ``function_calls``
    incomplete (see ``ReportingRecorder``), and its error is raised as
    before. A record, a simulated tool's environment changes and a forwarded
    call's effect on the real tool's system are separate: none of them waits
    for or undoes another.

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

    ``plan`` is the injection plan whose placements a forwarded call's result
    gets, and ``engine`` decides them: a ``midojo_sdk.placement.PlacementEngine``
    of your own, in place of the plan's own ``PlanPlacementEngine``. A server
    takes one or the other, not both: ``plan`` builds the default engine. The
    interceptor holds no plan state of its own; it forwards a call, applies
    the engine's finished result, and records, so an engine sees every call's
    start and its result and reports the call's outcomes.
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
        plan: InjectionPlan = InjectionPlan(),
        evaluation_id: str | None = None,
        engine: placement.PlacementEngine | None = None,
    ) -> None:
        if control_plane_url is None and recorder is None:
            raise ValueError("MidojoMCP needs a control_plane_url or a recorder.")
        if type(plan) is not InjectionPlan:
            raise TypeError("plan must be an InjectionPlan")
        if engine is not None and plan:
            raise ValueError("MidojoMCP takes a plan or a placement engine, not both.")
        if evaluation_id is not None and (type(evaluation_id) is not str or not evaluation_id):
            raise ValueError("evaluation_id must be a non-empty str or None.")
        if recorder is None and evaluation_id is not None:
            raise ValueError("evaluation_id is only used with an injected recorder.")
        if recorder is not None and plan and evaluation_id is None:
            raise ValueError("A non-empty plan with an injected recorder needs an evaluation_id.")
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
        self._evaluation_id = evaluation_id
        self._engine: placement.PlacementEngine = (
            placement.PlanPlacementEngine(plan, evaluation_id=evaluation_id) if engine is None else engine
        )
        self._validate_plan()
        if self.injection_plan and self._upstream is None:
            raise CapabilityError(
                tuple(
                    CapabilityGap(index, "delivery", "mirrored_upstream") for index, _ in enumerate(self.injection_plan)
                )
            )
        self._blocked: set[str] = set()
        self._denials: dict[str, Denial] = {}
        self._pass_protocol_errors()
        self._connect_upstream_per_session()

    @property
    def injection_capability(self) -> InterceptorCapability:
        """The exact plan vocabulary this interceptor supports on mirrored results."""
        return MIDOJO_MCP_CAPABILITY

    @property
    def injection_plan(self) -> InjectionPlan:
        """The plan whose placements the engine applies, when the engine is one built from a plan; else empty."""
        engine = self._engine
        return engine.plan if isinstance(engine, placement.Planned) else InjectionPlan()

    @property
    def injection_engine(self) -> placement.PlacementEngine:
        """What decides a forwarded call's placements."""
        return self._engine

    @property
    def injection_outcomes(self) -> tuple[InjectionOutcome, ...]:
        """Reported outcomes, ordered by call admission then instruction, as the engine recorded them."""
        return self._engine.outcomes()

    def _validate_plan(self, *, tools: Collection[str] | None = None) -> None:
        if gaps := capability_gaps(self.injection_plan, self.injection_capability, tools=tools):
            raise CapabilityError(gaps)

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
        """The recorder for the request being handled, reporting the records it loses, and its environment."""
        client = None
        if self._recorder is None or (world and self._world is None and self._control_plane_url is not None):
            client = self._session_client()
        recorder = self._recorder
        if recorder is None:
            assert client is not None
            recorder = RetryingRecorder(client, retryable=transient)
        recorder = reporting(recorder)
        if not world:
            return recorder, None
        return recorder, self._world or client

    def tool(self):
        def decorator(fn):
            gaps = tuple(
                CapabilityGap(index, "delivery", f"simulated_tool:{fn.__name__}")
                for index, instruction in enumerate(self.injection_plan)
                if instruction.target.tool in (None, fn.__name__)
            )
            if gaps:
                raise CapabilityError(gaps)
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

    def _scope(self) -> str:
        """The counter scope of the request being handled: its evaluation, or the session it arrives in."""
        if self._recorder is not None:
            return self._evaluation_id or ""
        incoming = get_http_headers().get(SESSION_HEADER.lower())
        return session_token(incoming)

    async def mirror_upstream(self, *, allowed: Collection[str] | None = None) -> list[str]:
        """Expose every upstream tool not already defined here, forwarded and recorded; return their names.

        Each mirrored tool is listed exactly as the upstream lists it: name,
        title, description, input and output schemas, annotations and _meta.
        A call is forwarded with its arguments as sent, and the agent gets the
        upstream's result as it was sent: content blocks, structured content,
        _meta and error flag. A JSON-RPC error from the upstream reaches the
        agent as the same JSON-RPC error. ``allowed`` mirrors only the tools it
        names, and a name the upstream doesn't have is not an error (the
        policy may name a tool this upstream version dropped); None mirrors
        every one. Raises when there's no upstream or it can't be listed: a
        mirror never starts empty by accident.
        """
        if self._upstream is None:
            raise RuntimeError("Cannot mirror: no upstream is configured.")
        defined = {tool.name for tool in await self._fastmcp.list_tools()}
        definitions = await self._upstream.list_tools()
        self._validate_plan(tools={definition.name for definition in definitions if definition.name not in defined})
        exposed = set(allowed) if allowed is not None else {definition.name for definition in definitions}
        mirrored = []
        for definition in definitions:
            if definition.name in defined or definition.name not in exposed:
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

    async def tool_names(self) -> list[str]:
        """The tools this server lists, in name order: its own, then the mirrored ones."""
        return sorted(tool.name for tool in await self._fastmcp.list_tools())

    async def _forward_call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        assert self._upstream is not None
        recorder, _ = self._bindings(world=False)
        await self._refuse(recorder, name, arguments)
        call_id = str(uuid.uuid4())
        await recorder.start_call(call_id=call_id, function=name, args=arguments, disposition="forwarded")
        call = placement.CallView(tool=name, arguments=arguments, call_id=call_id, scope=self._scope())
        decision = self._engine.decide(call, None)
        try:
            result = await self._upstream.call_tool_mcp(name, arguments)
        except McpError as e:
            outcomes = self._engine.settle(decision, placement.EmittedView(call, None))
            message = e.error.message
            await _finish_forwarded(recorder, call_id, name, arguments, message, message, None, outcomes)
            errors = _protocol_errors.get()
            if errors is not None:
                errors.append(e)
            raise
        except Exception as e:
            outcomes = self._engine.settle(decision, placement.EmittedView(call, None))
            await _finish_forwarded(recorder, call_id, name, arguments, str(e), str(e), None, outcomes)
            raise
        except BaseException:
            # Cancellation leaves the started disposition record unfinished. The call reached the upstream, so
            # the engine settles it as one that produced no result.
            self._engine.settle(decision, placement.EmittedView(call, None))
            raise
        decision = self._engine.decide(call, _view(result))
        result = _applied(result, decision)
        outcomes = self._engine.settle(decision, placement.EmittedView(call, _view(result)))
        # The record is what the interceptor returns, recorded before it does: its text blocks as one string,
        # and its structured content. It doesn't show the agent received it.
        seen = _text(result.content)
        error = seen if result.isError else None
        await _finish_forwarded(recorder, call_id, name, arguments, seen, error, result.structuredContent, outcomes)
        forwarded = _UpstreamResult(
            content=result.content, structured_content=result.structuredContent, meta=result.meta
        )
        forwarded._result = result
        return forwarded

    def http_app(self, path: str = "/") -> Any:
        return self._fastmcp.http_app(path=path)

    def run(self, **kwargs) -> None:
        self._fastmcp.run(**kwargs)


async def _finish_forwarded(
    recorder: Recorder,
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    result: str,
    error: str | None,
    structured_content: dict[str, Any] | None,
    outcomes: Sequence[InjectionOutcome],
) -> None:
    """Finish a forwarded call with its details, or with its text only when ``recorder`` can't record them.

    A recorder without the capability (see ``DetailedRecorder``) never fails
    the call over it: the agent still gets its result, and the record, with
    no ``injection_outcomes``, says the details weren't reported. Every such
    fallback logs a warning, even for a text-only result no instruction
    matched: its record still loses the empty outcomes that say so.
    """
    detailed = await finish_detailed(
        recorder,
        call_id=call_id,
        function=name,
        args=arguments,
        result=result,
        disposition="forwarded",
        error=error,
        structured_content=structured_content,
        injection_outcomes=_records(outcomes),
    )
    if not detailed:
        logger.warning(
            "%s can't record a finish's details: %s (%s) is recorded with its text only, without its structured "
            "content (%s) or its %d injection outcome(s)",
            type(recorder).__name__,
            name,
            call_id,
            "none" if structured_content is None else "present",
            len(outcomes),
        )


def _text(content: Sequence[ContentBlock]) -> str:
    """What the agent is sent, as one string: text blocks as sent, other blocks as their wire JSON."""
    return "\n".join(
        block.text if isinstance(block, TextContent) else block.model_dump_json(by_alias=True, exclude_none=True)
        for block in content
    )


def _view(result: CallToolResult) -> placement.ResultView:
    """``result`` as the engine reads it: its content blocks, its structured content and its error flag.

    Each block keeps the wire block it came from, so applying a decision
    writes over the blocks the upstream sent and builds only the ones an
    engine's placement added.
    """
    content: list[placement.TextContent | placement.ContentBlock] = []
    for block in result.content:
        if isinstance(block, TextContent):
            content.append(placement.TextContent(text=block.text, wire=block))
        else:
            content.append(placement.ContentBlock(type=block.type, wire=block))
    return placement.ResultView(content=content, structured_content=result.structuredContent, is_error=result.isError)


def _applied(result: CallToolResult, decision: placement.PlacementDecision) -> CallToolResult:
    """``result`` with the engine's finished result on the wire, when the engine replaced it."""
    if decision.kind != "replace" or decision.result is None:
        return result
    content: list[ContentBlock] = []
    for block in decision.result.content:
        if not isinstance(block, placement.TextContent):
            content.append(block.wire)
        elif block.wire is None:
            content.append(TextContent(type="text", text=block.text))
        else:
            content.append(block.wire.model_copy(update={"text": block.text}))
    return result.model_copy(update={"content": content, "structuredContent": decision.result.structured_content})


def _records(outcomes: Sequence[InjectionOutcome]) -> list[InjectionRecord]:
    """A call's outcomes as its finish carries them: no scope, call ID or admission order.

    What was placed and where is sent only for a placement, so any other
    outcome's record is as it was before those fields.
    """
    records: list[InjectionRecord] = []
    for outcome in outcomes:
        record: InjectionRecord = {
            "instruction_index": outcome.instruction_index,
            "probe_key": outcome.probe_key,
            "occurrence": outcome.occurrence,
            "status": outcome.status.value,
            "detail": outcome.detail,
            "selected": outcome.selected,
            "transformation": outcome.transformation.value if outcome.transformation is not None else None,
            "emitted": [surface.value for surface in outcome.emitted],
        }
        if outcome.payload_digest is not None:
            record["payload_digest"] = outcome.payload_digest
        if outcome.placed_digest is not None:
            record["placed_digest"] = outcome.placed_digest
        if outcome.slot is not None:
            record["slot"] = outcome.slot.to_data()
        records.append(record)
    return records
