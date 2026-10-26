"""MCP SDK — Python equivalent of the PI SDK.

Lets suite authors write standalone fake MCP servers whose calls are recorded
through a ``Recorder`` and whose simulated tools read and change the
environment through a ``WorldState`` (by default, both on the midojo control
plane), and mirror a real MCP server's tools through a forwarding layer.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import functools
import inspect
import json
import logging
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Sequence
from contextvars import ContextVar
from typing import Any, Literal

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
from midojo_sdk.injection import (
    CapabilityError,
    CapabilityGap,
    Channel,
    InjectionInstruction,
    InjectionMode,
    InjectionOutcome,
    InjectionPlan,
    InterceptorCapability,
    MismatchPolicy,
    OutcomeStatus,
    RecordSelector,
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
)
"""Placements the MCP forwarding interceptor implements for mirrored tool results without degradation."""

type _AdmittedInstruction = tuple[str, int, InjectionInstruction, int, int]


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
        if type(plan) is not InjectionPlan:
            raise TypeError("plan must be an InjectionPlan")
        self._plan = plan
        if evaluation_id is not None and (type(evaluation_id) is not str or not evaluation_id):
            raise ValueError("evaluation_id must be a non-empty str or None.")
        if recorder is None and evaluation_id is not None:
            raise ValueError("evaluation_id is only used with an injected recorder.")
        if recorder is not None and self._plan and evaluation_id is None:
            raise ValueError("A non-empty plan with an injected recorder needs an evaluation_id.")
        self._evaluation_id = evaluation_id
        self._validate_plan()
        if self._plan and self._upstream is None:
            raise CapabilityError(
                tuple(CapabilityGap(index, "delivery", "mirrored_upstream") for index, _ in enumerate(self._plan))
            )
        self._plan_lock = anyio.Lock()
        self._occurrences: dict[object, list[int]] = {}
        self._admissions: dict[object, int] = {}
        self._outcomes: list[InjectionOutcome] = []
        self._blocked: set[str] = set()
        self._denials: dict[str, Denial] = {}
        self._pass_protocol_errors()
        self._connect_upstream_per_session()

    @property
    def injection_capability(self) -> InterceptorCapability:
        """The exact plan vocabulary this interceptor supports on mirrored results."""
        return MIDOJO_MCP_CAPABILITY

    @property
    def injection_outcomes(self) -> tuple[InjectionOutcome, ...]:
        """Reported outcomes, ordered by call admission then instruction."""
        return tuple(sorted(self._outcomes, key=lambda item: (item.admission_order, item.instruction_index)))

    def _validate_plan(self, *, tools: Collection[str] | None = None) -> None:
        if gaps := capability_gaps(self._plan, self.injection_capability, tools=tools):
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
                for index, instruction in enumerate(self._plan)
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

    def _evaluation_scope(self) -> str:
        """The bound evaluation or the incoming session: one counter scope."""
        if self._recorder is not None:
            assert self._evaluation_id is not None  # A non-empty plan requires it.
            return self._evaluation_id
        incoming = get_http_headers().get(SESSION_HEADER.lower())
        return session_token(incoming)

    async def _admit_injections(
        self, scope: str, call_id: str, name: str
    ) -> tuple[list[_AdmittedInstruction], list[InjectionOutcome]]:
        """Number matches under a lock before forwarding; that is admission order.

        Returns the instructions admitted for the call, and the outcomes of
        those whose occurrence rule skipped it.
        """
        admitted: list[_AdmittedInstruction] = []
        skipped: list[InjectionOutcome] = []
        async with self._plan_lock:
            counts = self._occurrences.setdefault(scope, [0] * len(self._plan))
            order = self._admissions.get(scope, 0) + 1
            self._admissions[scope] = order
            for index, instruction in enumerate(self._plan):
                if instruction.target.tool not in (None, name):
                    continue
                counts[index] += 1
                occurrence = counts[index]
                if instruction.occurrence.selects(occurrence):
                    admitted.append((scope, index, instruction, occurrence, order))
                else:
                    skipped.append(
                        _outcome(
                            scope,
                            index,
                            instruction,
                            call_id,
                            name,
                            occurrence,
                            order,
                            OutcomeStatus.SKIPPED_OCCURRENCE,
                            "occurrence rule did not select this call",
                        )
                    )
            self._outcomes.extend(skipped)
        return admitted, skipped

    def _report_no_result(
        self, call_id: str, name: str, admitted: Sequence[_AdmittedInstruction]
    ) -> list[InjectionOutcome]:
        outcomes = [
            _outcome(
                scope,
                index,
                instruction,
                call_id,
                name,
                occurrence,
                order,
                OutcomeStatus.NO_RESULT,
                "upstream produced no tool result",
            )
            for scope, index, instruction, occurrence, order in admitted
        ]
        self._outcomes.extend(outcomes)
        return outcomes

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
        definitions = await self._upstream.list_tools()
        self._validate_plan(tools={definition.name for definition in definitions if definition.name not in defined})
        mirrored = []
        for definition in definitions:
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
        admitted, skipped = (
            await self._admit_injections(self._evaluation_scope(), call_id, name) if self._plan else ([], [])
        )
        try:
            result = await self._upstream.call_tool_mcp(name, arguments)
        except McpError as e:
            outcomes = [*skipped, *self._report_no_result(call_id, name, admitted)]
            message = e.error.message
            await _finish_forwarded(recorder, call_id, name, arguments, message, message, None, outcomes)
            errors = _protocol_errors.get()
            if errors is not None:
                errors.append(e)
            raise
        except Exception as e:
            outcomes = [*skipped, *self._report_no_result(call_id, name, admitted)]
            await _finish_forwarded(recorder, call_id, name, arguments, str(e), str(e), None, outcomes)
            raise
        except BaseException:
            # Cancellation leaves the started disposition record unfinished,
            # but this admitted call still consumed its occurrence.
            self._report_no_result(call_id, name, admitted)
            raise
        placements = []
        # One key per content block, tracking each block through the instructions (see _track).
        keys: list[object] = [object() for _ in result.content]
        for admission in admitted:
            result, placement = _inject_result(result, admission[2])
            keys, written = _track(keys, placement.edit, len(result.content))
            placements.append((admission, placement, written))
        # Emission is checked on the result as returned, so a later instruction that replaced a placement shows.
        outcomes = list(skipped)
        for (scope, index, instruction, occurrence, order), placement, written in placements:
            outcome = _outcome(
                scope,
                index,
                instruction,
                call_id,
                name,
                occurrence,
                order,
                placement.status,
                placement.detail,
                selected=placement.status in _SELECTED,
                transformation=instruction.transformation,
                emitted=_emitted(result, keys, instruction, placement, written),
            )
            self._outcomes.append(outcome)
            outcomes.append(outcome)
        # The record is what the agent gets: its text blocks as one string, and its structured content.
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


def _transform(instruction: InjectionInstruction) -> str:
    if instruction.transformation is Transformation.JSON_STRING:
        return json.dumps(instruction.payload)
    return instruction.payload


def _inject_path(
    value: Any, path: tuple[str | int | RecordSelector, ...], payload: str, mode: InjectionMode
) -> str | None:
    current = value
    for part in path[:-1]:
        current, mismatch = _resolve_path_part(current, part)
        if mismatch is not None:
            return mismatch
    return _inject_leaf(current, path[-1], payload, mode)


def _resolve_path_part(current: Any, part: str | int | RecordSelector) -> tuple[Any, str | None]:
    if type(part) is str:
        if not isinstance(current, dict):
            return None, f"field {part!r} requires an object, got {type(current).__name__}"
        if part not in current:
            return None, f"field {part!r} is missing"
        return current[part], None
    if type(part) is int:
        if not isinstance(current, list):
            return None, f"index {part} requires a list, got {type(current).__name__}"
        if part >= len(current):
            return None, f"index {part} is out of range"
        return current[part], None
    assert type(part) is RecordSelector
    label = f"[{part.field}={part.value!r}]"
    if not isinstance(current, list):
        return None, f"record selector {label} requires a list, got {type(current).__name__}"
    matches = [
        item
        for item in current
        if isinstance(item, dict)
        and part.field in item
        and type(item[part.field]) is type(part.value)
        and item[part.field] == part.value
    ]
    if not matches:
        return None, f"record selector {label} matched no records"
    if len(matches) != 1:
        return None, f"record selector {label} matched {len(matches)} records"
    return matches[0], None


def _inject_leaf(current: Any, part: str | int | RecordSelector, payload: str, mode: InjectionMode) -> str | None:
    if type(part) is str:
        if not isinstance(current, dict):
            return f"field {part!r} requires an object, got {type(current).__name__}"
        if part not in current:
            return f"field {part!r} is missing"
        if not isinstance(current[part], str):
            return f"field {part!r} does not contain a string"
        old = current[part]
        current[part] = payload if mode is InjectionMode.REPLACE else old + payload
        return None
    if type(part) is int:
        if not isinstance(current, list):
            return f"index {part} requires a list, got {type(current).__name__}"
        if part >= len(current):
            return f"index {part} is out of range"
        if not isinstance(current[part], str):
            return f"index {part} does not contain a string"
        old = current[part]
        current[part] = payload if mode is InjectionMode.REPLACE else old + payload
        return None
    assert isinstance(part, RecordSelector)
    _, mismatch = _resolve_path_part(current, part)
    return mismatch or f"record selector [{part.field}={part.value!r}] selects a record, not a string field"


def _inject_structured(
    result: CallToolResult, instruction: InjectionInstruction, payload: str
) -> tuple[CallToolResult | None, str | None]:
    """Place ``payload`` in the structured content, and in the text that mirrors it.

    Returns the updated result, or None with the mismatch. A result whose
    text content is a JSON mirror of its structured content gets that text
    re-serialised from the updated structured content; any other keeps its
    text, and the second value then says why the placement is only partial.
    """
    structured = copy.deepcopy(result.structuredContent)
    mismatch = _inject_path(structured, instruction.target.structured_content_path, payload, instruction.mode)
    if mismatch is not None:
        return None, mismatch
    mirror = _json_mirror(result)
    if isinstance(mirror, str):
        return result.model_copy(update={"structuredContent": structured}), mirror
    block, options = mirror
    content = [block.model_copy(update={"text": json.dumps(structured, **options)})]
    return result.model_copy(update={"structuredContent": structured, "content": content}), None


def _json_mirror(result: CallToolResult) -> tuple[TextContent, dict[str, Any]] | str:
    """The text block that is the JSON serialisation of the structured content, and how to serialise it again.

    The mirror is the common MCP shape: exactly one content block, a text
    block whose text parses as JSON equal, type for type, to the structured
    content. A result without one gets the reason instead.
    """
    content = result.content
    if len(content) != 1 or not isinstance(content[0], TextContent):
        return f"text content is not a single text block ({len(content)} content blocks)"
    block = content[0]
    try:
        parsed = json.loads(block.text)
    except ValueError:
        return "text content is not JSON"
    if not _same_json(parsed, result.structuredContent):
        return "text content is JSON that differs from the structured content"
    return block, _json_style(block.text, result.structuredContent)


def _same_json(a: Any, b: Any) -> bool:
    """Whether two JSON values are equal type for type: ``1`` isn't ``1.0`` or ``true``."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_json(a[key], b[key]) for key in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_same_json(x, y) for x, y in zip(a, b, strict=True))
    return a == b


_JSON_LAYOUTS: tuple[tuple[int | None, tuple[str, str]], ...] = (
    (None, (",", ":")),
    (None, (", ", ": ")),
    (2, (",", ": ")),
    (4, (",", ": ")),
)


def _json_style(text: str, value: Any) -> dict[str, Any]:
    """``json.dumps`` options for the injected value: the predefined style that reproduces ``text`` exactly, if any.

    The styles are compact, spaced, and two- and four-space indented layouts,
    each with and without ASCII escapes and with keys in the structured
    content's order or sorted; the first that gives exactly ``text`` from
    ``value`` is used. Nothing else is reproduced: not a key order of the
    text's own (other than sorted), trailing whitespace or a final newline,
    or another serialiser's number format. For such text it falls back to
    compact for one line, or indented by the width of the second line's
    leading spaces (two when it has none) otherwise, with keys in the
    structured content's order. Unescaped non-ASCII is tried first, so an
    all-ASCII original, which can't show which it used, and any fallback get
    a non-ASCII payload unescaped.
    """
    for indent, separators in _JSON_LAYOUTS:
        for ensure_ascii in (False, True):
            for sort_keys in (False, True):
                options = {
                    "indent": indent,
                    "separators": separators,
                    "ensure_ascii": ensure_ascii,
                    "sort_keys": sort_keys,
                }
                if json.dumps(value, **options) == text:
                    return options
    lines = text.splitlines()
    if len(lines) > 1:
        indent = len(lines[1]) - len(lines[1].lstrip(" ")) or 2
        return {"indent": indent, "separators": (",", ": "), "ensure_ascii": False}
    return {"separators": (",", ":"), "ensure_ascii": False}


def _inject_content(
    result: CallToolResult, instruction: InjectionInstruction, payload: str
) -> tuple[CallToolResult, _Edit] | None:
    target = instruction.target
    content = list(result.content)
    if target.content_block_index is None and target.content_block_type is None:
        block = TextContent(type="text", text=payload)
        if instruction.mode is InjectionMode.REPLACE:
            return result.model_copy(update={"content": [block]}), _Edit("replace_all", 0, 0, payload)
        edit = _Edit("append", len(content), 0, payload)
        return result.model_copy(update={"content": [*content, block]}), edit
    return _inject_selected_block(result, instruction, payload, content)


def _inject_selected_block(
    result: CallToolResult, instruction: InjectionInstruction, payload: str, content: list[ContentBlock]
) -> tuple[CallToolResult, _Edit] | None:
    target = instruction.target
    indexes = range(len(content)) if target.content_block_index is None else (target.content_block_index,)
    for index in indexes:
        if index >= len(content):
            continue
        block = content[index]
        if target.content_block_type is not None and block.type != target.content_block_type:
            continue
        if not isinstance(block, TextContent):
            continue
        if instruction.mode is InjectionMode.REPLACE:
            text, edit = payload, _Edit("rewrite", index, 0, payload)
        else:
            text, edit = block.text + payload, _Edit("extend", index, len(block.text), payload)
        content[index] = block.model_copy(update={"text": text})
        return result.model_copy(update={"content": content}), edit
    return None


@dataclasses.dataclass(frozen=True)
class _Edit:
    """How one instruction changed a result's content blocks, and the text it wrote where.

    ``kind`` is ``replace_all`` (new content, all of it), ``append`` (a block
    added at the end), ``rewrite`` (one block's text replaced) or ``extend``
    (text appended to one block's text, which stays its prefix). ``text`` was
    written at ``start`` in the text of the block at index ``block`` of the
    content after the edit.
    """

    kind: Literal["replace_all", "append", "rewrite", "extend"]
    block: int
    start: int
    text: str


# Where an instruction's text is: the key of the block it wrote into (see _track), the offset and the text.
_Written = tuple[object, int, str]


@dataclasses.dataclass(frozen=True)
class _Placement:
    """What one instruction did to a call's result: its outcome, the transformed payload and where it put it.

    ``placed`` names where: ``structured_content``, ``mirror`` (the text
    re-serialised from the structured content) and ``content_block`` (a text
    block the instruction wrote or appended). ``edit`` is how it changed the
    content blocks, None when it left them as they were.
    """

    status: OutcomeStatus
    detail: str
    payload: str
    placed: tuple[str, ...] = ()
    edit: _Edit | None = None


# The outcomes whose exact selector matched a place in the result.
_SELECTED = frozenset({OutcomeStatus.APPLIED, OutcomeStatus.PARTIAL})


def _inject_result(result: CallToolResult, instruction: InjectionInstruction) -> tuple[CallToolResult, _Placement]:
    payload = _transform(instruction)
    if instruction.size_limit is not None and len(payload.encode()) > instruction.size_limit:
        return result, _Placement(OutcomeStatus.TOO_LARGE, "transformed payload exceeds size_limit", payload)
    if instruction.target.surface is SelectorSurface.STRUCTURED_CONTENT:
        updated, problem = _inject_structured(result, instruction, payload)
        if updated is not None and problem is not None:
            detail = f"exact selector matched structured content only; {problem}"
            return updated, _Placement(OutcomeStatus.PARTIAL, detail, payload, ("structured_content",))
        if updated is not None:
            detail = "exact selector matched; mirrored text re-serialised"
            [mirror] = updated.content
            assert isinstance(mirror, TextContent)  # _inject_structured re-serialised it.
            edit = _Edit("replace_all", 0, 0, mirror.text)
            placement = _Placement(OutcomeStatus.APPLIED, detail, payload, ("structured_content", "mirror"), edit)
            return updated, placement
        mismatch = problem
    else:
        injected = _inject_content(result, instruction, payload)
        if injected is not None:
            updated, edit = injected
            placement = _Placement(OutcomeStatus.APPLIED, "exact selector matched", payload, ("content_block",), edit)
            return updated, placement
        mismatch = "content-block selector did not match"
    if instruction.mismatch_policy is MismatchPolicy.FALLBACK_APPEND:
        content = [*result.content, TextContent(type="text", text=payload)]
        updated = result.model_copy(update={"content": content})
        detail = f"{mismatch}; appended a text block"
        edit = _Edit("append", len(content) - 1, 0, payload)
        return updated, _Placement(OutcomeStatus.FALLBACK, detail, payload, ("content_block",), edit)
    return result, _Placement(OutcomeStatus.UNMATCHED, f"{mismatch}; result unchanged", payload)


def _track(keys: list[object], edit: _Edit | None, size: int) -> tuple[list[object], _Written | None]:
    """The block keys after ``edit``, and where it wrote its text.

    A block keeps its key while its text is unchanged or only extended, so
    text written into it stays where it was written; a rewritten block, an
    appended one and all new content get new keys, so text written into a
    block that was since replaced is found nowhere.
    """
    if edit is None:
        return keys, None
    if edit.kind == "replace_all":
        keys = [object() for _ in range(size)]
    elif edit.kind == "append":
        keys = [*keys, object()]
    elif edit.kind == "rewrite":
        keys = [object() if index == edit.block else key for index, key in enumerate(keys)]
    assert len(keys) == size
    return keys, (keys[edit.block], edit.start, edit.text)


def _still_written(result: CallToolResult, keys: list[object], written: _Written | None) -> bool:
    """Whether the block ``written`` names is still in ``result`` and holds its text where it was written."""
    if written is None:
        return False
    key, start, text = written
    index = next((index for index, block_key in enumerate(keys) if block_key is key), None)
    if index is None:
        return False
    block = result.content[index]
    return isinstance(block, TextContent) and block.text[start : start + len(text)] == text


def _emitted(
    result: CallToolResult,
    keys: list[object],
    instruction: InjectionInstruction,
    placement: _Placement,
    written: _Written | None,
) -> tuple[SelectorSurface, ...]:
    """The parts of the returned ``result`` that still carry the placement's payload where it was placed.

    Structured content carries it when the field the instruction selected
    holds the payload. The text carries a mirrored placement when it is still
    a JSON mirror of structured content that carries it, or the block the
    instruction re-serialised still holds that text; and a text-block
    placement when the block it wrote into or appended still holds the
    payload where it wrote it. Only that place counts: the same text
    elsewhere in the result, or in a block a later instruction replaced,
    isn't this instruction's placement.
    """
    carried: list[SelectorSurface] = []
    structured = "structured_content" in placement.placed and _field_carries(
        result.structuredContent, instruction.target.structured_content_path, placement.payload
    )
    if structured:
        carried.append(SelectorSurface.STRUCTURED_CONTENT)
    if "mirror" in placement.placed:
        mirrored = structured and not isinstance(_json_mirror(result), str)
        if mirrored or _still_written(result, keys, written):
            carried.append(SelectorSurface.CONTENT_BLOCK)
    elif "content_block" in placement.placed and _still_written(result, keys, written):
        carried.append(SelectorSurface.CONTENT_BLOCK)
    return tuple(carried)


def _field_carries(value: Any, path: tuple[str | int | RecordSelector, ...], payload: str) -> bool:
    """Whether the string field at ``path`` in ``value`` holds ``payload``."""
    current = value
    for part in path:
        current, mismatch = _resolve_path_part(current, part)
        if mismatch is not None:
            return False
    return isinstance(current, str) and payload in current


def _records(outcomes: Sequence[InjectionOutcome]) -> list[InjectionRecord]:
    """A call's outcomes as its finish carries them: no scope, call ID or admission order."""
    return [
        {
            "instruction_index": outcome.instruction_index,
            "probe_key": outcome.probe_key,
            "occurrence": outcome.occurrence,
            "status": outcome.status.value,
            "detail": outcome.detail,
            "selected": outcome.selected,
            "transformation": outcome.transformation.value if outcome.transformation is not None else None,
            "emitted": [surface.value for surface in outcome.emitted],
        }
        for outcome in outcomes
    ]


def _outcome(
    scope: str,
    index: int,
    instruction: InjectionInstruction,
    call_id: str,
    tool: str,
    occurrence: int,
    admission_order: int,
    status: OutcomeStatus,
    detail: str,
    *,
    selected: bool = False,
    transformation: Transformation | None = None,
    emitted: tuple[SelectorSurface, ...] = (),
) -> InjectionOutcome:
    return InjectionOutcome(
        instruction_index=index,
        probe_key=instruction.probe_key,
        scope=scope,
        call_id=call_id,
        tool=tool,
        occurrence=occurrence,
        admission_order=admission_order,
        status=status,
        detail=detail,
        selected=selected,
        transformation=transformation,
        emitted=emitted,
    )
