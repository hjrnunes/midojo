"""How the orchestrator reaches the agent, and what it hands the agent for each evaluation.

Each conversation carries what the agent may hold for its evaluation: the
evaluation's session token, for an interception layer in or beside the agent
that records on the control plane (the ``X-Midojo-Session`` header, or
``MIDOJO_SESSION_TOKEN`` and ``MIDOJO_URL`` for a PI subprocess); or, in host
mode, only the :class:`~midojo.tool_endpoint.ToolEndpoint` of the interceptor
the host runs. A client that can pass a tool endpoint has
``supports_host_interception``: the Responses API clients put it in their MCP
tool block, the A2A client in the ``X-Midojo-Tool-Url`` and
``X-Midojo-Tool-Token`` headers of its requests, the PI client in the
subprocess's environment, and the OpenShell client's sandbox already has it.
"""

from __future__ import annotations

import abc
import asyncio
import contextlib
import os
import uuid
from collections.abc import Sequence
from typing import TYPE_CHECKING

import httpx

from midojo.control_plane.config import CONTROL_TOKEN_ENV
from midojo.tool_endpoint import HOST_MODE_WITHHELD_ENV, TOOL_TOKEN_ENV, TOOL_URL_ENV, ToolEndpoint

if TYPE_CHECKING:
    from midojo.runtimes.openshell import OpenShellRuntime

# Carries the evaluation session to agents reached over HTTP. midojo_sdk.session
# reads it on the agent side.
SESSION_HEADER = "X-Midojo-Session"

# Carry the tool endpoint, in host mode, to an A2A agent: the interceptor's URL and the bearer token it accepts.
TOOL_URL_HEADER = "X-Midojo-Tool-Url"
TOOL_TOKEN_HEADER = "X-Midojo-Tool-Token"

# What a PI subprocess must never inherit from the orchestrator's environment: the
# control token, and the variables that give or name an evaluation's access.
_WITHHELD_ENV = frozenset({CONTROL_TOKEN_ENV, "MIDOJO_URL", "MIDOJO_SESSION_TOKEN", TOOL_URL_ENV, TOOL_TOKEN_ENV})


class Conversation(abc.ABC):
    """One evaluation's exchange with the agent, a user message at a time."""

    @abc.abstractmethod
    async def send(self, message: str) -> str:
        """Send the next user message and return the agent's reply."""
        ...

    async def aclose(self) -> None:
        """Release what the conversation holds once its last message is answered."""


class AgentClient(abc.ABC):
    # Whether a conversation carries the agent's context from one message to the
    # next. Clients without it get one message per evaluation.
    supports_multiturn: bool = False
    # Whether the client can give the agent a tool endpoint instead of a session (host mode).
    supports_host_interception: bool = False

    @abc.abstractmethod
    async def send_task(
        self, prompt: str, *, session_token: str | None = None, tool_endpoint: ToolEndpoint | None = None
    ) -> str:
        """Send a task prompt to the agent and return its final text output.

        The agent gets ``session_token`` or, in host mode, ``tool_endpoint``.
        """
        ...

    def start_conversation(
        self, *, session_token: str | None = None, turn_count: int = 1, tool_endpoint: ToolEndpoint | None = None
    ) -> Conversation:
        """Start one evaluation's conversation of ``turn_count`` messages.

        By default, its one message goes through ``send_task``.
        """
        return _SingleMessageConversation(self, session_token, tool_endpoint)


class _SingleMessageConversation(Conversation):
    def __init__(self, client: AgentClient, session_token: str | None, tool_endpoint: ToolEndpoint | None) -> None:
        self._client = client
        self._session_token = session_token
        self._tool_endpoint = tool_endpoint
        self._sent = False

    async def send(self, message: str) -> str:
        if self._sent:
            raise RuntimeError(f"{type(self._client).__name__} sends one message per evaluation")
        self._sent = True
        # A tool endpoint only in host mode, so clients written before it keep their signature.
        endpoint = {"tool_endpoint": self._tool_endpoint} if self._tool_endpoint is not None else {}
        return await self._client.send_task(message, session_token=self._session_token, **endpoint)


def _session_headers(session_token: str | None) -> dict[str, str]:
    return {SESSION_HEADER: session_token} if session_token is not None else {}


def _tool_endpoint_headers(tool_endpoint: ToolEndpoint) -> dict[str, str]:
    """The headers that give an A2A agent its tool endpoint: where its MCP client connects, and the token it sends."""
    return {TOOL_URL_HEADER: tool_endpoint.url, TOOL_TOKEN_HEADER: tool_endpoint.token}


def _mcp_server(
    url: str | None,
    label: str,
    session_token: str | None,
    tool_endpoint: ToolEndpoint | None,
    allowed_tools: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """The Responses API's MCP tool block: the interceptor and its token in host mode, else the session header.

    With ``allowed_tools``, the block ends with them, so the model is offered only those tools.
    """
    if tool_endpoint is not None:
        url, headers = tool_endpoint.url, tool_endpoint.headers
    else:
        headers = _session_headers(session_token)
    if url is None:
        raise ValueError("No MCP server URL: pass mcp_server_url, or a tool endpoint in host mode")
    block: dict[str, object] = {
        "type": "mcp",
        "server_label": label,
        "server_url": url,
        "headers": headers,
        "require_approval": "never",
    }
    if allowed_tools is not None:
        block["allowed_tools"] = list(allowed_tools)
    return block


def _tool_names(allowed_tools: Sequence[str]) -> tuple[str, ...]:
    """``allowed_tools`` as a tuple of distinct, non-empty tool names, or an error saying what's wrong."""
    if isinstance(allowed_tools, str | bytes | bytearray):
        raise TypeError("allowed_tools must be a sequence of tool names, not a single string")
    names = tuple(allowed_tools)
    if not names:
        raise ValueError("allowed_tools must name at least one tool")
    for name in names:
        if type(name) is not str:
            raise TypeError(f"allowed_tools names must be str, got {type(name).__name__}")
        if not name:
            raise ValueError("allowed_tools names must not be empty")
    if len(set(names)) != len(names):
        raise ValueError("allowed_tools names a tool more than once")
    return names


class SimpleHTTPAgentClient(AgentClient):
    """Sends a prompt via HTTP POST and returns the response text.

    Expects the agent to accept ``{"prompt": "..."}`` and return
    ``{"response": "..."}``.
    """

    def __init__(self, agent_url: str, timeout: float = 120.0) -> None:
        self.agent_url = agent_url
        self.timeout = timeout

    async def send_task(
        self, prompt: str, *, session_token: str | None = None, tool_endpoint: ToolEndpoint | None = None
    ) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.agent_url, json={"prompt": prompt}, headers=_session_headers(session_token))
            resp.raise_for_status()
            data = resp.json()
            for key in ("response", "output", "text"):
                if key in data:
                    return data[key]
            raise ValueError(
                f"Agent response JSON has no recognized key (expected 'response', 'output', or 'text'). "
                f"Got keys: {list(data.keys())}. Is the agent using the expected protocol? "
                f"(Use --protocol a2a for A2A agents.)"
            )


class A2AAgentClient(AgentClient):
    """Sends a task via the A2A protocol and returns the agent's text output.

    Every request to the agent carries the session in ``X-Midojo-Session``; in
    host mode, only the tool endpoint, in ``X-Midojo-Tool-Url`` (the
    interceptor's MCP URL) and ``X-Midojo-Tool-Token`` (the bearer token it
    accepts), which the agent's MCP client connects with for that task.
    """

    supports_host_interception = True

    def __init__(self, agent_url: str, timeout: float = 120.0) -> None:
        self.agent_url = agent_url
        self.timeout = timeout

    async def send_task(
        self, prompt: str, *, session_token: str | None = None, tool_endpoint: ToolEndpoint | None = None
    ) -> str:
        from a2a.client import ClientConfig, create_client
        from a2a.types import Message, Part, Role, SendMessageRequest

        if tool_endpoint is not None:
            headers = _tool_endpoint_headers(tool_endpoint)
        else:
            headers = _session_headers(session_token)
        config = ClientConfig(httpx_client=httpx.AsyncClient(timeout=self.timeout, headers=headers))
        client = await create_client(self.agent_url, client_config=config)
        try:
            request = SendMessageRequest(
                message=Message(
                    role=Role.ROLE_USER,
                    message_id=uuid.uuid4().hex,
                    parts=[Part(text=prompt)],
                ),
            )

            result_text = ""
            async for event in client.send_message(request):
                payload_type = event.WhichOneof("payload")
                if payload_type == "message":
                    for part in event.message.parts:
                        if part.text:
                            result_text += part.text
                elif payload_type == "task":
                    for artifact in event.task.artifacts:
                        for part in artifact.parts:
                            if part.text:
                                result_text += part.text
                elif payload_type == "artifact_update":
                    for part in event.artifact_update.artifact.parts:
                        if part.text:
                            result_text = part.text

            return result_text
        finally:
            await client.close()


class OpenAIResponsesAgentClient(AgentClient):
    """Calls any OpenAI Responses API endpoint (OpenAI cloud, Llama Stack, vLLM, etc.).

    Uses the standard ``openai`` SDK so it works with any server that implements
    the OpenAI Responses API spec — including local deployments. The tool loop
    runs server-side; the model discovers and calls MCP tools without client
    involvement.

    Unlike ``OGXResponsesClient``, this client is fully async and has no
    OGX-specific extensions (no ``guardrails``). The caller supplies the API key
    expected by the endpoint; the CLI reads it from ``OPENAI_API_KEY``.

    A conversation sends each message after the first with the previous reply's
    response ID as ``previous_response_id``, so the server carries the context.

    The MCP tool block names ``mcp_server_url`` with the session in
    ``X-Midojo-Session``; in host mode, the interceptor's URL with its bearer
    token, and no session.

    ``allowed_tools`` limits the MCP server's tools the model is offered: every
    request's MCP tool block then ends with ``"allowed_tools": [...]``, the names
    in the order given. Unset (None), the block has no such key and the model is
    offered every tool the server lists. The names must be distinct, non-empty
    strings, at least one (a bare string is refused: it isn't a list of names).
    """

    supports_multiturn = True
    supports_host_interception = True

    def __init__(
        self,
        base_url: str,
        model: str,
        mcp_server_url: str | None,
        *,
        mcp_server_label: str = "midojo",
        instructions: str = "",
        api_key: str,
        timeout: float = 120.0,
        allowed_tools: Sequence[str] | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.mcp_server_url = mcp_server_url
        self.mcp_server_label = mcp_server_label
        self.instructions = instructions
        self.api_key = api_key
        self.timeout = timeout
        self.allowed_tools: tuple[str, ...] | None = None if allowed_tools is None else _tool_names(allowed_tools)

    async def send_task(
        self, prompt: str, *, session_token: str | None = None, tool_endpoint: ToolEndpoint | None = None
    ) -> str:
        conversation = self.start_conversation(session_token=session_token, tool_endpoint=tool_endpoint)
        try:
            return await conversation.send(prompt)
        finally:
            await conversation.aclose()

    def start_conversation(
        self, *, session_token: str | None = None, turn_count: int = 1, tool_endpoint: ToolEndpoint | None = None
    ) -> Conversation:
        return _ResponsesConversation(self, session_token, tool_endpoint)

    async def _create_response(
        self,
        openai_client,
        message: str,
        *,
        session_token: str | None,
        previous_response_id: str | None,
        tool_endpoint: ToolEndpoint | None = None,
    ):
        continuation = {"previous_response_id": previous_response_id} if previous_response_id is not None else {}
        tool = _mcp_server(self.mcp_server_url, self.mcp_server_label, session_token, tool_endpoint, self.allowed_tools)
        return await openai_client.responses.create(
            model=self.model,
            input=message,
            instructions=self.instructions,
            tools=[tool],
            stream=False,
            **continuation,
        )


class _ResponsesConversation(Conversation):
    """Sends every turn through one ``AsyncOpenAI`` client, closed by ``aclose``."""

    def __init__(
        self, client: OpenAIResponsesAgentClient, session_token: str | None, tool_endpoint: ToolEndpoint | None
    ) -> None:
        self._client = client
        self._session_token = session_token
        self._tool_endpoint = tool_endpoint
        self._previous_response_id: str | None = None
        self._openai = None
        self._resources = contextlib.AsyncExitStack()

    async def send(self, message: str) -> str:
        if self._openai is None:
            from openai import AsyncOpenAI  # pyright: ignore[reportMissingImports] - in the [suites] extras

            client = self._client
            self._openai = await self._resources.enter_async_context(
                AsyncOpenAI(base_url=client.base_url, api_key=client.api_key, timeout=client.timeout)
            )
        response = await self._client._create_response(
            self._openai,
            message,
            session_token=self._session_token,
            previous_response_id=self._previous_response_id,
            tool_endpoint=self._tool_endpoint,
        )
        self._previous_response_id = response.id
        return response.output_text

    async def aclose(self) -> None:
        await self._resources.aclose()


class OGXResponsesClient(AgentClient):
    """Calls the OGX (Llama Stack) Responses API directly.

    The Responses API runs the tool loop server-side — the model discovers
    and calls MCP tools without client involvement. This is the vector
    tested by ogx-ai/ogx#5036: server-side tool results bypass guardrails.

    The MCP tool block is the same as ``OpenAIResponsesAgentClient``'s.
    """

    supports_host_interception = True

    def __init__(
        self,
        ogx_url: str,
        model: str,
        mcp_server_url: str | None,
        *,
        mcp_server_label: str = "midojo",
        instructions: str = "",
        shield_id: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.ogx_url = ogx_url
        self.model = model
        self.mcp_server_url = mcp_server_url
        self.mcp_server_label = mcp_server_label
        self.instructions = instructions
        self.shield_id = shield_id
        self.timeout = timeout

    async def send_task(
        self, prompt: str, *, session_token: str | None = None, tool_endpoint: ToolEndpoint | None = None
    ) -> str:
        from ogx_client import OgxClient  # pyright: ignore[reportMissingImports] - in the [suites] extras

        client = OgxClient(base_url=self.ogx_url, timeout=self.timeout)
        kwargs: dict = dict(
            model=self.model,
            input=prompt,
            instructions=self.instructions,
            tools=[_mcp_server(self.mcp_server_url, self.mcp_server_label, session_token, tool_endpoint)],
            stream=False,
        )
        if self.shield_id:
            kwargs["guardrails"] = [self.shield_id]

        response = client.responses.create(**kwargs)
        return response.output_text


class PIAgentClient(AgentClient):
    """Launches a PI coding agent as a subprocess for each task.

    The ``agent_dir`` should point to a directory containing a ``.pi/``
    configuration folder with extensions that use ``@midojo/pi-sdk``, which
    record with ``MIDOJO_SESSION_TOKEN`` at ``MIDOJO_URL``. In host mode the
    subprocess gets neither, only ``MIDOJO_TOOL_URL`` and ``MIDOJO_TOOL_TOKEN``:
    PI must be configured to reach its tools there. Either way it inherits the
    orchestrator's environment without the control token or any of these; in
    host mode, also without ``UPSTREAM_MCP_URL`` or ``MCP_SERVER_URL``, which
    name the real MCP server.
    """

    supports_host_interception = True

    def __init__(
        self,
        agent_dir: str,
        control_url: str,
        *,
        timeout: float = 120.0,
    ) -> None:
        self.agent_dir = os.path.abspath(agent_dir)
        self.control_url = control_url
        self.timeout = timeout

    async def send_task(
        self, prompt: str, *, session_token: str | None = None, tool_endpoint: ToolEndpoint | None = None
    ) -> str:
        # The agent gets its own evaluation's access only, never the orchestrator's control token.
        env = {name: value for name, value in os.environ.items() if name not in _WITHHELD_ENV}
        if tool_endpoint is not None:
            # Nor anything that names the real MCP server, which would let it call its tools unrecorded.
            env = {name: value for name, value in env.items() if name not in HOST_MODE_WITHHELD_ENV}
            env.update(tool_endpoint.environment())
        elif session_token is not None:
            env.update({"MIDOJO_URL": self.control_url, "MIDOJO_SESSION_TOKEN": session_token})

        proc = await asyncio.create_subprocess_exec(
            "pi",
            "-p",
            "--no-session",
            prompt,
            cwd=self.agent_dir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise TimeoutError(f"PI agent timed out after {self.timeout}s")

        if proc.returncode != 0:
            raise RuntimeError(f"PI agent exited with code {proc.returncode}: {stderr.decode()}")

        return stdout.decode().strip()


class OpenShellAgentClient(AgentClient):
    """Runs the agent command inside an OpenShell sandbox.

    Thin wrapper — the sandbox lifecycle (create, seed, teardown) is managed by
    ``OpenShellRuntime``. This class only executes the agent once the sandbox is ready.

    A multi-turn conversation runs the agent command once per message, in the
    evaluation's one sandbox, with the message as its last argument and
    ``MIDOJO_TURN_INDEX`` (0-based) and ``MIDOJO_TURN_COUNT`` in its environment.
    The command keeps its own state between runs, outside ``/sandbox/workdir``.
    A single-turn task runs the command as ``send_task`` does, without them.
    Each run gets the full ``timeout``.
    Without a ``timeout``, it uses the runtime's (the suite's ``timeout_seconds``, or 300 s).
    The session or tool endpoint is in the sandbox's environment already
    (``OpenShellRuntime.setup``).
    """

    supports_multiturn = True
    supports_host_interception = True

    def __init__(
        self,
        runtime: OpenShellRuntime,
        *,
        timeout: float | None = None,
    ) -> None:
        self._runtime = runtime
        self.timeout = runtime.agent_timeout_seconds if timeout is None else timeout

    async def send_task(
        self, prompt: str, *, session_token: str | None = None, tool_endpoint: ToolEndpoint | None = None
    ) -> str:
        return await self._run(prompt)

    def start_conversation(
        self, *, session_token: str | None = None, turn_count: int = 1, tool_endpoint: ToolEndpoint | None = None
    ) -> Conversation:
        if turn_count == 1:
            return super().start_conversation(session_token=session_token, tool_endpoint=tool_endpoint)
        return _OpenShellConversation(self, turn_count)

    async def _run(self, prompt: str, *, env: dict[str, str] | None = None, turn: str = "") -> str:
        extra = {"env": env} if env is not None else {}
        result = await asyncio.to_thread(self._runtime.exec_agent, prompt, timeout_seconds=self.timeout, **extra)
        if result.exit_code != 0:
            raise RuntimeError(f"OpenShell agent exited with code {result.exit_code}{turn}: {result.stderr}")
        return result.stdout.strip()


class _OpenShellConversation(Conversation):
    """Runs the agent command once per turn, telling it which turn it is."""

    def __init__(self, client: OpenShellAgentClient, turn_count: int) -> None:
        self._client = client
        self._turn_count = turn_count
        self._turn_index = 0

    async def send(self, message: str) -> str:
        index, count = self._turn_index, self._turn_count
        if index >= count:
            raise RuntimeError(f"OpenShell conversation has {count} turns; turn {index + 1} was not expected")
        self._turn_index += 1
        env = {"MIDOJO_TURN_INDEX": str(index), "MIDOJO_TURN_COUNT": str(count)}
        return await self._client._run(message, env=env, turn=f" on turn {index + 1}/{count}")
