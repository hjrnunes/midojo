from __future__ import annotations

import abc
import asyncio
import contextlib
import os
import uuid
from typing import TYPE_CHECKING

import httpx

if TYPE_CHECKING:
    from midojo.runtimes.openshell import OpenShellRuntime

# Carries the evaluation session to agents reached over HTTP. midojo_sdk.session
# reads it on the agent side.
SESSION_HEADER = "X-Midojo-Session"


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

    @abc.abstractmethod
    async def send_task(self, prompt: str, *, session_token: str) -> str:
        """Send a task prompt to the agent and return its final text output."""
        ...

    def start_conversation(self, *, session_token: str) -> Conversation:
        """Start one evaluation's conversation. By default, its one message goes through ``send_task``."""
        return _SingleMessageConversation(self, session_token)


class _SingleMessageConversation(Conversation):
    def __init__(self, client: AgentClient, session_token: str) -> None:
        self._client = client
        self._session_token = session_token
        self._sent = False

    async def send(self, message: str) -> str:
        if self._sent:
            raise RuntimeError(f"{type(self._client).__name__} sends one message per evaluation")
        self._sent = True
        return await self._client.send_task(message, session_token=self._session_token)


class SimpleHTTPAgentClient(AgentClient):
    """Sends a prompt via HTTP POST and returns the response text.

    Expects the agent to accept ``{"prompt": "..."}`` and return
    ``{"response": "..."}``.
    """

    def __init__(self, agent_url: str, timeout: float = 120.0) -> None:
        self.agent_url = agent_url
        self.timeout = timeout

    async def send_task(self, prompt: str, *, session_token: str) -> str:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.agent_url, json={"prompt": prompt}, headers={SESSION_HEADER: session_token})
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
    """Sends a task via the A2A protocol and returns the agent's text output."""

    def __init__(self, agent_url: str, timeout: float = 120.0) -> None:
        self.agent_url = agent_url
        self.timeout = timeout

    async def send_task(self, prompt: str, *, session_token: str) -> str:
        from a2a.client import ClientConfig, create_client
        from a2a.types import Message, Part, Role, SendMessageRequest

        config = ClientConfig(
            httpx_client=httpx.AsyncClient(timeout=self.timeout, headers={SESSION_HEADER: session_token}),
        )
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
    """

    supports_multiturn = True

    def __init__(
        self,
        base_url: str,
        model: str,
        mcp_server_url: str,
        *,
        mcp_server_label: str = "midojo",
        instructions: str = "",
        api_key: str,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.mcp_server_url = mcp_server_url
        self.mcp_server_label = mcp_server_label
        self.instructions = instructions
        self.api_key = api_key
        self.timeout = timeout

    async def send_task(self, prompt: str, *, session_token: str) -> str:
        conversation = self.start_conversation(session_token=session_token)
        try:
            return await conversation.send(prompt)
        finally:
            await conversation.aclose()

    def start_conversation(self, *, session_token: str) -> Conversation:
        return _ResponsesConversation(self, session_token)

    async def _create_response(
        self, openai_client, message: str, *, session_token: str, previous_response_id: str | None
    ):
        continuation = {"previous_response_id": previous_response_id} if previous_response_id is not None else {}
        return await openai_client.responses.create(
            model=self.model,
            input=message,
            instructions=self.instructions,
            tools=[
                {
                    "type": "mcp",
                    "server_label": self.mcp_server_label,
                    "server_url": self.mcp_server_url,
                    "headers": {SESSION_HEADER: session_token},
                    "require_approval": "never",
                },
            ],
            stream=False,
            **continuation,
        )


class _ResponsesConversation(Conversation):
    """Sends every turn through one ``AsyncOpenAI`` client, closed by ``aclose``."""

    def __init__(self, client: OpenAIResponsesAgentClient, session_token: str) -> None:
        self._client = client
        self._session_token = session_token
        self._previous_response_id: str | None = None
        self._openai = None
        self._resources = contextlib.AsyncExitStack()

    async def send(self, message: str) -> str:
        if self._openai is None:
            from openai import AsyncOpenAI  # openai is in the [suites] optional extras

            client = self._client
            self._openai = await self._resources.enter_async_context(
                AsyncOpenAI(base_url=client.base_url, api_key=client.api_key, timeout=client.timeout)
            )
        response = await self._client._create_response(
            self._openai,
            message,
            session_token=self._session_token,
            previous_response_id=self._previous_response_id,
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
    """

    def __init__(
        self,
        ogx_url: str,
        model: str,
        mcp_server_url: str,
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

    async def send_task(self, prompt: str, *, session_token: str) -> str:
        from ogx_client import OgxClient

        client = OgxClient(base_url=self.ogx_url, timeout=self.timeout)
        kwargs: dict = dict(
            model=self.model,
            input=prompt,
            instructions=self.instructions,
            tools=[
                {
                    "type": "mcp",
                    "server_label": self.mcp_server_label,
                    "server_url": self.mcp_server_url,
                    "headers": {SESSION_HEADER: session_token},
                    "require_approval": "never",
                },
            ],
            stream=False,
        )
        if self.shield_id:
            kwargs["guardrails"] = [self.shield_id]

        response = client.responses.create(**kwargs)
        return response.output_text


class PIAgentClient(AgentClient):
    """Launches a PI coding agent as a subprocess for each task.

    The ``agent_dir`` should point to a directory containing a ``.pi/``
    configuration folder with extensions that use ``@midojo/pi-sdk``.
    """

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

    async def send_task(self, prompt: str, *, session_token: str) -> str:
        env = {
            **os.environ,
            "MIDOJO_URL": self.control_url,
            "MIDOJO_SESSION_TOKEN": session_token,
        }

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
    """

    def __init__(
        self,
        runtime: OpenShellRuntime,
        *,
        timeout: float = 300.0,
    ) -> None:
        self._runtime = runtime
        self._timeout = timeout

    async def send_task(self, prompt: str, *, session_token: str) -> str:
        result = await asyncio.to_thread(self._runtime.exec_agent, prompt, timeout_seconds=int(self._timeout))
        if result.exit_code != 0:
            raise RuntimeError(f"OpenShell agent exited with code {result.exit_code}: {result.stderr}")
        return result.stdout.strip()
