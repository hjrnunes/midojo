"""A2A MiniBank agent for midojo E2E testing.

A minimal A2A-compliant agent that:
1. Publishes an AgentCard at /.well-known/agent-card.json
2. Connects, for each task, to the MCP tool endpoint the task's request names
3. Uses an OpenAI-compatible LLM (via LiteLLM) for reasoning
4. Speaks the A2A JSON-RPC protocol for task handling

midojo-run runs the suite in host mode (``--interception host --protocol
a2a``): each evaluation's tools are served by an interceptor on the host, and
every request midojo-run sends the agent carries that interceptor's MCP URL in
``X-Midojo-Tool-Url`` and its bearer token in ``X-Midojo-Tool-Token``. The
agent connects its MCP client there for that task only. It never gets the
evaluation's session or the control plane's URL or token, and has no MCP
server of its own to fall back on: a request without the endpoint is refused.

Usage::

    # With explicit flags
    minibank-a2a-agent --litellm-api-key=xxx --litellm-api-url=yyy --litellm-model=zzz

    # With env vars (via dotenv run or exported)
    dotenv run -- minibank-a2a-agent
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import click
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    Message,
    Part,
    Role,
)
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import TextContent
from starlette.applications import Starlette

from suites.minibank import SYSTEM_MESSAGE

if TYPE_CHECKING:
    from openai import OpenAI  # pyright: ignore[reportMissingImports] - in the [suites] extras

# Where each request names the tool endpoint for its task (midojo.agent_client sends them).
TOOL_URL_HEADER = "X-Midojo-Tool-Url"
TOOL_TOKEN_HEADER = "X-Midojo-Tool-Token"


class NoToolEndpointError(RuntimeError):
    """A task's request named no tool endpoint, so the agent has no tools to run it with."""


def tool_endpoint(headers: dict[str, str]) -> tuple[str, str]:
    """The MCP URL and bearer token a task's request headers name, or ``NoToolEndpointError``."""
    lowered = {name.lower(): value for name, value in headers.items()}
    url, token = lowered.get(TOOL_URL_HEADER.lower()), lowered.get(TOOL_TOKEN_HEADER.lower())
    if not url or not token:
        raise NoToolEndpointError(
            f"The request names no tool endpoint ({TOOL_URL_HEADER} and {TOOL_TOKEN_HEADER}): run the suite with "
            "midojo-run --interception host --protocol a2a."
        )
    return url, token


def mcp_tools_to_openai(mcp_tools: list) -> list[dict]:
    """Convert MCP tool definitions to OpenAI function-calling format."""
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                },
            }
        )
    return openai_tools


async def run_agent_loop(prompt: str, llm: OpenAI | Any, model: str, tool_url: str, tool_token: str) -> str:
    """Connect to the task's MCP tool endpoint, discover tools, run tool-use loop with LLM."""
    headers = {"Authorization": f"Bearer {tool_token}"}
    async with streamablehttp_client(tool_url, headers=headers) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            openai_tools = mcp_tools_to_openai(tools_result.tools)

            messages: list[dict] = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ]

            for _ in range(10):
                response = llm.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=openai_tools if openai_tools else None,
                )

                choice = response.choices[0]

                if choice.message.tool_calls:
                    messages.append(choice.message.model_dump())

                    for tool_call in choice.message.tool_calls:
                        fn = tool_call.function
                        args = json.loads(fn.arguments) if fn.arguments else {}
                        print(f"  Tool call: {fn.name}({args})")

                        result = await session.call_tool(fn.name, args)
                        content = ""
                        for block in result.content:
                            if isinstance(block, TextContent):
                                content += block.text

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": content,
                            }
                        )
                else:
                    return choice.message.content or ""

            return messages[-1].get("content", "") if messages else ""


class MinibankAgentExecutor(AgentExecutor):
    def __init__(self, llm: OpenAI | Any, model: str) -> None:
        self.llm = llm
        self.model = model

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_message = context.message
        prompt = ""
        if user_message and user_message.parts:
            prompt = user_message.parts[0].text

        tool_url, tool_token = tool_endpoint(context.call_context.state.get("headers", {}))
        print(f"Received prompt: {prompt[:100]}... (tools at {tool_url})")
        response_text = await run_agent_loop(prompt, self.llm, self.model, tool_url, tool_token)
        print(f"Response: {response_text[:100]}...")

        await event_queue.enqueue_event(Message(role=Role.ROLE_AGENT, parts=[Part(text=response_text)]))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel not supported")


minibank_skill = AgentSkill(
    id="banking_operations",
    name="Banking Operations",
    description="Perform banking operations: check balances, transfer funds, manage compliance",
    tags=["banking", "finance", "compliance"],
    examples=["What is the balance of ACC001?", "Transfer $500 from ACC001 to ACC003"],
)


def create_app(llm: OpenAI | Any, model: str, public_url: str) -> Starlette:
    """The agent's A2A app: its card, advertising ``public_url``, and its JSON-RPC endpoint."""
    agent_card = AgentCard(
        name="MiniBank Agent",
        description="A banking assistant that performs account operations using MCP tools.",
        version="1.0.0",
        supported_interfaces=[
            AgentInterface(url=public_url, protocol_binding="JSONRPC"),
        ],
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[minibank_skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=MinibankAgentExecutor(llm, model),
        task_store=InMemoryTaskStore(),
        agent_card=agent_card,
    )

    routes = []
    routes.extend(create_agent_card_routes(agent_card))
    routes.extend(create_jsonrpc_routes(request_handler, "/"))
    return Starlette(routes=routes)


@click.command()
@click.option("--host", default="127.0.0.1", help="Host to bind to.")
@click.option("--port", default=8000, type=int, help="Port to bind to.", envvar="AGENT_PORT")
@click.option("--litellm-api-key", required=True, help="LiteLLM API key.", envvar="LITELLM_API_KEY")
@click.option("--litellm-api-url", required=True, help="LiteLLM API URL.", envvar="LITELLM_API_URL")
@click.option("--litellm-model", required=True, help="LiteLLM model name.", envvar="LITELLM_MODEL")
@click.option(
    "--public-url",
    default=None,
    envvar="AGENT_PUBLIC_URL",
    help="Externally reachable base URL advertised in the agent card. "
    "Defaults to http://localhost:<port>/. Set this when the agent is reached "
    "from another host (e.g. a Kubernetes Service DNS name).",
)
def main(
    host: str,
    port: int,
    litellm_api_key: str,
    litellm_api_url: str,
    litellm_model: str,
    public_url: str | None,
) -> None:
    from openai import OpenAI  # pyright: ignore[reportMissingImports] - in the [suites] extras

    llm = OpenAI(api_key=litellm_api_key, base_url=litellm_api_url)
    public_url = public_url or f"http://localhost:{port}/"
    app = create_app(llm, litellm_model, public_url)

    print(f"Starting A2A MiniBank agent on port {port}")
    print(f"LLM model: {litellm_model}")
    print(f"Agent card: http://localhost:{port}/.well-known/agent-card.json")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
