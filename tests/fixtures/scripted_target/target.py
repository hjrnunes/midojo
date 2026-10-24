"""The scripted target: an MCP client that calls a fixed sequence of tools. No model.

It runs as an agent command does, with the user's message as its last
argument, and finds its tool endpoint where a host-mode runtime puts it:
``MIDOJO_TOOL_URL`` and ``MIDOJO_TOOL_TOKEN`` in its environment. Without them
it calls no tool. It writes what it was given and what it saw (its arguments,
its environment, the tools listed and each raw result) to ``transcript.json``
in its working directory, and prints a reply.

It also behaves as a curious agent would: it sends any session token or debug
value it finds in its environment (``MIDOJO_SESSION_TOKEN``, ``TARGET_DEBUG``)
along with its tool calls, in an ``X-Debug`` header.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

SCRIPT = [
    ("get_weather", {"city": "Boston"}),
    ("list_alerts", {}),
    ("forecast", {"city": "Boston", "days": 2}),
    ("book_shelter", {"city": "Boston"}),
]


async def main() -> None:
    url, token = os.environ.get("MIDOJO_TOOL_URL"), os.environ.get("MIDOJO_TOOL_TOKEN")
    tools, results = [], []
    if url and token:
        headers = {"Authorization": f"Bearer {token}"}
        if leaked := os.environ.get("MIDOJO_SESSION_TOKEN") or os.environ.get("TARGET_DEBUG"):
            headers["X-Debug"] = leaked
        async with Client(StreamableHttpTransport(url, headers=headers)) as client:
            tools = [tool.model_dump(mode="json") for tool in await client.list_tools()]
            results = [(await client.call_tool_mcp(name, args)).model_dump(mode="json") for name, args in SCRIPT]
    transcript = {"argv": sys.argv, "env": dict(os.environ), "tools": tools, "results": results}
    Path("transcript.json").write_text(json.dumps(transcript, indent=2))
    print(f"done: {len(results)} tool calls")


if __name__ == "__main__":
    asyncio.run(main())
