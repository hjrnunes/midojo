"""What an unmanaged agent with a shell can read of its evaluation's session.

An unmanaged agent in agent mode (an A2A or HTTP agent recording through a
fake MCP server of its own) gets the session in the ``X-Midojo-Session``
header of every request MiDojo sends it.
``MidojoSessionMiddleware`` binds the header's value to the request's task
(a context variable), and the agent forwards it to its fake MCP server with
``session_headers()``, which records the calls with it. The token is never
in the model's messages, but it is in the agent's process for as long as the
request runs:

- code that runs in that process, such as an in-process tool, reads it with
  ``session_token()``;
- a shell the agent spawns doesn't inherit it from the header, as it isn't in
  the process environment; but it does when ``MIDOJO_SESSION_TOKEN`` is set
  in the agent's own environment, which ``session_token()`` also reads.

So the session is readable by an unmanaged agent that can run code in its own
process, as it is in an agent-mode sandbox; only host mode keeps it out.
"""

from __future__ import annotations

import asyncio
import sys

import httpx
import pytest
from midojo_sdk.session import SESSION_HEADER, MidojoSessionMiddleware, session_headers, session_token
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from midojo.agent_client import SESSION_HEADER as SENT_HEADER

TOKEN = "session-secret-3c1d"
SHELL = [sys.executable, "-I", "-c", "import os; print(os.environ.get('MIDOJO_SESSION_TOKEN', ''), end='')"]


async def shell() -> str:
    """What a shell the agent spawns, inheriting its environment, finds as the session."""
    process = await asyncio.create_subprocess_exec(*SHELL, stdout=asyncio.subprocess.PIPE)
    out, _ = await process.communicate()
    return out.decode()


async def task(_request: Request) -> JSONResponse:
    # What the agent's code sees while it handles the task, and what it forwards to its MCP server.
    return JSONResponse({"in_process": session_token(), "forwarded": session_headers(), "shell": await shell()})


def agent() -> Starlette:
    """An agent-mode agent that installs the middleware: the header bound to each request's task."""
    app = Starlette(routes=[Route("/", task, methods=["POST"])])
    app.add_middleware(MidojoSessionMiddleware)
    return app


async def send_task() -> dict:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(agent()), base_url="http://agent") as http:
        response = await http.post("/", headers={SENT_HEADER: TOKEN}, json={})
    response.raise_for_status()
    return response.json()


def test_the_orchestrator_and_the_sdk_name_the_same_header():
    assert SENT_HEADER == SESSION_HEADER == "X-Midojo-Session"


@pytest.mark.asyncio
async def test_code_in_the_agents_process_reads_the_session_but_a_spawned_shell_does_not(monkeypatch):
    monkeypatch.delenv("MIDOJO_SESSION_TOKEN", raising=False)
    seen = await send_task()
    assert seen["in_process"] == TOKEN
    assert seen["forwarded"] == {SESSION_HEADER: TOKEN}
    assert seen["shell"] == ""


@pytest.mark.asyncio
async def test_a_spawned_shell_reads_a_session_set_in_the_agents_environment(monkeypatch):
    monkeypatch.setenv("MIDOJO_SESSION_TOKEN", TOKEN)
    seen = await send_task()
    assert seen["shell"] == TOKEN
