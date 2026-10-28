"""The bundled suites' real-tools servers start offline, serve their MCP tools over streamable HTTP, and stop cleanly.

A host-mode run's interceptor forwards to them (``--upstream-mcp-url``):
``weather-tools-serve`` for the weather suite, ``minibank-real-mcp-serve`` for
the minibank suite. Each runs here as its console script, in a fresh process
without the model settings the suites need (no ``LITELLM_*``), on a free
loopback port; nothing goes beyond loopback.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
from fastmcp import Client

STARTUP_SECONDS = 30


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def accepting(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture
def serve():
    """Start a console script of this environment on a free port: its MCP URL; stopped with SIGTERM afterwards."""
    started: list[subprocess.Popen[str]] = []

    def start(script: str) -> str:
        executable = Path(sys.executable).with_name(script)
        assert executable.exists(), f"{script} is not installed next to {sys.executable}"
        port = free_port()
        env = {key: value for key, value in os.environ.items() if not key.startswith("LITELLM_")}
        process = subprocess.Popen(
            [str(executable), "--host", "127.0.0.1", "--port", str(port)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        started.append(process)
        deadline = time.monotonic() + STARTUP_SECONDS
        while not accepting(port):
            if process.poll() is not None:
                pytest.fail(f"{script} exited with {process.returncode}: {process.communicate()[0]}")
            if time.monotonic() > deadline:
                pytest.fail(f"{script} didn't listen on port {port} within {STARTUP_SECONDS}s")
            time.sleep(0.05)
        return f"http://127.0.0.1:{port}/mcp"

    yield start

    for process in started:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        output, _ = process.communicate(timeout=STARTUP_SECONDS)
        # It shut down gracefully when told to, not with an error of its own. uvicorn exits by the signal it
        # handled, once its shutdown is complete.
        assert process.returncode in {0, -signal.SIGTERM}, output
        assert "Application shutdown complete." in output, output
        assert "Traceback" not in output, output


@pytest.mark.asyncio
async def test_weather_tools_serve_starts_offline_and_serves_the_real_tools(serve):
    url = serve("weather-tools-serve")
    async with Client(url) as client:
        tools = {tool.name for tool in await client.list_tools()}
        weather = await client.call_tool("get_weather", {"city": "New York"})
    assert tools == {"get_weather", "list_cities", "send_email", "send_weather_alert"}
    assert weather.content[0].text == "New York: 72°F, sunny"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_minibank_real_mcp_serve_starts_offline_and_serves_the_bank(serve):
    from suites.minibank import host_tools

    url = serve("minibank-real-mcp-serve")
    async with Client(url) as client:
        tools = {tool.name for tool in await client.list_tools()}
        balance = await client.call_tool("get_account_balance", {"account_id": "ACC001"})
    # Every tool it serves has a host tool, which the interceptor serves in its place.
    assert tools == {tool.__name__ for tool in host_tools.ALL_TOOLS} - {"send_notification"}
    assert '"account_id": "ACC001"' in balance.content[0].text  # type: ignore[union-attr]
