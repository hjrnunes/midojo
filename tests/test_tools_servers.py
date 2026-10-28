"""The bundled suites' real-tools servers start offline, serve their MCP tools over streamable HTTP, and stop cleanly.

A host-mode run's interceptor forwards to them (``--upstream-mcp-url``):
``weather-tools-serve`` for the weather suite, ``minibank-real-mcp-serve`` for
the minibank suite. Each runs here as its console script, in a fresh process
without the model settings the suites need (no ``LITELLM_*``), on a free
loopback port; nothing goes beyond loopback.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import sys
from pathlib import Path

import pytest
import pytest_asyncio
from fastmcp import Client

STARTUP_SECONDS = 30
# How many free ports a server is tried on: another process may take the one picked before the server binds it.
PORT_ATTEMPTS = 5


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class Server:
    """One run of a console script of this environment on ``port``, its output kept as it comes."""

    def __init__(self, script: str, port: int) -> None:
        self.script, self.port = script, port
        self.url = f"http://127.0.0.1:{port}/mcp"
        self.lines: list[str] = []
        self.process: asyncio.subprocess.Process | None = None
        self._reading: asyncio.Task[None] | None = None

    @property
    def output(self) -> str:
        return "".join(self.lines)

    async def start(self) -> bool:
        """Start it, and wait until it says it listens on its port: True; False when the port was taken first."""
        executable = Path(sys.executable).with_name(self.script)
        assert executable.exists(), f"{self.script} is not installed next to {sys.executable}"
        env = {key: value for key, value in os.environ.items() if not key.startswith("LITELLM_")}
        self.process = await asyncio.create_subprocess_exec(
            str(executable),
            *("--host", "127.0.0.1", "--port", str(self.port)),
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        # Its own log says when it bound the port, so a listener another process has there is never taken for it.
        ready = f"Uvicorn running on http://127.0.0.1:{self.port}"
        try:
            async with asyncio.timeout(STARTUP_SECONDS):
                while line := await self._readline():
                    if ready in line:
                        self._reading = asyncio.create_task(self._drain())
                        return True
                    if "address already in use" in line.lower():
                        await self.stop(expect_clean=False)
                        return False
        except TimeoutError:
            await self.stop(expect_clean=False)
            pytest.fail(f"{self.script} didn't listen on port {self.port} within {STARTUP_SECONDS}s:\n{self.output}")
        await self.process.wait()
        pytest.fail(f"{self.script} exited with {self.process.returncode}:\n{self.output}")

    async def _readline(self) -> str:
        assert self.process is not None and self.process.stdout is not None
        line = (await self.process.stdout.readline()).decode(errors="replace")
        self.lines.append(line)
        return line

    async def _drain(self) -> None:
        while await self._readline():
            pass

    async def stop(self, *, expect_clean: bool = True) -> None:
        """Stop it with SIGTERM; kill and reap it, and fail, if it doesn't stop in time."""
        process = self.process
        if process is None:
            return
        if process.returncode is None:
            process.send_signal(signal.SIGTERM)
        try:
            async with asyncio.timeout(STARTUP_SECONDS):
                await process.wait()
                if self._reading is not None:
                    await self._reading
        except TimeoutError:
            process.kill()
            await process.wait()
            pytest.fail(f"{self.script} didn't stop within {STARTUP_SECONDS}s of SIGTERM; killed:\n{self.output}")
        if expect_clean:
            # It shut down gracefully when told to, not with an error of its own. uvicorn exits by the signal it
            # handled, once its shutdown is complete.
            assert process.returncode in {0, -signal.SIGTERM}, self.output
            assert "Application shutdown complete." in self.output, self.output
            assert "Traceback" not in self.output, self.output


@pytest_asyncio.fixture
async def serve():
    """Start a console script on a free port and check it serves ``tools``: its MCP URL. Stopped afterwards."""
    running: list[Server] = []

    async def start(script: str, tools: set[str]) -> str:
        for _ in range(PORT_ATTEMPTS):
            server = Server(script, free_port())
            if not await server.start():
                continue  # the port was taken before it bound it: try another
            running.append(server)
            async with Client(server.url) as client:
                listed = {tool.name for tool in await client.list_tools()}
            assert listed == tools, f"{script} serves {sorted(listed)}"
            return server.url
        pytest.fail(f"{script} found no free port in {PORT_ATTEMPTS} attempts")

    yield start

    for server in running:
        await server.stop()


@pytest.mark.asyncio
async def test_weather_tools_serve_starts_offline_and_serves_the_real_tools(serve):
    url = await serve("weather-tools-serve", {"get_weather", "list_cities", "send_email", "send_weather_alert"})
    async with Client(url) as client:
        weather = await client.call_tool("get_weather", {"city": "New York"})
    assert weather.content[0].text == "New York: 72°F, sunny"  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_minibank_real_mcp_serve_starts_offline_and_serves_the_bank(serve):
    from suites.minibank import host_tools

    # Every tool it serves has a host tool, which the interceptor serves in its place.
    url = await serve(
        "minibank-real-mcp-serve", {tool.__name__ for tool in host_tools.ALL_TOOLS} - {"send_notification"}
    )
    async with Client(url) as client:
        balance = await client.call_tool("get_account_balance", {"account_id": "ACC001"})
    assert '"account_id": "ACC001"' in balance.content[0].text  # type: ignore[union-attr]


# --- the harness's own failure paths ---


@pytest.mark.asyncio
async def test_a_server_whose_port_is_taken_first_is_started_on_another(serve, monkeypatch):
    # Another process binds the port picked for the server before the server does: a listener that answers
    # connections, which the server's own log, not the port answering, tells apart from it.
    taken = socket.socket()
    taken.bind(("127.0.0.1", 0))
    taken.listen()
    taken_port, other_port = taken.getsockname()[1], free_port()
    ports = iter([taken_port, other_port])
    monkeypatch.setattr(sys.modules[__name__], "free_port", lambda: next(ports))
    try:
        url = await serve("weather-tools-serve", {"get_weather", "list_cities", "send_email", "send_weather_alert"})
    finally:
        taken.close()
    assert url == f"http://127.0.0.1:{other_port}/mcp"


@pytest.mark.asyncio
async def test_a_server_that_ignores_sigterm_is_killed_and_reaped(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "STARTUP_SECONDS", 1)
    server = Server("weather-tools-serve", 0)
    ignoring = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('up', flush=True); time.sleep(60)"
    )
    server.process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", ignoring, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    assert await server._readline() == "up\n"
    with pytest.raises(pytest.fail.Exception, match="didn't stop within 1s of SIGTERM; killed"):
        await server.stop()
    assert server.process.returncode == -signal.SIGKILL
