"""Spike: FastMCP ``http_app`` bindings mounted and unmounted on one uvicorn listener (bead asago-v9md.2).

One uvicorn listener serves a Starlette app. A binding is one ``MidojoMCP``'s
FastMCP ``http_app``, mounted at ``/b/<id>/mcp`` behind the binding's own
bearer token, mirroring its own upstream server. The binding's lifespan (the
FastMCP session manager) is entered when it mounts and left when it unbinds.
These tests show the mechanics v9md.3's service needs, including that
unmounting a binding leaves no task, listener or connection behind.

No model and no network beyond loopback.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from typing import Any

import httpx
import pytest
from fastmcp import Client, Context, FastMCP
from midojo_sdk.mcp import MidojoMCP
from midojo_sdk.recorder import Disposition, InjectionRecord
from starlette.applications import Starlette
from starlette.routing import WebSocketRoute
from starlette.websockets import WebSocket
from websockets.asyncio.client import connect as ws_connect

from tests.mount_spike import BindingHost, serve

ALPHA_TOKEN = "alpha-token"
BETA_TOKEN = "beta-token"
WARM_TOKEN = "warm-token"


class Calls:
    """Records the calls of one binding, so a test can show two bindings never share them."""

    def __init__(self) -> None:
        self.starts: list[str] = []
        self.finishes: list[str] = []

    async def start_call(self, *, call_id: str, function: str, args: dict[str, Any], disposition: Disposition) -> None:
        self.starts.append(function)

    async def finish_call(
        self,
        *,
        call_id: str,
        function: str,
        args: dict[str, Any],
        result: str,
        disposition: Disposition,
        error: str | None = None,
    ) -> None:
        self.finishes.append(function)

    async def finish_call_detailed(
        self,
        *,
        call_id: str,
        function: str,
        args: dict[str, Any],
        result: str,
        disposition: Disposition,
        error: str | None = None,
        structured_content: dict[str, Any] | None,
        injection_outcomes: Sequence[InjectionRecord],
    ) -> None:
        self.finishes.append(function)


def upstream(name: str) -> FastMCP:
    """An upstream MCP server whose one tool names the upstream session it is serving."""
    server = FastMCP(name)

    @server.tool
    def whoami(ctx: Context) -> str:
        return f"{name}:{ctx.session_id}"

    return server


async def bound(name: str) -> tuple[MidojoMCP, Calls]:
    """A MidojoMCP mirroring a fresh upstream server, recording into its own recorder."""
    calls = Calls()
    mcp = MidojoMCP(name, upstream_transport=upstream(name), recorder=calls)
    await mcp.mirror_upstream()
    return mcp, calls


def text(result: Any) -> str:
    return "\n".join(block.text for block in result.content if getattr(block, "text", None) is not None)


def probe_app() -> Starlette:
    """The root app: a WebSocket probe proving the one listener has WebSockets enabled."""

    async def probe(websocket: WebSocket) -> None:
        await websocket.accept()
        await websocket.send_text("pong")
        await websocket.close()

    return Starlette(routes=[WebSocketRoute("/ws-probe", probe)])


@pytest.mark.asyncio
async def test_a_binding_serves_at_its_path_only_while_it_is_mounted():
    app = probe_app()
    async with serve(app) as listener:
        host = BindingHost(app)
        alpha, calls = await bound("alpha")
        url = listener.url("/b/alpha/mcp")

        async with httpx.AsyncClient() as http:
            assert (await http.post(url, json={})).status_code == 404
        assert host.events == []

        await host.mount("alpha", token=ALPHA_TOKEN, mcp=alpha)
        assert host.events == [("alpha", "entered")]

        async with httpx.AsyncClient() as http:
            assert (await http.post(url, json={})).status_code == 401
        async with Client(url, auth=ALPHA_TOKEN) as target:
            assert text(await target.call_tool("whoami", {})).startswith("alpha:")
        assert calls.starts == ["whoami"]
        assert calls.finishes == ["whoami"]

        await host.unmount("alpha")
        assert host.events == [("alpha", "entered"), ("alpha", "left")]
        async with httpx.AsyncClient() as http:
            assert (
                await http.post(url, json={}, headers={"authorization": f"Bearer {ALPHA_TOKEN}"})
            ).status_code == 404


@pytest.mark.asyncio
async def test_two_bindings_on_one_listener_have_their_own_token_and_upstream_session():
    app = probe_app()
    async with serve(app) as listener:
        host = BindingHost(app)
        alpha, alpha_calls = await bound("alpha")
        beta, beta_calls = await bound("beta")
        await host.mount("alpha", token=ALPHA_TOKEN, mcp=alpha)
        await host.mount("beta", token=BETA_TOKEN, mcp=beta)
        assert host.events == [("alpha", "entered"), ("beta", "entered")]
        assert host.ids == ("alpha", "beta")
        alpha_url, beta_url = listener.url("/b/alpha/mcp"), listener.url("/b/beta/mcp")

        async with httpx.AsyncClient() as http:
            assert (await http.post(alpha_url, json={})).status_code == 401
            assert (await http.post(beta_url, json={})).status_code == 401
            crossed = await http.post(beta_url, json={}, headers={"authorization": f"Bearer {ALPHA_TOKEN}"})
            assert crossed.status_code == 401
            assert crossed.headers["www-authenticate"] == "Bearer"

        # Two client sessions on one binding take two upstream sessions; each keeps its one for its whole session.
        async with Client(alpha_url, auth=ALPHA_TOKEN) as one:
            first = text(await one.call_tool("whoami", {}))
            again = text(await one.call_tool("whoami", {}))
        async with Client(alpha_url, auth=ALPHA_TOKEN) as two:
            second = text(await two.call_tool("whoami", {}))
        async with Client(beta_url, auth=BETA_TOKEN) as other:
            third = text(await other.call_tool("whoami", {}))

        alpha_sessions = {first.removeprefix("alpha:"), second.removeprefix("alpha:")}
        assert first == again, "one downstream session must keep its one upstream session"
        assert len(alpha_sessions) == 2, "each downstream session must get its own upstream session"
        assert {first, second} == {f"alpha:{session}" for session in alpha_sessions}
        assert third.startswith("beta:")
        assert third.removeprefix("beta:") not in alpha_sessions, (
            "a binding must never reach another binding's upstream"
        )
        assert alpha_calls.starts == ["whoami"] * 3
        assert beta_calls.starts == ["whoami"]

        # Unbinding one binding leaves the other serving, and the listener itself untouched.
        await host.unmount("alpha")
        assert host.events[-1] == ("alpha", "left")
        assert host.ids == ("beta",)
        async with Client(beta_url, auth=BETA_TOKEN) as still_serving:
            assert text(await still_serving.call_tool("whoami", {})).startswith("beta:")
        async with httpx.AsyncClient() as http:
            assert (
                await http.post(alpha_url, json={}, headers={"authorization": f"Bearer {ALPHA_TOKEN}"})
            ).status_code == 404

        # The same interceptor mounts again: a new app, a new session manager, a new upstream session.
        await host.mount("alpha", token=ALPHA_TOKEN, mcp=alpha)
        async with Client(alpha_url, auth=ALPHA_TOKEN) as remounted:
            fourth = text(await remounted.call_tool("whoami", {}))
        assert fourth.startswith("alpha:")
        assert fourth.removeprefix("alpha:") not in alpha_sessions
        await host.unmount("alpha")
        await host.unmount("beta")


def tcp_states(port: int) -> list[str]:
    """The states of this process's TCP sockets on ``port``, as ``lsof`` reports them."""
    listing = subprocess.run(
        ["lsof", "-nP", "-p", str(os.getpid()), f"-iTCP:{port}", "-a"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.rsplit(None, 1)[-1].strip("()") for line in listing.stdout.splitlines()[1:] if line.strip()]


async def settled(condition: Callable[[], bool], timeout: float = 10.0) -> bool:
    """Whether ``condition`` came to hold within ``timeout``: teardown is asynchronous, so the test waits."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(0.01)
    return True


def described(tasks: set[asyncio.Task[Any]]) -> list[str]:
    return sorted(f"{task.get_name()} ({task.get_coro().__qualname__})" for task in tasks)


@pytest.mark.asyncio
async def test_unmounting_both_bindings_leaves_no_task_listener_or_connection_behind():
    app = probe_app()
    async with serve(app) as listener:
        host = BindingHost(app)
        alpha, _ = await bound("alpha")
        beta, _ = await bound("beta")

        # Warm-up: one binding serves one session and unbinds. A streamable HTTP session starts loop-lifetime
        # tasks of its own, sse_starlette's shutdown watcher included (one per loop); taking the baseline after
        # the warm-up keeps those out of the comparison, so what the two bindings under test leave behind is a leak.
        warm, _ = await bound("warm")
        before_warmup = asyncio.all_tasks()
        await host.mount("warm", token=WARM_TOKEN, mcp=warm)
        async with Client(listener.url("/b/warm/mcp"), auth=WARM_TOKEN) as client:
            await client.call_tool("whoami", {})
        await host.unmount("warm")
        assert await settled(lambda: not listener.server.server_state.connections), "the warm-up session did not close"
        await asyncio.sleep(0.05)
        # The warm-up itself leaks nothing but the loop-lifetime watcher, so the baseline below is not hiding a leak.
        leftovers = asyncio.all_tasks() - before_warmup
        watchers = {task for task in leftovers if task.get_coro().__qualname__ == "_shutdown_watcher"}
        assert len(leftovers) == 1 and len(watchers) == 1, (
            f"the warm-up left more than the loop-lifetime watcher: {described(leftovers)}"
        )
        before_tasks = asyncio.all_tasks()
        assert len(watchers) == 1, f"expected one loop-lifetime sse_starlette watcher, saw {described(watchers)}"
        assert tcp_states(listener.port).count("LISTEN") == 1, (
            "the test expects one listener before it mounts a binding"
        )

        await host.mount("alpha", token=ALPHA_TOKEN, mcp=alpha)
        await host.mount("beta", token=BETA_TOKEN, mcp=beta)
        assert tcp_states(listener.port).count("LISTEN") == 1, "mounting a binding must not open another listener"

        stack = contextlib.AsyncExitStack()
        one = await stack.enter_async_context(Client(listener.url("/b/alpha/mcp"), auth=ALPHA_TOKEN))
        other = await stack.enter_async_context(Client(listener.url("/b/beta/mcp"), auth=BETA_TOKEN))
        await one.call_tool("whoami", {})
        await other.call_tool("whoami", {})
        assert await settled(lambda: "ESTABLISHED" in tcp_states(listener.port)), (
            "the test expects live sessions before it unmounts"
        )

        # Unbind both while their downstream sessions are still open: each binding's lifespan ends under its session.
        await host.unmount("alpha")
        await host.unmount("beta")
        await stack.aclose()

        assert await settled(lambda: asyncio.all_tasks() == before_tasks), (
            f"leaked tasks: {described(asyncio.all_tasks() - before_tasks)}"
        )
        assert await settled(lambda: not listener.server.server_state.connections), (
            "the listener kept a connection open"
        )
        assert await settled(lambda: tcp_states(listener.port) == ["LISTEN"]), (
            f"leaked sockets: {tcp_states(listener.port)}"
        )
        assert tcp_states(listener.port) == ["LISTEN"], "the one listener must keep serving"
        watchers = {task for task in asyncio.all_tasks() if task.get_coro().__qualname__ == "_shutdown_watcher"}
        assert len(watchers) == 1, (
            f"unmounting started another sse_starlette watcher: {described(watchers - before_tasks)}"
        )

        # The one listener still serves after both bindings are gone, WebSockets included.
        async with ws_connect(f"ws://127.0.0.1:{listener.port}/ws-probe") as probe:
            assert await probe.recv() == "pong"
