"""The FastMCP internals that MidojoMCP wraps.

``MidojoMCP`` reaches into FastMCP's low-level server (``FastMCP._mcp_server``)
to pass an upstream's JSON-RPC errors through (``request_handlers`` keyed by
``CallToolRequest``) and to give each downstream session its own upstream
connection (``run``, called once per session). These are private, so a FastMCP
upgrade can move them; this test names each one, so such an upgrade fails here
with the hook's name rather than deep inside the fidelity tests.
"""

import inspect

from fastmcp import FastMCP
from mcp.types import CallToolRequest
from midojo_sdk.mcp import MidojoMCP


def test_fastmcp_keeps_the_low_level_server_hooks_midojo_wraps():
    server = getattr(FastMCP("compat"), "_mcp_server", None)
    assert server is not None, "FastMCP no longer exposes _mcp_server; MidojoMCP wraps it"
    assert isinstance(getattr(server, "request_handlers", None), dict), (
        "_mcp_server.request_handlers is no longer a dict"
    )
    assert inspect.iscoroutinefunction(getattr(server, "run", None)), (
        "_mcp_server.run is no longer a coroutine function"
    )


def test_midojo_installs_its_call_tool_handler():
    mcp = MidojoMCP("compat", control_plane_url="http://control.invalid")
    handlers = mcp._fastmcp._mcp_server.request_handlers
    assert CallToolRequest in handlers, "FastMCP registers no handler under CallToolRequest"
    assert handlers[CallToolRequest].__qualname__.endswith("_pass_protocol_errors.<locals>.call_tool")
