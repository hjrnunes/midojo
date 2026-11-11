"""Where an interceptor serves its tools, and how a host is written in that address."""

from __future__ import annotations

TOOL_PATH = "/mcp"
"""The path an interceptor serves its tools on."""


def bare_host(host: str) -> str:
    """``host`` without the brackets an IPv6 address has in a URL: how a socket and a policy name it."""
    return host[1:-1] if host.startswith("[") and host.endswith("]") else host


def tool_url(host: str, port: int) -> str:
    """The URL of an interceptor at ``host`` and ``port``, an IPv6 host bracketed once, bracketed or not."""
    host = bare_host(host)
    return f"http://{f'[{host}]' if ':' in host else host}:{port}{TOOL_PATH}"


__all__ = ["TOOL_PATH", "bare_host", "tool_url"]
