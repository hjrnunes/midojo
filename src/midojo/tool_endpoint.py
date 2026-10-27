"""What a target is given in host mode: its tool endpoint, and nothing about the evaluation.

In host mode (``midojo-run --interception host``) the host runs the
interceptor (:mod:`midojo.interception_host`) and keeps the evaluation's
session. The target gets a :class:`ToolEndpoint`: the URL of its evaluation's
interceptor and the credential that interceptor accepts. The credential grants
tool calls through that one interceptor, nothing on the control plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Where a target finds its tool endpoint: in an OpenShell sandbox's environment, or a PI subprocess's.
TOOL_URL_ENV = "MIDOJO_TOOL_URL"
TOOL_TOKEN_ENV = "MIDOJO_TOOL_TOKEN"

# The variables that give or name an evaluation's session or the control plane.
SESSION_ENV = frozenset({"MIDOJO_SESSION_TOKEN", "MIDOJO_URL", "MIDOJO_CONTROL_TOKEN"})

# What a host-mode target never gets, whoever set it: those, and the variables that name the real MCP
# server, a route past the interceptor.
HOST_MODE_WITHHELD_ENV = SESSION_ENV | {"UPSTREAM_MCP_URL", "MCP_SERVER_URL"}

# The path an interceptor serves its tools on.
TOOL_PATH = "/mcp"


def bare_host(host: str) -> str:
    """``host`` without the brackets an IPv6 address has in a URL: how a socket and a policy name it."""
    return host[1:-1] if host.startswith("[") and host.endswith("]") else host


def tool_url(host: str, port: int) -> str:
    """The URL of an interceptor at ``host`` and ``port``, an IPv6 host bracketed once, bracketed or not."""
    host = bare_host(host)
    return f"http://{f'[{host}]' if ':' in host else host}:{port}{TOOL_PATH}"


@dataclass(frozen=True)
class ToolEndpoint:
    """One evaluation's interceptor: its streamable HTTP MCP ``url`` and the bearer ``token`` it accepts."""

    url: str
    token: str = field(repr=False)

    @property
    def headers(self) -> dict[str, str]:
        """The headers an MCP client sends to the interceptor."""
        return {"Authorization": f"Bearer {self.token}"}

    def environment(self) -> dict[str, str]:
        """The endpoint as environment variables, for a target that reads its configuration there."""
        return {TOOL_URL_ENV: self.url, TOOL_TOKEN_ENV: self.token}
