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

# What a host-mode target never gets, whoever set it: the variables that give or name an evaluation's
# session or the control plane, and those that name the real MCP server, a route past the interceptor.
HOST_MODE_WITHHELD_ENV = frozenset(
    {"MIDOJO_SESSION_TOKEN", "MIDOJO_URL", "MIDOJO_CONTROL_TOKEN", "UPSTREAM_MCP_URL", "MCP_SERVER_URL"}
)


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
