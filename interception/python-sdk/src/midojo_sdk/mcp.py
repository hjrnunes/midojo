"""MCP SDK: the interceptor's MCP face, plus agent mode.

``MidojoMCP`` here is ``midojo_mim.mcp.MidojoMCP`` (the MCP face, which lives in
``midojo-mim``) with one addition: given ``control_plane_url`` and no recorder,
it records every call on the MiDojo control plane for the evaluation session
the agent's request names. That is agent mode: the agent gets the session and
runs the interception layer itself, so the layer can reach the control plane.

Everything else (mirroring, placement, refusals, simulated tools) is the MCP
face's, unchanged. A service or host that holds the evaluation itself gives
the face a recorder and a world and has no use for this module.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastmcp.server.dependencies import get_http_headers
from midojo_mim import mcp as face
from midojo_mim import placement
from midojo_mim.mcp import BLOCKED_REASON, UpstreamClient
from midojo_spec.context import ToolContext
from midojo_spec.injection import InjectionPlan
from midojo_spec.recorder import Recorder, RetryingRecorder, WorldState

from midojo_sdk.client import AgentControlPlaneClient, transient
from midojo_sdk.session import SESSION_HEADER, session_token

__all__ = ["BLOCKED_REASON", "ControlPlaneSessions", "MidojoMCP", "ToolContext", "UpstreamClient"]


class ControlPlaneSessions:
    """The sessions of a control plane: each request's recorder and environment are its evaluation session's.

    The session is the ``X-Midojo-Session`` header the agent sends, or
    ``MIDOJO_SESSION_TOKEN``. Records are resent after a transient failure
    (see ``RetryingRecorder``).
    """

    def __init__(self, control_plane_url: str, *, http: httpx.AsyncClient | None = None) -> None:
        self._control_plane_url = control_plane_url
        self._http = http

    def _client(self) -> AgentControlPlaneClient:
        """The control plane client for the session of the request being handled."""
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=300.0)
        incoming = get_http_headers().get(SESSION_HEADER.lower())
        return AgentControlPlaneClient(self._control_plane_url, session_token(incoming), http=self._http)

    def recorder(self) -> Recorder:
        return RetryingRecorder(self._client(), retryable=transient)

    def world(self) -> WorldState:
        return self._client()

    def scope(self) -> str:
        incoming = get_http_headers().get(SESSION_HEADER.lower())
        return session_token(incoming)


class MidojoMCP(face.MidojoMCP):
    """The MCP face, recording on a control plane when it is given ``control_plane_url`` instead of a recorder.

    Usage::

        mcp = MidojoMCP("weather", control_plane_url=...)

        @mcp.tool()
        async def get_weather(ctx: ToolContext, city: str) -> str:
            cities = await ctx.env("cities")
            ...

    Without ``recorder``, calls are recorded on the control plane at
    ``control_plane_url``, for the evaluation session the agent sends; without
    ``world``, the environment is that session's on the same control plane, or
    none when there's no ``control_plane_url``. A server with ``recorder`` and
    no ``control_plane_url`` mirrors and records with no environment at all.
    ``http`` is the client its control plane requests go through. See
    ``midojo_mim.mcp.MidojoMCP`` for the rest.
    """

    def __init__(
        self,
        name: str,
        *,
        control_plane_url: str | None = None,
        upstream_url: str | None = None,
        http: httpx.AsyncClient | None = None,
        upstream_transport: Any = None,
        recorder: Recorder | None = None,
        world: WorldState | None = None,
        plan: InjectionPlan = InjectionPlan(),
        evaluation_id: str | None = None,
        engine: placement.PlacementEngine | None = None,
    ) -> None:
        if control_plane_url is None and recorder is None:
            raise ValueError("MidojoMCP needs a control_plane_url or a recorder.")
        super().__init__(
            name,
            upstream_url=upstream_url,
            upstream_transport=upstream_transport,
            recorder=recorder,
            world=world,
            sessions=None if control_plane_url is None else ControlPlaneSessions(control_plane_url, http=http),
            plan=plan,
            evaluation_id=evaluation_id,
            engine=engine,
        )
