"""The two implementations of the SDK's seams, for tests that run over both.

``http`` reaches a control plane app through ``AgentControlPlaneClient`` (and
the orchestrator's ``ControlPlaneClient``) over an in-memory ASGI transport.
``in-process`` calls the app's ``ControlPlane`` through ``InProcessAgentClient``
(and ``InProcessControlClient``); every HTTP client it hands out fails the test
on any request, so an in-process run that sent one can't pass.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from fastapi import FastAPI
from midojo_sdk.client import AgentControlPlaneClient
from midojo_sdk.mcp import MidojoMCP

from midojo.control_plane.client import ControlClient, ControlPlaneClient
from midojo.control_plane.inprocess import InProcessAgentClient
from midojo.control_plane.inprocess_control import InProcessControlClient

CONTROL_URL = "http://control"

Agent = AgentControlPlaneClient | InProcessAgentClient

NAMES = ["http", "in-process"]


class Backend:
    """How a test reaches the ``/agent`` seams of a control plane app: over HTTP, or in process."""

    def __init__(self, name: str, app: FastAPI, monkeypatch: pytest.MonkeyPatch) -> None:
        self.name = name
        self.app = app
        self._monkeypatch = monkeypatch
        self._clients: dict[int, httpx.AsyncClient] = {}
        # Every request an in-process run sent over HTTP: none, or the test has failed already.
        self.forbidden_requests: list[httpx.Request] = []

    @property
    def in_process(self) -> bool:
        return self.name == "in-process"

    def http(self, app: FastAPI | None = None) -> httpx.AsyncClient:
        """An HTTP client for ``app``'s control plane: one that fails on any request, in process."""
        app = app or self.app
        if id(app) not in self._clients:
            transport = httpx.MockTransport(self._forbid) if self.in_process else httpx.ASGITransport(app)
            self._clients[id(app)] = httpx.AsyncClient(transport=transport)
        return self._clients[id(app)]

    def agent(self, session_token: str, app: FastAPI | None = None) -> Agent:
        """The ``Recorder`` and ``WorldState`` of the evaluation ``session_token`` is bound to."""
        app = app or self.app
        if self.in_process:
            return InProcessAgentClient(app.state.control, session_token)
        return AgentControlPlaneClient(CONTROL_URL, session_token, http=self.http(app))

    def control(self, app: FastAPI | None = None, *, token: str | None = None) -> ControlClient:
        """The orchestrator's control client for ``app``'s control plane, sending its control ``token`` over HTTP."""
        app = app or self.app
        if self.in_process:
            return InProcessControlClient(app.state.control)
        return ControlPlaneClient(CONTROL_URL, token=token, http=self.http(app))

    def mcp(self, name: str, session_token: str, **kwargs: Any) -> MidojoMCP:
        """A ``MidojoMCP`` recording in the evaluation ``session_token`` is bound to.

        Over HTTP it's wired the default way, to the control plane URL and the
        session in ``MIDOJO_SESSION_TOKEN``; in process, it's given the
        in-process client as its recorder and world.
        """
        if self.in_process:
            agent = self.agent(session_token)
            return MidojoMCP(name, recorder=agent, world=agent, **kwargs)
        self._monkeypatch.setenv("MIDOJO_SESSION_TOKEN", session_token)
        return MidojoMCP(name, control_plane_url=CONTROL_URL, http=self.http(), **kwargs)

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        assert self.forbidden_requests == [], f"the in-process run sent HTTP requests: {self.forbidden_requests}"

    def _forbid(self, request: httpx.Request) -> httpx.Response:
        self.forbidden_requests.append(request)
        pytest.fail(f"The in-process run sent an HTTP request: {request.method} {request.url}")
