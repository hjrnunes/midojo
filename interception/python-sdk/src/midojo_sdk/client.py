"""HTTP client for the control plane's /agent API."""

from __future__ import annotations

import asyncio
import os
import uuid
from typing import Any

import httpx

CALL_ID_MAX_LENGTH = 256
"""The longest ``call_id`` the control plane accepts."""

RECORD_ATTEMPTS = 3
"""How many times a function-call record is sent before its error is raised."""

RECORD_BACKOFF_SECONDS = 0.1
"""The wait before the first resend of a record, doubled before each later one."""


class CallIds:
    """Picks the ``call_id`` of each record one SDK instance sends.

    A harness's own tool-call ID becomes ``<prefix>:<instance UUID>:<harness ID>``,
    so it can't collide with another SDK instance's records in the same
    evaluation. A missing harness ID, one this instance has already used, or one
    too long for the control plane gets a fresh UUID instead.

    The instance UUID is chosen on first use in each process, so workers forked
    from a process that built the instance don't share it.
    """

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._pid: int | None = None
        self._namespace = ""
        self._used: set[str] = set()

    def for_call(self, harness_id: str | None) -> str:
        if self._pid != os.getpid():
            self._pid = os.getpid()
            self._namespace = f"{self._prefix}:{uuid.uuid4()}"
            self._used.clear()
        if harness_id:
            call_id = f"{self._namespace}:{harness_id}"
            if len(call_id) <= CALL_ID_MAX_LENGTH and call_id not in self._used:
                self._used.add(call_id)
                return call_id
        return str(uuid.uuid4())


class AgentControlPlaneClient:
    """Access interception endpoints for one evaluation session."""

    def __init__(self, base_url: str, session_token: str, *, http: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http
        self._headers = {"Authorization": f"Bearer {session_token}"}

    async def get_environment(self) -> dict[str, Any]:
        response = await self._http.get(f"{self._base_url}/agent/environment", headers=self._headers)
        response.raise_for_status()
        return response.json()

    async def put_environment(self, environment: dict[str, Any]) -> None:
        response = await self._http.put(f"{self._base_url}/agent/environment", json=environment, headers=self._headers)
        response.raise_for_status()

    async def record_function_call(
        self,
        *,
        function: str,
        args: dict,
        result: str,
        error: str | None = None,
        blocked: bool = False,
        call_id: str | None = None,
    ) -> None:
        """Record a call, at most once per ``call_id`` (a UUID when not given).

        A transport error or 5xx response is retried with the same ``call_id``,
        up to ``RECORD_ATTEMPTS`` sends in all, so resending a record that
        reached the control plane doesn't record it twice. This is at most one
        record per ``call_id``, not exactly-once: a record whose sends all fail
        is missing. A 4xx, including a 409 for a ``call_id`` recorded with
        another body, is raised at once.
        """
        body = {
            "call_id": call_id or str(uuid.uuid4()),
            "function": function,
            "args": args,
            "result": result,
            "error": error,
            "blocked": blocked,
        }
        for attempt in range(1, RECORD_ATTEMPTS + 1):
            try:
                response = await self._http.post(
                    f"{self._base_url}/agent/function-calls", headers=self._headers, json=body
                )
            except httpx.TransportError:
                if attempt == RECORD_ATTEMPTS:
                    raise
            else:
                if response.status_code < 500 or attempt == RECORD_ATTEMPTS:
                    response.raise_for_status()
                    return
            await asyncio.sleep(RECORD_BACKOFF_SECONDS * 2 ** (attempt - 1))
