"""HTTP client for the control plane's /agent API."""

from __future__ import annotations

import os
import uuid
from typing import Any

import httpx

from midojo_sdk.recorder import (
    RECORD_ATTEMPTS,
    RECORD_BACKOFF_SECONDS,
    REFUSED,
    ControlPlaneRejectedError,
    Disposition,
    EnvironmentRevision,
    InvalidRequestError,
    RejectionKind,
    retry,
)

__all__ = [
    "CALL_ID_MAX_LENGTH",
    "RECORD_ATTEMPTS",
    "RECORD_BACKOFF_SECONDS",
    "REVISION_HEADER",
    "AgentControlPlaneClient",
    "CallIds",
    "InvalidHTTPRequestError",
    "transient",
]

CALL_ID_MAX_LENGTH = 256
"""The longest ``call_id`` the control plane accepts."""

REVISION_HEADER = "X-Midojo-Environment-Revision"
"""The response header that carries the environment's revision."""


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


class InvalidHTTPRequestError(InvalidRequestError, httpx.HTTPStatusError):
    """The API's 422 for a request it can't take as sent: an :class:`InvalidRequestError` and an HTTP error."""

    def __init__(self, detail: str, *, request: httpx.Request, response: httpx.Response) -> None:
        httpx.HTTPStatusError.__init__(self, f"invalid request: {detail}", request=request, response=response)
        self.detail = detail


def transient(error: Exception) -> bool:
    """Whether resending a request that failed with ``error`` may succeed.

    A transport error is, and so is a 5xx response: a proxy's, or a server
    error. A :class:`~midojo_sdk.recorder.ControlPlaneRejectedError` never is,
    nor any other 4xx.
    """
    if isinstance(error, httpx.TransportError):
        return True
    return isinstance(error, httpx.HTTPStatusError) and error.response.status_code >= 500


class AgentControlPlaneClient:
    """Access interception endpoints for one evaluation session.

    The HTTP :class:`~midojo_sdk.recorder.Recorder` and
    :class:`~midojo_sdk.recorder.WorldState`, bound to the session token's
    evaluation. Their methods send each request once. A response that carries
    one of the control plane's rejection kinds raises
    :class:`~midojo_sdk.recorder.ControlPlaneRejectedError`; any other failure
    (a transport error, a proxy's error page, a request the API refuses before
    the control plane decides anything, such as one without a token) raises as
    an ``httpx`` error, which :func:`transient` classifies for a
    :class:`~midojo_sdk.recorder.RetryingRecorder`.
    """

    def __init__(self, base_url: str, session_token: str, *, http: httpx.AsyncClient) -> None:
        self._base_url = base_url.rstrip("/")
        self._http = http
        self._headers = {"Authorization": f"Bearer {session_token}"}

    # --- WorldState ---

    async def read(self) -> EnvironmentRevision:
        response = await self._http.get(f"{self._base_url}/agent/environment", headers=self._headers)
        _raise_for_status(response)
        return EnvironmentRevision(int(response.headers[REVISION_HEADER]), response.json())

    async def change(
        self, values: dict[str, Any], *, expected_revision: int | None = None
    ) -> EnvironmentRevision | None:
        response = await self._http.patch(
            f"{self._base_url}/agent/environment",
            json={"set": values, "expected_revision": expected_revision},
            headers=self._headers,
        )
        try:
            _raise_for_status(response)
        except ControlPlaneRejectedError as e:
            if e.kind is RejectionKind.REVISION_CONFLICT and expected_revision is not None:
                return None
            raise
        return EnvironmentRevision(int(response.headers[REVISION_HEADER]), response.json())

    async def get_environment(self) -> dict[str, Any]:
        """The environment's fields, without its revision."""
        return (await self.read()).environment

    async def put_environment(self, environment: dict[str, Any]) -> None:
        """Replace the whole environment, whatever changed it since it was read.

        Prefer :meth:`change`: a read followed by a replace loses any change
        made in between.
        """
        response = await self._http.put(f"{self._base_url}/agent/environment", json=environment, headers=self._headers)
        _raise_for_status(response)

    # --- Recorder ---

    async def start_call(self, *, call_id: str, function: str, args: dict[str, Any], disposition: Disposition) -> None:
        """Send a call's start once.

        The API has no field for the disposition: a started call is forwarded
        or simulated, never blocked, so the start carries nothing for it.
        """
        await self._post("/agent/function-calls/start", {"call_id": call_id, "function": function, "args": args})

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
        """Send a call's finish once.

        The API has no field for the disposition, so the record carries only
        what the disposition implies: ``blocked`` for a denied or blocked call.
        """
        body = _finish_body(call_id, function, args, result, error, disposition in REFUSED)
        await self._post("/agent/function-calls", body)

    # --- the /agent API, with resends ---

    async def start_function_call(self, *, function: str, args: dict, call_id: str) -> None:
        """Record a call's start: an unfinished record that ``record_function_call`` finishes.

        Send it before the tool runs, then finish the call with the same
        ``call_id``, ``function`` and ``args``. If the finish never arrives,
        the call stays on the control plane as unfinished. Resent like
        ``record_function_call``; a resent start doesn't record the call twice.
        """
        body = {"call_id": call_id, "function": function, "args": args}
        await retry(lambda: self._post("/agent/function-calls/start", body), retryable=transient)

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

        When ``call_id`` was started with ``start_function_call``, this
        finishes that record. A :func:`transient` failure is retried with the
        same ``call_id``, up to ``RECORD_ATTEMPTS`` sends in all, so resending
        a record that reached the control plane doesn't record it twice. This
        is at most one record per ``call_id``, not exactly-once: a record whose
        sends all fail is missing, or left unfinished when it was started. A
        rejection, such as ``call_conflict`` for a ``call_id`` recorded with
        another body, is raised at once, as is any other 4xx.
        """
        body = _finish_body(call_id or str(uuid.uuid4()), function, args, result, error, blocked)
        await retry(lambda: self._post("/agent/function-calls", body), retryable=transient)

    async def _post(self, path: str, body: dict[str, Any]) -> None:
        response = await self._http.post(f"{self._base_url}{path}", headers=self._headers, json=body)
        _raise_for_status(response)


def _finish_body(
    call_id: str, function: str, args: dict[str, Any], result: str, error: str | None, blocked: bool
) -> dict[str, Any]:
    return {
        "call_id": call_id,
        "function": function,
        "args": args,
        "result": result,
        "error": error,
        "blocked": blocked,
    }


def _raise_for_status(response: httpx.Response) -> None:
    """Raise the control plane's rejection as a ``ControlPlaneRejectedError``, and any other error as ``httpx``'s.

    Only a response whose body names one of the control plane's rejection
    kinds is a rejection. A 422 whose ``detail`` is a list of errors, the
    API's answer to a malformed request, is an :class:`InvalidHTTPRequestError`.
    Anything else, such as a proxy's 503 or a body that isn't JSON, is an HTTP
    error like any other.
    """
    if response.is_success:
        return
    try:
        body = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and isinstance(body.get("kind"), str):
        try:
            kind = RejectionKind(body["kind"])
        except ValueError:
            pass
        else:
            detail = body.get("detail")
            late_fact = body.get("late_fact") is True
            raise ControlPlaneRejectedError(
                kind, detail if isinstance(detail, str) else str(detail), late_fact=late_fact
            )
    if response.status_code == 422 and isinstance(body, dict) and isinstance(body.get("detail"), list):
        raise InvalidHTTPRequestError(str(body["detail"]), request=response.request, response=response)
    response.raise_for_status()
