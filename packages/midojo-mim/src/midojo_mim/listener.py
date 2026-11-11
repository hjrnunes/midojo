"""The listener plumbing every interceptor server shares: one socket, one uvicorn server, one refusal.

The interceptor service and the driver's in-process interceptor both serve an
ASGI app on a socket they bind themselves, so a taken port raises here and not
in the server's task.
"""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Iterator

import uvicorn
from starlette.types import Send

WILDCARD_HOSTS = {"0.0.0.0", "::", ""}
"""Hosts that mean every interface: a listener can bind them, but a client can't connect to them."""

SHUTDOWN_SECONDS = 2
"""How long stopping waits for open connections before closing them."""


class Server(uvicorn.Server):
    """A uvicorn server that leaves the process's signal handlers to its host.

    uvicorn's own handlers would take the host's Ctrl-C. And once a stopped
    server's handler is left installed (two servers running at once restore
    each other's), sse_starlette reads that server's ``should_exit`` and ends
    every event stream in the process, the next servers' included.
    """

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def listen(host: str, port: int) -> socket.socket:
    """A socket bound to ``host`` and ``port`` (0 picks a free one), reusable at once by the next server on it."""
    family, kind, proto, _, address = socket.getaddrinfo(
        host or None, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
    )[0]
    listener = socket.socket(family, kind, proto)
    try:
        # Each evaluation of a run may listen on the same fixed port as the last one did.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(address)
    except OSError:
        listener.close()
        raise
    return listener


async def respond(send: Send, status: int, body: bytes) -> None:
    """Answer with a plain-text ``body``; a 401 also names the scheme, so a client reads a refusal of its credential."""
    headers = [(b"content-type", b"text/plain; charset=utf-8")]
    if status == 401:
        headers.append((b"www-authenticate", b"Bearer"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


__all__ = ["SHUTDOWN_SECONDS", "WILDCARD_HOSTS", "Server", "listen", "respond"]
