"""Serve one interceptor service and one scripted upstream, for a driver in another process to dial.

Run as ``python -m tests.remote_interceptor_service``: the process prints one
JSON line with the service's ``url`` and the upstream's ``upstream_url`` once
both listen, then serves until SIGTERM or SIGINT. The listener test starts it
so the driver under test holds the only sockets that test counts: the service
and the upstream are the child's.
"""

from __future__ import annotations

import asyncio
import json
import signal

from midojo.interception_service import InterceptorService
from tests.binding_driver import serve_app
from tests.fixtures.scripted_target.upstream import scripted_upstream

CREDENTIAL = "the-bind-credential"


async def main() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for name in ("SIGTERM", "SIGINT"):
        loop.add_signal_handler(getattr(signal, name), stop.set)
    service = InterceptorService(CREDENTIAL)
    async with serve_app(scripted_upstream().http_app(path="/mcp")) as listener:
        url = await service.start()
        print(json.dumps({"url": url, "upstream_url": listener.url("/mcp")}), flush=True)
        try:
            await stop.wait()
        finally:
            await service.stop()


if __name__ == "__main__":
    asyncio.run(main())
