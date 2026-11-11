"""The MiDojo man-in-the-middle interceptor: the service, the MCP face, the placement engine and the link's service end.

Import what you use from its module; the package re-exports nothing.

- ``service``: the interceptor service, one listener with one MCP endpoint per binding.
- ``mcp``: the MCP face, ``MidojoMCP``, over a real MCP server.
- ``placement``: the engine that places an injection plan's payloads in a forwarded result.
- ``link``: the service's end of the ``midojo-binding-v1`` stream, ``Link`` with its recorder and world.
- ``endpoint`` and ``listener``: the tool endpoint's path and the server plumbing the service shares with a host.

The contracts it speaks (the plan, the binding frames, the recorder and world seams, the tool policy) are
``midojo_spec``'s, and nothing here imports MiDojo's harness: its verifiers, runtimes and injection spec.
"""
