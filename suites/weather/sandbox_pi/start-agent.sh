#!/bin/sh
# Starts the weather agent: points PI at the model and at its weather tools,
# then runs the agent command passed as arguments.
set -e

# The suite passes LITELLM_API_URL and LITELLM_MODEL into the sandbox at run
# time, like LITELLM_API_KEY, so one image works with any OpenAI-compatible
# model server. MiDojo runs the weather suite in host mode (midojo-run
# --interception host), and passes the agent its tool endpoint,
# MIDOJO_TOOL_URL and MIDOJO_TOOL_TOKEN: the URL and bearer token of the
# interceptor that serves the agent's weather tools on the host. The sandbox
# gets nothing else of the evaluation: no session token, no control-plane URL.
#
# PI can't read them itself: it expands environment variables only in apiKey
# and headers, not in a provider's baseUrl or model IDs, or an MCP server's
# URL. Fill the @NAME@ placeholders in PI's config, JSON-escaped. The config
# is in PI's global agent directory, outside the working directory MiDojo
# observes.
: "${LITELLM_API_URL:?LITELLM_API_URL is not set}"
: "${LITELLM_MODEL:?LITELLM_MODEL is not set}"
: "${MIDOJO_TOOL_URL:?MIDOJO_TOOL_URL is not set: run the weather suite with midojo-run --interception host}"
: "${MIDOJO_TOOL_TOKEN:?MIDOJO_TOOL_TOKEN is not set: run the weather suite with midojo-run --interception host}"
/sandbox/.venv/bin/python3 - /sandbox/.pi/agent/models.json /sandbox/.pi/agent/settings.json /sandbox/.pi/agent/mcp.json <<'PY'
import json
import os
import sys

for path in sys.argv[1:]:
    with open(path) as f:
        text = f.read()
    for name in ("LITELLM_API_URL", "LITELLM_MODEL", "MIDOJO_TOOL_URL", "MIDOJO_TOOL_TOKEN"):
        text = text.replace(f"@{name}@", json.dumps(os.environ[name])[1:-1])
    with open(path, "w") as f:
        f.write(text)
PY

exec "$@"
