# Shared image for midojo + its suites. One image, many entrypoints — each
# Deployment/Job overrides command/args (midojo-serve, minibank-real-mcp-serve,
# minibank-a2a-agent, midojo-run).
#
# Build & push to a registry your cluster can pull from, e.g.:
#   podman build -t quay.io/<you>/midojo:latest -f Containerfile .
#   podman push  quay.io/<you>/midojo:latest
# then point suites/minibank/deploy/kustomization.yaml at that image.
FROM python:3.12-slim

WORKDIR /app
COPY . /app

# Install midojo plus the suites' agent dependencies (openai, ogx, ...). The
# host-side interceptor midojo-run starts (--interception host) and the suites'
# host tools use midojo-sdk, installed from its folder.
RUN pip install --no-cache-dir "./interception/python-sdk[mcp]" ".[suites]" \
    # Make the tree arbitrary-UID friendly (OpenShift restricted SCC runs the
    # container as a random non-root UID with GID 0).
    && chgrp -R 0 /app && chmod -R g=u /app

# Default to a non-root UID for vanilla K8s; OpenShift overrides this anyway.
USER 1001

# 8000 the agent, 8080 the control plane, 8083 the bank's real MCP server, 8766 the
# host-side interceptor midojo-run serves the agent's tools on.
EXPOSE 8000 8080 8083 8766
