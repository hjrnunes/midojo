# Document Assistant — Example Sandbox

An example OpenShell sandbox for the `document_assistant` suite. Extends the
[pi community sandbox](https://github.com/NVIDIA/OpenShell-Community/tree/main/sandboxes/pi)
with an OpenAI-compatible model provider. By default it uses a local model
server, so no cloud API key is required.

## What this is

This is the **example agent** for the PoC — equivalent to the A2A agent in the
minibank suite. It lets you run the suite against a local model
server without any cloud credentials.

**Replace it with your own production sandbox image** when you're red-teaming a
real agent. The `suite.yaml` evaluation (workspace files, injection tasks,
predicates) does not change — only the `image:` field under `agent_runtime`.

## Prerequisites

- OpenShell gateway running locally (`openshell status` shows Connected)
- An OpenAI-compatible model server, by default at `localhost:8321` (e.g. Llama Stack, vLLM, Ollama) serving `ollama/qwen3.5:2b`

## Build

```bash
# from repo root (mind the trailing ".")
podman build --pull=always --platform linux/arm64 \
  -t localhost/document-assistant-local:latest \
  -f suites/document_assistant/sandbox_pi/Containerfile .
```

On Apple Silicon, use a native ARM64 image as above: under x86 emulation,
OpenShell identifies QEMU as the process, so the Node binary allowlist does not
match. Drop `--platform linux/arm64` on x86 hosts.

To use another model server or model, pass build args:

```bash
podman build --pull=always --platform linux/arm64 \
  --build-arg LITELLM_API_URL=https://your-model-server.example.com/v1 \
  --build-arg LITELLM_MODEL=your-model-id \
  -t localhost/document-assistant-local:latest \
  -f suites/document_assistant/sandbox_pi/Containerfile .
```

| Build arg | Default | Notes |
|---|---|---|
| `LITELLM_API_URL` | `http://host.openshell.internal:8321/v1` | `host.openshell.internal` resolves to the host from inside the sandbox |
| `LITELLM_MODEL` | `ollama/qwen3.5:2b` | must match what the server advertises |

The API key is not baked into the image. The suite passes `LITELLM_API_KEY`
into the sandbox when it is set.

## Run

```bash
uv run midojo-serve --load-suite document_assistant --host 127.0.0.1 --port 8090
uv run midojo-run --suite document_assistant --gateway GATEWAY_NAME \
  --control-url http://localhost:8090
```

`GATEWAY_NAME` is the gateway registered with the `openshell` CLI. For a model
server other than the default, set `LITELLM_API_HOST` and `LITELLM_API_PORT`
for `midojo-run` so the sandbox network policy allows it, and `LITELLM_API_KEY`
if the server needs one.

Each run gets a workspace such as `midojo-docum-k7p2xa`, with a shortened suite
name and a six-character suffix (retried if already taken). Each evaluation gets
a 10-character hexadecimal ID, checked for uniqueness across runs in the control
plane and used in the API and logs. As the sandbox gets no session, its name and
labels name no run, evaluation or task either: it is named `sbx-<hex>`, a random
name, and labelled only with `midojo.suite`, and `midojo-run` logs which sandbox
runs each evaluation. Both names fit OpenShell's 19-character limit.

Workspace labels retain `midojo.suite` and `midojo.run-id`. Session tokens are
never put in names or labels.

## How injections are observed

OpenShell observes the agent from outside the sandbox: files created,
modified or deleted in `/sandbox/workdir` (and new files' contents), the
processes it launched, its network calls, allowed or blocked, and security
findings. Every check of this suite reads those observations, so nothing in
the sandbox records the agent's tool calls, and the suite sets
`inject_session: false`: the sandbox gets no evaluation session, no
control-plane URL and no control token, and its network policy has no rule
for the control plane, which the sandbox never reaches. The control plane can
then listen on loopback (`--host 127.0.0.1`).

The evaluation's `function_calls` evidence is `incomplete` (no layer recorded
a call), which no check of this suite reads. A seeded file's payload has no
delivery fact (no interceptor placed it), so MiDojo reports its exposure as
unknown.

## Bring your own agent

Any OpenShell sandbox image that runs PI works as a drop-in replacement:

```yaml
# suite.yaml
agent_runtime:
  type: openshell
  image: my-production-pi-sandbox   # ← change this only
  agent_command: ["pi", "-p", "--no-session"]
```

The sandbox must have PI pre-configured to connect to whatever inference
endpoint it uses. The suite does not inject any agent configuration.
