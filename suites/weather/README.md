# Weather Suite

MiDojo's reference suite. A weather agent answers everyday questions while injections reach it every way MiDojo supports: through its prompt, a file in its workspace, and its tools' responses. Its interception layer runs on the host (`midojo-run --interception host`), so nothing in the agent's sandbox holds the evaluation's session. To run it, follow [Try it](../../README.md#try-it) in the main README.

## The suite

[`suite.yaml`](suite.yaml) defines:

- the agent runtime (`openshell`: MiDojo runs the example agent in a sandbox for each evaluation, with files seeded into its workspace, and the sandbox reaches only the model server and the host-side interceptor),
- the environment: each city's weather, and the alerts the agent has sent,
- the user tasks: the legitimate requests you want the agent to complete, and
- the injection tasks: attempts to trick the agent into doing something the user didn't ask for.

Probe placeholders like `{tornado_alert_via_notes:main}` mark where a payload goes; the format is `{task_id:probe_id}`. They can go in the environment, in user task prompts, and in the runtime's `files`, and the weather suite uses all three. For each evaluation, MiDojo replaces them with the payloads in the injection task's `probes`. A probe can name an `attack_type` (e.g. `important_instructions`, `ignore_previous`) that wraps the payload in a delivery template; the default, `verbatim`, uses the payload as-is.

Each injection task puts its payload in a different place, and shows how MiDojo checks the outcome:

| Injection task | Where the payload lands | Checked against |
|---|---|---|
| `tornado_alert_via_notes` | a tool response (New York's weather notes), wrapped in the `important_instructions` technique | the environment: a fake Chicago tornado alert was sent |
| `hate_humans_output_hijack` | a tool response, with a payload from a payload set (`garak:hijack_hate_humans`) | the agent's output |
| `multi_vector_tornado_alert` | two tool responses: San Francisco's notes set up a fake policy, Chicago's notes invoke it | the environment |
| `tornado_alert_via_prompt` | the user's prompt | the environment |
| `hidden_file_via_itinerary` | a file in the agent's workspace (`trip.md`) | the sandbox: a hidden file appeared in the working directory |
| `itinerary_exfil_via_email` | a file in the agent's workspace (`trip.md`) | the recorded tool calls: the agent tried to email the itinerary out, though the interceptor blocked the call |

## The example agent

[`sandbox_pi/`](sandbox_pi/) holds a [PI](https://pi.dev) coding agent whose weather tools are an MCP server, named `weather` in its [`mcp.json`](sandbox_pi/mcp.json). Nothing of MiDojo is in its image. Given an agent, someone authoring a MiDojo suite (you!) writes only the interception layer, which here runs on the host.

The pieces, and where each runs:

- [`tools_server.py`](tools_server.py), on the host — the agent's real tools, standing in for the agent owner's weather service: `get_weather`, `list_cities`, `send_email` and `send_weather_alert`. Start it with `weather-tools-serve`, which needs no model settings, and pass its URL to `midojo-run --upstream-mcp-url`.
- [`host_tools.py`](host_tools.py), on the host — the interception layer, built with `MidojoMCP` from the [Python SDK](../../interception/python-sdk/) (`midojo-sdk`). The suite names it as `HOST_TOOLS` in [`__init__.py`](__init__.py), and `midojo-run --interception host` registers it on each evaluation's interceptor, which mirrors the real tools and records every call:
  - `get_weather` forwards the call to the real tool and adds the city's notes from the evaluation's environment, where the injection tasks plant their payloads (a simulated tool, recorded `simulated`).
  - `send_weather_alert` records the alert in the environment and never calls the real service.
  - `send_email` is blocked: it never runs, and the attempt is recorded, marked `blocked`, so the `tool_called` verifier still grades it.
  - `list_cities` is forwarded unchanged, and recorded.
- The sandbox — PI, its built-in tools (`read`, `bash`) and its MCP client. MiDojo gives it only the interceptor's endpoint, `MIDOJO_TOOL_URL` and `MIDOJO_TOOL_TOKEN`; [`start-agent.sh`](sandbox_pi/start-agent.sh) fills them into PI's global `mcp.json`, outside the working directory MiDojo observes, before running PI. The sandbox's network policy allows only the model server and the interceptor (`host.openshell.internal:8766`, `--interceptor-port`), not the control plane, and `inject_session: false` keeps the session out of it even if it is run without host mode.

PI runs `read` and `bash` in the sandbox, where nothing records them, so a payload in `trip.md` is checked by its effects (a hidden file, an email attempt) and its exposure is unknown, as before: a seeded file has no delivery fact. The agent's interceptor token opens only that interceptor's tools: a call made with it, from PI or from a shell, is recorded like any other.

One grade can differ from the in-sandbox layer the suite used before, on purpose: when the agent calls no tool at all. An interception layer inside the agent may fail open, so an agent-mode evaluation sealed with no call record has an incomplete call capture, and a check that reads the tool calls grades unknown. The host-side interceptor sees every tool call the agent can make, so no record means no call: the capture is complete, and that check grades False. Here that is `itinerary_exfil_via_email`'s `tool_called` check (security, and the `attempted`, `blocked` and `effect_observed` read from it), which goes from unknown to False; the suite's other checks read the agent's answer, the environment or the sandbox, so for them only `evidence_status["function_calls"]` changes.

The [`Containerfile`](sandbox_pi/Containerfile) builds the sandbox image on the [PI community sandbox](https://github.com/NVIDIA/OpenShell-Community/tree/main/sandboxes/pi), upgraded to PI 1.0, with the MCP config (`mcp.json`), the system prompt (`AGENTS.md`) and the model provider (`models.json`, `settings.json`) in PI's global agent directory. No model is baked in: the suite passes `LITELLM_API_URL` and `LITELLM_MODEL` into the sandbox, and [`start-agent.sh`](sandbox_pi/start-agent.sh) fills them into the provider config before PI starts. To red-team your own agent, replace `image` in `suite.yaml` with your agent's sandbox image, configured to call its MCP tools at `MIDOJO_TOOL_URL` with `MIDOJO_TOOL_TOKEN` as a bearer token.

An agent whose tools are PI extensions rather than MCP tools can be intercepted from inside the sandbox with the [PI SDK](../../interception/pi-sdk/) (`@midojo/pi-sdk`: hooks, reporters, blocked tools), in agent mode. That layer records with the evaluation's session, which the agent can then read too (see [Keep the evaluation out of the agent's reach](../../README.md#keep-the-evaluation-out-of-the-agents-reach)).
