# midojo-sdk

Build [MiDojo](https://github.com/asago-ai/midojo) interception layers in Python. An interception layer stands between an agent and its tools during an evaluation: it reads and updates the evaluation's simulated environment, and attempts to record every call the agent makes, by default on the MiDojo control plane.

```sh
pip install "midojo-sdk[mcp]"        # fake MCP servers
pip install "midojo-sdk[langchain]"  # LangChain agents
```

## Fake MCP servers

`midojo_sdk.mcp.MidojoMCP` builds a fake MCP server on [FastMCP](https://gofastmcp.com). Each tool takes a `ToolContext` as its first parameter. The SDK passes it in and hides it from the tool schema the agent sees.

```python
import os

from midojo_sdk.mcp import MidojoMCP, ToolContext

mcp = MidojoMCP(
    "alerts",
    control_plane_url=os.environ["MIDOJO_URL"],
    upstream_url="http://localhost:8081/mcp",  # optional: the real MCP server
)


@mcp.tool()
async def get_alerts(ctx: ToolContext, city: str) -> str:
    """Get the weather alerts for a city."""
    real = await ctx.forward("get_alerts", {"city": city})
    return real + await ctx.env("injected_alert")


if __name__ == "__main__":
    mcp.run(transport="http", port=8082)
```

- `await ctx.env(field)` reads a field of the evaluation's environment, and `await ctx.read()` the whole environment with its revision
- `await ctx.env_update(field, value)` sets one field in one atomic step, so a concurrent change to another field isn't lost. Pass `expected_revision=` the revision `ctx.read()` returned to apply it only if nothing changed the environment since: it returns `False` otherwise
- `await ctx.forward(tool, args)` calls the tool on the upstream MCP server and returns its text

The SDK attempts to record each call with its arguments and the result the agent saw. The evaluation session comes from the incoming request's `X-Midojo-Session` header, or else from the `MIDOJO_SESSION_TOKEN` environment variable.

`await mcp.mirror_upstream()` exposes every upstream tool the server doesn't define itself; the upstream is `upstream_url`, or `upstream_transport` for anything else a fastmcp `Client` connects to. Each downstream session gets one upstream connection, opened on its first call and closed when the session ends. A mirrored tool is listed exactly as the upstream lists it (name, description, input and output schemas, annotations, `_meta`), and a call gets the upstream's result as sent: content blocks, structured content, error flag, and a JSON-RPC error as the same JSON-RPC error. `mcp.block(*names)` and `@mcp.deny(name)` refuse calls: the tool never runs and the agent gets an error result, "Tool execution was blocked" or the reason the denial check returned. Each call has a disposition: `forwarded` (mirrored), `simulated` (a tool defined with `@mcp.tool()`), `blocked` or `denied`. The control plane records each call's disposition, with the `blocked` flag it implies (see [the control plane API](../README.md#control-plane-api)).

## Injection plans

Pass a frozen `InjectionPlan` as `MidojoMCP(..., plan=plan)`. `append` is the one default because an unqualified content selector can preserve the upstream result without guessing a field. The interceptor supports exact content-block and structured-content selectors, and `mirror_upstream()` validates targeted tool names before serving calls.

Unsupported placements raise `CapabilityError` with every gap. Strict selector and size mismatches leave the result unchanged; `fallback_append` is explicit. `mcp.injection_outcomes` reports every result.

Matching calls are numbered per evaluation under a lock before forwarding. Concurrent calls therefore use interceptor-admission order, not completion order.

## Recorder and WorldState

`midojo_sdk.recorder` defines the two seams an interception layer talks to. A `Recorder` starts and finishes the calls of one evaluation, each with its disposition. A `WorldState` reads the evaluation's simulated environment (`read()`: its revision and fields) and changes it (`change(values, expected_revision=None)`: one atomic step, or `None` when `expected_revision` is stale). `midojo_sdk.client.AgentControlPlaneClient` implements both over HTTP for one session; `midojo.control_plane.inprocess.InProcessAgentClient` implements both in the control plane's own process, with no HTTP.

`MidojoMCP(name, recorder=..., world=...)` records through `recorder` and gives simulated tools `world`; without them it uses the control plane at `control_plane_url` for the request's session. Forwarding (`mirror_upstream`) needs only a recorder: a server built with a recorder and no `control_plane_url` has no environment, and `ctx.env` raises. Both implementations raise `midojo_sdk.recorder.ControlPlaneRejectedError` when the control plane refuses an event, with its `kind` (a `RejectionKind`, such as `call_conflict` or `session_closed`) and `late_fact`, and `midojo_sdk.recorder.InvalidRequestError` for a request the API can't take as sent, such as a `call_id` over 256 characters (over HTTP it's the 422, and also an `httpx.HTTPStatusError`); over HTTP, any other failure is an `httpx` error. Each recorder sends an event once: wrap one in `RetryingRecorder(recorder, retryable=midojo_sdk.client.transient)` to resend after a transport error or a 5xx, as `MidojoMCP` does with the control plane's. A rejection or an invalid request is never resent, whatever `retryable` says. `ControlPlaneRecorder(client)` and `ToolContext(client=...)` still work, deprecated.

A record is what the agent asked for and saw, not proof of an effect; the environment is what simulated tools did; what a forwarded call did to a real system is neither, and a runtime observes it. Nothing makes them one transaction: a record, an environment change and a real tool's effect each commit, or fail, on their own.

## LangChain agents

`midojo_sdk.langchain.MidojoMiddleware` is [agent middleware](https://docs.langchain.com/oss/python/langchain/middleware/overview) that intercepts the tool calls of a [LangChain](https://docs.langchain.com) agent. It picks tools by name, like the PI SDK's `hooks`, `blockTools` and `reportTools`. The agent keeps its real tools, so it sees the same tool names, descriptions and arguments as it does without midojo.

```python
import os

from langchain.agents import create_agent
from midojo_sdk.langchain import MidojoMiddleware, ToolContext

from my_agent.tools import get_alerts, read_file, send_email

midojo = MidojoMiddleware(control_plane_url=os.environ["MIDOJO_URL"])


@midojo.hook("get_alerts")
async def inject_alert(ctx: ToolContext, args: dict, real_result: str) -> str:
    return real_result + await ctx.env("injected_alert")


midojo.block("send_email")
midojo.report("read_file")

agent = create_agent(model, tools=[get_alerts, read_file, send_email], middleware=[midojo])  # model: your chat model
```

- `hook(name)`: the real tool runs, and the hook's return value is what the agent gets. When the hook raises, the agent gets the error message as an error result, and the run goes on. The tool must return a message: hooking a tool that returns a LangGraph `Command` stops the run with a `TypeError`.
- `block(*names)`: the tools never run. The call is recorded as blocked, and the agent gets "Tool execution was blocked" as an error result. Use it for a tool whose effect must not happen during an evaluation, such as sending email.
- `report(*names)`: the tools run unchanged, and each call is recorded with its result, so midojo can see what a tool returned, such as a file read, without perturbing the agent.

Calls to these tools are recorded on the control plane, and the agent's other tools run as usual. Run the agent with `ainvoke` or `astream`: the middleware is asynchronous. For a [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) graph with its own `ToolNode`, pass `ToolNode(tools, awrap_tool_call=midojo.awrap_tool_call)`. The evaluation session comes from `MIDOJO_SESSION_TOKEN`, or from `midojo_sdk.session.session_context()` for an agent that serves several evaluations.

## Records

Each record carries a `call_id`: the LangChain tool call's ID, namespaced as `langchain:<UUID>:<id>`, or a fresh UUID for a fake MCP server tool, a call without an ID, or an ID the middleware has already used. A record that fails with a transport error or a 5xx is resent with the same `call_id`, up to 3 sends in all, so the control plane records it at most once (`RetryingRecorder` for a fake MCP server; the client's `record_function_call` and `start_function_call` for the LangChain middleware). A 4xx, including a 409, is raised without a resend. The [interception README](https://github.com/asago-ai/midojo/blob/main/interception/README.md#record-ids-and-retries) describes the guarantee.

A call that runs a tool (a fake MCP server tool, or a hooked or reported LangChain tool) is started on the control plane before the tool runs and finished after, with the same `call_id`. When the start fails, the tool doesn't run; when the finish fails, the call stays recorded as unfinished. A blocked call never runs, so it's recorded once. See [Started and unfinished calls](https://github.com/asago-ai/midojo/blob/main/interception/README.md#started-and-unfinished-calls).

## More

The weather suite's [`fake_mcp.py`](https://github.com/asago-ai/midojo/blob/main/suites/weather/sandbox_pi/fake_mcp.py) is a complete example, and the [interception README](https://github.com/asago-ai/midojo/blob/main/interception/README.md) describes the control plane API the SDK calls.
