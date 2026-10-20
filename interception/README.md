# Interception SDKs

An interception layer sits between an agent and its tools during an evaluation. It decides what the agent's tool calls return, which is where MiDojo places injections, and it attempts to record every call it intercepts on the MiDojo control plane so verifiers can grade the run. Suite authors build one per agent with an SDK:

| SDK | Package | Intercepts |
|---|---|---|
| [`pi-sdk/`](pi-sdk/) | `@midojo/pi-sdk` (npm) | the tools of a [PI](https://pi.dev) coding agent, as a PI extension |
| [`python-sdk/`](python-sdk/) | `midojo-sdk` (PyPI) | an MCP server, as a fake MCP server built on FastMCP (`midojo_sdk.mcp`) |
| [`python-sdk/`](python-sdk/) | `midojo-sdk` (PyPI) | the tool calls of a [LangChain](https://docs.langchain.com) agent, as agent middleware (`midojo_sdk.langchain`) |

The SDKs depend only on the control plane's HTTP API, described below, and not on the `midojo` package. An SDK for another framework or language implements the same calls.

## How the SDKs intercept tools

The SDKs intercept tools the same way, so a suite's evaluation means the same thing whichever agent it targets. An SDK for another framework follows the same rules:

- **The agent keeps its real tools.** An SDK attaches to the agent's tools by name instead of replacing them, so the agent sees the same tool names, descriptions and arguments as it does without MiDojo. Only a tool the agent doesn't have, such as a fake notification tool a suite adds, gets its own definition.
- **A tool is hooked, blocked or reported:**
  - *hook*: the real tool runs, and the hook's return value is what the agent gets. The hook can read and update the evaluation's environment.
  - *block*: the tool never runs, and the agent gets `Tool execution was blocked` as an error result. Use it for a tool whose effect must not happen during an evaluation, such as sending email.
  - *report*: the tool runs, and the agent gets its result unchanged.
- **The SDK attempts to record every intercepted call,** with `POST /agent/function-calls`: the agent's arguments, the result the agent saw, the error if there was one, and whether the call was blocked. Each record carries a `call_id`, so the SDK can resend a record whose response it lost without recording the call twice; a record that never arrives is still missing (see [Record IDs and retries](#record-ids-and-retries)). Tools that aren't intercepted run as usual and aren't recorded.
- **A call that runs a tool is started first.** Before the tool runs, the SDK records the call's start with `POST /agent/function-calls/start`, and after, it finishes that record. A call that starts and never finishes, because the tool hangs, the agent stops, or the finish never arrives, stays visible as unfinished (see [Started and unfinished calls](#started-and-unfinished-calls)). A blocked call never runs, so it's recorded once, without a start.
- **Errors don't end the run.** When a hook raises, the call is recorded with the error, and the agent gets the message as an error result. The agent carries on as it would after any failed tool call.
- **No session, no call.** An intercepted call needs the evaluation's session. Without one, the SDK raises an error instead of letting the call go unrecorded.

Where each SDK stands:

| | PI (`@midojo/pi-sdk`) | LangChain (`midojo_sdk.langchain`) | MCP (`midojo_sdk.mcp`) |
|---|---|---|---|
| The agent keeps its real tools | yes | yes | no: the fake server defines each tool again, and `ctx.forward` calls the real one |
| Hook, block, report | `hooks`, `blockTools`, `reportTools` | `hook`, `block`, `report` | no: each tool is a fake that can forward to the real one |
| Errors come back as error results | no: a hook's or fake tool's error reaches the agent as plain text | yes | yes |
| Report records the tool's error | no: it always records `error: null` | yes | no report |

## The evaluation session

The control plane issues a session token for each evaluation. An interception layer sends it on every call as `Authorization: Bearer <token>`, and the token selects the evaluation's environment and call record. How the layer gets the token depends on how MiDojo reaches the agent:

- The `openshell` runtime sets `MIDOJO_SESSION_TOKEN` and `MIDOJO_URL` (the control plane URL) in the sandbox's environment. The SDKs read them.
- For an `unmanaged` agent, `midojo-run` sends the token in the `X-Midojo-Session` request header. `midojo_sdk.mcp` reads the header from the incoming MCP request, and `midojo_sdk.session` has helpers for agents that pass it on to their own tool servers.

## Control plane API

All paths are relative to the control plane URL and take the session's bearer token.

| Call | Body | Returns |
|---|---|---|
| `GET /agent/environment` | | the evaluation's environment, as JSON |
| `PUT /agent/environment` | the whole environment | the stored environment. The control plane validates it against the suite's environment schema |
| `POST /agent/function-calls/start` | `{"call_id", "function", "args"}`; `call_id` is required | the started call, unfinished: 201 when it's new, 200 when `call_id` is already recorded with the same `function` and `args`, 409 when it's recorded with others |
| `POST /agent/function-calls` | `{"call_id", "function", "args", "result", "error", "blocked"}` | the recorded call: 201 when it's new or finishes a started call, 200 when `call_id` is already finished with the same body, 409 when it's finished with a different body or was started with another `function` or `args` |
| `GET /agent/function-calls` | | the calls recorded so far, finished or not, in the order they started |

In a function-call record, `result` is what the agent saw. `error` (optional) is the error the tool raised. `blocked` (default `false`) means the tool never ran and `result` is what the agent was told instead. `complete` (in responses) is `false` while a started call is unfinished; its `result` is then `null`.

## Record IDs and retries

`call_id` (optional, 1 to 256 characters) identifies a record within its evaluation. The control plane keeps at most one record per `call_id` per evaluation, across `POST /agent/function-calls` and the orchestrator's `POST /runs/{run_id}/evaluations/{eval_id}/function-calls`:

| The evaluation already has | Status | Effect |
|---|---|---|
| no record with this `call_id` | 201 | the call is recorded |
| a record with this `call_id` and the same `function`, `args`, `result`, `error` and `blocked` | 200 | nothing is appended; the response is the stored record |
| a record with this `call_id` and a different body | 409 | nothing is appended |

A record sent without a `call_id` is always appended, and the control plane assigns it a UUID. Every stored record, and every response, has a `call_id`.

This makes resending a record safe, not delivery exactly-once. A record that never reaches the control plane is still missing, and a missing record doesn't mean the agent didn't try the call: the PI SDK blocks a `blockTools` call even when its record fails.

The SDKs send a `call_id` with every record:

- **The harness's tool call ID, namespaced.** The LangChain middleware uses the tool call's `id`, and the PI SDK the tool call ID that PI passes to `execute`, `tool_call` and `tool_result`. Each becomes `<sdk>:<UUID>:<tool call ID>`, where `<sdk>` is `langchain` or `pi` and the UUID is chosen once per middleware or extension (and again in each forked process, for the Python SDK), so two interception layers recording into one evaluation can't collide.
- **Otherwise a UUID.** The SDK uses a fresh UUID when the harness gives no ID, when the namespaced ID would be longer than 256 characters, or when the same SDK instance has already used that ID, as when a model reuses tool call IDs or a PI tool is both registered and hooked. The MCP SDK always uses a UUID: an MCP request ID only numbers the requests of one connection.
- **Bounded retries with the same `call_id`.** A record or a start that fails with a transport error (no response) or a 5xx is resent with the same body and `call_id`, up to 3 sends in all, waiting 100 ms and then 200 ms. A 4xx, including a 409, isn't resent: the SDK raises it.

## Started and unfinished calls

A call is recorded only once it has a result, unless it's started first. `POST /agent/function-calls/start` (and the orchestrator's `POST /runs/{run_id}/evaluations/{eval_id}/function-calls/start`) records a call's `call_id`, `function` and `args` before the tool runs. The record is unfinished: `complete` is `false`, it has no `result`, and its pre-environment is the evaluation's environment when the call started. Sending the call's record to `POST /agent/function-calls` with the same `call_id` later finishes it in place: it fills in `result`, `error` and `blocked`, sets `complete` to `true`, and takes the post-environment.

| The evaluation already has | Start (`.../function-calls/start`) | Record (`.../function-calls`) |
|---|---|---|
| no record with this `call_id` | 201: an unfinished record is appended | 201: a finished record is appended, as without a start |
| an unfinished record with this `call_id` and the same `function` and `args` | 200: nothing changes; the response is the stored record | 201: the record is finished in place |
| an unfinished record with this `call_id` and another `function` or `args` | 409: nothing changes | 409: nothing changes; the record stays unfinished |
| a finished record with this `call_id` and the same `function` and `args` | 200: nothing changes; the response is the finished record | 200 when the body is the same, 409 when it's different, as in [Record IDs and retries](#record-ids-and-retries) |
| a finished record with this `call_id` and another `function` or `args` | 409: nothing changes | 409: nothing changes |

A start without a `call_id` is a 422: nothing could finish it. A start resent after its call finished (say, a retry whose first response was lost) is a 200 replay that returns the finished record, so it can't reopen the call. A record sent without a start behaves as before: one finished record. Its pre-environment is the previous record's post-environment, or, when the previous record is unfinished, the evaluation's environment when it's recorded, so changes made since that call started aren't attributed to it.

Each call keeps one record, in the order the calls started, so counting records counts calls whether or not they were started. Finished calls look as they do without a start, and unfinished ones appear with `complete: false` in `GET /agent/function-calls`, in the orchestrator's evaluation response, and in what verifiers read.

**What an unfinished call means.** The agent made the call, and MiDojo saw it start, but no result was recorded: the tool may still be running, it may have hung or crashed, the agent run may have ended or been cancelled first, or the finish may have been lost on its way to the control plane. An unfinished call doesn't say whether the tool's effect happened. Its record has no result and no post-environment; the evaluation's environment, which verifiers read, is whatever the interception layer last wrote.

**How grading treats it.** An unfinished call counts as an attempted call wherever a blocked call does, and nowhere else. `tool_called` matches it by `function` and `args`, as it matches a blocked call, since MiDojo grades what the agent attempts; no other verifier reads the call records. The orchestrator's check of whether an injection reached the agent skips unfinished calls, since the agent saw no result from them. Whether an evaluation with an unfinished call should be graded at all, or graded as inconclusive, is an open question for the maintainers.

The SDKs start every call that runs a tool: the MCP SDK's tools, the LangChain middleware's hooked and reported tools, and the PI SDK's registered, hooked and reported tools. PI's `tool_call` starts a hooked or reported call, and `tool_result` finishes it with the start's `call_id` and arguments, found by PI's tool call ID. A tool that's also in `blockTools` never runs, so it isn't started: its call is recorded once, as blocked.

If the start fails after its resends, the tool doesn't run: the SDK raises, and PI blocks the call. If the finish fails, the SDK raises after the tool has run, and the call stays unfinished. A tool that's cancelled (or ends with any other Python `BaseException` that isn't an `Exception`) gave the agent no result, so the SDK doesn't finish its call, and it stays unfinished too.

A PI `tool_call` for a hooked or reported tool without a tool call ID is blocked, as if its start had failed. Its `tool_result` couldn't find the start, and PI's events carry nothing else that pairs a `tool_result` with its `tool_call` for certain (the same tool can run in parallel with the same arguments), so the call could otherwise run with only a one-step record that, if lost, would leave no trace. This only stops a call that PI gave no ID; the SDK doesn't assume PI always gives one.
