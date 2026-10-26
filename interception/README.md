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
- **The SDK attempts to record every intercepted call,** with `POST /agent/function-calls`: the agent's arguments, the result the layer returned to the agent, the error if there was one, and whether the call was blocked. Each record carries a `call_id`, so the SDK can resend a record whose response it lost without recording the call twice; a record that never arrives is still missing (see [Record IDs and retries](#record-ids-and-retries)). Tools that aren't intercepted run as usual and aren't recorded.
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
| Delivery facts (`injection_outcomes`) | no: its finishes are plain, so a payload's exposure is unknown | no: plain finishes | for forwarded (mirrored) calls; a tool the server defines finishes plainly |

## The evaluation session

The control plane issues a session token for each evaluation. An interception layer sends it on every call as `Authorization: Bearer <token>`, and the token selects the evaluation's environment and call record. How the layer gets the token depends on how MiDojo reaches the agent:

- The `openshell` runtime sets `MIDOJO_SESSION_TOKEN` and `MIDOJO_URL` (the control plane URL) in the sandbox's environment. The SDKs read them.
- For an `unmanaged` agent, `midojo-run` sends the token in the `X-Midojo-Session` request header. `midojo_sdk.mcp` reads the header from the incoming MCP request, and `midojo_sdk.session` has helpers for agents that pass it on to their own tool servers.

These layers run where the agent can reach the token, so their records have `provenance` `agent`: lower-trust evidence. An evaluation intercepted this way that is sealed with no call recorded at all has its `function_calls` evidence `incomplete`, since a layer that fails open records nothing. With `midojo-run --interception host`, the host runs a `MidojoMCP` interceptor for each evaluation and keeps the session itself. The agent gets only the interceptor's URL and a bearer token for that interceptor, and the records have `provenance` `interceptor` (see [Keep the evaluation out of the agent's reach](../README.md#keep-the-evaluation-out-of-the-agents-reach)).

The session ends when the evaluation is sealed (see [Closure and late facts](#closure-and-late-facts)), when the orchestrator revokes it after a failure, or when it expires (`midojo-serve --session-ttl`, default 3600 s). Every call then gets a 401, with one exception: after the seal, a start or record that exactly replays a recorded call (same `call_id` and body) is still a 200 with the stored record, so a retry whose first response was lost stays safe.

## Control plane API

All paths are relative to the control plane URL and take the session's bearer token.

| Call | Body | Returns |
|---|---|---|
| `GET /agent/environment` | | the evaluation's environment, as JSON, with its revision in the `X-Midojo-Environment-Revision` header |
| `PATCH /agent/environment` | `{"set": {field: value, ...}, "expected_revision"}`; `expected_revision` is optional | the changed environment, with its new revision in `X-Midojo-Environment-Revision`. The listed top-level fields change in one atomic step and the others are kept, so a concurrent change to another field is never lost. With `expected_revision`, the change applies only at that revision: 409 with `kind` `revision_conflict` at any other. A field the suite doesn't define, or a value that doesn't fit its schema, is a 422 with `kind` `invalid_environment` |
| `PUT /agent/environment` | the whole environment | the stored environment. The control plane validates it against the suite's environment schema. A read followed by a replace loses any change made in between: prefer `PATCH` |
| `POST /agent/function-calls/start` | `{"call_id", "function", "args", "disposition"}`; `call_id` is required, `disposition` optional | the started call, unfinished: 201 when it's new, 200 when `call_id` is already recorded with the same `function` and `args` (and `disposition`, see below), 409 when it's recorded with others, or when the evaluation is closing; 422 with `kind` `invalid_disposition` for a `denied` or `blocked` start |
| `POST /agent/function-calls` | `{"call_id", "function", "args", "result", "error", "blocked", "disposition"}`; `disposition` optional | the recorded call: 201 when it's new or finishes a started call, 200 when `call_id` is already finished with the same body, 409 when it's finished with a different body or was started with another `function`, `args` or `disposition`, or when the evaluation is closing and no longer admits it; 422 with `kind` `invalid_disposition` when `disposition` doesn't go with `blocked` |
| `GET /agent/function-calls` | | the calls recorded so far, finished or not, in the order they started |

An error response has `detail`, `kind` (the reason, such as `call_conflict`, `closing`, `session_closed` or `invalid_session`) and `late_fact` (whether the control plane kept the refused write as a late fact). Only a response with a `kind` is the control plane's own refusal; a response without one, such as a proxy's 503 or a 422 for a malformed body, isn't.

The environment's revision counts its changes: every `PATCH`, `PUT` and run-scoped replace gives a new one. A change to the environment is atomic on its own, and on its own only: nothing makes a call's record, an environment change and what a forwarded call did to a real system one transaction.

In a function-call record, `result` is what the layer returned to the agent. The layer records it before it returns it, so a record doesn't show the agent received it. `error` (optional) is the error the tool raised. `blocked` (default `false`) means the tool never ran and `result` is what the agent was told instead. `complete` (in responses) is `false` while a started call is unfinished; its `result` is then `null`.

`disposition` (optional, `null` when no one said) is what the interception layer did with the call: `forwarded` to the real tool, `simulated` by a tool the layer defines, or refused before the tool ran, `denied` by a check on its arguments or `blocked` because the tool is blocked. A `denied` or `blocked` call has `blocked: true` and a `forwarded` or `simulated` one `blocked: false`; any other pairing is a 422 with `kind` `invalid_disposition`, at any stage, and leaves nothing behind. A refused call never runs, so it's never started. What the layer does with a call is decided when it starts: a finish that leaves `disposition` out keeps its start's, and one that names another is a 409 `call_conflict`, like another `function` or `args` (so is a `blocked: true` finish of a call started `forwarded` or `simulated`). A `disposition` left out, on a resend or in a stored record, matches any, so a resend from a client that doesn't send it is still a replay. The SDK's HTTP client sends `blocked` alongside, so a control plane that predates the field still records a refused call as blocked.

## Record IDs and retries

`call_id` (optional, 1 to 256 characters) identifies a record within its evaluation. The control plane keeps at most one record per `call_id` per evaluation, across `POST /agent/function-calls` and the orchestrator's `POST /runs/{run_id}/evaluations/{eval_id}/function-calls`:

| The evaluation already has | Status | Effect |
|---|---|---|
| no record with this `call_id` | 201 | the call is recorded |
| a record with this `call_id` and the same `function`, `args`, `result`, `error` and `blocked` | 200 | nothing is appended; the response is the stored record |
| a record with this `call_id` and a different body | 409 | nothing is appended |

A record sent without a `call_id` is always appended, and the control plane assigns it a UUID. Every stored record, and every response, has a `call_id`.

This makes resending a record safe, not delivery exactly-once. A record that never reaches the control plane is still missing, and a missing record doesn't mean the agent didn't try the call: the PI SDK blocks a `blockTools` call even when its record fails. The Python SDK reports each start or finish it couldn't deliver (after its resends, or refused other than as a late fact) to `POST /agent/capture-failures` with its `call_id`, `function`, `phase` (`start` or `finish`) and `reason`: 201 when kept, 200 for the same `call_id`, `function` and `phase` again (whatever the `reason`, even after the seal) once a report of them was kept before the seal, 401 with `late_fact` true for a new report after the seal. A report first received after the seal is only a late fact, so sending it again is refused the same way and adds no other late fact. The report never adds a record. One received before the seal marks `function_calls` `incomplete` in the sealed snapshot, so a check that depends on a call's absence is unknown instead of passing on a lost record; one received after the seal is a late fact that marks it `incomplete` only in the evaluation's `evidence_status`, and the grade doesn't change.

The SDKs send a `call_id` with every record:

- **The harness's tool call ID, namespaced.** The LangChain middleware uses the tool call's `id`, and the PI SDK the tool call ID that PI passes to `execute`, `tool_call` and `tool_result`. Each becomes `<sdk>:<UUID>:<tool call ID>`, where `<sdk>` is `langchain` or `pi` and the UUID is chosen once per middleware or extension (and again in each forked process, for the Python SDK), so two interception layers recording into one evaluation can't collide.
- **Otherwise a UUID.** The SDK uses a fresh UUID when the harness gives no ID, when the namespaced ID would be longer than 256 characters, or when the same SDK instance has already used that ID, as when a model reuses tool call IDs or a PI tool is both registered and hooked. The MCP SDK always uses a UUID: an MCP request ID only numbers the requests of one connection.
- **Bounded retries with the same `call_id`.** A record or a start that fails with a transport error (no response) or a 5xx is resent with the same body and `call_id`, up to 3 sends in all, waiting 100 ms and then 200 ms. A 4xx, including a 409, isn't resent: the SDK raises it. In both SDKs, the resends are a recorder that wraps another (`RetryingRecorder`: `midojo_sdk.recorder`'s, or the PI SDK's), and a refusal with a `kind` is raised as a `ControlPlaneRejectedError` carrying it (in the PI SDK, a subclass of its `ControlPlaneError`).
- **A refusal is never resent.** A response with a `kind` gets the same answer every time. The LangChain middleware gives the agent the refusal as an error result instead of the tool's, and the run goes on; the MCP and PI SDKs raise it. Either way, a call whose start is refused never runs, and PI still blocks a `blockTools` call whose record is refused.

## Started and unfinished calls

A call is recorded only once it has a result, unless it's started first. `POST /agent/function-calls/start` (and the orchestrator's `POST /runs/{run_id}/evaluations/{eval_id}/function-calls/start`) records a call's `call_id`, `function` and `args` before the tool runs. The record is unfinished: `complete` is `false` and it has no `result`. Sending the call's record to `POST /agent/function-calls` with the same `call_id` later finishes it in place: it fills in `result`, `error` and `blocked`, and sets `complete` to `true`.

| The evaluation already has | Start (`.../function-calls/start`) | Record (`.../function-calls`) |
|---|---|---|
| no record with this `call_id` | 201: an unfinished record is appended | 201: a finished record is appended, as without a start |
| an unfinished record with this `call_id` and the same `function` and `args` | 200: nothing changes; the response is the stored record | 201: the record is finished in place |
| an unfinished record with this `call_id` and another `function` or `args` | 409: nothing changes | 409: nothing changes; the record stays unfinished |
| a finished record with this `call_id` and the same `function` and `args` | 200: nothing changes; the response is the finished record | 200 when the body is the same, 409 when it's different, as in [Record IDs and retries](#record-ids-and-retries) |
| a finished record with this `call_id` and another `function` or `args` | 409: nothing changes | 409: nothing changes |

A start without a `call_id` is a 422: nothing could finish it. A start resent after its call finished (say, a retry whose first response was lost) is a 200 replay that returns the finished record, so it can't reopen the call. A record sent without a start behaves as before: one finished record.

**Environment history.** By default a record carries no copy of the environment. A control plane created with `AppConfig(environment_history=True)` (`midojo-serve --environment-history`) keeps, on each record, the environment observed when the call started (its pre-environment) and when it finished (its post-environment). A record sent without a start takes as its pre-environment the previous record's post-environment, or, when the previous record is unfinished, the evaluation's environment when it's recorded, so changes made since that call started aren't attributed to it. These are observations over the call's interval, not its effect: another call, or the host, may have changed the environment in between. The run-scoped and `/agent` responses never include them.

Each call keeps one record, in the order the calls started, so counting records counts calls whether or not they were started. Finished calls look as they do without a start, and unfinished ones appear with `complete: false` in `GET /agent/function-calls`, in the orchestrator's evaluation response, and in what verifiers read.

**What an unfinished call means.** The agent made the call, and MiDojo saw it start, but no result was recorded: the tool may still be running, it may have hung or crashed, the agent run may have ended or been cancelled first, or the finish may have been lost on its way to the control plane. An unfinished call doesn't say whether the tool's effect happened. Its record has no result (and, with environment history, no post-environment); the evaluation's environment, which verifiers read, is whatever the interception layer last wrote.

**How grading treats it.** An unfinished call counts as an attempted call wherever a blocked call does, and nowhere else. `tool_called` matches it by `function` and `args`, as it matches a blocked call, since MiDojo grades what the agent attempts; no other verifier reads the call records. In the delivery and exposure report, an unfinished call to an instruction's tool is a gap: no finish recorded what the interceptor emitted in it, so that payload's exposure is unknown. Whether an evaluation with an unfinished call should be graded at all, or graded as inconclusive, is an open question for the maintainers.

The SDKs start every call that runs a tool: the MCP SDK's tools, the LangChain middleware's hooked and reported tools, and the PI SDK's registered, hooked and reported tools. PI's `tool_call` starts a hooked or reported call, and `tool_result` finishes it with the start's `call_id` and arguments, found by PI's tool call ID. A tool that's also in `blockTools` never runs, so it isn't started: its call is recorded once, as blocked.

If the start fails after its resends, the tool doesn't run: the SDK raises, and PI blocks the call. If the finish fails, the SDK raises after the tool has run, and the call stays unfinished. A tool that's cancelled (or ends with any other Python `BaseException` that isn't an `Exception`) gave the agent no result, so the SDK doesn't finish its call, and it stays unfinished too.

## Closure and late facts

When the agent has replied, the orchestrator closes the evaluation in stages:

1. **Closing** (`POST /runs/{run_id}/evaluations/{eval_id}/close`). No new call is admitted: a start, or a record whose `call_id` isn't recorded yet, is a 409 with `kind` `closing`. A call that's already started may still finish, and the environment may still change, until every started call has finished or the closure wait runs out. The wait is a control plane setting, `midojo-serve --closure-wait` (default 5 seconds). The deadline is checked against the time each write arrives, so it holds even if nothing ends the wait (for example, the orchestrator's close request was cancelled): a write that arrives after it is handled as in the collecting stage.
2. **Collecting.** The orchestrator records the runtime's final observations. A finish or an environment write that arrives now is a 409 with `kind` `closing`, and a call that hasn't finished stays unfinished.
3. **Sealed** (`POST .../complete`, which closes first if needed). The control plane takes a snapshot of the calls, turns, environment, observations and evidence status, and grading reads only that snapshot. The grade is recorded once; grading again returns it. Every write after this point is refused, and the session is closed: a call through `/agent` gets a 401 with `kind` `session_closed`, and a run-scoped write gets a 409 with `kind` `sealed`. That includes a start or record under a recorded `call_id` with another body: before the seal it's a 409 `call_conflict` and isn't recorded, after the seal it's refused like any other late write. Only an exact replay is answered from the record.

A write that closure refuses isn't lost: the control plane records it as a **late fact**, in the evaluation's `late_facts` (what was sent, when, and by whom), and the response says so with `late_fact: true`. A late fact never changes the snapshot or the grade. Instead it marks the evidence it carries `incomplete` in the evaluation's `evidence_status`: `function_calls` for a call, `environment`, `turns`, `agent_output`, or the observation source's `<source>.<part>` keys. Resending the same late write records it once.

Retries stay safe at every stage: a start or record that replays a stored `call_id` with the same body (for a start, the same `function` and `args`) is a 200 with the stored record, even after the seal. A session the orchestrator revoked (after a failure) records nothing: its calls get a 401 with `kind` `invalid_session`.


A PI `tool_call` for a hooked or reported tool without a tool call ID isn't started: its `tool_result` couldn't find the start, and PI's events carry nothing else that pairs a `tool_result` with its `tool_call` for certain (the same tool can run in parallel with the same arguments). The call runs, and its `tool_result` records it once with a fresh `call_id`, as before starts existed, so such a call that never finishes leaves no record. The SDK doesn't assume PI always gives an ID.
