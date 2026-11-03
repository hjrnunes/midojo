# Drive an evaluation from a synchronous script

`midojo.evaluation_handle.open` returns a handle: one injection spec, driven from a script that has no event loop. The handle is the same `EvaluationContext` the asynchronous API runs, on a loop thread of its own, so an agent script, a shell wrapper or a test needs no `async` and no `await`.

```python
from midojo import InjectionSpec
from midojo.evaluation_handle import open

spec = InjectionSpec(runtime={"type": "unmanaged"}, turns=["Look up order ORD-101.", "Sum that up in one line."])
with open(spec, upstream="http://127.0.0.1:8770/sse") as handle:
    for message in spec.turns:
        reply = handle.send_turn(message, send=lambda prompt: my_agent(prompt, handle.tool_endpoint))
    snapshot = handle.seal()
    verdict = handle.grade()
print(verdict.evaluation_id, verdict.snapshot_id, verdict.grade.security)
```

Entering the handle starts the runtime, opens the evaluation, starts the interceptor and sets the runtime up. Leaving it seals the evaluation if you didn't, tears the runtime down, stops the interceptor and closes the handle. Use it as a context manager or call `close()` yourself; a handle you never close is closed at process exit.

## The API

`open(spec, *, runtime=None, upstream=None, interception=None, control=None, run_id=None, reset=False, max_live=None, repeat_of=None)`:

| Argument | What it does |
| --- | --- |
| `runtime` | The agent's runtime. Without one, the injection spec's own runtime is built, so `open` first calls `spec.check_runtime_available()` and raises `MissingCapabilityError` for a provider that isn't installed. |
| `upstream` | The MCP server the interceptor mirrors. Without it, the spec's `interception.upstream`; with neither, `open` raises `InjectionSpecDocumentError` before it acquires anything. |
| `interception` | A whole `HostInterception` or `RemoteInterception`, which wins over the spec's settings. A `HostInterception` is how you change the interceptor's port or bind host; the spec's tool policy still applies. A `RemoteInterception` runs the interceptor in a service instead. See [Run the interceptor in a service](#run-the-interceptor-in-a-service). |
| `control` | A control plane or client. Without one, the spec runs on a control plane in this process. |
| `run_id` | The run to open the evaluation in, with a `ControlPlaneClient`, and only with the runtime that run already owns. See [When `run_id` applies](#when_run_id-applies). |
| `reset` | Reset the target before the evaluation, on a runtime that declares it can. |
| `max_live` | Lower this process's cap for open handles. It never raises the cap. |
| `repeat_of` | The key of the run this evaluation repeats. A runtime that neither resets nor replaces its target refuses the repeat with `RepeatWithoutResetError`, since the two attempts would share one target. |

`open` checks its arguments in a fixed order, so the first refusal names what to fix. Steps 1 to 5 allocate nothing, so they run before any sandbox, port or live slot. The last two steps are the cap and the bind, and each refuses before it takes the slot or binds the port:

1. The spec is an `InjectionSpec`; otherwise `TypeError`.
2. `interception=`: a `HostInterception`, a `RemoteInterception` or nothing; anything else is a `TypeError`.
3. The upstream: `upstream=` or the spec's `interception.upstream`; with neither, `InjectionSpecDocumentError`. A remote interception needs its URL, an `http(s)` string the service dials, else `ValueError`.
4. A remote interception's control: a control plane in this process, else `ValueError`.
5. The runtime: without `runtime=`, that the spec's own provider is installed, else `MissingCapabilityError`.
6. `run_id`: with `run_id=`, that the runtime you pass has a run open under it, else `ValueError`. See [When `run_id` applies](#when_run_id-applies).
7. The repeat: `repeat_of` on a runtime that can't isolate the target, else `RepeatWithoutResetError`.
8. The cap: an open handle too many, else `LiveEpisodeLimitError`.
9. The dial or the bind: a service's refusal of the binding (`BindingRefusedError`), or the interceptor's port and host (`HandleError`).

### When `run_id` applies

`run_id` opens the evaluation in a run that is already open on a control plane, so the served control plane can list it with the run's other evaluations. It requires both:

- A `ControlPlaneClient` (the `control=` argument, or the spec's), because a control plane in this process needs no run.
- The runtime that run owns: pass the runtime whose `workspace_name` is already set, and the handle neither starts nor ends that run, nor reaps it.

The handle refuses `run_id` without such a runtime, because the served control plane compiles the injection spec from the run's suite and needs the spec's `user_task_id`: it would fail after a sandbox exists. Leave `run_id` out instead, and the handle starts the spec's own run.

A handle reads `evaluation_id`, `binding`, `tool_endpoint`, `capabilities` and `turns`, and `sealed`, `closed` and `result`. It drives the evaluation with the methods in `HANDLE_METHODS`:

- `start_turn(message)` starts the next turn and returns its index. Calls the agent makes from then on carry that index.
- `record_reply(index, reply)` records the agent's reply. The last reply is the agent's output at the seal.
- `send_turn(message, *, send=None, timeout=None)` starts a turn, sends the message and records the reply. With `send`, your callable drives the agent. Without it, the runtime runs the agent itself, which is what OpenShell does: the handle runs the runtime's `exec_agent` command in the sandbox with `MIDOJO_TURN_INDEX` and `MIDOJO_TURN_COUNT` set, and returns the agent's output. On a runtime that doesn't run the agent, it raises `SendTurnUnsupported` before it starts the turn.
- `seal(agent_output=None)` closes the evaluation, collects the runtime's observations, tears the runtime down and seals it, and returns the snapshot. `agent_output` defaults to the last recorded reply. Sealing before closing keeps the interceptor up while you grade, so a late call is a late fact rather than a lost one.
- `tool_block(label="midojo")` returns the interceptor as an MCP tool block for a target that takes one: `server_url`, `authorization`, `require_approval` and the `allowed_tools` the policy exposes.

`close()` seals an unsealed evaluation as interrupted, then releases the context, the loop thread and the live slot. Closing twice is a no-op. Use `grade()` after sealing or after closing: it's a pure function of the injection spec and the sealed snapshot, so it still works on a handle whose close sealed the evaluation as interrupted. Sealing and grading wait for the loop thread, so a script can grade and print without a loop of its own.

## The tool policy

`open` enforces the spec's `tools` policy. The interceptor mirrors only the tools in `tools.allowed`, and the handle reports those in `tool_block()["allowed_tools"]`. A tool whose `write_disposition` is `block` is recorded as blocked and never reaches the upstream; the rest are forwarded. An injection spec with no `tools` mirrors every upstream tool and blocks nothing, and an empty `allowed` exposes none.

This replaces orch's `BlockingInterception`: the policy lives in the injection spec document, not in the caller, so the same spec blocks the same calls wherever it runs, and the blocked call is evidence, not an error.

The handle's interceptor is the only place that enforces a policy. `midojo.open_evaluation` starts no interceptor of its own, so it never applies `spec.tools`, and it refuses an injection spec whose policy blocks a tool unless the `interception` you pass blocks the same tools: a policy that silently did nothing would look like evidence of safety. Drive a spec that has a policy with this handle, or with an interception layer that enforces the policy itself.

## Run the interceptor in a service

`interception=RemoteInterception(url, credential)` runs each evaluation's interceptor in the interceptor service at `url` instead of in the driver's process. Without the option, a service named in both `MIDOJO_INTERCEPTOR_URL` and `MIDOJO_INTERCEPTOR_CREDENTIAL` is used the same way; set one without the other and `open` refuses it naming both, and set neither and the interceptor runs in this process, as before.

The service holds the interceptor and the agent's tool endpoint; the driver holds the evaluation. Each evaluation opens one WebSocket stream: the handshake offers the subprotocol `midojo-binding.v1`, and the frames inside it belong to the API `midojo-binding-v1`; the driver dials out to it, so a driver with a live remote binding holds no listening socket at all. The interceptor's records, its environment reads and changes, and the injection outcomes all cross that stream, so they land in the driver's own control plane: the binding records through a `LinkRecorder` (never retry-wrapped, since a lost answer must not cause a second forward) and its environment through a `LinkWorld`. `upstream` must then be the MCP server's URL, because the service dials it: `open` refuses a server object in the driver's process. `tool_block()` lists the tools the service answered the binding with, and `result.injection_outcomes` are the ones its release reported.

Start the service with `midojo-intercept-serve --credential ...` (or `InterceptorService(credential).start()`); it prints the driver's URL. A lost link is final: the evaluation keeps what the driver recorded, seals with `function_calls` incomplete, and the handle raises `LinkLostError` (`reason` `link-lost`) for every call after it. The dispatcher sends `ping` while it waits for frames (`HEARTBEAT_INTERVAL_SECONDS`, 5 seconds), so a stalled dispatcher lets the service's 15-second expiry end the binding, and a request to a binding's endpoint waits until the service answered `bound`. An acknowledged release after the seal is not a loss; one the service never acknowledges is: `close()`, and leaving the `with` block, raises `LinkLostError`, keeps the sealed snapshot unchanged and reports no injection outcomes, and `close_error` names the error afterwards.

## Ownership, caps and failures

Each handle runs one loop thread, named `midojo-episode-handle`. The context lives in the thread that opened the handle, so a handle used from another process (a `fork`ed child, say) raises `HandleOwnerError` and changes nothing: the message names the owning process and the current one. Calls that a handle made before returning are the only writes, which is why the check comes before any work.

A process keeps at most `DEFAULT_MAX_LIVE_EPISODES` (8) handles open. `MIDOJO_MAX_LIVE_EPISODES` overrides the cap for the process, and `max_live` lowers it for one call; a value that isn't a positive integer raises `ValueError` rather than falling back to the default. The limit counts live evaluations, so a handle that fails to open never holds a slot. Opening past the cap raises `LiveEpisodeLimitError` before any port is bound. The count belongs to the process that opened the handles: a `fork`ed child starts from zero, because it owns none of its parent's evaluations and none of its parent's listeners serve there.

An exception in the block, a `KeyboardInterrupt` or a `SystemExit` still closes the handle, and the evaluation is sealed as interrupted with its incomplete evidence. A driver killed outright can't run its cleanup, so a handle whose run it can reap starts a reaper process as soon as that run's workspace exists: right after it starts the run itself, and, when you pass no `runtime` and the spec's context builds and starts one, right after the context's `start_run` returns. Both points are before the runtime creates its sandbox, so a driver killed during setup is covered too. The reaper waits for the driver to exit, then deletes the sandboxes in the run's workspace and the workspace itself.

A close, or an open that fails after the run started, stops the reaper only once the run's `end_run` has succeeded, because that call is what deletes the workspace. That covers both cases: the handle ends a run it started, and the spec's context ends a run it started. When `end_run` fails, the handle logs the failure as it does for any teardown error and leaves the reaper running, so a workspace the failed teardown left behind is still removed when the driver exits. A handle's driver killed with `SIGKILL` therefore leaves no sandbox and no listening interceptor behind, whatever it was doing. A run the caller started itself (the runtime already names its workspace) belongs to the caller: the handle neither ends nor reaps it.

The reaper belongs to the handle, which starts it. An asynchronous driver, one that calls `midojo.open_evaluation` itself, starts no reaper, so a driver killed with `SIGKILL` in the middle of an evaluation can leave its OpenShell workspace and sandbox behind. Run the injection spec through the handle when a driver has to be leak-safe.

## Compatibility promise

`midojo.evaluation_handle` is the synchronous driver API, and this is what stays stable within a 0.x minor release:

- The names in `HANDLE_METHODS`, and the properties and methods above.
- The exception classes in the module, and their `reason` strings: `HandleError` (`handle-error`), `HandleOwnerError` (`handle-owner`), `HandleClosedError` (`handle-closed`), `SendTurnUnsupported` (`send-turn-unsupported`), `RepeatWithoutResetError` (`repeat-without-reset`), `LiveEpisodeLimitError` (`live-episode-limit`) and `LinkLostError` (`link-lost`).
- `HANDLE_API_VERSION`, which is 1, and `DEFAULT_MAX_LIVE_EPISODES`.
- New keyword arguments to `open` and `send_turn` are appended with defaults, so existing calls keep working.

A change that breaks any of these raises the major version. The asynchronous API is unchanged: `midojo.open_evaluation` and `EvaluationContext` behave as before, and a handle is exactly a context driven on a thread.
