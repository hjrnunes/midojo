# Drive an episode from a synchronous script

`midojo.episode_handle.open` returns a handle: one episode, driven from a script that has no event loop. The handle is the same `EpisodeContext` the asynchronous API runs, on a loop thread of its own, so an agent script, a shell wrapper or a test needs no `async` and no `await`.

```python
from midojo import Episode
from midojo.episode_handle import open

episode = Episode(runtime={"type": "unmanaged"}, turns=["Look up order ORD-101.", "Sum that up in one line."])
with open(episode, upstream="http://127.0.0.1:8770/sse") as handle:
    for message in episode.turns:
        reply = handle.send_turn(message, send=lambda prompt: my_agent(prompt, handle.tool_endpoint))
    snapshot = handle.seal()
    verdict = handle.grade()
print(verdict.evaluation_id, verdict.snapshot_id, verdict.grade.security)
```

Entering the handle starts the runtime, opens the evaluation, starts the interceptor and sets the runtime up. Leaving it seals the evaluation if you didn't, tears the runtime down, stops the interceptor and closes the handle. Use it as a context manager or call `close()` yourself; a handle you never close is closed at process exit.

## The API

`open(episode, *, runtime=None, upstream=None, interception=None, control=None, run_id=None, reset=False, max_live=None, repeat_of=None)`:

| Argument | What it does |
| --- | --- |
| `runtime` | The agent's runtime. Without one, the episode's own runtime is built, so `open` first calls `episode.check_runtime_available()` and raises `MissingCapabilityError` for a provider that isn't installed. |
| `upstream` | The MCP server the interceptor mirrors. Without it, the episode's `interception.upstream`; with neither, `open` raises `EpisodeDocumentError` before it acquires anything. |
| `interception` | A whole `HostInterception`, which wins over the episode's settings. It's how you change the interceptor's port or bind host; the episode's tool policy still applies. |
| `control` | A control plane or client. Without one, the episode runs on a control plane in this process. |
| `run_id` | The run to open the evaluation in, with a `ControlPlaneClient`, and only with the runtime that run already owns. See [When `run_id` applies](#when-run_id-applies). |
| `reset` | Reset the target before the episode, on a runtime that declares it can. |
| `max_live` | Lower this process's cap for open handles. It never raises the cap. |
| `repeat_of` | The key of the run this episode repeats. A runtime that neither resets nor replaces its target refuses the repeat with `RepeatWithoutResetError`, since the two attempts would share one target. |

`open` checks its arguments in a fixed order, so the first refusal names what to fix. Steps 1 to 5 allocate nothing, so they run before any sandbox, port or live-episode slot. The last two steps are the cap and the bind, and each refuses before it takes the slot or binds the port:

1. The episode is an `Episode`; otherwise `TypeError`.
2. The upstream: `upstream=` or the episode's `interception.upstream`; with neither, `EpisodeDocumentError`.
3. The runtime: without `runtime=`, that the episode's own provider is installed, else `MissingCapabilityError`.
4. `run_id`: with `run_id=`, that the runtime you pass has a run open under it, else `ValueError`. See [When `run_id` applies](#when-run_id-applies).
5. The repeat: `repeat_of` on a runtime that can't isolate the target, else `RepeatWithoutResetError`.
6. The cap: an open handle too many, else `LiveEpisodeLimitError`.
7. The bind: the interceptor's port and host, else `HandleError`.

### When `run_id` applies

`run_id` opens the evaluation in a run that is already open on a control plane, so the served control plane can list it with the run's other evaluations. It requires both:

- A `ControlPlaneClient` (the `control=` argument, or the episode's), because a control plane in this process needs no run.
- The runtime that run owns: pass the runtime whose `workspace_name` is already set, and the handle neither starts nor ends that run, nor reaps it.

The handle refuses `run_id` without such a runtime, because the served control plane compiles the episode from the run's suite and needs the episode's `user_task_id`: it would fail after a sandbox exists. Leave `run_id` out instead, and the handle starts the episode's own run.

A handle reads `evaluation_id`, `binding`, `tool_endpoint`, `capabilities` and `turns`, and `sealed`, `closed` and `result`. It drives the episode with the methods in `HANDLE_METHODS`:

- `start_turn(message)` starts the next turn and returns its index. Calls the agent makes from then on carry that index.
- `record_reply(index, reply)` records the agent's reply. The last reply is the agent's output at the seal.
- `send_turn(message, *, send=None, timeout=None)` starts a turn, sends the message and records the reply. With `send`, your callable drives the agent. Without it, the runtime runs the agent itself, which is what OpenShell does: the handle runs the runtime's `exec_agent` command in the sandbox with `MIDOJO_TURN_INDEX` and `MIDOJO_TURN_COUNT` set, and returns the agent's output. On a runtime that doesn't run the agent, it raises `SendTurnUnsupported` before it starts the turn.
- `seal(agent_output=None)` closes the evaluation, collects the runtime's observations, tears the runtime down and seals it, and returns the snapshot. `agent_output` defaults to the last recorded reply. Sealing before closing keeps the interceptor up while you grade, so a late call is a late fact rather than a lost one.
- `tool_block(label="midojo")` returns the interceptor as an MCP tool block for a target that takes one: `server_url`, `authorization`, `require_approval` and the `allowed_tools` the policy exposes.

`close()` seals an unsealed evaluation as interrupted, then releases the context, the loop thread and the live-episode slot. Closing twice is a no-op. Use `grade()` after sealing or after closing: it's a pure function of the episode and the sealed snapshot, so it still works on a handle whose close sealed the evaluation as interrupted. Sealing and grading wait for the loop thread, so a script can grade and print without a loop of its own.

## The tool policy

`open` enforces the episode's `tools` policy. The interceptor mirrors only the tools in `tools.allowed`, and the handle reports those in `tool_block()["allowed_tools"]`. A tool whose `write_disposition` is `block` is recorded as blocked and never reaches the upstream; the rest are forwarded. An episode with no `tools` mirrors every upstream tool and blocks nothing, and an empty `allowed` exposes none.

This replaces orch's `BlockingInterception`: the policy lives in the episode document, not in the caller, so the same episode blocks the same calls wherever it runs, and the blocked call is evidence, not an error.

The handle's interceptor is the only place that enforces a policy. `midojo.open_episode` starts no interceptor of its own, so it never applies `episode.tools`, and it refuses an episode whose policy blocks a tool unless the `interception` you pass blocks the same tools: a policy that silently did nothing would look like evidence of safety. Drive an episode that has a policy with this handle, or with an interception layer that enforces the policy itself.

## Ownership, caps and failures

Each handle runs one loop thread, named `midojo-episode-handle`. The context lives in the thread that opened the handle, so a handle used from another process (a `fork`ed child, say) raises `HandleOwnerError` and changes nothing: the message names the owning process and the current one. Calls that a handle made before returning are the only writes, which is why the check comes before any work.

A process keeps at most `DEFAULT_MAX_LIVE_EPISODES` (8) handles open. `MIDOJO_MAX_LIVE_EPISODES` overrides the cap for the process, and `max_live` lowers it for one call; a value that isn't a positive integer raises `ValueError` rather than falling back to the default. The limit counts live episodes, so a handle that fails to open never holds a slot. Opening past the cap raises `LiveEpisodeLimitError` before any port is bound. The count belongs to the process that opened the handles: a `fork`ed child starts from zero, because it owns none of its parent's episodes and none of its parent's listeners serve there.

An exception in the block, a `KeyboardInterrupt` or a `SystemExit` still closes the handle, and the evaluation is sealed as interrupted with its incomplete evidence. A driver killed outright can't run its cleanup, so a handle whose run it can reap starts a reaper process as soon as that run's workspace exists: right after it starts the run itself, and, when you pass no `runtime` and the episode's context builds and starts one, right after the context's `start_run` returns. Both points are before the runtime creates its sandbox, so a driver killed during setup is covered too. The reaper waits for the driver to exit, then deletes the sandboxes in the run's workspace and the workspace itself.

A close, or an open that fails after the run started, stops the reaper only once the run's `end_run` has succeeded, because that call is what deletes the workspace. That covers both cases: the handle ends a run it started, and the episode's context ends a run it started. When `end_run` fails, the handle logs the failure as it does for any teardown error and leaves the reaper running, so a workspace the failed teardown left behind is still removed when the driver exits. A handle's driver killed with `SIGKILL` therefore leaves no sandbox and no listening interceptor behind, whatever it was doing. A run the caller started itself (the runtime already names its workspace) belongs to the caller: the handle neither ends nor reaps it.

The reaper belongs to the handle, which starts it. An asynchronous driver, one that calls `midojo.open_episode` itself, starts no reaper, so a driver killed with `SIGKILL` in the middle of an episode can leave its OpenShell workspace and sandbox behind. Run the episode through the handle when a driver has to be leak-safe.

## Compatibility promise

`midojo.episode_handle` is the synchronous driver API, and this is what stays stable within a 0.x minor release:

- The names in `HANDLE_METHODS`, and the properties and methods above.
- The exception classes in the module, and their `reason` strings: `HandleError` (`handle-error`), `HandleOwnerError` (`handle-owner`), `HandleClosedError` (`handle-closed`), `SendTurnUnsupported` (`send-turn-unsupported`), `RepeatWithoutResetError` (`repeat-without-reset`) and `LiveEpisodeLimitError` (`live-episode-limit`).
- `HANDLE_API_VERSION`, which is 1, and `DEFAULT_MAX_LIVE_EPISODES`.
- New keyword arguments to `open` and `send_turn` are appended with defaults, so existing calls keep working.

A change that breaks any of these raises the major version. The asynchronous API is unchanged: `midojo.open_episode` and `EpisodeContext` behave as before, and a handle is exactly a context driven on a thread.
