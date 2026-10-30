# Changelog

What this fork changes in MiDojo's public API. Read [breaking changes](#breaking-changes) before you upgrade. `midojo` and `midojo-sdk` are released together at the same `0.x` version.

## Unreleased

### Added

- **Episode document v1.** `Episode.to_document`, `Episode.from_document`, `Episode.load_document`, `Episode.dump_document` and `midojo.episode_document` read and write `schema_version: midojo-episode-v1`, a closed document: every member is written, so two dumps of one episode are the same bytes. A v0 consumer wrapper (`{"episode": ...}`, no `schema_version`) still reads, with a `DeprecationWarning`; the writer never emits it. The JSON Schema ships in `midojo_sdk.schemas` and `midojo_sdk.episode_schema.load_schema()` loads it.
- **Verdict document version.** A verdict names its format, `midojo-verdict-v1`, so a reader can tell an older record from a newer one. A verdict without the key reads as v0.
- **Plan codec version.** `midojo_sdk.injection.PLAN_CODEC_VERSION` (`midojo-plan-v1`) names the codec `InjectionPlan.to_data` writes, which the v1 episode document's `plan` member holds. The plan carries no version of its own: the document that encloses it names the codec through its `schema_version`, so a change to the bytes the codec writes needs a new document version.
- **Synchronous handle.** `midojo.episode_handle.open` returns an `EpisodeHandle` that drives one episode from a script with no event loop: `send_turn`, `record_reply`, `seal`, `grade` and `close`. It runs the same `EpisodeContext` on a loop thread of its own, and it states what stays stable (`HANDLE_METHODS`, `HANDLE_API_VERSION`, the exception classes and their `reason` strings). See [the driver guide](docs/guide-driving-midojo.md).
- **Live-episode cap.** A process keeps at most `DEFAULT_MAX_LIVE_EPISODES` (8) handles open; `MIDOJO_MAX_LIVE_EPISODES` overrides the cap and `max_live` lowers it for one call. Opening past the cap raises `LiveEpisodeLimitError` before any port is bound.
- **Runtime capability query.** `midojo.runtimes.runtime_capabilities` answers what a runtime can do from its definition alone: no runtime built, no sandbox created, no provider imported.
- **Placement engine protocol.** `midojo_sdk.placement.PlacementEngine` decides what a plan places in a tool call's result, apart from the wire: the interceptor forwards and records a call, and the engine answers each placement. `PlanPlacementEngine` is the plan's own engine.
- **Judge replay.** A grade keeps the judgements it made, and `grade(episode, snapshot, judgements=first.grade.judgements)` replays them with no judge service. A request without a record is unknown, never False. See [Replay](docs/judge-service.md#replay).
- **`midojo-sdk` as a base dependency.** `midojo` now depends on `midojo-sdk[mcp]` instead of reaching into the SDK's checkout: the episode document, the snapshot encoding, the plan codec and the host interceptor's MCP client all live in the published SDK, and `[tool.uv.sources]` keeps the workspace link for development. The `suites`, `pyrit` and `dev` extras are unchanged; `jsonschema>=4.23` joins `dev`, since the tests import it directly.

### Changed

- **The snapshot encoding moved to `midojo_sdk.snapshot`.** The digest, its canonical encoding and `SNAPSHOT_DIGEST_VERSION` live in the SDK, so a client verifies a snapshot without importing `midojo`. `midojo.control_plane.state.snapshot_digest` remains as a wrapper over it.
- **The served OpenAPI text states the unfinished-call rule.** `_INTERCEPTION_DESCRIPTION` now says that an unfinished call the session can no longer report — revoked or expired — marks `function_calls` evidence incomplete: the sealed record holds a call whose result is unknown.
- **A refused tool call logs one line, not a traceback.** A blocked or denied call still reaches the agent as the same `ToolError`; the driver now gets an INFO line from the SDK and no FastMCP traceback on stderr.
- **`episode_handle.open` refuses `run_id` without an open run.** Passing `run_id` with a runtime whose run is not open raises `ValueError` naming `run_id`, before any live-episode slot, port or sandbox. Start the run and pass the runtime it owns, or leave `run_id` out.
- **`open_episode` refuses an unenforced tool policy.** The asynchronous API has no interceptor of its own, so it never applied `episode.tools`. It now refuses an episode whose policy blocks a tool, unless the `interception` you pass blocks the same tools, rather than opening with the block silently unenforced. The handle's interceptor remains the only place a policy is enforced.

## Breaking changes

- **Episode documents are v1.** A v0 consumer wrapper still reads, with a `DeprecationWarning`, but new dumps are v1: `schema_version: midojo-episode-v1` at the top level, no `episode` wrapper. A reader that requires the wrapper must read v1, or keep reading the v0 files it already holds. The deprecation is scheduled for removal at the next major release.
- **A verdict names its version.** The document carries `schema_version: midojo-verdict-v1`; code that compares a verdict document byte for byte, or rejects unknown keys, must accept the key. A verdict without it still reads as v0.
- **The synchronous handle owns the run it starts.** `episode_handle.open` starts the episode's run, starts the sandbox reaper for it, and ends the run on close; a driver script that used to start the run itself must pass the runtime that run owns. An asynchronous `open_episode` driver starts no reaper, so a driver killed with `SIGKILL` can leave its OpenShell workspace and sandbox behind: the handle is the leak-safe path.
- **The snapshot encoding moved to `midojo_sdk.snapshot`.** `midojo.control_plane.state.snapshot_digest` still works as a wrapper, but code that imported the encoding's helpers from `midojo` must import them from `midojo_sdk.snapshot`. Snapshots computed with `SNAPSHOT_DIGEST_VERSION` 1 keep their IDs.
- **`midojo` depends on `midojo-sdk[mcp]`.** Installing `midojo` now installs the SDK and FastMCP. An environment that pinned the SDK's source to a checkout without the published package, or that installed `midojo` without its dependencies, must install both. The `mcp` extra pins FastMCP as `fastmcp>=3.2.4,<3.3` in `interception/python-sdk/pyproject.toml`: FastMCP 4.0 removed `fastmcp.tools.tool`, which the SDK imports `ToolResult` from, and the SDK wraps FastMCP's private low-level server, so a new minor is a deliberate upgrade.
