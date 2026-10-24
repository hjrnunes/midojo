# MiDojo

Man-in-the-middle red teaming framework for AI agents. Tests whether agents can be tricked into unsafe actions via prompt injection.

## Build & Test

```bash
uv sync --extra dev                                   # install all deps (including ruff, pyright, pytest)
uv run pytest                                         # run all unit tests
uv run pytest tests/test_attacks.py                   # single test file
uv run pytest tests/test_attacks.py -k test_verbatim  # single test
uv run ruff check .                                   # lint (whole project)
uv run ruff check --fix src/midojo/serve.py           # lint + autofix single file
uv run ruff format .                                  # format (whole project)
uv run ruff format src/midojo/serve.py                # format single file
uv run pyright                                        # type check (whole project)
uv run pyright src/midojo/serve.py                    # type check single file
```

- prefer `uv run` over activating a virtualenv
- the test suite is fast (<5s); you can run the full suite rather than guessing which tests are affected

## Architecture

The control plane (`src/midojo/control_plane/`) is a FastAPI server whose routes submit events to one writer per evaluation (see Evaluation lifecycle below). The orchestrator (`src/midojo/orchestrator.py`) drives benchmark runs via CLI. Suites define the scenarios; the attack library wraps payloads; verifiers check outcomes.

The orchestrator drives the control plane through a `ControlClient` (`control_plane/client.py`): `ControlPlaneClient` over HTTP, or `InProcessControlClient` (`control_plane/inprocess_control.py`) in the control plane's own process, passed as `run_benchmark(control=...)`. The in-process client calls `ControlPlane` directly, with no HTTP request, and returns the same response models; a refusal raises the control plane's `RejectedError` instead of an `httpx` error.

```mermaid
graph LR
    O[Orchestrator<br/><i>midojo-run CLI</i>] -->|creates eval, grades| CP
    O -->|sends prompt| A
    A[Agent] <-->|tool calls| F[Fake MCP Server<br/><i>intercepts calls</i>]
    F <-->|env state, recording| CP[Control Plane<br/><i>injects attacks, records calls, grades results</i>]
    F <-->|forwards calls| R[Real MCP Server<br/><i>actual tool logic</i>]
```

The fake MCP server is the man-in-the-middle: it sits between the Agent and real Tools, letting the Control Plane inject payloads into tool responses and record every call the Agent makes.

`midojo-serve` (control plane) and `midojo-run` (orchestrator) are 2 separate processes:
- the control plane is long-lived and must be running before the orchestrator starts
- the orchestrator is short-lived — it drives one benchmark run and exit

Start them in order, for example for the weather suite on an OpenShell gateway:

```sh
podman build -t localhost/weather-pi:latest -f suites/weather/sandbox_pi/Containerfile .   # 1. the example agent image (model-agnostic)
export LITELLM_API_HOST=... LITELLM_MODEL=... LITELLM_API_KEY=...                 # the suite needs these to load (optional: LITELLM_API_PORT, LITELLM_API_SCHEME)
export MIDOJO_CONTROL_TOKEN=...                                                   # optional: midojo-serve requires it on all but /agent and /health; midojo-run sends it (or --control-token)
midojo-serve --load-suite weather --port 8090                                     # 2. Control Plane (must be UP before anything else talks to it)
midojo-run --suite weather --gateway GATEWAY --control-url http://localhost:8090  # 3. runs the benchmark (exits when done)
```

for additional examples refer to [README.md](./README.md).

## Key Concepts

- **Suite** (`suites/<name>/suite.yaml`): defines the agent runtime, environment state, user tasks (benign), and injection tasks (adversarial)
- **Episode** (`src/midojo/episode.py`): a frozen description of what to run: the turns (or None when the host drives the conversation), its `midojo_sdk.injection.InjectionPlan`, the security and utility checks as authored specs (either may be absent), the initial environment state, and the runtime definition it needs. Plan instructions separate channel and exact selector, occurrence, transformation, size limit, mode and mismatch policy; `episode.injections` is the plan's derived compatibility substitution map. All the data it owns is frozen (read-only mappings, tuples and exact built-in scalars); checks, environments, the runtime definition and runtimes are built fresh from it each time, so no caller-held data can change how an opened evaluation is graded. The environment type and the verifier implementations are trusted code, shared by reference. An evaluation is one run of an episode. `suite.episode(user_task_id, injection_task_id, injections)` compiles a suite pair into one (the run routes and the orchestrator go through it); a direct caller builds one and opens it with `ControlPlane.open_evaluation(episode)`, with no suite or run registered (`run_id` None, addressed as `ById(None, eval_id)`). Run, suite and task IDs on an evaluation are optional provenance (an episode opened in a run must name its `user_task_id`); grading uses the episode's checks
- **Turns**: a user task has either a `prompt` (one message) or `turns` (messages sent in order in one evaluation and session). The control plane records each turn's message and reply, and stamps every function call with the index of the latest started turn. `--protocol openai` (it chains turns with `previous_response_id`) and the `openshell` runtime support more than one turn; other protocols refuse a multi-turn task before the run starts. The `openshell` runtime runs `agent_command` once per turn in the evaluation's one sandbox, with the turn's message as the last argument and `MIDOJO_TURN_INDEX` (0-based) and `MIDOJO_TURN_COUNT` set (not set for a single turn); each run gets the full timeout. The command keeps its own state between runs, outside `/sandbox/workdir`, or it shows up in the workdir observation
- **Evaluation lifecycle** (`src/midojo/control_plane/lifecycle.py`, `service.py`): the rules are pure transitions, `transition(evaluation, authority, event)` returning `Accepted` (next value, result, effects), `Late` (refused, recorded as a late fact) or `Rejected` (a domain `Rejection` kind; HTTP status codes are chosen only in `dependencies.REJECTION_STATUS`). `ControlPlane.apply` is the single writer: it runs each transition as a step of the store's writer capability (`Store.writer()`, given out once, to the control plane), under one lock per evaluation from look-up to commit, for the HTTP routes and in-process callers alike, never held across an `await`, a tool call or grading. The `Evaluation` is a frozen value; a transition decides on deep copies of its arguments, the writer keeps a copy of the next value, and every public store read returns a copy. Closure deadlines are checked against each event's arrival time (`at`), so they hold even when no one ends the wait. Closure is staged: `open` → `closing` (no new call; admitted calls may finish for `AppConfig.closure_wait_seconds`, `midojo-serve --closure-wait`, default 5) → `collecting` (final observations) → `sealed` (an immutable `Snapshot`; `grading.grade_snapshot` grades it once, and the grade is never replaced). A write closure refuses is a late fact (`Evaluation.late_facts`): it never changes the snapshot and marks its evidence `incomplete` in `evidence_status`
- **Agent runtime** (`agent_runtime`, `src/midojo/runtimes/`): where the agent runs and how MiDojo observes it. `openshell` (the default) runs the agent in an OpenShell sandbox per evaluation, seeds its `files`, and needs a running gateway (`midojo-run --gateway`); the agent command may run in the sandbox for `timeout_seconds` per run, one run per turn (default 300; sandbox startup is not included), which `midojo-run --agent-timeout` overrides. `unmanaged` (experimental) is an agent MiDojo doesn't control, reached with `--agent-uri`/`--protocol` and passed the evaluation session in the `X-Midojo-Session` header (`--agent-timeout` sets its client's timeout, default 120). OpenShell predicates require the `openshell` runtime. A suite or episode holds only the runtime's definition; `suite.new_runtime()`, `episode.new_runtime()` (or `runtimes.build_runtime`) build a fresh runtime that shares no state with another. `midojo-run` builds one per run and runs its evaluations in it one at a time
- **Interception layer** (`interception/`): what stands between the agent and its tools, built with an SDK: `pi-sdk/` (`@midojo/pi-sdk`, PI extensions) or `python-sdk/` (`midojo-sdk`: fake MCP servers with `midojo_sdk.mcp`, LangChain agent middleware with `midojo_sdk.langchain`). The SDKs talk to the control plane's `/agent` API only and don't import `midojo`; the root `pyproject.toml` makes `midojo-sdk` a uv workspace member. `MidojoMCP` alone declares its injection capability before mirrored calls and reports strict mismatches and explicit fallbacks; pass it the tool-output part selected from an episode plan. The Python SDK's seams (`midojo_sdk/recorder.py`) are a `Recorder` (start and finish calls) and an optional `WorldState` (read the simulated environment, change it atomically or at a revision), implemented over HTTP by `AgentControlPlaneClient` and in process by `midojo.control_plane.inprocess.InProcessAgentClient`; both raise `ControlPlaneRejectedError` with the control plane's `kind`, and resends are a wrapping `RetryingRecorder`. A start or finish the Python SDK can't deliver (after its resends, or refused other than as a late fact) is reported to `POST /agent/capture-failures` (`CaptureReporter.report_capture_failure`, sent by the wrapping `ReportingRecorder` that `MidojoMCP` and `MidojoMiddleware` record through); it's kept as an `Evaluation.capture_failures` entry (after the seal, a late fact) that marks `function_calls` incomplete in `evidence_status`, and the record's own error still reaches the agent as before (the PI SDK doesn't report them yet). The PI SDK (`interception/pi-sdk/src/index.ts`, one file with type-only imports so PI's jiti loads it as is) mirrors the seams: `Recorder` and `WorldState` interfaces, `ControlPlaneClient` over HTTP (environment changes are one `PATCH`, at `expectedRevision` when given), `ControlPlaneRejectedError` (a `ControlPlaneError` with `kind`) and `RetryingRecorder`; `createMidojoExtension({ recorder, world })` takes them, and defaults to the control plane's. Its tests are `cd interception/pi-sdk && npm test`. Each start and finish carries the call's disposition (`forwarded`, `simulated`, `denied`, `blocked`; optional, None in older records), stored on `FunctionCallRecord.disposition`: the lifecycle refuses one that doesn't go with `blocked` as `invalid_disposition`, and a finish naming another than its start's as `call_conflict`. Call records carry environment snapshots only with `AppConfig(environment_history=True)` (`midojo-serve --environment-history`), as observations over the call's interval
- **Runtime observations** (`src/midojo/observations.py`): evidence a runtime collects outside the agent, keyed by source (e.g. `openshell`), recorded by the orchestrator and read by verifiers. `openshell` also records what ran: `image` (the reference the suite gave), `image_digest` (null: OpenShell 0.1.2 doesn't report the digest it resolved) and `policy_digest` (`sha256:<hex>` of the effective policy as sorted-key proto JSON, or `builtin` when the suite gives no policy or `policy: {}`)
- **Probe**: a slot in the environment, runtime files, or a user task's prompt or turns where an injection payload gets placed. Referenced via `{injection_task_id:probe_name}` placeholders; suite compilation turns each referenced active probe into an explicit template instruction in the episode plan
- **Attack technique** (`src/midojo/attacks/builtin.py`): wraps a raw payload in a delivery technique (e.g. `important_instructions`, `ignore_previous`, `verbatim`)
- **Payload set** (`src/midojo/attacks/data/`): curated corpus of payloads from external sources (e.g. Garak). Referenced in suite YAML via `source: "garak:<name>"`
- **Verifier** (`src/midojo/verifiers/`): checks whether an injection succeeded, for example with `output_contains`, `env_field_equals`, `env_list_any_match`, `env_field_contains`, `env_field_unchanged`, `env_list_count`, `tool_called` (counts blocked and unfinished calls; unknown rather than False with no matching call when `function_calls` is incomplete), `tool_call_condition` (a `tool-call-condition-v1` condition over the tool calls; unknown when inconclusive, e.g. an unmet condition while a call is unfinished or `function_calls` is incomplete), composable with `any_of`, `all_of`, `not`. A check is three-valued: True, False, or None (unknown) when the evidence it needs wasn't completely collected. The combinators use Kleene logic, and a grade whose top-level check is unknown is invalid (`grade_validity`)

## Patterns for Common Changes

**Add a new suite (bundled)** — follow `suites/weather/` as the minimal example:
1. create `suites/<name>/suite.yaml` with `agent_runtime`, `environment`, `user_tasks`, `injection_tasks`
2. create `suites/<name>/__init__.py` exporting `task_suite` (see below)
3. for the `openshell` runtime, build an agent sandbox image (see `suites/weather/sandbox_pi/`). For an `unmanaged` agent, create fake and real MCP servers under `suites/<name>/a2a_agent/` (see `suites/minibank/`)
4. for convenience, register CLI entrypoints in `pyproject.toml` under `[project.scripts]`

**Use an external suite (out-of-tree)** — suites can live in any Python package; midojo does not need to be forked:
1. expose `task_suite` in your suite's `__init__.py` (the only required attribute):
   ```python
   from pathlib import Path
   from midojo.yaml_task_suite import YAMLTaskSuite
   task_suite = YAMLTaskSuite("my_suite", suite_yaml_path=Path(__file__).parent / "suite.yaml")
   ```
2. optionally export `SYSTEM_MESSAGE` — if defined, midojo forwards it as the system prompt for `--protocol ogx` and `--protocol openai`. Not required: if absent, the agent runs without one (your model endpoint may already have it configured)
3. reference it by dotted module path: `midojo-serve --load-suite my_package.my_suite`; `midojo-run --suite my_package.my_suite ...`
4. fake MCP servers depend on `midojo-sdk[mcp]` and import `from midojo_sdk.mcp import MidojoMCP, ToolContext`

**Add a new attack technique** — add an `AttackTechnique` to the `BUILTIN_TECHNIQUES` list in `src/midojo/attacks/builtin.py`. Each attack technique is a function `(payload: str) -> str` that wraps the payload in a delivery template.

**Add a new verifier** — define a `Verifier` implementation in `src/midojo/verifiers/builtin.py` and register it via `register_verifier()`. The key in suite YAML maps to the verifier name.

**Add a runtime observation source** — define a Pydantic model for the evidence, register it with `register_observation_type(source, model)` from `src/midojo/observations.py`, and return it from the runtime's `observe()` under that source. The control plane rejects observations for unregistered sources or data that doesn't match the model, and verifiers read the validated model from `ctx.observations[source]`. To report how completely each part was collected, give the model an `evidence_status` sub-model of `EvidenceStatus` fields (see `OpenShellEvidence`); grades report it as `evidence_status["<source>.<part>"]`, and a predicate should return unknown rather than False when the part it needs isn't `complete`.

**Add a vendored payload set** — drop a JSON file in `src/midojo/attacks/data/`. It's auto-loaded at import time (`src/midojo/attacks/registry.py`). Use MiDojo's `PayloadSet` shape.

## Conventions

- Python code uses type annotations; pyright runs in `basic` mode
- tests use `pytest-asyncio` for async tests. Fixtures in `tests/conftest.py` provide a loaded suite, environment, FastAPI app, and `TestClient`
- attack taxonomy follows OWASP Top 10 for Agentic Applications (ASI-01 through ASI-10)

## PR Conventions

- ensure commits are signed off (`-s` flag)
