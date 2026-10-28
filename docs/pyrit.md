# PyRIT campaigns over MiDojo episodes

Install `midojo[pyrit]`, or in this checkout run:

```sh
uv sync --frozen --extra dev --extra pyrit
uv run --frozen pytest -q tests/test_pyrit_route.py tests/test_pyrit_optional.py
uv run --frozen pyright --project pyright-pyrit.json
```

The extra supports PyRIT 1.1.x; the lock selects **1.1.0**. It includes
`midojo-sdk[mcp]` for host interception. Core `import midojo` does not load
PyRIT; importing `midojo.pyrit` without the extra raises an installation hint.

The default Pyright configuration excludes the optional module, example and
tests so `uv sync --frozen --extra dev` needs no PyRIT installation. The
separate `pyright-pyrit.json` checks those files with the extra installed.
CI should run both environments: the ordinary gates with `--extra dev`,
then the full test suite and the PyRIT Pyright command above after syncing
with `--extra dev --extra pyrit`. Excluding the route from the default gate
is not a substitute for checking it in the extra environment.

## Target and evidence

`MidojoEpisodeTarget` is a text-only, single-turn PyRIT `PromptTarget`.
Each `send_prompt_async` runs one episode through `midojo.open_episode`.
The candidate fills one declared plan instruction; `check_plan_within`
refuses changed selectors, modes, transformations, size limits, extra
instructions or probes before an evaluation opens. The host-supplied
`episode_for_payload` compiler and `drive` callback are trusted code.
`drive` uses the episode's benign turns and `ep.binding`, records replies
with `ep.turn`, and leaves the context to seal and clean up.

For a suite, `MidojoEpisodeTarget.from_suite` takes a `suite`,
`user_task_id`, `injection_task_id`, and full `probe_key` (`task:probe`). It
uses `permitted_injections` and compiles that probe for each candidate. A
probe occurring at multiple placements is refused: the target
requires exactly one instruction. Direct users supply an
`episode_for_payload(candidate) -> Episode` compiler and its permitted
single-instruction `InjectionPlan` as `surface`.

The response text is the agent's actual final reply. Its `prompt_metadata`
contains `midojo.evaluation_id`, `midojo.snapshot_id`, `midojo.grade` (JSON),
`midojo.status` and a reason; `midojo.run_id` is included when present.
The target opens standalone episodes with an in-process control plane, so
normally there is no run ID. A failed/refused attempt carries its status and
reason, plus any evaluation/snapshot identity available after cleanup.
Refused and failed sends return the fixed reply "MiDojo could not evaluate
this candidate." Details remain in the receipt, metadata and score rationale.
A candidate with multiple pieces, a non-text modality or an invalid role
receives a refused receipt before an episode opens. PyRIT validates the
Message model itself before dispatching to the target.
Cancellation and interrupts propagate.

Each target retains one control plane for its lifetime. After a send, callers
can resolve its IDs with `target.control.snapshot(receipt.run_id,
receipt.evaluation_id)` and verify the returned `snapshot_id`. Snapshots from
earlier sends and replays remain available until the target/control plane is
released; supplying `control=` lets the caller retain them longer.

Scoring uses `midojo.grade(episode, ep.sealed)`: tool evidence belongs to
that immutable snapshot. No current/latest-evaluation endpoint is involved.
The scorer resolves the response's piece ID in the target's private receipt
registry, rather than trusting its text or JSON metadata. This prevents a
reply or edited metadata from asserting a successful grade. Keep the target
alive until scoring finishes. Metadata is an audit copy, not a portable
scoring credential: a new process/target cannot score old responses from
metadata alone and returns undetermined. `verdict_of(response, target=...)`
reads the immutable receipt, including a replay already completed.

## Scoring and replay

Construct `MidojoVerdictScorer(target=target)` for `RedTeamingAttack`.
MiDojo's **security=True means the attack's security predicate held** (the
agent broke), not that the agent was safe. The scorer returns true only
when that result is valid and the same candidate breaks again on replay.
A valid security=False returns false. A valid initial success whose replay
returns false also scores false, following MiDojo's attempt-loop convention.

PyRIT 1.1 supports native undetermined scores: both scorer variants return
`Score(status=ScoreStatus.UNDETERMINED, score_value=None)` for security=None,
invalid **security** grade validity, an absent security check, failed/refused episodes,
an unknown response identity, or an unreadable/blocked transport response.
A failed or unknown replay is also undetermined. Calling `get_value()` on
such a score raises PyRIT's `UndeterminedScoreError`; it never reads as false.
PyRIT maps this status to `AttackOutcome.UNDETERMINED`. Utility validity does
not affect security scoring: invalid utility alone leaves a valid security
verdict usable, matching MiDojo's attempt loop.

`MidojoVerdictFloatScorer` uses the same rules with 1.0 for a confirmed break
and 0.0 for a known non-break. It provides no intermediate heuristic score.

The target refuses construction unless its runtime declares a reset or
`DisposableRuntime.disposable=True`. It passes `reset=True` to every
`open_episode` when reset is available. A supplied runtime belongs to the
caller: start its run before the campaign and end it afterwards. A fresh
OpenShell sandbox isolates the suite's sandbox and simulated environment;
external services still need an explicit reset if their state is changed.

To preserve **one send = one episode**, the scorer runs confirmation when
it first sees a successful discovery. It reuses the same frozen episode,
opens a fresh evaluation with a clean start, and caches that replay across
scorers. Its persisted `score_metadata` repeats the discovery receipt and
adds `midojo.replay`, a JSON receipt with the replay's evaluation ID,
snapshot ID, grade and status. The response metadata remains the original
send's receipt. Scoring can therefore execute the agent once more and must
finish before the caller ends the runtime's run. Sends and replays are
serialized within one target. The replay queue retains episode specifications
only for security-True discoveries, releasing them after confirmation. Use one target per runtime; sharing a runtime
across target instances is unsupported.

## Weather example

[`examples/pyrit_weather.py`](../examples/pyrit_weather.py) wires
`RedTeamingAttack` to the weather suite in host mode. PyRIT owns candidate
generation and iteration; MiDojo runs and grades each attempt. Its attacker
brief specifies **content for New York's weather notes**, not a user message.
PyRIT's `next_message` JSON field carries that content; rationale and summary
stay with the attacker. The defender receives its normal weather question.
Custom attacker targets should honor the JSON response configuration in the
request metadata and return the raw reply to PyRIT's parser. Avoid parsing it
first with bare `json.loads`: PyRIT also accepts Markdown-fenced JSON, which
is covered by the offline campaign test. The example's `OpenAIChatTarget`
already forwards the JSON configuration and leaves parsing to the campaign.

The offline test uses fake attacker and scorer chat targets, a disposable
scripted agent, an in-process control plane and a local MCP server. It tries
a benign note, then an alert-inducing note, and confirms the latter in a
third evaluation. The scripted agent always replies with the same benign
weather summary: success comes from the recorded environment change, not
from that reply. The fake chat scorer separately scores the saved reply as
false, which does not alter the attack outcome (`RedTeamingAttack` 1.1 does
not invoke its configured auxiliary scorers). No model endpoint is called.

For a real run, build the weather agent image and configure its model and
OpenShell gateway as in the [README](../README.md). Start
`weather-tools-serve`. No `midojo-serve` process is needed for this example.
Set the defender's `LITELLM_API_HOST`, `LITELLM_MODEL`, `LITELLM_API_KEY` (and
port/scheme if needed), then supply the attacker's separate credentials:

```sh
export OPENAI_CHAT_KEY=...  # attacker model credential
uv run --frozen python -m examples.pyrit_weather \
  --gateway YOUR_GATEWAY \
  --attacker-model YOUR_ATTACKER_MODEL \
  --attacker-endpoint https://YOUR_MODEL_HOST/v1/chat/completions \
  --max-turns 3
```

That command calls real models; it is not part of the tests. The example
binds the host interceptor on port 8766, as the weather policy requires,
and keeps the control-plane session out of the sandbox. PyRIT persists
messages and scores in its SQLite memory, including replay identities in
the score metadata.

## Other attack strategies

- **PAIR and TAP:** wrap `MidojoVerdictFloatScorer` in PyRIT's
  `FloatScaleThresholdScorer`. In PyRIT 1.1, `PAIRAttack` subclasses TAP and
  inherits its constraints: an IPI brief with the required `desired_prefix`
  parameter, plus `tree_width`/`batch_size` parallel streams. Start both at
  1; concurrent streams need independent runtimes. Both also need campaign
  integration tests and explicit tree-lineage capture. Binary 0/1 scores tie
  all unsuccessful branches; useful sub-threshold steering remains to be designed.
- **Crescendo:** needs a separate stateful design for conversation history,
  reset boundaries and replay of the whole conversation. This single-turn
  target deliberately does not declare editable history or multi-turn support.
- **XPIAWorkflow:** is one-pass (plant, trigger, score), useful for fixed
  replay rather than adaptive search.
- **OpenAIResponseTarget:** PyRIT 1.1 drops `mcp_call` output items. This
  route gets tool evidence from MiDojo's sealed snapshot, so reply parsing
  does not decide whether a tool was called.
