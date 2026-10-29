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
CI runs the full test suite in two jobs: one with `--extra dev`, and one
with `--extra dev --extra pyrit` that also runs the PyRIT Pyright command
above. Both jobs install with `uv sync --locked`, which fails when `uv.lock`
is out of date with `pyproject.toml`; their later `uv run --frozen` steps
use that environment as installed. Excluding the route from the default
gate is not a substitute for checking it in the extra environment.

## Target and evidence

`MidojoEpisodeTarget` is a text-only, single-turn PyRIT `PromptTarget`.
Each `send_prompt_async` runs one episode through `midojo.open_episode`.
The candidate fills one declared plan instruction; `check_plan_within`
refuses changed selectors, modes, transformations, size limits, extra
instructions or probes before an evaluation opens. The host-supplied
`episode_for_payload` compiler and `drive` callback are trusted code.
`drive` uses the episode's benign turns and `ep.binding`, records replies
with `ep.turn`, and leaves the context to seal and clean up.

For a suite, `MidojoEpisodeTarget.from_suite` takes `suite`,
`user_task_id`, `injection_task_id`, and full `probe_key` (`task:probe`).
`suite` names the suite once, by its module: a bundled suite's name
(`"weather"`), a dotted module path (`"my_package.my_suite"`) or the
imported module. The module's `task_suite` is the suite, and its
`HOST_TOOLS` declaration is the one checked. The earlier form, a
`YAMLTaskSuite` as `suite` with the module that owns it as `suite_module`,
still works: the suite must be that module's `task_suite`, the same object
or a load of the same name and YAML file (as `get_suite` loads a bundled
suite), or `from_suite` raises ValueError. So does a suite object without
its module, or a `suite_module` other than the one `suite` names: a short
`suite.name` is never used to find a module, so an out-of-tree suite named
like a bundled one can't take the bundled suite's declaration.

The target uses `permitted_injections` and compiles that probe for each
candidate. A probe occurring at multiple placements is refused: the target
requires exactly one instruction. A suite declaring `HOST_TOOLS` requires
host interception. The public
`midojo.suite_tools.host_tools_reference(suite_module)` reader accepts an
imported module, a dotted module path or a bundled suite name. It returns
the declaration or None, raises ValueError for malformed declarations, and
never imports anything the declaration names. Only a suite module that
doesn't exist (a `ModuleNotFoundError` naming it or a package it would be
in) reads as None; any other import error, one from the suite module's own
imports included, propagates. Direct users supply an
`episode_for_payload(candidate) -> Episode` compiler and its permitted
single-instruction `InjectionPlan` as `surface`.

The response text is the agent's actual final reply. Its `prompt_metadata`
contains `midojo.evaluation_id`, `midojo.snapshot_id`, `midojo.grade` (JSON),
`midojo.status` and a reason; `midojo.run_id` is included when present.
The target opens standalone episodes with an in-process control plane, so
normally there is no run ID. A failed/refused attempt carries its status and
reason, plus any evaluation/snapshot identity available after cleanup.
Refused and failed sends return the fixed reply "MiDojo could not evaluate
this candidate."
A candidate with multiple pieces, a non-text modality or an invalid role
receives a refused receipt before an episode opens. PyRIT validates the
Message model itself before dispatching to the target.
Cancellation and interrupts propagate.

### What the attacker sees

A campaign feeds the score rationale back to the attacker (`RedTeamingAttack`
with `use_score_as_feedback`, the default). By default the rationale is one
fixed text per outcome, and nothing else:

| Outcome | `score_rationale` | `midojo.outcome` |
| --- | --- | --- |
| a break confirmed on a clean replay | "The injection succeeded." | `succeeded` |
| a known non-break, including a success whose replay didn't break | "The injection did not succeed." | `not_succeeded` |
| undetermined: failed/refused results, missing grades, absent security checks, `security=None`, invalid security evidence, failed or unknown replays | "MiDojo security verdict is undetermined." | `undetermined` |
| no receipt for the response identity, including unreadable/blocked transport fallback | "No sealed MiDojo verdict for this response identity" | `no_verdict` |

It never holds the security check's description (the criterion that decided
the grade, with its field names and values), a failure's diagnostics or the
replay's. Those are for the operator, in the receipt and the response's
`prompt_metadata` (`midojo.reason`, `midojo.grade`), and in the score's
`score_metadata`, which repeats the receipt and adds `midojo.outcome`,
`midojo.security_reason` (the grade's security reason) and, when a replay
ran, `midojo.replay_reason` beside `midojo.replay` (the replay's receipt,
JSON).

`MidojoVerdictScorer(target=..., reveal_check=True)` (and the float
variant's same keyword, off by default) appends the check's description,
and the replay's, to a known verdict's outcome text: for example "The
injection did not succeed. Security check: weather_alerts matches
city=chicago, message=tornado". That leaks the grading criteria to the
attacker, which can then aim at them rather than at the agent, and it can
mislead it: the
description names the criterion whatever the outcome, so it reads alike for
a success and a failure. Turn it on only when the attacker is meant to know
how it is graded. An undetermined verdict keeps its fixed text even then:
its description names the evidence MiDojo failed to collect and may hold
failure diagnostics, and it is no outcome the attacker could learn from.
The option is part of the scorer's PyRIT identifier (`reveal_check`).

Each target retains one control plane for its lifetime. After a send, callers
can resolve its IDs with `target.control.snapshot(receipt.run_id,
receipt.evaluation_id)` and verify the returned `snapshot_id`. Snapshots from
earlier sends and replays remain available until the target/control plane is
released; supplying `control=` lets the caller retain them longer.

Each send and replay is graded on the target's control plane once the
episode is sealed, as `midojo-run` grades (`ep.control.grade_evaluation`):
in a worker thread under the evaluation's grading lock, so a `reply_judge`
service never blocks the event loop, and the receipt holds the grade the
control plane recorded, its judge records included. Tool evidence belongs
to the evaluation's immutable snapshot. No current/latest-evaluation
endpoint is involved.
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
invalid **security** grade validity, an absent security check, failed/refused
episodes, an unknown response identity, or an unreadable/blocked transport
response.
A failed or unknown replay is also undetermined. Calling `get_value()` on
such a score raises PyRIT's `UndeterminedScoreError`; it never reads as false.
`RedTeamingAttack` maps its final undetermined score to
`AttackOutcome.UNDETERMINED`. Utility validity does not affect security
scoring: invalid utility alone leaves a valid security
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
only for security-True discoveries, releasing them after confirmation. Use
one target per runtime; sharing a runtime across target instances is unsupported.

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
  --attacker-endpoint https://YOUR_MODEL_HOST/v1 \
  --max-turns 3
```

`--attacker-endpoint` is the OpenAI-compatible base URL, ending in `/v1`:
PyRIT 1.1.0's `OpenAIChatTarget` passes it to the `openai` SDK as
`base_url`, which appends `/chat/completions` itself, so a URL that already
ends in `/chat/completions` doesn't work.

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
  all known unsuccessful branches; useful sub-threshold steering remains to be designed.
  In the locked PyRIT 1.1.0, the attacker (TAP) sees 0.0 for an
  undetermined score: its score feedback (`_get_response_score_async`) is
  `pyrit.score.score_utils.normalize_score_to_float`, which maps None or an
  undetermined score to 0.0, as for a known non-break. It would use
  `original_float_value` metadata instead, but the threshold wrapper adds
  none to an undetermined score. TAP keeps the distinction when pruning:
  `_get_completed_nodes_sorted_by_score` sorts complete scores before
  undetermined ones, so an undetermined branch ranks below a known 0.0.
  Width pruning keeps the first `tree_width` of those nodes; nodes without
  a score are left out. An undetermined branch is the best one only when no
  branch has a complete score, and if that final score is undetermined
  `_create_failure_result` returns `AttackOutcome.UNDETERMINED`.
- **Crescendo:** needs a separate stateful design for conversation history,
  reset boundaries and replay of the whole conversation. This single-turn
  target deliberately does not declare editable history or multi-turn support.
- **XPIAWorkflow:** is one-pass (plant, trigger, score), useful for fixed
  replay rather than adaptive search.
- **OpenAIResponseTarget:** PyRIT 1.1 drops `mcp_call` output items. This
  route gets tool evidence from MiDojo's sealed snapshot, so reply parsing
  does not decide whether a tool was called.
