# midojo-spec

The contracts MiDojo, the interceptor service, a consumer and Garak must read the same
way, on the standard library alone. The distribution is `midojo-spec`; you import
`midojo_spec`. It declares no dependency and runs on Python 3.11 and later, so a Garak or
consumer environment installs it without MiDojo's runtime.

```sh
pip install midojo-spec
python -c "from midojo_spec import judge; print(judge.SCHEMA)"
```

## Modules

| Module | Holds |
| --- | --- |
| `judge` | The `midojo.judge.v1` request and response rules: `request_id`, `request_problem`, `response_problem`. |
| `judge_exchange` | The record a judge service keeps of each model request (`midojo.judge-exchange.v1`): the row, redaction, the append-only JSONL log and its strict reader. |
| `judge_replay` | Strict offline replay of exchanges keyed by request and judge identity, with stable refusal reasons. |
| `rubric_judge` | The rubric judge, a port of Garak's at 968cc22, and `http_judge`, its one-shot HTTP call with an observer for the exchange record. |
| `snapshot` | A sealed snapshot's ID, `snapshot_digest`, and the lossless encoding it hashes (with the rules that keep older digests valid). |
| `injection` | The immutable injection plan (`InjectionPlan`, instructions, selectors, targets, outcomes, capability gaps) and its codec, and `MIDOJO_MCP_CAPABILITY`, what MiDojo's MCP interceptor places. |
| `injection_spec_schema` | The injection-spec document's version and closed JSON Schema (package data), and the plan codec's version. |
| `recorder` | The two seams an interception layer talks to, `Recorder` and `WorldState`, the facts and dispositions they carry, and the retrying and reporting recorders. |
| `policy` | The tool policy (`ToolPolicy`): which upstream tools an agent may call and what happens to its writes, in the document form a driver states and the interceptor decodes, and `tool_policy`, the interceptor's view of it. |
| `link` | The `midojo-binding-v1` frame and message codec: encode and decode of frames, answers, rejections, `bind`, `bound` and `unbind`, and the errors a link raises. |

Import the module you need: the package imports none of them.

## Rules for a change

- Add no dependency. A test reads every module's imports and fails on anything outside
  the standard library and this package.
- Write nothing Python 3.11 cannot parse. A test parses every module as 3.11, and bans
  `type` statements (use `TypeAlias`).
- Wire IDs (`midojo.judge.v1`, `midojo.judge-exchange.v1`) are independent of the package
  name and stay when it changes.

## Vectors

`vectors/` holds the conformance vectors: inputs and the output each rule must give,
computed by the implementation the rule came from (or by hand, for a format that is new)
and cross-checked with tools that share no code with it (`jq -cS`, `shasum -a 256`). A
reader in another language runs the same files. The vectors ship in the source
distribution, not the wheel.

| Directory | Files |
| --- | --- |
| `vectors/judge/` | `request_id.json`, `request.json`, `response.json`, `exchange.json`, `replay.json`; see its README |
| `vectors/rubric/` | `vectors.json`, `fidelity.json`, `turns.json`; see its README |
| `vectors/snapshot/` | `vectors.json`: snapshots and their digests, and the snapshots a reader must refuse (`refusal`; `refused_value` names the `bytes` or non-finite `float` that JSON cannot carry, and the `paths` it replaces) |
| `vectors/plan/` | `plans.json` (accepted plans, defaults and refusals; an accepted plan is a fixpoint of decode then encode) and `before_call_arguments.json` (a plan as an older codec wrote it) |
| `vectors/binding/` | `codec.json`: frames, answers, rejections and messages the link codec accepts and refuses |
| `vectors/policy/` | `policy.json`: the tool policy's accepted and refused document data, and the tools it mirrors and blocks |

## Tests

```sh
uv run --no-project --python 3.11 --with pytest --with-editable packages/midojo-spec python -m pytest packages/midojo-spec
```

## Replay

`judge_replay.replay(exchanges, request_id, ask, model=..., prompt_version=...)`
returns the first row with that key and identity if all matching rows agree on
`reply`, `error` and `response_sha256`. Redaction can hide different original
replies behind the same text, but their response digests still differ. An
`Exchange` has no error digest; a matching failed call returns its row with
`error`, not a reply. The requested model and prompt version must each be a non-empty string.
`identity_problem` checks each candidate's model and prompt version in that order:
missing identity, wrong model, then wrong prompt version. A missing key raises
`UnrecordedExchange`; a key without the requested identity raises `IdentityRefused`
with `reason` (`missing`, `model`, or `prompt_version`); disagreeing matching rows
raise `AmbiguousExchange`. Their stable `why` values are
`judge-replay-unrecorded`, `judge-replay-identity`, and
`judge-replay-ambiguous`. The functions read no environment and open no connection.
A live re-judge with another model remains possible; each model's recorded rows
replay under its own identity.

Exchange replay compares raw reply text, error and response digest. MiDojo's
graded `JudgeRecord` replay instead compares verdict and failure: records with
the same verdict but different reply text are ambiguous here, not in graded
replay. Both layers apply the same strict model and prompt-version identity check.

## The judge exchange record

A judge service appends one row for each model request, answered or failed, to a JSONL file
(`ExchangeLog`). A row holds the key (`request_id`, `ask`), the judge's identity (`model`,
`prompt_version`), what was sent and received (`messages`, `reply`, `error`, `http_status`,
`finish_reason`, `response_model`, `usage`), the timing (`started_at`, `duration_ms`) and the
SHA-256 of the judge request and of the reply (`request_sha256`, `response_sha256`), which
equal the digests the sealed grade's judgements hold. A row that was not redacted must hold
the digests of its own text, and the log and the reader refuse one that does not;
`digests_hold` says `None` for a redacted row, which cannot be checked.

Redaction replaces each occurrence of each secret with `[redacted]` in the messages, the
reply, the error, the finish reason, the response model and the usage, and never in the
digests. It does not touch `model` or `prompt_version`: they are the identity a replay
checks, so a secret that occurs in either (or in the marker, or that the marker would spell
with its neighbours) is refused with `ExchangeFormatError`. A secret that two messages hold
in halves is in neither, and is not found.

Every integer in a row fits a signed 64-bit integer (`MAX_INTEGER`, `MIN_INTEGER`); `usage`
drops one that does not, and any other field refuses it. A log whose last row holds the
largest sequence refuses the next row and stays readable.

`ExchangeLog` assumes one process appends to a file at a time. It makes its path absolute
when you make it, and every log on the same file in a process shares a lock and a count,
whatever path each names (the file's device and inode decide, so a relative path, a symlink or
a renamed directory cannot make two logs number rows alike). It reads the file again, strictly,
when its size or modification time is not what the process left, and refuses a file that does
not read as a log or ends in a partial row; an edit that keeps both size and modification time
goes unnoticed until the file is read. A write that fails is cut back so no partial row stays.
The module docstring of `judge_exchange` is the field reference.

```python
from midojo_spec.judge_exchange import Exchange, ExchangeLog
from midojo_spec.rubric_judge import PROMPT_VERSION, http_judge

log = ExchangeLog("judge-record.jsonl")


def observe(messages, observed):
    log.append(
        Exchange.observed(
            observed,
            request_id=request["id"],
            ask=ask,
            model=model,
            prompt_version=PROMPT_VERSION,
            messages=messages,
            secrets=[service_token, model_key],
        )
    )


call = http_judge(model_url, model, observer=observe)
```
