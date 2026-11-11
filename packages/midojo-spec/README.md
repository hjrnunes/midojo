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
| `rubric_judge` | The rubric judge, a port of Garak's at 968cc22, and `http_judge`, its one-shot HTTP call with an observer for the exchange record. |

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
| `vectors/judge/` | `request_id.json`, `request.json`, `response.json`, `exchange.json` |
| `vectors/rubric/` | `vectors.json`, `fidelity.json`, `turns.json`; see its README |

## Tests

```sh
uv run --no-project --python 3.11 --with pytest --with-editable packages/midojo-spec python -m pytest packages/midojo-spec
```

## The judge exchange record

A judge service appends one row for each model request, answered or failed, to a JSONL file
(`ExchangeLog`). A row holds the key (`request_id`, `ask`), the judge's identity (`model`,
`prompt_version`), what was sent and received (`messages`, `reply`, `error`, `http_status`,
`finish_reason`, `response_model`, `usage`), the timing (`started_at`, `duration_ms`) and the
SHA-256 of the judge request and of the reply (`request_sha256`, `response_sha256`), which
equal the digests the sealed grade's judgements hold. `digests_hold` recomputes them for a
row that was not redacted. The module docstring of `judge_exchange` is the field reference.

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
