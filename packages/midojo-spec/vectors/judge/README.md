# Judge vectors

What the `midojo.judge.v1` rules and the `midojo.judge-exchange.v1` record must give.

| File | What it holds | Test (in `packages/midojo-spec/tests/`) |
| --- | --- | --- |
| `request_id.json` | Request bodies and their `sha256:` ids | `test_judge_vectors.py` |
| `request.json` | Values and why each is not a judge request, or `null` | `test_judge_vectors.py` |
| `response.json` | Answers and why each is not a response to the request id `expected_id`, or `null` | `test_judge_vectors.py` |
| `exchange.json` | Exchange rows that are valid (and whether their digests hold), rows that must be refused (a digest that is not of the row's text included), how a model request becomes a row under redaction, the redactions that are refused (`redaction_refused`), and SHA-256 of texts | `test_judge_exchange.py` |

## How they were computed

- `request_id.json`, `request.json` and `response.json` come from the implementations the
  rules were taken from: the MiDojo runtime's `verifiers/judge.py` (`request_id`,
  `response_problem`) and the consumer judge service's `request_problem`, both at the
  commits the reset branch started from. Their reason texts are kept as they were, so a
  service that moved to `midojo_spec.judge` gives the same reasons it gave before. The
  ids were also recomputed from `jq -cS` output hashed with `shasum -a 256`, which share no
  code with the implementation (the cases with a lone surrogate or a float that JSON writes
  more than one way are the exceptions: `jq` cannot read the first).
- `exchange.json` is new, so its rows and expected reasons are written by hand. Only the
  SHA-256 digests are computed, and `shasum -a 256` agrees with them (a lone surrogate
  hashes as `surrogatepass`, which `shasum` cannot take).

A change to a rule changes the vector in the same commit, and says why.
