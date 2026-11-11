# Rubric judge fixtures

These files hold what Garak's own judge computes, so the port in
`packages/midojo-spec/src/midojo_spec/rubric_judge.py` can be checked against it.

| File | What it holds | Test (in `packages/midojo-spec/tests/`) |
| --- | --- | --- |
| `vectors.json` | Every evidence, request, assessment and judged record the Garak functions give for the cases the port must match | `test_rubric_judge_vectors.py` |
| `fidelity.json` | For each recorded tool call, the sink's text and Garak's evidence value, the failed-call case the port's decoding corrects included | `test_rubric_judge_fidelity.py` |
| `turns.json` | The judge requests, HTTP bodies and records Garak makes for two- and three-turn conversations, under `judge_turns` `final` and `all` | `test_rubric_judge_turns.py` |
| `fences.json` | Canned judge replies (bare, in one markdown code block, malformed, plain prose) with the record Garak's judge gives for each, and what Garak's `_strip_code_fence` returns for a set of texts. The records hold the `model` and `prompt` Garak writes, which the port keeps in the judge exchange row | `test_rubric_judge_fences.py`, `test_rubric_judge_identity.py` |

`vectors.json`, `fidelity.json` and `turns.json` name the Garak revision they were computed
at (`968cc22`). `fences.json` names `3e601af3`, the revision that holds the code-fence reading
(`97d6597f`) and the `model` and `prompt` of the record (`3097e41b`).

The generators of the first three stay in `asago-artifact-generator`, under
`packages/asago-bundle-garak/tests/fixtures/rubric/`: they import Garak's probe and
detector, and two of them also import the consumer's sink and adapter, which a fork
checkout does not have. Regenerate there and copy the JSON files here.

`generate_fences.py` imports only Garak, so it lives here. Run it with a Garak interpreter at
the revision it records, read-only:
`<garak venv python> generate_fences.py <garak revision> fences.json`. It makes no request
and writes only the file it names.
