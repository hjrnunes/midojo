# Rubric judge fixtures

These files hold what Garak's own judge computes, so the port in
`interception/python-sdk/src/midojo_sdk/rubric_judge.py` can be checked against it.

| File | What it holds | Test |
| --- | --- | --- |
| `vectors.json` | Every evidence, request, assessment and judged record the Garak functions give for the cases the port must match | `tests/test_rubric_judge_vectors.py` |
| `fidelity.json` | For each recorded tool call, the sink's text and Garak's evidence value, the failed-call case the port's decoding corrects included | `tests/test_rubric_judge_fidelity.py` |
| `turns.json` | The judge requests, HTTP bodies and records Garak makes for two- and three-turn conversations, under `judge_turns` `final` and `all` | `tests/test_rubric_judge_turns.py` |

Each fixture names the Garak revision it was computed at (`968cc22`).

The generators stay in `asago-artifact-generator`, under
`packages/asago-bundle-midojo/tests/fixtures/rubric/`: they import Garak's probe and
detector, and two of them also import the consumer's sink and adapter, which a fork
checkout does not have. Regenerate there and copy the JSON files here.
