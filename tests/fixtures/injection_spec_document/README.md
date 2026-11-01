# Injection-spec document goldens

These files are the contract for `midojo-injection-spec-v1`: one JSON document
per shape MiDojo writes, and the documents it must refuse. Every file is
generated. Do not edit one by hand.

## Regenerate

Run the generator in MiDojo's interpreter, from the package root:

```sh
uv run --frozen python tests/fixtures/injection_spec_document/generate.py
```

`generate.py` builds each valid document from MiDojo's own objects
(`InjectionSpec`, `InjectionPlan`, `InjectionInstruction`, and their settings)
and writes it with `midojo.injection_spec_document.dump_document`, so the bytes
are the writer's canonical output. It builds each refused document from
`v1_single.json` with one change. Run it after changing the writer or the
schema, and commit the bytes it wrote; a re-run over an unchanged tree writes
the same bytes, and `tests/test_injection_spec_document.py` checks that by
running the generator into a scratch directory and comparing.

`--out DIR` writes the documents somewhere other than this directory.

## What each file is

| File | Shape |
| --- | --- |
| `v1_single.json` | One tool-output placement, a tool-call condition, and a tool policy that forwards every write. |
| `v1_openshell.json` | The same spec under an OpenShell runtime, with an HTTP-request condition and a utility check. |
| `v1_reply_judge.json` | No placement: a reply judged against a rubric, and a utility check over the reply. |
| `v1_sequential.json` | Two placements (tool output and a host template), two turns, and an environment model. |
| `refused_*.json` | `v1_single.json` with one change each: the reader must refuse every one by name. |

The valid documents all carry `"schema_version": "midojo-injection-spec-v1"`.
The reader also accepts the document's old ID, `midojo-episode-v1`, and the
consumer's v0 wrapper, `midojo-episode-v2`, each with a deprecation warning;
those legacy documents live in `tests/fixtures/legacy_documents/`, because the
writer never produces them.
