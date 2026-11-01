# Versioning the tool-call-condition contract

The `asago-scenario-generator` owns this contract. A version is a directory and
a string: `tool-call-condition-v1` is `tool-call-condition-v1/`, and
`tool-call-condition-v2` is `tool-call-condition-v2/`. A condition does not name
its version; an evaluator states the version it implements.

## What each version means

- **`tool-call-condition-v1`** is the first published version. Its vectors were
  computed by running Garak's `garak/detectors/toolcall.py`
  (`validate_condition`, `evaluate_condition`) at revision
  `877932916b7b3e1148171a49b97eeb75a44c9b39`. Scenario handoffs reference v1,
  so its files must never change.
- **`tool-call-condition-v2`** is the current version. The condition structure
  and validation are unchanged from v1; three evaluation rules change, as
  `tool-call-condition-v2/README.md` states. Its vectors were computed with
  Garak at revision `4f4917d666feea947731ec1588739d8427bc15d9`, which
  evaluates v2. Every v1 evaluate vector is also a v2 vector with the same id,
  and three of them give a different result.

A version changes when a result changes or a validation rule changes. A
clarification that changes no result updates the files in place: the version
string stays, the digests change, and consumers resync the copy. Consumers must
not edit expected values by hand; recompute them with
`scripts/gen_tool_call_condition_vectors.py`.

## Citing a version

A consumer cites the contract by the version string, exactly as written here: a
document, test, or comment names `tool-call-condition-v2` (or the version it
implements). A consumer that evaluates conditions states which version its
evaluator implements, because the vectors hold the expected results for one
version. Reference a version, never a file path alone, so a copy can move
without changing the citation.

## Vendoring a copy

Copy the version directory byte for byte. A consumer that keeps the whole
contract (both versions, the index, and the integrity record) copies this
directory; a consumer that only evaluates one version copies
`tool-call-condition-v2/` alone. A vendored copy needs no other file from the
producer.

The producer's `scripts/sync_tool_call_condition.py` refreshes the known copies:

```bash
python3 scripts/sync_tool_call_condition.py --check \
    --consumer ../asago-artifact-generator --midojo ../midojo --garak ../garak
python3 scripts/sync_tool_call_condition.py --write --midojo ../midojo
```

`--check` verifies the producer tree and every copy it is given, writes
nothing, and exits non-zero on drift. `--write` replaces a copy with the
producer's files. Without a checkout argument the script reads nothing outside
this directory.

## Integrity

`CONTRACT.lock` records the SHA-256 digest of every file in this directory, by
path relative to this directory, and names the current version. That file is
the integrity record: verify a copy by comparing each file with the digest the
copy's `CONTRACT.lock` records, or run the sync script's `--check` mode. A
vendored copy that carries `CONTRACT.lock` references the contract, because the
lock names the version and every digest.
