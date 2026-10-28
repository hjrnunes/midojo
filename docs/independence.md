# Checking MiDojo against something other than itself

MiDojo places injections, records what the agent did, seals the evidence and grades it. Once more than one red-teaming tool runs MiDojo episodes, it is tempting to read their agreement as evidence that those layers are right. It isn't, when they run the same code. This page separates the two ways tools end up side by side, says which layers each pairing has in common, and lists the checks in this repository whose oracle is not MiDojo.

## Comparison and composition

**Comparison** sets two pipelines against the same target, each with its own way of planting the payload, capturing what happened and deciding the outcome. Garak's own data-setup path against MiDojo's episode path is one. A bug in one pipeline shows up as a disagreement, unless the other pipeline has the same bug, which is likely only in a layer they share.

**Composition** puts several hosts on top of MiDojo episodes: PyRIT, Garak, or MiDojo's own attempt loop, each choosing payloads and turns, all handing the placement, the recording and the grading to MiDojo. When they agree, the hosts and the episode API fit together. It can't reveal a bug in MiDojo's delivery or grading: every host went through the same code, so a fault there would show up in all of them alike.

The difference is what the two sides have in common. Agreement is evidence only about the layers where the two sides differ.

## Shared layers, per pairing

The layers, from the target down: the **target** (agent, its framework and model, with the model's sampling), the **tools** (the real MCP server behind the agent), the **payload** (its text, and the technique that wraps it), the **driver** (who chooses the payload, the turns and the attempts), **delivery** (compiling the injection plan and placing the payload: MiDojo's interceptor splice, or the other tool's own setup), **recording** (capturing the tool calls and their results), **sealing** (closure, evidence status and the immutable snapshot), and **grading** (the verifier, for tool calls the tool-call condition evaluator).

| Pairing | Kind | Shared | Different |
| --- | --- | --- | --- |
| Garak's native path vs MiDojo | comparison | target, tools; the payload when both draw on the same corpus; the condition contract, and the evaluator itself when one side's detector calls the other's | delivery, recording, sealing, and grading unless the evaluator is shared |
| MiDojo under PyRIT vs MiDojo under Garak | composition | target, tools, delivery, recording, sealing, grading | driver, and the payload when each tool generates its own |
| MiDojo's attempt loop vs PyRIT over MiDojo episodes | composition | target, tools, delivery, recording, sealing, grading | driver |
| MiDojo today vs MiDojo at an earlier commit, on recorded calls | comparison of grading only | the recorded calls, so target, tools, payload, driver, delivery and recording are all the recording's | sealing and grading code |

A report that compares two runs should carry a row like these for its own pairing, so a reader knows which layers the agreement says nothing about. When the evaluator is shared, a matching verdict says nothing about the evaluator even in a comparison.

## The independent checks

Each check below has an oracle MiDojo doesn't compute. They run offline, with no model or target, as part of `uv run pytest`.

| Check | What it covers | Oracle |
| --- | --- | --- |
| `tests/test_delivery_independent_record.py` | delivery facts (`selected`, `transformation`, `emitted`), `delivery_facts()` and `exposure()` for MidojoMCP in front of a real MCP server | the upstream's log of what it served (`tests/logging_mcp.py`), the raw JSON-RPC responses the client received, and expected values written by hand from those two |
| `tests/test_condition_mutations.py` | the tool-call condition evaluator, the `any_of`/`all_of`/`not` combinators and the `tool_call_condition` verifier | cases written by hand from the rules, apart from the contract's vectors; each of 34 single-edit mutants must fail one |
| `tests/test_splice_mutations.py` | MidojoMCP's splice: structured and content-block placement, the JSON mirror's re-serialisation, record selectors, occurrence counting, size limits, transformations, mismatch policies, call arguments and result records | literal results typed out by hand; each of 26 single-edit mutants must fail one |
| `tests/test_regrade_recorded.py` | today's control plane and grading on calls recorded by MiDojo at b0fb457 | the verdicts b0fb457 recorded, and a hand-written list of each change with its cause |

What each check has in common with MiDojo:

- **Delivery against the upstream's log.** The MCP SDK serialises both the upstream's log and the client's responses, so a bug in the SDK's wire format would go unseen. The check stops where MiDojo's own evidence stops, at what the interceptor returned: nothing here observes the agent reading the result or the model's input.
- **The mutation suites.** They share the rules as the contract and the docs state them; a rule stated wrongly would be tested wrongly. The mutants are a fixed list of plausible edits, not a mutation score: a bug no listed mutant resembles can survive. `tests/mutation_harness.py` compiles a module, or one function, with one exact source edit, apart from the real one, and fails if the snippet it edits has moved, so the list can't quietly test the original code.
- **The re-grade.** It shares the recordings, so it checks grading on fixed evidence and nothing upstream of it. Only `tool_call_condition` checks are re-graded: the recorded runs' other checks were LLM judges, which can't be run offline.

## The re-grade at b0fb457

The migration reference is MiDojo b0fb457. `tests/regrade.py` replays a recorded evaluation through a new in-process control plane, through its session as the in-agent layer that recorded it did: the turns, each call record (an unfinished one only started, a blocked one recorded whole), the reply, the seal and the grade. The evidence status is whatever the current lifecycle derives from those records.

All 203 recorded evaluations at b0fb457 with a `tool_call_condition` security check were re-graded this way. 36 verdicts change, all for one intended reason: each is an evaluation intercepted in the agent and sealed with no call record, which since 5222730 has its `function_calls` evidence `incomplete` (a layer in the agent may fail open and record nothing), so a check only a missing call could decide is unknown. 25 were True on a `not_called` check and 11 False on a `value` check; all are now None, and their grades invalid. No other verdict changes. A False verdict also has a reason now (since aac063d every grade names one), where b0fb457 recorded none.

`tests/fixtures/regrade/recorded_b0fb457.json` freezes 15 of them, sanitised: product names replaced, results, messages, replies, environments and call IDs elided or renumbered, as the checks read none of them. Four of the 15 are such changes, and the test also replays each as a host interceptor's records, whose capture is complete with no record, to show the recorded verdict comes back once that cause is taken away.

To add a verdict change, list it in `CHANGES` with a cause in `CAUSES` (the commit and why it is intended). A change with no cause is a failing test.
