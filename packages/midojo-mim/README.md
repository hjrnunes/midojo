# midojo-mim

The man-in-the-middle interceptor of [MiDojo](https://github.com/asago-ai/midojo): the service a remote evaluation binds over one WebSocket, the MCP face the agent calls, the placement of injection payloads in tool results, and the recorder and world adapters that send every record and environment read back to the driver. It depends on [`midojo-spec`](../midojo-spec) and FastMCP, and on nothing of MiDojo's harness: the interceptor runs beside the agent, so an image of it holds no verifier, runtime or control plane.

```sh
pip install midojo-mim
midojo-intercept-serve --host 0.0.0.0 --port 8767        # the credential: --credential, or $MIDOJO_INTERCEPTOR_CREDENTIAL
```

Python 3.11 and later.

| Module | What it holds |
| --- | --- |
| `midojo_mim.service` | `InterceptorService`, the listener, and `main`, the `midojo-intercept-serve` entry point. |
| `midojo_mim.mcp` | `MidojoMCP`, the MCP face: simulated tools, mirroring the upstream, refusals, recording and placement. |
| `midojo_mim.placement` | The placement engine: matching, occurrence counting, transformation, size limits and the built result. Standard library only. |
| `midojo_mim.link` | `Link`, `LinkRecorder` and `LinkWorld`: the service's end of a `midojo-binding-v1` stream. |
| `midojo_mim.endpoint`, `midojo_mim.listener` | The tool endpoint's path and URL, and the listener plumbing the service and the driver's in-process interceptor share. |

## The service

One listener serves the bind upgrade, `GET /v1/bindings` with `Authorization: Bearer <bind credential>` and the subprotocol `midojo-binding.v1`, and one MCP endpoint for each bound evaluation, `/b/<binding id>/mcp`, behind that binding's own token. The first frame the driver sends is `bind {api, plan, policy, upstream_url, host_tools, faces}`; the service answers `bound {tool_url, tool_token, tools}` or `rejected {kind, detail}`. From then on every call, delivery fact and environment read crosses the socket as a `midojo-binding-v1` frame, and the service holds no record and never sees the evaluation's session token. `unbind` is answered by `unbound {injection_outcomes}`. A closed socket, an answer past the deadline or a missed heartbeat is a loss: pending requests fail, the socket closes with 1011 and the binding's token gets 401 from then on. The service runs no host tools in v1 (`host_tools` is `[]`) and serves the MCP face only (`faces` is `["mcp"]`); a bind that asks for more is `rejected`. The frame codec is `midojo_spec.link`, and the driver's end of the link is `midojo_sdk.binding` in `midojo-sdk`.

On start the service prints `midojo-intercept-serve listening on <url>/v1/bindings`: the line a launcher waits for.

## The image

[`Containerfile`](Containerfile) builds the service's image from this package and `midojo-spec` alone. From the repository root:

```sh
docker build -f packages/midojo-mim/Containerfile -t midojo-intercept-serve .
docker run --rm -e MIDOJO_INTERCEPTOR_CREDENTIAL=... -p 8767:8767 midojo-intercept-serve
```

The image runs as a non-root user, and with any user ID, so an orchestrator that assigns one can run it.

## The MCP face

`midojo_mim.mcp.MidojoMCP` builds a fake MCP server on [FastMCP](https://gofastmcp.com). Each tool takes a `ToolContext` (`midojo_spec.context`) as its first parameter. The face passes it in and hides it from the tool schema the agent sees. You give the face the evaluation's `Recorder` and `WorldState` (see [Recorder and WorldState](#recorder-and-worldstate)); the interceptor service gives each binding one whose every call crosses the link to the driver.

```python
from midojo_mim.mcp import MidojoMCP
from midojo_spec.context import ToolContext

mcp = MidojoMCP(
    "alerts",
    recorder=recorder,
    world=world,
    evaluation_id="run-1:eval-7",  # needed when the plan is not empty
    upstream_url="http://localhost:8081/mcp",  # the real MCP server
)


@mcp.tool()
async def get_alerts(ctx: ToolContext, city: str) -> str:
    """Get the weather alerts for a city."""
    real = await ctx.forward("get_alerts", {"city": city})
    return real + await ctx.env("injected_alert")
```

- `await ctx.env(field)` reads a field of the evaluation's environment, and `await ctx.read()` the whole environment with its revision
- `await ctx.env_update(field, value)` sets one field in one atomic step, so a concurrent change to another field isn't lost. Pass `expected_revision=` the revision `ctx.read()` returned to apply it only if nothing changed the environment since: it returns `False` otherwise
- `await ctx.change({field: value, ...}, expected_revision=...)` sets several fields in one atomic step: all of them change, or none. A tool that reads the environment, computes new values and writes them back passes the revision it read, and reads again when it gets `False` (see `suites/minibank/host_tools.py`)
- `await ctx.forward(tool, args)` calls the tool on the upstream MCP server and returns its text

The face attempts to record each call with its arguments and the result it returns to the agent (recorded before it is returned, so not proof the agent received it): its text, and for a mirrored call its structured content and injection outcomes. Those details go through an optional capability, `DetailedRecorder.finish_call_detailed` (`records_details(recorder)` says whether a recorder has it): The driver's recorders have it, and `RetryingRecorder` and `ReportingRecorder` (`midojo_spec.recorder`) have it when the recorder they wrap does. A custom recorder with only `finish_call` keeps working unchanged: MidojoMCP finishes the call with `finish_call` (`finish_detailed` falls back), the agent gets its result, a warning is logged for every call finished that way (a text-only result no instruction matched included), and the control plane keeps the record with `injection_outcomes` null, which says the details weren't reported. 

`await mcp.mirror_upstream()` exposes every upstream tool the server doesn't define itself; the upstream is `upstream_url`, or `upstream_transport` for anything else a fastmcp `Client` connects to. Each downstream session gets one upstream connection, opened on its first call and closed when the session ends. A mirrored tool is listed exactly as the upstream lists it (name, description, input and output schemas, annotations, `_meta`), and a call gets the upstream's result as sent: content blocks, structured content, error flag, and a JSON-RPC error as the same JSON-RPC error. `mcp.block(*names)` and `@mcp.deny(name)` refuse calls: the tool never runs and the agent gets an error result, "Tool execution was blocked" or the reason the denial check returned. Each call has a disposition: `forwarded` (mirrored), `simulated` (a tool defined with `@mcp.tool()`), `blocked` or `denied`. The control plane records each call's disposition, with the `blocked` flag it implies (see [the control plane API](../README.md#control-plane-api)).

## Injection plans

Pass an exact frozen `InjectionPlan` as `MidojoMCP(..., plan=plan)`. For an injection spec's plan containing host template channels, select the MCP-owned part with `spec.plan.for_channels(Channel.TOOL_OUTPUT)`, where `spec` is the injection spec (`midojo.InjectionSpec`). `append` is the one default because an unqualified content selector can preserve the upstream result without guessing a field. Unqualified content-block `replace` replaces the first text block and leaves `structuredContent` unchanged. MidojoMCP supports exact content-block and structured-content selectors on mirrored results, and `mirror_upstream()` validates targeted tool names before serving calls. A non-empty plan requires an upstream; registering a simulated tool is rejected when a named or all-tools instruction would match it.

A structured-content path uses string keys, non-negative integer indexes, and frozen `RecordSelector("NOTE-1")` steps to select the sole list record whose `id` equals `"NOTE-1"`; `field="sku"` selects a custom identifier field. The plain-data/JSON selector shape is `{"field": "id", "equals": "NOTE-1"}` (`field` may be omitted for `id`), exposed by `InjectionPlan.to_data()`/`from_data()` and `to_json()`/`from_json()`. A plain key against a list fails strictly—for example, `field 'NOTE-1' requires an object, got list`—so record ids must use a selector step. A single-record lookup chosen by the call's argument is matched on the call instead: `InjectionTarget(tool="lookup_order", surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=("item",), call_arguments={"order_id": "ORD-102"})` (or `(name, value)` pairs, `(("order_id", "ORD-102"),)`: either way `target.call_arguments` is the pairs sorted by name), in JSON `"call_arguments": {"order_id": "ORD-102"}` (left out when empty), places the payload only on calls whose top-level arguments include each named one with an equal JSON scalar of the same type; any other call to the tool is left unchanged with an `unmatched` outcome at occurrence 0, and occurrence rules count only matching calls. A single-object result whose record id is in the result only (`get_referral(patient_id)` answering `{"referral_id": "REF-1", ...}`) is matched on the result instead: `InjectionTarget(tool="get_referral", surface=SelectorSurface.STRUCTURED_CONTENT, structured_content_path=("reason",), result_record=RecordSelector("REF-1", field="referral_id"))`, in JSON `"result_record": {"field": "referral_id", "equals": "REF-1"}` (left out when None; `field` omitted means `id`), places the payload only on a result whose structured content is an object whose `field` equals the value exactly, as the list record selector matches. Use `call_arguments` when the id is an argument of the call and `result_record` when it is only in the result; both may be given, and both must match. Any other result, one without structured content included (a `content_block` target is decided by the structured content too), is left unchanged with an `unmatched` outcome at occurrence 0 (`result field 'referral_id' is 'REF-2', not 'REF-1'; result unchanged`, `result field 'referral_id' is missing; result unchanged`, `result has no structured content to match field 'referral_id'; result unchanged`), and so is a call whose upstream produced no result; occurrence rules count only the results that match, in the order they come back. `InterceptorCapability.result_record` declares support (`MidojoMCP` has it), and `capability_gaps` reports a `result_record` gap otherwise.

A structured-content placement rewrites the text too when the result's text mirrors its structured content: exactly one content block, a text block whose text parses as JSON equal (type for type) to the original `structuredContent`, as FastMCP sends it. The text is serialised again from the injected structured content with the first predefined `json.dumps` style (compact, spaced, or two- or four-space indented; ASCII escapes or not; keys in the structured content's order or sorted) that reproduces the original text exactly. A key order of the text's own, trailing whitespace or a final newline, and another serialiser's number format aren't reproduced: such text is written compact when it was one line, else indented by its second line's leading spaces. Unescaped non-ASCII is tried first, so an all-ASCII original and any fallback get a non-ASCII payload unescaped. The outcome is `applied` with detail `exact selector matched; mirrored text re-serialised`. Text that isn't a mirror is left as sent and the outcome is `partial`: the payload is in `structuredContent` only, and `detail` says why the text was left alone. `fallback_append` doesn't change that: a structured placement that matched is never relocated.

Unsupported placements raise `CapabilityError` with every gap. Strict selector and size mismatches leave the result unchanged; `fallback_append` is explicit. `mcp.injection_outcomes` reports every result with its evaluation/session `scope` and recorded `call_id`, and each forwarded call's detailed finish carries the call's own outcomes (`injection_outcomes`, `InjectionRecord`s without the scope: every instruction that matched the call's tool: admitted, skipped by its occurrence rule, or `unmatched` at occurrence 0 when its `call_arguments` didn't match the call's arguments or its `result_record` the call's result; only that outcome, with empty delivery facts, has occurrence 0: `InjectionOutcome` refuses any other, and so does the control plane) and the structured content it returns to the agent (`structured_content`; recorded before it is returned, so not proof the agent received it), which the control plane keeps on the call record. A returned result with `isError=True` still receives the selected payload; an exception or cancellation instead reports `no_result`.

Each outcome is also the instruction's delivery fact for the call: what MidojoMCP did at its own boundary. `selected` says whether the instruction's exact selector matched a place in the result (`applied` and `partial`), `transformation` names the transformation applied to the payload (`identity` or `json_string`; `None` for `skipped_occurrence`, `no_result` and an `unmatched` call whose arguments didn't match the target's `call_arguments` or whose result didn't match its `result_record`, where it never was), and `emitted` names the parts of the returned result that carry the transformed payload where it was placed: `structured_content` when the field it wrote still holds the payload where it wrote it, with no later replace of that field (even with the same text), and `content_block` when the text block the instruction wrote into or appended still holds the payload where it wrote it or, for a mirrored placement, while the text is still a JSON mirror of that structured content or the block it re-serialised still holds that text. Only the place the instruction wrote counts: MidojoMCP follows each block and each structured field (by the object keys and list indexes its path resolved to, a record selector's included) through the call's instructions, and the same text in another block or field, or in a block or field a later instruction replaced, isn't the placement; an append keeps what was there. A call whose arguments or result didn't match places nothing: its `unmatched` outcome is a fact with `selected` false and `emitted` empty. Emission is checked on the result as returned, after every instruction for the call, so a placement a later instruction replaced is emitted nowhere. A fact says what MidojoMCP sent, never what the agent or its model saw: the call is recorded before its result is returned. The control plane keeps the facts on `CallInjectionOutcome` (`None` when a layer doesn't report them), seals them in the snapshot, and `midojo-run` derives each payload's exposure from them.

A placement (`applied`, `partial`, `fallback`) also says what it placed and where, so a generated payload is identifiable from the sealed record without the plan: `payload_digest` is `sha256:<hex>` of the UTF-8 bytes of the instruction's payload before its transformation (`payload_digest(text)`), `placed_digest` the same of the transformed text it wrote (equal for `identity`), and `slot` the instruction's `InjectionSlot`, sent as `InjectionSlot.to_data()`: `{"id", "target", "mode"}`, the target in the plan codec's form (as `InjectionPlan.to_data()` gives an instruction's `target`) and `id` `sha256:<hex>` of that target's sorted-key compact JSON, one per place whatever the mode. `InjectionSlot.of(instruction)` gives the same data for an instruction of a fixed plan. Any other outcome leaves them out: `InjectionOutcome` refuses them there, and so does the control plane.

Matching calls are numbered per evaluation under a lock after `start_call` returns and before forwarding, by the default engine's first `decide`. Concurrent calls therefore use interceptor-admission order, not arrival, recorded-start, or completion order: two recorder starts can be stored in one order and return in another. Blocked and denied calls are never admitted; an upstream exception or cancellation consumes the admitted occurrence and reports `no_result`. A cancelled call remains unfinished in the recorder.

When `recorder=` is injected, a non-empty plan also requires `evaluation_id=`. That binds the interceptor and all its counters to one evaluation; build a separate `MidojoMCP` for another evaluation. Without an injected recorder, the incoming evaluation session is the scope.

There is no session-end hook that both identifies the scope and leaves outcomes available for inspection. A long-lived MidojoMCP serving many session-backed evaluations therefore retains its engine's counters and outcomes for each seen scope until process exit. Prefer one interceptor per evaluation where the deployment permits it.

Automated generator output should go through `instruction_from_generator_for_plan(output, spec.plan)`. It accepts only tools, structured-content field paths, and modes listed in that plan's `generator_permissions`.

## Placement engines

`midojo_mim.placement` holds the plan logic: matching a call's arguments or result to an instruction, counting occurrences, transforming the payload, applying size limits and the mismatch policy, and building the placed result, the text mirror included. `MidojoMCP` forwards the call, applies the finished result and records; it keeps no plan state, and `block()`/`deny()` stay on the server. The module uses the standard library only, so you can call it without FastMCP.

The seam is `PlacementEngine`:

- `decide(call, result)` runs once before the upstream call, with `result` `None`, and once on the result it returned. It returns a `PlacementDecision`: `leave(...)` or `replace(result, outcomes)`.
- `settle(decision, emitted)` runs once the result as returned is known and gives the call's outcomes.
- `outcomes()` lists every outcome so far, which `mcp.injection_outcomes` reports.

`PlanPlacementEngine(plan, evaluation_id=...)` is the default: `MidojoMCP(..., plan=plan)` builds one. Pass `engine=` to place with an engine of your own. A server takes a plan or an engine, not both, and one given an engine holds no plan (`mcp.injection_plan` is empty unless that engine has one). `place_field(result, path, text, mode=...)` returns a result with one structured-content field placed, for an engine that places on its own; `mcp.injection_engine` is the engine in use.

A server checks the plan only when its engine declares one (`placement.Planned`). An engine without a plan skips the capability gaps, the mirrored-upstream requirement and the simulated-tool gaps, because each of those reads `mcp.injection_plan`; such an engine is responsible for the placements it makes. A `PlanPlacementEngine` counts occurrences per evaluation, so a server built from one with a recorder needs an `evaluation_id`, whether the plan arrives through `plan=` or through `engine=`.

## Recorder and WorldState

`midojo_spec.recorder` defines the two seams the interceptor talks to. A `Recorder` starts and finishes the calls of one evaluation, each with its disposition. A `WorldState` reads the evaluation's simulated environment (`read()`: its revision and fields) and changes it (`change(values, expected_revision=None)`: one atomic step, or `None` when `expected_revision` is stale). `MidojoMCP(name, recorder=..., world=...)` records through `recorder` and gives simulated tools `world`. Forwarding (`mirror_upstream`) needs only a recorder: a face built with a recorder and no world has no environment, and `ctx.env` raises. A face with no recorder takes `sessions=`, an object that names each request's recorder and world (`midojo_mim.mcp.Sessions`); the SDK's agent mode is one.

In the service, `LinkRecorder` and `LinkWorld` (`midojo_mim.link`) are the two seams over one binding's stream: every call is one frame, sent once and never resent, because the driver may already have applied it. A record is what the agent asked for and what the layer returned to it, not proof the agent received the result, nor of an effect; the environment is what simulated tools did; what a forwarded call did to a real system is neither, and a runtime observes it. Nothing makes them one transaction: a record, an environment change and a real tool's effect each commit, or fail, on their own.

## Where the rest lives

- [`midojo-spec`](../midojo-spec): the plan codec, the frame codec, the recorder seams, the tool policy and every other contract the interceptor speaks.
- [`midojo-sdk`](../../interception/python-sdk): the driver's end of the link (`midojo_sdk.binding`), the control plane client, and the agent-mode interception layers.
- [MiDojo](https://github.com/asago-ai/midojo#run-the-interceptor-as-a-separate-service): how a driver binds an evaluation to the service.
