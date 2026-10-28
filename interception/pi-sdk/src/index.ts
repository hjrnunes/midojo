import { createHash, randomUUID } from "node:crypto";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { TSchema } from "typebox";

// --- The seams: recording calls, and the simulated world ---
//
// A `Recorder` records the calls the extension intercepts, each with what the
// extension returned to the agent and what it did with it (its `Disposition`). A
// `WorldState` reads and changes the simulated environment that the
// extension's own tools act on. Both are bound to one evaluation by whoever
// builds them, as in the Python SDK (`midojo_sdk.recorder`). `ControlPlaneClient`
// implements both over the control plane's /agent API. Nothing here is a
// transaction across them: a record and an environment change commit, or fail,
// on their own.

/**
 * What the extension did with a call:
 *
 * - `forwarded`: the real tool ran, and the extension returned its answer (rewritten by a hook, or not)
 * - `simulated`: answered by a tool the extension registers, instead of a real one
 * - `denied`: refused by a check on the call's arguments; the tool never ran
 * - `blocked`: refused because the tool is blocked for the evaluation; the tool never ran
 */
export type Disposition = "forwarded" | "simulated" | "denied" | "blocked";

/** The dispositions of a call whose tool never ran: the record's `blocked` flag. */
export const REFUSED: ReadonlySet<Disposition> = new Set<Disposition>(["denied", "blocked"]);

/** Why the control plane refused an event: the `kind` of its response. */
export const REJECTION_KINDS = [
	"unknown_run",
	"unknown_evaluation",
	"unknown_task",
	"idempotency_conflict",
	"control_token_required",
	"invalid_session",
	"session_closed",
	"forbidden",
	"closing",
	"sealed",
	"closure_pending",
	"not_sealed",
	"call_conflict",
	"invalid_disposition",
	"turn_pending",
	"turn_not_started",
	"turn_answered",
	"revision_conflict",
	"invalid_environment",
	"unsupported_value",
] as const;

export type RejectionKind = (typeof REJECTION_KINDS)[number];

/** A call's start: an unfinished record, sent before the tool runs. */
export interface CallStart {
	callId: string;
	function: string;
	args: Record<string, unknown>;
	disposition: Disposition;
}

/**
 * A call's finish, or the whole record of a call that wasn't started: the
 * result's text only. Sent with `finishCall`, a plain finish, it carries no
 * injection outcomes, so the control plane keeps the record with
 * `injection_outcomes` null (not reported), has no delivery fact for a payload
 * in the result and reports its exposure as unknown. A `DetailedCallFinish`
 * carries the rest. The result is recorded before the agent gets it, so a
 * record doesn't show the agent received it.
 */
export interface CallFinish extends CallStart {
	result: string;
	error?: string | null;
}

/**
 * How one injection instruction fared on one call: the values of the Python
 * SDK's `OutcomeStatus`. `partial` is a structured-content placement the
 * result's text doesn't carry. `unmatched` is also a call to the instruction's
 * tool whose arguments didn't match its `call_arguments`, or whose result
 * didn't match its `result_record`: occurrence 0.
 */
export type OutcomeStatus =
	| "applied"
	| "partial"
	| "fallback"
	| "unmatched"
	| "no_result"
	| "too_large"
	| "skipped_occurrence";

/** The transformation applied to a payload: the values of the Python SDK's `Transformation`. */
export type Transformation = "identity" | "json_string";

/** A part of the result returned to the agent: its text (content) blocks or its structured content. */
export type ResultPart = "content_block" | "structured_content";

/** How an instruction writes its payload: the values of the Python SDK's `InjectionMode`. */
export type InjectionMode = "append" | "replace";

/** The outcomes that wrote the payload into the result: only these say what was placed and where. */
const PLACED: ReadonlySet<OutcomeStatus> = new Set<OutcomeStatus>(["applied", "partial", "fallback"]);

/** A payload digest: `sha256:` and the SHA-256 of the text's UTF-8 bytes in lowercase hex. */
const DIGEST = /^sha256:[0-9a-f]{64}$/;

/**
 * Where an instruction placed its payload: the Python SDK's
 * `InjectionSlot.to_data()`, sent as given. `target` is the instruction's
 * tool-output target in the plan codec's form (`InjectionPlan.to_data`'s
 * instruction `target`), `mode` its mode, and `id` `sha256:<hex>` of the
 * target's JSON with sorted keys, compact, UTF-8 (`injectionSlotId`); left
 * out, the control plane fills it in, and one given must be the target's.
 */
export interface InjectionSlot {
	id?: string;
	target: Record<string, unknown>;
	mode: InjectionMode;
}

/** `sha256:<hex>` of `text`'s UTF-8 bytes: the Python SDK's `payload_digest`. Throws a `RangeError` for a lone surrogate, which has no UTF-8. */
export function payloadDigest(text: string): string {
	return `sha256:${createHash("sha256").update(utf8(text)).digest("hex")}`;
}

/**
 * The ID of a slot's `target`: `sha256:<hex>` of the target in the plan codec's
 * form, as the Python SDK's `InjectionSlot.id` computes it (its JSON with keys
 * sorted by code point, compact, non-ASCII unescaped, floats as Python writes
 * them, as UTF-8). The target is read as the JSON it is sent as, and throws a
 * `RangeError` when it isn't a tool-output target the Python SDK accepts.
 */
export function injectionSlotId(target: unknown): string {
	return slotIdOf(canonicalTarget(asJson(target, "a slot's target")));
}

function slotIdOf(target: Record<string, unknown>): string {
	return payloadDigest(pythonJson(target));
}

const SLOT_FIELDS = new Set(["id", "target", "mode"]);
const TARGET_FIELDS = new Set([
	"tool",
	"surface",
	"content_block_index",
	"content_block_type",
	"structured_content_path",
	"template_path",
	"template_key",
	"call_arguments",
	"result_record",
]);
const TOOL_OUTPUT_SURFACES = new Set(["content_block", "structured_content"]);
const MODES = new Set(["append", "replace"]);

/**
 * Throw a `RangeError` for a slot the Python SDK (`InjectionSlot.from_data`) and
 * the control plane refuse: not `{target, mode}` with an optional `id`; a target
 * that isn't a tool-output target, by the same field rules; a mode other than
 * `append` or `replace`; or an `id` that isn't its target's. It reads the slot as
 * the JSON it is sent as (a field left undefined is left out).
 */
function checkSlot(slot: unknown): void {
	const data = asJson(slot, "a slot");
	if (!isObject(data)) throw new RangeError("a slot must be a JSON object");
	onlyFields(data, SLOT_FIELDS, "slot");
	if (!("target" in data) || !("mode" in data)) throw new RangeError("a slot must contain target and mode");
	const target = canonicalTarget(data.target);
	if (typeof data.mode !== "string" || !MODES.has(data.mode)) {
		throw new RangeError(`a slot's mode is append or replace, not ${JSON.stringify(data.mode)}`);
	}
	// Computed even when the id is left out, as the control plane fills it in: a target with no UTF-8 has none.
	const id = slotIdOf(target);
	if ("id" in data && data.id !== id) {
		throw new RangeError(`a slot's id ${JSON.stringify(data.id)} is not its target's ${id}`);
	}
}

/** `value` as the JSON it is sent as. */
function asJson(value: unknown, what: string): unknown {
	try {
		const text = JSON.stringify(value);
		if (text === undefined) throw new RangeError(`${what} is not JSON`);
		return JSON.parse(text);
	} catch (error) {
		throw error instanceof RangeError ? error : new RangeError(`${what} is not JSON: ${String(error)}`);
	}
}

function isObject(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

/** Whether a JSON number is one Python reads as an `int`: written without a fraction or an exponent. */
function isPythonInt(value: unknown): value is number {
	return typeof value === "number" && /^-?\d+$/.test(JSON.stringify(value));
}

function onlyFields(value: Record<string, unknown>, allowed: ReadonlySet<string>, what: string): void {
	const unknown = Object.keys(value).filter((key) => !allowed.has(key));
	if (unknown.length > 0) throw new RangeError(`${what} has unknown fields: ${JSON.stringify(unknown.sort())}`);
}

function optionalString(value: unknown, what: string): string | null {
	if (value !== null && typeof value !== "string") throw new RangeError(`${what} must be a string or null`);
	return value;
}

/**
 * A slot's target in the plan codec's form, as the Python SDK decodes and
 * encodes it again; a `RangeError` for any target it refuses as a tool-output
 * target. Absent fields take their defaults, a record selector its `field`
 * (`id`), and `call_arguments` and `result_record` are left out when empty.
 */
function canonicalTarget(value: unknown): Record<string, unknown> {
	if (!isObject(value)) throw new RangeError("a slot's target must be a JSON object");
	onlyFields(value, TARGET_FIELDS, "target");
	const tool = optionalString(value.tool ?? null, "target tool");
	const surface = "surface" in value ? value.surface : "content_block";
	if (typeof surface !== "string" || !TOOL_OUTPUT_SURFACES.has(surface)) {
		throw new RangeError(`tool_output needs a content_block or structured_content selector, not ${JSON.stringify(surface)}`);
	}
	const index = value.content_block_index ?? null;
	if (index !== null && !(isPythonInt(index) && index >= 0)) {
		throw new RangeError("content_block_index must be a non-negative int");
	}
	const blockType = optionalString(value.content_block_type ?? null, "content block type");
	// A path left out is empty; null isn't a path.
	const structuredPath = path(
		"structured_content_path" in value ? value.structured_content_path : [],
		"structured content path",
		true,
	);
	const templatePath = path("template_path" in value ? value.template_path : [], "template path", false);
	const templateKey = optionalString(value.template_key ?? null, "template key");
	const callArguments = "call_arguments" in value ? argumentsOf(value.call_arguments) : {};
	const resultRecord = "result_record" in value ? selector(value.result_record, "result_record") : null;
	if ((Object.keys(callArguments).length > 0 || resultRecord !== null) && tool === null) {
		throw new RangeError("call_arguments and result_record need a target tool");
	}
	const template = templatePath.length > 0 || templateKey !== null;
	if (surface === "content_block" && (structuredPath.length > 0 || template)) {
		throw new RangeError("content_block selectors cannot include structured-content or template fields");
	}
	if (surface === "structured_content") {
		if (index !== null || blockType !== null || template) {
			throw new RangeError("structured_content selectors cannot include content-block or template fields");
		}
		if (structuredPath.length === 0) throw new RangeError("structured_content needs a non-empty field path");
	}
	const target: Record<string, unknown> = {
		tool,
		surface,
		content_block_index: index,
		content_block_type: blockType,
		structured_content_path: structuredPath,
		template_path: templatePath,
		template_key: templateKey,
	};
	if (Object.keys(callArguments).length > 0) target.call_arguments = callArguments;
	if (resultRecord !== null) target.result_record = resultRecord;
	return target;
}

function path(value: unknown, what: string, recordSelectors: boolean): unknown[] {
	if (!Array.isArray(value)) throw new RangeError(`${what} must be a JSON array`);
	return value.map((part) => {
		if (isObject(part)) {
			if (!recordSelectors) throw new RangeError(`${what} cannot contain a record selector`);
			return selector(part, "record selector");
		}
		if (typeof part === "string" || (isPythonInt(part) && part >= 0)) return part;
		throw new RangeError(`${what} parts must be strings, non-negative ints${recordSelectors ? " or record selectors" : ""}`);
	});
}

function selector(value: unknown, what: string): { field: string; equals: string | number } {
	if (!isObject(value)) throw new RangeError(`${what} must be a JSON object`);
	onlyFields(value, new Set(["field", "equals"]), what);
	if (!("equals" in value)) throw new RangeError(`${what} must contain equals`);
	const field = "field" in value ? value.field : "id";
	if (typeof field !== "string" || field === "") throw new RangeError(`${what}'s field must be a non-empty string`);
	const equals = value.equals;
	if (typeof equals !== "string" && !isPythonInt(equals)) throw new RangeError(`${what}'s value must be a string or an int`);
	return { field, equals };
}

function argumentsOf(value: unknown): Record<string, unknown> {
	if (!isObject(value)) throw new RangeError("call_arguments must be a JSON object");
	for (const [name, argument] of Object.entries(value)) {
		if (name === "") throw new RangeError("call argument names must be non-empty");
		if (argument !== null && !["string", "number", "boolean"].includes(typeof argument)) {
			throw new RangeError(`call argument ${JSON.stringify(name)} must be a string, number, boolean or null`);
		}
	}
	return value;
}

/** `value` (JSON data) as Python's `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)` writes it. */
function pythonJson(value: unknown): string {
	if (value === null || typeof value === "boolean") return JSON.stringify(value);
	if (typeof value === "number") return pythonNumber(value);
	if (typeof value === "string") return pythonString(value);
	if (Array.isArray(value)) return `[${value.map(pythonJson).join(",")}]`;
	const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => byCodePoint(a, b));
	return `{${entries.map(([key, item]) => `${pythonString(key)}:${pythonJson(item)}`).join(",")}}`;
}

/** A string as Python's JSON writes it without ASCII escapes: the same escapes as JSON.stringify, with no lone surrogate to escape. */
function pythonString(text: string): string {
	utf8(text);
	return JSON.stringify(text);
}

/**
 * A number as Python writes the value it reads from this number's JSON: an int
 * as its digits, a float as `repr` does (the shortest digits that read back,
 * fixed for exponents -4 to 15, else scientific with a signed exponent of at
 * least two digits).
 */
function pythonNumber(value: number): string {
	const text = JSON.stringify(value);
	if (!/[.e]/.test(text)) return text;
	const [mantissa, power] = value.toExponential().split("e");
	const exponent = Number(power);
	const sign = mantissa.startsWith("-") ? "-" : "";
	const digits = mantissa.replace(/[-.]/g, "");
	if (exponent >= -4 && exponent < 16) {
		if (exponent < 0) return `${sign}0.${"0".repeat(-exponent - 1)}${digits}`;
		const whole = digits.slice(0, exponent + 1).padEnd(exponent + 1, "0");
		return `${sign}${whole}.${digits.slice(exponent + 1) || "0"}`;
	}
	const fraction = digits.length > 1 ? `.${digits.slice(1)}` : "";
	return `${sign}${digits[0]}${fraction}e${exponent < 0 ? "-" : "+"}${String(Math.abs(exponent)).padStart(2, "0")}`;
}

/** Order two strings by code point, as Python sorts them (not by UTF-16 code unit). */
function byCodePoint(a: string, b: string): number {
	const x = Array.from(a);
	const y = Array.from(b);
	for (let i = 0; i < Math.min(x.length, y.length); i++) {
		const difference = (x[i].codePointAt(0) ?? 0) - (y[i].codePointAt(0) ?? 0);
		if (difference !== 0) return difference;
	}
	return x.length - y.length;
}

/** `text`'s UTF-8 bytes; a `RangeError` for a lone surrogate, which Python can't encode either. */
function utf8(text: string): Buffer {
	for (const character of text) {
		const point = character.codePointAt(0) ?? 0;
		if (point >= 0xd800 && point <= 0xdfff) throw new RangeError("a string with a lone surrogate has no UTF-8");
	}
	return Buffer.from(text, "utf8");
}

/**
 * How one injection instruction fared on a call, as the call's detailed finish
 * carries it (the Python SDK's `InjectionRecord`). `instructionIndex` names the
 * instruction by its index in the plan the interceptor was given; a record
 * names neither the payload nor a counter scope. `occurrence` is one-based, or
 * 0 for a call that isn't counted (its arguments didn't match the
 * instruction's `call_arguments`, or its result its `result_record`): only an
 * `unmatched` record that placed nothing has occurrence 0, which
 * `checkInjectionRecord` enforces.
 *
 * The delivery facts say what the interceptor did at its own boundary:
 * `selected`, whether the instruction's exact selector matched a place in the
 * result; `transformation`, the transformation applied to the payload (null
 * when it was never transformed); `emitted`, the parts of the returned result
 * that carry the transformed payload (empty when none does). A layer that
 * doesn't report them leaves them out, and they're sent only when given. They
 * never say what the agent saw: a result is recorded before it is returned.
 *
 * A placement (`applied`, `partial`, `fallback`) may also say what it placed and
 * where, sent only when given: `payloadDigest`, `sha256:<hex>` of the UTF-8
 * bytes of the instruction's payload before its transformation;
 * `placedDigest`, the same of the transformed text it wrote; and `slot`, the
 * instruction's target and mode (`InjectionSlot`). Any other record has none.
 */
export interface InjectionRecord {
	instructionIndex: number;
	probeKey: string | null;
	occurrence: number;
	status: OutcomeStatus;
	detail: string;
	selected?: boolean;
	transformation?: Transformation | null;
	emitted?: ResultPart[];
	payloadDigest?: string;
	placedDigest?: string;
	slot?: InjectionSlot;
}

/**
 * A call's finish with the parts of its result beyond the text: the structured
 * content returned to the agent with `result` (null when it had none) and how
 * each injection instruction that matched the call fared. The list is always
 * sent, empty when none matched, so the record says the details were reported.
 */
export interface DetailedCallFinish extends CallFinish {
	structuredContent: Record<string, unknown> | null;
	injectionOutcomes: InjectionRecord[];
}

/**
 * Records one evaluation's tool calls: each call's start, then its finish with
 * the same `callId`. Each method sends its event once, and throws a
 * `ControlPlaneRejectedError` when the control plane refuses it. Wrap a
 * recorder in `RetryingRecorder` to resend after a transient failure.
 */
export interface Recorder {
	/** Record that a call started, with its disposition (`forwarded` or `simulated`), before the tool runs. */
	startCall(call: CallStart): Promise<void>;
	/** Finish the call started with `callId`, or record it whole when it wasn't started. */
	finishCall(call: CallFinish): Promise<void>;
}

/**
 * A `Recorder` that also records the parts of a result beyond its text: an
 * optional capability, which `recordsDetails` says a recorder has.
 * `ControlPlaneClient` has it, and `RetryingRecorder` has the capability of the
 * recorder it wraps. A recorder without it still records every call, with
 * `finishCall` (`finishDetailed` falls back to it): the text only, kept with
 * `injection_outcomes` null, which says the details weren't reported, not that
 * there were none.
 */
export interface DetailedRecorder extends Recorder {
	/** Finish the call as `finishCall` does, with its structured content and injection outcomes too. */
	finishCallDetailed(call: DetailedCallFinish): Promise<void>;
}

/**
 * Whether `recorder` is a `DetailedRecorder`. A wrapping recorder says so with
 * its own boolean `recordsDetails`, taken from the recorder it wraps; any other
 * recorder is one when it has a `finishCallDetailed` method.
 */
export function recordsDetails(recorder: unknown): recorder is DetailedRecorder {
	if (recorder === null || typeof recorder !== "object") return false;
	const { recordsDetails: declared, finishCallDetailed } = recorder as {
		recordsDetails?: unknown;
		finishCallDetailed?: unknown;
	};
	if (typeof declared === "boolean") return declared;
	return typeof finishCallDetailed === "function";
}

/**
 * Throw a `RangeError` for an injection record the control plane would refuse
 * at occurrence 0: one that isn't `unmatched`, or that reports a delivery fact
 * (`selected` true, a `transformation`, a non-empty `emitted`) or a placement
 * (a digest or a slot). Unreported facts are allowed, and so is `unmatched` at
 * occurrence 1 or more (a strict selector that didn't match a counted call).
 * It also throws for a digest or slot on any record but a placement, for a
 * digest not in the `sha256:<hex>` form, and for a slot the Python SDK and the
 * control plane refuse (`checkSlot`: its shape, a target that isn't a
 * tool-output target, its mode, or an `id` that isn't its target's).
 */
export function checkInjectionRecord(record: InjectionRecord): void {
	const placement = record.payloadDigest !== undefined || record.placedDigest !== undefined || record.slot !== undefined;
	if (
		record.occurrence === 0 &&
		(record.status !== "unmatched" ||
			record.selected === true ||
			(record.transformation !== undefined && record.transformation !== null) ||
			(record.emitted !== undefined && record.emitted.length > 0) ||
			placement)
	) {
		throw new RangeError("only an unmatched outcome that placed nothing has occurrence 0");
	}
	if (placement && !PLACED.has(record.status)) {
		throw new RangeError(`a ${record.status} outcome placed nothing: it has no payload digest or slot`);
	}
	for (const digest of [record.payloadDigest, record.placedDigest]) {
		if (digest !== undefined && !DIGEST.test(digest)) {
			throw new RangeError(`a payload digest is 'sha256:' and 64 lowercase hex digits, not ${JSON.stringify(digest)}`);
		}
	}
	if (record.slot !== undefined) checkSlot(record.slot);
}

/**
 * Finish a call with its details when `recorder` records them, else with
 * `finishCall`: the text only. Resolves to whether the details were recorded.
 * The fallback never fails the finish because of the details: a recorder
 * written before them records the call as it always did. Either way, an
 * invalid injection record (`checkInjectionRecord`) throws before anything is
 * sent.
 */
export async function finishDetailed(recorder: Recorder, call: DetailedCallFinish): Promise<boolean> {
	call.injectionOutcomes.forEach(checkInjectionRecord);
	if (recordsDetails(recorder)) {
		await recorder.finishCallDetailed(call);
		return true;
	}
	const { structuredContent: _structured, injectionOutcomes: _outcomes, ...plain } = call;
	await recorder.finishCall(plain);
	return false;
}

/** The simulated environment's fields, as of one revision. Every change, through any caller, gives a new revision. */
export interface EnvironmentRevision {
	revision: number;
	environment: Record<string, unknown>;
}

/**
 * Reads and changes one evaluation's simulated environment. A change is atomic
 * on its own: no other change interleaves with it.
 */
export interface WorldState {
	/** The current environment and its revision. */
	read(): Promise<EnvironmentRevision>;
	/**
	 * Set the top-level fields in `values` in one atomic step, and return the
	 * environment it made. With `expectedRevision`, the change applies only at
	 * that revision: at any other it changes nothing and returns `null`, so a
	 * caller that read the environment, computed new values and names the
	 * revision it read loses no concurrent change. Throws a
	 * `ControlPlaneRejectedError` with `invalid_environment` when the changed
	 * environment isn't valid for the evaluation's suite.
	 */
	change(values: Record<string, unknown>, options?: { expectedRevision?: number }): Promise<EnvironmentRevision | null>;
}

/** What a tool registered with `tools`, or a hook, gets to read and change the evaluation's environment. */
export interface ToolContext {
	/** The current value of one field of the environment. */
	env<T = unknown>(field: string): Promise<T>;
	/** The whole environment and its revision, for a later `envUpdate` or `change` at that revision. */
	read(): Promise<EnvironmentRevision>;
	/**
	 * Set one field in one atomic step: no concurrent change to another field
	 * is lost. Without `expectedRevision` the change always applies, so a value
	 * computed from an earlier read can overwrite a concurrent change to the
	 * same field. With the revision `read()` returned, it applies only if
	 * nothing changed the environment since, and resolves to `false` otherwise
	 * (read again and recompute).
	 */
	envUpdate(field: string, value: unknown, options?: { expectedRevision?: number }): Promise<boolean>;
	/** Set several fields in one atomic step: all of them change, or none. `expectedRevision` works as in `envUpdate`. */
	change(values: Record<string, unknown>, options?: { expectedRevision?: number }): Promise<boolean>;
}

/**
 * What a registered tool or a hook may return instead of a string, to give the
 * agent structured content too, or to report what it did with an injection.
 *
 * The extension can't see what a tool's or a hook's code placed in its result,
 * so only the code can report it. With `injectionOutcomes` the call is finished
 * with its details when the recorder records them (`finishDetailed`): the
 * structured content returned to the agent and these outcomes, an empty list
 * saying no instruction matched the call. Without it, as for a plain string,
 * the call is finished with its text only, which the control plane keeps with
 * `injection_outcomes` null: not reported, so a payload in the result has
 * unknown exposure rather than a claim that nothing was placed.
 */
export interface MidojoToolOutput {
	/** The result's text: the one text content block the agent gets. */
	result: string;
	/**
	 * Structured content returned with the text. Null or left out: none, and PI
	 * drops any the real tool returned, since the hook replaced its content.
	 */
	structuredContent?: Record<string, unknown> | null;
	/** How each injection instruction that matched the call fared (see `InjectionRecord`); left out: not reported. */
	injectionOutcomes?: InjectionRecord[];
}

export interface MidojoToolDef {
	name: string;
	label: string;
	description: string;
	parameters: TSchema;
	execute: (params: Record<string, unknown>, ctx: ToolContext) => Promise<string | MidojoToolOutput>;
}

export interface MidojoToolHook {
	toolName: string;
	/**
	 * Rewrite a real tool's result: `realResult` is its text content joined by
	 * newlines, and `realStructuredContent` the structured content it returned
	 * (undefined when none), which the agent loses unless the hook returns it.
	 */
	execute: (
		args: Record<string, unknown>,
		realResult: string,
		ctx: ToolContext,
		realStructuredContent?: unknown,
	) => Promise<string | MidojoToolOutput>;
}

export interface MidojoExtensionConfig {
	/**
	 * The control plane the calls are recorded on, and whose environment tools
	 * and hooks get, for the evaluation session in `MIDOJO_SESSION_TOKEN`.
	 * Defaults to `MIDOJO_URL`. Not needed when both `recorder` and `world` are given.
	 */
	controlPlaneUrl?: string;
	/**
	 * Records every intercepted call. Defaults to the control plane's, wrapped
	 * in a `RetryingRecorder` that resends after a transient failure.
	 */
	recorder?: Recorder;
	/** The environment tools and hooks get through their `ToolContext`. Defaults to the control plane's. */
	world?: WorldState;
	tools?: MidojoToolDef[];
	hooks?: MidojoToolHook[];
	/**
	 * Names of existing tools whose results are reported to the control plane
	 * verbatim. Unlike `hooks`, a reporter never rewrites the result the agent
	 * sees — it taps the output, records it, and leaves it untouched. Use this to
	 * record a tool's result (e.g. a file `read`) in the evaluation's tool calls
	 * without perturbing the agent. Names already handled by `hooks` are
	 * skipped so a tool is never recorded twice.
	 */
	reportTools?: string[];
	/**
	 * Names of tools that never run. When the agent calls one, the call is
	 * recorded to the control plane as blocked and the agent gets an error
	 * instead of a result. Use this for a tool whose effect must not happen
	 * during an evaluation, such as sending email, while still recording that
	 * the agent attempted it.
	 */
	blockTools?: string[];
}

/** What the agent is told when it calls a tool in `blockTools`: PI's own message for a blocked call. */
export const BLOCKED_REASON = "Tool execution was blocked";

/** The longest `call_id` the control plane accepts. */
export const CALL_ID_MAX_LENGTH = 256;

/** The response header that carries the environment's revision. */
export const REVISION_HEADER = "X-Midojo-Environment-Revision";

/** How many times `RetryingRecorder` (and `retry`) sends an event before its error is thrown. */
export const RECORD_ATTEMPTS = 3;

/** The wait before the first resend of a record, doubled before each later one. */
export const RECORD_BACKOFF_MS = 100;

// --- Errors ---

/** A failed control plane request. `status` is undefined when no response arrived. */
export class ControlPlaneError extends Error {
	status: number | undefined;

	constructor(message: string, status?: number, options?: ErrorOptions) {
		super(message, options);
		this.name = "ControlPlaneError";
		this.status = status;
	}

	/** A lost request or a server error may succeed when resent; a 4xx won't. */
	get retryable(): boolean {
		return this.status === undefined || this.status >= 500;
	}
}

/**
 * The control plane refused an event: a record, or a read or change of the
 * environment. `kind` says why. `lateFact` is true when the control plane kept
 * the refused write as a late fact: the evaluation was closing or sealed, and
 * its evidence is now marked incomplete. A refusal is never resent: sending the
 * same event again gets the same answer.
 */
export class ControlPlaneRejectedError extends ControlPlaneError {
	kind: RejectionKind;
	detail: string;
	lateFact: boolean;

	constructor(kind: RejectionKind, detail: string, options: { lateFact?: boolean; status?: number } = {}) {
		super(`${kind}: ${detail}`, options.status);
		this.name = "ControlPlaneRejectedError";
		this.kind = kind;
		this.detail = detail;
		this.lateFact = options.lateFact ?? false;
	}

	get retryable(): boolean {
		return false;
	}
}

/**
 * The control plane can't take the request as sent: a field is missing, too
 * long, or of the wrong type (the API's 422 without a `kind`). Nothing was
 * decided about the evaluation. Resending it gets the same answer.
 */
export class InvalidRequestError extends ControlPlaneError {
	detail: string;

	constructor(detail: string) {
		super(`invalid request: ${detail}`, 422);
		this.name = "InvalidRequestError";
		this.detail = detail;
	}

	get retryable(): boolean {
		return false;
	}
}

/** Whether resending a request that failed with `error` may succeed: a lost request or a 5xx, never a refusal. */
export function transient(error: unknown): boolean {
	return error instanceof ControlPlaneError && error.retryable;
}

/** Whether no resend can change `error`'s answer: a refusal, or an invalid request. */
function final(error: unknown): boolean {
	return error instanceof ControlPlaneRejectedError || error instanceof InvalidRequestError;
}

export interface RetryOptions {
	/** Which errors are transient: `transient` for the control plane's client. */
	retryable: (error: unknown) => boolean;
	/** How many sends in all; `RECORD_ATTEMPTS` by default. */
	attempts?: number;
	/** The wait before the first resend, doubled before each later one; `RECORD_BACKOFF_MS` by default. */
	backoffMs?: number;
}

/**
 * Call `send` until it resolves, resending after an error `retryable` accepts.
 * Any other error, and the last attempt's, is thrown at once, and so is a
 * refusal or an invalid request, whatever `retryable` says.
 */
export async function retry<T>(send: () => Promise<T>, options: RetryOptions): Promise<T> {
	const attempts = options.attempts ?? RECORD_ATTEMPTS;
	const backoffMs = options.backoffMs ?? RECORD_BACKOFF_MS;
	for (let attempt = 1; ; attempt++) {
		try {
			return await send();
		} catch (e) {
			if (final(e) || attempt >= attempts || !options.retryable(e)) throw e;
		}
		await new Promise((resolve) => setTimeout(resolve, backoffMs * 2 ** (attempt - 1)));
	}
}

/**
 * A `Recorder` that resends each start and finish after a transient failure.
 * Resends keep the `callId`, and the control plane keeps at most one record per
 * `call_id`, so a resent event that had already arrived isn't recorded twice.
 * This is at most one record per call, not exactly once: a finish whose sends
 * all fail leaves its call unfinished, and a call whose every send fails is
 * missing. A refusal is never resent.
 */
export class RetryingRecorder implements Recorder {
	recorder: Recorder;
	private options: RetryOptions;

	constructor(recorder: Recorder, options: RetryOptions) {
		this.recorder = recorder;
		this.options = options;
	}

	async startCall(call: CallStart): Promise<void> {
		await retry(() => this.recorder.startCall(call), this.options);
	}

	async finishCall(call: CallFinish): Promise<void> {
		await retry(() => this.recorder.finishCall(call), this.options);
	}

	/** Whether the wrapped recorder records a finish's details (see `DetailedRecorder`). */
	get recordsDetails(): boolean {
		return recordsDetails(this.recorder);
	}

	/** Finish the call through the wrapped recorder, with its details when that recorder records them, resent like `finishCall`. */
	async finishCallDetailed(call: DetailedCallFinish): Promise<void> {
		await retry(() => finishDetailed(this.recorder, call), this.options);
	}
}

/**
 * Picks the `call_id` of each record one extension sends. PI's tool call ID
 * becomes `<prefix>:<extension UUID>:<tool call ID>`, so it can't collide with
 * another extension's records in the same evaluation. A missing tool call ID,
 * one this extension has already used (say, a tool that is both registered and
 * hooked), or one too long for the control plane gets a fresh UUID instead.
 */
export class CallIds {
	private namespace: string;
	private used = new Set<string>();

	constructor(prefix: string) {
		this.namespace = `${prefix}:${randomUUID()}`;
	}

	forCall(harnessId: string | undefined): string {
		if (harnessId) {
			const callId = `${this.namespace}:${harnessId}`;
			if (callId.length <= CALL_ID_MAX_LENGTH && !this.used.has(callId)) {
				this.used.add(callId);
				return callId;
			}
		}
		return randomUUID();
	}
}

/** A `ToolContext` over `world`; without one, reading or changing the environment throws. */
export function createToolContext(world: WorldState | undefined): ToolContext {
	const required = (): WorldState => {
		if (world === undefined) throw new Error("This tool has no environment: no WorldState is configured.");
		return world;
	};
	const change = async (values: Record<string, unknown>, options?: { expectedRevision?: number }) =>
		(await required().change(values, options)) !== null;
	return {
		env: async <T = unknown>(field: string): Promise<T> => (await required().read()).environment[field] as T,
		read: () => required().read(),
		envUpdate: (field, value, options) => change({ [field]: value }, options),
		change,
	};
}

/**
 * The control plane's /agent API: the `Recorder` and `WorldState` of the
 * evaluation session, over HTTP. The session is `sessionToken`, or else
 * `MIDOJO_SESSION_TOKEN`, read on every request.
 *
 * `startCall`, `finishCall`, `finishCallDetailed`, `read` and `change` send
 * each request once. A response that carries one of the control plane's
 * rejection kinds throws a `ControlPlaneRejectedError`; a 422 without one, an
 * `InvalidRequestError`; any other failure (a lost request, a proxy's error
 * page, a 401 without a session) a `ControlPlaneError`, which `transient`
 * classifies for a `RetryingRecorder`.
 */
export class ControlPlaneClient implements DetailedRecorder, WorldState {
	private baseUrl: string;
	private sessionToken: string | undefined;

	constructor(baseUrl: string | undefined = process.env.MIDOJO_URL, options: { sessionToken?: string } = {}) {
		if (!baseUrl) throw new Error("MiDojo control plane URL is required. Set MIDOJO_URL or configure controlPlaneUrl.");
		let end = baseUrl.length;
		while (end > 0 && baseUrl[end - 1] === "/") end--;
		this.baseUrl = `${baseUrl.slice(0, end)}/agent`;
		this.sessionToken = options.sessionToken;
	}

	private getSessionToken(): string {
		const token = this.sessionToken ?? process.env.MIDOJO_SESSION_TOKEN;
		if (!token) throw new Error("No MiDojo evaluation session. Set MIDOJO_SESSION_TOKEN.");
		return token;
	}

	private async request(path: string, method: string = "GET", body?: unknown): Promise<Response> {
		const init = {
			method,
			headers: {
				"Content-Type": "application/json",
				"Authorization": `Bearer ${this.getSessionToken()}`,
			},
			body: body === undefined ? undefined : JSON.stringify(body),
		};
		let resp: Response;
		try {
			resp = await fetch(`${this.baseUrl}${path}`, init);
		} catch (e) {
			const reason = e instanceof Error ? e.message : String(e);
			throw new ControlPlaneError(`MiDojo ${method} ${path} failed: ${reason}`, undefined, { cause: e });
		}
		if (!resp.ok) throw await failure(resp, `MiDojo ${method} ${path} failed (${resp.status})`);
		return resp;
	}

	// --- WorldState ---

	async read(): Promise<EnvironmentRevision> {
		return revisionOf(await this.request("/environment"));
	}

	async change(
		values: Record<string, unknown>,
		options: { expectedRevision?: number } = {},
	): Promise<EnvironmentRevision | null> {
		const body = { set: values, expected_revision: options.expectedRevision ?? null };
		try {
			return revisionOf(await this.request("/environment", "PATCH", body));
		} catch (e) {
			if (e instanceof ControlPlaneRejectedError && e.kind === "revision_conflict" && options.expectedRevision !== undefined) {
				return null;
			}
			throw e;
		}
	}

	/** The environment's fields, without its revision. */
	async getEnvironment(): Promise<Record<string, unknown>> {
		return (await this.read()).environment;
	}

	/**
	 * Replace the whole environment, whatever changed it since it was read.
	 *
	 * @deprecated Use `change`: a read followed by a replace loses any change made in between.
	 */
	async putEnvironment(env: Record<string, unknown>): Promise<void> {
		await this.request("/environment", "PUT", env);
	}

	// --- Recorder ---

	/**
	 * Send a call's start once, with its disposition (`forwarded` or
	 * `simulated`). A denied or blocked call never runs, so it's never started:
	 * the control plane refuses its start as `invalid_disposition`.
	 */
	async startCall(call: CallStart): Promise<void> {
		await this.request("/function-calls/start", "POST", startBody(call));
	}

	/**
	 * Send a call's finish once, with its disposition. The body also carries the
	 * `blocked` flag the disposition implies, so a control plane that predates
	 * the disposition field still records a refused call as blocked.
	 */
	async finishCall(call: CallFinish): Promise<void> {
		await this.request("/function-calls", "POST", finishBody(call));
	}

	/**
	 * Send a call's finish once, as `finishCall` does, with its structured
	 * content and injection outcomes. `injection_outcomes` is always sent, empty
	 * when no instruction matched the call, so the record says the details were
	 * reported. An invalid injection record (`checkInjectionRecord`) throws
	 * before anything is sent.
	 */
	async finishCallDetailed(call: DetailedCallFinish): Promise<void> {
		call.injectionOutcomes.forEach(checkInjectionRecord);
		await this.request("/function-calls", "POST", {
			...finishBody(call),
			structured_content: call.structuredContent,
			injection_outcomes: call.injectionOutcomes.map(injectionRecordBody),
		});
	}

	// --- the earlier API, with resends ---

	/**
	 * Record a call's start: an unfinished record that `recordFunctionCall`
	 * finishes. Resent like `recordFunctionCall`; a resent start doesn't record
	 * the call twice.
	 *
	 * @deprecated Use `startCall`, which carries the call's disposition, wrapped in a `RetryingRecorder` for resends.
	 */
	async startFunctionCall(entry: { function: string; args: Record<string, unknown>; callId: string }): Promise<void> {
		const body = { call_id: entry.callId, function: entry.function, args: entry.args };
		await retry(() => this.request("/function-calls/start", "POST", body), { retryable: transient });
	}

	/**
	 * Record a call, at most once per `callId` (a UUID when not given). When
	 * `callId` was started with `startFunctionCall`, this finishes that record.
	 * A lost request or a 5xx is resent with the same `call_id`, up to
	 * `RECORD_ATTEMPTS` sends in all. A 4xx, including a 409 for a `call_id`
	 * recorded with another body, is thrown at once.
	 *
	 * @deprecated Use `finishCall`, which carries the call's disposition, wrapped in a `RetryingRecorder` for resends.
	 */
	async recordFunctionCall(entry: {
		function: string;
		args: Record<string, unknown>;
		result: string;
		error?: string | null;
		blocked?: boolean;
		callId?: string;
	}): Promise<void> {
		const { callId, ...fields } = entry;
		const body = { call_id: callId ?? randomUUID(), ...fields };
		await retry(() => this.request("/function-calls", "POST", body), { retryable: transient });
	}

	/** A `ToolContext` over this client's environment. */
	createToolContext(): ToolContext {
		return createToolContext(this);
	}
}

function startBody(call: CallStart): Record<string, unknown> {
	return { call_id: call.callId, function: call.function, args: call.args, disposition: call.disposition };
}

/** A finish's body. It also carries the `blocked` flag the disposition implies. */
function finishBody(call: CallFinish): Record<string, unknown> {
	return { ...startBody(call), result: call.result, error: call.error ?? null, blocked: REFUSED.has(call.disposition) };
}

/** An injection record as the control plane takes it: unreported delivery facts and placements are left out. */
function injectionRecordBody(record: InjectionRecord): Record<string, unknown> {
	const body: Record<string, unknown> = {
		instruction_index: record.instructionIndex,
		probe_key: record.probeKey,
		occurrence: record.occurrence,
		status: record.status,
		detail: record.detail,
	};
	if (record.selected !== undefined) body.selected = record.selected;
	if (record.transformation !== undefined) body.transformation = record.transformation;
	if (record.emitted !== undefined) body.emitted = record.emitted;
	if (record.payloadDigest !== undefined) body.payload_digest = record.payloadDigest;
	if (record.placedDigest !== undefined) body.placed_digest = record.placedDigest;
	if (record.slot !== undefined) body.slot = record.slot;
	return body;
}

async function revisionOf(resp: Response): Promise<EnvironmentRevision> {
	const revision = Number(resp.headers.get(REVISION_HEADER) ?? Number.NaN);
	if (!Number.isInteger(revision)) throw new ControlPlaneError(`MiDojo response has no ${REVISION_HEADER}`, resp.status);
	return { revision, environment: (await resp.json()) as Record<string, unknown> };
}

/**
 * The error for a failed response: a `ControlPlaneRejectedError` when its body
 * names one of the control plane's rejection kinds, an `InvalidRequestError`
 * for a 422 whose `detail` is a list of errors, a `ControlPlaneError` otherwise.
 */
async function failure(resp: Response, message: string): Promise<ControlPlaneError> {
	let body: unknown;
	try {
		body = await resp.json();
	} catch {
		body = undefined;
	}
	if (body !== null && typeof body === "object") {
		const { kind, detail, late_fact } = body as { kind?: unknown; detail?: unknown; late_fact?: unknown };
		if (typeof kind === "string" && (REJECTION_KINDS as readonly string[]).includes(kind)) {
			const text = typeof detail === "string" ? detail : String(detail);
			return new ControlPlaneRejectedError(kind as RejectionKind, text, { lateFact: late_fact === true, status: resp.status });
		}
		if (resp.status === 422 && Array.isArray(detail)) return new InvalidRequestError(JSON.stringify(detail));
	}
	return new ControlPlaneError(message, resp.status);
}

/** A registered tool's or a hook's output, as the extension records it and returns it to the agent. */
interface ToolOutput {
	result: string;
	structuredContent: Record<string, unknown> | null;
	/** Undefined: not reported, so the call is finished with its text only. */
	injectionOutcomes: InjectionRecord[] | undefined;
}

function outputOf(returned: string | MidojoToolOutput): ToolOutput {
	if (typeof returned === "string") return { result: returned, structuredContent: null, injectionOutcomes: undefined };
	return {
		result: returned.result,
		structuredContent: returned.structuredContent ?? null,
		injectionOutcomes: returned.injectionOutcomes,
	};
}

/** What PI returns to the agent for `output`: its text, and its structured content when it has some. */
function agentResult(output: ToolOutput): {
	content: { type: "text"; text: string }[];
	structuredContent?: Record<string, unknown>;
} {
	const content = [{ type: "text" as const, text: output.result }];
	return output.structuredContent === null ? { content } : { content, structuredContent: output.structuredContent };
}

/**
 * Finish `call` with its details (`finishDetailed`, which falls back to the
 * text only for a recorder without them) when its injection outcomes were
 * reported, and with its text only (`finishCall`) when they weren't.
 */
async function finish(
	recorder: Recorder,
	call: CallFinish,
	details: Pick<ToolOutput, "structuredContent" | "injectionOutcomes">,
): Promise<void> {
	if (details.injectionOutcomes === undefined) {
		await recorder.finishCall(call);
		return;
	}
	await finishDetailed(recorder, {
		...call,
		structuredContent: details.structuredContent,
		injectionOutcomes: details.injectionOutcomes,
	});
}

function isJsonObject(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

export function createMidojoExtension(config: MidojoExtensionConfig): (pi: ExtensionAPI) => void {
	return (pi: ExtensionAPI) => {
		// Built once per extension: the control plane's client, the recorder
		// every call goes through, and the environment tools and hooks get.
		const client =
			config.recorder !== undefined && config.world !== undefined
				? undefined
				: new ControlPlaneClient(config.controlPlaneUrl);
		const recorder = config.recorder ?? new RetryingRecorder(client as ControlPlaneClient, { retryable: transient });
		const ctx = createToolContext(config.world ?? client);
		const callIds = new CallIds("pi");

		// Hooked and reported calls, started on tool_call and keyed by PI's tool
		// call ID, until tool_result finishes them with the same call_id and args.
		// A tool_result with no started call (one whose tool_call this extension
		// never saw) is recorded once, with a call_id of its own.
		const started = new Map<string, { callId: string; args: Record<string, unknown> }>();
		const finishing = (toolCallId: string | undefined, args: Record<string, unknown>) => {
			const call = toolCallId === undefined ? undefined : started.get(toolCallId);
			if (call === undefined) return { callId: callIds.forCall(toolCallId), args };
			started.delete(toolCallId as string);
			return call;
		};

		for (const toolDef of config.tools ?? []) {
			pi.registerTool({
				name: toolDef.name,
				label: toolDef.label,
				description: toolDef.description,
				parameters: toolDef.parameters,
				async execute(toolCallId, params) {
					const typedParams = params as Record<string, unknown>;
					const call = {
						callId: callIds.forCall(toolCallId),
						function: toolDef.name,
						args: typedParams,
						disposition: "simulated" as const,
					};
					await recorder.startCall(call);

					let output: ToolOutput;
					let error: string | null = null;
					try {
						output = outputOf(await toolDef.execute(typedParams, ctx));
					} catch (e) {
						error = e instanceof Error ? e.message : String(e);
						output = outputOf(error);
					}

					await finish(recorder, { ...call, result: output.result, error }, output);

					return { ...agentResult(output), details: { tool: toolDef.name, params: typedParams } };
				},
			});
		}

		for (const hook of config.hooks ?? []) {
			pi.on("tool_result", async (event) => {
				if (event.toolName !== hook.toolName) return;

				const realResult = event.content
					.filter((c): c is { type: "text"; text: string } => c.type === "text")
					.map((c) => c.text)
					.join("\n");

				let output: ToolOutput;
				let error: string | null = null;
				try {
					output = outputOf(await hook.execute(event.input, realResult, ctx, event.structuredContent));
				} catch (e) {
					error = e instanceof Error ? e.message : String(e);
					output = outputOf(error);
				}

				const call = { function: hook.toolName, disposition: "forwarded" as const, ...finishing(event.toolCallId, event.input) };
				await finish(recorder, { ...call, result: output.result, error }, output);

				return agentResult(output);
			});
		}

		// Passive reporters: record a tool's result to the control plane without
		// altering it. Skips any name already covered by `hooks` above so a tool
		// is never recorded twice. A reporter places nothing, so its finish
		// reports the details when the recorder records them: the structured
		// content the tool returned and no injection outcomes. A structured
		// content that isn't a JSON object can't go in a call record, so that
		// call is finished with its text only.
		const hookedTools = new Set((config.hooks ?? []).map((h) => h.toolName));
		for (const reportToolName of new Set(config.reportTools ?? [])) {
			if (hookedTools.has(reportToolName)) continue;
			pi.on("tool_result", async (event) => {
				if (event.toolName !== reportToolName) return;

				const result = event.content
					.filter((c): c is { type: "text"; text: string } => c.type === "text")
					.map((c) => c.text)
					.join("\n");

				const call = { function: reportToolName, disposition: "forwarded" as const, ...finishing(event.toolCallId, event.input) };
				const structured: unknown = event.structuredContent ?? null;
				await finish(
					recorder,
					{ ...call, result, error: null },
					structured === null || isJsonObject(structured)
						? { structuredContent: structured, injectionOutcomes: [] }
						: { structuredContent: null, injectionOutcomes: undefined },
				);
				// No return value: leave the tool result untouched so the agent
				// sees the real output — this observes, it does not mutate.
			});
		}

		// Start each hooked or reported call before the tool runs, so a call
		// whose tool_result never records stays visible as unfinished. If the
		// start still fails after its resends (or is refused, which is never
		// resent), the error blocks the call, so the tool doesn't run
		// unrecorded. A call without a tool call ID isn't started, since
		// tool_result couldn't find the start: it runs and its tool_result
		// records it once, as before starts existed. A blocked tool never runs,
		// so it isn't started either: the handler below records it once.
		const blockedTools = new Set(config.blockTools ?? []);
		const startedTools = new Set(
			[...hookedTools, ...(config.reportTools ?? [])].filter((name) => !blockedTools.has(name)),
		);
		if (startedTools.size > 0) {
			pi.on("tool_call", async (event) => {
				if (!startedTools.has(event.toolName)) return;
				if (!event.toolCallId) return;

				const call = { callId: callIds.forCall(event.toolCallId), args: event.input as Record<string, unknown> };
				started.set(event.toolCallId, call);
				try {
					await recorder.startCall({ function: event.toolName, disposition: "forwarded", ...call });
				} catch (e) {
					started.delete(event.toolCallId);
					throw e;
				}
			});
		}

		// Blocked tools: record the attempted call, then stop it. PI skips
		// tool_result for a blocked call, so the hooks and reporters above never
		// see it. If recording still fails after its resends, the error blocks
		// the call too, so a missing record doesn't mean the call wasn't tried.
		if (blockedTools.size > 0) {
			pi.on("tool_call", async (event) => {
				if (!blockedTools.has(event.toolName)) return;

				await recorder.finishCall({
					callId: callIds.forCall(event.toolCallId),
					function: event.toolName,
					args: event.input as Record<string, unknown>,
					result: BLOCKED_REASON,
					disposition: "blocked",
				});
				return { block: true, reason: BLOCKED_REASON };
			});
		}
	};
}
