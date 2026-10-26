import { randomUUID } from "node:crypto";
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
 * - `forwarded`: the real tool ran, and the agent got its answer (rewritten by a hook, or not)
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
 * A call's finish, or the whole record of a call that wasn't started. It is a
 * plain finish: it carries no injection outcomes, so the control plane has no
 * delivery fact for a payload in the result and reports its exposure as
 * unknown. The result is recorded before the agent gets it, so a record doesn't
 * show the agent received it.
 */
export interface CallFinish extends CallStart {
	result: string;
	error?: string | null;
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

export interface MidojoToolDef {
	name: string;
	label: string;
	description: string;
	parameters: TSchema;
	execute: (params: Record<string, unknown>, ctx: ToolContext) => Promise<string>;
}

export interface MidojoToolHook {
	toolName: string;
	execute: (
		args: Record<string, unknown>,
		realResult: string,
		ctx: ToolContext,
	) => Promise<string>;
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
 * `startCall`, `finishCall`, `read` and `change` send each request once. A
 * response that carries one of the control plane's rejection kinds throws a
 * `ControlPlaneRejectedError`; a 422 without one, an `InvalidRequestError`; any
 * other failure (a lost request, a proxy's error page, a 401 without a session)
 * a `ControlPlaneError`, which `transient` classifies for a `RetryingRecorder`.
 */
export class ControlPlaneClient implements Recorder, WorldState {
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
		await this.request("/function-calls", "POST", {
			...startBody(call),
			result: call.result,
			error: call.error ?? null,
			blocked: REFUSED.has(call.disposition),
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

					let result: string;
					let error: string | null = null;
					try {
						result = await toolDef.execute(typedParams, ctx);
					} catch (e) {
						error = e instanceof Error ? e.message : String(e);
						result = error;
					}

					await recorder.finishCall({ ...call, result, error });

					return {
						content: [{ type: "text" as const, text: result }],
						details: { tool: toolDef.name, params: typedParams },
					};
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

				let result: string;
				let error: string | null = null;
				try {
					result = await hook.execute(event.input, realResult, ctx);
				} catch (e) {
					error = e instanceof Error ? e.message : String(e);
					result = error;
				}

				await recorder.finishCall({
					function: hook.toolName,
					result,
					error,
					disposition: "forwarded",
					...finishing(event.toolCallId, event.input),
				});

				return {
					content: [{ type: "text" as const, text: result }],
				};
			});
		}

		// Passive reporters: record a tool's result to the control plane without
		// altering it. Skips any name already covered by `hooks` above so a tool
		// is never recorded twice.
		const hookedTools = new Set((config.hooks ?? []).map((h) => h.toolName));
		for (const reportToolName of new Set(config.reportTools ?? [])) {
			if (hookedTools.has(reportToolName)) continue;
			pi.on("tool_result", async (event) => {
				if (event.toolName !== reportToolName) return;

				const result = event.content
					.filter((c): c is { type: "text"; text: string } => c.type === "text")
					.map((c) => c.text)
					.join("\n");

				await recorder.finishCall({
					function: reportToolName,
					result,
					error: null,
					disposition: "forwarded",
					...finishing(event.toolCallId, event.input),
				});
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
