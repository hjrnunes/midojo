import { randomUUID } from "node:crypto";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import type { TSchema } from "typebox";

export interface ToolContext {
	env<T = unknown>(field: string): Promise<T>;
	envUpdate(field: string, value: unknown): Promise<void>;
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
	controlPlaneUrl: string;
	tools?: MidojoToolDef[];
	hooks?: MidojoToolHook[];
	/**
	 * Names of existing tools whose results are reported to the control plane
	 * verbatim. Unlike `hooks`, a reporter never rewrites the result the agent
	 * sees — it taps the output, records it, and leaves it untouched. Use this to
	 * make a tool's result (e.g. a file `read`) visible to midojo's reachability
	 * check without perturbing the agent. Names already handled by `hooks` are
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

/** How many times a function-call record, or a call's start, is sent before its error is thrown. */
export const RECORD_ATTEMPTS = 3;

/** The wait before the first resend of a record, doubled before each later one. */
export const RECORD_BACKOFF_MS = 100;

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

/** Read the evaluation process's session token from the environment on every request. */
export class ControlPlaneClient {
	private baseUrl: string;

	constructor(baseUrl: string | undefined = process.env.MIDOJO_URL) {
		if (!baseUrl) throw new Error("MiDojo control plane URL is required. Set MIDOJO_URL or configure controlPlaneUrl.");
		let end = baseUrl.length;
		while (end > 0 && baseUrl[end - 1] === "/") end--;
		this.baseUrl = `${baseUrl.slice(0, end)}/agent`;
	}

	private getSessionToken(): string {
		const token = process.env.MIDOJO_SESSION_TOKEN;
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
		if (!resp.ok) throw new ControlPlaneError(`MiDojo ${method} ${path} failed (${resp.status})`, resp.status);
		return resp;
	}

	async getEnvironment(): Promise<Record<string, unknown>> {
		return (await this.request("/environment")).json() as Promise<Record<string, unknown>>;
	}

	async putEnvironment(env: Record<string, unknown>): Promise<void> {
		await this.request("/environment", "PUT", env);
	}

	/**
	 * Record a call's start: an unfinished record that `recordFunctionCall`
	 * finishes. Send it before the tool runs, then finish the call with the
	 * same `callId`, `function` and `args`. If the finish never arrives, the
	 * call stays on the control plane as unfinished. Resent like
	 * `recordFunctionCall`; a resent start doesn't record the call twice.
	 */
	async startFunctionCall(entry: { function: string; args: Record<string, unknown>; callId: string }): Promise<void> {
		await this.postRecord("/function-calls/start", {
			call_id: entry.callId,
			function: entry.function,
			args: entry.args,
		});
	}

	/**
	 * Record a call, at most once per `callId` (a UUID when not given). When
	 * `callId` was started with `startFunctionCall`, this finishes that record.
	 * A lost request or a 5xx is resent with the same `call_id`, up to
	 * `RECORD_ATTEMPTS` sends in all, so resending a record that reached the
	 * control plane doesn't record it twice. This is at most one record per
	 * `call_id`, not exactly-once: a record whose sends all fail is missing, or
	 * left unfinished when it was started. A 4xx, including a 409 for a
	 * `call_id` recorded with another body, is thrown at once.
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
		await this.postRecord("/function-calls", { call_id: callId ?? randomUUID(), ...fields });
	}

	/** POST `body`, resending it after a lost request or a 5xx, and throwing a 4xx at once. */
	private async postRecord(path: string, body: unknown): Promise<void> {
		for (let attempt = 1; ; attempt++) {
			try {
				await this.request(path, "POST", body);
				return;
			} catch (e) {
				if (attempt >= RECORD_ATTEMPTS || !(e instanceof ControlPlaneError && e.retryable)) throw e;
			}
			await new Promise((resolve) => setTimeout(resolve, RECORD_BACKOFF_MS * 2 ** (attempt - 1)));
		}
	}

	createToolContext(): ToolContext {
		return {
			env: async <T = unknown>(field: string): Promise<T> => {
				const env = await this.getEnvironment();
				return env[field] as T;
			},
			envUpdate: async (field: string, value: unknown): Promise<void> => {
				const env = await this.getEnvironment();
				env[field] = value;
				await this.putEnvironment(env);
			},
		};
	}
}

export function createMidojoExtension(config: MidojoExtensionConfig): (pi: ExtensionAPI) => void {
	return (pi: ExtensionAPI) => {
		const client = new ControlPlaneClient(config.controlPlaneUrl);
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
					const ctx = client.createToolContext();
					const callId = callIds.forCall(toolCallId);
					await client.startFunctionCall({ function: toolDef.name, args: typedParams, callId });

					let result: string;
					let error: string | null = null;
					try {
						result = await toolDef.execute(typedParams, ctx);
					} catch (e) {
						error = e instanceof Error ? e.message : String(e);
						result = error;
					}

					await client.recordFunctionCall({ function: toolDef.name, args: typedParams, result, error, callId });

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

				const ctx = client.createToolContext();
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

				await client.recordFunctionCall({
					function: hook.toolName,
					result,
					error,
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

				await client.recordFunctionCall({
					function: reportToolName,
					result,
					error: null,
					...finishing(event.toolCallId, event.input),
				});
				// No return value: leave the tool result untouched so the agent
				// sees the real output — this observes, it does not mutate.
			});
		}

		// Start each hooked or reported call before the tool runs, so a call
		// whose tool_result never records stays visible as unfinished. If the
		// start still fails after its resends, the error blocks the call, so the
		// tool doesn't run unrecorded. A call without a tool call ID isn't
		// started, since tool_result couldn't find the start: it runs and its
		// tool_result records it once, as before starts existed. A blocked tool
		// never runs, so it isn't started either: the handler below records it once.
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
					await client.startFunctionCall({ function: event.toolName, ...call });
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

				await client.recordFunctionCall({
					function: event.toolName,
					args: event.input as Record<string, unknown>,
					result: BLOCKED_REASON,
					blocked: true,
					callId: callIds.forCall(event.toolCallId),
				});
				return { block: true, reason: BLOCKED_REASON };
			});
		}
	};
}
