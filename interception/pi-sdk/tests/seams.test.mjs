import assert from "node:assert/strict";
import { test } from "node:test";
import {
	ControlPlaneClient,
	ControlPlaneError,
	ControlPlaneRejectedError,
	InvalidRequestError,
	RECORD_ATTEMPTS,
	REVISION_HEADER,
	RetryingRecorder,
	createMidojoExtension,
	transient,
} from "../src/index.ts";

/**
 * Answer each fetch with the next of `responses` (a `Response` or a function
 * returning one, or an error to throw), and collect `[method, path, body]` sent.
 */
async function withFetch(responses, fn) {
	const originalFetch = globalThis.fetch;
	const previousToken = process.env.MIDOJO_SESSION_TOKEN;
	const sent = [];
	globalThis.fetch = async (url, init) => {
		sent.push([init.method, new URL(url).pathname, init.body === undefined ? undefined : JSON.parse(init.body)]);
		const response = responses[sent.length - 1];
		if (response === undefined) throw new Error(`unexpected request ${init.method} ${url}`);
		if (response instanceof Error) throw response;
		return typeof response === "function" ? response() : response;
	};
	try {
		process.env.MIDOJO_SESSION_TOKEN = "session";
		await fn(sent);
	} finally {
		globalThis.fetch = originalFetch;
		if (previousToken === undefined) delete process.env.MIDOJO_SESSION_TOKEN;
		else process.env.MIDOJO_SESSION_TOKEN = previousToken;
	}
}

/** The control plane's environment response at `revision`. */
const environment = (revision, body) => () =>
	new Response(JSON.stringify(body), { status: 200, headers: { [REVISION_HEADER]: String(revision) } });

/** The control plane's refusal of an event. */
const refusal = (status, kind, lateFact = false) => () =>
	new Response(JSON.stringify({ detail: `refused: ${kind}`, kind, late_fact: lateFact }), { status });

const call = { callId: "call-1", function: "get_weather", args: { city: "Boston" }, disposition: "forwarded" };

test("read returns the environment with its revision", async () => {
	await withFetch([environment(4, { cities: ["Boston"] })], async (sent) => {
		const read = await new ControlPlaneClient("http://control").read();
		assert.deepEqual(read, { revision: 4, environment: { cities: ["Boston"] } });
		assert.deepEqual(sent, [["GET", "/agent/environment", undefined]]);
	});
});

test("a change is one PATCH of the fields it sets, at the expected revision when given", async () => {
	await withFetch([environment(5, { alerts: ["storm"] }), environment(6, { alerts: [] })], async (sent) => {
		const client = new ControlPlaneClient("http://control");
		assert.deepEqual(await client.change({ alerts: ["storm"] }), { revision: 5, environment: { alerts: ["storm"] } });
		assert.equal((await client.change({ alerts: [] }, { expectedRevision: 5 })).revision, 6);
		assert.deepEqual(sent, [
			["PATCH", "/agent/environment", { set: { alerts: ["storm"] }, expected_revision: null }],
			["PATCH", "/agent/environment", { set: { alerts: [] }, expected_revision: 5 }],
		]);
	});
});

test("a change at a stale revision returns null, and envUpdate false", async () => {
	await withFetch([refusal(409, "revision_conflict"), refusal(409, "revision_conflict")], async (sent) => {
		const client = new ControlPlaneClient("http://control");
		assert.equal(await client.change({ alerts: [] }, { expectedRevision: 3 }), null);
		assert.equal(await client.createToolContext().envUpdate("alerts", [], { expectedRevision: 3 }), false);
		assert.equal(sent.length, 2);
	});
});

test("envUpdate sets one field atomically, never reading and replacing the whole environment", async () => {
	await withFetch([environment(2, { alerts: ["storm"], cities: [] })], async (sent) => {
		const ctx = new ControlPlaneClient("http://control").createToolContext();
		assert.equal(await ctx.envUpdate("alerts", ["storm"]), true);
		assert.deepEqual(sent, [["PATCH", "/agent/environment", { set: { alerts: ["storm"] }, expected_revision: null }]]);
	});
});

test("a tool reads, then changes several fields at the revision it read", async () => {
	await withFetch([environment(7, { a: 1, b: 2 }), environment(8, { a: 2, b: 3 })], async (sent) => {
		const ctx = new ControlPlaneClient("http://control").createToolContext();
		const { revision, environment: env } = await ctx.read();
		assert.equal(await ctx.change({ a: env.a + 1, b: env.b + 1 }, { expectedRevision: revision }), true);
		assert.deepEqual(sent[1], ["PATCH", "/agent/environment", { set: { a: 2, b: 3 }, expected_revision: 7 }]);
	});
});

test("a refusal carries the control plane's kind and late-fact flag", async () => {
	await withFetch([refusal(422, "invalid_environment"), refusal(409, "closing", true)], async () => {
		const client = new ControlPlaneClient("http://control");
		await assert.rejects(client.change({ nope: 1 }), (e) => {
			assert.ok(e instanceof ControlPlaneRejectedError);
			assert.ok(e instanceof ControlPlaneError); // caught where a ControlPlaneError was
			assert.equal(e.kind, "invalid_environment");
			assert.equal(e.detail, "refused: invalid_environment");
			assert.equal(e.lateFact, false);
			assert.equal(transient(e), false);
			return true;
		});
		await assert.rejects(client.finishCall({ ...call, result: "sunny" }), (e) => {
			assert.equal(e.kind, "closing");
			assert.equal(e.lateFact, true);
			assert.equal(e.status, 409);
			return true;
		});
	});
});

test("a 422 without a kind is an invalid request; other failures are plain control plane errors", async () => {
	const invalid = () => new Response(JSON.stringify({ detail: [{ loc: ["body", "call_id"] }] }), { status: 422 });
	const proxy = () => new Response("<html>bad gateway</html>", { status: 502 });
	const unknownKind = () => new Response(JSON.stringify({ detail: "x", kind: "something_new" }), { status: 409 });
	await withFetch([invalid, proxy, unknownKind, new TypeError("fetch failed")], async () => {
		const client = new ControlPlaneClient("http://control");
		await assert.rejects(client.startCall(call), (e) => e instanceof InvalidRequestError && !transient(e));
		for (const retryable of [true, false, true]) {
			await assert.rejects(client.startCall(call), (e) => {
				assert.ok(e instanceof ControlPlaneError && !(e instanceof ControlPlaneRejectedError));
				assert.equal(transient(e), retryable);
				return true;
			});
		}
	});
});

test("the records carry the disposition and the blocked flag it implies", async () => {
	const created = () => new Response("{}", { status: 201 });
	await withFetch([created, created, created], async (sent) => {
		const client = new ControlPlaneClient("http://control", { sessionToken: "explicit" });
		await client.startCall(call);
		await client.finishCall({ ...call, result: "sunny" });
		await client.finishCall({ ...call, callId: "call-2", disposition: "denied", result: "no", error: "no" });
		assert.deepEqual(
			sent.map(([method, path, body]) => [method, path, body]),
			[
				["POST", "/agent/function-calls/start", { call_id: "call-1", function: "get_weather", args: call.args, disposition: "forwarded" }],
				["POST", "/agent/function-calls", { call_id: "call-1", function: "get_weather", args: call.args, disposition: "forwarded", result: "sunny", error: null, blocked: false }],
				["POST", "/agent/function-calls", { call_id: "call-2", function: "get_weather", args: call.args, disposition: "denied", result: "no", error: "no", blocked: true }],
			],
		);
	});
});

test("the client sends the session token it was given, or else MIDOJO_SESSION_TOKEN", async () => {
	const originalFetch = globalThis.fetch;
	const tokens = [];
	globalThis.fetch = async (_url, init) => {
		tokens.push(init.headers.Authorization);
		return new Response("{}", { status: 201 });
	};
	try {
		process.env.MIDOJO_SESSION_TOKEN = "from-env";
		await new ControlPlaneClient("http://control", { sessionToken: "explicit" }).startCall(call);
		await new ControlPlaneClient("http://control").startCall(call);
		assert.deepEqual(tokens, ["Bearer explicit", "Bearer from-env"]);
	} finally {
		globalThis.fetch = originalFetch;
		delete process.env.MIDOJO_SESSION_TOKEN;
	}
});

test("the retrying recorder resends a transient failure and never a refusal", async () => {
	const created = () => new Response("{}", { status: 201 });
	await withFetch([() => new Response("", { status: 503 }), created, refusal(409, "call_conflict")], async (sent) => {
		const recorder = new RetryingRecorder(new ControlPlaneClient("http://control"), {
			retryable: transient,
			backoffMs: 0,
		});
		await recorder.startCall(call);
		assert.equal(sent.length, 2);
		await assert.rejects(recorder.finishCall({ ...call, result: "sunny" }), ControlPlaneRejectedError);
		assert.equal(sent.length, 3);
	});
});

test("the deprecated record methods never resend a refusal either", async () => {
	await withFetch([refusal(409, "closing", true)], async (sent) => {
		const client = new ControlPlaneClient("http://control");
		const record = client.recordFunctionCall({ function: "read", args: {}, result: "x", callId: "call-1" });
		await assert.rejects(record, (e) => e.kind === "closing" && e.lateFact === true);
		assert.equal(sent.length, 1);
	});
});

test("a refused blocked-call record is sent once and still blocks the call", async () => {
	await withFetch([refusal(401, "session_closed")], async (sent) => {
		const handlers = {};
		createMidojoExtension({ controlPlaneUrl: "http://control", blockTools: ["send_email"] })({
			on: (event, callback) => {
				handlers[event] = callback;
			},
		});
		const blocked = handlers.tool_call({ toolName: "send_email", toolCallId: "call-1", input: {} });
		await assert.rejects(blocked, (e) => e instanceof ControlPlaneRejectedError && e.kind === "session_closed");
		assert.equal(sent.length, 1);
	});
});

test("an extension records through the recorder and world it's given, with no request", async () => {
	await withFetch([], async (sent) => {
		const records = [];
		const recorder = {
			startCall: async (start) => records.push(["start", start]),
			finishCall: async (finish) => records.push(["finish", finish]),
		};
		let state = { revision: 1, environment: { alerts: [] } };
		const world = {
			read: async () => structuredClone(state),
			change: async (values, { expectedRevision } = {}) => {
				if (expectedRevision !== undefined && expectedRevision !== state.revision) return null;
				state = { revision: state.revision + 1, environment: { ...state.environment, ...values } };
				return structuredClone(state);
			},
		};
		const tools = [];
		const handlers = {};
		createMidojoExtension({
			recorder,
			world,
			tools: [
				{
					name: "send_alert",
					label: "Send alert",
					description: "",
					parameters: {},
					execute: async (params, ctx) => {
						const { revision, environment: env } = await ctx.read();
						const changed = await ctx.envUpdate("alerts", [...env.alerts, params.message], {
							expectedRevision: revision,
						});
						return changed ? "sent" : "retry";
					},
				},
			],
			hooks: [{ toolName: "get_weather", execute: async (_args, real, ctx) => `${real} (${(await ctx.env("alerts")).length})` }],
			blockTools: ["send_email"],
		})({
			registerTool: (tool) => tools.push(tool),
			on: (event, callback) => {
				(handlers[event] ??= []).push(callback);
			},
		});

		const output = await tools[0].execute("call-1", { message: "storm" });
		for (const handler of handlers.tool_call) await handler({ toolName: "get_weather", toolCallId: "call-2", input: {} });
		const [hooked] = await Promise.all(
			handlers.tool_result.map((handler) =>
				handler({ toolName: "get_weather", toolCallId: "call-2", input: {}, content: [{ type: "text", text: "sunny" }] }),
			),
		);
		const blocked = await Promise.all(
			handlers.tool_call.map((handler) => handler({ toolName: "send_email", toolCallId: "call-3", input: {} })),
		);

		assert.deepEqual(sent, []);
		assert.equal(output.content[0].text, "sent");
		assert.equal(hooked.content[0].text, "sunny (1)");
		assert.deepEqual(blocked.filter(Boolean), [{ block: true, reason: "Tool execution was blocked" }]);
		assert.deepEqual(state, { revision: 2, environment: { alerts: ["storm"] } });
		assert.deepEqual(
			records.map(([event, record]) => [event, record.function, record.disposition, record.callId.split(":").at(-1)]),
			[
				["start", "send_alert", "simulated", "call-1"],
				["finish", "send_alert", "simulated", "call-1"],
				["start", "get_weather", "forwarded", "call-2"],
				["finish", "get_weather", "forwarded", "call-2"],
				["finish", "send_email", "blocked", "call-3"],
			],
		);
	});
});

test("RECORD_ATTEMPTS bounds the retrying recorder's sends", async () => {
	await withFetch(Array(RECORD_ATTEMPTS).fill(new TypeError("fetch failed")), async (sent) => {
		const recorder = new RetryingRecorder(new ControlPlaneClient("http://control"), { retryable: transient, backoffMs: 0 });
		await assert.rejects(recorder.startCall(call), ControlPlaneError);
		assert.equal(sent.length, RECORD_ATTEMPTS);
	});
});
