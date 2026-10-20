import assert from "node:assert/strict";
import { test } from "node:test";
import { CallIds, ControlPlaneClient, RECORD_ATTEMPTS, createMidojoExtension } from "../src/index.ts";

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/** Answer each fetch with the next of `responses` (a status, or an error to throw) and collect the bodies sent. */
async function withFetch(responses, fn) {
	const originalFetch = globalThis.fetch;
	const previousToken = process.env.MIDOJO_SESSION_TOKEN;
	const sent = [];
	globalThis.fetch = async (_url, init) => {
		sent.push(JSON.parse(init.body));
		const response = responses[sent.length - 1];
		if (response instanceof Error) throw response;
		return new Response("{}", { status: response });
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

const entry = { function: "get_weather", args: { city: "Boston" }, result: "sunny" };

test("a record is resent with the same call_id after a 5xx or a lost request", async () => {
	await withFetch([503, new TypeError("fetch failed"), 201], async (sent) => {
		await new ControlPlaneClient("http://control").recordFunctionCall({ ...entry, callId: "call-1" });
		assert.equal(sent.length, 3);
		for (const body of sent) assert.deepEqual(body, { call_id: "call-1", ...entry });
	});
});

test("a record without a callId is resent with one UUID", async () => {
	await withFetch([500, 200], async (sent) => {
		await new ControlPlaneClient("http://control").recordFunctionCall(entry);
		assert.equal(sent.length, 2);
		assert.match(sent[0].call_id, UUID);
		assert.deepEqual(sent[1], sent[0]);
	});
});

for (const status of [400, 401, 409, 422]) {
	test(`a record is not resent after a ${status}`, async () => {
		await withFetch([status, 201], async (sent) => {
			const record = new ControlPlaneClient("http://control").recordFunctionCall({ ...entry, callId: "call-1" });
			await assert.rejects(record, (e) => e.status === status && e.retryable === false);
			assert.equal(sent.length, 1);
		});
	});
}

test("record attempts are bounded", async () => {
	await withFetch(Array(RECORD_ATTEMPTS + 1).fill(503), async (sent) => {
		await assert.rejects(new ControlPlaneClient("http://control").recordFunctionCall(entry), /503/);
		assert.equal(sent.length, RECORD_ATTEMPTS);
	});
});

test("call ids are namespaced, and fall back to a UUID", () => {
	const callIds = new CallIds("pi");
	const callId = callIds.forCall("call-1");
	assert.match(callId, /^pi:[0-9a-f-]{36}:call-1$/);
	assert.notEqual(new CallIds("pi").forCall("call-1"), callId);
	assert.match(callIds.forCall("call-1"), UUID); // already used by this extension
	assert.match(callIds.forCall(undefined), UUID);
	assert.match(callIds.forCall("x".repeat(300)), UUID);
});

test("each tool kind records the tool call ID", async () => {
	await withFetch([201, 201, 201], async (sent) => {
		const tools = [];
		const handlers = {};
		createMidojoExtension({
			controlPlaneUrl: "http://control",
			tools: [{ name: "alert", label: "Alert", description: "", parameters: {}, execute: async () => "ok" }],
			hooks: [{ toolName: "get_weather", execute: async (_args, real) => `${real}!` }],
			reportTools: ["read"],
		})({
			registerTool: (tool) => tools.push(tool),
			on: (event, callback) => {
				(handlers[event] ??= []).push(callback);
			},
		});
		const content = [{ type: "text", text: "sunny" }];
		await tools[0].execute("call-1", {});
		for (const handler of handlers.tool_result) {
			await handler({ toolName: "get_weather", toolCallId: "call-2", input: {}, content });
			await handler({ toolName: "read", toolCallId: "call-3", input: {}, content });
		}
		assert.deepEqual(
			sent.map((body) => [body.function, body.call_id.split(":").at(-1)]),
			[["alert", "call-1"], ["get_weather", "call-2"], ["read", "call-3"]],
		);
	});
});
