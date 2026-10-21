import assert from "node:assert/strict";
import { test } from "node:test";
import { BLOCKED_REASON, ControlPlaneError, RECORD_ATTEMPTS, createMidojoExtension } from "../src/index.ts";

const START = "/agent/function-calls/start";
const FINISH = "/agent/function-calls";

/** Answer each POST to a path with the next status scripted for it (201 when none is left), and collect what was sent. */
async function withControlPlane(script, fn) {
	const originalFetch = globalThis.fetch;
	const previousToken = process.env.MIDOJO_SESSION_TOKEN;
	const sent = [];
	globalThis.fetch = async (url, init) => {
		const path = new URL(url).pathname;
		sent.push([path, JSON.parse(init.body)]);
		const status = script[path]?.shift() ?? 201;
		return new Response("{}", { status });
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

/** Load an extension with `config`, and return its registered tools and event handlers. */
function load(config) {
	const tools = [];
	const handlers = {};
	createMidojoExtension({ controlPlaneUrl: "http://control", ...config })({
		registerTool: (tool) => tools.push(tool),
		on: (event, callback) => {
			(handlers[event] ??= []).push(callback);
		},
	});
	const emit = async (event, payload) => {
		const results = [];
		for (const handler of handlers[event] ?? []) results.push(await handler(payload));
		return results;
	};
	return { tools, emit };
}

const input = { city: "Boston" };
const content = [{ type: "text", text: "sunny" }];

test("a registered tool's call is started before it runs and finished after", async () => {
	await withControlPlane({}, async (sent) => {
		const ran = [];
		const { tools } = load({
			tools: [
				{
					name: "alert",
					label: "Alert",
					description: "",
					parameters: {},
					execute: async () => {
						ran.push(sent.length);
						return "ok";
					},
				},
			],
		});

		await tools[0].execute("call-1", input);

		assert.deepEqual(ran, [1]); // the tool ran after the start was sent
		const [[startPath, start], [finishPath, finish]] = sent;
		assert.deepEqual([startPath, finishPath], [START, FINISH]);
		assert.match(start.call_id, /^pi:[0-9a-f-]{36}:call-1$/);
		assert.deepEqual(start, { call_id: start.call_id, function: "alert", args: input });
		assert.deepEqual(finish, { ...start, result: "ok", error: null });
	});
});

for (const kind of ["hooks", "reportTools"]) {
	test(`a ${kind} call is started on tool_call and finished on tool_result`, async () => {
		await withControlPlane({}, async (sent) => {
			const config =
				kind === "hooks"
					? { hooks: [{ toolName: "get_weather", execute: async (_args, real) => real }] }
					: { reportTools: ["get_weather"] };
			const { emit } = load(config);

			assert.deepEqual(await emit("tool_call", { toolName: "get_weather", toolCallId: "call-1", input }), [undefined]);
			assert.equal(sent.length, 1);
			await emit("tool_result", { toolName: "get_weather", toolCallId: "call-1", input, content });

			const [[startPath, start], [finishPath, finish]] = sent;
			assert.deepEqual([startPath, finishPath], [START, FINISH]);
			assert.deepEqual(start, { call_id: start.call_id, function: "get_weather", args: input });
			assert.equal(finish.call_id, start.call_id);
			assert.deepEqual(finish.args, input);
			assert.equal(finish.result, "sunny");
		});
	});
}

test("a finish that fails leaves the start as the only record sent", async () => {
	await withControlPlane({ [FINISH]: Array(RECORD_ATTEMPTS).fill(503) }, async (sent) => {
		const { emit } = load({ reportTools: ["get_weather"] });

		await emit("tool_call", { toolName: "get_weather", toolCallId: "call-1", input });
		await assert.rejects(
			emit("tool_result", { toolName: "get_weather", toolCallId: "call-1", input, content }),
			ControlPlaneError,
		);

		const paths = sent.map(([path]) => path);
		assert.deepEqual(paths, [START, ...Array(RECORD_ATTEMPTS).fill(FINISH)]);
		assert.ok(sent.every(([, body]) => body.call_id === sent[0][1].call_id));
	});
});

test("a start is resent with the same body after a 5xx", async () => {
	await withControlPlane({ [START]: [503] }, async (sent) => {
		const { emit } = load({ reportTools: ["get_weather"] });

		await emit("tool_call", { toolName: "get_weather", toolCallId: "call-1", input });

		assert.deepEqual(
			sent.map(([path]) => path),
			[START, START],
		);
		assert.deepEqual(sent[1][1], sent[0][1]);
	});
});

test("a start that fails throws, so PI blocks the call, and the tool doesn't run", async () => {
	await withControlPlane({ [START]: [409, 409] }, async (sent) => {
		const ran = [];
		const { tools, emit } = load({
			reportTools: ["get_weather"],
			tools: [{ name: "alert", label: "Alert", description: "", parameters: {}, execute: async () => ran.push(1) }],
		});

		await assert.rejects(emit("tool_call", { toolName: "get_weather", toolCallId: "call-1", input }), ControlPlaneError);
		await assert.rejects(tools[0].execute("call-2", input), ControlPlaneError);

		assert.deepEqual(ran, []);
		assert.deepEqual(
			sent.map(([path]) => path),
			[START, START],
		);
	});
});

for (const kind of ["hooks", "reportTools"]) {
	test(`a ${kind} call without a tool call ID isn't started and is recorded once on tool_result`, async () => {
		await withControlPlane({}, async (sent) => {
			const config =
				kind === "hooks"
					? { hooks: [{ toolName: "get_weather", execute: async (_args, real) => real }] }
					: { reportTools: ["get_weather"] };
			const { emit } = load(config);

			const results = await emit("tool_call", { toolName: "get_weather", input });
			assert.ok(results.every((result) => result === undefined));
			assert.deepEqual(sent, []);

			await emit("tool_result", { toolName: "get_weather", input, content });

			assert.deepEqual(
				sent.map(([path]) => path),
				[FINISH],
			);
			assert.equal(sent[0][1].function, "get_weather");
			assert.ok(sent[0][1].call_id);
		});
	});

	test(`a ${kind} tool that is also blocked is recorded once, as blocked, without a start`, async () => {
		await withControlPlane({}, async (sent) => {
			const config =
				kind === "hooks"
					? { hooks: [{ toolName: "send_email", execute: async (_args, real) => real }] }
					: { reportTools: ["send_email"] };
			const { emit } = load({ ...config, blockTools: ["send_email"] });

			const results = await emit("tool_call", { toolName: "send_email", toolCallId: "call-1", input });

			assert.deepEqual(results.filter(Boolean), [{ block: true, reason: BLOCKED_REASON }]);
			assert.equal(sent.length, 1);
			const [[path, body]] = sent;
			assert.equal(path, FINISH);
			assert.equal(body.blocked, true);
			assert.match(body.call_id, /^pi:[0-9a-f-]{36}:call-1$/);
		});
	});
}

test("other tools and blocked tools aren't started", async () => {
	await withControlPlane({}, async (sent) => {
		const { emit } = load({ reportTools: ["get_weather"], blockTools: ["send_email"] });

		await emit("tool_call", { toolName: "read", toolCallId: "call-1", input });
		await emit("tool_call", { toolName: "send_email", toolCallId: "call-2", input });

		assert.deepEqual(
			sent.map(([path, body]) => [path, body.blocked]),
			[[FINISH, true]],
		);
	});
});
