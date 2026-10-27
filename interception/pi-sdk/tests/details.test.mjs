import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
	ControlPlaneClient,
	ControlPlaneRejectedError,
	RetryingRecorder,
	checkInjectionRecord,
	createMidojoExtension,
	finishDetailed,
	recordsDetails,
	transient,
} from "../src/index.ts";

/** The body the PI and Python clients both send for one detailed finish, pinned for both languages. */
const FIXTURE = JSON.parse(readFileSync(new URL("../../../tests/fixtures/pi_sdk_detailed_finish.json", import.meta.url), "utf8"));

/** Answer each fetch with the next of `responses` (a status, or an error to throw), and collect `[path, raw body]` sent. */
async function withFetch(responses, fn) {
	const originalFetch = globalThis.fetch;
	const previousToken = process.env.MIDOJO_SESSION_TOKEN;
	const sent = [];
	globalThis.fetch = async (url, init) => {
		sent.push([new URL(url).pathname, init.body]);
		const response = responses[sent.length - 1];
		if (response === undefined) throw new Error(`unexpected request ${init.method} ${url}`);
		if (response instanceof Error) throw response;
		if (typeof response === "function") return response();
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

/** The fixture's call (the Python client's keywords) as a PI `DetailedCallFinish`. */
function piCall({ call_id, function: name, args, result, error, disposition, structured_content, injection_outcomes }) {
	return {
		callId: call_id,
		function: name,
		args,
		result,
		error,
		disposition,
		structuredContent: structured_content,
		injectionOutcomes: injection_outcomes.map(
			({ instruction_index, probe_key, selected, transformation, emitted, ...rest }) => ({
				instructionIndex: instruction_index,
				probeKey: probe_key,
				...rest,
				...(selected === undefined ? {} : { selected }),
				...(transformation === undefined ? {} : { transformation }),
				...(emitted === undefined ? {} : { emitted }),
			}),
		),
	};
}

const applied = {
	instructionIndex: 0,
	probeKey: "p:main",
	occurrence: 1,
	status: "applied",
	detail: "exact selector matched",
	selected: true,
	transformation: "identity",
	emitted: ["content_block"],
};
const finished = { callId: "call-1", function: "get_weather", args: { city: "Boston" }, disposition: "forwarded", result: "sunny" };
const detailed = { ...finished, structuredContent: { forecast: "sunny" }, injectionOutcomes: [applied] };

/** A recorder that keeps what it's sent; `detailed` gives it `finishCallDetailed`. */
function keeping({ detailed: withDetails }) {
	const records = [];
	const recorder = {
		startCall: async (call) => records.push(["start", call]),
		finishCall: async (call) => records.push(["finish", call]),
	};
	if (withDetails) recorder.finishCallDetailed = async (call) => records.push(["detailed", call]);
	return { recorder, records };
}

test("the client sends the pinned body for a detailed finish, byte for byte", async () => {
	await withFetch([201], async (sent) => {
		await new ControlPlaneClient("http://control").finishCallDetailed(piCall(FIXTURE.call));
		assert.deepEqual(sent, [[FIXTURE.request.path, JSON.stringify(FIXTURE.request.body)]]);
	});
});

test("a detailed finish always sends its outcomes, an empty list included, and null structured content", async () => {
	await withFetch([201], async (sent) => {
		await new ControlPlaneClient("http://control").finishCallDetailed({ ...finished, structuredContent: null, injectionOutcomes: [] });
		assert.deepEqual(JSON.parse(sent[0][1]), {
			call_id: "call-1",
			function: "get_weather",
			args: { city: "Boston" },
			disposition: "forwarded",
			result: "sunny",
			error: null,
			blocked: false,
			structured_content: null,
			injection_outcomes: [],
		});
	});
});

test("an outcome's unreported delivery facts are left out, not sent as null", async () => {
	await withFetch([201], async (sent) => {
		const { selected, transformation, emitted, ...unreported } = applied;
		await new ControlPlaneClient("http://control").finishCallDetailed({ ...detailed, injectionOutcomes: [unreported] });
		assert.deepEqual(JSON.parse(sent[0][1]).injection_outcomes, [
			{ instruction_index: 0, probe_key: "p:main", occurrence: 1, status: "applied", detail: "exact selector matched" },
		]);
	});
});

test("occurrence 0 is accepted only as an unmatched outcome that placed nothing", async () => {
	const uncounted = { instructionIndex: 0, probeKey: null, occurrence: 0, status: "unmatched", detail: "d" };
	for (const accepted of [
		uncounted,
		{ ...uncounted, selected: false, transformation: null, emitted: [] },
		{ ...uncounted, occurrence: 1, selected: true, transformation: "identity", emitted: ["content_block"] },
	]) {
		assert.doesNotThrow(() => checkInjectionRecord(accepted));
	}
	for (const refused of [
		{ ...uncounted, status: "applied" },
		{ ...uncounted, status: "skipped_occurrence" },
		{ ...uncounted, selected: true },
		{ ...uncounted, transformation: "identity" },
		{ ...uncounted, emitted: ["structured_content"] },
	]) {
		assert.throws(() => checkInjectionRecord(refused), RangeError);
		// Refused before anything is sent, by the client and by finishDetailed, whatever the recorder.
		await withFetch([], async (sent) => {
			const call = { ...detailed, injectionOutcomes: [refused] };
			await assert.rejects(new ControlPlaneClient("http://control").finishCallDetailed(call), RangeError);
			await assert.rejects(finishDetailed(keeping({ detailed: false }).recorder, call), RangeError);
			assert.deepEqual(sent, []);
		});
	}
});

test("recordsDetails is true for a recorder with finishCallDetailed, and a wrapper's declaration wins", () => {
	assert.equal(recordsDetails(new ControlPlaneClient("http://control")), true);
	assert.equal(recordsDetails(keeping({ detailed: true }).recorder), true);
	assert.equal(recordsDetails(keeping({ detailed: false }).recorder), false);
	const options = { retryable: transient };
	assert.equal(recordsDetails(new RetryingRecorder(keeping({ detailed: true }).recorder, options)), true);
	// A RetryingRecorder always has the method, so only its declaration keeps the check truthful.
	assert.equal(recordsDetails(new RetryingRecorder(keeping({ detailed: false }).recorder, options)), false);
	assert.equal(recordsDetails(new RetryingRecorder(new RetryingRecorder(keeping({ detailed: false }).recorder, options), options)), false);
});

test("finishDetailed sends the details to a recorder that records them, and falls back to finishCall", async () => {
	const withDetails = keeping({ detailed: true });
	assert.equal(await finishDetailed(withDetails.recorder, detailed), true);
	assert.deepEqual(withDetails.records, [["detailed", detailed]]);

	const plain = keeping({ detailed: false });
	assert.equal(await finishDetailed(plain.recorder, detailed), false);
	assert.deepEqual(plain.records, [["finish", finished]]);
});

test("the retrying recorder forwards the detailed finish, resending it, or falls back for a plain recorder", async () => {
	await withFetch([503, 201, () => new Response(JSON.stringify({ detail: "x", kind: "call_conflict" }), { status: 409 })], async (sent) => {
		const recorder = new RetryingRecorder(new ControlPlaneClient("http://control"), { retryable: transient, backoffMs: 0 });
		await recorder.finishCallDetailed(detailed);
		assert.equal(sent.length, 2);
		assert.equal(sent[1][1], sent[0][1]);
		assert.deepEqual(JSON.parse(sent[1][1]).injection_outcomes[0].emitted, ["content_block"]);
		await assert.rejects(recorder.finishCallDetailed(detailed), ControlPlaneRejectedError);
		assert.equal(sent.length, 3);
	});

	const plain = keeping({ detailed: false });
	await new RetryingRecorder(plain.recorder, { retryable: transient }).finishCallDetailed(detailed);
	assert.deepEqual(plain.records, [["finish", finished]]);
});

/** An extension over `recorder`, with its registered tools and event handlers. */
function extension(recorder, config) {
	const tools = [];
	const handlers = {};
	createMidojoExtension({ recorder, world: { read: async () => ({ revision: 0, environment: {} }), change: async () => null }, ...config })({
		registerTool: (tool) => tools.push(tool),
		on: (event, callback) => {
			(handlers[event] ??= []).push(callback);
		},
	});
	const emit = async (event, payload) => {
		const answers = [];
		for (const handler of handlers[event] ?? []) answers.push(await handler(payload));
		return answers;
	};
	return { tools, emit };
}

const placed = {
	instructionIndex: 0,
	probeKey: "p:main",
	occurrence: 1,
	status: "applied",
	detail: "hook placed the payload",
	selected: true,
	transformation: "identity",
	emitted: ["content_block", "structured_content"],
};

test("a hook that reports its outcomes is finished with them and the structured content the agent gets", async () => {
	const { recorder, records } = keeping({ detailed: true });
	const seen = [];
	const { emit } = extension(recorder, {
		hooks: [
			{
				toolName: "get_weather",
				execute: async (_args, real, _ctx, structured) => {
					seen.push(structured);
					return {
						result: `${real} INJECTED`,
						structuredContent: { ...structured, note: "INJECTED" },
						injectionOutcomes: [placed],
					};
				},
			},
		],
	});
	const input = { city: "Boston" };
	await emit("tool_call", { toolName: "get_weather", toolCallId: "call-1", input });
	const [answer] = await emit("tool_result", {
		toolName: "get_weather",
		toolCallId: "call-1",
		input,
		content: [{ type: "text", text: "sunny" }],
		structuredContent: { forecast: "sunny" },
	});

	assert.deepEqual(seen, [{ forecast: "sunny" }]);
	// The agent gets the text and structured content that were recorded.
	assert.deepEqual(answer, {
		content: [{ type: "text", text: "sunny INJECTED" }],
		structuredContent: { forecast: "sunny", note: "INJECTED" },
	});
	const [[, start], [kind, finish]] = records;
	assert.equal(kind, "detailed");
	assert.equal(finish.callId, start.callId);
	assert.equal(finish.result, "sunny INJECTED");
	assert.deepEqual(finish.structuredContent, answer.structuredContent);
	assert.deepEqual(finish.injectionOutcomes, [placed]);
});

test("a hook's output without outcomes, or a string, is finished with its text only", async () => {
	const { recorder, records } = keeping({ detailed: true });
	const outputs = [async (_a, real) => `${real}!`, async (_a, real) => ({ result: `${real}?`, structuredContent: { kept: true } })];
	let next = 0;
	const { emit } = extension(recorder, { hooks: [{ toolName: "get_weather", execute: (...a) => outputs[next++](...a) }] });
	const answers = [];
	for (const id of ["call-1", "call-2"]) {
		answers.push(...(await emit("tool_result", { toolName: "get_weather", toolCallId: id, input: {}, content: [{ type: "text", text: "sunny" }] })));
	}

	assert.deepEqual(answers, [
		{ content: [{ type: "text", text: "sunny!" }] },
		{ content: [{ type: "text", text: "sunny?" }], structuredContent: { kept: true } },
	]);
	// Not reported: the record's injection_outcomes stay null, never an empty list the hook didn't vouch for.
	assert.deepEqual(records.map(([kind, call]) => [kind, call.result, "injectionOutcomes" in call]), [
		["finish", "sunny!", false],
		["finish", "sunny?", false],
	]);
});

test("a registered tool can report its outcomes and structured content too", async () => {
	const { recorder, records } = keeping({ detailed: true });
	const { tools } = extension(recorder, {
		tools: [
			{
				name: "lookup",
				label: "Lookup",
				description: "",
				parameters: {},
				execute: async () => ({ result: "found", structuredContent: { id: 1 }, injectionOutcomes: [] }),
			},
		],
	});
	const answer = await tools[0].execute("call-1", { q: "x" });

	assert.deepEqual(answer.structuredContent, { id: 1 });
	assert.deepEqual(answer.content, [{ type: "text", text: "found" }]);
	assert.deepEqual(records.map(([kind, call]) => [kind, call.disposition, call.structuredContent, call.injectionOutcomes]), [
		["start", "simulated", undefined, undefined],
		["detailed", "simulated", { id: 1 }, []],
	]);
});

test("a reporter finishes with the tool's structured content and no outcomes, or its text only for non-object content", async () => {
	const { recorder, records } = keeping({ detailed: true });
	const { emit } = extension(recorder, { reportTools: ["read"] });
	const content = [{ type: "text", text: "body" }];
	const answers = [
		...(await emit("tool_result", { toolName: "read", toolCallId: "call-1", input: {}, content, structuredContent: { lines: 1 } })),
		...(await emit("tool_result", { toolName: "read", toolCallId: "call-2", input: {}, content })),
		...(await emit("tool_result", { toolName: "read", toolCallId: "call-3", input: {}, content, structuredContent: [1, 2] })),
	];

	assert.deepEqual(answers, [undefined, undefined, undefined]); // the result is left as it was
	assert.deepEqual(records.map(([kind, call]) => [kind, call.structuredContent, call.injectionOutcomes]), [
		["detailed", { lines: 1 }, []],
		["detailed", null, []],
		["finish", undefined, undefined],
	]);
});

test("over a plain recorder the extension records every call with finishCall", async () => {
	const { recorder, records } = keeping({ detailed: false });
	const { emit, tools } = extension(recorder, {
		tools: [{ name: "lookup", label: "", description: "", parameters: {}, execute: async () => ({ result: "found", injectionOutcomes: [] }) }],
		hooks: [{ toolName: "get_weather", execute: async () => ({ result: "x", structuredContent: { a: 1 }, injectionOutcomes: [placed] }) }],
		reportTools: ["read"],
	});
	await tools[0].execute("call-1", {});
	const content = [{ type: "text", text: "body" }];
	const [hooked] = await emit("tool_result", { toolName: "get_weather", toolCallId: "call-2", input: {}, content });
	await emit("tool_result", { toolName: "read", toolCallId: "call-3", input: {}, content, structuredContent: { lines: 1 } });

	// The agent still gets the hook's output; only the record is the text alone.
	assert.deepEqual(hooked, { content: [{ type: "text", text: "x" }], structuredContent: { a: 1 } });
	assert.deepEqual(records.map(([kind, call]) => [kind, call.function, Object.keys(call).sort()]), [
		["start", "lookup", ["args", "callId", "disposition", "function"]],
		["finish", "lookup", ["args", "callId", "disposition", "error", "function", "result"]],
		["finish", "get_weather", ["args", "callId", "disposition", "error", "function", "result"]],
		["finish", "read", ["args", "callId", "disposition", "error", "function", "result"]],
	]);
});

test("by default the extension sends the detailed finish to the control plane", async () => {
	await withFetch([201], async (sent) => {
		const handlers = {};
		createMidojoExtension({
			controlPlaneUrl: "http://control",
			hooks: [{ toolName: "get_weather", execute: async () => ({ result: "x", injectionOutcomes: [placed] }) }],
		})({ on: (event, callback) => (handlers[event] = callback) });
		await handlers.tool_result({ toolName: "get_weather", toolCallId: "call-1", input: {}, content: [] });
		const body = JSON.parse(sent[0][1]);
		assert.equal(body.structured_content, null);
		assert.deepEqual(body.injection_outcomes, [
			{
				instruction_index: 0,
				probe_key: "p:main",
				occurrence: 1,
				status: "applied",
				detail: "hook placed the payload",
				selected: true,
				transformation: "identity",
				emitted: ["content_block", "structured_content"],
			},
		]);
	});
});
