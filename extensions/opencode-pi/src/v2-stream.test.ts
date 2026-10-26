import assert from "node:assert/strict";
import type { ChildProcess } from "node:child_process";
import { createServer, type ServerResponse } from "node:http";
import test from "node:test";
import type { Api, AssistantMessage, Model } from "@earendil-works/pi-ai";
import {
  createMarkerStreamFilter,
  createV2EventMapper,
  parseToolCallResponse,
  runV2ServerTurn,
  v2PromptBody,
  v2SessionBody,
  type OpenCodeStreamSink,
  type OpenCodeTurnState,
} from "./index.js";
import { EventStreamClosedBeforeConnect, type OpenCodeServer } from "./server.js";

const SESSION = "ses_probe";

function freshState(): OpenCodeTurnState {
  return {
    text: "",
    thinking: "",
    stderr: "",
    sessionId: SESSION,
    finishReason: "stop",
    timedOut: false,
  };
}

function collector(): {
  sink: OpenCodeStreamSink;
  text: Array<[string, string]>;
  thinking: Array<[string, string]>;
} {
  const text: Array<[string, string]> = [];
  const thinking: Array<[string, string]> = [];
  return {
    text,
    thinking,
    sink: {
      text: (partId, delta) => text.push([partId, delta]),
      thinking: (partId, delta) => thinking.push([partId, delta]),
    },
  };
}

function part(ordinal = 0, message = "msg_1") {
  return { sessionID: SESSION, assistantMessageID: message, ordinal };
}

function event(type: string, data: Record<string, unknown> = {}) {
  return { type, data };
}

function askEvent(sessionID: string, n: number) {
  return {
    type: "permission.asked",
    data: { sessionID, id: "per_" + n, action: "read", resources: [] },
  };
}

test("v2 text deltas stream in order and the closing event does not repeat them", () => {
  const state = freshState();
  const { sink, text } = collector();
  const mapper = createV2EventMapper({ state, sink });

  mapper.handle(event("session.text.started", part()));
  mapper.handle(event("session.text.delta", { ...part(), delta: "Hello " }));
  mapper.handle(event("session.text.delta", { ...part(), delta: "world" }));

  // The full text is already on screen; the closing event must not append it again.
  mapper.handle(event("session.text.ended", { ...part(), text: "Hello world" }));

  assert.deepEqual(text, [
    ["msg_1:0", "Hello "],
    ["msg_1:0", "world"],
  ]);
  assert.equal(state.text, "Hello world");
});

test("v2 reasoning deltas go to the thinking sink and never to text", () => {
  const state = freshState();
  const { sink, text, thinking } = collector();
  const mapper = createV2EventMapper({ state, sink });

  mapper.handle(event("session.reasoning.started", part()));
  mapper.handle(event("session.reasoning.delta", { ...part(), delta: "let me " }));
  mapper.handle(event("session.reasoning.delta", { ...part(), delta: "think" }));
  mapper.handle(event("session.reasoning.ended", { ...part(), text: "let me think" }));

  assert.deepEqual(thinking, [
    ["msg_1:0", "let me "],
    ["msg_1:0", "think"],
  ]);
  assert.deepEqual(text, []);
  assert.equal(state.thinking, "let me think");
  assert.equal(state.text, "");
});

test("a reply delivered only through the closing event is not lost", () => {
  const state = freshState();
  const { sink, text } = collector();
  const mapper = createV2EventMapper({ state, sink });

  mapper.handle(event("session.text.started", part()));
  mapper.handle(event("session.text.ended", { ...part(), text: "all at once" }));

  assert.deepEqual(text, [["msg_1:0", "all at once"]]);
  assert.equal(state.text, "all at once");
});

test("ordinals become separate content blocks", () => {
  const state = freshState();
  const { sink, text } = collector();
  const mapper = createV2EventMapper({ state, sink });

  mapper.handle(event("session.text.delta", { ...part(0), delta: "first" }));
  mapper.handle(event("session.text.delta", { ...part(1), delta: "second" }));

  assert.deepEqual(text, [
    ["msg_1:0", "first"],
    ["msg_1:1", "second"],
  ]);
  assert.equal(state.text, "firstsecond");
});

test("events from another session are ignored", () => {
  const state = freshState();
  const { sink, text } = collector();
  const mapper = createV2EventMapper({ state, sink });

  mapper.handle(
    event("session.text.delta", {
      sessionID: "ses_other",
      assistantMessageID: "msg_9",
      ordinal: 0,
      delta: "not mine",
    }),
  );

  assert.deepEqual(text, []);
  assert.equal(state.text, "");
});

test("a marker split across v2 deltas never reaches the visible sink", () => {
  const state = freshState();
  const filter = createMarkerStreamFilter();
  const visible: string[] = [];
  let closed = false;
  const sink: OpenCodeStreamSink = {
    text: (_partId, delta) => {
      const out = filter.push(delta);
      if (out) visible.push(out);
    },
    thinking: () => undefined,
  };
  const mapper = createV2EventMapper({ state, sink });

  for (const chunk of [
    "<pi_",
    'tool_call>{"name":"read",',
    '"arguments":{}}</pi_tool_call>',
  ]) {
    mapper.handle(event("session.text.delta", { ...part(), delta: chunk }));
  }
  closed = mapper.handle(event("session.execution.succeeded", { sessionID: SESSION }));

  assert.equal(visible.join(""), "");
  assert.equal(filter.text, '<pi_tool_call>{"name":"read","arguments":{}}</pi_tool_call>');
  assert.equal(closed, true);
});

test("splitting the same reply into single characters changes nothing visible", () => {
  const reply = 'Checking. <pi_tool_call>{"name":"read","arguments":{}}</pi_tool_call> Done.';

  function run(chunks: string[]): string {
    const state = freshState();
    const filter = createMarkerStreamFilter();
    const visible: string[] = [];
    const sink: OpenCodeStreamSink = {
      text: (_partId, delta) => {
        const out = filter.push(delta);
        if (out) visible.push(out);
      },
      thinking: () => undefined,
    };
    const mapper = createV2EventMapper({ state, sink });
    for (const chunk of chunks) {
      mapper.handle(event("session.text.delta", { ...part(), delta: chunk }));
    }
    return visible.join("");
  }

  const whole = run([reply]);
  const bytes = run([...reply]);

  assert.equal(whole, bytes);
  assert.ok(!bytes.includes("<pi_tool_call>"));
  assert.ok(bytes.startsWith("Checking. "));
  assert.ok(bytes.endsWith(" Done."));
});

test("a length finish is reported as a length stop", () => {
  const state = freshState();
  const { sink } = collector();
  const mapper = createV2EventMapper({ state, sink });

  mapper.handle(event("session.step.ended", { ...part(), finish: "length" }));

  assert.equal(state.finishReason, "length");

  const other = freshState();
  const second = createV2EventMapper({ state: other, sink });
  second.handle(event("session.step.ended", { ...part(), finish: "stop" }));
  assert.equal(other.finishReason, "stop");
});

test("only an execution terminal event closes the turn", () => {
  const state = freshState();
  const { sink } = collector();
  const mapper = createV2EventMapper({ state, sink });

  assert.equal(mapper.handle(event("session.step.ended", { ...part(), finish: "stop" })), false);
  assert.equal(mapper.handle(event("session.usage.updated", { ...part() })), false);
  assert.equal(mapper.handle(event("session.execution.failed", { sessionID: SESSION })), true);
  assert.equal(mapper.handle(event("session.execution.interrupted", { sessionID: SESSION })), true);
});

test("a failed v2 execution surfaces the server's reason instead of an empty turn", () => {
  const state = freshState();
  const { sink } = collector();
  const mapper = createV2EventMapper({ state, sink });

  const closed = mapper.handle(
    event("session.execution.failed", {
      sessionID: SESSION,
      error: { type: "unknown", message: 'Agent not found: "pi-model"' },
    }),
  );

  assert.equal(closed, true);
  assert.equal(state.error, 'Agent not found: "pi-model"');
  assert.match(state.stderr, /Agent not found/);
});

// Reply shapes the v1 parser is pinned on. The v2 path must rebuild every one
// of them byte for byte, whatever the provider's chunking happens to be.
const FIXTURES: Array<{ name: string; text: string }> = [
  { name: "plain prose", text: "Just a normal answer." },
  {
    name: "one valid marker",
    text: 'Before <pi_tool_call>{"name":"read","arguments":{"path":"README.md"}}</pi_tool_call> after',
  },
  {
    name: "valid call plus malformed sibling",
    text:
      '<pi_tool_call>{"name":"read","arguments":{"path":"README.md"}}</pi_tool_call>' +
      "<pi_tool_call>{not json}</pi_tool_call>",
  },
  { name: "every payload invalid", text: "<pi_tool_call>{not json}</pi_tool_call>" },
  {
    name: "quoted marker",
    text: '"<pi_tool_call>{\\"name\\":\\"bash\\",\\"arguments\\":{}}</pi_tool_call>"',
  },
  {
    name: "array payload",
    text: '<pi_tool_call>[{"name":"read","arguments":{"path":"a"}}]</pi_tool_call>',
  },
  {
    name: "bounded closing tag",
    text: '<pi_tool_call>{"name":"bash","arguments":{}}</pi_tool_call >',
  },
  {
    name: "marker followed by prose",
    text:
      '<pi_tool_call>{"name":"bash","arguments":{"command":"pwd"}}</pi_tool_call>' +
      "\nDone.",
  },
];

const CHUNKINGS: Array<{ name: string; split: (text: string) => string[] }> = [
  { name: "whole", split: (text) => [text] },
  { name: "characters", split: (text) => [...text] },
  { name: "pairs", split: (text) => text.match(/[\s\S]{1,2}/g) ?? [] },
  {
    name: "halves",
    split: (text) => {
      const cut = Math.ceil(text.length / 2);
      return [text.slice(0, cut), text.slice(cut)];
    },
  },
];

function accumulate(chunks: string[]): OpenCodeTurnState {
  const state = freshState();
  const { sink } = collector();
  const mapper = createV2EventMapper({ state, sink });
  for (const chunk of chunks) {
    mapper.handle(event("session.text.delta", { ...part(), delta: chunk }));
  }
  return state;
}

test("every v2 chunking reconstructs the reply exactly", () => {
  for (const fixture of FIXTURES) {
    for (const chunk of CHUNKINGS) {
      assert.equal(
        accumulate(chunk.split(fixture.text)).text,
        fixture.text,
        `${fixture.name} / ${chunk.name}`,
      );
    }
  }
});

test("v2 accumulated text parses exactly like the v1 path", () => {
  const allowed = new Set(["read", "bash"]);
  for (const fixture of FIXTURES) {
    const expected = parseToolCallResponse(fixture.text, allowed);
    for (const chunk of CHUNKINGS) {
      assert.deepEqual(
        parseToolCallResponse(accumulate(chunk.split(fixture.text)).text, allowed),
        expected,
        `${fixture.name} / ${chunk.name}`,
      );
    }
  }

  // One concrete outcome, so parity cannot pass with both sides empty.
  const marker = '<pi_tool_call>{"name":"read","arguments":{"path":"README.md"}}</pi_tool_call>';
  const parsed = parseToolCallResponse(accumulate([...marker]).text, allowed);
  assert.equal(parsed.ok, true);
  assert.deepEqual(parsed.calls, [{ name: "read", arguments: { path: "README.md" } }]);
});

test("the visible text is identical whether the reply arrives whole or chunked", () => {
  function visible(chunks: string[]): string {
    const state = freshState();
    const filter = createMarkerStreamFilter();
    const out: string[] = [];
    const sink: OpenCodeStreamSink = {
      text: (_partId, delta) => {
        const shown = filter.push(delta);
        if (shown) out.push(shown);
      },
      thinking: () => undefined,
    };
    const mapper = createV2EventMapper({ state, sink });
    for (const chunk of chunks) {
      mapper.handle(event("session.text.delta", { ...part(), delta: chunk }));
    }
    out.push(filter.finish(""));
    return out.join("");
  }

  for (const fixture of FIXTURES) {
    const whole = visible([fixture.text]);
    assert.ok(!whole.includes("<pi_tool_call>"), `${fixture.name} leaked a marker`);
    for (const chunk of CHUNKINGS) {
      assert.equal(
        visible(chunk.split(fixture.text)),
        whole,
        `${fixture.name} / ${chunk.name}`,
      );
    }
  }
});

// ---------------------------------------------------------------------------
// T4: the v2 wire. A real HTTP server in this process speaks the v2 subset,
// so the turn is exercised end to end without spawning a binary.
// ---------------------------------------------------------------------------

type Captured = { method: string; path: string; directory?: string | null; body?: any };

function bridgeModel(): Model<Api> {
  return {
    id: "opencode/deepseek-test",
    name: "Deepseek Test",
    api: "opencode-cli-runner",
    provider: "opencode-cli",
    baseUrl: "",
    reasoning: true,
    input: ["text", "image"],
    cost: { input: 1, output: 2, cacheRead: 0.5, cacheWrite: 0.25 },
    contextWindow: 128_000,
    maxTokens: 16_384,
  };
}

function emptyOutput(model: Model<Api>): AssistantMessage {
  return {
    role: "assistant",
    content: [],
    api: model.api,
    provider: model.provider,
    model: model.id,
    usage: {
      input: 0,
      output: 0,
      cacheRead: 0,
      cacheWrite: 0,
      totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    },
    stopReason: "stop",
    timestamp: 0,
  };
}

async function withV2Server(
  events: (
    sessionID: string,
    promptIndex: number,
  ) => Array<{ type: string; data: Record<string, unknown> }>,
  run: (
    server: OpenCodeServer,
    seen: Captured[],
    stats: { closes: number },
  ) => Promise<void>,
  options: { sendConnected?: boolean } = {},
): Promise<void> {
  const seen: Captured[] = [];
  const stats = { closes: 0 };
  const clients = new Set<ServerResponse>();
  let promptIndex = 0;
  const http = createServer((req, res) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
    });
    req.on("end", () => {
      const url = new URL(req.url ?? "/", "http://fake");
      if (url.pathname === "/api/event") {
        res.writeHead(200, { "content-type": "text/event-stream" });
        clients.add(res);
        res.on("close", () => {
          clients.delete(res);
          stats.closes += 1;
        });
        if (options.sendConnected !== false) {
          res.write(
            "data: " + JSON.stringify({ type: "server.connected", data: {} }) + "\n\n",
          );
        } else {
          // Close immediately to simulate stream closing before server.connected
          res.end();
        }
        return;
      }
      seen.push({
        method: req.method ?? "",
        path: url.pathname,
        directory: url.searchParams.get("directory"),
        body: body ? JSON.parse(body) : undefined,
      });
      if (url.pathname === "/api/session") {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ data: { id: "ses_fake" } }));
        return;
      }
      const prompt = /^\/api\/session\/([^/]+)\/prompt$/.exec(url.pathname);
      if (prompt) {
        res.writeHead(200, { "content-type": "application/json" });
        res.end(JSON.stringify({ data: {} }));
        for (const event of events(prompt[1]!, promptIndex++)) {
          for (const client of clients) {
            client.write("data: " + JSON.stringify(event) + "\n\n");
          }
        }
        return;
      }
      res.writeHead(200, { "content-type": "application/json" });
      res.end("true");
    });
  });
  await new Promise<void>((resolve) => http.listen(0, "127.0.0.1", resolve));
  const address = http.address();
  const port = typeof address === "object" && address ? address.port : 0;
  try {
    await run(
      {
        bin: "fake",
        prefixArgs: [],
        url: "http://127.0.0.1:" + port,
        authorization: "Basic fake",
        child: {} as ChildProcess,
        pure: false,
      },
      seen,
      stats,
    );
  } finally {
    for (const client of clients) client.end();
    await new Promise<void>((resolve) => http.close(() => resolve()));
  }
}

test("v2SessionBody names the bridge agent, the model ref and the variant", () => {
  assert.deepEqual(v2SessionBody({ id: "opencode/deepseek-v4.1-flash-free" }), {
    agent: "pi-model",
    model: { providerID: "opencode", id: "deepseek-v4.1-flash-free" },
  });
  assert.deepEqual(v2SessionBody({ id: "deepseek-v4.1-flash-free" }), {
    agent: "pi-model",
    model: { providerID: "opencode", id: "deepseek-v4.1-flash-free" },
  });
  // A `#variant` suffix that OpenCode's CLI accepts moves into the model ref.
  assert.deepEqual(v2SessionBody({ id: "opencode/deepseek#high" }), {
    agent: "pi-model",
    model: { providerID: "opencode", id: "deepseek", variant: "high" },
  });
  // An explicit variant wins over the inline one.
  assert.deepEqual(v2SessionBody({ id: "opencode/deepseek#high" }, "low"), {
    agent: "pi-model",
    model: { providerID: "opencode", id: "deepseek", variant: "low" },
  });
});

test("v2SessionBody names the project directory, which the query cannot", () => {
  assert.deepEqual(v2SessionBody({ id: "opencode/deepseek" }, undefined, "/tmp/project"), {
    agent: "pi-model",
    model: { providerID: "opencode", id: "deepseek" },
    location: { directory: "/tmp/project" },
  });
});

test("v2PromptBody sends files only when there are files", () => {
  assert.deepEqual(v2PromptBody("hi"), { text: "hi" });
  assert.deepEqual(v2PromptBody("hi", [{ uri: "file:///a.png", name: "a.png" }]), {
    text: "hi",
    files: [{ uri: "file:///a.png", name: "a.png" }],
  });
});

test("a v2 turn creates the session with the bridge agent and streams its reply", async () => {
  const model = bridgeModel();
  const output = emptyOutput(model);
  const { sink, text } = collector();
  const sessions: string[] = [];

  await withV2Server(
    (sessionID) => [
      {
        type: "session.step.started",
        data: { sessionID, agent: "pi-model", assistantMessageID: "msg_v2" },
      },
      {
        type: "session.text.started",
        data: { sessionID, assistantMessageID: "msg_v2", ordinal: 0 },
      },
      {
        type: "session.text.delta",
        data: { sessionID, assistantMessageID: "msg_v2", ordinal: 0, delta: "hello " },
      },
      {
        type: "session.text.delta",
        data: { sessionID, assistantMessageID: "msg_v2", ordinal: 0, delta: "world" },
      },
      {
        type: "session.text.ended",
        data: {
          sessionID,
          assistantMessageID: "msg_v2",
          ordinal: 0,
          text: "hello world",
        },
      },
      {
        type: "session.step.ended",
        data: {
          sessionID,
          assistantMessageID: "msg_v2",
          finish: "stop",
          cost: 0,
          tokens: { input: 65, output: 10, reasoning: 4, cache: { read: 6272, write: 0 } },
        },
      },
      { type: "session.execution.succeeded", data: { sessionID } },
    ],
    async (server, seen) => {
      const turn = await runV2ServerTurn({
        server,
        directory: "/tmp/project",
        prompt: "hi",
        files: [{ uri: "file:///tmp/one.png", name: "one.png" }],
        timeoutMs: 5000,
        model,
        output,
        sink,
        onSessionId: (id) => sessions.push(id),
      });

      assert.equal(turn.code, 0);
      assert.equal(turn.state.sessionId, "ses_fake");
      assert.equal(turn.state.text, "hello world");
      assert.equal(turn.state.error, undefined);

      assert.deepEqual(
        seen.map((entry) => entry.path),
        ["/api/session", "/api/session/ses_fake/prompt"],
      );
      assert.deepEqual(seen[0]!.body, {
        agent: "pi-model",
        model: { providerID: "opencode", id: "deepseek-test" },
        location: { directory: "/tmp/project" },
      });
      assert.deepEqual(seen[1]!.body, {
        text: "hi",
        files: [{ uri: "file:///tmp/one.png", name: "one.png" }],
      });
      assert.equal(seen[0]!.directory, "/tmp/project");
    },
  );

  assert.deepEqual(text, [
    ["msg_v2:0", "hello "],
    ["msg_v2:0", "world"],
  ]);
  assert.deepEqual(sessions, ["ses_fake"]);
  assert.equal(output.usage.input, 65);
  assert.equal(output.usage.output, 14, "reasoning tokens count as output");
  assert.equal(output.usage.cacheRead, 6272);
  const expected = (65 * 1 + 14 * 2 + 6272 * 0.5) / 1_000_000;
  assert.ok(Math.abs(output.usage.cost.total - expected) < 1e-12);
});

test("a v2 turn that runs under another agent fails instead of answering", async () => {
  const model = bridgeModel();
  const output = emptyOutput(model);
  const { sink } = collector();

  await withV2Server(
    (sessionID) => [
      {
        type: "session.step.started",
        data: { sessionID, agent: "build", assistantMessageID: "msg_v2" },
      },
      {
        type: "session.text.delta",
        data: { sessionID, assistantMessageID: "msg_v2", ordinal: 0, delta: "no bridge" },
      },
      { type: "session.execution.succeeded", data: { sessionID } },
    ],
    async (server) => {
      const turn = await runV2ServerTurn({
        server,
        directory: "/tmp/project",
        prompt: "hi",
        timeoutMs: 5000,
        model,
        output,
        sink,
      });
      assert.match(turn.state.error ?? "", /"build".*"pi-model"/);
      assert.equal(turn.state.text, "", "the turn stops before the wrong agent answers");
    },
  );
});

test("a continued v2 turn prompts the recorded session and creates no new one", async () => {
  const model = bridgeModel();
  const output = emptyOutput(model);
  const { sink } = collector();

  await withV2Server(
    (sessionID) => [
      {
        type: "session.step.started",
        data: { sessionID, agent: "pi-model", assistantMessageID: "msg_v2" },
      },
      { type: "session.execution.succeeded", data: { sessionID } },
    ],
    async (server, seen) => {
      const turn = await runV2ServerTurn({
        server,
        directory: "/tmp/project",
        sessionId: "ses_recorded",
        prompt: "again",
        timeoutMs: 5000,
        model,
        output,
        sink,
      });
      assert.equal(turn.code, 0);
      assert.equal(turn.state.sessionId, "ses_recorded");
      assert.deepEqual(
        seen.map((entry) => entry.path),
        ["/api/session/ses_recorded/prompt"],
      );
    },
  );
});

// Regression from real use on a weak model: `parseToolCallResponse` recovers a
// complete payload whose closing tag never arrived and the turn then uses that
// call, so the filter must not report the same bytes as dropped text.
test("a recovered marker without its closing tag is not reported as dropped text", () => {
  const filter = createMarkerStreamFilter();
  const shown = filter.push(
    'Need this. <pi_tool_call>{"name":"bash","arguments":{"command":"echo hi"}}',
  );

  assert.equal(shown, "Need this. ", "prose before the marker still streams");
  assert.equal(
    filter.finish(""),
    "",
    "the payload was recovered and used, so nothing was dropped",
  );
});

// S6: the reply used to be cut where an unclosed marker began, and the tail
// was dropped with no trace. Prose before the marker is already released, so
// the only honest ending is a visible note saying the rest was not shown.
test("an unfinished tool-call block ends the reply with a visible note", () => {
  const filter = createMarkerStreamFilter();

  assert.equal(filter.push("El texto visible sigue completo."), "El texto visible sigue completo.");
  filter.push('<pi_tool_call>{"name":"read","argu');
  const tail = filter.finish("");

  assert.match(tail, /unfinished tool-call block/, "the drop is explained, not silent");
  assert.ok(
    !tail.includes("<pi_tool_call>"),
    "the note never leaks the raw marker",
  );
});

// S6: a reply that carries one valid call and one unparseable block keeps the
// call, but the removed block is reported instead of vanishing.
test("an unparseable tool-call block is reported instead of silently removed", () => {
  const allowed = new Set(["read"]);
  const text = [
    '<pi_tool_call>{"name":"read","arguments":{"path":"README.md"}}</pi_tool_call>',
    '<pi_tool_call>{"name": broken</pi_tool_call>',
  ].join("\n");

  const parsed = parseToolCallResponse(text, allowed);

  assert.equal(parsed.ok, true);
  if (!parsed.ok) return;
  assert.deepEqual(parsed.calls, [{ name: "read", arguments: { path: "README.md" } }]);
  assert.match(
    parsed.cleanedText ?? "",
    /could not be parsed/,
    "the reply says a block was removed instead of ending mid-sentence",
  );
});

test("a v2 permission ask is rejected with the marker instruction and the turn survives", async () => {
  const model = bridgeModel();
  const output = emptyOutput(model);
  const { sink, text } = collector();

  await withV2Server(
    (sessionID) => [
      {
        type: "session.step.started",
        data: { sessionID, agent: "pi-model", assistantMessageID: "msg_v2" },
      },
      {
        type: "permission.asked",
        data: { sessionID, id: "per_1", action: "read", resources: ["/etc/hosts"] },
      },
      {
        type: "session.tool.failed",
        data: {
          sessionID,
          id: "per_1",
          name: "read",
          error: { type: "permission.rejected", message: "declined" },
        },
      },
      // v2 starts a new step by itself after a rejection that carries a message.
      {
        type: "session.step.ended",
        data: { sessionID, assistantMessageID: "msg_v2", finish: "stop" },
      },
      {
        type: "session.step.started",
        data: { sessionID, agent: "pi-model", assistantMessageID: "msg_v2b" },
      },
      {
        type: "session.text.delta",
        data: { sessionID, assistantMessageID: "msg_v2b", ordinal: 0, delta: "marker instead" },
      },
      { type: "session.execution.succeeded", data: { sessionID } },
    ],
    async (server, seen) => {
      const turn = await runV2ServerTurn({
        server,
        directory: "/tmp/project",
        prompt: "read the file",
        timeoutMs: 5000,
        model,
        output,
        sink,
      });

      assert.equal(turn.code, 0);
      assert.equal(turn.state.text, "marker instead");
      assert.equal(turn.state.toolUse, undefined, "a refusal is not tool use");
      assert.equal(turn.state.error, undefined);

      const reply = seen.find((entry) => entry.path.endsWith("/reply"));
      assert.equal(reply?.path, "/api/session/ses_fake/permission/per_1/reply");
      assert.equal(reply?.body?.decision, "reject");
      assert.match(
        String(reply?.body?.message),
        /<pi_tool_call>\{"name":"\.\.\.","arguments":\{\}\}<\/pi_tool_call>/,
        "a bare reject would abort the execution",
      );
    },
  );

  assert.deepEqual(text, [["msg_v2b:0", "marker instead"]]);
});

// S5: the ask cap used to interrupt the turn outright. A model that keeps
// reaching for OpenCode's disabled native tools now gets a corrective prompt
// inside the same session, bounded by OPENCODE_PI_TOOL_RESCUES, and only an
// exhausted budget interrupts.
test("a v2 turn that exceeds the ask cap is rescued by an in-session correction", async () => {
  const model = bridgeModel();
  const output = emptyOutput(model);
  const { sink } = collector();

  await withV2Server(
    (sessionID, promptIndex) =>
      promptIndex === 0
        ? [
            {
              type: "session.step.started",
              data: { sessionID, agent: "pi-model", assistantMessageID: "msg_v2" },
            },
            askEvent(sessionID, 1),
            askEvent(sessionID, 2),
            askEvent(sessionID, 3),
            askEvent(sessionID, 4),
          ]
        : [
            {
              type: "session.step.started",
              data: { sessionID, agent: "pi-model", assistantMessageID: "msg_rescue" },
            },
            {
              type: "session.text.delta",
              data: {
                sessionID,
                assistantMessageID: "msg_rescue",
                ordinal: 0,
                delta:
                  '{"name":"read","arguments":{"path":"README.md"}}',
              },
            },
            { type: "session.execution.succeeded", data: { sessionID } },
          ],
    async (server, seen) => {
      const turn = await runV2ServerTurn({
        server,
        directory: "/tmp/project",
        prompt: "keep calling native tools",
        timeoutMs: 5000,
        model,
        output,
        sink,
      });

      assert.equal(turn.code, 0);
      assert.equal(turn.state.error, undefined, "a rescue keeps the turn alive");
      assert.equal(
        seen.filter((entry) => entry.path.endsWith("/interrupt")).length,
        0,
        "nothing is interrupted while the rescue budget lasts",
      );
      const prompts = seen.filter((entry) => /\/prompt$/.test(entry.path));
      assert.equal(
        prompts.length,
        2,
        "the initial prompt plus one corrective prompt in the same session",
      );
      assert.match(
        JSON.stringify(prompts[1]?.body),
        /pi_tool_call/,
        "the corrective prompt restates the marker protocol",
      );
      assert.match(turn.state.text, /"name":"read"/, "the correction produced a payload");
    },
  );
});

test("a v2 turn stops asking only after the rescue budget is spent", async () => {
  const model = bridgeModel();
  const output = emptyOutput(model);
  const { sink } = collector();

  const capBatch = (sessionID: string) => [
    {
      type: "session.step.started",
      data: { sessionID, agent: "pi-model", assistantMessageID: "msg_v2" },
    },
    askEvent(sessionID, 1),
    askEvent(sessionID, 2),
    askEvent(sessionID, 3),
    askEvent(sessionID, 4),
  ];

  await withV2Server(
    (sessionID) => capBatch(sessionID),
    async (server, seen) => {
      const turn = await runV2ServerTurn({
        server,
        directory: "/tmp/project",
        prompt: "keep calling native tools",
        timeoutMs: 5000,
        model,
        output,
        sink,
      });

      assert.equal(turn.code, 0);
      assert.equal(
        seen.filter((entry) => /\/prompt$/.test(entry.path)).length,
        3,
        "the initial prompt plus two corrective prompts",
      );
      assert.equal(
        seen.filter((entry) => entry.path.endsWith("/interrupt")).length,
        1,
        "an exhausted budget interrupts instead of looping",
      );
      assert.match(
        turn.state.error ?? "",
        /12 times/,
        "the diagnostic counts asks across every rescued step",
      );
    },
  );
});

test("a v2 turn that keeps asking for native tools is stopped at the cap", async () => {
  const model = bridgeModel();
  const output = emptyOutput(model);
  const { sink } = collector();

  const previous = process.env.OPENCODE_PI_TOOL_RESCUES;
  process.env.OPENCODE_PI_TOOL_RESCUES = "0";
  try {
    await withV2Server(
      (sessionID) => [
        {
          type: "session.step.started",
          data: { sessionID, agent: "pi-model", assistantMessageID: "msg_v2" },
        },
        askEvent(sessionID, 1),
        askEvent(sessionID, 2),
        askEvent(sessionID, 3),
        askEvent(sessionID, 4),
        { type: "session.execution.succeeded", data: { sessionID } },
      ],
      async (server, seen) => {
        const turn = await runV2ServerTurn({
          server,
          directory: "/tmp/project",
          prompt: "keep calling native tools",
          timeoutMs: 5000,
          model,
          output,
          sink,
        });

        assert.equal(turn.code, 0);
        assert.match(turn.state.error ?? "", /4 times/);
        assert.match(turn.state.error ?? "", /marker/);
        assert.equal(
          seen.filter((entry) => entry.path.endsWith("/reply")).length,
          3,
          "only three asks are answered; the fourth stops the turn",
        );
        assert.equal(
          seen.filter((entry) => entry.path.endsWith("/interrupt")).length,
          1,
          "the stalled execution is interrupted",
        );
      },
    );
  } finally {
    if (previous === undefined) delete process.env.OPENCODE_PI_TOOL_RESCUES;
    else process.env.OPENCODE_PI_TOOL_RESCUES = previous;
  }
});

// D1: `called` fires before the permission decision, so only a success or a
// failure that is not a refusal proves a native tool ran.
test("a v2 tool that ran fails the turn while a refused one does not", () => {
  const state = freshState();
  const { sink } = collector();
  const mapper = createV2EventMapper({ state, sink });

  mapper.handle(
    event("session.tool.failed", {
      sessionID: SESSION,
      name: "read",
      error: { type: "permission.rejected", message: "declined" },
    }),
  );
  assert.equal(state.toolUse, undefined, "a refused tool changed nothing");

  mapper.handle(event("session.tool.called", { sessionID: SESSION, name: "shell" }));
  assert.equal(state.toolUse, undefined, "`called` proves nothing");

  mapper.handle(event("session.tool.success", { sessionID: SESSION, name: "shell" }));
  assert.equal(state.toolUse, "shell", "a tool that ran fails the turn");
});

// T6: a turn that never finishes must be bounded by the caller's abort or the
// turn timeout, and both paths must interrupt the execution and close the
// subscription instead of leaving a reader attached to a dead server.
test("an aborted v2 turn interrupts the execution, keeps partial text and closes the subscription", async () => {
  const model = bridgeModel();
  const output = emptyOutput(model);
  const { sink, text } = collector();
  const controller = new AbortController();

  await withV2Server(
    (sessionID) => [
      {
        type: "session.step.started",
        data: { sessionID, agent: "pi-model", assistantMessageID: "msg_v2" },
      },
      {
        type: "session.text.delta",
        data: { sessionID, assistantMessageID: "msg_v2", ordinal: 0, delta: "partial" },
      },
      // No terminal execution event: the turn hangs until it is stopped.
    ],
    async (server, seen, stats) => {
      const timer = setTimeout(() => controller.abort(), 150);
      const turn = await runV2ServerTurn({
        server,
        directory: "/tmp/project",
        prompt: "hang",
        signal: controller.signal,
        timeoutMs: 10000,
        model,
        output,
        sink,
      });
      clearTimeout(timer);

      assert.equal(turn.code, null);
      assert.equal(turn.state.timedOut, false);
      assert.equal(turn.state.text, "partial", "what already streamed is kept");
      assert.deepEqual(text, [["msg_v2:0", "partial"]]);
      assert.equal(
        seen.filter((entry) => entry.path.endsWith("/interrupt")).length,
        1,
        "the hanging execution is interrupted",
      );

      // The SSE reader releases its socket shortly after the turn returns.
      for (let attempt = 0; attempt < 20 && stats.closes === 0; attempt += 1) {
        await new Promise((resolve) => setTimeout(resolve, 25));
      }
      assert.ok(stats.closes >= 1, "the event subscription is cancelled");
    },
  );
});

test("a v2 turn that never finishes is bounded by its timeout", async () => {
  const model = bridgeModel();
  const output = emptyOutput(model);
  const { sink } = collector();

  await withV2Server(
    (sessionID) => [
      {
        type: "session.step.started",
        data: { sessionID, agent: "pi-model", assistantMessageID: "msg_v2" },
      },
    ],
    async (server, seen) => {
      const turn = await runV2ServerTurn({
        server,
        directory: "/tmp/project",
        prompt: "never finishes",
        timeoutMs: 150,
        model,
        output,
        sink,
      });

      assert.equal(turn.code, null);
      assert.equal(turn.state.timedOut, true);
      assert.equal(
        seen.filter((entry) => entry.path.endsWith("/interrupt")).length,
        1,
        "the timeout interrupts the execution instead of leaving it running",
      );
    },
  );
});

// T8: v2 stream closes before server.connected -> fallback to opencode run.
// This tests the V2StreamClosedBeforeConnected error is thrown when the
// event stream ends without ever sending server.connected.
test("v2 stream closed before server.connected throws V2StreamClosedBeforeConnected", async () => {
  const model = bridgeModel();
  const output = emptyOutput(model);
  const { sink } = collector();

  await withV2Server(
    (sessionID) => [
      // No server.connected event - stream ends immediately after subscription
      {
        type: "session.step.started",
        data: { sessionID, agent: "pi-model", assistantMessageID: "msg_v2" },
      },
    ],
    async (server) => {
      let threw = false;
      try {
        await runV2ServerTurn({
          server,
          directory: "/tmp/project",
          prompt: "will fall back",
          timeoutMs: 5000,
          model,
          output,
          sink,
        });
      } catch (error) {
        threw = true;
        assert.ok(error instanceof EventStreamClosedBeforeConnect);
        assert.match(String(error), /closed before it connected/);
      }
      assert.equal(threw, true, "must throw EventStreamClosedBeforeConnect");
    },
    { sendConnected: false },
  );
});