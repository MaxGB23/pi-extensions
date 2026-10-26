import assert from "node:assert/strict";
import {
  chmodSync,
  existsSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import test, { mock } from "node:test";
import type {
  Api,
  AssistantMessageEvent,
  Context,
  Model,
  SimpleStreamOptions,
} from "@earendil-works/pi-ai";
import {
  cleanupPiSessionState,
  createMarkerStreamFilter,
  streamOpenCode,
} from "./index.js";
import {
  parseEvents,
  runningOpenCodeServer,
  stopOpenCodeServer,
  withOpenCodeServer,
} from "./server.js";

// Keep the model cache out of the real Pi agent directory.
process.env.PI_CODING_AGENT_DIR = mkdtempSync(join(tmpdir(), "opencode-pi-agent-"));
delete process.env.OPENCODE_PI_STREAM;

const TOOL_CALL = '<pi_tool_call>{"name":"read","arguments":{"path":"a.txt"}}</pi_tool_call>';

function fakeModel(): Model<Api> {
  return {
    id: "opencode/fake-free",
    name: "Fake",
    api: "opencode-cli-runner",
    provider: "opencode-cli",
    baseUrl: "",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow: 128_000,
    maxTokens: 16_384,
  };
}

function fakeContext(toolNames: string[] = []): Context {
  return {
    messages: [{ role: "user", content: "hi", timestamp: 1 }],
    tools: toolNames.map((name) => ({
      name,
      description: `${name} tool`,
      parameters: {},
    })) as Context["tools"],
  };
}

type FakeServerConfig = {
  /** Appends one JSON line per server request (except event streams). */
  capturePath?: string;
  /** Returned by POST /session. */
  sessionId?: string;
  /** Server events sent after a prompt; `{ waitFor: path }` pauses until the file exists. */
  events?: unknown[];
  /** Send session.status idle after the events (default true). */
  idle?: boolean;
  /** Permission IDs whose reply answers 404, as if already settled. */
  settledPermissions?: string[];
};

/**
 * A fake `opencode` binary whose `serve` command speaks the subset of the
 * OpenCode v1 server API the bridge uses. Each prompt echoes the prompt as a
 * user message, as OpenCode does, then replays the scripted events with the
 * session ID filled in. Any other command exits with code 1.
 */
function fakeServer(config: FakeServerConfig = {}): string {
  return `const fs = require("node:fs");
const http = require("node:http");
const config = ${JSON.stringify({ ...config, events: config.events ?? [textPart("prt_hello", "hello")] })};
if (process.argv[2] === "session") process.exit(0);
if (process.argv[2] !== "serve") process.exit(1);
const auth = "Basic " + Buffer.from("opencode:" + process.env.OPENCODE_SERVER_PASSWORD).toString("base64");
const clients = new Set();
const record = (entry) => { if (config.capturePath) fs.appendFileSync(config.capturePath, JSON.stringify(entry) + "\\n"); };
const send = (event) => { for (const res of clients) res.write("data: " + JSON.stringify(event) + "\\n\\n"); };
const wait = (file) => new Promise((resolve) => { const check = () => fs.existsSync(file) ? resolve() : setTimeout(check, 5); check(); });
const withSession = (event, sessionID) => {
  const copy = JSON.parse(JSON.stringify(event));
  copy.properties = copy.properties || {};
  if (copy.properties.part) copy.properties.part.sessionID = sessionID;
  else if (copy.properties.info) copy.properties.info.sessionID = sessionID;
  else copy.properties.sessionID = sessionID;
  return copy;
};
http.createServer((req, res) => {
  let body = "";
  req.on("data", (chunk) => { body += chunk; });
  req.on("end", async () => {
    if (req.headers.authorization !== auth) { res.writeHead(401); res.end(); return; }
    const url = new URL(req.url, "http://fake");
    if (url.pathname === "/event") {
      res.writeHead(200, { "content-type": "text/event-stream" });
      clients.add(res);
      res.on("close", () => clients.delete(res));
      res.write("data: " + JSON.stringify({ type: "server.connected", properties: {} }) + "\\n\\n");
      return;
    }
    const entry = { method: req.method, path: url.pathname, directory: url.searchParams.get("directory"), body: body ? JSON.parse(body) : undefined };
    record(entry);
    if (url.pathname === "/session") {
      res.writeHead(200, { "content-type": "application/json" });
      res.end(JSON.stringify({ id: config.sessionId || "ses-fake" }));
      return;
    }
    if ((config.settledPermissions || []).some((id) => url.pathname === "/permission/" + id + "/reply")) {
      res.writeHead(404, { "content-type": "application/json" });
      res.end(JSON.stringify({ _tag: "PermissionNotFoundError" }));
      return;
    }
    const prompt = url.pathname.match(/^\\/session\\/([^/]+)\\/prompt_async$/);
    if (!prompt) {
      res.writeHead(200, { "content-type": "application/json" });
      res.end("true");
      return;
    }
    const sessionID = prompt[1];
    res.writeHead(204);
    res.end();
    send(withSession({ type: "message.updated", properties: { info: { id: "msg_user", role: "user" } } }, sessionID));
    send(withSession({ type: "message.part.updated", properties: { part: { id: "prt_prompt", messageID: "msg_user", type: "text", text: entry.body.parts.at(-1).text } } }, sessionID));
    for (const event of config.events) {
      if (event.waitFor) await wait(event.waitFor);
      else send(withSession(event, sessionID));
    }
    if (config.idle !== false) send(withSession({ type: "session.status", properties: { status: { type: "idle" } } }, sessionID));
  });
}).listen(0, "127.0.0.1", function () {
  console.log("opencode server listening on http://127.0.0.1:" + this.address().port);
});
`;
}

function textPart(id: string, text: string, type = "text") {
  return { type: "message.part.updated", properties: { part: { id, messageID: "msg_reply", type, text, time: { start: 1, end: 2 } } } };
}

function partStart(id: string, type = "text") {
  return { type: "message.part.updated", properties: { part: { id, messageID: "msg_reply", type, text: "", time: { start: 1 } } } };
}

function partDelta(partID: string, delta: string) {
  return { type: "message.part.delta", properties: { partID, field: "text", delta } };
}

/** One text part streamed as the given deltas, then finished. */
function streamedText(id: string, deltas: string[]) {
  return [partStart(id), ...deltas.map((delta) => partDelta(id, delta)), textPart(id, deltas.join(""))];
}

function stepFinish(reason: string, tokens?: unknown) {
  return { type: "message.part.updated", properties: { part: { id: `prt_finish_${reason}`, messageID: "msg_reply", type: "step-finish", reason, tokens } } };
}

type Captured = { method: string; path: string; directory: string; body?: any };

function readCaptured(capturePath: string): Captured[] {
  if (!existsSync(capturePath)) return [];
  return readFileSync(capturePath, "utf8")
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

function prompts(capturePath: string): Captured[] {
  return readCaptured(capturePath).filter((entry) => entry.path.endsWith("/prompt_async"));
}

function restoreEnv(key: string, value: string | undefined): void {
  if (value === undefined) delete process.env[key];
  else process.env[key] = value;
}

async function withFakeOpenCode<T>(
  script: string,
  run: () => Promise<T>,
  version = "opencode v1.19.3",
): Promise<T> {
  const dir = mkdtempSync(join(tmpdir(), "opencode-pi-fake-bin-"));
  const binPath = join(dir, "opencode-fake.js");
  writeFileSync(binPath, `#!/usr/bin/env node\nif (process.argv[2] === "--version") { process.stdout.write(${JSON.stringify(version)}); process.exit(0); }\n${script}`, "utf8");
  chmodSync(binPath, 0o755);

  const previousBin = process.env.OPENCODE_PI_BIN;
  process.env.OPENCODE_PI_BIN = binPath;
  try {
    return await run();
  } finally {
    stopOpenCodeServer();
    restoreEnv("OPENCODE_PI_BIN", previousBin);
    rmSync(dir, { recursive: true, force: true });
  }
}

async function collect(
  script: string,
  context: Context = fakeContext(),
  options?: SimpleStreamOptions,
): Promise<AssistantMessageEvent[]> {
  return withFakeOpenCode(script, async () => {
    const events: AssistantMessageEvent[] = [];
    for await (const event of streamOpenCode(fakeModel(), context, options)) events.push(event);
    return events;
  });
}

function textDeltas(events: AssistantMessageEvent[]): string[] {
  return events.flatMap((event) => (event.type === "text_delta" ? [event.delta] : []));
}

function withCapture<T>(run: (capturePath: string) => Promise<T>): Promise<T> {
  const dir = mkdtempSync(join(tmpdir(), "opencode-pi-capture-"));
  return run(join(dir, "requests.jsonl")).finally(() => rmSync(dir, { recursive: true, force: true }));
}

/** Push `text` through a fresh filter in the given chunks; returns what it showed. */
function filtered(chunks: string[], displayText?: string): string {
  const filter = createMarkerStreamFilter();
  let shown = chunks.map((chunk) => filter.push(chunk)).join("");
  shown += filter.finish(displayText ?? filter.text);
  return shown;
}

function everySplit(text: string): string[][] {
  const splits: string[][] = [[text], [...text]];
  for (let cut = 1; cut < text.length; cut += 1) splits.push([text.slice(0, cut), text.slice(cut)]);
  return splits;
}

test("marker filter releases plain text as it arrives", () => {
  const filter = createMarkerStreamFilter();
  assert.equal(filter.push("Hello "), "Hello ");
  assert.equal(filter.push("world"), "world");
  assert.equal(filter.finish(filter.text), "");
});

test("marker filter never releases a marker, however the reply is split", () => {
  const text = `Let me look.\n${TOOL_CALL}\nDone.`;
  for (const chunks of everySplit(text)) {
    assert.equal(filtered(chunks), "Let me look.\n\nDone.", JSON.stringify(chunks));
  }
});

test("marker filter holds a possible marker start until it is ruled out", () => {
  const filter = createMarkerStreamFilter();
  assert.equal(filter.push("a <pi_to"), "a ");
  assert.equal(filter.push("ol> b"), "<pi_tool> b");
  assert.equal(filter.push(" <"), " ");
  assert.equal(filter.finish(filter.text), "<");
});

test("marker filter replaces an unclosed marker with a visible note at the end of the reply", () => {
  const filter = createMarkerStreamFilter();
  assert.equal(filter.push('Sure. <pi_tool_call>{"name":"read",'), "Sure. ");
  assert.equal(filter.push('"arguments":{}}'), "");
  const tail = filter.finish(filter.text);
  assert.match(tail, /unfinished tool-call block/, "the dropped tail is explained");
  assert.ok(!tail.includes("<pi_tool_call>"), "no marker byte reaches the chat");
});

test("marker filter ignores a closing tag inside a JSON string", () => {
  const text = '<pi_tool_call>{"name":"write","arguments":{"content":"</pi_tool_call>"}}</pi_tool_call>ok';
  for (const chunks of everySplit(text)) assert.equal(filtered(chunks), "ok", JSON.stringify(chunks));
});

test("marker filter holds a JSON-quoted reply whole", () => {
  const filter = createMarkerStreamFilter();
  assert.equal(filter.push("  "), "");
  assert.equal(filter.push('"quoted'), "");
  assert.equal(filter.push(' reply"'), "");
  assert.equal(filter.finish("display text"), "display text");
});

test("streamOpenCode streams v1 text and reasoning deltas before the turn finishes", async () => {
  const gateDir = mkdtempSync(join(tmpdir(), "opencode-pi-gate-"));
  const gate = join(gateDir, "first-delta-seen");
  try {
    const events = await withFakeOpenCode(
      fakeServer({
        events: [
          partStart("prt_think", "reasoning"),
          partDelta("prt_think", "Plan"),
          textPart("prt_think", "Plan.", "reasoning"),
          partStart("prt_text"),
          partDelta("prt_text", "Hel"),
          // The server holds the rest of the turn until Pi has seen a delta.
          { waitFor: gate },
          partDelta("prt_text", "lo"),
          textPart("prt_text", "Hello!"),
          stepFinish("stop", { total: 10, input: 4, output: 6, reasoning: 0, cache: { read: 0, write: 0 } }),
        ],
      }),
      async () => {
        const seen: AssistantMessageEvent[] = [];
        for await (const event of streamOpenCode(fakeModel(), fakeContext())) {
          seen.push(event);
          if (event.type === "text_delta") writeFileSync(gate, "");
        }
        return seen;
      },
    );

    assert.deepEqual(
      events.map((event) => (event.type.endsWith("_delta") ? `${event.type}:${(event as { delta: string }).delta}` : event.type)),
      [
        "start",
        "thinking_start", "thinking_delta:Plan", "thinking_delta:.", "thinking_end",
        "text_start", "text_delta:Hel", "text_delta:lo", "text_delta:!", "text_end",
        "done",
      ],
    );
    const done = events.at(-1);
    if (done?.type !== "done") return assert.fail("expected done");
    assert.equal(done.reason, "stop");
    assert.deepEqual(done.message.content, [
      { type: "thinking", thinking: "Plan." },
      { type: "text", text: "Hello!" },
    ]);
    assert.equal(done.message.usage.totalTokens, 10);
  } finally {
    rmSync(gateDir, { recursive: true, force: true });
  }
});

test("streamOpenCode never streams raw tool-call markers into Pi", async () => {
  const reply = `Reading it.\n${TOOL_CALL}`;
  // Five-character deltas cut the marker in several places.
  const deltas = reply.match(/[\s\S]{1,5}/g)!;
  const events = await collect(fakeServer({ events: streamedText("prt_text", deltas) }), fakeContext(["read"]));

  assert.ok(textDeltas(events).every((delta) => !delta.includes("<")), JSON.stringify(textDeltas(events)));
  const done = events.at(-1);
  if (done?.type !== "done") return assert.fail(`expected done, got ${JSON.stringify(done)}`);
  assert.equal(done.reason, "toolUse");
  assert.deepEqual(
    done.message.content.map((block) => (block.type === "toolCall" ? { name: block.name, arguments: block.arguments } : block)),
    [
      { type: "text", text: "Reading it.\n" },
      { name: "read", arguments: { path: "a.txt" } },
    ],
  );
});

test("streamOpenCode releases held text that turns out not to be a marker", async () => {
  const events = await collect(fakeServer({ events: streamedText("prt_text", ["Use <pi_", "tool> tags"]) }));
  assert.deepEqual(textDeltas(events), ["Use ", "<pi_tool> tags"]);
  const done = events.at(-1);
  if (done?.type !== "done") return assert.fail("expected done");
  assert.deepEqual(done.message.content, [{ type: "text", text: "Use <pi_tool> tags" }]);
});

test("streamOpenCode still applies the marker rules to streamed replies", async () => {
  const events = await collect(
    fakeServer({ events: streamedText("prt_text", ['<pi_tool_call>{"name":"bash","arguments":{}}</pi_tool_call>']) }),
    fakeContext(["read"]),
  );
  const error = events.at(-1);
  if (error?.type !== "error") return assert.fail("expected error");
  assert.match(error.error.errorMessage ?? "", /unavailable Pi tool "bash"/);
  assert.deepEqual(textDeltas(events), []);
});

test("streamOpenCode does not stream the prompt back as the reply", async () => {
  const events = await collect(fakeServer());
  assert.deepEqual(textDeltas(events), ["hello"]);
});

test("streamOpenCode sends the agent, model, variant, and image files to the server", async () => {
  await withCapture(async (capturePath) => {
    const model: Model<Api> = {
      ...fakeModel(),
      reasoning: true,
      input: ["text", "image"],
      thinkingLevelMap: { high: "max" },
    };
    const context: Context = {
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "look" },
            { type: "image", mimeType: "image/png", data: Buffer.from("png").toString("base64") },
          ],
          timestamp: 1,
        },
      ],
    };
    const message = await withFakeOpenCode(fakeServer({ capturePath }), () =>
      streamOpenCode(model, context, { reasoning: "high" }).result(),
    );

    assert.equal(message.stopReason, "stop");
    const [prompt] = prompts(capturePath);
    assert.equal(prompt?.body.agent, "pi-model");
    assert.deepEqual(prompt?.body.model, { providerID: "opencode", modelID: "fake-free" });
    assert.equal(prompt?.body.variant, "max");
    const [file, text] = prompt?.body.parts ?? [];
    assert.equal(file.type, "file");
    assert.equal(file.mime, "image/png");
    assert.equal(file.filename, "image-001.png");
    assert.equal(readFileSync(fileURLToPath(file.url), "utf8"), "png");
    assert.match(text.text, /# Pi\/OpenCode bridge instructions/);
    assert.match(text.text, /OpenCode's <env> block describes the bridge's scratch directory/);
    assert.match(prompt?.directory ?? "", /opencode-pi-/);
  });
});

test("streamOpenCode writes the v1 agent and disables OpenCode's title agent", async () => {
  await withCapture(async (capturePath) => {
    await withFakeOpenCode(
      fakeServer({ capturePath }).replace(
        "res.writeHead(204);",
        `entry.agent = fs.readFileSync(require("node:path").join(entry.directory, ".opencode", "agents", "pi-model.md"), "utf8");
    entry.title = fs.readFileSync(require("node:path").join(entry.directory, ".opencode", "agents", "title.md"), "utf8");
    record(entry);
    res.writeHead(204);`,
      ),
      () => streamOpenCode(fakeModel(), fakeContext()).result(),
    );
    const prompt = readCaptured(capturePath).find((entry) => "agent" in entry) as Captured & { agent: string; title: string };
    assert.match(prompt.agent, /permission:\n  "\*": deny/);
    assert.equal(prompt.title, "---\ndisable: true\n---\n");
  });
});

test("streamOpenCode continues the recorded server session with only the new transcript", async () => {
  await withCapture(async (capturePath) => {
    const sessionId = "pi-session-stream";
    const first: Context = { messages: [{ role: "user", content: "first question", timestamp: 1 }] };
    await withFakeOpenCode(fakeServer({ capturePath, sessionId: "ses-reused" }), async () => {
      try {
        const one = await streamOpenCode(fakeModel(), first, { sessionId }).result();
        assert.equal(one.stopReason, "stop");
        const second: Context = {
          messages: [...first.messages, one, { role: "user", content: "second question", timestamp: 3 }],
        };
        const two = await streamOpenCode(fakeModel(), second, { sessionId }).result();
        assert.equal(two.stopReason, "stop");
      } finally {
        await cleanupPiSessionState(sessionId);
      }
    });

    const captured = readCaptured(capturePath);
    assert.equal(captured.filter((entry) => entry.path === "/session").length, 1);
    const [one, two] = prompts(capturePath);
    assert.equal(one?.path, "/session/ses-reused/prompt_async");
    assert.equal(two?.path, "/session/ses-reused/prompt_async");
    assert.equal(one?.directory, two?.directory);
    assert.match(two?.body.parts.at(-1).text, /# Continued Pi\/OpenCode bridge turn/);
    assert.match(two?.body.parts.at(-1).text, /second question/);
    assert.doesNotMatch(two?.body.parts.at(-1).text, /first question/);
    // Teardown unloads the directory from the server.
    assert.ok(captured.some((entry) => entry.path === "/instance/dispose" && entry.directory === one?.directory));
  });
});

test("streamOpenCode rejects permission asks and tolerates already-settled ones", async () => {
  await withCapture(async (capturePath) => {
    const events = await collect(
      fakeServer({
        capturePath,
        settledPermissions: ["per_2"],
        events: [
          { type: "permission.asked", properties: { id: "per_1", permission: "read" } },
          { type: "permission.asked", properties: { id: "per_2", permission: "read" } },
          { type: "message.part.updated", properties: { part: { id: "prt_tool", messageID: "msg_reply", type: "tool", tool: "read", state: { status: "error", error: "The user rejected permission to use this specific tool call." } } } },
          ...streamedText("prt_text", [TOOL_CALL]),
        ],
      }),
      fakeContext(["read"]),
    );
    const done = events.at(-1);
    if (done?.type !== "done") return assert.fail(`expected done, got ${JSON.stringify(done)}`);
    assert.equal(done.reason, "toolUse");
    const replies = readCaptured(capturePath).filter((entry) => entry.path.startsWith("/permission/"));
    assert.deepEqual(replies.map((entry) => [entry.path, entry.body]), [
      ["/permission/per_1/reply", { reply: "reject" }],
      ["/permission/per_2/reply", { reply: "reject" }],
    ]);
  });
});

test("streamOpenCode fails a streamed turn when an OpenCode tool executed", async () => {
  const events = await collect(
    fakeServer({
      events: [
        { type: "message.part.updated", properties: { part: { id: "prt_tool", messageID: "msg_reply", type: "tool", tool: "bash", state: { status: "completed", output: "ran" } } } },
        textPart("prt_text", "done"),
      ],
    }),
  );
  const error = events.at(-1);
  if (error?.type !== "error") return assert.fail("expected error");
  assert.match(error.error.errorMessage ?? "", /disabled native tool \("bash"\)/);
});

test("streamOpenCode reports session errors from the server", async () => {
  const events = await collect(
    fakeServer({ events: [{ type: "session.error", properties: { error: { name: "APIError", data: { message: "rate limited" } } } }] }),
  );
  const error = events.at(-1);
  if (error?.type !== "error") return assert.fail("expected error");
  assert.equal(error.error.errorMessage, "rate limited");
});

test("streamOpenCode diagnoses an empty streamed reply", async () => {
  const message = await withFakeOpenCode(fakeServer({ events: [] }), () =>
    streamOpenCode(fakeModel(), fakeContext()).result(),
  );
  assert.equal(message.stopReason, "error");
  assert.match(message.errorMessage ?? "", /empty assistant response/);
});

test("streamOpenCode aborts the server session when Pi aborts mid-turn", async () => {
  await withCapture(async (capturePath) => {
    const controller = new AbortController();
    const events = await withFakeOpenCode(
      fakeServer({
        capturePath,
        sessionId: "ses-abort",
        events: [partStart("prt_text"), partDelta("prt_text", "partial")],
        idle: false,
      }),
      async () => {
        const seen: AssistantMessageEvent[] = [];
        for await (const event of streamOpenCode(fakeModel(), fakeContext(), { signal: controller.signal })) {
          seen.push(event);
          if (event.type === "text_delta") controller.abort();
        }
        return seen;
      },
    );

    const error = events.at(-1);
    if (error?.type !== "error") return assert.fail("expected error");
    assert.equal(error.reason, "aborted");
    assert.equal(error.error.errorMessage, "Request was aborted");
    // The streamed text block is closed and kept.
    assert.deepEqual(error.error.content, [{ type: "text", text: "partial" }]);
    assert.ok(events.some((event) => event.type === "text_end"));
    assert.ok(readCaptured(capturePath).some((entry) => entry.path === "/session/ses-abort/abort"));
  });
});

test("streamOpenCode bounds a streamed turn with the turn timeout", async () => {
  const message = await withFakeOpenCode(fakeServer({ events: [], idle: false }), () =>
    streamOpenCode(fakeModel(), fakeContext(), { timeoutMs: 200 }).result(),
  );
  assert.equal(message.stopReason, "error");
  assert.equal(message.errorMessage, "opencode timed out after 200ms");
});

test("streamOpenCode falls back to opencode run when the server cannot start", async () => {
  const events = [
    { type: "text", sessionID: "ses-run", part: { text: "from run" } },
    { type: "step_finish", sessionID: "ses-run", part: { reason: "stop" } },
  ];
  const message = await withFakeOpenCode(
    `if (process.argv[2] === "serve") { console.error("port in use"); process.exit(3); }
if (process.argv[2] === "session") process.exit(0);
for (const event of ${JSON.stringify(events)}) process.stdout.write(JSON.stringify(event) + "\\n");`,
    () => streamOpenCode(fakeModel(), fakeContext()).result(),
  );
  assert.equal(message.stopReason, "stop");
  assert.deepEqual(message.content, [{ type: "text", text: "from run" }]);
});

test("streamOpenCode keeps OpenCode v2 turns on opencode run", async () => {
  const message = await withFakeOpenCode(
    `if (process.argv[2] === "serve") process.exit(9);
if (process.argv[2] === "session") process.exit(0);
process.stdin.resume();
process.stdin.on("end", () => {
  process.stdout.write(JSON.stringify({ type: "text", sessionID: "ses-v2", part: { text: process.argv.includes("--standalone") ? "standalone" : "other" } }) + "\\n");
});`,
    () => streamOpenCode(fakeModel(), fakeContext()).result(),
    "opencode v2.1.0",
  );
  assert.equal(message.stopReason, "stop");
  assert.deepEqual(message.content, [{ type: "text", text: "standalone" }]);
  assert.equal(runningOpenCodeServer(), undefined);
});

test("parseEvents keeps CRLF event separators split across chunks", async () => {
  const encoder = new TextEncoder();
  const text = 'data: {"type":"a"}\r\n\r\ndata: {"type":"b"}\r\n\r\n';
  // Split inside the first separator, between its \r and \n.
  const cut = text.indexOf("\r\n\r\n") + 3;
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoder.encode(text.slice(0, cut)));
      controller.enqueue(encoder.encode(text.slice(cut)));
      controller.close();
    },
  });
  const types: string[] = [];
  for await (const event of parseEvents(body)) types.push(event.type);
  assert.deepEqual(types, ["a", "b"]);
});

test("the shared server stops parsing its output once it is ready", async () => {
  await withFakeOpenCode(
    `if (process.argv[2] !== "serve") process.exit(1);
require("node:http").createServer(() => undefined).listen(0, "127.0.0.1", function () {
  console.log("opencode server listening on http://127.0.0.1:" + this.address().port);
  let n = 0;
  const timer = setInterval(() => { console.log("request log " + n); if (++n === 30) clearInterval(timer); }, 2);
});`,
    async () => {
      const server = await withOpenCodeServer(process.env.OPENCODE_PI_BIN!, process.env, async (server) => server);
      await new Promise((resolve) => setTimeout(resolve, 300));
      // One exit listener for the running server, plus Node's own.
      assert.ok(server.child.listenerCount("exit") <= 2, `exit listeners: ${server.child.listenerCount("exit")}`);
      assert.equal(runningOpenCodeServer(), server);
    },
  );
});

test("the watchdog kills a server that ignores SIGTERM", async () => {
  const dir = mkdtempSync(join(tmpdir(), "opencode-pi-pid-"));
  const pidFile = join(dir, "serve.pid");
  await withFakeOpenCode(
    `if (process.argv[2] !== "serve") process.exit(1);
process.on("SIGTERM", () => undefined);
require("node:fs").writeFileSync(${JSON.stringify(pidFile)}, String(process.pid));
require("node:http").createServer(() => undefined).listen(0, "127.0.0.1", function () {
  console.log("opencode server listening on http://127.0.0.1:" + this.address().port);
});`,
    async () => {
      const server = await withOpenCodeServer(process.env.OPENCODE_PI_BIN!, process.env, async (server) => server);
      const pid = Number(readFileSync(pidFile, "utf8"));
      stopOpenCodeServer();
      const alive = () => { try { process.kill(pid, 0); return true; } catch { return false; } };
      const deadline = Date.now() + 10_000;
      while ((alive() || server.child.exitCode === null && server.child.signalCode === null) && Date.now() < deadline) {
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
      assert.equal(alive(), false, "opencode serve survived the watchdog");
      assert.notEqual(server.child.exitCode ?? server.child.signalCode, null, "the watchdog is still running");
    },
  ).finally(() => rmSync(dir, { recursive: true, force: true }));
});

// Mock timers landed in Node 20.4; the package still supports Node 18.
const mockTimersSkip = !("timers" in mock) && "needs Node >=20.4 mock timers";

test("the shared server stops after 10 idle minutes unless a turn starts first", { skip: mockTimersSkip }, async (t) => {
  const IDLE_SHUTDOWN_MS = 10 * 60_000;
  await withFakeOpenCode(
    `if (process.argv[2] !== "serve") process.exit(1);
require("node:http").createServer(() => undefined).listen(0, "127.0.0.1", function () {
  console.log("opencode server listening on http://127.0.0.1:" + this.address().port);
});`,
    async () => {
      const bin = process.env.OPENCODE_PI_BIN!;
      // Mock timers only once the server is up, so its real start timeout is unaffected.
      const server = await withOpenCodeServer(bin, process.env, async (server) => {
        t.mock.timers.enable({ apis: ["setTimeout"] });
        return server;
      });
      t.mock.timers.tick(IDLE_SHUTDOWN_MS - 1);
      assert.equal(runningOpenCodeServer(), server, "stopped before the idle window ended");

      // A turn inside the window cancels the pending stop, however long it runs.
      await withOpenCodeServer(bin, process.env, async (reused) => {
        assert.equal(reused, server);
        t.mock.timers.tick(IDLE_SHUTDOWN_MS);
        assert.equal(runningOpenCodeServer(), server, "stopped during a turn");
      });

      // The window restarts from the end of the last turn.
      t.mock.timers.tick(IDLE_SHUTDOWN_MS - 1);
      assert.equal(runningOpenCodeServer(), server, "the idle window did not restart");
      t.mock.timers.tick(1);
      assert.equal(runningOpenCodeServer(), undefined, "still running after the idle window");
      assert.equal(server.child.killed, true, "the server was not signalled to stop");

      t.mock.timers.reset();
      if (server.child.exitCode === null && server.child.signalCode === null) {
        await new Promise((resolve) => server.child.once("exit", resolve));
      }
    },
  );
});

// PR B, S6: both start-failure shapes must still answer through `opencode run`.
// A v2 `serve` that prints its banner and then closes `/api/event` before the
// `server.connected` frame used to answer and then lose the answer: the turn
// ran the fallback with the streaming finalizer still armed, so the reply was
// replaced by empty streamed text and the turn failed as an empty response.
const v2StreamClosingBeforeConnect = `
const http = require("node:http");
if (process.argv[2] === "serve") {
  http.createServer((req, res) => {
    if (req.url.startsWith("/api/event")) {
      res.writeHead(200, { "content-type": "text/event-stream" });
      res.end();
      return;
    }
    res.writeHead(404, { "content-type": "application/json" });
    res.end("{}");
  }).listen(0, "127.0.0.1", function () {
    console.log("opencode server listening on http://127.0.0.1:" + this.address().port);
  });
} else {
  let body = "";
  process.stdin.on("data", (chunk) => { body += chunk; });
  process.stdin.on("end", () => {
    process.stdout.write(JSON.stringify({ type: "text", part: { text: "fallback answer", sessionID: "ses-fallback" } }) + "\\n");
    process.exit(0);
  });
}
`;

test("a v2 stream that closes before server.connected still answers through opencode run", async () => {
  const notices: string[] = [];
  const context = {
    ...fakeContext(),
    ui: { notify: (message: string) => notices.push(message) },
  } as unknown as Context;

  const message = await withFakeOpenCode(
    v2StreamClosingBeforeConnect,
    () => streamOpenCode(fakeModel(), context).result(),
    "opencode v2.1.2",
  );

  assert.equal(message.stopReason, "stop", message.errorMessage ?? "");
  assert.deepEqual(
    message.content
      .filter((block) => block.type === "text")
      .map((block) => (block as { text: string }).text),
    ["fallback answer"],
    "the fallback's reply is shown, not the empty streamed text",
  );
  assert.equal(notices.length, 1, "the fallback is announced once");
});

// The other shape: the server never prints a banner at all.
const v2ServerDyingBeforeBanner = `
if (process.argv[2] === "serve") {
  process.stderr.write("banner never printed\\n");
  process.exit(1);
}
let body = "";
process.stdin.on("data", (chunk) => { body += chunk; });
process.stdin.on("end", () => {
  process.stdout.write(JSON.stringify({ type: "text", part: { text: "fallback answer" } }) + "\\n");
  process.exit(0);
});
`;

test("a v2 server that dies before its banner answers through opencode run", async () => {
  const notices: string[] = [];
  const context = {
    ...fakeContext(),
    ui: { notify: (message: string) => notices.push(message) },
  } as unknown as Context;

  const message = await withFakeOpenCode(
    v2ServerDyingBeforeBanner,
    () => streamOpenCode(fakeModel(), context).result(),
    "opencode v2.1.2",
  );

  assert.equal(message.stopReason, "stop", message.errorMessage ?? "");
  assert.equal(notices.length, 1, "the fallback is announced once");
});
