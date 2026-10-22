// Offline tests for the extension logic (jsdom + a mocked chrome API). Run: node extension/test.js
const fs = require("fs"), path = require("path"), vm = require("vm");
let JSDOM; try { ({ JSDOM } = require("jsdom")); } catch { ({ JSDOM } = require(path.join(process.env.JSDOM_PATH || "/tmp/exptest/node_modules", "jsdom"))); }
const read = (f) => fs.readFileSync(path.join(__dirname, f), "utf8");
let pass = 0, fail = 0;
const ok = (c, m) => { if (c) { pass++; console.log("  ok  " + m); } else { fail++; console.log("  FAIL " + m); } };

const SAMPLE = {
  ai_probability: 0.86,
  band: { key: "review", label: "Worth a closer look", detail: "Several measurable patterns resemble AI-written text." },
  evidence: { key: "moderate", label: "Moderate evidence base", detail: "x", words: 140, chunks: 3 },
  stats: { words: 140, sentences: 9 }, warnings: ["<b>warn</b>"],
  reasons: [{ feature: "surface_pattern", label: "Spelling habits", direction: "ai", impact: 61.5, text: "<img src=x onerror=alert(1)> points toward AI", examples: ["<script>alert(1)</script>", "কোনো"] }],
  chunks: [{ text: "<svg onload=alert(2)> পাঠ্য", probability: 0.9, words: 60 }, { text: "দ্বিতীয়", probability: 0.1, words: 50 }],
  disclaimer: "A probability, not a verdict.",
};

(async () => {
  console.log("ui.js / content.js (jsdom)");
  const dom = new JSDOM("<!DOCTYPE html><body><p>page</p></body>", { runScripts: "outside-only", pretendToBeVisual: true });
  const w = dom.window;
  const realAttach = w.Element.prototype.attachShadow;
  let shadow = null;
  w.Element.prototype.attachShadow = function (o) { shadow = realAttach.call(this, { ...o, mode: "open" }); return shadow; };
  let listener = null;
  w.chrome = { runtime: { onMessage: { addListener: (f) => (listener = f) } } };
  w.eval(read("ui.js")); w.eval(read("content.js"));
  ok(typeof w.ProhoriUI.renderResult === "function", "UI exposes renderResult");
  ok(typeof listener === "function", "content script registered a message listener");

  listener({ type: "prohori:loading", message: "Analyzing…" });
  ok(shadow && shadow.textContent.includes("Analyzing"), "loading state shown in the shadow root");
  listener({ type: "prohori:result", data: SAMPLE, fullReportUrl: "https://example.org/" });
  const text = shadow.textContent;
  ok(text.includes("86%") && text.includes("Worth a closer look"), "result shows percentage and band");
  ok(text.includes("<img src=x onerror=alert(1)>"), "hostile reason text is shown literally");
  ok(!shadow.querySelector("img") && !shadow.querySelector("script") && !shadow.querySelector("svg"), "no HTML from the server was parsed into elements");
  ok(text.includes("Passage 1") && text.includes("কোনো"), "passages and example patterns are rendered");
  ok(shadow.querySelector("a.ph-link").getAttribute("rel").includes("noopener"), "external link uses rel=noopener");
  ok(!!w.document.getElementById("prohori-overlay-host"), "overlay host is attached to the page");
  listener({ type: "prohori:error", message: "Text too short" });
  ok(shadow.textContent.includes("Text too short"), "error state shown");
  w.document.dispatchEvent(new w.KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
  ok(!w.document.getElementById("prohori-overlay-host"), "Escape closes the overlay");

  console.log("background.js (mocked chrome + fetch)");
  const calls = { exec: [], sent: [], menus: [], badge: [] };
  let healthFails = 2, analyzeBody = null, analyzeStatus = 200, healthLoaded = true;
  const sandbox = {
    console, setTimeout, clearTimeout, AbortController, JSON, Date, Promise, String, Number, Array, Object, Error, URL,
    self: { PROHORI_SLEEP_MS: 1 },
    chrome: {
      storage: { sync: { get: async (d) => d } },
      scripting: { executeScript: async (o) => { calls.exec.push(o); if (o.func) { return [{ result: o.func.name === "pageSelection" ? "  নির্বাচিত বাংলা লেখা  " : "পৃষ্ঠার লেখা" }]; } return []; } },
      tabs: { sendMessage: async (id, m) => { calls.sent.push(m); }, query: async () => [{ id: 7 }] },
      contextMenus: { create: (m) => calls.menus.push(m.id), onClicked: { addListener: (f) => (sandbox.onClicked = f) } },
      commands: { onCommand: { addListener: (f) => (sandbox.onCommand = f) } },
      runtime: { onInstalled: { addListener: (f) => (sandbox.onInstalled = f) }, onMessage: { addListener: (f) => (sandbox.onMessage = f) } },
      action: { setBadgeText: (o) => calls.badge.push(o.text), setTitle: () => {} },
    },
    fetch: async (url, opts) => {
      if (url.endsWith("/health")) {
        if (healthFails-- > 0) return { ok: false, status: 503, json: async () => ({}) };
        return { ok: true, status: 200, json: async () => ({ status: "ok", model_loaded: healthLoaded, detail: "no model" }) };
      }
      analyzeBody = JSON.parse(opts.body);
      return { ok: analyzeStatus === 200, status: analyzeStatus, json: async () => (analyzeStatus === 200 ? SAMPLE : { detail: "Text too short: need at least 10 words, got 3." }) };
    },
  };
  sandbox.self.self = sandbox.self;
  vm.createContext(sandbox);
  vm.runInContext(read("background.js"), sandbox);
  sandbox.onInstalled();
  ok(calls.menus.includes("prohori-selection") && calls.menus.includes("prohori-page"), "both context-menu items are created");

  await sandbox.onClicked({ menuItemId: "prohori-selection", selectionText: "  বাংলা লেখা এখানে  " }, { id: 7 });
  const types = calls.sent.map((m) => m.type);
  ok(types.join() === "prohori:loading,prohori:result", "flow: loading then result (server needed 2 retries to wake)");
  ok(analyzeBody && analyzeBody.text === "বাংলা লেখা এখানে", "selected text is trimmed and sent to /analyze");
  ok(calls.exec[0].files && calls.exec[0].files.join() === "ui.js,content.js", "UI + content script are injected on demand, not declared for all sites");

  calls.sent.length = 0; healthFails = 0; analyzeStatus = 422;
  await sandbox.onClicked({ menuItemId: "prohori-selection", selectionText: "ছোট" }, { id: 7 });
  const last = calls.sent[calls.sent.length - 1];
  ok(last.type === "prohori:error" && last.message.includes("too short"), "server error message is relayed to the user");

  calls.sent.length = 0; analyzeStatus = 200; healthLoaded = false;
  await sandbox.onClicked({ menuItemId: "prohori-selection", selectionText: "বাংলা লেখা" }, { id: 7 });
  ok(calls.sent[calls.sent.length - 1].type === "prohori:error" && calls.sent[calls.sent.length - 1].message.includes("no model"), "server with no model fails fast with a clear message");

  calls.sent.length = 0; healthLoaded = true; analyzeStatus = 200;
  await sandbox.onClicked({ menuItemId: "prohori-page" }, { id: 7 });
  ok(analyzeBody.text === "পৃষ্ঠার লেখা" && calls.sent.some((m) => m.type === "prohori:result"), "page mode extracts main text and analyzes it");

  calls.sent.length = 0; calls.exec.length = 0;
  const longText = ("বাংলা লেখা। ").repeat(5000);
  await sandbox.onClicked({ menuItemId: "prohori-selection", selectionText: longText }, { id: 7 });
  ok(analyzeBody.text.length <= 20000, "oversized text is clipped to the API limit");
  const res = calls.sent.find((m) => m.type === "prohori:result");
  ok(res && res.data.warnings.some((x) => x.includes("first 20,000")), "user is told the text was clipped");

  let resp = null;
  const keep = sandbox.onMessage({ type: "prohori:analyze", text: "পপআপ থেকে লেখা" }, {}, (r) => (resp = r));
  await new Promise((r) => setTimeout(r, 30));
  ok(keep === true && resp && resp.ok && resp.data.ai_probability === 0.86, "popup message path returns the result asynchronously");

  console.log(`\n${pass} passed, ${fail} failed`);
  process.exit(fail ? 1 : 0);
})();
