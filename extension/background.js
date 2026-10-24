// Service worker: context menu, keyboard shortcut, text extraction, API calls.
// Nothing is read from a page or sent anywhere until the user triggers a check.
const DEFAULT_API = "https://prohori-mhuf.onrender.com";
const SLEEP_MS = (self.PROHORI_SLEEP_MS ?? 4000);
const WAKE_TIMEOUT_MS = 100000;
const MAX_CHARS = 20000;

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function getApi() {
  const { apiUrl } = await chrome.storage.sync.get({ apiUrl: DEFAULT_API });
  return String(apiUrl || DEFAULT_API).replace(/\/+$/, "");
}

async function fetchTimeout(url, opts, ms) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), ms);
  try { return await fetch(url, { ...opts, signal: ctl.signal }); } finally { clearTimeout(t); }
}

// Render's free plan sleeps when idle; the first request can take up to a minute.
async function waitForServer(api) {
  const t0 = Date.now();
  while (Date.now() - t0 < WAKE_TIMEOUT_MS) {
    try {
      const r = await fetchTimeout(api + "/health", {}, 15000);
      if (r.ok) {
        const j = await r.json();
        if (j.model_loaded) return;
        const e = new Error(j.detail || "The server is running but has no model loaded.");
        e.fatal = true;
        throw e;
      }
    } catch (e) {
      if (e.fatal) throw e;
    }
    // Any extension API call resets the service worker's idle timer, so it is not shut down mid-wait.
    try { await chrome.runtime.getPlatformInfo(); } catch (e) { /* ignore */ }
    await sleep(SLEEP_MS);
  }
  throw new Error("The server did not wake up in time. Wait a minute and try again.");
}

async function analyze(text) {
  const api = await getApi();
  await waitForServer(api);
  let r;
  try {
    r = await fetchTimeout(api + "/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) }, 60000);
  } catch (e) {
    throw new Error("Could not reach the Prohori server. Check your connection and the server address in the options.");
  }
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : `Request failed (${r.status}).`);
  if (typeof j.ai_probability !== "number") throw new Error("The server returned an unexpected reply. Check the server address in Settings, or redeploy the server.");
  return { data: j, fullReportUrl: api + "/" };
}

// ---- functions that run INSIDE the page (must be self-contained) ----
function pageSelection() { return window.getSelection ? window.getSelection().toString() : ""; }
function pageMainText() {
  const pick = document.querySelector("article") || document.querySelector("main") || document.querySelector("[role=main]");
  let t = "";
  if (pick) t = pick.innerText;
  if (!t || t.trim().split(/\s+/).length < 40) {
    t = Array.from(document.querySelectorAll("p")).map((p) => p.innerText).filter((x) => x.trim().length > 40).join("\n\n");
  }
  if (!t || t.trim().split(/\s+/).length < 40) t = document.body ? document.body.innerText : "";
  return t;
}

async function readFromTab(tabId, fn, allFrames) {
  const res = await chrome.scripting.executeScript({ target: { tabId, allFrames: !!allFrames }, func: fn });
  return (res || []).map((r) => r.result).filter((x) => typeof x === "string" && x.trim());
}

function clip(text) {
  text = text.trim();
  if (text.length <= MAX_CHARS) return { text, clipped: false };
  const cut = text.slice(0, MAX_CHARS);
  const end = Math.max(cut.lastIndexOf("।"), cut.lastIndexOf(". "), cut.lastIndexOf("\n"));
  return { text: end > MAX_CHARS * 0.6 ? cut.slice(0, end + 1) : cut, clipped: true };
}

async function runCheck(tab, mode, directSelection) {
  const send = (m) => chrome.tabs.sendMessage(tab.id, m).catch(() => {});
  try {
    await chrome.scripting.executeScript({ target: { tabId: tab.id }, files: ["ui.js", "content.js"] });
  } catch (e) {
    chrome.action.setBadgeText({ text: "!" });
    chrome.action.setTitle({ title: "Prohori cannot read this page. Open the toolbar popup and paste the text instead." });
    setTimeout(() => { chrome.action.setBadgeText({ text: "" }); chrome.action.setTitle({ title: "Prohori: check Bangla text" }); }, 6000);
    return;
  }
  try {
    let text = directSelection || "";
    if (mode === "selection" && !text.trim()) text = (await readFromTab(tab.id, pageSelection, true))[0] || "";
    if (mode === "page") text = (await readFromTab(tab.id, pageMainText, false))[0] || "";
    if (!text.trim()) { await send({ type: "prohori:error", message: mode === "page" ? "I could not find readable text on this page." : "Select some Bangla text first, then try again." }); return; }
    const { text: body, clipped } = clip(text);
    await send({ type: "prohori:loading", message: "Analyzing… (the first request after a quiet period can take up to a minute while the server wakes up)" });
    const out = await analyze(body);
    if (clipped) out.data.warnings = [...(out.data.warnings || []), `Only the first ${MAX_CHARS.toLocaleString()} characters were checked.`];
    await send({ type: "prohori:result", data: out.data, fullReportUrl: out.fullReportUrl });
  } catch (e) {
    await send({ type: "prohori:error", message: e.message || "Something went wrong." });
  }
}

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: "prohori-selection", title: "Check selection with Prohori", contexts: ["selection"] });
  chrome.contextMenus.create({ id: "prohori-page", title: "Check this page's main text with Prohori", contexts: ["page"] });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (!tab) return;
  if (info.menuItemId === "prohori-selection") return runCheck(tab, "selection", info.selectionText);
  if (info.menuItemId === "prohori-page") return runCheck(tab, "page");
});

chrome.commands.onCommand.addListener(async (cmd) => {
  if (cmd !== "check-selection") return;
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab) runCheck(tab, "selection");
});

// Popup -> background
chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg && msg.type === "prohori:analyze") {
    analyze(clip(msg.text || "").text).then((o) => sendResponse({ ok: true, ...o })).catch((e) => sendResponse({ ok: false, error: e.message }));
    return true; // async response
  }
});
