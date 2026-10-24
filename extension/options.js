const DEFAULT_API = "https://prohori-mhuf.onrender.com";
const $ = (id) => document.getElementById(id);
const say = (t, ok) => { $("msg").textContent = t; $("msg").className = ok ? "ok" : "bad"; };

chrome.storage.sync.get({ apiUrl: DEFAULT_API }).then(({ apiUrl }) => { $("url").value = apiUrl; });

function parse() {
  let u;
  try { u = new URL($("url").value.trim()); } catch { return null; }
  const local = u.hostname === "localhost" || u.hostname === "127.0.0.1";
  if (!(u.protocol === "https:" || (local && u.protocol === "http:"))) return null;
  return u;
}

$("save").addEventListener("click", async () => {
  const u = parse();
  if (!u) { say("Enter a full address starting with https:// (http:// only for localhost).", false); return; }
  const origin = u.origin + "/*";
  const has = await chrome.permissions.contains({ origins: [origin] });
  if (!has) {
    const granted = await chrome.permissions.request({ origins: [origin] });
    if (!granted) { say("Permission to contact that server was not granted.", false); return; }
  }
  await chrome.storage.sync.set({ apiUrl: u.origin });
  say("Saved.", true);
});

$("reset").addEventListener("click", async () => {
  await chrome.storage.sync.set({ apiUrl: DEFAULT_API });
  $("url").value = DEFAULT_API;
  say("Reset to the default server.", true);
});

$("test").addEventListener("click", async () => {
  const u = parse();
  if (!u) { say("Enter a valid address first.", false); return; }
  say("Contacting the server… (can take a minute if it is asleep)", true);
  try {
    const r = await fetch(u.origin + "/health");
    const j = await r.json();
    say(j.model_loaded ? "Connected. The model is loaded and ready." : "Server reachable, but no model is loaded: " + (j.detail || ""), !!j.model_loaded);
  } catch (e) { say("Could not reach that server. Save it first to grant permission, then test again.", false); }
});

// Full self-test: server awake? model loaded? version? does /analyze return every field the extension displays?
const SAMPLE = "আমরা সবাই মিলে আজ সকালে বাজারে গিয়েছিলাম। সেখানে অনেক মানুষের ভিড় ছিল এবং দোকানিরা নানা রকম সবজি বিক্রি করছিলেন। আমি কিছু তাজা শাক আর মাছ কিনলাম। বাড়ি ফিরে মা রান্না করলেন আর আমরা সবাই একসঙ্গে দুপুরের খাবার খেলাম।";
$("self").addEventListener("click", async () => {
  const u = parse();
  if (!u) { say("Enter a valid address first.", false); return; }
  const lines = [];
  const out = () => { $("msg").textContent = lines.join("  |  "); $("msg").className = lines.some((l) => l.startsWith("FAIL")) ? "bad" : "ok"; };
  try {
    lines.push("waking server…"); out();
    let health = null;
    for (let i = 0; i < 20 && !health; i++) {
      try { const r = await fetch(u.origin + "/health"); if (r.ok) health = await r.json(); } catch (e) { /* retry */ }
      if (!health) await new Promise((r) => setTimeout(r, 4000));
    }
    lines.pop();
    if (!health) { lines.push("FAIL: server did not respond (check the address, or save it first to grant permission)"); out(); return; }
    lines.push(health.model_loaded ? "server up, model loaded" : "FAIL: server up but NO model loaded (" + (health.detail || "") + ")"); out();
    if (!health.model_loaded) return;
    const info = await (await fetch(u.origin + "/model-info")).json().catch(() => ({}));
    lines.push("server version: " + (info.version || "unknown (old version?)")); out();
    const r = await fetch(u.origin + "/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text: SAMPLE }) });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) { lines.push("FAIL: /analyze said " + (j.detail || r.status)); out(); return; }
    const missing = ["ai_probability", "band", "evidence", "reasons", "stats", "disclaimer"].filter((k) => !(k in j));
    lines.push(missing.length ? "WARN: reply is missing " + missing.join(", ") + " (the server is an older version; the extension still works but shows less)" : "reply format OK");
    out();
  } catch (e) { lines.push("FAIL: " + e.message); out(); }
});
