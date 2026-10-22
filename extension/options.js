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
