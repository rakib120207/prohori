const $ = (id) => document.getElementById(id);
const UI = window.ProhoriUI;
const style = document.createElement("style");
style.textContent = UI.CSS.replace(":host{all:initial}", "");
document.head.appendChild(style);

function show(node) { const wrap = UI.h("div", { class: "ph" }, node); $("out").replaceChildren(wrap); }

$("opt").addEventListener("click", (e) => { e.preventDefault(); chrome.runtime.openOptionsPage(); });

$("sel").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  try {
    const res = await chrome.scripting.executeScript({ target: { tabId: tab.id, allFrames: true }, func: () => window.getSelection().toString() });
    const text = (res || []).map((r) => r.result).find((x) => x && x.trim());
    if (text) $("t").value = text; else show(UI.renderError("Nothing is selected on this page."));
  } catch (e) { show(UI.renderError("This page cannot be read by extensions. Paste the text here instead.")); }
});

$("go").addEventListener("click", async () => {
  const text = $("t").value.trim();
  if (!text) { show(UI.renderError("Paste or select some Bangla text first.")); return; }
  $("go").disabled = true;
  show(UI.renderLoading("Analyzing… (the first request after a quiet period can take up to a minute)"));
  try {
    const r = await chrome.runtime.sendMessage({ type: "prohori:analyze", text });
    if (!r || !r.ok) show(UI.renderError((r && r.error) || "No response from the extension."));
    else show(UI.renderResult(r.data, { fullReportUrl: r.fullReportUrl }));
  } catch (e) { show(UI.renderError(e.message)); }
  $("go").disabled = false;
});
