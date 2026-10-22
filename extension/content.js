// Injected on demand (activeTab) when the user triggers a check. Shows a result card in a closed shadow root,
// so the page's own CSS and scripts can neither restyle nor read it.
(function () {
  if (window.__prohoriContent) return;
  window.__prohoriContent = true;
  let host = null, wrap = null;

  function ensure() {
    if (host && host.isConnected) return;
    host = document.createElement("div");
    host.id = "prohori-overlay-host";
    host.setAttribute("style", "all:initial;position:fixed;top:16px;right:16px;z-index:2147483647;");
    const root = host.attachShadow({ mode: "closed" });
    const style = document.createElement("style");
    style.textContent = window.ProhoriUI.CSS;
    wrap = document.createElement("div");
    wrap.className = "ph";
    root.append(style, wrap);
    document.documentElement.appendChild(host);
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); }, true);
  }
  function close() { if (host) { host.remove(); host = null; } }

  function frame(body, apiBase) {
    ensure();
    const { h } = window.ProhoriUI;
    wrap.replaceChildren(
      h("div", { class: "ph-top" },
        h("div", { class: "ph-brand" }, "PROHORI", h("span", {}, "প্রহরী")),
        h("button", { class: "ph-x", title: "Close (Esc)", onclick: close }, "×")),
      body);
  }

  chrome.runtime.onMessage.addListener((msg) => {
    const UI = window.ProhoriUI;
    if (msg.type === "prohori:loading") frame(UI.renderLoading(msg.message));
    else if (msg.type === "prohori:error") frame(UI.renderError(msg.message));
    else if (msg.type === "prohori:result") frame(UI.renderResult(msg.data, { fullReportUrl: msg.fullReportUrl }));
    else if (msg.type === "prohori:close") close();
  });
})();
