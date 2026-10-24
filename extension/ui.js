// Shared renderer for the popup and the on-page overlay. Builds DOM with textContent only:
// nothing from the page or the server is ever parsed as HTML.
(function () {
  const CSS = `
  :host{all:initial}
  .ph{--bg:#0b0d12;--panel:#11151d;--panel2:#171c27;--line:#232b3a;--text:#e8ecf3;--muted:#8d98ab;--ai:#ff7a59;--human:#4ade9a;--mid:#e6c55a;--accent:#7c9cff;
      font:14px/1.5 -apple-system,"Segoe UI",Roboto,"Noto Sans Bengali","Nirmala UI",sans-serif;color:var(--text);background:var(--bg);
      border:1px solid var(--line);border-radius:14px;box-shadow:0 12px 40px rgba(0,0,0,.45);padding:14px;box-sizing:border-box;width:360px;max-height:80vh;overflow:auto}
  .ph *{box-sizing:border-box}
  .ph-top{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
  .ph-brand{font-weight:700;letter-spacing:.12em;font-size:13px}.ph-brand span{color:var(--accent);letter-spacing:0;margin-left:6px;font-weight:500}
  .ph-x{background:none;border:0;color:var(--muted);font-size:18px;cursor:pointer;line-height:1}
  .ph-row{display:flex;gap:12px;align-items:center}
  .ph-ring{--p:0;--c:var(--mid);width:78px;height:78px;border-radius:50%;flex:none;display:grid;place-items:center;background:conic-gradient(var(--c) calc(var(--p)*1%),var(--panel2) 0)}
  .ph-ring div{width:62px;height:62px;border-radius:50%;background:var(--bg);display:grid;place-items:center;font-weight:700;font-size:18px}
  .ph-band{font-weight:600;font-size:15px}.ph-detail{color:var(--muted);font-size:12px;margin-top:2px}
  .ph-ev{display:inline-block;margin-top:6px;font-size:11px;font-weight:600;padding:1px 8px;border-radius:99px;background:var(--panel2);border:1px solid var(--line);color:var(--muted)}
  .ph-h{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin:14px 0 6px;font-weight:600}
  .ph-reason{margin-bottom:9px}.ph-rtop{display:flex;justify-content:space-between;gap:6px;font-weight:600;font-size:13px}
  .ph-chip{font-size:10.5px;padding:1px 7px;border-radius:99px;white-space:nowrap}.ph-chip.ai{background:#3a1d14;color:var(--ai)}.ph-chip.human{background:#13301f;color:var(--human)}
  .ph-bar{height:4px;background:var(--panel2);border-radius:9px;margin:5px 0}.ph-bar i{display:block;height:100%;border-radius:9px}
  .ph-p{color:var(--muted);font-size:12px;margin:0}
  .ph-ex{display:flex;gap:5px;flex-wrap:wrap;margin-top:5px}.ph-ex span{background:var(--panel2);border:1px solid var(--line);border-radius:6px;padding:0 7px;font-size:12.5px;color:var(--text)}
  .ph-warn{background:#2a2410;border:1px solid #4a3f14;color:#e9d58a;border-radius:9px;padding:6px 10px;font-size:12px;margin-top:10px}
  .ph-err{background:#2a1414;border:1px solid #4a1f1f;color:#ff9a9a;border-radius:9px;padding:8px 12px;font-size:13px}
  .ph-pass{border-left:3px solid var(--line);padding:4px 9px;margin:5px 0;background:var(--panel2);border-radius:0 8px 8px 0;font-size:12.5px;line-height:1.6}
  .ph-pass small{display:block;color:var(--muted);font-size:10.5px}
  .ph-disc{margin-top:12px;color:var(--muted);font-size:11px;border-left:2px solid var(--line);padding-left:8px}
  .ph-spin{display:flex;gap:10px;align-items:center;color:var(--muted);font-size:13px}
  .ph-spin i{width:16px;height:16px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:phs 0.9s linear infinite}
  @keyframes phs{to{transform:rotate(360deg)}}
  a.ph-link{color:var(--accent);font-size:12px;text-decoration:none}
  details summary{cursor:pointer;color:var(--muted);font-size:12px}
  `;

  function h(tag, props, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(props || {})) {
      if (k === "class") el.className = v;
      else if (k === "style") el.setAttribute("style", v);
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return el;
  }
  const bandColor = (k) => (k === "review" ? "var(--ai)" : k === "human_like" ? "var(--human)" : "var(--mid)");
  const passColor = (p) => (p >= 0.8 ? "var(--ai)" : p <= 0.2 ? "var(--human)" : p >= 0.5 ? "#c98f4d" : "#6fae8a");

  function renderLoading(msg) {
    return h("div", { class: "ph-spin" }, h("i"), h("span", {}, msg || "Analyzing…"));
  }
  function renderError(msg) {
    return h("div", { class: "ph-err" }, msg);
  }
  // Works with every server version: missing optional fields (evidence, chunks, examples ...) are simply skipped.
  function renderResult(j, opts) {
    opts = opts || {};
    if (!j || typeof j.ai_probability !== "number") throw new Error("The server's reply was not in the expected format.");
    const pct = Math.round(j.ai_probability * 100);
    const band = j.band || { key: "inconclusive", label: "Result", detail: "" };
    const color = bandColor(band.key);
    const words = (j.stats && j.stats.words) || (j.evidence && j.evidence.words) || null;
    const ev = j.evidence;
    const evText = ev && ev.label ? `${ev.label}${words ? " · " + words + " words" : ""}${ev.chunks > 1 ? " · " + ev.chunks + " passages" : ""}` : (words ? `${words} words` : "");
    const frag = h("div", {});
    const ring = h("div", { class: "ph-ring", style: `--c:${color}` }, h("div", {}, `${pct}%`));
    frag.append(
      h("div", { class: "ph-row" }, ring,
        h("div", {},
          h("div", { class: "ph-band", style: `color:${color}` }, band.label),
          band.detail ? h("div", { class: "ph-detail" }, band.detail) : null,
          evText ? h("span", { class: "ph-ev" }, evText) : null)),
    );
    requestAnimationFrame(() => ring.style.setProperty("--p", String(pct)));
    for (const w of j.warnings || []) frag.append(h("div", { class: "ph-warn" }, w));
    if ((j.reasons || []).length) frag.append(h("div", { class: "ph-h" }, "Why this score"));
    for (const r of j.reasons || []) {
      const impact = typeof r.impact === "number" ? r.impact : 0;
      frag.append(
        h("div", { class: "ph-reason" },
          h("div", { class: "ph-rtop" }, h("span", {}, r.label || r.feature || ""),
            h("span", { class: `ph-chip ${r.direction === "ai" ? "ai" : "human"}` }, `${r.direction === "ai" ? "toward AI-like" : "toward human-like"} · ${impact}%`)),
          h("div", { class: "ph-bar" }, h("i", { style: `width:${Math.max(4, impact)}%;background:${r.direction === "ai" ? "var(--ai)" : "var(--human)"}` })),
          r.text ? h("p", { class: "ph-p" }, r.text) : null,
          r.examples && r.examples.length ? h("div", { class: "ph-ex" }, r.examples.map((e) => h("span", {}, e))) : null));
    }
    if (j.chunks && j.chunks.length) {
      const d = h("details", {}, h("summary", {}, `Passage by passage (${j.chunks.length})`));
      for (const [i, c] of j.chunks.entries()) {
        d.append(h("div", { class: "ph-pass", style: `border-left-color:${passColor(c.probability)}` },
          h("small", {}, `Passage ${i + 1} · ${c.words} words · ${Math.round(c.probability * 100)}%`), c.text));
      }
      frag.append(d);
    }
    if (j.disclaimer) frag.append(h("div", { class: "ph-disc" }, j.disclaimer));
    if (opts.fullReportUrl) frag.append(h("div", { style: "margin-top:8px" }, h("a", { class: "ph-link", href: opts.fullReportUrl, target: "_blank", rel: "noopener noreferrer" }, "Open the full app ›")));
    return frag;
  }
  // Never leave a spinner on screen: any rendering problem becomes a readable message.
  function safeResult(j, opts) {
    try { return renderResult(j, opts); }
    catch (e) { return renderError((e && e.message) || "Could not display the result.") ; }
  }
  window.ProhoriUI = { CSS, h, renderLoading, renderError, renderResult, safeResult };
})();
