// Review desk client. All page text is set with textContent (never innerHTML)
// so user text and model output cannot inject markup.
"use strict";

const $ = (id) => document.getElementById(id);
let session = null;
let view = "rules";

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) if (c != null) node.append(c instanceof Node ? c : String(c));
  return node;
}

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({ error: "Unexpected reply from the server." }));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function showError(msg) {
  $("error").textContent = msg || "";
  $("error").hidden = !msg;
}

async function health() {
  try {
    const h = await api("/api/health");
    const box = $("health");
    box.replaceChildren(
      el("span", { class: "dot" + (h.ollama ? " on" : "") }), `Model ${h.model}${h.ollama ? "" : " (offline)"}`,
      el("span", { class: "dot" + (h.analyzer ? " on" : "") }), h.analyzer ? "Analyzer ready" : "Analyzer missing",
    );
    if (!h.ollama) $("mode").value = "rules";
  } catch { $("health").textContent = "server unreachable"; }
}

function renderDiff(target, segments, side) {
  target.replaceChildren(...segments
    .filter((s) => s.op === "equal" || (side === "left" ? s.op === "delete" : s.op === "insert"))
    .map((s) => s.op === "equal" ? document.createTextNode(s.text) : el(side === "left" ? "del" : "ins", {}, s.text)));
}

function renderEdits() {
  const list = $("edits");
  const applied = session.edits.filter((e) => e.applied);
  $("edit-count").textContent = `${applied.length} applied · ${session.edits.length - applied.length} skipped`;
  if (!session.edits.length) { list.replaceChildren(el("li", { class: "empty" }, "No rule edits proposed.")); return; }
  list.replaceChildren(...session.edits.map((e) => {
    const cls = ["edit", e.applied ? (e.user === "rejected" ? "rejected" : "applied") : "skipped"].join(" ");
    const change = el("div", { class: "change" }, el("del", {}, e.from), " → ", e.to ? el("ins", {}, e.to) : el("em", {}, "(removed)"));
    const meta = el("div", { class: "meta" },
      `${e.rule} · confidence ${e.confidence} · risk ${e.risk}` + (e.probability != null ? ` · picked p=${e.probability}` : ""));
    const li = el("li", { class: cls }, el("div", { class: "what" }, e.applied ? e.label : `Skipped: ${e.label}`), change);
    if (e.applied) {
      li.append(el("div", {}, e.reason), meta, el("div", { class: "actions" },
        decisionButton(e, "accept", "Keep"), decisionButton(e, "reject", "Reject"),
        e.user !== "pending" ? decisionButton(e, "reset", "Clear") : null));
    } else {
      li.append(el("div", { class: "meta" }, e.skipped_because || ""));
    }
    return li;
  }));
}

function decisionButton(e, decision, label) {
  const pressed = (decision === "accept" && e.user === "accepted") || (decision === "reject" && e.user === "rejected");
  return el("button", { "aria-pressed": String(pressed), onclick: () => act(`/api/session/${session.id}/edit`, { edit_id: e.id, decision }) }, label);
}

function checkBlock(title, rep) {
  const wrap = el("div", {});
  if (!rep) { wrap.append(el("p", { class: "empty" }, `${title}: no changes to check.`)); return wrap; }
  const badge = rep.is_valid ? el("span", { class: "badge ok" }, "passes") : el("span", { class: "badge bad" }, "meaning risk");
  wrap.append(el("p", {}, badge, el("strong", {}, title),
    ` — score ${rep.overall_score.toFixed(2)}, confidence ${rep.confidence.toFixed(2)}`));
  const items = [
    ...rep.errors.map((t) => ["critical", t]),
    ...rep.warnings.map((t) => [t.startsWith("[major]") ? "major" : "minor", t.replace(/^\[\w+\]\s*/, "")]),
  ];
  if (items.length) wrap.append(el("ul", { class: "checklist" }, ...items.map(([c, t]) => el("li", { class: c }, t))));
  return wrap;
}

function renderChecks() {
  const box = $("checks");
  box.replaceChildren(checkBlock("Rule version", session.rules_validation));
  if (session.settings.mode !== "rules") {
    for (const c of session.chunks) {
      const src = { llm: "model", repaired: "model (repaired)", rules: "rule version used", original: "original kept" }[c.source] || c.source;
      const head = el("p", {}, el("span", { class: "badge " + (c.success ? "ok" : "warn") }, src), `Part ${c.index + 1}`,
        c.error ? ` — ${c.error}` : "");
      box.append(head);
      if (c.validation) box.append(checkBlock("Model version", c.validation));
      const extra = [
        ...c.model_rejected.map((r) => `Model declined ${r.id}: ${r.reason || "no reason given"}`),
        ...c.uncertain.map((u) => `Model unsure: ${u}`), ...c.warnings,
      ];
      if (extra.length) box.append(el("ul", { class: "checklist" }, ...extra.map((t) => el("li", { class: "minor" }, t))));
    }
  }
}

function render() {
  $("results").hidden = false;
  const diff = view === "rules" ? session.diff_rules : session.diff_model;
  renderDiff($("original"), diff, "left");
  renderDiff($("rewrite"), diff, "right");
  $("rewrite-title").textContent = view === "rules" ? "Rule version" : "Model version";
  for (const t of document.querySelectorAll(".tabs button")) t.setAttribute("aria-selected", String(t.dataset.view === view));
  document.querySelector('.tabs [data-view="model"]').disabled = session.settings.mode === "rules";
  $("undo").disabled = !session.can_undo;
  $("redo").disabled = !session.can_redo;
  renderEdits();
  renderChecks();
}

async function act(path, body) {
  try { session = await api(path, body); render(); showError(""); }
  catch (err) { showError(err.message); }
}

$("run").addEventListener("click", async () => {
  const btn = $("run");
  btn.disabled = true; btn.lastChild.textContent = " Rewriting…"; showError(""); $("saved").hidden = true;
  try {
    session = await api("/api/rewrite", {
      text: $("input").value, mode: $("mode").value, intensity: $("intensity").value,
      strictness: $("strictness").value, seed: $("seed").value === "" ? null : Number($("seed").value),
      protected: $("protected").value.split(",").map((s) => s.trim()).filter(Boolean), tone: $("tone").value, style: $("style").value,
    });
    view = session.settings.mode === "rules" ? "rules" : "model";
    if (session.settings.mode === "hybrid" && session.status === "failed") {
      view = "rules";
      showError("Local model is offline or unavailable. Showing rule-based rewrite.");
    }
    $("final").value = view === "rules" ? session.rules_text : session.model_text;
    render();
  } catch (err) { showError(err.message); }
  finally { btn.disabled = false; btn.lastChild.textContent = " Rewrite"; }
});

for (const t of document.querySelectorAll(".tabs button")) t.addEventListener("click", () => { view = t.dataset.view; render(); });
$("undo").addEventListener("click", () => act(`/api/session/${session.id}/undo`, {}));
$("redo").addEventListener("click", () => act(`/api/session/${session.id}/redo`, {}));
for (const b of document.querySelectorAll("[data-use]")) b.addEventListener("click", () => {
  $("final").value = { rules: session.rules_text, model: session.model_text, original: session.original }[b.dataset.use];
});

async function sendFeedback(verdict) {
  try {
    const r = await api(`/api/session/${session.id}/feedback`, {
      verdict, final_text: verdict === "reject" ? null : $("final").value,
      rating: $("rating").value || null, comment: $("comment").value,
    });
    const v = r.final_validation;
    $("saved").textContent = verdict === "reject" ? "Saved: rewrite rejected." :
      `Saved.${v && !v.is_valid ? " Note: your final text may change the meaning — " + (v.errors[0] || v.warnings[0]) : ""}`;
    $("saved").hidden = false;
    if (verdict !== "reject") navigator.clipboard?.writeText($("final").value).catch(() => {});
  } catch (err) { showError(err.message); }
}
$("accept").addEventListener("click", () => sendFeedback("accept"));
$("reject").addEventListener("click", () => sendFeedback("reject"));

health();
