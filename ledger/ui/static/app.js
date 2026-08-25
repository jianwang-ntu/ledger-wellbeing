/* Ledger's interface behaviour.

   Deliberately plain: no framework, no bundler, no build step. That is not
   minimalism for its own sake — `export/INCREMENT_9_PREREGISTRATION.md` R9-8(b)
   requires that every byte the page loads comes from this repository, and the
   cheapest way to be sure of that is for there to be nothing to fetch.

   Two habits are load-bearing and are worth naming so they are not undone later:

   * Every piece of the person's own writing reaches the DOM through
     `textContent`, never `innerHTML`. Their journal is the untrusted input here,
     and it is also the thing most worth not corrupting.
   * Every state change that a sighted user learns from a repaint is also written
     into a live region (R9-6). If a change is only visible, it did not happen for
     everyone.
*/

"use strict";

const TOKEN = document.querySelector('meta[name="ledger-token"]').content;

/* The run token above is in the page, so any local process can read it. It is a
   same-origin guard, not an authenticator. SESSION is minted by the server on a
   successful unlock and returned only in that response body — it is held in this
   closure, never written to the DOM, localStorage, sessionStorage or a cookie,
   so a second local client that did not supply the passphrase cannot obtain it.
   Round-1 audit finding F-04 is what this exists for. */
let SESSION = null;

const VIEWS = ["unlock", "write", "entry", "history", "report", "settings"];

const el = (id) => document.getElementById(id);
const statusLine = el("status");

let state = null;
let latest = null;

/* ---- transport ---------------------------------------------------------- */

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "X-Ledger-Token": TOKEN,
      "Content-Type": "application/json",
      ...(SESSION ? { "X-Ledger-Session": SESSION } : {}),
      ...(options.headers || {}),
    },
    /* Same-origin by construction: there is no other origin this page may reach.
       Stated explicitly so a later edit has to argue with it. */
    credentials: "omit",
    cache: "no-store",
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    /* 401 on a journal endpoint means the server dropped the key — either the
       idle timer fired or this client never had a session. Forget ours and send
       the user back to the passphrase rather than leaving a dead interface. */
    if (response.status === 401 && SESSION !== null) {
      SESSION = null;
      state = null;
      showView("unlock");
      announce("The journal locked itself. Enter the passphrase to reopen it.");
    }
    throw new Error(payload.error || `request failed (${response.status})`);
  }
  return payload;
}

const post = (path, body) => api(path, { method: "POST", body: JSON.stringify(body || {}) });

/* ---- announcements ------------------------------------------------------ */

function announce(message) {
  /* Clearing first makes a repeat of the same message announce again; some
     screen readers ignore an identical live-region write. */
  statusLine.textContent = "";
  window.setTimeout(() => { statusLine.textContent = message; }, 30);
}

function setError(node, message) {
  node.textContent = message || "";
}

/* ---- navigation --------------------------------------------------------- */

function showView(name, { focus = true } = {}) {
  VIEWS.forEach((view) => {
    const section = el(`view-${view}`);
    if (section) section.hidden = view !== name;
  });
  document.querySelectorAll(".rail__link").forEach((link) => {
    if (link.dataset.view === name) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  if (focus) {
    const heading = el(`view-${name}`).querySelector("h2");
    if (heading) heading.focus();
  }
}

document.querySelectorAll(".rail__link").forEach((link) => {
  link.addEventListener("click", () => {
    const target = link.dataset.view;
    if (!state || !state.unlocked) {
      showView("unlock");
      announce("Open your journal first.");
      return;
    }
    showView(target);
    if (target === "history") refreshHistory();
  });
});

/* ---- rendering ---------------------------------------------------------- */

function makeSpanNodes(text, spans) {
  /* Rebuild the person's entry from its span offsets, so the words on screen are
     exactly the words on disk. The gaps between spans are emitted verbatim: a
     renderer that dropped whitespace would quietly rewrite the entry. */
  const fragment = document.createDocumentFragment();
  const ordered = [...spans].sort((a, b) => a.start - b.start);
  const magnitudes = ordered.map((s) => Math.abs(s.attribution));
  const peak = Math.max(...magnitudes, 1e-9);
  let cursor = 0;
  ordered.forEach((span) => {
    if (span.start > cursor) fragment.append(text.slice(cursor, span.start));
    const node = document.createElement("span");
    node.textContent = span.text;
    const share = Math.abs(span.attribution) / peak;
    node.className = `span ${span.attribution >= 0 ? "span--pos" : "span--neg"}`;
    node.style.setProperty("--t", `${(1 + Math.round(share * 3))}px`);
    fragment.append(node);
    cursor = span.end;
  });
  if (cursor < text.length) fragment.append(text.slice(cursor));
  return fragment;
}

function spanTable(spans, limit = 3) {
  const ranked = [...spans].sort((a, b) => Math.abs(b.attribution) - Math.abs(a.attribution));
  const wrap = document.createElement("div");
  wrap.className = "tablewrap";
  wrap.tabIndex = 0;
  wrap.setAttribute("role", "region");
  wrap.setAttribute("aria-label", "Largest contributions from your own words, scrollable");

  const table = document.createElement("table");
  table.className = "table";
  const caption = document.createElement("caption");
  caption.className = "visually-hidden";
  caption.textContent = "Each phrase and the amount it contributed to this signal";
  table.append(caption);

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Your words", "Direction", "Contribution"].forEach((label) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = label;
    headRow.append(th);
  });
  thead.append(headRow);
  table.append(thead);

  const tbody = document.createElement("tbody");
  ranked.slice(0, limit).forEach((span) => {
    const row = document.createElement("tr");
    const words = document.createElement("td");
    words.textContent = span.text.trim();
    const direction = document.createElement("td");
    /* The direction is spelled out rather than left to the underline style, so
       the information survives without colour, without CSS and out loud. */
    direction.textContent = span.attribution >= 0 ? "raised" : "lowered";
    const value = document.createElement("td");
    value.className = "num";
    value.textContent = span.attribution >= 0
      ? `+${span.attribution.toFixed(3)}` : span.attribution.toFixed(3);
    row.append(words, direction, value);
    tbody.append(row);
  });
  table.append(tbody);
  wrap.append(table);
  return { wrap, ranked };
}

function renderDimension(dim, text) {
  const card = document.createElement("section");
  card.className = "dimension";
  card.setAttribute("aria-labelledby", `dim-${dim.dimension}`);

  const head = document.createElement("div");
  head.className = "dimension__head";
  const name = document.createElement("h3");
  name.className = "dimension__name";
  name.id = `dim-${dim.dimension}`;
  name.style.margin = "0";
  name.textContent = dim.label;
  const score = document.createElement("p");
  score.className = "dimension__score";
  score.style.margin = "0";
  score.textContent = dim.probability.toFixed(2);
  head.append(name, score);
  card.append(head);

  if (!dim.established) {
    const badge = document.createElement("p");
    badge.style.margin = "0";
    const tag = document.createElement("span");
    tag.className = "badge badge--warn";
    tag.textContent = "Not established";
    badge.append(tag);
    card.append(badge);

    const note = document.createElement("p");
    note.className = "dimension__note";
    /* R9-9. The number and its basis travel with the label every time it is
       shown. A prettier surface may not be a quieter one about what it rests on.

       The sentence after the number comes from `ledger/app/evidence.py` and is
       not restated here: an earlier draft prefixed its own "against the 0.70
       threshold fixed before it was measured", which the rendered card then said
       twice (DEFECT-INC9-005). One source, one sentence. */
    note.textContent = `Held-out AUC ${dim.held_out_auc.toFixed(3)}. ${dim.evidence_note || ""}`;
    card.append(note);
  }

  const attributed = document.createElement("p");
  attributed.className = "attributed";
  attributed.append(makeSpanNodes(text, dim.spans));
  card.append(attributed);

  const { wrap, ranked } = spanTable(dim.spans);
  card.append(wrap);

  const rest = ranked.slice(3).reduce((total, span) => total + span.attribution, 0);
  const arithmetic = document.createElement("p");
  arithmetic.className = "arith";
  const parts = [];
  if (ranked.length > 3) parts.push(`${rest >= 0 ? "+" : ""}${rest.toFixed(3)} from ${ranked.length - 3} more`);
  parts.push(`${dim.structural_attribution >= 0 ? "+" : ""}${dim.structural_attribution.toFixed(3)} structural`);
  parts.push(`${dim.bias >= 0 ? "+" : ""}${dim.bias.toFixed(3)} offset`);
  arithmetic.textContent = `Everything above ${parts.join(", ")} = logit `
    + `${dim.logit >= 0 ? "+" : ""}${dim.logit.toFixed(3)} `
    + `(residual ${dim.additivity_residual.toExponential(1)})`;
  card.append(arithmetic);

  return card;
}

function renderAnalysis(entry) {
  latest = entry;
  const analysis = entry.analysis;
  const crisis = el("crisis");
  const result = el("entry-result");
  el("entry-empty").hidden = true;

  if (!analysis.scored) {
    result.hidden = true;
    crisis.hidden = false;
    el("crisis-reason").textContent = analysis.reason_not_scored || "";
    const list = el("crisis-helplines");
    list.replaceChildren();
    (analysis.routed.helplines || []).forEach((line) => {
      const item = document.createElement("li");
      const strong = document.createElement("strong");
      strong.textContent = line.name;
      item.append(strong, ` — ${line.contact} (${line.hours})`);
      list.append(item);
    });
    announce("This entry matched a crisis rule. It was stored but not scored, and helplines are shown.");
    return;
  }

  crisis.hidden = true;
  result.hidden = false;
  el("entry-contract").textContent = analysis.contract;
  /* R8-7 carried forward: the page that shows the scores also names what they
     were evaluated on, so a reader does not have to visit another view to find
     out that the evaluation is 25 withheld anchor-sentence pairs written for
     this repository rather than clinical data. */
  el("entry-basis").textContent = state ? `Evaluated on: ${state.evaluation_basis}` : "";
  const container = el("dimensions");
  container.replaceChildren();
  analysis.dimensions.forEach((dim) => container.append(renderDimension(dim, entry.text)));
  announce(`Entry saved and explained across ${analysis.dimensions.length} signals.`);
}

/* ---- history and report -------------------------------------------------- */

async function refreshHistory() {
  const { entries } = await api("/api/entries");
  const body = el("history-body");
  body.replaceChildren();
  entries.forEach((entry) => {
    const row = document.createElement("tr");
    const when = document.createElement("td");
    when.className = "num";
    when.textContent = entry.written_at;
    const outcome = document.createElement("td");
    outcome.textContent = entry.analysis.scored ? "Scored" : "Routed to help";
    const preview = document.createElement("td");
    preview.textContent = entry.text.replace(/\s+/g, " ").slice(0, 64);
    row.append(when, outcome, preview);
    body.append(row);
  });
  el("history-empty").hidden = entries.length > 0;
  el("history-table").hidden = entries.length === 0;
}

/* ---- forms --------------------------------------------------------------- */

el("unlock-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setError(el("unlock-error"), "");
  try {
    const result = await post("/api/unlock", { passphrase: el("passphrase").value });
    el("passphrase").value = "";
    SESSION = result.session || null;
    state = await api("/api/state");
    renderSettings();
    showView("write");
    announce(result.created
      ? "New encrypted journal created on this machine."
      : `Journal opened. ${result.entries} entr${result.entries === 1 ? "y" : "ies"} stored.`);
  } catch (error) {
    setError(el("unlock-error"), error.message);
    announce("The journal did not open.");
  }
});

el("entry-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setError(el("entry-error"), "");
  const granularity = document.querySelector('input[name="granularity"]:checked').value;
  try {
    announce("Reading your entry on this machine…");
    const entry = await post("/api/entry", { text: el("entry-text").value, granularity });
    el("entry-text").value = "";
    showView("entry");
    renderAnalysis(entry);
  } catch (error) {
    setError(el("entry-error"), error.message);
    announce("The entry was not saved.");
  }
});

el("report-refresh").addEventListener("click", async () => {
  try {
    const { report, entries } = await api("/api/report");
    el("report-text").textContent = report;
    announce(`Report rendered from ${entries} entr${entries === 1 ? "y" : "ies"}.`);
  } catch (error) {
    el("report-text").textContent = "";
    announce(error.message);
  }
});

el("wipe-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  setError(el("wipe-error"), "");
  try {
    const result = await post("/api/wipe", { confirm: el("wipe-confirm").value });
    el("wipe-confirm").value = "";
    SESSION = null;   /* the server dropped the key; do not keep a stale handle */
    state = await api("/api/state");
    renderSettings();
    showView("unlock");
    announce(result.wiped
      ? "Journal overwritten and removed from this machine."
      : `Nothing was destroyed: ${result.reason}`);
  } catch (error) {
    setError(el("wipe-error"), error.message);
    announce("Nothing was destroyed.");
  }
});

/* ---- settings ------------------------------------------------------------ */

function renderSettings() {
  if (!state) return;
  el("fact-store").textContent = state.store;
  el("evidence-basis").textContent = state.evaluation_basis;
  el("contract").textContent = "Nothing you write here leaves this machine.";
  el("colophon-contract").textContent = state.contract;

  const body = el("evidence-body");
  body.replaceChildren();
  state.dimensions.forEach((dim) => {
    const row = document.createElement("tr");
    const name = document.createElement("td");
    name.textContent = dim.label;
    const auc = document.createElement("td");
    auc.className = "num";
    auc.textContent = dim.held_out_auc === null ? "not measured" : dim.held_out_auc.toFixed(3);
    const established = document.createElement("td");
    established.textContent = dim.established ? "Yes" : "No — below the 0.70 threshold";
    row.append(name, auc, established);
    body.append(row);
  });
}

/* ---- start --------------------------------------------------------------- */

(async function start() {
  state = await api("/api/state");
  renderSettings();
  if (state.unlocked) {
    showView("write", { focus: false });
  } else {
    showView("unlock", { focus: false });
    el("unlock-lede").textContent = state.store_exists
      ? "This machine already holds an encrypted journal. Enter its passphrase to open it."
      : "No journal exists on this machine yet. The passphrase you choose now encrypts it, is not stored anywhere, and cannot be recovered.";
    el("unlock-submit").textContent = state.store_exists ? "Open journal" : "Create journal";
  }
  document.body.dataset.ready = "true";
})();
