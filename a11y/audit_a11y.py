"""Measure increment 9's accessibility rules against the running interface.

R9-1 keyboard completeness, R9-2 visible focus, R9-3 axe-core, R9-4 contrast and
forced colours, R9-5 reduced motion, R9-6 announcements, R9-8(b) no third-party
request. Every threshold is fixed in `export/INCREMENT_9_PREREGISTRATION.md`,
which was committed before this file existed.

Why a real browser rather than jsdom
------------------------------------
Contrast is a property of what the compositor drew. jsdom has no layout and no
compositing, so `color-contrast` comes back `incomplete` there — the check that
matters most for a designed interface is exactly the one a DOM-only harness
cannot run. Playwright drives the same Chromium a judge would open, and R9-4
requires `color-contrast` to have actually *run*, so the weaker harness could not
have passed this rule by omission.

Scope, stated so the artifact is not read as more than it is
------------------------------------------------------------
This measures the machine-checkable half of accessibility: rule violations,
keyboard operability, focus visibility, announced state changes, motion and
forced colours. It does **not** measure what a screen-reader user experiences.
No assistive technology is driven here and none is claimed;
`export/INCREMENT_9_PREREGISTRATION.md` records that as NEEDS_HUMAN.

axe-core is MPL-2.0 and Playwright is Apache-2.0. Both are measurement tools,
imported by this harness only; neither is a dependency of the application.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image                                            # noqa: E402
from playwright.sync_api import sync_playwright                  # noqa: E402

from ledger.ui.server import serve_in_thread                     # noqa: E402

AXE = ROOT / "a11y" / "node_modules" / "axe-core" / "axe.min.js"
OUT = ROOT / "artifacts" / "a11y"
SHOTS = OUT / "screens"

#: WCAG 2.0 and 2.1, levels A and AA. Fixed in the pre-registration; not narrowed
#: afterwards to make a number look better.
AXE_TAGS = ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]

#: R9-2. A focus indicator has to change at least this share of the pixels in the
#: element's own box. 0.5% of a small control is a few dozen pixels — enough to
#: rule out a repaint nobody can see, low enough that a thin ring still passes.
FOCUS_MIN_CHANGED = 0.005

#: A per-channel difference below this is compositor noise, not an indicator.
CHANNEL_TOLERANCE = 8

PASSPHRASE = "an accessibility harness passphrase"

ORDINARY_ENTRY = (
    "I slept badly again and dragged through the whole day. The meeting went "
    "fine but I could not settle afterwards and kept going over it."
)
CRISIS_ENTRY = "I want to k1ll myself."

VIEWS = ["unlock", "write", "entry", "history", "report", "settings"]

#: Counts real pointer input. A button activated with Enter still fires a `click`
#: event, but with `detail === 0`; a mouse click has `detail >= 1`. Counting
#: `click` alone would have made R9-1 unpassable and counting nothing would have
#: made it unfalsifiable, so both are recorded separately.
POINTER_COUNTER = """
window.__pointerEvents = {pointerdown: 0, mousedown: 0, mouseup: 0, clickWithDetail: 0};
for (const type of ['pointerdown', 'mousedown', 'mouseup']) {
  window.addEventListener(type, (e) => { if (e.isTrusted) window.__pointerEvents[type] += 1; }, true);
}
window.addEventListener('click', (e) => {
  if (e.isTrusted && e.detail > 0) window.__pointerEvents.clickWithDetail += 1;
}, true);
"""


# ---------------------------------------------------------------------------
# helpers


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tab_until(page, predicate: str, limit: int = 80) -> int:
    """Press Tab until `predicate` is true of the page. Keyboard input only."""
    for pressed in range(1, limit + 1):
        page.keyboard.press("Tab")
        if page.evaluate(predicate):
            return pressed
    raise AssertionError(f"never reached: {predicate} after {limit} Tab presses")


def active_id(page) -> str:
    return page.evaluate("() => document.activeElement ? document.activeElement.id : ''")


def run_axe(page, context_label: str) -> dict:
    result = page.evaluate(
        """(tags) => axe.run(document, {
             runOnly: {type: 'tag', values: tags},
             resultTypes: ['violations', 'incomplete']
           }).then(r => ({
             violations: r.violations.map(v => ({
               id: v.id, impact: v.impact, help: v.help,
               nodes: v.nodes.map(n => ({target: n.target, summary: n.failureSummary}))
             })),
             incomplete: r.incomplete.map(v => ({
               id: v.id, impact: v.impact, help: v.help,
               nodes: v.nodes.map(n => ({target: n.target, summary: n.failureSummary}))
             })),
             passes: r.passes.map(p => p.id),
             inapplicable: r.inapplicable.map(p => p.id)
           }))""",
        AXE_TAGS,
    )
    result["view"] = context_label
    result["violation_count"] = len(result["violations"])
    result["incomplete_count"] = len(result["incomplete"])
    result["color_contrast_ran"] = (
        "color-contrast" in result["passes"]
        or any(v["id"] == "color-contrast" for v in result["violations"])
        or any(v["id"] == "color-contrast" for v in result["incomplete"])
    )
    result["color_contrast_passed"] = "color-contrast" in result["passes"]
    return result


def diff_ratio(a: bytes, b: bytes) -> float:
    """Share of pixels that differ by more than the tolerance in any channel."""
    left = Image.open(io.BytesIO(a)).convert("RGB")
    right = Image.open(io.BytesIO(b)).convert("RGB")
    if left.size != right.size:
        return 1.0
    lp, rp = left.load(), right.load()
    width, height = left.size
    changed = 0
    for y in range(height):
        for x in range(width):
            l, r = lp[x, y], rp[x, y]
            if (abs(l[0] - r[0]) > CHANNEL_TOLERANCE or abs(l[1] - r[1]) > CHANNEL_TOLERANCE
                    or abs(l[2] - r[2]) > CHANNEL_TOLERANCE):
                changed += 1
    return changed / float(width * height)


def motion_scan(page) -> list[dict]:
    """Every element whose computed animation or transition is not zero."""
    return page.evaluate("""() => {
      const bad = [];
      for (const node of document.querySelectorAll('*')) {
        const s = getComputedStyle(node);
        const durations = (s.animationDuration + ',' + s.transitionDuration)
          .split(',').map(v => v.trim()).filter(Boolean);
        const nonZero = durations.filter(v => v !== '0s' && v !== '0ms');
        if (nonZero.length) {
          bad.push({tag: node.tagName.toLowerCase(), cls: node.className.toString().slice(0, 60),
                    durations: nonZero});
        }
      }
      return bad;
    }""")


def focusable_boxes(page, view: str) -> int:
    """Tag every focusable element inside the visible view and return the count."""
    return page.evaluate("""(view) => {
      document.querySelectorAll('[data-a11y-idx]').forEach(n => n.removeAttribute('data-a11y-idx'));
      const root = document.getElementById('view-' + view);
      const sel = 'a[href], button, input:not([type=hidden]), textarea, select, [tabindex]';
      const nodes = [...root.querySelectorAll(sel)]
        .filter(n => !n.disabled && n.tabIndex >= 0 && n.getClientRects().length);
      nodes.forEach((n, i) => n.setAttribute('data-a11y-idx', String(i)));
      return nodes.length;
    }""", view)


def measure_focus(page, view: str) -> dict:
    """R9-2, optically, over every focusable element in the visible view."""
    count = focusable_boxes(page, view)
    elements = []
    for idx in range(count):
        if idx == 0:
            page.evaluate("() => { if (document.activeElement) document.activeElement.blur(); }")
            page.evaluate("() => window.scrollTo(0, 0)")
            tab_until(page, "() => document.activeElement "
                            "&& document.activeElement.getAttribute('data-a11y-idx') === '0'")
        else:
            page.evaluate("(i) => document.querySelector(`[data-a11y-idx=\"${i}\"]`).focus()", idx - 1)
            page.keyboard.press("Tab")
        landed = page.evaluate("() => document.activeElement "
                               "? document.activeElement.getAttribute('data-a11y-idx') : null")
        if landed != str(idx):
            elements.append({"index": idx, "reached": False, "landed_on": landed})
            continue

        box = page.evaluate("""(i) => {
          const n = document.querySelector(`[data-a11y-idx="${i}"]`);
          const r = n.getBoundingClientRect();
          return {x: r.x, y: r.y, w: r.width, h: r.height,
                  tag: n.tagName.toLowerCase(), id: n.id || '', cls: n.className.toString().slice(0,50)};
        }""", idx)
        pad = 8
        clip = {
            "x": max(0.0, box["x"] - pad), "y": max(0.0, box["y"] - pad),
            "width": box["w"] + pad * 2, "height": box["h"] + pad * 2,
        }
        if clip["width"] < 2 or clip["height"] < 2:
            elements.append({"index": idx, "reached": True, "skipped": "zero-size box"})
            continue
        focused = page.screenshot(clip=clip)
        page.evaluate("() => document.activeElement.blur()")
        unfocused = page.screenshot(clip=clip)
        ratio = diff_ratio(focused, unfocused)
        styles = page.evaluate("""(i) => {
          const s = getComputedStyle(document.querySelector(`[data-a11y-idx="${i}"]`));
          return {outline: s.outlineStyle + ' ' + s.outlineWidth, boxShadow: s.boxShadow};
        }""", idx)
        elements.append({
            "index": idx, "reached": True, "tag": box["tag"], "id": box["id"],
            "class": box["cls"], "changed_pixel_ratio": round(ratio, 5),
            "passes": ratio >= FOCUS_MIN_CHANGED, "unfocused_styles": styles,
        })
    failures = [e for e in elements
                if e.get("reached") and "skipped" not in e and not e.get("passes", False)]
    unreached = [e for e in elements if not e.get("reached")]
    return {"view": view, "focusable": count, "elements": elements,
            "failures": len(failures), "unreachable": len(unreached),
            "verdict": "PASS" if not failures and not unreached else "FAIL"}


# ---------------------------------------------------------------------------
# the keyboard-only walk


def keyboard_flow(page, store: Path, *, label: str) -> dict:
    """R9-1. The whole primary flow, driven with keys and nothing else."""
    steps = []

    def step(name, detail):
        steps.append({"step": name, "detail": detail})

    page.wait_for_function("() => document.body.dataset.ready === 'true'")

    tab_until(page, "() => document.activeElement && document.activeElement.id === 'passphrase'")
    page.keyboard.type(PASSPHRASE)
    step("reach passphrase", "Tab only")
    tab_until(page, "() => document.activeElement && document.activeElement.id === 'unlock-submit'")
    page.keyboard.press("Enter")
    page.wait_for_function("() => !document.getElementById('view-write').hidden", timeout=30000)
    step("create journal", f"store exists: {store.exists()}")

    focused_after_unlock = active_id(page)
    tab_until(page, "() => document.activeElement && document.activeElement.id === 'entry-text'")
    page.keyboard.type(ORDINARY_ENTRY)
    tab_until(page, "() => document.activeElement && document.activeElement.id === 'entry-submit'")
    page.keyboard.press("Enter")
    page.wait_for_function("() => !document.getElementById('entry-result').hidden", timeout=300000)
    dimensions = page.evaluate("() => document.querySelectorAll('.dimension').length")
    scored_status = page.text_content("#status")
    step("write and explain an entry", f"{dimensions} dimension cards rendered")

    # R9-6, part two: the crisis path must announce, not merely repaint.
    tab_until(page, "() => document.activeElement "
                    "&& document.activeElement.dataset.view === 'write'")
    page.keyboard.press("Enter")
    page.wait_for_function("() => !document.getElementById('view-write').hidden")
    tab_until(page, "() => document.activeElement && document.activeElement.id === 'entry-text'")
    page.keyboard.type(CRISIS_ENTRY)
    tab_until(page, "() => document.activeElement && document.activeElement.id === 'entry-submit'")
    page.keyboard.press("Enter")
    page.wait_for_function("() => !document.getElementById('crisis').hidden", timeout=300000)
    crisis_status = page.text_content("#status")
    crisis_helplines = page.evaluate("() => document.querySelectorAll('#crisis-helplines li').length")
    crisis_alert_role = page.get_attribute("#crisis", "role")
    step("crisis entry routes without scoring",
         f"{crisis_helplines} helplines, role={crisis_alert_role}")

    tab_until(page, "() => document.activeElement "
                    "&& document.activeElement.dataset.view === 'history'")
    page.keyboard.press("Enter")
    page.wait_for_function("() => !document.getElementById('view-history').hidden")
    page.wait_for_function("() => document.querySelectorAll('#history-body tr').length >= 2",
                           timeout=30000)
    rows = page.evaluate("() => document.querySelectorAll('#history-body tr').length")
    step("history", f"{rows} rows read back from the encrypted store")

    tab_until(page, "() => document.activeElement "
                    "&& document.activeElement.dataset.view === 'report'")
    page.keyboard.press("Enter")
    page.wait_for_function("() => !document.getElementById('view-report').hidden")
    tab_until(page, "() => document.activeElement && document.activeElement.id === 'report-refresh'")
    page.keyboard.press("Enter")
    page.wait_for_function(
        "() => document.getElementById('report-text').textContent.length > 400", timeout=60000)
    report_chars = page.evaluate("() => document.getElementById('report-text').textContent.length")
    step("render the report", f"{report_chars} characters")

    tab_until(page, "() => document.activeElement "
                    "&& document.activeElement.dataset.view === 'settings'")
    page.keyboard.press("Enter")
    page.wait_for_function("() => !document.getElementById('view-settings').hidden")
    tab_until(page, "() => document.activeElement && document.activeElement.id === 'wipe-confirm'")
    page.keyboard.type("no")
    tab_until(page, "() => document.activeElement && document.activeElement.id === 'wipe-submit'")
    page.keyboard.press("Enter")
    page.wait_for_function("() => document.getElementById('wipe-error').textContent.length > 0",
                           timeout=30000)
    declined = page.text_content("#wipe-error")
    step("decline the wipe", f"store still present: {store.exists()}; message: {declined!r}")

    pointer = page.evaluate("() => window.__pointerEvents")
    return {
        "label": label,
        "steps": steps,
        "pointer_events": pointer,
        "focus_moved_to_after_unlock": focused_after_unlock,
        "status_after_scored_entry": scored_status,
        "status_after_crisis_entry": crisis_status,
        "store_survived_declined_wipe": store.exists(),
        "verdict": "PASS" if (sum(pointer.values()) == 0 and store.exists()) else "FAIL",
    }


def visit_view(page, view: str) -> None:
    """Show a view using the keyboard, or directly for the locked-only view."""
    if view == "unlock":
        page.evaluate("() => { document.querySelectorAll('.view').forEach(v => v.hidden = true);"
                      "document.getElementById('view-unlock').hidden = false; }")
        return
    tab_until(page, f"() => document.activeElement "
                    f"&& document.activeElement.dataset.view === '{view}'")
    page.keyboard.press("Enter")
    page.wait_for_function(f"() => !document.getElementById('view-{view}').hidden")


# ---------------------------------------------------------------------------


def main() -> int:
    if not AXE.exists():
        print(f"axe-core is absent at {AXE}. Run `npm install` in a11y/.", file=sys.stderr)
        return 2
    axe_source = AXE.read_text(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    SHOTS.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "measured_at_utc": now(),
        "rules": "export/INCREMENT_9_PREREGISTRATION.md",
        "harness": {
            "axe_core": json.loads(
                (ROOT / "a11y" / "node_modules" / "axe-core" / "package.json").read_text()
            )["version"],
            "axe_tags": AXE_TAGS,
            "browser": "chromium via playwright",
            "scope_not_measured": [
                "screen-reader output — no assistive technology is driven here and none is claimed",
                "any judgement about whether the design is good, as opposed to conformant",
            ],
        },
    }

    with tempfile.TemporaryDirectory(prefix="ledger-a11y-") as tmp:
        store = Path(tmp) / "journal.enc"
        server = serve_in_thread(store=store, region="SG")
        report["server"] = {"url": server.url, "bound_host": server.bound_host,
                            "port": server.server_port}
        requests_seen: list[dict] = []

        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                context = browser.new_context(viewport={"width": 1280, "height": 900},
                                              reduced_motion="no-preference")
                context.on("request", lambda r: requests_seen.append(
                    {"url": r.url, "resource_type": r.resource_type}))
                page = context.new_page()
                page.add_init_script(axe_source)
                page.add_init_script(POINTER_COUNTER)
                console: list[str] = []
                page.on("console", lambda m: console.append(f"{m.type}: {m.text}"))
                page.goto(server.url, wait_until="load")

                report["keyboard_flow"] = keyboard_flow(page, store, label="default")

                # R9-3/R9-4a: axe on every view, and a screenshot of each for the record.
                axe_results = []
                for view in VIEWS:
                    visit_view(page, view)
                    page.wait_for_timeout(120)
                    axe_results.append(run_axe(page, view))
                    page.screenshot(path=str(SHOTS / f"{view}.png"), full_page=True)
                report["axe"] = axe_results

                # R9-2 on the two views that carry the most controls, plus the
                # entry view, which is the one built from user text at runtime.
                report["focus_visibility"] = [measure_focus(page, v)
                                              for v in ("write", "settings", "entry")]
                report["console"] = console
                context.close()

                # R9-5: a second context that asks for reduced motion.
                rm = browser.new_context(viewport={"width": 1280, "height": 900},
                                         reduced_motion="reduce")
                rm.on("request", lambda r: requests_seen.append(
                    {"url": r.url, "resource_type": r.resource_type}))
                rm_page = rm.new_page()
                rm_page.add_init_script(axe_source)
                rm_page.add_init_script(POINTER_COUNTER)
                rm_page.goto(server.url, wait_until="load")
                rm_page.wait_for_function("() => document.body.dataset.ready === 'true'")
                tab_until(rm_page, "() => document.activeElement "
                                   "&& document.activeElement.id === 'passphrase'")
                rm_page.keyboard.type(PASSPHRASE)
                tab_until(rm_page, "() => document.activeElement "
                                   "&& document.activeElement.id === 'unlock-submit'")
                rm_page.keyboard.press("Enter")
                rm_page.wait_for_function("() => !document.getElementById('view-write').hidden",
                                          timeout=30000)
                motion = []
                for view in ("unlock", "write", "history", "report", "settings"):
                    if view != "unlock":
                        visit_view(rm_page, view)
                    else:
                        rm_page.evaluate(
                            "() => { document.querySelectorAll('.view').forEach(v => v.hidden = true);"
                            "document.getElementById('view-unlock').hidden = false; }")
                    rm_page.wait_for_timeout(80)
                    offenders = motion_scan(rm_page)
                    motion.append({"view": view, "non_zero": offenders})
                report["reduced_motion"] = {
                    "views": motion,
                    "total_non_zero": sum(len(m["non_zero"]) for m in motion),
                    "verdict": "PASS" if not any(m["non_zero"] for m in motion) else "FAIL",
                }
                rm.close()

                # R9-4b: the whole flow again with the platform's colours forced.
                fc_store = Path(tmp) / "forced.enc"
                fc_server = serve_in_thread(store=fc_store, region="SG")
                fc = browser.new_context(viewport={"width": 1280, "height": 900},
                                         forced_colors="active")
                fc.on("request", lambda r: requests_seen.append(
                    {"url": r.url, "resource_type": r.resource_type}))
                fc_page = fc.new_page()
                fc_page.add_init_script(axe_source)
                fc_page.add_init_script(POINTER_COUNTER)
                fc_page.goto(fc_server.url, wait_until="load")
                fc_flow = keyboard_flow(fc_page, fc_store, label="forced-colors")
                fc_axe = []
                for view in VIEWS:
                    visit_view(fc_page, view)
                    fc_page.wait_for_timeout(120)
                    fc_axe.append(run_axe(fc_page, view))
                    fc_page.screenshot(path=str(SHOTS / f"forced-colors-{view}.png"), full_page=True)
                report["forced_colors"] = {
                    "flow": fc_flow, "axe": fc_axe,
                    "violation_total": sum(a["violation_count"] for a in fc_axe),
                }
                fc.close()
                fc_server.shutdown_now()
            finally:
                browser.close()
        server.shutdown_now()

    origin = report["server"]["url"].rstrip("/")
    external = [r for r in requests_seen
                if not (r["url"].startswith(origin) or r["url"].startswith("data:"))]
    report["page_requests"] = {
        "total": len(requests_seen),
        "origin": origin,
        "unique_urls": sorted({r["url"] for r in requests_seen}),
        "external": external,
        "verdict": "PASS" if not external else "FAIL",
    }

    # ---- roll-up ---------------------------------------------------------
    violations = sum(a["violation_count"] for a in report["axe"])
    contrast_ran = all(a["color_contrast_ran"] for a in report["axe"]
                       if a["view"] not in ())
    contrast_passed = all(a["color_contrast_passed"] for a in report["axe"])
    focus_fail = [f for f in report["focus_visibility"] if f["verdict"] != "PASS"]

    report["verdicts"] = {
        "R9-1_keyboard_complete": report["keyboard_flow"]["verdict"],
        "R9-2_visible_focus": "PASS" if not focus_fail else "FAIL",
        "R9-3_axe_zero_violations": "PASS" if violations == 0 else "FAIL",
        "R9-4_contrast_and_forced_colors": "PASS" if (
            contrast_ran and contrast_passed
            and report["forced_colors"]["flow"]["verdict"] == "PASS"
            and report["forced_colors"]["violation_total"] == 0) else "FAIL",
        "R9-5_reduced_motion": report["reduced_motion"]["verdict"],
        "R9-6_announcements": "PASS" if (
            report["keyboard_flow"]["status_after_scored_entry"].strip()
            and report["keyboard_flow"]["status_after_crisis_entry"].strip()) else "FAIL",
        "R9-8b_no_third_party_request": report["page_requests"]["verdict"],
    }
    report["axe_violation_total"] = violations
    report["axe_incomplete_total"] = sum(a["incomplete_count"] for a in report["axe"])
    report["verdict"] = ("PASS" if all(v == "PASS" for v in report["verdicts"].values())
                         else "FAIL")

    (ROOT / "artifacts" / "a11y_report.json").write_text(json.dumps(report, indent=1) + "\n")
    summary = {k: v for k, v in report.items()
               if k in ("measured_at_utc", "server", "verdicts", "verdict",
                        "axe_violation_total", "axe_incomplete_total")}
    print(json.dumps(summary, indent=1))
    for entry in report["axe"]:
        if entry["violations"]:
            print(f"\nVIOLATIONS in {entry['view']}:", file=sys.stderr)
            print(json.dumps(entry["violations"], indent=1)[:6000], file=sys.stderr)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
