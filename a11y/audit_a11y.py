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
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image                                            # noqa: E402
from playwright.sync_api import sync_playwright                  # noqa: E402

from ledger.ui.server import serve_in_thread                     # noqa: E402

# One source for the banned list. R9-9 requires R8-8's check to run over the
# RENDERED page text rather than the source, and a second copy of the list here
# would be a copy that could drift from the one the report is held to.
sys.path.insert(0, str(ROOT / "tests"))
from test_report import BANNED                                   # noqa: E402

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


def launch_chromium(playwright):
    """Launch the browser, and say which binary was used.

    The Playwright build pinned by this environment ships a headless shell that
    exits immediately here, while a newer one in the same cache runs. Rather than
    silently taking whichever works, the chosen executable is returned and
    recorded in the report: a measurement that depends on a browser build should
    name the build.
    """
    attempts = []
    try:
        return playwright.chromium.launch(), {"executable": "playwright default",
                                              "attempts": attempts}
    except Exception as exc:                                # noqa: BLE001
        attempts.append({"executable": "playwright default", "error": str(exc)[:200]})

    cache = Path.home() / ".cache" / "ms-playwright"
    candidates = sorted(
        cache.glob("chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell"),
        key=lambda p: int(p.parents[1].name.split("-")[-1]), reverse=True)
    for candidate in candidates:
        try:
            return (playwright.chromium.launch(executable_path=str(candidate)),
                    {"executable": str(candidate), "attempts": attempts})
        except Exception as exc:                            # noqa: BLE001
            attempts.append({"executable": str(candidate), "error": str(exc)[:200]})
    raise RuntimeError(f"no chromium build would launch: {attempts}")


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


CONTRAST_SCAN = r"""() => {
  const channel = (v) => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); };
  const lum = ([r, g, b]) => 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b);
  const parse = (s) => { const m = String(s).match(/rgba?\(([^)]+)\)/); if (!m) return null;
    const p = m[1].split(',').map(Number); return {rgb: [p[0], p[1], p[2]], a: p.length > 3 ? p[3] : 1}; };
  const bgOf = (node) => { let n = node;
    while (n) { const c = parse(getComputedStyle(n).backgroundColor); if (c && c.a > 0) return c.rgb;
      n = n.parentElement; } return [255, 255, 255]; };
  const out = [];
  for (const n of document.querySelectorAll('*')) {
    if (![...n.childNodes].some(c => c.nodeType === 3 && c.textContent.trim().length)) continue;
    const s = getComputedStyle(n);
    if (s.visibility === 'hidden' || s.display === 'none' || !n.getClientRects().length) continue;
    if (n.closest('.visually-hidden')) continue;
    const fg = parse(s.color); if (!fg) continue;
    const bg = bgOf(n);
    const l1 = lum(fg.rgb), l2 = lum(bg);
    const ratio = (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
    const px = parseFloat(s.fontSize), bold = parseInt(s.fontWeight, 10) >= 700;
    const required = (px >= 24 || (bold && px >= 18.66)) ? 3 : 4.5;
    out.push({tag: n.tagName.toLowerCase(), id: n.id || '', cls: String(n.className).slice(0, 40),
              fg: s.color, bg: 'rgb(' + bg.join(', ') + ')', ratio: Math.round(ratio * 100) / 100,
              required, ok: ratio >= required});
  }
  return out;
}"""


def contrast_scan(page, view: str) -> dict:
    """Compute WCAG contrast from the live computed styles, independently of axe.

    axe reported `#unlock-submit` as a 1:1 `incomplete` under emulated forced
    colours while `getComputedStyle` on the same element returned black on white.
    Rather than take either on trust, this walks every text-bearing element,
    resolves the first opaque background above it, and computes the ratio from
    the same formula WCAG defines. Both numbers go in the artifact.
    """
    rows = page.evaluate(CONTRAST_SCAN)
    failures = [r for r in rows if not r["ok"]]
    return {"view": view, "elements_measured": len(rows), "failures": failures,
            "min_ratio": min((r["ratio"] for r in rows), default=None),
            "verdict": "PASS" if not failures and rows else "FAIL"}


#: The contract has to be able to say "diagnosis" in order to deny making one.
#: The exemption is by exact rendered text, taken from the page itself, and
#: covers nothing else — the same discipline `tests/test_report.py` uses.
RENDERED_TEXT = r"""() => {
  const flat = (s) => s.replace(/\s+/g, ' ').trim().toLowerCase();
  let text = flat(document.body.innerText);
  const exempt = [];
  for (const id of ['entry-contract', 'colophon-contract']) {
    const n = document.getElementById(id);
    if (n && n.textContent.trim()) {
      const f = flat(n.textContent);
      exempt.push(f);
      text = text.split(f).join(' ');
    }
  }
  return {text, exempt, length: text.length};
}"""


def vocabulary_scan(page, view: str) -> dict:
    """R9-9 / R8-8, over what the page actually renders."""
    rendered = page.evaluate(RENDERED_TEXT)
    found = [word for word in BANNED if word in rendered["text"]]
    return {"view": view, "rendered_chars": rendered["length"],
            "exempted": rendered["exempt"], "found": found,
            "verdict": "PASS" if not found and rendered["length"] > 200 else "FAIL"}


#: axe-core's `label` rule accepts a non-empty `placeholder` as an accessible
#: name. A placeholder disappears the moment someone types, so a control named
#: only by one is not really named. Mutation testing found that removing a
#: `<label>` left R9-3 green because the placeholder covered for it, so this
#: check was added and folded into R9-3 — tightening the rule, not relaxing it.
NAME_SCAN = """() => {
  const out = [];
  for (const n of document.querySelectorAll('input:not([type=hidden]), textarea, select')) {
    if (!n.getClientRects().length) continue;
    const explicit = n.id ? document.querySelectorAll(
      'label[for="' + CSS.escape(n.id) + '"]').length : 0;
    const wrapped = !!n.closest('label');
    const ariaLabel = (n.getAttribute('aria-label') || '').trim();
    const labelledby = (n.getAttribute('aria-labelledby') || '').trim();
    const named = explicit > 0 || wrapped || !!ariaLabel || !!labelledby;
    out.push({id: n.id || '', tag: n.tagName.toLowerCase(), type: n.type || '',
              explicit_labels: explicit, wrapped_in_label: wrapped,
              aria_label: !!ariaLabel, aria_labelledby: !!labelledby,
              placeholder: !!(n.getAttribute('placeholder') || '').trim(),
              named});
  }
  return out;
}"""


def name_scan(page, view: str) -> dict:
    """Every visible form control must be named by something durable."""
    controls = page.evaluate(NAME_SCAN)
    unnamed = [c for c in controls if not c["named"]]
    # No vacuity guard on the count here: `entry`, `history` and `report` have no
    # form controls at all, and demanding one would fail a view for being what it
    # is. The views that do have controls — unlock, write, settings — carry the
    # weight, and an unnamed control anywhere fails.
    return {"view": view, "controls": len(controls), "unnamed": unnamed,
            "verdict": "PASS" if not unnamed else "FAIL"}


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


def motion_scan(page) -> dict:
    """Every element whose computed animation or transition is not zero.

    Returns the scanned count as well as the offenders: "no offenders" out of a
    DOM of zero elements is not the same result as out of a DOM of two hundred,
    and only one of them is evidence.
    """
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
      return {scanned: document.querySelectorAll('*').length, non_zero: bad};
    }""")


def focusable_boxes(page) -> int:
    """Tag every focusable element that is currently rendered, in tab order.

    Whole document, not just the view section. The skip link and the section rail
    live outside every `<section class="view">`, and they are as much a part of
    "every focusable element" as the controls inside one — the first version of
    this function scanned only the section and so never looked at either.
    """
    return page.evaluate("""() => {
      document.querySelectorAll('[data-a11y-idx]').forEach(n => n.removeAttribute('data-a11y-idx'));
      const sel = 'a[href], button, input:not([type=hidden]), textarea, select, [tabindex]';
      const nodes = [...document.querySelectorAll(sel)]
        .filter(n => !n.disabled && n.tabIndex >= 0 && n.getClientRects().length);
      nodes.forEach((n, i) => n.setAttribute('data-a11y-idx', String(i)));
      return nodes.length;
    }""")


#: R9-2 is only meaningful if there was something to look at. A view with no
#: focusable element found is a broken measurement, not a clean result — the
#: first run of this harness "passed" two views this way, because it scanned a
#: section that was hidden at the time (DEFECT-INC9-002).
MIN_FOCUSABLE_PER_VIEW = 3


def measure_focus(page, view: str) -> dict:
    """R9-2, optically, over every focusable element currently rendered."""
    visit_view(page, view)
    page.wait_for_timeout(80)
    count = focusable_boxes(page)
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
        reached_by = "Tab"
        if landed != str(idx):
            # A radio group is ONE tab stop. Tab moving from the checked radio
            # straight past its unchecked siblings is correct behaviour, not an
            # unreachable control, and the arrow keys are how a keyboard user
            # gets to the others. Treating that as a failure was a defect in this
            # harness, not in the interface (DEFECT-INC9-004).
            if page.evaluate("""(i) => {
                  const n = document.querySelector(`[data-a11y-idx="${i}"]`);
                  return !!n && n.tagName === 'INPUT' && n.type === 'radio' && !!n.name;
                }""", idx):
                # Tab has already carried focus past the group, so step back into
                # it before using the arrow keys — pressing ArrowDown while a
                # button is focused does nothing at all, which is why the first
                # attempt at this fallback still recorded the radio unreachable.
                page.evaluate("""(i) => {
                      const n = document.querySelector(`[data-a11y-idx="${i}"]`);
                      const group = document.getElementsByName(
                        document.querySelector(`[data-a11y-idx="${i}"]`).name);
                      const checked = [...group].find(r => r.checked) || group[0];
                      checked.focus();
                    }""", idx)
                for _ in range(8):
                    page.keyboard.press("ArrowDown")
                    if page.evaluate("(i) => document.activeElement && "
                                     "document.activeElement.getAttribute('data-a11y-idx') === String(i)",
                                     idx):
                        break
                landed = page.evaluate("() => document.activeElement "
                                       "? document.activeElement.getAttribute('data-a11y-idx') : null")
                reached_by = "ArrowDown within the radio group"
        if landed != str(idx):
            elements.append({"index": idx, "reached": False, "landed_on": landed})
            continue

        box = page.evaluate("""(i) => {
          const n = document.querySelector(`[data-a11y-idx="${i}"]`);
          const r = n.getBoundingClientRect();
          return {x: r.x, y: r.y, w: r.width, h: r.height,
                  tag: n.tagName.toLowerCase(), id: n.id || '', cls: n.className.toString().slice(0,50)};
        }""", idx)
        # `clip` is viewport-relative for a non-full-page screenshot — measured,
        # not assumed: a page-coordinate clip at y=1500 errors with "outside the
        # resulting image", and a viewport-coordinate one captures the element
        # exactly. `getBoundingClientRect()` is therefore the right source, and
        # the box is clamped to the viewport so a control at the edge does not
        # throw instead of being measured.
        pad = 8
        view_w, view_h = page.viewport_size["width"], page.viewport_size["height"]
        x0 = max(0.0, box["x"] - pad)
        y0 = max(0.0, box["y"] - pad)
        clip = {
            "x": x0, "y": y0,
            "width": min(box["w"] + pad * 2, view_w - x0),
            "height": min(box["h"] + pad * 2, view_h - y0),
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
            "index": idx, "reached": True, "reached_by": reached_by, "tag": box["tag"],
            "id": box["id"], "class": box["cls"], "changed_pixel_ratio": round(ratio, 5),
            "passes": ratio >= FOCUS_MIN_CHANGED, "unfocused_styles": styles,
        })
    failures = [e for e in elements
                if e.get("reached") and "skipped" not in e and not e.get("passes", False)]
    unreached = [e for e in elements if not e.get("reached")]
    measured = [e for e in elements if "changed_pixel_ratio" in e]
    vacuous = len(measured) < MIN_FOCUSABLE_PER_VIEW
    return {"view": view, "focusable": count, "measured": len(measured),
            "elements": elements, "failures": len(failures),
            "unreachable": len(unreached), "vacuous": vacuous,
            "min_ratio": min((e["changed_pixel_ratio"] for e in measured), default=None),
            "verdict": "PASS" if not failures and not unreached and not vacuous else "FAIL"}


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
    # The live region is written on a short timer so that a repeated message
    # re-announces. Reading it straight after the repaint catches the *previous*
    # message, which is how the first run of this harness recorded a pass for
    # R9-6 on the text "Reading your entry on this machine…" (DEFECT-INC9-003).
    page.wait_for_function(
        "() => document.getElementById('status').textContent.includes('explained')",
        timeout=15000)
    scored_status = page.text_content("#status")
    # The attribution view is the one artifact a C6 reader most needs to see, and
    # it is only on screen between the scored entry and the crisis entry that
    # follows it. Captured here rather than in the later screenshot sweep, which
    # would find this view showing the crisis panel instead.
    page.screenshot(path=str(SHOTS / f"attribution-{label}.png"), full_page=True)
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
    page.wait_for_function(
        "() => document.getElementById('status').textContent.includes('crisis rule')",
        timeout=15000)
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
    # Mutation testing found R9-6 passing after `role="status" aria-live="polite"`
    # was stripped off the element: the text still arrived, so a check that only
    # read textContent could not tell an announcement from a repaint. The
    # attributes are now part of the rule.
    live = page.evaluate("""() => {
      const s = document.getElementById('status');
      const c = document.getElementById('crisis');
      return {status_aria_live: s ? s.getAttribute('aria-live') : null,
              status_role: s ? s.getAttribute('role') : null,
              crisis_role: c ? c.getAttribute('role') : null};
    }""")
    live["is_live_region"] = (live["status_aria_live"] in ("polite", "assertive")
                              or live["status_role"] == "status")
    live["crisis_is_alert"] = live["crisis_role"] == "alert"
    return {
        "label": label,
        "live_region": live,
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
        #: Every origin this harness itself started. R9-8(b) is a claim about
        #: 127.0.0.1, not about one port: the forced-colors pass runs a second
        #: server on a second ephemeral port, and its requests are as local as
        #: the first server's. Comparing against a single origin marked those
        #: twelve requests external and failed a rule the product had not broken
        #: (DEFECT-INC9-001).
        loopback_origins: set[str] = {server.url.rstrip("/")}

        with sync_playwright() as p:
            browser, browser_info = launch_chromium(p)
            report["harness"]["browser_binary"] = browser_info
            report["harness"]["browser_version"] = browser.version
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
                contrast_results = []
                vocabulary_results = []
                name_results = []
                for view in VIEWS:
                    visit_view(page, view)
                    page.wait_for_timeout(120)
                    axe_results.append(run_axe(page, view))
                    contrast_results.append(contrast_scan(page, view))
                    vocabulary_results.append(vocabulary_scan(page, view))
                    name_results.append(name_scan(page, view))
                    page.screenshot(path=str(SHOTS / f"{view}.png"), full_page=True)
                report["axe"] = axe_results
                report["computed_contrast"] = contrast_results
                report["rendered_vocabulary"] = vocabulary_results
                report["accessible_names"] = name_results

                # R9-2 on the two views that carry the most controls, plus the
                # entry view, which is the one built from user text at runtime.
                report["focus_visibility"] = [measure_focus(page, v) for v in VIEWS]
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
                # The journal opened in the first context is still open on the
                # server, so this page loads straight into `write`. Unlocking
                # again would fail on a store that already exists, so the state
                # is read rather than assumed.
                if rm_page.evaluate("() => !document.getElementById('view-unlock').hidden"):
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
                    scan = motion_scan(rm_page)
                    motion.append({"view": view, "scanned": scan["scanned"],
                                   "non_zero": scan["non_zero"]})
                report["reduced_motion"] = {
                    "views": motion,
                    "elements_scanned": sum(m["scanned"] for m in motion),
                    "total_non_zero": sum(len(m["non_zero"]) for m in motion),
                    "verdict": ("PASS" if (not any(m["non_zero"] for m in motion)
                                           and all(m["scanned"] > 50 for m in motion))
                                else "FAIL"),
                }
                rm.close()

                # R9-4b: the whole flow again with the platform's colours forced.
                fc_store = Path(tmp) / "forced.enc"
                fc_server = serve_in_thread(store=fc_store, region="SG")
                loopback_origins.add(fc_server.url.rstrip("/"))
                fc = browser.new_context(viewport={"width": 1280, "height": 900},
                                         forced_colors="active")
                fc.on("request", lambda r: requests_seen.append(
                    {"url": r.url, "resource_type": r.resource_type}))
                fc_page = fc.new_page()
                fc_page.add_init_script(axe_source)
                fc_page.add_init_script(POINTER_COUNTER)
                fc_page.goto(fc_server.url, wait_until="load")
                fc_flow = keyboard_flow(fc_page, fc_store, label="forced-colors")
                fc_axe, fc_contrast = [], []
                for view in VIEWS:
                    visit_view(fc_page, view)
                    fc_page.wait_for_timeout(120)
                    fc_axe.append(run_axe(fc_page, view))
                    fc_contrast.append(contrast_scan(fc_page, view))
                    fc_page.screenshot(path=str(SHOTS / f"forced-colors-{view}.png"), full_page=True)
                report["forced_colors"] = {
                    "flow": fc_flow, "axe": fc_axe, "computed_contrast": fc_contrast,
                    "violation_total": sum(a["violation_count"] for a in fc_axe),
                    "incomplete_total": sum(a["incomplete_count"] for a in fc_axe),
                    "computed_contrast_failures": sum(len(c["failures"]) for c in fc_contrast),
                }
                fc.close()
                fc_server.shutdown_now()
            finally:
                browser.close()
        server.shutdown_now()

    def is_local(url: str) -> bool:
        if url.startswith("data:"):
            return True
        parsed = urlparse(url)
        # Both halves are required. The host check is the rule; the origin check
        # stops a request to some *other* process listening on loopback from
        # being waved through as "well, it is 127.0.0.1".
        return (parsed.hostname == "127.0.0.1"
                and f"{parsed.scheme}://{parsed.netloc}" in loopback_origins)

    external = [r for r in requests_seen if not is_local(r["url"])]
    report["page_requests"] = {
        "total": len(requests_seen),
        "loopback_origins": sorted(loopback_origins),
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
        # R9-3 as pre-registered is "zero axe violations". It is reported here
        # together with the accessible-name check, because a mutation that
        # removed a <label> left axe green. Both must hold.
        "R9-3_axe_zero_violations": "PASS" if (
            violations == 0
            and all(n["verdict"] == "PASS" for n in report["accessible_names"])) else "FAIL",
        "R9-4_contrast_and_forced_colors": "PASS" if (
            contrast_ran and contrast_passed
            and all(c["verdict"] == "PASS" for c in report["computed_contrast"])
            and report["forced_colors"]["flow"]["verdict"] == "PASS"
            and report["forced_colors"]["violation_total"] == 0
            and report["forced_colors"]["computed_contrast_failures"] == 0) else "FAIL",
        "R9-5_reduced_motion": report["reduced_motion"]["verdict"],
        "R9-6_announcements": "PASS" if (
            report["keyboard_flow"]["status_after_scored_entry"].strip()
            and report["keyboard_flow"]["status_after_crisis_entry"].strip()
            and report["keyboard_flow"]["live_region"]["is_live_region"]
            and report["keyboard_flow"]["live_region"]["crisis_is_alert"]) else "FAIL",
        "R9-8b_no_third_party_request": report["page_requests"]["verdict"],
        "R9-9_no_clinical_vocabulary_rendered": "PASS" if all(
            v["verdict"] == "PASS" for v in report["rendered_vocabulary"]) else "FAIL",
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
        if entry["incomplete"]:
            print(f"\nINCOMPLETE in {entry['view']} (not counted as a pass):", file=sys.stderr)
            print(json.dumps(entry["incomplete"], indent=1)[:4000], file=sys.stderr)
    for entry in report["forced_colors"]["axe"]:
        if entry["violations"] or entry["incomplete"]:
            print(f"\nforced-colors {entry['view']}: "
                  f"{entry['violation_count']} violation(s), "
                  f"{entry['incomplete_count']} incomplete", file=sys.stderr)
            print(json.dumps({"violations": entry["violations"],
                              "incomplete": entry["incomplete"]}, indent=1)[:4000],
                  file=sys.stderr)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
