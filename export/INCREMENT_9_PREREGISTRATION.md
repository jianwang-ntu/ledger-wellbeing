# Increment 9 — pre-registration

Written and committed **before** the code it governs, per the discipline of
`INCREMENT_6/7/8_PREREGISTRATION.md`.

## The gap this increment closes

`plan.md` C6 — *UI/UX & Accessibility*, "Visual design quality, ease of
navigation, intuitive user flows, and adherence to accessibility standards" — is
the one row in the criteria table with **zero artifact**. Increment 8 shipped a
CLI and said so plainly: a CLI answers none of that criterion's four named
qualities. C6 was declared a deliberate hole *before* increment 8 started and
owed here.

It is also the largest remaining scoring hole. C1–C5 each have at least one
artifact; C6 has none. An audit run before this increment would spend a revision
round learning something already written down.

## The fork recorded in increment 8, now taken

`INCREMENT_8_PREREGISTRATION.md` fixed this in advance:

> The claim this project can defend indefinitely is **"nothing leaves the
> device"**, measured. The claim "no server component at all" is an
> implementation detail that happens to be true today. If increment 9 binds a
> loopback listener, the second sentence is **dropped and replaced by the
> measured one** — the same way "in the browser" was dropped in increment 6 —
> and it is not quietly reinterpreted to mean "no *remote* server".

**Increment 9 binds a loopback listener.** So the sentence "no server component
at all" is dropped from `plan.md`, `README.md` and every artifact, and replaced
by two claims that are measured here rather than asserted:

1. nothing leaves the device (R9-8), and
2. the listener is bound to `127.0.0.1` only and is **refused** from the host's
   own non-loopback address (R9-7).

### Why a loopback UI rather than a native GUI

A native GUI (tkinter/Qt) would open no socket at all and would preserve the
dropped sentence for free. It was rejected, and the reason is that it would make
this criterion's evidence *worse*:

- `axe-core` and every automated WCAG check operate on a DOM. There is no DOM in
  tkinter, so C6 would be back to being asserted rather than measured.
- Contrast ratios need real layout and real compositing. A rendered browser can
  be measured; a Tk widget tree cannot, without hand-computing what the
  compositor did.
- `prefers-reduced-motion` and `forced-colors` are platform signals the browser
  exposes and honours; Tk has no equivalent to measure against.

The honest trade is: give up a sentence that was true by accident, and gain the
ability to *measure* the criterion instead of describing it. The listener is
loopback-only and its unreachability is itself measured, so nothing about the
privacy property is weakened — only the wording is, and only where the wording
was over-broad.

## Scope, fixed now

`ledger/ui/` — an HTTP server bound to `127.0.0.1` on an ephemeral port, serving
a single-page application from files inside the package. Launched with
`python -m ledger.ui`. No CDN, no external font, no analytics, no framework
fetched at runtime: every byte the page loads comes from this repository, and
R9-8 measures that in the browser rather than trusting the markup.

Views: **unlock/create** → **write** → **entry with per-span attribution** →
**history** → **report** → **wipe**. The same `LedgerEngine`, `Journal`,
`crisis_router` and `render` the CLI uses; the UI adds a presentation layer and
**no new inference, storage or safety behaviour**.

**Explicitly NOT in increment 9**, so the holes stay deliberate:

| Not built | Why | Owed to |
|---|---|---|
| The 4-minute video | Requires a human. Must not be faked. | NEEDS_HUMAN, unchanged |
| Any training | `plan.md` R-1 uncleared. | blocked, unchanged |
| `eval/bias_slices.json` | No demographically sliceable evaluation data exists. Stated as a real shortfall in C4, not a deferral. | unchanged |
| A packaged installer | Not asked for by `overview.txt`. | out of scope |
| A recorded screen-reader traversal | Needs a screen reader and a human listener. What is measured instead is the *machine-checkable* half: accessible names, roles, live-region updates and keyboard operability. The difference is stated in the artifact, and the assistive-technology half is **not claimed**. | NEEDS_HUMAN |

That last row matters. `plan.md`'s evidence plan says "plus a recorded
screen-reader traversal". That artifact will not exist this increment, and C6
must not be reported as fully evidenced on the strength of an axe run.

## Adoption rules, fixed now

Every rule below is blind: the threshold is written here, before the measurement.

| | Rule | Blind? |
|---|---|---|
| **R9-1** | **Keyboard completeness.** The whole primary flow — create journal → write an entry → submit → read the attribution → open history → render the report → open and cancel the wipe dialog — completes with **keyboard events only**, zero mouse events dispatched. Measured by driving a real browser with `keyboard.press` alone and asserting the terminal state of each step. A flow that needs one click fails. | **yes** |
| **R9-2** | **Visible focus, measured optically.** For every focusable element: a screenshot of its bounding box while focused differs from the same box unfocused by **≥ 0.5 % of pixels**. Computed styles are also recorded, but the pixel diff is the rule — `outline: none` plus a repaint that no one can see would pass a style check and fails this one. | **yes** |
| **R9-3** | **axe-core, zero violations.** `axe-core` run in the real Chromium against **every** view, with tags `wcag2a, wcag2aa, wcag21a, wcag21aa`. **Zero** violations. `incomplete` results are printed in full and reasoned about in the artifact rather than dropped; they are not counted as passes. | **yes** |
| **R9-4** | **Contrast and forced colors.** axe's `color-contrast` rule must run (not be `incomplete`) and pass on every view — this is the check that needs real layout, and is the reason for a browser. Separately, under `forced-colors: active` the app renders and stays operable: the primary flow of R9-1 completes again, unchanged. | **yes** |
| **R9-5** | **Reduced motion, measured over the live DOM.** Under `prefers-reduced-motion: reduce`, **every** element in every view reports computed `animation-duration` and `transition-duration` of `0s`. Enumerated over the rendered DOM, not read off the stylesheet. | **yes** |
| **R9-6** | **Announcements.** After an analysis completes and after a crisis route fires, an `aria-live` region contains text naming what happened. Read back from the DOM after the action. A visual-only state change fails. | **yes** |
| **R9-7** | **Loopback-only listener, measured from outside.** The server binds `127.0.0.1` exclusively. A TCP connect to the **same port** on the host's own non-loopback address is **refused or times out**. If any non-loopback address accepts, the UI is withdrawn rather than shipped with a caveat. | **yes** |
| **R9-8** | **Zero egress survives the UI**, on two independent measurements: (a) `export/egress_audit.py --ui` starts the server and drives it in-process — every socket call is loopback, zero DNS resolution of a non-local name; (b) in the browser, **every** network request the page issues has origin `127.0.0.1`. One external request of any kind — a font, a favicon, a telemetry beacon — fails this rule, and the claim is dropped rather than softened. | **yes** |
| **R9-9** | **No new claims.** `head_is_trained` stays `false`. Any dimension below the 0.70 held-out floor is labelled `[NOT ESTABLISHED]` with its AUC **in the rendered DOM**, not only in the CLI. R8-8's banned-vocabulary test runs over the **rendered page text**, not the source. | **yes** |
| **R9-10** | No value in `CEILINGS` edited, no entry in `ENFORCED_BY_TARGET` changed, `verify.py` still exits 0 with `int8_embed` selected, and the increment-8 suite still passes. | **yes** |

**If R9-3 does not reach zero**, the violations are listed in the artifact and in
`plan.md` C6, and C6 is reported as partially evidenced. The count is not
re-scoped by narrowing the tag set.

**If R9-7 fails**, the UI does not ship.

## Attribution (rules.txt rule 2)

`axe-core` (MPL-2.0) and Playwright (Apache-2.0) are **measurement tools**, used
in the test harness only. Neither is imported by the application, and neither is
a runtime dependency. Both are recorded in `data/MANIFEST.md` with their
licences. The page itself loads no third-party asset at all — that is R9-8(b).

## Predictions

Written before the measurements, so the record shows what was expected.

1. **R9-3 needs more than one pass.** *Likely.* Hand-written markup reaching zero
   axe violations first try would be unusual; the common misses are an unlabelled
   control, a landmark gap and a heading-order jump.
2. **R9-4 color-contrast is where a designed palette gets caught.** *Medium.* The
   greys that look right are usually the ones that fail 4.5:1.
3. **R9-5 holds trivially if the CSS is written for it, and is worth measuring
   anyway** — a `transition` on a hover state is exactly the thing that survives
   a media query by accident. *High.*
4. **R9-7 holds.** *High.* It is a one-line bind address, which is precisely why
   it gets measured: "obviously right" is how it would be wrong.
5. **R9-8(b) is the one most likely to surprise.** *Medium.* A favicon request to
   the page's own origin is fine; a browser default that reaches elsewhere, or a
   stray `@font-face`, is not, and would not be visible by reading the HTML.

## What this increment must not be allowed to erase

A visual interface is persuasive in a way a CLI is not. The numbers it displays
are the same numbers, computed by the same untrained anchor head, evaluated on 25
withheld anchor-sentence pairs per dimension written for this repository. The
prettier the surface, the more load R9-9 carries: if the UI ever shows a score
without showing what established it, the interface has started overstating the
project's evidence, and that is a defect of the same kind as a false claim in the
README.
