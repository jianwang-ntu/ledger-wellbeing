# Increment 6 — pre-registration

Written and committed **before** `export/common.py` was edited and before any
number in this increment was produced. Timestamp of writing: **2026-08-24T17:03Z**.
The commit that carries this file contains no measurement.

Increment 5 established, and did not merely assert, that no entailment-supervised
bi-encoder at hidden ≤ 384 both separates held-out anchors and fits CEIL-1. That
closed the size route. `plan.md` R-4 — written at step D, before any of this was
measured — says what happens next:

> If the web target fails, fall back to a local desktop app; the zero-egress
> claim survives, the "in-browser" claim is dropped rather than fudged.

This increment **exercises** that fallback instead of declaring it.

## The question

With delivery re-targeted from an in-browser download to a locally installed
desktop application, does the scorer increment 4 selected —
`sentence-transformers/nli-distilroberta-base-v2` @ `cc35a0bf`, centroid-difference
readout, macro held-out AUC **0.880** — produce a build that clears every ceiling
that still applies to it?

## What the target change is, and what it is not

The delivery target is a property of *distribution*, not of the model. So:

| Ceiling | Property of | Still enforced on the desktop target? |
|---|---|---|
| CEIL-1 model bytes ≤ 32 MiB | the **download** | **No** — measured and reported, not enforced |
| CEIL-2 tokenizer ≤ 2 MiB | the download | **Yes** (it is met with room to spare either way) |
| CEIL-3 cold first-load ≤ 64 MiB | the **download** | **No** — measured and reported, not enforced |
| CEIL-4 p95 latency ≤ 500 ms | the **model + runtime** | **Yes**, on the desktop runtime — see the relaxation below |
| CEIL-5 int8-vs-fp32 agreement | the **model** | **Yes**, unchanged |

**This is a relaxation and is recorded as one.** Two ceilings the current
artifact fails stop binding because the artifact stopped being a web download.
Three guards against that being self-serving, all fixed here before measurement:

1. **No ceiling value is edited.** `CEILINGS` in `export/common.py` keeps every
   number byte-identical, and
   `tests/test_size_feasible_scorer.py::TestNoCeilingMoved` continues to assert
   CEIL-1 is 32 MiB. What changes is a separate `ENFORCED_BY_TARGET` map.
2. **CEIL-1 and CEIL-3 are still measured** for every build and printed in
   `verify_report.json` with an explicit `would_fail_web_target` flag, so the
   claim that was dropped stays visible rather than disappearing from the report.
3. **The web claim is deleted from the artifacts that make it** — `README.md`,
   `plan.md` C1/C4 and the project description — in this same increment. The
   plan and the artifact do not get to disagree.

### The one genuine loosening, stated plainly

CEIL-4 was written as *"p95 inference latency, 256-token entry, single-threaded
**WASM**"*, chosen because WASM single-threaded was the pessimistic case for a
browser. A desktop app does not run WASM, so on the desktop target CEIL-4 is
judged on **native onnxruntime, CPU, 1 intra-op / 1 inter-op thread**, at the
same 256 tokens and the same 500 ms. Native is *faster* than WASM, so this makes
the ceiling easier, and that is a relaxation of a measurement basis rather than
of a number. It is logged here rather than absorbed. The WASM number is still
measured where it can be obtained and reported alongside; if it cannot be
obtained inside this increment, it is reported as `NOT_MEASURED`, not as absent.

## Adoption rule — fixed before measurement

The build is adopted only if **all** of the following hold. Any failure means the
adoption does not happen and the reason is recorded.

1. **R6-1 — additivity survives the body swap.** `max |logits − (Σ token_attr +
   bias)| ≤ 1e-4` on every build produced. The incumbent measured 2.4e-7. If this
   fails, criterion C4's explainability claim is **dropped**, not softened.
2. **R6-2 — CEIL-5 decides which build ships.** Ship the smallest build whose
   per-dimension Pearson r ≥ 0.99 **and** max abs score delta ≤ 0.02 against
   fp32. If no int8 build passes, ship fp32 — this is `SIZE_BUDGET.md`'s own
   written fallback ("a CEIL-5 failure means the int8 build is not shipped"),
   not a new rule.
3. **R6-3 — CEIL-4 passes on the desktop runtime** for the build R6-2 selects.
4. **R6-4 — CEIL-2 passes**, unchanged.
5. **R6-5 — the head is still not trained**, and nothing produced here may be
   described as fine-tuned. `head_is_trained` stays `false` (plan.md R-1).
6. **R6-6 — `activation` is not presented as a working dimension.** Its held-out
   AUC is 0.600, below the 0.700 usability floor fixed in increment 3. Wherever
   per-dimension numbers appear, that row carries the failure. Adopting a scorer
   whose macro is 0.880 does not repair its worst dimension, and the macro may
   not be quoted without it.

## Prediction, recorded so it can be wrong

Written before running anything:

- R6-1 **will** hold. The head is the same linear row over a mean-pooled vector;
  only the body changed. Predicted residual < 1e-5.
- R6-2 **will select `int8_embed`**. On the incumbent, `int8_full` failed CEIL-5
  at max delta 0.0404 while `int8_embed` passed at 0.0094, and distilroberta has
  more, not fewer, MatMuls to damage.
- R6-3 is the one I am least sure of. distilroberta-base is 6 layers at hidden
  768 against the incumbent's 6 at 384 — roughly 4× the matmul work — and the
  incumbent's `int8_embed` measured 236 ms p95 in WASM and well under that
  natively. Native p95 under 500 ms is likely; the **WASM** number almost
  certainly is not, which is a second, independent reason the web target is gone.
- CEIL-1 will be missed by roughly 6×, and CEIL-3 by roughly 3×.

If R6-3 fails natively, the fallback is not another target: it is chunking or a
smaller body, and the increment closes with no adoption.
