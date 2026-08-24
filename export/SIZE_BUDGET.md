# SIZE_BUDGET.md — the in-browser ceiling

Closes plan.md risk **R-4** ("In-browser inference is a real engineering risk —
model size, WASM/WebGPU support, cold-start latency. *Size ceiling fixed on day 1
and the export pipeline built before the UI.*").

These ceilings were **fixed before the model was exported or measured**, so that
the measurement in `../audit/runs/export_verify_*.json` is a test the pipeline
can fail, not a description of whatever came out. Written 2026-08-24, build
increment 2.

## The one number we do not control

`onnxruntime-web` is the runtime. Its cold payload is fixed by the package, not
by us. Measured from `web/node_modules/onnxruntime-web/dist` at version
`1.23.0`:

| Artifact | Bytes | Note |
|---|---:|---|
| `ort.wasm.min.mjs` | 49,856 | wasm-only ESM entry point (not the `ort.all` bundle) |
| `ort-wasm-simd-threaded.wasm` | 11,815,498 | SIMD + threads, **non-JSEP** |
| **Runtime floor** | **11,865,354** | **11.32 MiB** |

The JSEP build (`ort-wasm-simd-threaded.jsep.wasm`, 23,696,346 B) doubles the
floor to buy WebGPU. A 22M-parameter encoder over a 256-token journal entry does
not need a GPU, so the plain SIMD+threads build is chosen and WebGPU is dropped.
That decision costs 11.3 MiB of budget we would otherwise have spent for nothing.

## The ceilings

| ID | Ceiling | Value | Why this number |
|---|---|---:|---|
| **CEIL-1** | Quantized model weights (`.onnx`, int8) | **≤ 32 MiB** | Leaves ≥ 20 MiB of headroom under CEIL-3 after the fixed runtime floor, so the tokenizer, the app shell and a future larger head all fit without reopening this decision. |
| **CEIL-2** | Tokenizer + vocabulary | **≤ 2 MiB** | A WordPiece vocab of 30k entries is ~230 KiB of text; 2 MiB is generous and forces a rejection if someone swaps in a 250k-entry multilingual vocab without saying so. |
| **CEIL-3** | Total cold first-load payload | **≤ 64 MiB** | At a deliberately pessimistic 10 Mbit/s effective downlink this is ~54 s once, and the Cache API makes every later visit a local read. At 50 Mbit/s it is ~11 s. Stated as an assumption, not a measurement: we have not measured a user's connection and will not claim to have. |
| **CEIL-4** | p95 inference latency, 256-token entry, single-threaded WASM | **≤ 500 ms** | A journal entry is scored on submit, not per keystroke. 500 ms is under the 1 s limit at which a user's flow of thought stays uninterrupted (Nielsen, *Usability Engineering*, 1993, §5.5). Single-threaded is the pessimistic case: `crossOriginIsolated` is false without COOP/COEP headers, and a static host may not send them. |
| **CEIL-5** | int8-vs-fp32 agreement, per dimension, over the fixed probe set | **Pearson r ≥ 0.99 AND max abs difference ≤ 0.02 on the 0–1 normalised score** | Quantization that moves a displayed score by more than 2 points out of 100 would change what a clinician reads. This is the tolerance at which the int8 build may be substituted for the fp32 build without qualification. |

## What happens if a ceiling fails

Per plan.md R-4: *"If the web target fails, fall back to a local desktop app; the
zero-egress claim survives, the 'in-browser' claim is dropped rather than
fudged."* A CEIL-1/2/3 failure means a smaller encoder. A CEIL-4 failure means
shorter chunking or a smaller encoder. A **CEIL-5 failure means the int8 build is
not shipped** and the fp32 build is measured against CEIL-1 instead — the
accuracy claim is not weakened to keep the size claim.

No ceiling is to be edited to match a measurement. If one is genuinely wrong,
it is superseded in a new row with the reason, and the old row stays.

---

## Build increment 3 — the fallback rule was invoked, and the route through was not taken

No ceiling in the table above has been edited. This section records what happened
when the increment-2 route through was tested.

Increment 2 left `verify.py` at exit 1 with `shippable_builds == []` and named a
narrower encoder as the way out: hidden 256 rather than 384, encoder layers fp32,
int8 on the embedding table only, projected at ~26.7 MiB.

`export/encoder_ablation.py` measured that route before it was adopted, and the
size half of it held — `google/bert_uncased_L-6_H-256_A-4` projects to 25.91 MiB
against CEIL-1's 32 MiB. The quality half did not. Every candidate encoder,
including the incumbent, scores at chance on held-out anchor separation (macro
AUC 0.392–0.504) while scoring 1.000 in-sample, and the positive control at
0.82–0.86 rules out the protocol as the explanation. Details in
`../docs/limitations.md` §1.

**The encoder was therefore not switched**, and CEIL-1 is still unmet by a
shippable build. Shrinking a scorer that does not discriminate on held-out text
would have made the size number better and the product no more real.

This is the SIZE_BUDGET fallback rule working as written, one level up from where
it was expected to fire: plan.md R-4's fallback ("drop the in-browser claim
rather than fudge it") remains live and unexercised, because the binding problem
turned out not to be size.

---

## Build increment 5 — the fallback rule fired, and this time it was taken

No ceiling in the table above has been edited. This section records the run that
ended the attempt to meet them.

Increment 4 left the project holding two artifacts that are not the same
artifact: a scorer measured to separate held-out text at macro AUC 0.880
(`sentence-transformers/nli-distilroberta-base-v2`, 201.68 MiB) and a scorer that
fits (`all-MiniLM-L6-v2`, at chance). `export/size_feasible_scorer.py` was
written to close that gap from the size end, with its candidate list committed
before it ran (`c1727e1`) and with **shippability promoted from a post-hoc report
to a selection criterion** — a strict tightening of increment 4's rule.

The envelope was arithmetic, not a search. The int8-embed build stores embedding
parameters at one byte and every other parameter at four, so at a 30522-entry
vocabulary CEIL-1's 32 MiB buys roughly three encoder layers at hidden 384 or
eight at hidden 256. Four entailment-supervised bi-encoders at hidden ≤ 384 were
measured against it, with three references.

**Nothing was selected.** The full table is in `../docs/limitations.md` §1b and
`../artifacts/size_feasible_scorer.json`. The two rows that define the outcome:

| | Model | int8-embed | CEIL-1 | Held-out AUC | Control |
|---|---|---:|---:|---:|---:|
| fits, does not separate | `xtremedistil-l6-h256-zeroshot-v1.1-all-33` | **25.91 MiB** | pass | **0.504** | 0.628 (under floor) |
| separates, does not fit | `nli-distilroberta-base-v2` | 201.68 MiB | **fail** | **0.880** | 0.916 |

The closest thing to a near miss, `paraphrase-MiniLM-L3-v2`, projects to **32.24
MiB against CEIL-1's 32.00** — 0.24 MiB over — and separates at 0.592 against a
0.70 threshold. It fails both halves, and it is named here because raising CEIL-1
by a quarter of a megabyte is the specific edit this document exists to forbid.
`tests/test_size_feasible_scorer.py::TestNoCeilingMoved::test_ceil_1_is_still_32_mib`
fails if it ever happens.

So the fallback in "What happens if a ceiling fails" is now **invoked rather than
merely available**. plan.md R-4 as written: *fall back to a local desktop app;
the zero-egress claim survives, the "in-browser" claim is dropped rather than
fudged.* CEIL-1 through CEIL-3 stop being the binding constraint on the artifact,
because the artifact is no longer a web download — and CEIL-4 (latency) and
CEIL-5 (int8-vs-fp32 agreement) continue to apply unchanged, because they are
properties of the model, not of the delivery target.

`export/common.py` is unchanged. `BASE_MODEL` still holds the incumbent, and no
build has been shipped.

---

## Build increment 6 — the fallback was taken, and it was not enough

No ceiling in the table above has been edited. `CEILINGS` in `export/common.py`
is byte-identical to the day-1 values and
`tests/test_size_feasible_scorer.py::TestNoCeilingMoved` still asserts it.

What changed is **which** ceilings gate, and that change is scoped to a delivery
target rather than applied to a number. `export/common.py` now carries
`DELIVERY_TARGET = "desktop"` and an `ENFORCED_BY_TARGET` map:

| Ceiling | Bounds | Enforced on `desktop` |
|---|---|---|
| CEIL-1 model bytes | an HTTP first-load | **no** — measured, reported, not gating |
| CEIL-2 tokenizer bytes | an HTTP first-load | **yes** |
| CEIL-3 cold payload | an HTTP first-load | **no** — measured, reported, not gating |
| CEIL-4 p95 latency | the model + runtime | **yes**, on native ORT rather than WASM |
| CEIL-5 int8-vs-fp32 | the model | **yes**, unchanged |

Two ceilings this artifact fails stopped gating. That is a relaxation and is
recorded as one, with the decision written down in
`export/INCREMENT_6_PREREGISTRATION.md` **before** the body was swapped or
anything was measured. Every build still reports `would_fail_web_target_on`, so
the claim that was dropped is visible in the artifact rather than absent from it.
The CEIL-4 basis change from single-threaded WASM to single-threaded native ORT
makes that ceiling *easier* and is logged in the same file as the one genuine
loosening in this increment.

### What was measured

Body swapped to `sentence-transformers/nli-distilroberta-base-v2` @ `cc35a0bf`
— 82,118,400 parameters, 6 layers, hidden 768 — the scorer increment 4 measured
at macro held-out AUC 0.880.

| Build | Model | Cold | Native p95 | Pearson r (worst dim) | max Δ score | CEIL-5 |
|---|---:|---:|---:|---:|---:|---|
| `int8_full` | 78.20 MiB | 92.91 MiB | **75.8 ms** | 0.99282 | **0.0770** | **fail** |
| `int8_embed` | 199.49 MiB | 214.20 MiB | 224.98 ms | 0.99995 | 0.00694 | pass |
| `fp32` | 311.07 MiB | 325.78 MiB | 226.41 ms | 1.0 | 0.0 | pass (reference) |

Three of the four pre-registered predictions held. Additivity survived the body
swap at a residual of 2.8–3.2e-07 against a 1e-04 rule (R6-1). CEIL-5 selected
`int8_embed` exactly as predicted, and `int8_full`'s 0.077 max delta is nearly
4× the tolerance — quantizing every MatMul in a hidden-768 body moves a displayed
score by almost 8 points out of 100. CEIL-4 passed natively with 2.2× headroom,
which was the prediction held with least confidence.

### The prediction that was wrong, and it is the one that decides the increment

> "CEIL-2 … it is met with room to spare either way."

**CEIL-2 fails. 3,559,258 bytes — 3.394 MiB against a 2.000 MiB ceiling, 70%
over.** distilroberta's tokenizer is a 50,265-entry byte-level BPE with a 50,000
-entry merges table, against the incumbent's 30,522-entry WordPiece. CEIL-2's
stated purpose on day 1 was to "force a rejection if someone swaps in a
[much larger] vocab without saying so". A vocabulary 1.65× larger was swapped in,
and the tripwire fired on it.

**So no build clears every enforced ceiling, and the 0.880 scorer is NOT
adopted.** `verify.py` exits 1. `shippable_builds` is empty.

CEIL-2 bounds a download, exactly as CEIL-1 and CEIL-3 do, so there is an
argument that it should not gate a desktop target either. That argument is not
being made here, because the enforcement map was fixed before the measurement and
rewriting it in the same increment that measured a 70% overage would make the
map a description of the result rather than a test of it. If CEIL-2's scope is
genuinely wrong for this target, it gets revisited in a later increment as its
own decision, on the record, with this failing number already published.

The open question for increment 7 is therefore narrow and testable: **is there a
serialization of this tokenizer, with the same 50,265-entry vocabulary and
identical encode() output, that fits 2 MiB?** If yes, CEIL-2 is met rather than
argued away. If no, the choice between re-scoping CEIL-2 and abandoning this body
is made explicitly and is not disguised as an engineering detail.

### DEFECT-INC6-001 — a stale benchmark was being read as a current one

Found by re-deriving rather than inheriting. `artifacts/wasm/bench_*.json`
carried no identity of the model it timed, and `export/verify.py` keyed those
files on the build *name* alone. The three bench files on disk described the
**previous** encoder, and on the first verify run of the new body they would have
been reported as its WASM latency — a MiniLM number printed as a distilroberta
number, in the artifact an auditor reads for CEIL-4.

`web/bench_wasm.mjs` now emits `model_sha256`; `export/verify.py` rejects any
bench whose sha does not match the file on disk and records it under
`wasm_benches_rejected_as_stale`. On its first run the guard rejected all three
stale files and reported `latency_wasm_1thread: "NOT_MEASURED"`, which is the
correct answer and the one the old code could not give.

### The WASM number, measured after the guard was in place

With `model_sha256` emitted and checked, `int8_embed` was re-benched under WASM
against the new body. The fresh bench was accepted; the two remaining stale files
(`fp32`, `int8_full`) were still rejected, so the guard was observed working in
both directions on the same run.

| | native ORT, 1 thread | WASM, 1 thread | CEIL-4 |
|---|---:|---:|---|
| `int8_embed` p95 | **224.98 ms** | **845.98 ms** | 500 ms |

**On the original web basis CEIL-4 fails too, at 1.7× the ceiling.** This was
pre-registered as the expected outcome, and it matters more than it looks: the
web target is not lost only on size. Even with CEIL-1 and CEIL-3 set aside, a
hidden-768 body does not score a 256-token entry inside the latency budget in
single-threaded WASM. So CEIL-1, CEIL-2, CEIL-3 **and** CEIL-4 all fail on the
web target, and the desktop fallback fails on CEIL-2 alone.

The attribution identity holds in the WASM runtime as well — residual 1.4e-07 —
and WASM and native agree on the logits to 3.3e-07, so the two runtimes are
measuring the same model.
