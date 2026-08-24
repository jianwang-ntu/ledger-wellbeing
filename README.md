# Ledger

A client-side longitudinal well-being instrument, built for
**Hack for Humanity | Summer 2026** (focus: Mental health).

> **Status: in progress.** This README records only what is built and has been
> run. Claims about parts that do not exist yet are not made here. Every claim
> below is backed by a checked-in artifact under `../audit/`.

## What it is meant to be

An instrument that makes a person's own between-visits experience legible to the
clinician actually treating them — not an AI that acts as a therapist. Inference
runs locally over the user's own writing and nothing leaves the device; the
output is per-span attribution over their words and a trajectory over time, not
generated advice.

**The delivery target changed, and the reason is a measurement, not a
preference.** The plan was a browser app. Meeting the size ceiling set on day one
requires an encoder small enough that no entailment-supervised checkpoint we could
find both fits it and separates held-out text — the one that fits scores at
chance. Rather than move the ceiling or drop the claim quietly, plan.md's
pre-written R-4 fallback was taken in increment 5 and **exercised** in increment
6: `export/common.py` now pins the 0.880 scorer and `DELIVERY_TARGET = "desktop"`.
Zero-egress survives, because it never depended on the browser. The in-browser
claim does not.

Increment 6 also settled that the browser was not lost on size alone. Re-measured
on the new body, the web target fails **four** ceilings — model bytes, tokenizer
bytes, cold payload, and latency at **845.98 ms in single-threaded WASM against a
500 ms ceiling**. And it did not end in a shippable build either: **CEIL-2 fails
at 3.394 MiB against 2.000 MiB**, so no build was adopted. `export/SIZE_BUDGET.md`
has both tables; `export/INCREMENT_6_PREREGISTRATION.md` has the rule that decided
it, written before the measurement.

## What is built and verified so far

| Component | State | Runtime evidence |
|---|---|---|
| `ledger/safety/` — deterministic crisis router | **built, tested** | `../audit/runs/safety_tests_20260824T1350Z.txt` (8/8), `../audit/runs/crisis_matrix_20260824T1350Z.json` (23/23) |
| `ledger/model/` — encoder + additive attribution head | **built** | `artifacts/verify_report.json` |
| `export/` — ONNX export, int8 quantization, verification | **built, and currently failing its own ceiling** | `artifacts/verify_report.json`, `artifacts/quant_sensitivity.json` |
| In-browser inference (`onnxruntime-web`, WASM) | **runs**, and **dropped as the target** — fails 4 of 6 ceilings on the current body, including latency at 845.98 ms | `artifacts/wasm/bench_int8_embed.json`, `export/SIZE_BUDGET.md` |
| Desktop target with the 0.880 scorer | **built and measured; NOT adopted** — CEIL-2 (tokenizer) fails at 3.394 MiB vs 2.000 MiB | `artifacts/verify_report.json`, `export/SIZE_BUDGET.md` |
| Head training | **not started** — blocked on a permissively-licensed corpus | `data/MANIFEST.md` |
| Held-out separation of the five dimensions | **measured, and at chance on 4 of 5** | `artifacts/encoder_ablation.json`, `docs/limitations.md` §1 |
| A scorer that *does* separate held-out text | **found, and 6.3× too large to ship** | `artifacts/scorer_ablation.json`, `docs/limitations.md` §1a |
| A *small* scorer that separates **and** fits | **searched for, and does not exist at hidden ≤ 384** | `artifacts/size_feasible_scorer.json`, `docs/limitations.md` §1b |
| UI | not started | — |

All tests pass offline: `python -m unittest discover -s tests`. What they assert is that the numbers in this README are re-derivable from the artifacts, not that the project works.

## The safety layer

Crisis routing is rule-based and runs **before** any inference. The model is
never in the crisis path, so no model output can suppress a routing decision —
the router does not consult one. `tests/test_crisis_router.py` is written to try
to make it fail to fire: leetspeak, letter-spacing, separator insertion,
homoglyphs, and five prompt-injection attempts that instruct the system to stay
silent.

That suite has already caught one real evasion in this code: the dotless-i
homoglyph `U+0131` survived NFKD normalisation, so `suıcıde` routed past an
earlier version of the normaliser. The fix is the `_CONFUSABLES` table in
`ledger/safety/crisis_router.py`; the case is now a permanent test.

```
python3 -m unittest discover -s tests -v
```

## The model, and the one claim it is built to support

`ledger/model/scorer.py` is a frozen `all-MiniLM-L6-v2` encoder, mean pooling,
and a linear head over five language dimensions. That shape is chosen for a
single reason: mean pooling followed by a linear layer is a **sum of per-token
terms**, so

```
logit_k  ==  sum over tokens i of token_attr[i, k]  +  bias_k
```

holds *exactly*, not approximately. The explanation is not a saliency map fitted
to the score afterwards — it is the arithmetic the score is made of. Measured
residual on the current body: **1.4e-07 in the WASM runtime**, 2.8–3.2e-07
natively across all three builds (`artifacts/wasm/bench_int8_embed.json`,
`artifacts/verify_report.json`), asserted on every build by
`tests/test_export_pipeline.py` and by
`tests/test_delivery_target.py::TestAdditivitySurvivedTheBodySwap`. The identity
survived the increment-6 encoder swap; it was re-measured rather than inherited.

**The head is not trained.** Each row is the difference between two centroids of
anchor phrases written for this project (`ledger/model/dimensions.py`), affinely
calibrated so the poles land near ±2. It is labelled `anchor_v0` and
`head_is_trained` is `false` in `artifacts/torch/build_report.json`; a test fails
if this README ever claims a fine-tune while that flag is false. Supervised
training is blocked on `data/MANIFEST.md` — no mental-health corpus with a clear
permissive licence has been cleared yet, and hackathon rule 2 requires that it be
cleared rather than assumed.

## Where the in-browser plan stands: a ceiling we set, and did not meet

`export/SIZE_BUDGET.md` fixes five ceilings **before** anything was exported, so
that `export/verify.py` is a test the pipeline can fail. It failed. Reported here
rather than in a footnote:

| Build | Size | Cold load | WASM p95 | Agreement with fp32 (max Δ on the 0–1 score) | Verdict |
|---|---:|---:|---:|---:|---|
| int8, all ops | 21.78 MiB | 33.78 MiB | 124 ms | **0.0404** (ceiling 0.02) | **fails CEIL-5** |
| int8 embeddings only | 52.04 MiB | 64.04 MiB | 236 ms | 0.0094 | **fails CEIL-1**, and CEIL-3 by 39,892 bytes |
| fp32 reference | 86.14 MiB | 98.1 MiB | 238 ms | — | not a shipping candidate |

`export/verify.py` exits **1**, and `shippable_builds` is `[]`.

`export/quant_sensitivity.py` is the search that was run before accepting that,
kept in the repository so nobody repeats it. Its finding is the useful part:
**size and accuracy live in different halves of this model.** The 11.7M-parameter
embedding table is 51% of the weights and quantizing it costs almost nothing
(max Δ 0.0094); the 10.6M parameters in the six encoder layers are where int8
loses agreement, and excluding any single layer costs 5.5 MiB while moving the
error by less than 0.01 — two of the six exclusions make it *worse*. There is no
cheap subset that buys CEIL-5 back.

A second finding, which matters more than the first: under WASM the int8 build
disagrees with native onnxruntime by **1.3e-02** on the logits, while the fp32
and embeddings-only builds agree to **5e-07**. A user's score would depend on
which kernel path their browser took. That is an argument against the int8 build
independent of any ceiling.

Neither the ceilings nor the claims were adjusted to fit these numbers.

## The small-scorer route was measured, and it closed

Increment 4 left a working scorer (0.880 held-out) at 201.68 MiB and a shippable
scorer at chance, so increment 5 asked the only remaining cheap question: is
there an entailment-supervised bi-encoder small enough to fit? The candidate list
was committed before the run (`c1727e1`) and shippability was promoted into the
selection rule, so a scorer that separates and does not fit could not be
"selected" again.

| Scorer | H | L | Held-out AUC | Control | int8-embed | Fits CEIL-1? |
|---|---:|---:|---:|---:|---:|:---:|
| `xtremedistil-l6-h256-zeroshot` | 256 | 6 | 0.504 | 0.628 | **25.91 MiB** | **yes** |
| `paraphrase-MiniLM-L3-v2` | 384 | 3 | 0.592 | 0.792 | 32.24 MiB | no, by 0.24 MiB |
| `nli-deberta-v3-xsmall` (body) | 384 | 12 | 0.608 | 0.648 | 128.90 MiB | no |
| `deberta-v3-xsmall-zeroshot` | 384 | 12 | 0.616 | 0.644 | 128.90 MiB | no |
| `nli-distilroberta-base-v2` *(reference)* | 768 | 6 | **0.880** | 0.916 | 201.68 MiB | no |

The one that fits separates at 0.504 — the incumbent's number — on a positive
control below the 0.75 floor, so it is not even readable. **Nothing was
selected**, `export/common.py` is unchanged, and no ceiling moved: the 32.24 MiB
row missing a 32.00 MiB ceiling by a quarter of a megabyte is exactly the edit
`tests/test_size_feasible_scorer.py` exists to forbid.

That closes the last route that would have kept the browser target, so R-4 was
invoked. It is worth being plain about what this costs and what it does not: the
scoring defect in `docs/limitations.md` §1 is **not** fixed by this — a desktop
target makes the 0.880 scorer *shippable*, it does not make its 0.600
`activation` dimension work.

## The narrower encoder was measured, and not taken

Increment 2 ended by naming the route through: a **narrower encoder**, hidden 256
rather than 384, encoder layers left fp32 and int8 on the embedding table only,
projected at ~26.7 MiB. Increment 3 measured it before adopting it, because
`all-MiniLM-L6-v2` is a sentence-embedding distillation and the
`google/bert_uncased_*` miniatures are general pretrained checkpoints — trading
26 MiB for an unmeasured loss of signal is exactly the sort of thing that should
not be done quietly.

The size half of the projection held. `google/bert_uncased_L-6_H-256_A-4` comes
out at **25.91 MiB**, comfortably inside CEIL-1. The other half did not survive
contact with a measurement, and it took the whole increment with it:

| Encoder | int8-embed size | CEIL-1 | Held-out AUC | In-sample AUC | Positive control |
|---|---:|:---:|---:|---:|---:|
| `all-MiniLM-L6-v2` (incumbent, H384 L6) | 52.55 MiB | fail | **0.504** | 1.000 | 0.860 |
| `bert_uncased_L-8_H-256_A-4` | 31.93 MiB | pass | 0.400 | 1.000 | 0.844 |
| `bert_uncased_L-6_H-256_A-4` | 25.91 MiB | pass | 0.392 | 1.000 | 0.824 |
| `bert_uncased_L-4_H-256_A-4` | 19.88 MiB | pass | 0.432 | 1.000 | 0.844 |

**Held-out AUC 0.504 is chance, and every candidate is at chance, incumbent
included.** The zero-shot anchor head separates the sentences it was built from
perfectly (in-sample 1.000 — which a 5-versus-5 centroid split in 384 dimensions
does regardless of the encoder) and separates sentences it has not seen no better
than a coin. Only `low_mood` generalises, at 0.960; `anxiety` is 0.240.

The positive control is what makes that readable rather than dismissible: the
same held-out protocol asked to separate the positive poles of two *different*
dimensions scores 0.860, with 4 of its 10 pairs at 1.000. The protocol works. So
does the encoder, at what it is for. The head is what does not.
`representation_geometry` says why — opposite poles of one dimension sit at
cosine 0.324 to each other against 0.243 to other dimensions, so the space
carries subject matter and not direction, which is why more anchors and a
different frozen encoder both fail to help.

So the encoder was **not** switched. `export/common.py` still holds the
incumbent, and `tests/test_encoder_ablation.py` fails if it ever disagrees with
the ablation's selection. Shrinking the model was set aside because it would have
optimised the size of a scorer that does not yet discriminate.

Full numbers, protocol and the routes that remain: `docs/limitations.md` §1 and
`artifacts/encoder_ablation.json`.

```
python export/encoder_ablation.py    # exits 2 while no candidate is selectable
```

## The scorer that works, and why it is not in this build

Increment 3 left four routes out of a head that scores at chance. Increment 4
measured the one that was unmeasured — polarity-aware scoring — under the same
held-out protocol, the same positive control, and the same 0.70 / 0.75
thresholds, imported from the increment-3 script rather than re-typed.

| Scorer | Held-out AUC | Positive control | Additive? | Projected int8-embed |
|---|---:|---:|:---:|---:|
| `all-MiniLM-L6-v2`, centroid difference *(incumbent)* | 0.504 | 0.860 | yes | 52.55 MiB |
| `all-MiniLM-L6-v2`, shrunk LDA | 0.544 | 0.864 | yes | 52.55 MiB |
| `distilbert…sst-2` body, mean-pooled | 0.672 | **0.584** | yes | 184.96 MiB |
| **`nli-distilroberta-base-v2`, centroid difference** | **0.880** | **0.916** | **yes** | **201.68 MiB** |
| `cross-encoder/nli-distilroberta-base` | 0.784 | **0.656** | no | 201.69 MiB |
| `cross-encoder/nli-MiniLM2-L6-H768` | 0.896 | **0.684** | no | 201.69 MiB |
| one global sentiment score *(diagnostic)* | 0.792 | **0.608** | no | 187.22 MiB |

Three things in that table, in the order they matter.

**A bold control number is a disqualification, not a footnote.** Four scorers —
including the 0.896, the best held-out number anywhere in this project — have a
positive control under the 0.75 floor. That floor means the scorer cannot tell
two different dimensions apart, so its pole separation is not evidence about the
dimension it is credited to. The floor was fixed in increment 3 before any of
this was run. That is the only thing that makes it legitimate to apply it to a
number we would have liked to keep.

**One scorer clears every gate.** An NLI-supervised bi-encoder, still mean-pooled
and still read out by a fixed linear row, takes held-out macro from 0.504 to
**0.880** on the highest control in the table. `anxiety` goes 0.240 → 0.880,
`sleep_disruption` 0.360 → 0.960. The attribution identity is untouched, because
the head did not change. **`activation` is 0.600 and remains below the usable
threshold** — the macro clears, that dimension does not, and it is named here
rather than averaged away.

**And it is 201.68 MiB against a 32 MiB ceiling.** So it was *not* adopted.
`export/common.py` still holds the incumbent, and
`tests/test_scorer_ablation.py` fails if the repository adopts a scorer that
breaches a ceiling *or* declines to adopt one that does not. The state of this
project is: the defect is understood and demonstrably fixable, and it is not
fixed in any artifact here.

```
python export/scorer_ablation.py     # exits 0; the selection it makes is not adopted
```

Full numbers and the routes remaining: `docs/limitations.md` §1a.

Reproduce all of the above:

```
bash export/run_all.sh          # exits non-zero while no build clears every ceiling
```

## How this was created — disclosure

This entry is being developed by an autonomous coding agent (Claude, Anthropic)
operating as the entrant's tool, on the entrant's own machine and accounts. The
entrant is a solo participant.

This is disclosed deliberately and in advance. Rule 3 of the hackathon states
"You may not receive any assistance from others outside of your group"; we read
that clause as governing assistance from other *people*, and rule 2 — "You must
only use resources and assets you have access to and must attribute
appropriately" — as the clause governing tools, which requires attribution
rather than abstention. If the organisers read rule 3 as covering AI agents,
they learn it here rather than discovering it.

## Licence

MIT — see `LICENSE`.

## Build increment 6 — the fallback taken, and what it cost

The body is now `sentence-transformers/nli-distilroberta-base-v2` @ `cc35a0bf`
(82.1M parameters, 6 layers, hidden 768, Apache-2.0), the scorer increment 4
measured at macro held-out AUC **0.880** where the previous body sits at 0.504 —
chance. `DELIVERY_TARGET` is `desktop`.

| Build | Model | Cold | native p95 | WASM p95 | worst-dim r | max Δ score | CEIL-5 |
|---|---:|---:|---:|---:|---:|---:|---|
| `int8_full` | 78.20 MiB | 92.91 MiB | 75.8 ms | not measured | 0.99282 | **0.0770** | **fail** |
| `int8_embed` | 199.49 MiB | 214.20 MiB | 224.98 ms | **845.98 ms** | 0.99995 | 0.00694 | pass |
| `fp32` (reference) | 311.07 MiB | 325.78 MiB | 226.41 ms | not measured | 1.0 | 0.0 | pass |

**No build was adopted.** `export/verify.py` exits 1 and `shippable_builds` is
empty, because CEIL-2 — the tokenizer, at **3,559,258 B against a 2,097,152 B
ceiling** — is enforced on the desktop target and fails. distilroberta's 50,265
-entry byte-level BPE is 1.65× the previous WordPiece vocabulary, and CEIL-2's
written purpose on day one was to force exactly this rejection.

CEIL-1 and CEIL-3 bound an HTTP first-load and stopped gating when the target
stopped being a download. That is the one relaxation in this increment and it is
bounded three ways: no ceiling *value* was edited, every build still reports
`would_fail_web_target_on`, and `tests/test_delivery_target.py` fails if anything
beyond CEIL-1 and CEIL-3 is ever dropped. CEIL-4's basis moved from WASM to native
ORT for the same reason, which makes it easier, and that is logged as a relaxation
in the pre-registration rather than absorbed silently.

CEIL-2 also bounds a download, so there is an argument it should not gate a
desktop target either. That argument is not made here. The enforcement map was
fixed before the measurement, and rewriting it in the increment that measured a
70% overage would turn the map into a description of the result. It is a decision
for a later increment, on the record, with the failing number already published.

### A defect found by re-deriving rather than inheriting

`artifacts/wasm/bench_*.json` carried no identity of the model it timed, and
`export/verify.py` keyed those files on the build *name*. The three benches on
disk described the **previous** encoder, so on the first run against the new body
a MiniLM latency would have been reported as a distilroberta latency, in the
artifact an auditor reads for CEIL-4. `web/bench_wasm.mjs` now emits
`model_sha256`; `export/verify.py` rejects any bench whose sha does not match the
file on disk. On its first run the guard rejected all three stale files.

87 tests pass (73 before). The 22 new guards were verified against 12 targeted
mutations — including raising CEIL-2 to 4 MiB, dropping CEIL-2 or CEIL-4 from the
desktop enforcement map, and re-pinning the encoder to a model no ablation
selected — and caught 12 of 12.
