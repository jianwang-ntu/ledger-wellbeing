# Ledger

A client-side longitudinal well-being instrument, built for
**Hack for Humanity | Summer 2026** (focus: Mental health).

> **Status: in progress.** This README records only what is built and has been
> run. Claims about parts that do not exist yet are not made here. Every claim
> below is backed by a checked-in artifact under `audit/`.
>
> As of build increment 9 there is a working application **with an interface**.
> As of build increment 7 there is a shippable model build. None of that makes
> the score good — `docs/limitations.md` §1 is the thing to read before believing
> any number this produces.

## What it is meant to be

An instrument that makes a person's own between-visits experience legible to the
clinician actually treating them — not an AI that acts as a therapist. Inference
runs locally over the user's own writing and nothing leaves the device; the
output is per-span attribution over their words and a trajectory over time, not
generated advice.

The interface is a local one: `python -m ledger.ui` binds `127.0.0.1` and opens
in the browser you already have. Earlier versions of this README said the project
had **"no server component at all"**. That sentence is **withdrawn** as of
increment 9, because it stopped being true when the interface arrived — see
`docs/limitations.md` §7.6 for what replaced it and how each replacement was
measured. It is not reinterpreted to mean "no *remote* server"; the wording was
pre-committed in `export/INCREMENT_8_PREREGISTRATION.md` before the measurement
that would have made it awkward.

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
| `ledger/safety/` — deterministic crisis router | **built, tested** | `audit/runs/safety_tests_20260824T1350Z.txt` (8/8), `audit/runs/crisis_matrix_20260824T1350Z.json` (23/23) |
| `ledger/model/` — encoder + additive attribution head | **built** | `artifacts/verify_report.json` |
| `export/` — ONNX export, int8 quantization, verification | **built; clears every enforced ceiling since increment 7** | `artifacts/verify_report.json`, `artifacts/quant_sensitivity.json` |
| In-browser inference (`onnxruntime-web`, WASM) | **runs**, and **dropped as the target** — fails 4 of 6 ceilings on the current body, including latency at 845.98 ms | `artifacts/wasm/bench_int8_embed.json`, `export/SIZE_BUDGET.md` |
| Desktop target with the 0.880 scorer | **built, measured and ADOPTED** since increment 7 — CEIL-2 met at 1.484 MiB vs 2.000 MiB, `verify.py` exits 0, `int8_embed` selected | `artifacts/verify_report.json`, `export/SIZE_BUDGET.md` |
| `ledger/store/` — encrypted local journal, one-click wipe | **built, tested** | `tests/test_store.py` (25 guards), `docs/limitations.md` §7.2–7.3 |
| `ledger/app/` — entry → route → score → per-span attribution → store → report | **built, runs end to end** | `audit/runs/inc8_cli_demo_*.txt`, `artifacts/span_additivity.json` |
| Zero egress | **measured twice**: 0 socket calls of any kind with the CLI; 39 calls, all loopback, with the interface running | `artifacts/egress_audit.json`, `artifacts/egress_audit_ui.json`, `docs/limitations.md` §7.1 |
| Head training | **not started** — blocked on a permissively-licensed corpus | `data/MANIFEST.md` |
| Held-out separation of the five dimensions | **measured, and at chance on 4 of 5** | `artifacts/encoder_ablation.json`, `docs/limitations.md` §1 |
| A scorer that *does* separate held-out text | **found, and 6.3× too large to ship** | `artifacts/scorer_ablation.json`, `docs/limitations.md` §1a |
| A *small* scorer that separates **and** fits | **searched for, and does not exist at hidden ≤ 384** | `artifacts/size_feasible_scorer.json`, `docs/limitations.md` §1b |
| CLI front-end (`python -m ledger.app.cli`) | **built** | `audit/runs/inc8_cli_demo_*.txt` |
| Visual interface (`python -m ledger.ui`) | **built** — six views, served from 127.0.0.1, no third-party asset | `artifacts/a11y/screens/`, `tests/test_ui.py` (26 guards) |
| Accessibility (plan.md C6) | **measured in a real browser**: axe-core 0 violations / 0 incomplete on all six views; whole flow keyboard-only; focus verified optically | `artifacts/a11y_report.json` |
| Screen-reader traversal | **not done, not claimed** — needs a human | `docs/limitations.md` §7.4 |
| The listener is local and unreachable | **measured**: only bind `127.0.0.1`; port refused on all 9 non-loopback addresses | `artifacts/egress_audit_ui.json` |
| 4-minute submission video | **not produced**, and will not be faked | — |

## Install, and what a clone can and cannot do

Read this before the quickstarts below. A round-1 audit cloned this repository
and found that it did not run, while the README said "All tests pass offline".
That sentence has been replaced by the two true ones.

```bash
git clone https://github.com/jianwang-ntu/ledger-wellbeing && cd ledger-wellbeing
python3 -m pip install -r requirements.txt      # ~120 MiB: onnxruntime, numpy, tokenizers, cryptography
python3 -m pytest -q                            # exits 0 offline
```

**What ships in the clone**: all source, all `artifacts/*.json` reports, and the
1,556,504-byte compacted tokenizer — so a clone can tokenize, run the crisis
router, run the store and run every test that does not need model weights.

**What does not ship**: the ONNX and PyTorch weights. `artifacts/onnx/` is
900 MiB and `artifacts/torch/` is 314 MiB; the selected build alone,
`ledger_scorer_int8embed.onnx`, is 209,181,357 bytes. Putting them in git would
make a `git clone` of a $100 hackathon entry a 1.2 GiB download.

**So scoring an entry needs one build step**, and its true cost is:

```bash
python3 -m pip install -r requirements-build.txt   # adds torch + transformers, ~2.5 GiB
bash export/run_all.sh                             # downloads the encoder, exports, quantizes, verifies
```

| | cost |
|---|---|
| downloaded | ~2.5 GiB of build-only wheels, plus the ~330 MiB encoder from HuggingFace |
| peak disk under `artifacts/` | ~1.2 GiB |
| wall clock | tens of minutes on CPU, dominated by the fp32 ONNX export |
| network | **only here.** The build is the one step that touches the network; nothing at runtime does, and that is measured in `artifacts/egress_audit.json` |

**Two true sentences replacing "All tests pass offline":**

1. `python3 -m pytest -q` on a fresh clone **exits 0** — **230 passed, 24
   skipped**, measured at revision 1 and recorded verbatim in
   `audit/revision1/clean_clone_pytest.txt`. On a machine that has run
   `export/run_all.sh` the same command gives 245 passed, 9 skipped.
2. **15 of those 24 skips** are the tests that need the weights, and each names
   the command that produces them (`run `bash export/run_all.sh``). The other 9
   are pre-existing conditional guards that do not apply to the selected build.
   They **skip, never pass** — a green run on a clone never means more than it
   says.

What the tests assert is that the numbers in this README are re-derivable from
the artifacts, and — since increment 8 — that the application's own behaviour
matches what is claimed for it.

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

**It also missed one, and an independent audit found it.** The leetspeak table
mapped `1 → i` and nothing else, so `1` standing in for the letter `l` was never
folded — and every rule phrase containing an `l` (`kill myself`, `end my life`,
`self harm`, `suicidal`) could be written straight past the router. Ten of the
auditor's twelve probes evaded that way; `k1ll myse1f` normalised to
`kill myseif` and scored as an ordinary entry. This is exactly the failure the
suite existed to prevent, and the suite's single leetspeak case happened to be
the one direction where `1 → i` is correct.

The fix does not guess. Ambiguous glyphs (`1`, `!`, `|`) are no longer folded at
all; the *phrase* is expanded instead, so one glyph satisfies either letter it
could stand for. The matrix that guards it is generated from `RULES` itself —
every phrase × every ambiguous glyph × every letter that glyph can mean — so a
clinician adding a phrase cannot forget to test its evasions.

**And measuring it turned up something the audit did not find.** A 40-entry
ordinary-language control corpus fires **11 times**: `RULES` matches literal
substrings with no negation or topic handling, so "the article was about suicide
prevention funding" and "I am not suicidal, just tired" both route to a helpline.
The same 11 fired on the pre-fix router — 0 of them are caused by the widening
(`audit/revision1/fp_control_corpus.json`) — so this is a property the project
has always had and had never quantified. It is **not fixed**: negation handling
in an acute path is an evasion surface, and anything that learns "not" suppresses
a match can be written past with "I would never say I want to kill myself". The
asymmetry is chosen — a false positive costs a helpline card and a skipped trend
line, a false negative costs a missed crisis — and the 11 are pinned in
`tests/test_crisis_router.py::AcceptedFalsePositives` so the published rate
cannot drift from the measured one.

```
python3 -m pytest tests/test_crisis_router.py -v
```

## The model, and the one claim it is built to support

`ledger/model/scorer.py` is a frozen encoder — `nli-distilroberta-base-v2` since
increment 6, `all-MiniLM-L6-v2` before it — with mean pooling and a linear head
over five language dimensions. That shape is chosen for a
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

## Build increment 7 — the first shippable build

Increment 6 ended with exactly one enforced ceiling unmet: **CEIL-2, the
tokenizer, at 3,559,258 B against 2,097,152 B**. Increment 7 had one question,
pre-registered in `export/INCREMENT_7_PREREGISTRATION.md` before the decisive
measurement: is there a serialization of *this* tokenizer — same 50,265-entry
vocabulary, identical `encode()` output — that fits?

There is. `tokenizer.json` was written pretty-printed; JSON whitespace carries no
semantics. Compact re-serialization gives **1,556,504 B (1.484 MiB)**, and
re-tokenizing afterwards returns `input_ids`, `attention_mask`, `offset_mapping`
and `decode()` round-trips **elementwise identical** over the 64 probe entries and
all 50 anchor sentences. Zero mismatches. The parsed JSON document compares equal;
vocab 50,265 and merges 50,000 both unchanged.

`export/verify.py` exits **0** for the first time in this project.
`shippable_builds` is `["int8_embed", "fp32"]` and `int8_embed` is selected at
CEIL-4 228.15 ms (ceiling 500), CEIL-5 r 0.99995 / max Δ 0.00694 (ceilings 0.99 /
0.02), additivity residual 3.2e-07.

Two things this does **not** retire, both published in `export/SIZE_BUDGET.md`
rather than left to be inferred:

- The vocabulary really did grow 30,522 → 50,265, and CEIL-2's day-one tripwire
  fired on a real change. Meeting the ceiling by re-serialization satisfies it
  **as written** — it bounds bytes on disk — and does not unmake that signal.
- The win is thin against a ceiling denominated in *transfer* size, since
  compression would have removed most of that whitespace anyway. CEIL-2 is
  denominated in bytes on disk. Both readings are in the budget file.

One defect was found and fixed in the same increment: two guards had been green
for four increments only because `shippable_builds` was empty and they could not
fail (DEFECT-INC7-001). They were replaced with strictly stronger guards rather
than relaxed. 103 tests pass; 16 new guards caught 14 of 14 targeted mutations.

## Build increment 8 — there is an application now

Everything before this was a model and a set of measurements. The competition
asks for **functioning source code**, and there was none: no way to enter
anything, no storage, no attribution surface, no report. Increment 8 builds the
smallest honest end-to-end path.

```
python3 -m ledger.app.cli init
echo "I lay awake until nearly four again." | python3 -m ledger.app.cli --region SG add
python3 -m ledger.app.cli report
python3 -m ledger.app.cli wipe
```

| Piece | What it is |
|---|---|
| `ledger/store/` | Append-only encrypted journal. scrypt (n=2^15, r=8, p=1) → AES-256-GCM per record, fresh nonce, file identity and record position bound into the AAD. `0600` from creation. One-click wipe: two overwrite passes, then unlink. |
| `ledger/app/spans.py` | Regroups per-token attribution into the user's own words and sentences, as a **partition** — no token dropped, none double-counted, structural tokens reported rather than hidden. |
| `ledger/app/engine.py` | Route → score → attribute. On an acute crisis match the model is **not loaded, let alone run**. |
| `ledger/app/report.py` | The clinician-shareable output: trajectory, the spans that moved each score, and the strength of the evidence for each dimension. |
| `ledger/app/cli.py` | A front-end. Not a designed interface — see below. |

### The four things that were measured rather than asserted

Rules fixed in `export/INCREMENT_8_PREREGISTRATION.md` **before** any of this
existed.

| | Rule | Result |
|---|---|---|
| **R8-1** | Span aggregation is a partition | **PASS** — 128 checks, 0 failures |
| **R8-2** | Span attributions still sum to the logit, ≤ 1e-4 | **PASS** — max residual **4.41e-07** over 640 checks, at both granularities. The aggregation step got no tolerance of its own. |
| **R8-3** | Zero egress, on the running application | **PASS** — **0 socket calls of any kind** across init, 5 entries, scoring, read-back, report and wipe |
| **R8-4** | The model never runs on an acute entry | **PASS** — session `run()` count 0, and the session is never even constructed |

Plus R8-5/R8-6 on the store (no 12-character run of any entry appears in the
file; a flipped byte, a reordered record and a record from another store are all
rejected rather than decrypted; wipe overwrites before unlinking) and R8-7/R8-8
on what the product says.

`export/egress_audit.py` was committed before it was run. It carries its own
positive control: `tests/test_egress.py` makes a real outbound connect under the
same instrumentation and fails if it is not recorded — because a green guard over
an instrument that cannot fail is exactly DEFECT-INC7-001 again.

### Three defects, two found by running the thing

**DEFECT-INC8-001, found by the first end-to-end run.** The application loaded its
tokenizer through `transformers`, which imports `sklearn`, which imports
`pyarrow` — and on this machine that fails outright once `onnxruntime` or `torch`
has been imported first (`CXXABI_1.3.15 not found`). The test suite never saw it,
because pytest's collection order happens to import things in an order that
works. A product whose startup depends on import order is broken.

Fixed by reading `tokenizer.json` with the `tokenizers` library directly and the
head bias from the build report, which removes `transformers`, `sklearn`,
`pyarrow` **and** `torch` from everything the shipped application touches. The
runtime dependency set is now `onnxruntime`, `numpy`, `tokenizers`,
`cryptography`. That is only legitimate if the encodings are identical, so
`export/tokenizer_parity.py` compares `input_ids`, `attention_mask` and
`offset_mapping` elementwise over 114 texts: **0 mismatches**, and the head bias
is bit-identical after its float32 cast.

**A claim defect, also found by running it.** The report said "the contributions
listed above are the score: they sum to it exactly" while listing only the
largest three. The first real entry came out with a dimension at 0.60 and all
three displayed spans negative — the offset and the unshown spans carry the rest.
The text now says what is true: only the largest few are listed, and the complete
set (every span, the structural tokens, one fixed offset per dimension) is what
adds up. The CLI prints the remainder terms so the sum visibly closes.

**A guard gap and a harness bug, found by the mutation run.** Blanking the
`[NOT ESTABLISHED]` marker in the trajectory section left it in the other section
and the guard stayed green; it now requires the flag in both. And the harness
itself replaced the first occurrence of its target string, which for one mutation
was a docstring — reporting MISSED against a guard it had never given anything to
catch. It now refuses any target that does not occur exactly once.

### Where it stands

212 tests pass, 9 skipped (all pre-existing conditional guards). The increment-8
guards were checked against **30 targeted mutations and caught 30 of 30** —
including lowering the scrypt cost, skipping records that fail authentication,
appending entries as plaintext, blanking the not-established marker, importing
`transformers` again, and widening the additivity tolerance in either of the two
files that hold it.

Still open, and stated plainly:

- **`plan.md` C6 has no artifact.** A CLI is not a visual design. No
  `a11y/axe_report.json`, no recorded screen-reader traversal.
- **The 4-minute video does not exist** and will not be faked.
- **The head is still untrained** (plan.md R-1), `activation` still separates
  held-out text at 0.600 against a 0.700 floor, and the application labels it
  `[NOT ESTABLISHED]` everywhere it appears rather than averaging it away.
- **The egress measurement is process-level, not a packet capture.**
  `docs/limitations.md` §7.1 says exactly how far it reaches.


## Build increment 9 — the interface, and what measuring it actually found

`plan.md` C6 is graded on "visual design quality, ease of navigation, intuitive
user flows, and adherence to accessibility standards". Through increment 8 it had
**zero** artifact, declared as a deliberate hole before that increment started.
This increment closes it — partly.

Run it with `python -m ledger.ui`. Six views: open the journal, write, read the
per-span attribution, history, report, journal & safety. Screenshots of every one
of them, as rendered by the browser the harness drove, are in
`artifacts/a11y/screens/`.

### What was measured, and against what

Every threshold below was written into `export/INCREMENT_9_PREREGISTRATION.md`
and committed **before** the interface existed.

| Rule | Result |
|---|---|
| R9-1 keyboard completeness | The whole primary flow — create journal → write → attribution → history → report → decline the wipe — with **0 pointer events**. A keyboard-activated button fires `click` with `detail === 0`; a mouse one does not, so the counter can tell them apart. |
| R9-2 visible focus, optically | **48 of 48** focusable elements. The element's own box is screenshotted focused and unfocused and diffed; minimum **1.9%** of pixels changed against a 0.5% rule. A `:focus { outline: none }` that repaints invisibly passes a style check and fails this one. |
| R9-3 axe-core | **0 violations, 0 incomplete**, six views, `wcag2a, wcag2aa, wcag21a, wcag21aa`. |
| R9-4 contrast + forced colours | axe's `color-contrast` **ran** (it needs real layout — this is why a browser, not jsdom) and passed on every view. An independent pass over every text-bearing element measures **7.38:1 minimum** by default and **13.99:1** under `forced-colors: active`, where the flow completes again unchanged. |
| R9-5 reduced motion | **744 elements** enumerated in the live DOM, every one reporting `0s` animation and transition. |
| R9-6 announcements | The live region carries the scored result and the crisis routing, and it **is** a live region — `aria-live="polite"`, `role="status"`, crisis panel `role="alert"`. |
| R9-7 the listener is local | Only bind is `127.0.0.1`. The port is **refused on all nine non-loopback addresses** on this host. The interface list is read from the kernel by `ioctl`, not from a name lookup. |
| R9-8 zero egress survives | 39 socket calls driving the interface in-process, **all loopback**, zero DNS. In the browser, **30 of 30** requests to the loopback origin. |
| R9-9 no new claims | `head_is_trained` still `false`. Every dimension below the 0.70 floor renders a **NOT ESTABLISHED** badge with its AUC. The banned-vocabulary check now runs over the **rendered page text**. |
| R9-10 nothing else moved | No `CEILINGS` value edited, `export/common.py` byte-identical, `verify.py` exits 0 with `int8_embed`, suite green. |

### Five defects, four of them found by measuring rather than by reading

The first run of the harness came back green on every rule. Four of those greens
were wrong, and the way they were found is the point.

1. **DEFECT-INC9-001** — R9-8(b) counted the forced-colours pass's own loopback
   server as "external", failing a rule the product had not broken. The rule is
   about `127.0.0.1`; the check had been written about one port.
2. **DEFECT-INC9-002** — R9-2 reported **PASS on two views where it measured
   nothing**. It scanned a section that was hidden at the time, found zero
   focusable elements, and found zero failures among them. A vacuity guard now
   fails a view that measures nothing, and the scan covers the whole document —
   the skip link and the section rail live outside every view and were never
   being looked at.
3. **DEFECT-INC9-003** — R9-6 read the live region 30 ms too early and passed on
   the **previous** message ("Reading your entry on this machine…"). It now waits
   for the text it expects.
4. **DEFECT-INC9-004** — R9-2 called a radio button "unreachable" because Tab
   skipped it. Tab is *supposed* to skip it: a radio group is one tab stop and
   the arrow keys move within it. The harness was wrong, not the interface.
5. **DEFECT-INC9-005** — found by looking at a rendered screenshot: the
   NOT ESTABLISHED card said "threshold fixed before it was measured" **twice**,
   once from the page's own template and once from `ledger/app/evidence.py`. One
   source, one sentence.

### Then the guards were mutated, and two of them were too weak

`export/mutation_check_inc9.py` breaks each property and checks that the guard
notices. The first full run caught **17 of 19**. The two misses were real:

- Stripping `role="status" aria-live="polite"` off the status element left R9-6
  **green** — the text still arrived, so a check that only read `textContent`
  could not tell an announcement from a repaint. R9-6 now checks the attributes.
- Removing a `<label>` from the textarea left axe **green**, because axe-core
  accepts a non-empty `placeholder` as an accessible name. A placeholder vanishes
  the moment someone types. R9-3 now also requires every control to be named by a
  label, `aria-label` or `aria-labelledby` — a **tightening** of the rule after a
  mutation showed it was too loose, recorded here rather than quietly folded in.

After both fixes: **19 of 19 caught**.

### What is still not true

- **No screen-reader traversal exists**, and none is claimed. Conformance is a
  floor, not a substitute for someone listening to the thing.
- **The 4-minute video is still missing** and will not be faked.
- **The head is still untrained** and `activation` still sits at 0.600 against a
  0.700 floor, labelled everywhere it appears.
- **The packet-level capture is still owed**; the egress measurement is
  process-level and is described as such.
