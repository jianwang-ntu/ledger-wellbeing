# docs/limitations.md — what Ledger does not establish

plan.md criterion **C2** ("Feasibility & Safety") names this file. It exists to
state, in the place a reader will actually look, what this tool has *not* shown.
Everything here is a measurement in this repository, not a caveat written to
sound careful.

Last updated: build increment 5, 2026-08-24.

---

## 1. The five-dimension score does not generalise. This is the headline.

**Measured:** `artifacts/encoder_ablation.json`, reproducible with
`python export/encoder_ablation.py`.

The head is a zero-shot construction: each dimension's row is the difference
between the centroids of five "positive-pole" and five "negative-pole" anchor
sentences (`ledger/model/dimensions.py`). Build increment 3 asked the obvious
question — does that separate sentences it was *not* built from? — and measured
it three ways.

| Protocol | What it does | Result (incumbent encoder) |
|---|---|---:|
| In-sample AUC | direction from all 5+5 anchors, scored on those same anchors | **1.000** |
| Positive control | same held-out protocol, but separating the positive poles of two *different* dimensions | **0.860** |
| **Held-out AUC** | withhold one positive and one negative together, rebuild the direction from the remaining 4+4, rank the two withheld sentences | **0.504** |

Held-out AUC by dimension:

| Dimension | Held-out AUC |
|---|---:|
| `low_mood` | 0.960 |
| `social_withdrawal` | 0.560 |
| `activation` | 0.400 |
| `sleep_disruption` | 0.360 |
| `anxiety` | 0.240 |

**0.504 is chance.** Four of the five dimensions are at or below it. Only
`low_mood` separates text the head was not built from.

The in-sample 1.000 is what makes this worth stating plainly: a 5-versus-5
centroid split in a 384-dimensional space separates its own training points
perfectly no matter what the encoder does, so the perfect number is a
memorisation artifact and means nothing. Quoting it alone would be the natural
way to make this head look like it works.

The positive control at 0.860 is what rules out the alternative explanation. A
held-out protocol on ten sentences per dimension could easily be measuring its
own bias rather than the head; the control shows it separates topically distinct
sets perfectly well (4 of its 10 pairs score 1.000). The protocol works. The head
does not.

*(An earlier revision of the measurement withheld one sentence at a time rather
than one from each pole. That biases against the withheld point — its own
centroid moves away from it while the opposing centroid keeps all five members —
and it produced systematically below-chance numbers for every encoder. It was
wrong and was replaced before anything was concluded from it. Recorded here
because a discarded measurement that nobody mentions is indistinguishable from
one that was never run.)*

### Why, and what does not fix it

`representation_geometry` in the same artifact: the two poles of one dimension
sit at mean cosine **0.324** to each other, against **0.243** to the sentences of
other dimensions. The space is organised by *topic*, not by *direction*. Mean-
pooled sentence embeddings put "I could not get to sleep until four" and "I slept
straight through and woke up rested" close together, because they are both about
sleep. `sleep_disruption` is the most topic-locked dimension (cross-pole cosine
0.520) and it is also one of the worst performing (0.360).

That geometry rules out the cheap fixes. **More anchor sentences cannot buy
polarity out of a representation that does not carry it**, and neither can a
different frozen encoder: the same measurement over three `google/bert_uncased_*`
miniatures gives held-out macro AUC 0.392–0.432, all at chance, with positive
controls of 0.82–0.84.

Increment 4 tested the last cheap explanation and it did not survive either.
**It is not the readout.** A covariance-aware linear readout — shrunk LDA,
ridge 0.1 of the mean eigenvalue, refitted inside the same held-out loop — over
the *same* incumbent representation reaches 0.544 against the centroid
difference's 0.504. Both are chance. The direction really is absent from the
space, exactly as the geometry said, and no linear rearrangement of that space
recovers it.

### §1a. A polarity-supervised encoder does fix the separation — and cannot ship

**Measured:** `artifacts/scorer_ablation.json`, reproducible with
`python export/scorer_ablation.py`. Same held-out protocol, same positive
control, same thresholds — 0.70 usable, 0.75 control floor — imported from
`export/encoder_ablation.py` rather than re-typed so they could not drift.

Seven scorers, three families. The question was whether *polarity-aware* scoring
recovers what mean-pooled similarity cannot, and at what cost to the exactly
additive attribution head (§6).

| Scorer | Held-out AUC | In-sample | Positive control | Additive? | Projected int8-embed |
|---|---:|---:|---:|:---:|---:|
| `all-MiniLM-L6-v2`, centroid difference *(incumbent)* | 0.504 | 1.000 | 0.860 | yes | 52.55 MiB |
| `all-MiniLM-L6-v2`, shrunk LDA | 0.544 | 1.000 | 0.864 | yes | 52.55 MiB |
| `distilbert…sst-2`, body mean-pooled | 0.672 | 0.920 | **0.584** | yes | 184.96 MiB |
| **`nli-distilroberta-base-v2`, centroid difference** | **0.880** | 1.000 | **0.916** | **yes** | **201.68 MiB** |
| `cross-encoder/nli-distilroberta-base`, entailment | 0.784 | — | **0.656** | no | 201.69 MiB |
| `cross-encoder/nli-MiniLM2-L6-H768`, entailment | 0.896 | — | **0.684** | no | 201.69 MiB |
| `distilbert…sst-2`, one global sentiment score *(diagnostic)* | 0.792 | — | **0.608** | no | 187.22 MiB |

**Four of the seven have a positive control below the 0.75 floor, and their
held-out numbers are therefore not readable** — including the two that score
highest of all. A control below the floor means the scorer cannot tell two
*different* dimensions apart, so a high pole-separation number is being produced
by something other than the dimension it is attributed to. The floor was fixed in
increment 3, before any of this was run, which is the only reason it can be
applied to a 0.896 without it looking like an excuse. This is not a finding that
cross-encoders are bad at this task: it is a finding that **under this protocol
their numbers cannot be read**, and a protocol that can read them was not built.

The global-sentiment row is a deliberate diagnostic, excluded from selection
before it was run. It separates poles well (0.792) and fails the control (0.608)
because one number cannot distinguish five dimensions. That is the clean
demonstration of the trade-off: polarity without topic is as useless here as
topic without polarity.

What survives every gate is one scorer. `nli-distilroberta-base-v2` —
NLI-supervised, still mean-pooled, still a fixed linear row — reaches **held-out
macro 0.880 on the highest positive control in the table (0.916)**, and it keeps
the attribution identity intact, because nothing about the head changed.

Per dimension, under the selected scorer, against the incumbent:

| Dimension | Incumbent | `nli-distilroberta-base-v2` |
|---|---:|---:|
| `low_mood` | 0.960 | 1.000 |
| `sleep_disruption` | 0.360 | 0.960 |
| `social_withdrawal` | 0.560 | 0.960 |
| `anxiety` | 0.240 | 0.880 |
| **`activation`** | 0.400 | **0.600** |

**`activation` is still below the 0.70 usable threshold.** The macro clears it;
that one dimension does not, and a macro average is exactly the number that would
hide it. Whatever ships, `activation` is not a dimension this project has shown
to work.

### And then the size, which is the reason it is not adopted

`sentence-transformers/nli-distilroberta-base-v2` projects to **201.68 MiB** as
an embeddings-only int8 build, against CEIL-1's 32 MiB, and to 213 MiB of cold payload against CEIL-3's
64 MiB. It is **6.3× over** a ceiling fixed in `export/SIZE_BUDGET.md` before
anything was ever exported.

So `export/common.py` **still holds the incumbent encoder.** The scorer that
separates was selected and not adopted, and
`tests/test_scorer_ablation.py::TestASelectionIsNotAnAdoption` fails if the
repository ever adopts a scorer that breaches a ceiling, or quietly declines to
adopt one that does not.

That leaves R4-CEIL-001 as the binding defect again, with its terms changed: the
problem is no longer "no build clears the ceilings", it is "the only scorer
measured to work is six times the budget". The routes through, in the order they
will be measured, are a *small* NLI-supervised bi-encoder (the candidate list is
to be fixed before the next increment runs, not chosen from its results), and
plan.md R-4 — a local desktop app, which keeps the zero-egress claim and drops
the in-browser one rather than fudging it.

### §1b. There is no small NLI encoder that both separates and fits. R-4 fires.

**Measured:** `artifacts/size_feasible_scorer.json`, reproducible with
`python export/size_feasible_scorer.py`. Same held-out protocol, same 0.75
control floor and 0.70 usable threshold, imported not re-typed — and this time
**size is a selection criterion, not a footnote**. Increment 4 selected on
separation and reported shippability afterwards, which is how it produced a
selection it could not adopt. That is a tightening, so nothing that failed in
increment 4 can pass here because the rule changed.

The candidate list was closed in commit `c1727e1`, which contains the list and
no measurement. The size envelope was arithmetic, fixed before any of it ran:
the int8-embed build stores embedding parameters at one byte and everything else
at four, so at a 30522-entry vocab CEIL-1 buys about **three encoder layers at
hidden 384, or about eight at hidden 256**. That is why the candidates vary
depth rather than width.

| Scorer | H | L | Held-out AUC | Positive control | int8-embed | Fits? |
|---|---:|---:|---:|---:|---:|:---:|
| `xtremedistil-l6-h256-zeroshot-v1.1-all-33` | 256 | 6 | 0.504 | **0.628** | **25.91 MiB** | **yes** |
| `paraphrase-MiniLM-L3-v2` | 384 | 3 | 0.592 | 0.792 | 32.24 MiB | no |
| `cross-encoder/nli-deberta-v3-xsmall` (body) | 384 | 12 | 0.608 | **0.648** | 128.90 MiB | no |
| `deberta-v3-xsmall-zeroshot-v1.1-all-33` | 384 | 12 | 0.616 | **0.644** | 128.90 MiB | no |
| `all-MiniLM-L6-v2` *(baseline, recomputed)* | 384 | 6 | 0.504 | 0.860 | 52.55 MiB | no |
| `nli-distilroberta-base-v2` *(§1a, recomputed)* | 768 | 6 | **0.880** | 0.916 | 201.68 MiB | no |
| `MoritzLaurer/MiniLM-L6-mnli` *(unlicensed)* | 384 | 6 | 0.728 | **0.516** | 52.55 MiB | no |

**Nothing is selected.** The two rows that matter are the two ends of the gap:

* `MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33` **fits** — 25.91 MiB
  inside CEIL-1, 37.2 MiB of cold payload inside CEIL-3 — and separates at
  **0.504**, which is the incumbent's number, on a positive control of 0.628
  that is under the floor. It is entailment-supervised and it recovers nothing.
  Adopting it would be a size fix presented as a scorer fix.
* `sentence-transformers/nli-distilroberta-base-v2` separates at 0.880 and does
  not fit, unchanged from §1a.

`paraphrase-MiniLM-L3-v2` is the near miss in both directions and clears
neither: 32.24 MiB against a 32.00 MiB ceiling — **0.24 MiB over** — and 0.592
against a 0.70 threshold. Editing CEIL-1 by a quarter of a megabyte would have
"closed" R4-CEIL-001 on a model that is at chance anyway.
`tests/test_size_feasible_scorer.py::TestNoCeilingMoved` fails if CEIL-1 or
CEIL-3 ever moves.

Two things are recorded because they cut against us. First, one pre-registered
expectation was **wrong**: `paraphrase-MiniLM-L3-v2` was written down as
expected-to-fit before the run and missed by 0.24 MiB, and the expectation is in
the artifact rather than quietly corrected. Second, the second-highest held-out
number in the whole table (0.728) belongs to `MoritzLaurer/MiniLM-L6-mnli`, and
it is **not readable**: its positive control is 0.516, so the protocol cannot
attribute that separation to the dimension it is claimed for. That model is also
excluded from selection on licence — its repository and card declare none, and
`data/MANIFEST.md` requires a cleared licence for anything redistributed. Both
reasons are recorded, and it is worth being plain that the licence exclusion
cost nothing: the row was unreadable on its own control regardless.

**Conclusion: option B is closed at hidden ≤ 384.** R4-CEIL-001 does not close
by shrinking the scorer. plan.md **R-4** is therefore the answer, exactly as it
was written before any of this was measured — *a local desktop app, where the
zero-egress claim survives and the in-browser claim is dropped rather than
fudged*. `export/common.py` is unchanged and still holds the incumbent.

### What is therefore not claimed anywhere

- That a Ledger score reflects the user's mood, anxiety, sleep or activity.
- That a trajectory of these scores over time means anything.
- That any of this is diagnostic, screening, or clinically validated. There has
  been no clinical validation of any kind and none is planned inside this event.
- That the 0.880 in §1a is a property of anything that ships. It is a property of
  a 201.68 MiB scorer that this project has **not** adopted, measured on 50
  sentences written for it. It says the defect is fixable. It does not say it is
  fixed.

`tests/test_encoder_ablation.py` fails if the README starts making those claims
while the measurement stands.

## 2. The head is not trained

`head_is_trained` is `false` in `artifacts/torch/build_report.json` and the
version string is `anchor_v0`. Supervised fine-tuning is blocked on plan.md risk
R-1: no mental-health corpus with a clear permissive licence has been cleared,
and hackathon rule 2 requires clearing it rather than assuming it. Nothing in
this repository may be described as fine-tuned; a test enforces that.

## 3. There is no shippable in-browser build yet

`export/verify.py` exits **1** and `shippable_builds` is `[]`. The int8 build
(21.78 MiB) misses CEIL-5 at max score delta 0.0404 against 0.02; the
embeddings-only int8 build (52.04 MiB) meets CEIL-5 at 0.0094 but breaches CEIL-1
and CEIL-3. Details and the full ceiling table are in `export/SIZE_BUDGET.md`.

Increment 4 made this worse before it made it better, and the honest order to
read the two facts in is: the only scorer measured to separate held-out text
(§1a) is **201.68 MiB**, six times CEIL-1, so there is now a working scorer and a
shippable size and they are not the same artifact. Neither ceiling was moved to
close that gap.

Increment 5 closed the last route that would have kept the in-browser target:
of four entailment-supervised bi-encoders at hidden ≤ 384, the only one that
fits the ceilings separates at chance (§1b). **The in-browser claim is therefore
dropped rather than fudged, and plan.md R-4's desktop fallback is the target.**
The zero-egress claim is unaffected — it never depended on the browser, only on
there being no server component.

The narrower-encoder route recorded at the end of increment 2 was measured in
increment 3 and **not taken**: `google/bert_uncased_L-6_H-256_A-4` does project
to 25.91 MiB, comfortably inside CEIL-1, but §1 above shows it scores no better
than chance on held-out separation, exactly like every other candidate. Shrinking
the model was set aside because it would have optimised the size of a scorer that
does not yet discriminate.

## 4. What the safety layer does and does not cover

`ledger/safety/crisis_router.py` is deterministic and runs before any inference,
so no model output can suppress a routing decision. It is tested against
leetspeak, letter-spacing, separator insertion, homoglyphs and five
prompt-injection attempts (`tests/test_crisis_router.py`, 8/8; matrix 23/23).

It is a **term-and-pattern router**, not an intent classifier. It will not
recognise crisis expressed without any of the terms it knows, in a language it
does not cover, or through pure implication. It is a floor, not a safety net.

## 5. Quantization is runtime-dependent

Under WASM the int8 build's logits differ from native onnxruntime by **1.3e-02**,
while the fp32 and embeddings-only builds agree to ~5e-07
(`artifacts/wasm/bench_*.json`). A displayed score would depend on the user's
browser kernel path. This is an argument against the int8 build that is
independent of any size ceiling.

## 6. What *is* established

Stated so the list above is not read as "nothing works":

- **The attribution identity holds exactly.** `logit_k` equals the sum of its
  per-token attributions plus bias to a measured residual of 2.4e-07 natively and
  2.5e-07 under WASM, asserted on every build. Whatever the score is worth, the
  explanation is arithmetically the score and not a saliency map fitted to it.
- **In-browser inference runs.** `onnxruntime-web` WASM p95 is 124 ms for the
  int8 build and 238 ms for fp32 at 256 tokens, both inside CEIL-4's 500 ms.
- **The crisis router fires** on all 23 adversarial cases in the matrix,
  including one real evasion this suite caught (the dotless-i homoglyph
  `U+0131`, which survived NFKD normalisation).
- **Nothing leaves the device**, by construction: there is no server component.
  The packet-level demonstration of that claim is still owed.
