# docs/limitations.md — what Ledger does not establish

plan.md criterion **C2** ("Feasibility & Safety") names this file. It exists to
state, in the place a reader will actually look, what this tool has *not* shown.
Everything here is a measurement in this repository, not a caveat written to
sound careful.

Last updated: build increment 3, 2026-08-24.

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
controls of 0.82–0.84. The remaining routes are supervision (blocked on plan.md
R-1, corpus licence), a polarity-aware encoder, or reducing the product to the
one dimension that is measured to work.

### What is therefore not claimed anywhere

- That a Ledger score reflects the user's mood, anxiety, sleep or activity.
- That a trajectory of these scores over time means anything.
- That any of this is diagnostic, screening, or clinically validated. There has
  been no clinical validation of any kind and none is planned inside this event.

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
