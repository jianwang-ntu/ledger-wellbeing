# docs/limitations.md — what Ledger does not establish

plan.md criterion **C2** ("Feasibility & Safety") names this file. It exists to
state, in the place a reader will actually look, what this tool has *not* shown.
Everything here is a measurement in this repository, not a caveat written to
sound careful.
> **Reading the numbers in this file.** Measured figures for the build that
> ships today are re-derived from their artifacts by
> `python export/check_published_numbers.py`. Figures from earlier build
> increments are kept as they were measured and tagged **SUPERSEDED**, because a
> limitations document that quietly rewrites its own history is worth less than
> one that shows it.

Last updated: build increment 9, 2026-08-25 (revision round 1).

---

## 1. The five-dimension score does not generalise. This is the headline.

**What ships, stated before the history so it cannot be missed.** The body in
`export/common.py` is `sentence-transformers/nli-distilroberta-base-v2`
(`artifacts/torch/build_report.json`), and under the held-out protocol below it
reaches **macro 0.880**, not the 0.504 this section opens with. 0.504 is the
**SUPERSEDED** incumbent `all-MiniLM-L6-v2` figure, kept because the argument
that forced the change is built on it. Read §1 as the record of *why the body
changed*, and §1a for the number that describes what ships. Two things do not
improve with the body: `activation` is still at 0.600, below the 0.70 usable
threshold fixed in increment 3, and the whole protocol is 50 anchor sentences
written for this project — not an evaluation set. The round-1 audit was right
that this section had gone stale at its own headline (F-05); this paragraph is
the correction, and none of the measurements below were altered.

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
fudged*. `export/common.py` held the incumbent when this was written. **SUPERSEDED at
increment 7**: it now pins `nli-distilroberta-base-v2` and `export/verify.py`
exits 0 on the desktop target. The conclusion above — that option B does not
close by shrinking the scorer, and that R-4's desktop fallback is the answer —
is unchanged; only the sentence about `common.py` was overtaken.

### What is therefore not claimed anywhere

- That a Ledger score reflects the user's mood, anxiety, sleep or activity.
- That a trajectory of these scores over time means anything.
- That any of this is diagnostic, screening, or clinically validated. There has
  been no clinical validation of any kind and none is planned inside this event.
- That the 0.880 in §1a means the instrument works. **Corrected at revision
  round 1 (F-05):** an earlier version of this bullet said the 0.880 was "a
  property of a 201.68 MiB scorer that this project has not adopted". That
  stopped being true at increment 7. `export/common.py` pins
  `nli-distilroberta-base-v2`, `artifacts/torch/build_report.json` confirms it,
  and `artifacts/verify_report.json` selects the `int8_embed` build made from
  it — so the 0.880 *is* a property of the shipped body. What it is still not:
  it is measured on 50 anchor sentences written for this project, under a
  leave-one-pair-out protocol, with `activation` at 0.600 below the usable
  threshold. It says the pole-separation defect is fixable and was fixed on
  this anchor set. It does not say the score means anything about a person.

`tests/test_encoder_ablation.py` fails if the README starts making those claims
while the measurement stands.

## 2. The head is not trained

`head_is_trained` is `false` in `artifacts/torch/build_report.json` and the
version string is `anchor_v0`. Supervised fine-tuning is blocked on plan.md risk
R-1: no mental-health corpus with a clear permissive licence has been cleared,
and hackathon rule 2 requires clearing it rather than assuming it. Nothing in
this repository may be described as fine-tuned; a test enforces that.

## 3. There is no shippable in-browser build, and there will not be

**Updated at increment 7.** `export/verify.py` now exits **0** on the *desktop*
target with `int8_embed` selected: CEIL-2 was met by re-serializing the same
tokenizer document compactly, 3,559,258 B -> 1,556,504 B, with `encode()` output
elementwise identical. That is the first shippable build in this project. It is a
statement about **size and fidelity only** and says nothing about whether the
score is any good; §1 is unchanged by it.

The **web** target stays lost, on four ceilings rather than two: model bytes,
cold payload, and 836.61 ms p95 in single-threaded WASM against 500 ms. Nothing
below is retracted.

The history that got here, kept because the numbers are the argument:

`export/verify.py` used to exit **1** with `shippable_builds` `[]`. The int8 build
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
The zero-egress claim is unaffected — it never depended on the browser. It was
written here as depending on "there being no server component", and **increment 9
withdrew that wording**: the interface now binds a loopback listener. What the
claim rests on is stated in §7.6 and measured there — the only bind is
`127.0.0.1`, and the port is refused on every non-loopback address on this host.

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

- **The attribution identity holds exactly — and the residual is a regression
  guard, not the evidence for it.** `logit_k` equals the sum of its per-token
  attributions plus bias. The architectural claim is correct: mean pooling
  followed by a linear head *is* a sum of per-token terms, so the explanation is
  arithmetically the score and not a saliency map fitted to it. But
  `LedgerScorer.forward` computes `logits = token_attr.sum(dim=1) + bias`, so in
  PyTorch the identity holds **by construction** and the residual we report
  cannot fail — round-1 audit finding F-06 was right that presenting it as
  verification invited the wrong reading. What the residual actually measures is
  whether the **export and quantization path** preserved an identity the source
  enforces: ONNX conversion, int8 quantization and the WASM runtime each get a
  chance to break it. Measured on the shipped `int8_embed` build:
  **3.2e-07** natively (`artifacts/verify_report.json`) and **1.4e-07** in the
  WASM runtime (`artifacts/wasm/bench_int8_embed.json`). An earlier revision of
  this bullet quoted 2.4e-07 / 2.5e-07, which match neither artifact.
- **What the identity is not: encoder-level faithfulness.** Token attributions
  are computed from **contextual** embeddings, so token *i*'s vector already
  depends on its neighbours. Exact additivity over the *pooling* layer therefore
  says the score decomposes exactly into per-token terms; it does not say those
  terms are what the encoder "used" for that token. Removing a word would change
  the vectors of the words around it. This caveat was undisclosed before
  revision round 1.
- **In-browser inference runs**, but not within the latency ceiling on the
  shipped body. The only WASM bench `export/verify.py` accepts is the shipped
  `int8_embed` build, at **836.61 ms** p95 — *over* CEIL-4's 500 ms, which is one
  of the four reasons the web target was dropped (§3). The two other bench files
  in `artifacts/wasm/` (124 ms for `int8_full`, 238 ms for `fp32`) are
  **SUPERSEDED**: `verify_report.json.wasm_benches_rejected_as_stale` rejects
  both because they predate `DEFECT-INC6-001` and carry no `model_sha256`, so
  they are not evidence for anything and are not quoted as such. An earlier
  revision of this file cited them as though they were current and inside the
  ceiling; that was wrong in both halves and is corrected here.
- **The crisis router fires** on all 23 adversarial cases in the matrix,
  including one real evasion this suite caught (the dotless-i homoglyph
  `U+0131`, which survived NFKD normalisation).
- **Nothing leaves the device**, and as of increment 8 that is measured rather
  than asserted: a full end-to-end exercise of the application under instrumented
  `socket` primitives records **zero socket calls of any kind**, loopback
  included (`artifacts/egress_audit.json`). See §7.1 for exactly how far that
  measurement reaches, which is not as far as a packet capture.
- **The application exists and runs.** Entry -> deterministic crisis route ->
  score -> per-span attribution -> encrypted local store -> report, driven from
  `python -m ledger.app.cli`. Span-level attribution still sums to the logit
  exactly: max residual **4.41e-07** against a 1e-4 rule, over 64 probe entries x
  5 dimensions x 2 granularities (`artifacts/span_additivity.json`).

## 7. What the application does and does not do (increments 8–9)

### 7.1 The zero-egress measurement is process-level, not packet-level

`export/egress_audit.py` wraps `socket.socket.connect`, `connect_ex`, `sendto`,
`sendmsg`, `socket.create_connection`, `getaddrinfo`, `gethostbyname` and
`gethostbyname_ex` before the application is imported, then drives it end to end.
It observes **what this Python process asked the kernel to do**.

It does not observe a subprocess, a native library that opens a socket without
going through the `socket` module, or a kernel-level send. A packet capture with
the interface down — which `plan.md`'s evidence plan asks for — is strictly
stronger and needs privileges this build environment does not have. **The weaker
measurement is not described as the stronger one**, and the stronger one remains
owed.

What the weaker one is genuinely sufficient for: the realistic failure mode here
was a Hugging Face library issuing a revision check on first use, which goes
through `getaddrinfo` and would have been caught. Since increment 8 the
application does not import `transformers` at all, so there is no library left in
it that knows how to fetch anything.

### 7.2 The encrypted store leaks metadata, and truncation is not detectable

Records are AES-256-GCM with a per-record nonce, keyed by scrypt
(n=2^15, r=8, p=1) from a passphrase the user holds and the file never contains.
Position and file identity are bound into the AAD, so a record cannot be
reordered, duplicated, or moved between stores without the tag failing.

Three things it does not hide or prevent, stated rather than left to be found:

- **Record count and approximate entry length are visible** to anyone holding the
  file. Length-prefixed records make that unavoidable without padding, and
  padding was not implemented.
- **Truncation of trailing records is detectable only as absence.** Someone who
  can write to the file can delete the most recent entries; the remaining ones
  still authenticate. A running counter in each record would fix this and is not
  built.
- **The header is plaintext**, deliberately: it holds the KDF parameters, which
  cannot be hidden from an attacker without also hiding them from the owner.

### 7.3 Wipe is as good as userspace gets, which is not as good as it sounds

`Journal.wipe()` overwrites the file with random bytes, then with zeros, then
unlinks it. On a copy-on-write, journalled, or wear-levelled filesystem — which
is to say on most SSDs — the superseded blocks may survive both passes, and **no
userspace program can promise otherwise**. Full-disk encryption is the actual
answer to that threat and is outside what this project can provide.

Wipe deliberately does **not** require the passphrase. Someone who needs their
journal gone should not have to remember how to open it first.

### 7.4 The interface is measured for conformance, not for what a screen-reader user experiences

Increment 9 built the visual interface (`python -m ledger.ui`) and measured it in
a real Chromium: axe-core at **zero violations and zero incomplete** across all
six views on `wcag2a, wcag2aa, wcag21a, wcag21aa`; the whole primary flow driven
with **zero pointer events**; a focus indicator verified **optically** on every
focusable element; every element reporting `0s` motion under
`prefers-reduced-motion`; the flow completing again under `forced-colors: active`.
`artifacts/a11y_report.json` has all of it.

What that is **not**: a screen-reader traversal. No assistive technology is driven
anywhere in this repository, and none is claimed. `plan.md`'s evidence plan asks
for "an axe run *plus* a recorded screen-reader traversal"; the first half exists
and the second does not, so **C6 is evidenced in part, not in full**. That half
needs a screen reader and a human listening to it, and it is recorded as
NEEDS_HUMAN rather than approximated.

Two further limits worth naming:

- **Conformance is a floor, not a design review.** Zero violations means no rule
  was broken. It does not mean the interface is good, and no automated check can.
- **axe disagreed with the computed styles under emulated forced colours.** It
  reported the three primary buttons as a 1:1 `incomplete`, while
  `getComputedStyle` on the same elements returns black on white. An independent
  contrast pass built into the harness — same WCAG formula, live computed styles,
  every text-bearing element — measures a **minimum of 13.99:1** in that mode and
  **7.38:1** in the default one. Both numbers are in the artifact; neither is
  dropped in favour of the flattering one.

### 7.5 The application shows numbers, and numbers persuade

Four of five dimensions clear the 0.70 held-out floor and one does not. Both the
per-entry output and the report label `activation` `[NOT ESTABLISHED]` every
time it appears, with its 0.600 and the threshold, and say the evaluation was 25
withheld anchor-sentence pairs per dimension written for this repository — not
clinical data, not real journal entries, not an external benchmark. That labelling
is a guard (`tests/test_report.py`), not a convention, and a mutation that blanks
it in either section fails the suite.

Increment 9 made this sharper, not softer. The interface renders the same numbers
in a form that is far more persuasive than a JSON file, so it carries the same
labelling: every dimension card below the floor shows a **NOT ESTABLISHED** badge,
its held-out AUC, and the sentence from `ledger/app/evidence.py` saying what that
means; the entry view names the evaluation basis above the cards; and the
banned-vocabulary check now runs over the **rendered page text**, not the source.
That last check caught real copy on the write view during increment 9 — the
wording was changed rather than the check's exemption widened.

The residual risk stands anyway: a person reading a sparkline of their own moods
will over-read it. The chart is fixed to a 0..1 scale rather than autoscaled for
exactly this reason, and it is not enough.
### 7.6 The interface binds a loopback listener, and that is a change of claim

Until increment 9 the application opened no socket at all, so "no server
component" was true — by accident of there being no interface. There is now an
HTTP server on `127.0.0.1` and an ephemeral port.

`export/INCREMENT_8_PREREGISTRATION.md` fixed in advance what would happen if this
came to pass: the sentence is **withdrawn**, not reinterpreted to mean "no
*remote* server". What replaces it is measured:

- The only address the process binds is `127.0.0.1`
  (`artifacts/egress_audit_ui.json:binds`), and `socket.socket.bind` is
  instrumented so a wildcard bind would be recorded rather than argued about.
- The port is **refused** on all nine non-loopback IPv4 addresses on this host —
  `ECONNREFUSED` on each (`…:ui_exercise.reachability`). The probe reads the
  interface list from the kernel by `ioctl`, not from a name lookup.
- Driving the whole interface over loopback produced **39 socket calls, all
  loopback, zero DNS resolutions**.
- In the browser, **every one of the 30 requests the page issued** went to the
  loopback origin, and the served Content-Security-Policy pins `default-src` and
  `connect-src` to `'self'`.

What this costs, stated plainly: the passphrase is posted to `127.0.0.1` in a
request body. It does not leave the host and it is never written to disk, to a
log line, or to a URL — but it does cross a socket, which it did not before.

**The argument that used to sit here was wrong, and it was wrong in our own
favour.** It said the listener "does not widen the trust boundary", because "any
process running as this user could already read the journal file and the process
memory". Round-1 audit finding **F-04** falsified both halves:

* `journal.enc` is *ciphertext*. A process that reads the file without the
  passphrase gets bytes it cannot decrypt. That is the whole point of the store.
* This host runs `yama ptrace_scope=1` (`/proc/sys/kernel/yama/ptrace_scope`),
  which blocks same-user process-memory reads outside a descendant. So the
  derived key in the server's memory was not already readable either.

And the listener really did widen the boundary. The auditor unlocked the journal
in one browser, then acted as an unrelated local process: `GET /` (which cannot
require a secret — the browser has to load it), scraped the per-run token out of
the served HTML, and read the full decrypted journal with it, never supplying the
passphrase. The store is mode `0600` precisely to exclude other local users, and
the interface was undoing that for as long as anything was unlocked.

**What changed at revision round 1.** Unlocking now mints a second secret,
returned *only* in the body of the successful `/api/unlock` response and written
nowhere else — not into the page, not into `localStorage`, not into a cookie.
Every endpoint that can reach journal plaintext (`/api/entries`, `/api/report`,
`/api/entry`) requires it in `X-Ledger-Session`. A local client that did not
supply the passphrase has no way to obtain it. The derived key is also dropped
after `IDLE_LOCK_SECONDS` (default 900) without an authenticated request, which
answers the other half of the finding — the server used never to re-challenge.

Measured, with a negative control, in
`audit/revision1/probe_session_unlock.rerun.json`: against the **pre-fix** server
at `git HEAD` the attack returns HTTP 200 and the planted entry text; against the
revised server the same attack returns **401** on `/api/entries` and
`/api/report` with no entry text in either body. Seven tests in
`tests/test_ui.py::TestUnlockingOneClientDoesNotUnlockTheMachine` pin it,
including the negative control that a client which *did* supply the passphrase
still works.

**What is still true and is not claimed away.** `GET /` remains unauthenticated
and the per-run token remains scrapeable by any local process — that is asserted
by a test rather than defended, because the browser must be able to load the
page. The token was demoted from authenticator to same-origin guard; it is not
the thing protecting the journal. And the transport is still loopback TCP with no
UID restriction, so this is an application-layer fix, not the OS-enforced one a
unix socket with filesystem permissions would give. That remains the better
design and is not built.

The `Host` header is pinned and every `/api` call needs the per-run token, so a
page on the open internet cannot reach the interface by pointing a name at
127.0.0.1. Both are guarded in `tests/test_ui.py` and both are mutation-tested.

