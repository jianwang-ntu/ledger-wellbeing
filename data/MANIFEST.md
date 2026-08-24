# data/MANIFEST.md — everything third-party, and its licence

Hackathon rule 2: *"You must only use resources and assets you have access to and
must attribute appropriately."* This file is that attribution. It is written to
be complete: if something is used and is not ours, it is listed here.

## Model weights

| Asset | `sentence-transformers/all-MiniLM-L6-v2` |
|---|---|
| Pinned revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (pinned by commit, not by tag, in `export/common.py`) |
| Licence | **Apache-2.0** — read from the Hugging Face model API, `cardData.license` and the `license:apache-2.0` tag, 2026-08-24 |
| Gated | No |
| Parameters | 22,713,216 (6 layers, hidden 384, vocab 30,522) |
| Used for | The encoder. Its weights are frozen — no gradient step is taken on them anywhere in this repository. |
| Upstream | Reimers & Gurevych, *Sentence-BERT* (EMNLP-IJCNLP 2019); the checkpoint is the sentence-transformers community distillation of MiniLM. |

### Downloaded and measured, not shipped

Two ablations download checkpoints purely to measure them. None of these is
shipped and none appears in any exported artifact — `export/common.py` still
holds the incumbent above — but rule 2 requires attribution for resources
*used*, not only for resources shipped, so every one of them is listed.

`export/encoder_ablation.py` (build increment 3) downloads three checkpoints to
measure whether a narrower encoder could clear CEIL-1.

| Asset | Pinned revision | Licence | Gated |
|---|---|---|---|
| `google/bert_uncased_L-8_H-256_A-4` | `fff21c203abcc9365418f2e46bb6801a2b98e3da` | **Apache-2.0** | No |
| `google/bert_uncased_L-6_H-256_A-4` | `67ada51801f40684c01ca3f20c97a35fa7a67d36` | **Apache-2.0** | No |
| `google/bert_uncased_L-4_H-256_A-4` | `387825ce42dbb39b87911cdf8e383ee3b25184f8` | **Apache-2.0** | No |

Upstream: Turc, Chang, Lee & Toutanova, *Well-Read Students Learn Better* (2019).

`export/scorer_ablation.py` (build increment 4) downloads four further
checkpoints to measure whether any *polarity-aware* scorer separates held-out
anchors where the mean-pooled anchor head does not.

| Asset | Pinned revision | Licence | Gated | Measured as |
|---|---|---|---|---|
| `distilbert-base-uncased-finetuned-sst-2-english` | `714eb0fa89d2f80546fda750413ed43d93601a13` | **Apache-2.0** | No | sentiment-tuned encoder body (classification head discarded), and separately as a global-polarity diagnostic |
| `sentence-transformers/nli-distilroberta-base-v2` | `cc35a0bfb6251228a6fb8c797bca5fef0ece3c1d` | **Apache-2.0** | No | NLI-supervised bi-encoder, mean-pooled |
| `cross-encoder/nli-distilroberta-base` | `b14d131f9d32668a5e6a982729b57ff6ed5dfcbd` | **Apache-2.0** | No | zero-shot entailment cross-encoder |
| `cross-encoder/nli-MiniLM2-L6-H768` | `b95119ce93d3e065de6214e38cd4a97b0f2f2c6d` | **Apache-2.0** | No | zero-shot entailment cross-encoder |

Upstream for the two cross-encoders and the NLI bi-encoder: Reimers & Gurevych,
*Sentence-BERT* (2019) and the SBERT cross-encoder NLI models trained on SNLI +
MultiNLI; for the SST-2 checkpoint, Sanh et al., *DistilBERT* (2019) fine-tuned
on SST-2 (Socher et al., 2013).

`export/size_feasible_scorer.py` (build increment 5) downloads five further
checkpoints to ask whether any entailment-supervised bi-encoder at hidden <= 384
both separates held-out anchors and fits the ceilings in
`export/SIZE_BUDGET.md`. The candidate list was committed before the run.

| Asset | Pinned revision | Licence | Gated | Measured as |
|---|---|---|---|---|
| `MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33` | `c07f66d9cbf781191bee66edfe8ad7856f045781` | **MIT** | No | entailment-supervised body, mean-pooled; the one candidate inside the size envelope |
| `sentence-transformers/paraphrase-MiniLM-L3-v2` | `4ca70771034acceecb2e72475f72050fcdde4ddc` | **Apache-2.0** | No | 3-layer bi-encoder with AllNLI in its training mixture |
| `cross-encoder/nli-deberta-v3-xsmall` | `a150876415327c80daeff35ca6f68f5ed8cf5c24` | **Apache-2.0** | No | NLI body, sequence-classification head discarded, mean-pooled |
| `MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33` | `262ae02f29173eec1c250f90804dc7edc677dcff` | **MIT** | No | same body under a broader zero-shot mixture |
| `MoritzLaurer/MiniLM-L6-mnli` | `6e0917f1a395b7a6c0f054a56b91c45d8e3af92f` | **NONE DECLARED** | No | MNLI-supervised at the incumbent's exact shape — **measured, and excluded from selection on licence** |

**`MoritzLaurer/MiniLM-L6-mnli` declares no licence** in `cardData.license`, in
the repository tags, or anywhere in its model card (all three checked via the
Hugging Face model API on 2026-08-24). An undeclared licence is not a permissive
one. It was measured because it isolates supervision from capacity — it is the
incumbent's exact shape with MNLI training — and it was excluded from selection
before its number was read, so no weight of it enters any artifact here.
`tests/test_size_feasible_scorer.py::TestLicenceIsAGateNotAPreference` fails if a
model with a null licence is ever made a selection candidate.

Upstream: for the MiniLM checkpoints, Wang et al., *MiniLM* (2020) and Reimers &
Gurevych, *Sentence-BERT* (2019); for the DeBERTa-v3 checkpoints, He, Gao & Chen,
*DeBERTaV3* (2021); for the xtremedistil body, Mukherjee & Awadallah,
*XtremeDistil* (2020). The zero-shot fine-tunes are Laurer et al., *Less Annotating,
More Classifying* (2023).

All licences on this page were read from the Hugging Face model API
(`cardData.license` and the `license:apache-2.0` tag) on 2026-08-24, not from a
README badge. `tests/test_encoder_ablation.py`, `tests/test_scorer_ablation.py` and
`tests/test_size_feasible_scorer.py` fail if a model any ablation downloads is
missing from this file, or is listed without its pinned revision.

## Runtime

| Asset | `onnxruntime-web` 1.23.0 |
|---|---|
| Licence | MIT (Microsoft) |
| Used for | In-browser inference, and the WASM latency measurement in `web/bench_wasm.mjs`. |

Build-time only, not shipped to a user: `torch`, `transformers`, `onnx`,
`onnxruntime`, `onnxscript`, `numpy` — all BSD/MIT/Apache-2.0.

## Text

**There is no third-party text corpus in this project.** Everything the model is
calibrated against is original text written for it:

- `ledger/model/dimensions.py` — 50 anchor sentences, 10 per dimension, written
  for this project.
- `export/common.py:probe_entries` — the 64 probe entries are a seeded
  recombination of those same anchor sentences. Nothing external is drawn in.

This is deliberate and it is also a limitation, recorded here rather than in a
footnote: plan.md risk **R-1** rules out the Reddit-derived mental-health corpora
whose terms are restrictive or unclear, and no permissively-licensed substitute
has been cleared yet. Until one is, **the head is not trained** — see
`export/SIZE_BUDGET.md` and `artifacts/torch/build_report.json`, where
`head_is_trained` is `false`. No artifact in this repository may be described as
fine-tuned.

## Clinical instruments

None. No PHQ-9, GAD-7 or other validated instrument is reproduced, scored or
approximated (plan.md risk R-2). The five dimensions in
`ledger/model/dimensions.py` are named for language, not for conditions, for
exactly this reason.
