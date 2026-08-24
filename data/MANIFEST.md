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
