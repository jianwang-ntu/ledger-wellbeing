# Ledger

A client-side longitudinal well-being instrument, built for
**Hack for Humanity | Summer 2026** (focus: Mental health).

> **Status: in progress.** This README records only what is built and has been
> run. Claims about parts that do not exist yet are not made here. Every claim
> below is backed by a checked-in artifact under `../audit/`.

## What it is meant to be

An instrument that makes a person's own between-visits experience legible to the
clinician actually treating them — not an AI that acts as a therapist. Inference
runs in the user's browser over their own writing; the output is per-span
attribution over their words and a trajectory over time, not generated advice.

## What is built and verified so far

| Component | State | Runtime evidence |
|---|---|---|
| `ledger/safety/` — deterministic crisis router | **built, tested** | `../audit/runs/safety_tests_20260824T1350Z.txt` (8/8), `../audit/runs/crisis_matrix_20260824T1350Z.json` (23/23) |
| `ledger/model/` — encoder + additive attribution head | **built** | `artifacts/verify_report.json` |
| `export/` — ONNX export, int8 quantization, verification | **built, and currently failing its own ceiling** | `artifacts/verify_report.json`, `artifacts/quant_sensitivity.json` |
| In-browser inference (`onnxruntime-web`, WASM) | **runs**; no shippable build yet | `artifacts/wasm/bench_*.json` |
| Head training | **not started** — blocked on a permissively-licensed corpus | `data/MANIFEST.md` |
| Held-out separation of the five dimensions | **measured, and at chance on 4 of 5** | `artifacts/encoder_ablation.json`, `docs/limitations.md` §1 |
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
residual: **2.4e-07 in the browser runtime**, 2.4e-07 natively
(`artifacts/wasm/bench_*.json`, `artifacts/verify_report.json`), asserted on
every build by `tests/test_export_pipeline.py`.

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
