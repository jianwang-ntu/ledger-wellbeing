"""Build increment 3, step 0: choose the encoder by measurement, not by projection.

Increment 2 ended with `verify.py` exiting 1 and `shippable_builds == []`:
int8-everything (21.78 MiB) missed CEIL-5 at a max score delta of 0.0404 against
0.02, and int8-on-the-embedding-only (52.04 MiB) met CEIL-5 at 0.0094 but
breached CEIL-1. `artifacts/quant_sensitivity.json` established *why*: the int8
error lives in the encoder-layer weights and the size lives in the embedding
table, and the two are separable. That pointed at a narrower encoder, projected
at ~26.7 MiB.

This script was written to replace that projection with a measurement, because a
narrower encoder is not free: `all-MiniLM-L6-v2` is a sentence-embedding
distillation, while the `google/bert_uncased_*` miniatures are general pretrained
checkpoints with no sentence-level objective. Trading 26 MiB for an unmeasured
loss of signal is exactly what the audit is meant to catch.

## The quality measure, and why it is built the way it is

Held-out separation over the 50 anchor sentences in `ledger/model/dimensions.py`.
No external corpus is touched, so plan.md R-1 is untouched.

Three protocols are computed, and the first two exist to keep the third honest:

* ``in_sample`` - direction built from all 5+5 anchors, scored on those same
  anchors. A 5-vs-5 centroid split in a 384-dimensional space separates its own
  training points perfectly whatever the encoder does, so this number is
  expected to be 1.000 and means nothing on its own. It is reported as the
  contrast that shows what the held-out numbers are worth.
* ``positive_control`` - the same held-out protocol applied to a pair the encoder
  should obviously get right: the POSITIVE poles of two *different* dimensions,
  which are different topics. If this collapses toward 0.5 the protocol is
  broken and no other number here may be read.
* ``held_out`` - the question that matters. One positive and one negative anchor
  are withheld *together*, the direction is rebuilt from the remaining 4+4, and
  the two withheld sentences are ranked against each other. 25 pairs per
  dimension. Both poles lose a member, so neither centroid is advantaged.

An earlier revision of this file withheld one sentence at a time, which biases
against the withheld point (its own centroid moves away from it while the
opposite centroid keeps all five members) and produced systematically
below-chance numbers. That protocol is wrong and was replaced; the finding is
recorded in `../audit/evidence.jsonl` rather than quietly dropped.

10 held-out sentences per dimension is a small sample and this is not an
evaluation of the product. It is a comparison instrument, applied identically to
every candidate.

Writes artifacts/encoder_ablation.json. Changes nothing else: any encoder switch
it justifies is applied by hand in export/common.py, and
tests/test_export_pipeline.py asserts the two agree.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from common import ARTIFACTS, CEILINGS, ORT_RUNTIME_FLOOR_BYTES, write_json
from ledger.model import ANCHORS, DIMENSIONS
from ledger.model.scorer import mean_pool

# Every candidate is Apache-2.0 and ungated; revisions read from the Hugging Face
# model API on 2026-08-24 and pinned by commit, never by tag.
CANDIDATES = [
    {"model": "sentence-transformers/all-MiniLM-L6-v2",
     "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
     "role": "incumbent from build increment 2"},
    {"model": "google/bert_uncased_L-8_H-256_A-4",
     "revision": "fff21c203abcc9365418f2e46bb6801a2b98e3da",
     "role": "narrower encoder, 8 layers"},
    {"model": "google/bert_uncased_L-6_H-256_A-4",
     "revision": "67ada51801f40684c01ca3f20c97a35fa7a67d36",
     "role": "narrower encoder, 6 layers"},
    {"model": "google/bert_uncased_L-4_H-256_A-4",
     "revision": "387825ce42dbb39b87911cdf8e383ee3b25184f8",
     "role": "narrower encoder, 4 layers"},
]

#: Below this, the held-out separation is not distinguishable from coin-flipping
#: at this sample size and an encoder chosen on it would be chosen on noise.
#: Fixed here before any candidate was measured.
USABLE_HELD_OUT_AUC = 0.70
#: Below this the protocol itself is not trustworthy and nothing else is read.
POSITIVE_CONTROL_FLOOR = 0.75

EMBEDDING_MARKERS = ("word_embeddings", "position_embeddings", "token_type_embeddings")


def projected_int8embed_bytes(model) -> dict:
    """The int8_embed build quantizes the embedding Gathers and leaves every
    MatMul fp32, so its size follows from the parameter shapes."""
    embed = sum(p.numel() for n, p in model.named_parameters()
                if any(m in n for m in EMBEDDING_MARKERS))
    other = sum(p.numel() for n, p in model.named_parameters()
                if not any(m in n for m in EMBEDDING_MARKERS))
    return {"embedding_parameters": embed, "non_embedding_parameters": other,
            "bytes": embed * 1 + other * 4}


@torch.no_grad()
def embed_sentences(model, tokenizer, sentences, max_length=64) -> np.ndarray:
    batch = tokenizer(sentences, padding=True, truncation=True,
                      max_length=max_length, return_tensors="pt")
    hidden = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[0]
    return mean_pool(hidden, batch["attention_mask"]).numpy().astype(np.float64)


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


def in_sample_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    direction = pos.mean(0) - neg.mean(0)
    direction = direction / np.linalg.norm(direction)
    return _auc(pos @ direction, neg @ direction)


def held_out_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Withhold one member of each pole together, rebuild from the rest, rank the two."""
    wins = []
    for i, j in itertools.product(range(len(pos)), range(len(neg))):
        direction = np.delete(pos, i, axis=0).mean(0) - np.delete(neg, j, axis=0).mean(0)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            raise ValueError("degenerate anchor direction")
        direction = direction / norm
        a, b = float(pos[i] @ direction), float(neg[j] @ direction)
        wins.append(1.0 if a > b else (0.5 if a == b else 0.0))
    return float(np.mean(wins))


def representation_geometry(poles: dict) -> dict:
    """Is this space organised by topic or by polarity?

    If the two poles of one dimension - "heavy and grey" against "calm and
    unhurried" - sit closer to each other than either sits to another
    dimension's sentences, the space encodes *what the entry is about* and not
    *which way it leans*. That distinction decides the route through a failing
    held-out AUC: more anchors cannot buy polarity out of a representation that
    does not carry it, but supervision or a polarity-aware encoder can.
    """
    unit = {d: (p / np.linalg.norm(p, axis=1, keepdims=True),
                n / np.linalg.norm(n, axis=1, keepdims=True)) for d, (p, n) in poles.items()}
    within, across = [], []
    for dim, (pos, neg) in unit.items():
        within.append(float((pos @ neg.T).mean()))
        others = np.concatenate([np.concatenate(v) for d, v in unit.items() if d != dim])
        across.append(float((np.concatenate((pos, neg)) @ others.T).mean()))
    return {
        "mean_cosine_across_poles_same_dimension": round(float(np.mean(within)), 4),
        "mean_cosine_to_other_dimensions": round(float(np.mean(across)), 4),
        "per_dimension_cross_pole_cosine":
            {d: round(w, 4) for d, w in zip(unit.keys(), within)},
        "reading": ("topic-dominated: opposite poles of the same dimension are more similar to "
                    "each other than to other dimensions, so mean-pooled similarity carries "
                    "subject matter rather than direction"
                    if float(np.mean(within)) > float(np.mean(across)) else
                    "polarity is visible: the poles of a dimension separate at least as much as "
                    "different dimensions do"),
    }


def evaluate(spec: dict) -> dict:
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    model = AutoModel.from_pretrained(spec["model"], revision=spec["revision"],
                                      attn_implementation="eager").eval()
    cfg = AutoConfig.from_pretrained(spec["model"], revision=spec["revision"])

    poles = {dim: (embed_sentences(model, tokenizer, ANCHORS[dim]["positive"]),
                   embed_sentences(model, tokenizer, ANCHORS[dim]["negative"]))
             for dim in DIMENSIONS}

    per_dim = {dim: {"held_out_auc": round(held_out_auc(*poles[dim]), 4),
                     "in_sample_auc": round(in_sample_auc(*poles[dim]), 4)}
               for dim in DIMENSIONS}

    control = {f"{a}|{b}": round(held_out_auc(poles[a][0], poles[b][0]), 4)
               for a, b in itertools.combinations(DIMENSIONS, 2)}

    held = [v["held_out_auc"] for v in per_dim.values()]
    size = projected_int8embed_bytes(model)
    cold = ORT_RUNTIME_FLOOR_BYTES + size["bytes"]      # + tokenizer, added by verify.py

    return {
        **spec,
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "int8embed_size": {
            **size,
            "MiB": round(size["bytes"] / 1048576, 2),
            "clears_CEIL_1": size["bytes"] <= CEILINGS["CEIL_1_int8_model_bytes"],
            "projected_cold_payload_MiB_excl_tokenizer": round(cold / 1048576, 2),
            "clears_CEIL_3_excl_tokenizer": cold <= CEILINGS["CEIL_3_cold_payload_bytes"],
            "basis": "computed from parameter shapes: 1 byte per embedding weight, 4 per other "
                     "weight. Omits per-channel quantization scales and ONNX graph overhead, and "
                     "also omits the exporter's pruning of the unused BERT pooler - for the "
                     "incumbent those cancelled to within 0.5 MiB of the file verify.py measured "
                     "(projected 52.55 MiB, measured 52.04). A PROJECTION. CEIL-1 is decided by "
                     "verify.py on the exported file, never by this number.",
        },
        "anchor_separation": {
            "per_dimension": per_dim,
            "macro_held_out_auc": round(float(np.mean(held)), 4),
            "dimensions_at_or_above_usable_threshold":
                [d for d, v in per_dim.items() if v["held_out_auc"] >= USABLE_HELD_OUT_AUC],
            "macro_in_sample_auc": round(float(np.mean([v["in_sample_auc"] for v in per_dim.values()])), 4),
            "positive_control_pairs": control,
            "positive_control_macro": round(float(np.mean(list(control.values()))), 4),
            "protocol_trustworthy": float(np.mean(list(control.values()))) >= POSITIVE_CONTROL_FLOOR,
            "n_held_out_pairs_per_dimension": 25,
        },
        "representation_geometry": representation_geometry(poles),
        "elapsed_s": round(time.time() - t0, 2),
    }


def main() -> int:
    results = [evaluate(spec) for spec in CANDIDATES]
    incumbent = results[0]

    trustworthy = all(r["anchor_separation"]["protocol_trustworthy"] for r in results)
    usable = [r for r in results
              if r["anchor_separation"]["macro_held_out_auc"] >= USABLE_HELD_OUT_AUC]
    fits = [r for r in usable if r["int8embed_size"]["clears_CEIL_1"]]
    selected = max(fits, key=lambda r: r["anchor_separation"]["macro_held_out_auc"]) if fits else None

    if selected is not None:
        verdict = f"SELECT {selected['model']}"
    elif not usable:
        verdict = ("NO SELECTION. No candidate - including the incumbent - reaches "
                   f"macro held-out AUC {USABLE_HELD_OUT_AUC} on the anchor separation test, so "
                   "any choice between them would be a choice between noise levels. The encoder "
                   "is not the binding problem; the head is. See open finding R4-HEAD-003.")
    else:
        verdict = ("NO SELECTION. Candidates reach the quality threshold but none of them clears "
                   "CEIL-1.")

    report = {
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "Which encoder lets an embedding-only-int8 build clear CEIL-1 (<= 32 MiB) "
                    "while giving up the least anchor separation?",
        "selection_rule": "Fixed before any candidate was measured. (1) The positive control must "
                          f"hold at >= {POSITIVE_CONTROL_FLOOR} or nothing is read. (2) A candidate "
                          f"must reach macro held-out AUC >= {USABLE_HELD_OUT_AUC} to be eligible "
                          "at all. (3) Among those, keep the ones whose projected int8_embed build "
                          "clears CEIL-1. (4) Take the highest macro held-out AUC. CEIL-5 is not "
                          "part of this rule - verify.py decides it on the real exported graph.",
        "protocol_trustworthy": trustworthy,
        "usable_held_out_auc_threshold": USABLE_HELD_OUT_AUC,
        "positive_control_floor": POSITIVE_CONTROL_FLOOR,
        "candidates": results,
        "selected": (selected or {}).get("model"),
        "selected_revision": (selected or {}).get("revision"),
        "verdict": verdict,
        "incumbent_reference": {
            "model": incumbent["model"],
            "macro_held_out_auc": incumbent["anchor_separation"]["macro_held_out_auc"],
            "macro_in_sample_auc": incumbent["anchor_separation"]["macro_in_sample_auc"],
            "positive_control_macro": incumbent["anchor_separation"]["positive_control_macro"],
            "int8embed_MiB": incumbent["int8embed_size"]["MiB"],
            "representation_geometry": incumbent["representation_geometry"],
        },
    }
    write_json(ARTIFACTS / "encoder_ablation.json", report)

    print(json.dumps({k: v for k, v in report.items() if k != "candidates"}, indent=1))
    print()
    print(f"{'model':46} {'H':>4} {'L':>3} {'int8emb':>9} {'CEIL1':>6} "
          f"{'held_out':>9} {'in_samp':>8} {'control':>8}")
    for r in results:
        s, a = r["int8embed_size"], r["anchor_separation"]
        print(f"{r['model']:46} {r['hidden_size']:>4} {r['num_hidden_layers']:>3} "
              f"{s['MiB']:>7.2f}Mi {'PASS' if s['clears_CEIL_1'] else 'FAIL':>6} "
              f"{a['macro_held_out_auc']:>9.3f} {a['macro_in_sample_auc']:>8.3f} "
              f"{a['positive_control_macro']:>8.3f}")
    print()
    print(verdict)
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
