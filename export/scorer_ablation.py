"""Build increment 4: decide R4-HEAD-003 by measuring option B, not by arguing it.

Increment 3 established the binding defect. The zero-shot anchor head separates
*held-out* text at chance — macro AUC 0.504 against an in-sample 1.000, with a
positive control at 0.824-0.860 proving the protocol works — and every
alternative frozen encoder measured was the same or worse. The geometry said why:
the opposite poles of one dimension sit at mean cosine 0.324 to each other
against 0.243 to other dimensions, so mean-pooled similarity carries *what the
entry is about* and not *which way it leans*.

That ruled out two routes (more anchors, a different frozen encoder of the same
family) and left four, of which exactly one was unmeasured:

  A. Train the head            BLOCKED on plan.md R-1 (no cleared corpus)
  B. Polarity-aware scoring    UNMEASURED  <-- this file
  C. Ship only what generalises (low_mood, held-out 0.960)
  D. Drop the score entirely

This file measures B. It changes nothing else: any scorer it selects is applied
by hand afterwards, exactly as encoder_ablation.py's selection would have been.

## What "polarity-aware" is allowed to mean here

Three families, because "use an NLI or sentiment-tuned encoder" is one idea and
there are three genuinely different ways to act on it, with different costs to
criterion C4 (the exactly-additive attribution head):

* **A different linear readout of the same representation.** Free, and it is the
  control that decides whether increment 3's diagnosis was right. If polarity is
  simply absent from the space, no linear readout can recover it; if the
  centroid-difference direction was merely a *bad* readout, a covariance-aware
  one will find it. Additivity survives untouched: any fixed row over the pooled
  vector is still a sum of per-token terms.
* **A polarity-tuned encoder body, still mean-pooled, still a linear head.** A
  sentiment- or NLI-trained body has seen a supervision signal about direction.
  Additivity survives: the head is unchanged, only the body differs.
* **A cross-encoder entailment scorer.** Genuinely polarity-aware by
  construction, and it **breaks C4**: the score is a nonlinear function of a
  (text, hypothesis) pair, not a linear row over a pooled vector, so the
  attribution identity `sum(token_attr) == logit` no longer holds. It is measured
  anyway, because the cost of a route is not a reason to leave it unmeasured. If
  it is the only thing that works, the honest response is to drop the C4 claim,
  not to bury the number.

Every variant carries `additivity_preserved`, and the selection rule below is
explicit about what that costs.

## The measure, and why it is the same one

Identical protocol to increment 3, imported from encoder_ablation.py rather than
re-typed, so it cannot drift: withhold one positive and one negative anchor
*together*, rebuild the direction from the remaining 4+4, rank the two withheld
against each other, 25 pairs per dimension. Same USABLE_HELD_OUT_AUC (0.70) and
POSITIVE_CONTROL_FLOOR (0.75), fixed in increment 3 before any of this was seen.

For the cross-encoders there is nothing to rebuild — they fit no parameters — so
`held_out` and `in_sample` collapse to the same thing and `macro_in_sample_auc`
is reported as null rather than as a flattering 1.000. The AUC itself is over the
same 25 (positive_i, negative_j) comparisons per dimension, so the column is
comparable across every row of the table. This is recorded because it is the one
place the protocols are not literally identical.

The hypothesis strings for the cross-encoders are written once, below, before any
of them was run, and are mechanical restatements of the dimension labels in
`ledger/model/dimensions.py`. Tuning them against the AUC would turn a
measurement into a search, so they are frozen in this file and any later edit is
visible in the diff.

Writes artifacts/scorer_ablation.json. Exit 0 if a scorer is selected, 2 if none
is — in which case option B is closed and the fallback is C or D.
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from common import ARTIFACTS, CEILINGS, ORT_RUNTIME_FLOOR_BYTES, write_json
from encoder_ablation import (
    POSITIVE_CONTROL_FLOOR,
    USABLE_HELD_OUT_AUC,
    _auc,
    embed_sentences,
    held_out_auc,
    in_sample_auc,
    projected_int8embed_bytes,
)
from ledger.model import ANCHORS, DIMENSIONS

#: Ridge on the pooled within-pole covariance for the LDA readout. Eight samples
#: in 384 dimensions make the empirical covariance singular, so some shrinkage is
#: mandatory; 0.1 of the mean eigenvalue is a conventional default and is fixed
#: here before the readout was ever run. It is not swept.
LDA_SHRINKAGE = 0.1

#: One hypothesis per dimension, phrased as a statement about the *writing*, in
#: the same register as DIMENSION_LABELS. Frozen before measurement.
HYPOTHESES = {
    "low_mood": "The writer describes feeling low, flat or down.",
    "anxiety": "The writer describes feeling anxious, worried or on edge.",
    "sleep_disruption": "The writer describes sleeping badly.",
    "social_withdrawal": "The writer describes avoiding or withdrawing from other people.",
    "activation": "The writer describes having energy and getting things done.",
}
assert tuple(HYPOTHESES) == DIMENSIONS, "HYPOTHESES must be in DIMENSIONS order"

VARIANTS = [
    {"variant": "incumbent_centroid",
     "family": "linear readout over a mean-pooled frozen encoder",
     "model": "sentence-transformers/all-MiniLM-L6-v2",
     "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
     "readout": "centroid_difference",
     "role": "baseline - increment 3's incumbent, recomputed inside this run so every "
             "number in the table comes from one execution",
     "additivity_preserved": True,
     "selection_candidate": True},

    {"variant": "incumbent_shrunk_lda",
     "family": "linear readout over a mean-pooled frozen encoder",
     "model": "sentence-transformers/all-MiniLM-L6-v2",
     "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
     "readout": "shrunk_lda",
     "role": "control on increment 3's diagnosis: same representation, a covariance-aware "
             "linear readout instead of the centroid difference. If this recovers polarity, "
             "the readout was the problem and not the space.",
     "additivity_preserved": True,
     "selection_candidate": True},

    {"variant": "sst2_body_centroid",
     "family": "linear readout over a mean-pooled polarity-tuned encoder",
     "model": "distilbert-base-uncased-finetuned-sst-2-english",
     "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
     "readout": "centroid_difference",
     "role": "sentiment-supervised body, classification head discarded. Its encoder has "
             "seen a direction signal that the sentence-similarity distillations have not.",
     "additivity_preserved": True,
     "selection_candidate": True},

    {"variant": "nli_sbert_centroid",
     "family": "linear readout over a mean-pooled polarity-tuned encoder",
     "model": "sentence-transformers/nli-distilroberta-base-v2",
     "revision": "cc35a0bfb6251228a6fb8c797bca5fef0ece3c1d",
     "readout": "centroid_difference",
     "role": "NLI-supervised bi-encoder: entailment/contradiction structure pushed into a "
             "pooled embedding, which is the strongest polarity signal available without "
             "giving up additivity.",
     "additivity_preserved": True,
     "selection_candidate": True},

    {"variant": "nli_crossenc_distilroberta",
     "family": "cross-encoder entailment (breaks C4)",
     "model": "cross-encoder/nli-distilroberta-base",
     "revision": "b14d131f9d32668a5e6a982729b57ff6ed5dfcbd",
     "readout": "p_entailment",
     "role": "zero-shot entailment of the dimension hypothesis. Polarity-aware by "
             "construction; the attribution identity does not survive it.",
     "additivity_preserved": False,
     "selection_candidate": True},

    {"variant": "nli_crossenc_minilm2",
     "family": "cross-encoder entailment (breaks C4)",
     "model": "cross-encoder/nli-MiniLM2-L6-H768",
     "revision": "b95119ce93d3e065de6214e38cd4a97b0f2f2c6d",
     "readout": "p_entailment",
     "role": "second cross-encoder, so the family is not judged on one checkpoint's quirk.",
     "additivity_preserved": False,
     "selection_candidate": True},

    {"variant": "sst2_global_polarity",
     "family": "diagnostic - not a scorer",
     "model": "distilbert-base-uncased-finetuned-sst-2-english",
     "revision": "714eb0fa89d2f80546fda750413ed43d93601a13",
     "readout": "p_negative_sentiment_for_every_dimension",
     "role": "DIAGNOSTIC ONLY, excluded from selection by construction: one global "
             "sentiment probability used as the score for all five dimensions. It cannot "
             "tell two dimensions apart, so it is *expected* to fail the positive control. "
             "It is here to separate the two things that could be missing - polarity, or "
             "the ability to attach polarity to a topic - because the route through differs.",
     "additivity_preserved": False,
     "selection_candidate": False},
]


# --------------------------------------------------------------------------
# readouts
# --------------------------------------------------------------------------

def _lda_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Shrunk within-pole-covariance direction. Still one fixed linear row."""
    zp, zn = pos - pos.mean(0), neg - neg.mean(0)
    dof = max(len(pos) + len(neg) - 2, 1)
    cov = (zp.T @ zp + zn.T @ zn) / dof
    ridge = LDA_SHRINKAGE * float(np.trace(cov)) / cov.shape[0]
    direction = np.linalg.solve(cov + ridge * np.eye(cov.shape[0]), pos.mean(0) - neg.mean(0))
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        raise ValueError("degenerate LDA direction")
    return direction / norm


def held_out_auc_lda(pos: np.ndarray, neg: np.ndarray) -> float:
    """The increment-3 held-out protocol, with the LDA readout fitted inside the loop.

    Fitting the covariance on all ten sentences and then scoring two of them would
    leak; the direction is rebuilt from the 4+4 that remain, exactly as the
    centroid version rebuilds its centroids.
    """
    wins = []
    for i, j in itertools.product(range(len(pos)), range(len(neg))):
        direction = _lda_direction(np.delete(pos, i, axis=0), np.delete(neg, j, axis=0))
        a, b = float(pos[i] @ direction), float(neg[j] @ direction)
        wins.append(1.0 if a > b else (0.5 if a == b else 0.0))
    return float(np.mean(wins))


def in_sample_auc_lda(pos: np.ndarray, neg: np.ndarray) -> float:
    direction = _lda_direction(pos, neg)
    return _auc(pos @ direction, neg @ direction)


READOUTS = {
    "centroid_difference": (held_out_auc, in_sample_auc),
    "shrunk_lda": (held_out_auc_lda, in_sample_auc_lda),
}


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------

def measure_linear(spec: dict) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    model = AutoModel.from_pretrained(spec["model"], revision=spec["revision"],
                                      attn_implementation="eager").eval()
    cfg = AutoConfig.from_pretrained(spec["model"], revision=spec["revision"])
    held_fn, in_fn = READOUTS[spec["readout"]]

    poles = {dim: (embed_sentences(model, tokenizer, ANCHORS[dim]["positive"]),
                   embed_sentences(model, tokenizer, ANCHORS[dim]["negative"]))
             for dim in DIMENSIONS}

    per_dim = {dim: {"held_out_auc": round(held_fn(*poles[dim]), 4),
                     "in_sample_auc": round(in_fn(*poles[dim]), 4)}
               for dim in DIMENSIONS}
    control = {f"{a}|{b}": round(held_fn(poles[a][0], poles[b][0]), 4)
               for a, b in itertools.combinations(DIMENSIONS, 2)}

    size = projected_int8embed_bytes(model)
    return {
        "hidden_size": cfg.hidden_size,
        "num_hidden_layers": cfg.num_hidden_layers,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "int8embed_projected_MiB": round(size["bytes"] / 1048576, 2),
        "int8embed_projected_bytes": size["bytes"],
        "per_dimension": per_dim,
        "macro_held_out_auc": round(float(np.mean([v["held_out_auc"] for v in per_dim.values()])), 4),
        "macro_in_sample_auc": round(float(np.mean([v["in_sample_auc"] for v in per_dim.values()])), 4),
        "positive_control_pairs": control,
        "positive_control_macro": round(float(np.mean(list(control.values()))), 4),
        "protocol_note": "identical to increment 3: withhold one anchor from each pole together, "
                         "refit from the remaining 4+4, rank the two withheld. 25 pairs per dimension.",
    }


@torch.no_grad()
def _entail_probs(model, tokenizer, sentences, hypothesis, entail_idx, max_length=128) -> np.ndarray:
    batch = tokenizer(text=list(sentences), text_pair=[hypothesis] * len(sentences),
                      padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    logits = model(**batch).logits
    return torch.softmax(logits, dim=-1)[:, entail_idx].numpy().astype(np.float64)


def measure_cross_encoder(spec: dict) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["model"], revision=spec["revision"]).eval()
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    if "entailment" not in labels:
        raise ValueError(f"{spec['model']}: no entailment label in {model.config.id2label}")
    entail_idx = labels["entailment"]

    scored = {dim: {pole: _entail_probs(model, tokenizer, ANCHORS[dim][pole],
                                        HYPOTHESES[dim], entail_idx)
                    for pole in ("positive", "negative")}
              for dim in DIMENSIONS}

    per_dim = {dim: {"held_out_auc": round(_auc(scored[dim]["positive"], scored[dim]["negative"]), 4),
                     "in_sample_auc": None}
               for dim in DIMENSIONS}
    # Control: dimension a's own hypothesis, ranking a's positives above b's.
    control = {f"{a}|{b}": round(_auc(scored[a]["positive"],
                                      _entail_probs(model, tokenizer, ANCHORS[b]["positive"],
                                                    HYPOTHESES[a], entail_idx)), 4)
               for a, b in itertools.combinations(DIMENSIONS, 2)}

    size = projected_int8embed_bytes(model)
    return {
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "int8embed_projected_MiB": round(size["bytes"] / 1048576, 2),
        "int8embed_projected_bytes": size["bytes"],
        "id2label": dict(model.config.id2label),
        "entailment_index": entail_idx,
        "per_dimension": per_dim,
        "macro_held_out_auc": round(float(np.mean([v["held_out_auc"] for v in per_dim.values()])), 4),
        "macro_in_sample_auc": None,
        "positive_control_pairs": control,
        "positive_control_macro": round(float(np.mean(list(control.values()))), 4),
        "protocol_note": "NOT LITERALLY THE INCREMENT-3 COMPUTATION, and the difference is in our "
                         "favour to state: a cross-encoder fits nothing, so there is no direction "
                         "to rebuild and no in-sample number to report. The AUC is over the same 25 "
                         "(positive_i, negative_j) comparisons per dimension, which is what makes "
                         "the column comparable. The control uses dimension a's hypothesis to rank "
                         "a's positives above b's, the same question the linear control asks.",
    }


@torch.no_grad()
def measure_global_polarity(spec: dict) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(spec["model"], revision=spec["revision"])
    model = AutoModelForSequenceClassification.from_pretrained(
        spec["model"], revision=spec["revision"]).eval()
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    neg_idx = labels["negative"]

    def p_neg(sentences):
        batch = tokenizer(list(sentences), padding=True, truncation=True,
                          max_length=128, return_tensors="pt")
        return torch.softmax(model(**batch).logits, dim=-1)[:, neg_idx].numpy().astype(np.float64)

    scored = {dim: {pole: p_neg(ANCHORS[dim][pole]) for pole in ("positive", "negative")}
              for dim in DIMENSIONS}
    per_dim = {dim: {"held_out_auc": round(_auc(scored[dim]["positive"], scored[dim]["negative"]), 4),
                     "in_sample_auc": None}
               for dim in DIMENSIONS}
    control = {f"{a}|{b}": round(_auc(scored[a]["positive"], scored[b]["positive"]), 4)
               for a, b in itertools.combinations(DIMENSIONS, 2)}
    size = projected_int8embed_bytes(model)
    return {
        "hidden_size": model.config.hidden_size,
        "num_hidden_layers": model.config.num_hidden_layers,
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "int8embed_projected_MiB": round(size["bytes"] / 1048576, 2),
        "int8embed_projected_bytes": size["bytes"],
        "id2label": dict(model.config.id2label),
        "negative_index": neg_idx,
        "per_dimension": per_dim,
        "macro_held_out_auc": round(float(np.mean([v["held_out_auc"] for v in per_dim.values()])), 4),
        "macro_in_sample_auc": None,
        "positive_control_pairs": control,
        "positive_control_macro": round(float(np.mean(list(control.values()))), 4),
        "protocol_note": "One score for all five dimensions, so the control is a formality: it "
                         "asks whether a single number can tell two dimensions apart, and it "
                         "cannot. Excluded from selection before it was run.",
    }


MEASURERS = {
    "centroid_difference": measure_linear,
    "shrunk_lda": measure_linear,
    "p_entailment": measure_cross_encoder,
    "p_negative_sentiment_for_every_dimension": measure_global_polarity,
}


def main() -> int:
    results = []
    for spec in VARIANTS:
        t0 = time.time()
        measured = MEASURERS[spec["readout"]](spec)
        results.append({**spec, **measured, "elapsed_s": round(time.time() - t0, 2)})
        r = results[-1]
        print(f"  {r['variant']:30} held_out={r['macro_held_out_auc']:.3f} "
              f"control={r['positive_control_macro']:.3f} "
              f"additive={r['additivity_preserved']}", flush=True)

    candidates = [r for r in results if r["selection_candidate"]]
    readable = [r for r in candidates if r["positive_control_macro"] >= POSITIVE_CONTROL_FLOOR]
    usable = [r for r in readable if r["macro_held_out_auc"] >= USABLE_HELD_OUT_AUC]
    additive = [r for r in usable if r["additivity_preserved"]]
    pool = additive or usable
    selected = max(pool, key=lambda r: r["macro_held_out_auc"]) if pool else None

    baseline = next(r for r in results if r["variant"] == "incumbent_centroid")
    # Option C's basis, under the best READABLE variant rather than the baseline only.
    best_readable = max(readable, key=lambda r: r["macro_held_out_auc"]) if readable else baseline
    option_c_dimensions = {
        v: best_readable["per_dimension"][v]["held_out_auc"] for v in DIMENSIONS
        if best_readable["per_dimension"][v]["held_out_auc"] >= USABLE_HELD_OUT_AUC}

    # A selection is not an adoption. The ceilings below were fixed in
    # export/SIZE_BUDGET.md before anything was exported (build increment 2), and
    # a scorer that wins on separation can still be unshippable on size. This
    # block is REPORTING ONLY: it was added after the variants above were
    # measured, it changes no threshold and no selection, and it exists so that
    # "selected but not adopted" has to be stated rather than left implicit.
    adoption_blockers = []
    if selected is not None:
        sel_bytes = selected["int8embed_projected_bytes"]
        if sel_bytes > CEILINGS["CEIL_1_int8_model_bytes"]:
            adoption_blockers.append(
                f"CEIL_1_int8_model_bytes: projected {sel_bytes} > "
                f"{CEILINGS['CEIL_1_int8_model_bytes']}")
        if ORT_RUNTIME_FLOOR_BYTES + sel_bytes > CEILINGS["CEIL_3_cold_payload_bytes"]:
            adoption_blockers.append(
                f"CEIL_3_cold_payload_bytes: projected "
                f"{ORT_RUNTIME_FLOOR_BYTES + sel_bytes} > "
                f"{CEILINGS['CEIL_3_cold_payload_bytes']} (excl. tokenizer)")

    if selected is not None and selected["additivity_preserved"]:
        verdict = (f"SELECT {selected['variant']} ({selected['model']}), macro held-out "
                   f"{selected['macro_held_out_auc']}. Option B is open on separation and C4 "
                   "survives: the head is still a fixed linear row over a mean-pooled vector, so "
                   "the attribution identity is untouched.")
        if adoption_blockers:
            verdict += (" NOT ADOPTED THIS INCREMENT: it breaches "
                        f"{len(adoption_blockers)} size ceiling(s) - "
                        + "; ".join(adoption_blockers) + ". The scorer that separates is "
                        f"{selected['int8embed_projected_MiB']} MiB against CEIL-1's 32, so "
                        "R4-CEIL-001 returns as the binding defect and export/common.py is left "
                        "on the incumbent.")
    elif selected is not None:
        verdict = (f"SELECT {selected['variant']} ({selected['model']}), macro held-out "
                   f"{selected['macro_held_out_auc']} - but it is a cross-encoder and the "
                   "exactly-additive attribution identity does NOT survive it. Taking it means "
                   "deleting criterion C4's claim from the submission, not restating it.")
    else:
        verdict = ("NO SELECTION. Option B is CLOSED: no polarity-aware scorer with a readable "
                   f"positive control reaches macro held-out AUC {USABLE_HELD_OUT_AUC}. The "
                   f"fallback is option C - ship only {sorted(option_c_dimensions) or 'nothing'} "
                   f"- or option D.")

    report = {
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "increment": 4,
        "question": "R4-HEAD-003 option B: does any polarity-aware scorer separate held-out "
                    "anchors, and does the exactly-additive attribution head survive it?",
        "selection_rule": "Fixed before any variant was measured. (1) A variant is READABLE only "
                          f"if its positive control holds at >= {POSITIVE_CONTROL_FLOOR}. (2) It is "
                          f"USABLE only if macro held-out AUC >= {USABLE_HELD_OUT_AUC}. (3) Among "
                          "usable variants, additivity-preserving ones are preferred outright - a "
                          "non-additive scorer is chosen only if NO additive one is usable, and "
                          "choosing it costs the C4 claim. (4) Tie-break on macro held-out AUC. "
                          "(5) If nothing is usable, option B is closed and the fallback is C or D.",
        "thresholds_inherited_from_increment_3": {
            "usable_held_out_auc": USABLE_HELD_OUT_AUC,
            "positive_control_floor": POSITIVE_CONTROL_FLOOR,
            "note": "imported from export/encoder_ablation.py, not re-typed, so they cannot drift",
        },
        "lda_shrinkage": LDA_SHRINKAGE,
        "hypotheses": HYPOTHESES,
        "variants": results,
        "selected": (selected or {}).get("variant"),
        "selected_model": (selected or {}).get("model"),
        "selected_revision": (selected or {}).get("revision"),
        "selection_preserves_additivity": (selected or {}).get("additivity_preserved"),
        "adoption": {
            "adoption_blockers": adoption_blockers,
            "adopted": selected is not None and not adoption_blockers,
            "rule": "A selected scorer is adopted into export/common.py only if its projected "
                    "int8_embed build clears CEIL-1 and CEIL-3, which were fixed in "
                    "export/SIZE_BUDGET.md before build increment 2 exported anything. If it "
                    "does not, export/common.py stays on the incumbent and the blocker is "
                    "recorded in docs/limitations.md. tests/test_scorer_ablation.py enforces "
                    "both directions.",
            "added_after_measurement": "This block is reporting only. It was written after the "
                                       "variants were measured; it changes no threshold, no "
                                       "measured number and no selection.",
        },
        "readable_variants": [r["variant"] for r in readable],
        "unreadable_variants": [r["variant"] for r in candidates
                                if r["positive_control_macro"] < POSITIVE_CONTROL_FLOOR],
        "usable_variants": [r["variant"] for r in usable],
        "baseline_reference": {
            "variant": baseline["variant"],
            "macro_held_out_auc": baseline["macro_held_out_auc"],
            "increment_3_value": 0.504,
            "note": "recomputed here; must reproduce increment 3 or one of the two runs is wrong",
        },
        "option_c_basis": {
            "measured_under": best_readable["variant"],
            "dimensions_at_or_above_usable_threshold": option_c_dimensions,
            "per_dimension_held_out_auc": {
                d: best_readable["per_dimension"][d]["held_out_auc"] for d in DIMENSIONS},
        },
        "verdict": verdict,
    }
    write_json(ARTIFACTS / "scorer_ablation.json", report)

    print()
    print(f"{'variant':30} {'model':52} {'held_out':>9} {'in_samp':>8} {'control':>8} "
          f"{'additive':>9} {'int8emb':>9}")
    for r in results:
        ins = "-" if r["macro_in_sample_auc"] is None else f"{r['macro_in_sample_auc']:.3f}"
        print(f"{r['variant']:30} {r['model']:52} {r['macro_held_out_auc']:>9.3f} {ins:>8} "
              f"{r['positive_control_macro']:>8.3f} {str(r['additivity_preserved']):>9} "
              f"{r['int8embed_projected_MiB']:>7.2f}Mi")
    print()
    print("per-dimension held-out AUC")
    for r in results:
        per = "  ".join(f"{d[:4]}={r['per_dimension'][d]['held_out_auc']:.3f}" for d in DIMENSIONS)
        print(f"  {r['variant']:30} {per}")
    print()
    print(verdict)
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
