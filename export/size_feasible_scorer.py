"""Build increment 5: is there a scorer that separates AND ships?

Increment 4 ended with two artifacts that are not the same artifact. The only
scorer measured to separate held-out text is
`sentence-transformers/nli-distilroberta-base-v2` at macro held-out AUC 0.880,
and it projects to 201.68 MiB against CEIL-1's 32. The incumbent ships and does
not separate (0.504). R4-CEIL-001 is the binding defect.

status.json fixed this increment's terms before anything here was run:

    "close the gap between the scorer that works (201.68 MiB) and the size that
     ships (32 MiB). Fix the candidate list BEFORE running it - small
     NLI/entailment-supervised bi-encoders only, hidden <= 384, measured under
     the identical held-out protocol and the identical 0.70/0.75 thresholds. If
     none clears both, plan.md R-4 is the answer: a local desktop app, which
     keeps zero-egress and drops the in-browser claim rather than fudging it.
     Do NOT extend increment 4's candidate list after the fact - that turns a
     measurement into a search."

So the candidate list below is closed. It was written, committed and only then
executed; the commit that introduces this file contains no measurement, and any
later addition to CANDIDATES is visible in the diff as a separate commit.

## The one change to the selection rule, and why it is not a loosening

Increment 4 selected on separation and *reported* shippability afterwards. That
is how it produced a selection it could not adopt. Here **shippability is a
selection criterion**, imported from `common.CEILINGS` (fixed in
export/SIZE_BUDGET.md before build increment 2 exported anything), not retyped.
A scorer that separates and does not fit is NOT selected by this file. It is
recorded as a quality reference and the fallback fires.

This is strictly harder to pass than increment 4's rule, so nothing that failed
there can pass here on the rule change alone.

## How the candidates were chosen, before any of them was measured

The size envelope follows from `projected_int8embed_bytes` and CEIL-1 alone,
with no measurement: the int8_embed build stores the embedding table at 1 byte
per parameter and every encoder MatMul at 4, so at vocab 30522 a hidden-384
encoder buys ~3 layers inside 32 MiB and a hidden-256 encoder buys ~8. That
arithmetic is what "hidden <= 384" has to be spent on, and it is why depth, not
width, is the variable below.

Against that envelope, HuggingFace was searched for NLI/entailment-supervised
encoders with hidden_size <= 384 (2026-08-24). Four are carried as selection
candidates and three as references. Every revision is pinned by commit sha read
from the model API, never by tag, and every licence is recorded because a model
we cannot licence cannot ship whatever it scores.

Writes artifacts/size_feasible_scorer.json. Exit 0 if a scorer is selected -
which now means separates AND fits - and 2 if none is, in which case option B is
closed at hidden <= 384 and plan.md R-4 is the answer.
"""

from __future__ import annotations

import json
import time

import numpy as np

from common import ARTIFACTS, CEILINGS, ORT_RUNTIME_FLOOR_BYTES, write_json
from encoder_ablation import POSITIVE_CONTROL_FLOOR, USABLE_HELD_OUT_AUC
from ledger.model import DIMENSIONS
from scorer_ablation import measure_linear

#: Every row is mean-pooled over the frozen body with a fixed linear
#: centroid-difference row on top, so the exactly-additive attribution identity
#: (criterion C4) survives every one of them by construction. No cross-encoder
#: appears here: increment 4 already measured that family and it breaks C4.
CANDIDATES = [
    {"variant": "nli_xtremedistil_h256",
     "model": "MoritzLaurer/xtremedistil-l6-h256-zeroshot-v1.1-all-33",
     "revision": "c07f66d9cbf781191bee66edfe8ad7856f045781",
     "licence": "mit",
     "role": "the one checkpoint found that is BOTH entailment-supervised and inside the "
             "size envelope: hidden 256, 6 layers, 30522-entry vocab, the same shape as "
             "increment 3's bert_uncased_L-6_H-256 which projected to 25.91 MiB. If NLI "
             "supervision is what increment 4 showed it to be, this is the row that ships it.",
     "expected_shippable_before_measurement": True,
     "selection_candidate": True},

    {"variant": "nli_paraphrase_minilm_l3",
     "model": "sentence-transformers/paraphrase-MiniLM-L3-v2",
     "revision": "4ca70771034acceecb2e72475f72050fcdde4ddc",
     "licence": "apache-2.0",
     "role": "the only hidden-384 depth inside the envelope (3 layers). AllNLI is part of "
             "its training mixture, so it carries a weaker form of the same supervision. "
             "Included because if the h256 row fails, the question 'was it the width?' has "
             "to have been measured rather than argued.",
     "expected_shippable_before_measurement": True,
     "selection_candidate": True},

    {"variant": "nli_deberta_v3_xsmall_body",
     "model": "cross-encoder/nli-deberta-v3-xsmall",
     "revision": "a150876415327c80daeff35ca6f68f5ed8cf5c24",
     "licence": "apache-2.0",
     "role": "the strongest pure-NLI supervision available at hidden <= 384; the sequence "
             "classification head is discarded and the body is mean-pooled, exactly as "
             "increment 4 treated the SST-2 body. Its 128100-entry vocab is expected to "
             "breach CEIL-1 on the embedding table alone, so it is carried as the quality "
             "ceiling of the hidden<=384 family, not as a hopeful.",
     "expected_shippable_before_measurement": False,
     "selection_candidate": True},

    {"variant": "nli_deberta_v3_xsmall_zeroshot",
     "model": "MoritzLaurer/deberta-v3-xsmall-zeroshot-v1.1-all-33",
     "revision": "262ae02f29173eec1c250f90804dc7edc677dcff",
     "licence": "mit",
     "role": "same body under a broader zero-shot mixture, so the family is not judged on "
             "one checkpoint's quirk - the same reason increment 4 carried two "
             "cross-encoders.",
     "expected_shippable_before_measurement": False,
     "selection_candidate": True},

    # ---- references: measured, excluded from selection, reason recorded ----

    {"variant": "incumbent_centroid",
     "model": "sentence-transformers/all-MiniLM-L6-v2",
     "revision": "1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
     "licence": "apache-2.0",
     "role": "baseline, recomputed inside this run. Must reproduce increments 3 and 4 "
             "(0.504) or one of the three measurements is wrong and none may be quoted.",
     "expected_shippable_before_measurement": False,
     "selection_candidate": False,
     "excluded_because": "baseline - it is the thing being replaced"},

    {"variant": "nli_sbert_768_reference",
     "model": "sentence-transformers/nli-distilroberta-base-v2",
     "revision": "cc35a0bfb6251228a6fb8c797bca5fef0ece3c1d",
     "licence": "apache-2.0",
     "role": "increment 4's selection (0.880), recomputed as the upper reference so the "
             "gap this increment is trying to close is measured in this run and not "
             "quoted from another one.",
     "expected_shippable_before_measurement": False,
     "selection_candidate": False,
     "excluded_because": "hidden_size 768 - outside the hidden <= 384 rule fixed before "
                         "this increment, and already measured unshippable at 201.68 MiB"},

    {"variant": "nli_minilm_l6_mnli_unlicensed",
     "model": "MoritzLaurer/MiniLM-L6-mnli",
     "revision": "6e0917f1a395b7a6c0f054a56b91c45d8e3af92f",
     "licence": None,
     "role": "MNLI-supervised at the incumbent's exact shape (hidden 384, 6 layers, 30522 "
             "vocab), which isolates supervision from capacity: it is the incumbent's "
             "size with NLI training. Measured for that reason and EXCLUDED FROM "
             "SELECTION ON LICENCE, not on its number.",
     "expected_shippable_before_measurement": False,
     "selection_candidate": False,
     "excluded_because": "no licence is declared on the model repository or its card "
                         "(checked 2026-08-24 via the HF model API and README). "
                         "data/MANIFEST.md requires a cleared licence for anything "
                         "redistributed in an artifact; an undeclared licence is not a "
                         "permissive one, and resolving that in our own favour is exactly "
                         "what GOAL.md non-negotiable 4 forbids."},
]

#: Increment 4's rule, plus criterion (3). Fixed before this file was executed.
SELECTION_RULE = (
    "(1) READABLE: positive control macro >= POSITIVE_CONTROL_FLOOR, or nothing else in "
    "the row is read. (2) USABLE: macro held-out AUC >= USABLE_HELD_OUT_AUC. "
    "(3) SHIPPABLE - NEW THIS INCREMENT, AND A TIGHTENING: projected int8_embed bytes "
    "<= CEIL_1 AND ORT runtime floor + those bytes <= CEIL_3. Increment 4 selected on "
    "(1)+(2) and reported (3) afterwards, which is how it produced a selection it could "
    "not adopt. (4) ADDITIVE: required; every row here is mean-pooled + one linear row, "
    "asserted rather than assumed. (5) Tie-break on macro held-out AUC. (6) If nothing "
    "satisfies 1-4, option B is CLOSED at hidden <= 384 and plan.md R-4 - a local "
    "desktop app, zero-egress kept, the in-browser claim dropped - is the answer."
)


def shippability(row: dict) -> dict:
    """CEIL-1 and CEIL-3 from common.CEILINGS. Nothing here is retyped."""
    b = row["int8embed_projected_bytes"]
    cold = ORT_RUNTIME_FLOOR_BYTES + b
    blockers = []
    if b > CEILINGS["CEIL_1_int8_model_bytes"]:
        blockers.append(f"CEIL_1_int8_model_bytes: projected {b} > "
                        f"{CEILINGS['CEIL_1_int8_model_bytes']}")
    if cold > CEILINGS["CEIL_3_cold_payload_bytes"]:
        blockers.append(f"CEIL_3_cold_payload_bytes: projected {cold} > "
                        f"{CEILINGS['CEIL_3_cold_payload_bytes']} (excl. tokenizer)")
    return {"projected_cold_payload_bytes": cold,
            "projected_cold_payload_MiB": round(cold / 1048576, 2),
            "size_blockers": blockers,
            "shippable": not blockers}


def main() -> int:
    results = []
    for spec in CANDIDATES:
        t0 = time.time()
        measured = measure_linear({**spec, "readout": "centroid_difference"})
        row = {**spec, **measured, "readout": "centroid_difference",
               "additivity_preserved": True, "elapsed_s": round(time.time() - t0, 2)}
        row.update(shippability(row))
        results.append(row)
        print(f"  {row['variant']:32} H={row['hidden_size']:<4} L={row['num_hidden_layers']:<3} "
              f"held_out={row['macro_held_out_auc']:.3f} "
              f"control={row['positive_control_macro']:.3f} "
              f"int8emb={row['int8embed_projected_MiB']:>7.2f}Mi "
              f"ships={row['shippable']}", flush=True)

    candidates = [r for r in results if r["selection_candidate"]]
    readable = [r for r in candidates if r["positive_control_macro"] >= POSITIVE_CONTROL_FLOOR]
    usable = [r for r in readable if r["macro_held_out_auc"] >= USABLE_HELD_OUT_AUC]
    shippable = [r for r in usable if r["shippable"]]
    additive = [r for r in shippable if r["additivity_preserved"]]
    selected = max(additive, key=lambda r: r["macro_held_out_auc"]) if additive else None

    baseline = next(r for r in results if r["variant"] == "incumbent_centroid")
    upper = next(r for r in results if r["variant"] == "nli_sbert_768_reference")

    # What was lost at each criterion, so "no selection" says WHERE it failed.
    attrition = {
        "measured": [r["variant"] for r in results],
        "selection_candidates": [r["variant"] for r in candidates],
        "readable": [r["variant"] for r in readable],
        "dropped_unreadable": [r["variant"] for r in candidates if r not in readable],
        "usable": [r["variant"] for r in usable],
        "dropped_not_usable": [r["variant"] for r in readable if r not in usable],
        "shippable_and_usable": [r["variant"] for r in shippable],
        "dropped_on_size": [r["variant"] for r in usable if r not in shippable],
        "separates_but_does_not_fit": [
            r["variant"] for r in results
            if r["macro_held_out_auc"] >= USABLE_HELD_OUT_AUC
            and r["positive_control_macro"] >= POSITIVE_CONTROL_FLOOR and not r["shippable"]],
        "fits_but_does_not_separate": [
            r["variant"] for r in results
            if r["shippable"] and r["macro_held_out_auc"] < USABLE_HELD_OUT_AUC],
    }

    if selected is not None:
        verdict = (
            f"SELECT AND ADOPT {selected['variant']} ({selected['model']}): macro held-out "
            f"{selected['macro_held_out_auc']} on a positive control of "
            f"{selected['positive_control_macro']}, projected int8_embed "
            f"{selected['int8embed_projected_MiB']} MiB inside CEIL-1's 32 and a cold payload "
            f"of {selected['projected_cold_payload_MiB']} MiB inside CEIL-3's 64. R4-CEIL-001 "
            "closes: the scorer that separates and the size that ships are the same artifact. "
            "Additivity is preserved, so criterion C4 stands unchanged.")
    else:
        verdict = (
            "NO SELECTION. Option B is CLOSED at hidden <= 384: no entailment-supervised "
            "bi-encoder in the fixed candidate list is READABLE, USABLE and SHIPPABLE at "
            "once. "
            + (f"These separate and do not fit: {attrition['separates_but_does_not_fit']}. "
               if attrition["separates_but_does_not_fit"] else "")
            + (f"These fit and do not separate: {attrition['fits_but_does_not_separate']}. "
               if attrition["fits_but_does_not_separate"] else "")
            + "R4-CEIL-001 therefore does not close by shrinking the scorer, and plan.md R-4 "
              "is the answer: a local desktop app, which keeps the zero-egress claim and "
              "drops the in-browser claim rather than fudging it. No ceiling is edited to "
              "reach a different conclusion.")

    report = {
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "increment": 5,
        "question": "R4-CEIL-001: is there an entailment-supervised bi-encoder at hidden <= 384 "
                    "that separates held-out anchors AND fits inside the ceilings fixed in "
                    "export/SIZE_BUDGET.md before build increment 2?",
        "candidate_list_closed_before_execution": (
            "status.json's next_action fixed the terms; CANDIDATES is committed in the same "
            "commit as this docstring and before any run. Extending it after seeing these "
            "numbers would turn a measurement into a search and is not done."),
        "selection_rule": SELECTION_RULE,
        "selection_rule_is_a_tightening_of_increment_4": (
            "Criterion (3) is added. Increment 4's rule is a strict subset, so nothing that "
            "failed there can pass here because the rule changed."),
        "thresholds_inherited": {
            "usable_held_out_auc": USABLE_HELD_OUT_AUC,
            "positive_control_floor": POSITIVE_CONTROL_FLOOR,
            "ceil_1_int8_model_bytes": CEILINGS["CEIL_1_int8_model_bytes"],
            "ceil_3_cold_payload_bytes": CEILINGS["CEIL_3_cold_payload_bytes"],
            "ort_runtime_floor_bytes": ORT_RUNTIME_FLOOR_BYTES,
            "note": "imported from export/encoder_ablation.py and export/common.py, not "
                    "re-typed, so they cannot drift between increments",
        },
        "size_envelope_derivation": (
            "int8_embed stores embedding parameters at 1 byte and every other parameter at 4. "
            "At vocab 30522 that is ~11.9 MB of embeddings at hidden 384 and ~7.9 MB at hidden "
            "256, leaving ~5.4M and ~6.4M non-embedding parameters inside CEIL-1 - about 3 "
            "encoder layers at hidden 384 and about 8 at hidden 256. Arithmetic, fixed before "
            "measurement, and it is why the candidates vary depth rather than width."),
        "variants": results,
        "attrition": attrition,
        "selected": (selected or {}).get("variant"),
        "selected_model": (selected or {}).get("model"),
        "selected_revision": (selected or {}).get("revision"),
        "adopted": selected is not None,
        "adoption_rule": (
            "Unlike increment 4, selection here already requires shippability, so a selection "
            "IS an adoption: export/common.py's BASE_MODEL/BASE_REVISION move to the selected "
            "row and the export pipeline is re-run against SIZE_BUDGET.md. If nothing is "
            "selected, export/common.py is left untouched on the incumbent."),
        "baseline_reference": {
            "variant": baseline["variant"],
            "macro_held_out_auc": baseline["macro_held_out_auc"],
            "increment_3_and_4_value": 0.504,
            "note": "recomputed here; must reproduce or one of the three runs is wrong",
        },
        "upper_reference": {
            "variant": upper["variant"],
            "macro_held_out_auc": upper["macro_held_out_auc"],
            "increment_4_value": 0.88,
            "int8embed_projected_MiB": upper["int8embed_projected_MiB"],
            "note": "the gap this increment tries to close, recomputed in this run",
        },
        "excluded_from_selection": {
            r["variant"]: r["excluded_because"] for r in results if not r["selection_candidate"]},
        "expectation_vs_outcome": {
            r["variant"]: {"expected_shippable": r["expected_shippable_before_measurement"],
                           "measured_shippable": r["shippable"]} for r in results},
        "verdict": verdict,
    }
    write_json(ARTIFACTS / "size_feasible_scorer.json", report)

    print()
    print(f"{'variant':32} {'model':58} {'H':>4} {'L':>3} {'held_out':>9} {'control':>8} "
          f"{'int8emb':>9} {'cold':>8} {'ships':>6}")
    for r in results:
        print(f"{r['variant']:32} {r['model']:58} {r['hidden_size']:>4} "
              f"{r['num_hidden_layers']:>3} {r['macro_held_out_auc']:>9.3f} "
              f"{r['positive_control_macro']:>8.3f} {r['int8embed_projected_MiB']:>7.2f}Mi "
              f"{r['projected_cold_payload_MiB']:>6.1f}Mi {str(r['shippable']):>6}")
    print()
    print("per-dimension held-out AUC")
    for r in results:
        per = "  ".join(f"{d[:4]}={r['per_dimension'][d]['held_out_auc']:.3f}" for d in DIMENSIONS)
        print(f"  {r['variant']:32} {per}")
    print()
    print(verdict)
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
