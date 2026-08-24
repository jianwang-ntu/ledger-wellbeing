"""Measure R8-1 and R8-2: does the *span* view still add up to the score?

The token-level identity is already measured on every build (`verify.py`,
residual ~3e-07). This script asks the question the product actually depends on:
after tokens are regrouped into the user's own words and sentences, do the
visible contributions still sum to the logit?

If they do not, the span view is a decoration over a decision made elsewhere,
and `plan.md` C4 — "explainable AI mechanisms to show why the model reached a
health conclusion" — is not satisfied by it.

Two things are checked on every probe entry, at both granularities, for all five
dimensions:

**R8-1, partition.** Every token with a live attention mask lands in exactly one
bucket. Counted directly: the sum of per-span token counts plus the structural
count must equal the number of live tokens, and no token index may appear twice.

**R8-2, additivity.** ``abs(logit − (Σ span + structural + bias)) ≤ 1e-4`` —
the same tolerance as the token-level rule. The aggregation step does not get a
tolerance of its own.

Run: ``python3 export/span_additivity.py``
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "export"))

from ledger.app import offline  # noqa: F401,E402  - offline before transformers

import numpy as np  # noqa: E402

from common import MAX_LENGTH, probe_entries, sha256_file  # noqa: E402
from ledger.app.engine import ADDITIVITY_MAX_RESIDUAL, BUILD_FILES, selected_build  # noqa: E402
from ledger.app.spans import _assign, attribute_spans, sentence_spans, word_spans  # noqa: E402
from ledger.model.dimensions import DIMENSIONS  # noqa: E402

GRANULARITIES = {"sentence": sentence_spans, "word": word_spans}


def main() -> int:
    import onnxruntime as ort
    import torch
    from transformers import AutoTokenizer

    build = selected_build()
    model_path = BUILD_FILES[build]
    tokenizer = AutoTokenizer.from_pretrained(ROOT / "artifacts" / "tokenizer",
                                              local_files_only=True)
    ckpt = torch.load(ROOT / "artifacts" / "torch" / "head.pt",
                      map_location="cpu", weights_only=True)
    bias = ckpt["head_bias"].numpy().astype(np.float64)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

    entries = probe_entries()
    results = {name: {"max_residual": 0.0, "checks": 0, "worst": None}
               for name in GRANULARITIES}
    partition = {"checks": 0, "failures": []}

    for entry_index, text in enumerate(entries):
        encoded = tokenizer([text], padding="max_length", truncation=True,
                            max_length=MAX_LENGTH, return_offsets_mapping=True,
                            return_tensors="np")
        feeds = {"input_ids": encoded["input_ids"].astype(np.int64),
                 "attention_mask": encoded["attention_mask"].astype(np.int64)}
        logits, token_attr = session.run(["logits", "token_attr"], feeds)
        logits = logits.astype(np.float64)[0]
        token_attr = token_attr.astype(np.float64)[0]
        offsets, mask = encoded["offset_mapping"][0], encoded["attention_mask"][0]
        live = int(np.asarray(mask).sum())

        for name, splitter in GRANULARITIES.items():
            spans = splitter(text)

            # R8-1: partition, checked on the assignment itself.
            buckets, structural = _assign(offsets, mask, spans)
            assigned = [i for bucket in buckets for i in bucket] + structural
            partition["checks"] += 1
            if len(assigned) != live or len(set(assigned)) != len(assigned):
                partition["failures"].append({
                    "entry": entry_index, "granularity": name,
                    "live_tokens": live, "assigned": len(assigned),
                    "distinct": len(set(assigned)),
                })

            # R8-2: additivity after aggregation.
            for k, dim in enumerate(DIMENSIONS):
                span_attr, structural_attr, _ = attribute_spans(
                    text, offsets, mask, token_attr[:, k], spans)
                total = sum(s.attribution for s in span_attr) + structural_attr + float(bias[k])
                residual = abs(float(logits[k]) - total)
                results[name]["checks"] += 1
                if residual > results[name]["max_residual"]:
                    results[name]["max_residual"] = residual
                    results[name]["worst"] = {"entry": entry_index, "dimension": dim}

    r81 = not partition["failures"]
    r82 = all(r["max_residual"] <= ADDITIVITY_MAX_RESIDUAL for r in results.values())

    report = {
        "measured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rules": "R8-1, R8-2 - export/INCREMENT_8_PREREGISTRATION.md",
        "build": build,
        "model_sha256": sha256_file(model_path),
        "probe_entries": len(entries),
        "max_length": MAX_LENGTH,
        "tolerance": ADDITIVITY_MAX_RESIDUAL,
        "tolerance_note": "identical to R6-1; the aggregation step gets no tolerance of its own",
        "R8_1_partition": {
            "checks": partition["checks"],
            "failures": partition["failures"],
            "pass": r81,
        },
        "R8_2_additivity": {
            name: {"checks": r["checks"], "max_residual": r["max_residual"],
                   "worst": r["worst"], "pass": r["max_residual"] <= ADDITIVITY_MAX_RESIDUAL}
            for name, r in results.items()
        },
        "verdict": "PASS" if (r81 and r82) else "FAIL",
    }
    out = ROOT / "artifacts" / "span_additivity.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
