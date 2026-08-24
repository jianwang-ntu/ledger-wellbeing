"""Measure DEFECT-INC8-001's fix: does the app's tokenizer encode identically?

The whole measured pipeline — `verify.py`, `quant_sensitivity.py`,
`span_additivity.py` — was run through `transformers.AutoTokenizer`. The shipped
application now uses the `tokenizers` library directly (see
`ledger/app/local_tokenizer.py` for why). Every one of those measurements is
transferable to the application only if the two paths produce the same tensors.

Protocol, identical to R7-2's: elementwise comparison of `input_ids`,
`attention_mask` and `offset_mapping` over

* the 64 deterministic probe entries at `max_length=256`, and
* all 50 anchor sentences in `ledger/model/dimensions.py`.

**A single mismatch anywhere fails.** There is no tolerance: these are integers.

The head bias is checked in the same run, because the application now reads it
from `artifacts/torch/build_report.json` instead of importing `torch` to open
`head.pt`. The stored bias is a float32 cast of the calibration offset, so the
comparison is exact after that cast, and the residual before it is reported.

Import order note: `transformers` is imported **first**, before `torch` or
`onnxruntime`, because that is the ordering under which it loads on this machine.
That fragility is the defect being fixed, and this script has to be able to run
the broken path in order to compare against it.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "export"))

from transformers import AutoTokenizer  # noqa: E402  - must precede torch/onnxruntime

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import MAX_LENGTH, probe_entries  # noqa: E402
from ledger.app import local_tokenizer  # noqa: E402
from ledger.model.dimensions import ANCHORS  # noqa: E402

TOKENIZER_DIR = ROOT / "artifacts" / "tokenizer"
TORCH_DIR = ROOT / "artifacts" / "torch"


def corpus() -> list[str]:
    anchors = [s for dim in ANCHORS.values() for pole in dim.values() for s in pole]
    return probe_entries() + anchors


def compare_encodings() -> dict:
    reference = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
    texts = corpus()

    mismatches = []
    for index, text in enumerate(texts):
        ref = reference([text], padding="max_length", truncation=True,
                        max_length=MAX_LENGTH, return_offsets_mapping=True,
                        return_tensors="np")
        ids, mask, offsets = local_tokenizer.encode(text, MAX_LENGTH)

        checks = {
            "input_ids": (list(ref["input_ids"][0]), ids),
            "attention_mask": (list(ref["attention_mask"][0]), mask),
            "offset_mapping": ([tuple(int(v) for v in pair)
                                for pair in ref["offset_mapping"][0]],
                               [tuple(int(v) for v in pair) for pair in offsets]),
        }
        for field, (expected, actual) in checks.items():
            expected = [int(v) if not isinstance(v, tuple) else v for v in expected]
            if list(expected) != list(actual):
                first = next(i for i, (a, b) in enumerate(zip(expected, actual)) if a != b)
                mismatches.append({
                    "text_index": index, "field": field, "first_differing_token": first,
                    "expected": str(expected[first]), "actual": str(actual[first]),
                    "text_preview": text[:60],
                })
    return {"texts_compared": len(texts), "mismatches": mismatches,
            "pass": not mismatches}


def compare_bias() -> dict:
    ckpt = torch.load(TORCH_DIR / "head.pt", map_location="cpu", weights_only=True)
    stored = ckpt["head_bias"].numpy().astype(np.float64)
    dimensions = list(ckpt["dimensions"])

    report = json.loads((TORCH_DIR / "build_report.json").read_text())
    calibration = report["anchor_calibration"]
    from_report_f64 = np.array([calibration[d]["offset"] for d in dimensions], np.float64)
    from_report_f32 = from_report_f64.astype(np.float32).astype(np.float64)

    return {
        "dimensions": dimensions,
        "max_abs_delta_before_float32_cast": float(np.abs(stored - from_report_f64).max()),
        "max_abs_delta_after_float32_cast": float(np.abs(stored - from_report_f32).max()),
        "bit_identical_after_cast": bool(np.array_equal(stored, from_report_f32)),
        "pass": bool(np.array_equal(stored, from_report_f32)),
        "note": ("head.pt stores a float32 cast of the same calibration offsets, so "
                 "reading them from build_report.json and casting to float32 "
                 "reproduces the shipped bias exactly and needs no torch."),
    }


def main() -> int:
    encodings = compare_encodings()
    bias = compare_bias()
    report = {
        "measured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "defect": "DEFECT-INC8-001",
        "question": ("Does the application's transformers-free tokenizer encode "
                     "identically to the path every prior measurement was run through?"),
        "protocol": "elementwise, no tolerance; identical to R7-2",
        "max_length": MAX_LENGTH,
        "encodings": encodings,
        "head_bias": bias,
        "verdict": "PASS" if (encodings["pass"] and bias["pass"]) else "FAIL",
    }
    out = ROOT / "artifacts" / "tokenizer_parity.json"
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
