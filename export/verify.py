"""Step 4: judge every candidate build against the ceilings in SIZE_BUDGET.md.

Every number here is measured in this process. The script exits non-zero when no
candidate build clears all five ceilings, so the pipeline can fail rather than
describe. CEIL-4 is not decided here - it is a WASM number and comes from
web/bench_wasm.mjs, whose JSON this script reads if it has been produced.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoTokenizer

from common import (ARTIFACTS, CEILINGS, MAX_LENGTH, ONNX_INT8, ORT_RUNTIME_FLOOR_BYTES,
                    TOKENIZER_DIR, TORCH_DIR, dir_bytes, probe_entries, write_json)
from quantize import FP32_SINGLE, ONNX_INT8_EMBED

WARMUP, TIMED = 5, 50
BUILDS = {"fp32": FP32_SINGLE, "int8_full": ONNX_INT8, "int8_embed": ONNX_INT8_EMBED}
CANDIDATES = ("int8_full", "int8_embed")   # fp32 is the reference, not a shipping candidate


def session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def pearson(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std()))


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def wasm_results() -> dict:
    out = {}
    for path in sorted((ARTIFACTS / "wasm").glob("bench_*.json")):
        rec = json.loads(path.read_text())
        out[rec["build"]] = rec
    return out


def main() -> int:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    ckpt = torch.load(TORCH_DIR / "head.pt", map_location="cpu", weights_only=True)
    bias = ckpt["head_bias"].numpy().astype(np.float64)
    dims = ckpt["dimensions"]

    entries = probe_entries()
    enc = tokenizer(entries, padding="max_length", truncation=True,
                    max_length=MAX_LENGTH, return_tensors="np")
    feeds = {"input_ids": enc["input_ids"].astype(np.int64),
             "attention_mask": enc["attention_mask"].astype(np.int64)}
    one = {k: v[:1] for k, v in feeds.items()}

    measured, identity, latency = {}, {}, {}
    for name, path in BUILDS.items():
        sess = session(path)
        logits, token_attr = sess.run(["logits", "token_attr"], feeds)
        logits = logits.astype(np.float64)
        measured[name] = sigmoid(logits)
        identity[name] = float(np.abs(logits - (token_attr.astype(np.float64).sum(1) + bias)).max())

        for _ in range(WARMUP):
            sess.run(None, one)
        samples = []
        for _ in range(TIMED):
            t = time.perf_counter()
            sess.run(None, one)
            samples.append((time.perf_counter() - t) * 1000.0)
        samples.sort()
        latency[name] = {"p50_ms": round(statistics.median(samples), 3),
                         "p95_ms": round(samples[int(0.95 * (len(samples) - 1))], 3),
                         "runs": TIMED}

    wasm = wasm_results()
    tok_bytes = dir_bytes(TOKENIZER_DIR)
    per_build = {}

    for name in CANDIDATES:
        per_dim, worst_r, worst_delta = {}, 1.0, 0.0
        for k, dim in enumerate(dims):
            r = pearson(measured["fp32"][:, k], measured[name][:, k])
            d = float(np.abs(measured["fp32"][:, k] - measured[name][:, k]).max())
            per_dim[dim] = {"pearson_r": round(r, 6), "max_abs_score_delta": round(d, 6)}
            worst_r, worst_delta = min(worst_r, r), max(worst_delta, d)

        size = BUILDS[name].stat().st_size
        cold = ORT_RUNTIME_FLOOR_BYTES + size + tok_bytes
        w = wasm.get(name)
        p95_wasm = w["p95_ms"] if w else None

        checks = {
            "CEIL_1_int8_model_bytes": (size, CEILINGS["CEIL_1_int8_model_bytes"],
                                        size <= CEILINGS["CEIL_1_int8_model_bytes"]),
            "CEIL_2_tokenizer_bytes": (tok_bytes, CEILINGS["CEIL_2_tokenizer_bytes"],
                                       tok_bytes <= CEILINGS["CEIL_2_tokenizer_bytes"]),
            "CEIL_3_cold_payload_bytes": (cold, CEILINGS["CEIL_3_cold_payload_bytes"],
                                          cold <= CEILINGS["CEIL_3_cold_payload_bytes"]),
            "CEIL_4_p95_latency_ms_wasm": (p95_wasm, CEILINGS["CEIL_4_p95_latency_ms"],
                                           None if p95_wasm is None else p95_wasm <= CEILINGS["CEIL_4_p95_latency_ms"]),
            "CEIL_5_min_pearson_r": (round(worst_r, 6), CEILINGS["CEIL_5_min_pearson_r"],
                                     worst_r >= CEILINGS["CEIL_5_min_pearson_r"]),
            "CEIL_5_max_abs_score_delta": (round(worst_delta, 6), CEILINGS["CEIL_5_max_abs_score_delta"],
                                           worst_delta <= CEILINGS["CEIL_5_max_abs_score_delta"]),
        }
        per_build[name] = {
            "bytes": size,
            "MiB": round(size / 1048576, 2),
            "cold_first_load_bytes": cold,
            "cold_first_load_MiB": round(cold / 1048576, 2),
            "vs_fp32_per_dimension": per_dim,
            "attribution_identity_max_residual": identity[name],
            "latency_native_ort_cpu_1thread": latency[name],
            "latency_wasm_1thread": {k: w[k] for k in ("p50_ms", "p95_ms", "session_load_ms")} if w else None,
            "wasm_vs_native_ort_max_abs_logit_diff": w.get("wasm_vs_native_ort_max_abs_logit_diff") if w else None,
            "ceiling_checks": {k: {"measured": m, "ceiling": c, "pass": p}
                               for k, (m, c, p) in checks.items()},
            "clears_every_ceiling": all(p is True for _, _, p in checks.values()),
        }

    shippable = [n for n, v in per_build.items() if v["clears_every_ceiling"]]
    report = {
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "machine": {"cpu": cpu_name(), "python": platform.python_version(),
                    "onnxruntime": ort.__version__, "os": platform.platform()},
        "probe_set": {"n": len(entries), "padded_to_tokens": MAX_LENGTH,
                      "source": "export/common.py:probe_entries - seeded recombination of the "
                                "anchor sentences in this repository; a numerical probe, not an "
                                "evaluation set. No accuracy claim is made from it."},
        "reference_build": {"name": "fp32", "bytes": FP32_SINGLE.stat().st_size,
                            "attribution_identity_max_residual": identity["fp32"],
                            "latency_native_ort_cpu_1thread": latency["fp32"],
                            "latency_wasm_1thread": ({k: wasm["fp32"][k] for k in ("p50_ms", "p95_ms")}
                                                     if "fp32" in wasm else None)},
        "tokenizer_bytes": tok_bytes,
        "ort_runtime_floor_bytes": ORT_RUNTIME_FLOOR_BYTES,
        "candidate_builds": per_build,
        "shippable_builds": shippable,
        "verdict": ("SHIPPABLE: " + ", ".join(shippable)) if shippable else
                   "NO BUILD CLEARS EVERY CEILING - see SIZE_BUDGET.md fallback rule",
    }
    write_json(ARTIFACTS / "verify_report.json", report)
    print(json.dumps(report, indent=1))
    return 0 if shippable else 1


if __name__ == "__main__":
    raise SystemExit(main())
