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

from common import (ARTIFACTS, BASE_MODEL, BASE_REVISION, CEIL_4_RUNTIME_BY_TARGET, CEILINGS,
                    DELIVERY_TARGET, ENFORCED_BY_TARGET, MAX_LENGTH, ONNX_INT8,
                    ORT_RUNTIME_FLOOR_BYTES, TOKENIZER_DIR, TORCH_DIR, dir_bytes,
                    enforced_ceilings, probe_entries, sha256_file, write_json)
from quantize import FP32_SINGLE, ONNX_INT8_EMBED

WARMUP, TIMED = 5, 50
BUILDS = {"fp32": FP32_SINGLE, "int8_full": ONNX_INT8, "int8_embed": ONNX_INT8_EMBED}

# fp32 was the reference and not a shipping candidate while CEIL-1 gated. On the
# desktop target it becomes a legitimate candidate, and SIZE_BUDGET.md's own
# fallback names it: "a CEIL-5 failure means the int8 build is not shipped and
# the fp32 build is measured against CEIL-1 instead". It is judged last, so it
# can only win when both int8 builds have failed CEIL-5.
CANDIDATES = (("int8_full", "int8_embed") if DELIVERY_TARGET == "web"
              else ("int8_full", "int8_embed", "fp32"))

#: pre-registered in export/INCREMENT_6_PREREGISTRATION.md as rule R6-1
ADDITIVITY_MAX_RESIDUAL = 1e-4


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
    """Read the WASM benches, and refuse any that does not identify its own model.

    DEFECT-INC6-001. Before this check, `bench_*.json` carried no model identity,
    so a bench produced against a previous encoder was read as if it described the
    current build - and the previous encoder is exactly what increment 6 replaced.
    A bench whose sha256 does not match the file on disk is dropped and recorded
    as STALE rather than quietly believed.
    """
    out, stale = {}, {}
    for path in sorted((ARTIFACTS / "wasm").glob("bench_*.json")):
        rec = json.loads(path.read_text())
        name = rec["build"]
        target = BUILDS.get(name)
        actual = sha256_file(target) if target and target.exists() else None
        claimed = rec.get("model_sha256")
        if claimed is None:
            stale[name] = "bench predates DEFECT-INC6-001 and carries no model_sha256"
        elif actual is not None and claimed != actual:
            stale[name] = f"bench sha {claimed[:12]} != build on disk {actual[:12]}"
        else:
            out[name] = rec
    return out, stale


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

    wasm, wasm_stale = wasm_results()
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

        p95_native = latency[name]["p95_ms"]
        ceil4_runtime = CEIL_4_RUNTIME_BY_TARGET[DELIVERY_TARGET]
        p95_gating = p95_native if ceil4_runtime == "native_ort_cpu_1thread" else p95_wasm

        checks = {
            "CEIL_1_int8_model_bytes": (size, CEILINGS["CEIL_1_int8_model_bytes"],
                                        size <= CEILINGS["CEIL_1_int8_model_bytes"]),
            "CEIL_2_tokenizer_bytes": (tok_bytes, CEILINGS["CEIL_2_tokenizer_bytes"],
                                       tok_bytes <= CEILINGS["CEIL_2_tokenizer_bytes"]),
            "CEIL_3_cold_payload_bytes": (cold, CEILINGS["CEIL_3_cold_payload_bytes"],
                                          cold <= CEILINGS["CEIL_3_cold_payload_bytes"]),
            "CEIL_4_p95_latency_ms": (p95_gating, CEILINGS["CEIL_4_p95_latency_ms"],
                                      None if p95_gating is None else p95_gating <= CEILINGS["CEIL_4_p95_latency_ms"]),
            "CEIL_5_min_pearson_r": (round(worst_r, 6), CEILINGS["CEIL_5_min_pearson_r"],
                                     worst_r >= CEILINGS["CEIL_5_min_pearson_r"]),
            "CEIL_5_max_abs_score_delta": (round(worst_delta, 6), CEILINGS["CEIL_5_max_abs_score_delta"],
                                           worst_delta <= CEILINGS["CEIL_5_max_abs_score_delta"]),
        }
        enforced = enforced_ceilings()

        # A ceiling that stopped gating is still measured and still reported. The
        # web verdict is computed in full alongside the desktop one so the claim
        # this increment dropped stays visible in the artifact rather than
        # disappearing from it.
        web_checks = {k: v for k, v in checks.items() if k in ENFORCED_BY_TARGET["web"]}
        web_fail = sorted(k for k, (_, _, ok) in web_checks.items()
                          if k != "CEIL_4_p95_latency_ms" and ok is False)

        per_build[name] = {
            "bytes": size,
            "MiB": round(size / 1048576, 2),
            "cold_first_load_bytes": cold,
            "cold_first_load_MiB": round(cold / 1048576, 2),
            "vs_fp32_per_dimension": per_dim,
            "attribution_identity_max_residual": identity[name],
            "attribution_identity_holds": identity[name] <= ADDITIVITY_MAX_RESIDUAL,
            "latency_native_ort_cpu_1thread": latency[name],
            "latency_wasm_1thread": {k: w[k] for k in ("p50_ms", "p95_ms", "session_load_ms")} if w
                                    else "NOT_MEASURED",
            "wasm_vs_native_ort_max_abs_logit_diff": w.get("wasm_vs_native_ort_max_abs_logit_diff") if w else None,
            "ceiling_checks": {k: {"measured": m, "ceiling": c, "pass": p,
                                   "enforced_on_this_target": k in enforced}
                               for k, (m, c, p) in checks.items()},
            "ceil_4_judged_on": ceil4_runtime,
            "would_fail_web_target_on": web_fail,
            "clears_every_enforced_ceiling": all(checks[k][2] is True for k in enforced),
        }

    shippable = [n for n in CANDIDATES
                 if per_build[n]["clears_every_enforced_ceiling"]
                 and per_build[n]["attribution_identity_holds"]]
    selected = shippable[0] if shippable else None      # CANDIDATES order = smallest first
    report = {
        "measured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "delivery_target": DELIVERY_TARGET,
        "delivery_target_note": (
            "plan.md R-4, exercised in build increment 6. CEIL-1 and CEIL-3 bound an HTTP "
            "first-load and are NOT enforced on a desktop target; they are still measured "
            "and every build carries would_fail_web_target_on. CEIL-4 is judged on "
            + CEIL_4_RUNTIME_BY_TARGET[DELIVERY_TARGET] + " rather than WASM, which is a "
            "relaxation of the measurement basis and is logged as one in "
            "export/INCREMENT_6_PREREGISTRATION.md. No ceiling VALUE was edited."),
        "enforced_ceilings": list(enforced_ceilings()),
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
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
                                                     if "fp32" in wasm else "NOT_MEASURED")},
        "tokenizer_bytes": tok_bytes,
        "wasm_benches_rejected_as_stale": wasm_stale,
        "ort_runtime_floor_bytes": ORT_RUNTIME_FLOOR_BYTES,
        "candidate_builds": per_build,
        "shippable_builds": shippable,
        "selected_build": selected,
        "selection_rule": ("R6-2, pre-registered: ship the smallest build that clears every "
                           "ceiling enforced on this target AND holds the additivity identity. "
                           "CANDIDATES is ordered smallest-first, so fp32 can only win once both "
                           "int8 builds have failed."),
        "verdict": (("SHIPPABLE on target=" + DELIVERY_TARGET + ": " + ", ".join(shippable))
                    if shippable else
                    "NO BUILD CLEARS EVERY ENFORCED CEILING - see SIZE_BUDGET.md fallback rule"),
    }
    write_json(ARTIFACTS / "verify_report.json", report)
    print(json.dumps(report, indent=1))
    return 0 if shippable else 1


if __name__ == "__main__":
    raise SystemExit(main())
