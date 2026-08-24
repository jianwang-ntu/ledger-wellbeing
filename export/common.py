"""Shared paths, the base-model pin, and the deterministic probe set."""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARTIFACTS = ROOT / "artifacts"
TORCH_DIR = ARTIFACTS / "torch"
ONNX_FP32 = ARTIFACTS / "onnx" / "ledger_scorer_fp32.onnx"
ONNX_INT8 = ARTIFACTS / "onnx" / "ledger_scorer_int8.onnx"
TOKENIZER_DIR = ARTIFACTS / "tokenizer"

# Pinned by revision, not by tag. See data/MANIFEST.md for the licence.
#
# Increment 6 swapped the body from all-MiniLM-L6-v2 to this one. The reason is
# increments 3-5, not a preference: the MiniLM body separates held-out anchors at
# macro AUC 0.504 (chance) and this one at 0.880, and increment 5 established that
# nothing at hidden <= 384 does both that and CEIL-1. The cost is that the
# artifact stopped fitting a browser download, which is why the delivery target
# below is `desktop`. See export/INCREMENT_6_PREREGISTRATION.md.
BASE_MODEL = "sentence-transformers/nli-distilroberta-base-v2"
BASE_REVISION = "cc35a0bfb6251228a6fb8c797bca5fef0ece3c1d"

#: Superseded pin, kept so the swap is legible in the file and not only in git.
PREVIOUS_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PREVIOUS_BASE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

MAX_LENGTH = 256          # tokens per journal entry; CEIL-4 is stated at this length
PROBE_SEED = 20260824
PROBE_N = 64

# Ceilings from export/SIZE_BUDGET.md, fixed before anything was measured.
CEILINGS = {
    "CEIL_1_int8_model_bytes": 32 * 1024 * 1024,
    "CEIL_2_tokenizer_bytes": 2 * 1024 * 1024,
    "CEIL_3_cold_payload_bytes": 64 * 1024 * 1024,
    "CEIL_4_p95_latency_ms": 500.0,
    "CEIL_5_min_pearson_r": 0.99,
    "CEIL_5_max_abs_score_delta": 0.02,
}

# Measured from web/node_modules/onnxruntime-web/dist at 1.23.0 (see SIZE_BUDGET.md).
# Retained on the desktop target because CEIL-3 is still *measured* there, and a
# measurement that silently changed its own definition would be worse than one
# that no longer gates.
ORT_RUNTIME_FLOOR_BYTES = 49_856 + 11_815_498

# ---------------------------------------------------------------------------
# Delivery target (plan.md R-4, exercised in build increment 6)
# ---------------------------------------------------------------------------
# R-4 was written at step D, before anything was measured: "If the web target
# fails, fall back to a local desktop app; the zero-egress claim survives, the
# 'in-browser' claim is dropped rather than fudged." Increment 5 measured that
# failure and increment 6 takes the fallback.
#
# A delivery target is a property of DISTRIBUTION. It changes which ceilings
# bind; it changes no ceiling's value. CEILINGS above is byte-identical to the
# version fixed on day 1 and tests/test_size_feasible_scorer.py asserts it.
DELIVERY_TARGET = "desktop"

#: Which ceilings gate a build, per target. Everything is measured on every
#: target regardless -- a ceiling that stops gating still gets reported, with a
#: `would_fail_web_target` flag, so a dropped claim stays visible.
ENFORCED_BY_TARGET = {
    "web": (
        "CEIL_1_int8_model_bytes",
        "CEIL_2_tokenizer_bytes",
        "CEIL_3_cold_payload_bytes",
        "CEIL_4_p95_latency_ms",
        "CEIL_5_min_pearson_r",
        "CEIL_5_max_abs_score_delta",
    ),
    # CEIL-1 and CEIL-3 bound the size of a first-visit HTTP download. A desktop
    # application is installed once from a release artifact, so neither is a
    # property of the thing being shipped any more. They remain measured.
    "desktop": (
        "CEIL_2_tokenizer_bytes",
        "CEIL_4_p95_latency_ms",
        "CEIL_5_min_pearson_r",
        "CEIL_5_max_abs_score_delta",
    ),
}

#: The runtime CEIL-4 is judged on, per target. CEIL-4 was written as
#: "single-threaded WASM" because that was the pessimistic browser case. A
#: desktop app does not execute WASM, so on that target the same 500 ms is judged
#: against native onnxruntime at 1 intra-op / 1 inter-op thread. Native is
#: FASTER than WASM, so this makes the ceiling EASIER: it is a relaxation of the
#: measurement basis and is logged as one in INCREMENT_6_PREREGISTRATION.md. The
#: WASM number is still measured where obtainable and reported alongside.
CEIL_4_RUNTIME_BY_TARGET = {"web": "wasm_1thread", "desktop": "native_ort_cpu_1thread"}


def enforced_ceilings(target: str = None) -> tuple:
    return ENFORCED_BY_TARGET[target or DELIVERY_TARGET]


def probe_entries() -> list[str]:
    """A fixed set of multi-sentence journal-style entries.

    Built by deterministically recombining the anchor sentences in
    ledger/model/dimensions.py, so the probe set is reproducible from the
    repository with no external data and no licence attached to it. It is a
    *numerical* probe for quantization parity and latency, not an evaluation
    set: no accuracy claim is made from it anywhere.
    """
    from ledger.model.dimensions import ANCHORS

    pool = [s for dim in ANCHORS.values() for pole in dim.values() for s in pole]
    rng = random.Random(PROBE_SEED)
    entries = []
    for _ in range(PROBE_N):
        k = rng.randint(3, 6)
        entries.append(" ".join(rng.sample(pool, k)))
    return entries


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=False) + "\n")
    return path
