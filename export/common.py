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
BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BASE_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

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
ORT_RUNTIME_FLOOR_BYTES = 49_856 + 11_815_498


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
