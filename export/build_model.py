"""Step 1 of the export pipeline: fetch the pinned encoder, derive the anchor head.

Deterministic and offline-repeatable once the base model is in the HF cache.
Writes artifacts/torch/ and artifacts/tokenizer/.
"""

from __future__ import annotations

import json
import time

import torch
from transformers import AutoModel, AutoTokenizer

from common import (BASE_MODEL, BASE_REVISION, TOKENIZER_DIR, TORCH_DIR, dir_bytes, write_json)
from ledger.model import DIMENSIONS, HEAD_VERSION, build_anchor_head


def main() -> None:
    t0 = time.time()
    torch.manual_seed(0)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    encoder = AutoModel.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, attn_implementation="eager"
    ).eval()

    weight, bias, report = build_anchor_head(encoder, tokenizer)

    TORCH_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_weight": weight,
            "head_bias": bias,
            "head_version": HEAD_VERSION,
            "dimensions": list(DIMENSIONS),
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
        },
        TORCH_DIR / "head.pt",
    )
    encoder.save_pretrained(TORCH_DIR / "encoder")
    tokenizer.save_pretrained(TOKENIZER_DIR)

    n_params = sum(p.numel() for p in encoder.parameters())
    out = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "encoder_parameters": n_params,
        "hidden_size": encoder.config.hidden_size,
        "num_hidden_layers": encoder.config.num_hidden_layers,
        "head_version": HEAD_VERSION,
        "head_is_trained": False,
        "head_note": (
            "Zero-shot anchor head. No gradient step has been taken on any corpus. "
            "Nothing produced from this checkpoint may be described as fine-tuned."
        ),
        "dimensions": list(DIMENSIONS),
        "anchor_calibration": report,
        "tokenizer_bytes": dir_bytes(TOKENIZER_DIR),
        "elapsed_s": round(time.time() - t0, 2),
    }
    write_json(TORCH_DIR / "build_report.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "anchor_calibration"}, indent=1))


if __name__ == "__main__":
    main()
