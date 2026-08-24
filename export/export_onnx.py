"""Step 2: export LedgerScorer to ONNX (fp32), with dynamic batch and sequence axes."""

from __future__ import annotations

import json
import time

import torch
from transformers import AutoModel, AutoTokenizer

from common import MAX_LENGTH, ONNX_FP32, TOKENIZER_DIR, TORCH_DIR, sha256_file, write_json
from ledger.model import DIMENSIONS, LedgerScorer

OPSET = 17  # onnxruntime-web 1.23 supports opset <= 22; 17 keeps older browsers viable


def load_scorer() -> LedgerScorer:
    ckpt = torch.load(TORCH_DIR / "head.pt", map_location="cpu", weights_only=True)
    encoder = AutoModel.from_pretrained(TORCH_DIR / "encoder", attn_implementation="eager").eval()
    model = LedgerScorer(encoder, encoder.config.hidden_size, len(DIMENSIONS))
    with torch.no_grad():
        model.weight.copy_(ckpt["head_weight"])
        model.bias.copy_(ckpt["head_bias"])
    return model.eval()


def main() -> None:
    t0 = time.time()
    model = load_scorer()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)

    sample = tokenizer(
        ["A short entry used only to trace the graph."],
        padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt",
    )
    args = (sample["input_ids"], sample["attention_mask"])

    ONNX_FP32.parent.mkdir(parents=True, exist_ok=True)
    used = "dynamo"
    try:
        torch.onnx.export(
            model, args, str(ONNX_FP32),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits", "token_attr"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
                "token_attr": {0: "batch", 1: "sequence"},
            },
            opset_version=OPSET, dynamo=True,
        )
    except Exception as exc:                                   # pragma: no cover - fallback path
        print(f"dynamo export failed ({type(exc).__name__}: {exc}); falling back to torchscript")
        used = "torchscript"
        torch.onnx.export(
            model, args, str(ONNX_FP32),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits", "token_attr"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch"},
                "token_attr": {0: "batch", 1: "sequence"},
            },
            opset_version=OPSET, dynamo=False,
        )

    out = {
        "exporter": used,
        "opset": OPSET,
        "path": str(ONNX_FP32.relative_to(ONNX_FP32.parents[2])),
        "bytes": ONNX_FP32.stat().st_size,
        "sha256": sha256_file(ONNX_FP32),
        "max_length_traced": MAX_LENGTH,
        "dynamic_axes": ["batch", "sequence"],
        "elapsed_s": round(time.time() - t0, 2),
    }
    write_json(ONNX_FP32.parent / "export_fp32_report.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
