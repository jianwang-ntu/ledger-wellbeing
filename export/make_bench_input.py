"""Freeze one tokenized 256-token entry plus its native-ORT logits.

export/bench_wasm.mjs replays exactly this input under onnxruntime-web, so the
WASM run is comparable to the native run rather than being a separate experiment,
and any divergence between the two runtimes shows up as a number.
"""

from __future__ import annotations

import json

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from common import ARTIFACTS, MAX_LENGTH, ONNX_INT8, TOKENIZER_DIR, probe_entries, write_json
from quantize import FP32_SINGLE, ONNX_INT8_EMBED


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    entry = probe_entries()[0]
    enc = tokenizer([entry], padding="max_length", truncation=True,
                    max_length=MAX_LENGTH, return_tensors="np")
    feeds = {"input_ids": enc["input_ids"].astype(np.int64),
             "attention_mask": enc["attention_mask"].astype(np.int64)}

    expected = {}
    for name, path in (("fp32", FP32_SINGLE), ("int8_full", ONNX_INT8), ("int8_embed", ONNX_INT8_EMBED)):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        expected[name] = sess.run(["logits"], feeds)[0][0].astype(float).tolist()

    import torch
    from common import TORCH_DIR
    head_bias = torch.load(TORCH_DIR / "head.pt", map_location="cpu",
                           weights_only=True)["head_bias"].tolist()

    write_json(ARTIFACTS / "bench_input.json", {
        "entry_text": entry,
        "head_bias": head_bias,
        "sequence_length": MAX_LENGTH,
        "real_tokens": int(feeds["attention_mask"].sum()),
        "input_ids": feeds["input_ids"][0].tolist(),
        "attention_mask": feeds["attention_mask"][0].tolist(),
        "expected_logits_native_ort": expected,
    })
    print(json.dumps({"real_tokens": int(feeds["attention_mask"].sum()),
                      "padded_to": MAX_LENGTH, "builds": list(expected)}, indent=1))


if __name__ == "__main__":
    main()
