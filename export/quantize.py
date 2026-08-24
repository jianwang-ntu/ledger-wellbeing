"""Step 3: consolidate the fp32 graph and produce the candidate deployable builds.

Dynamic (weight-only) quantization is the right tool here: activations stay
float, so no calibration set is needed and no user text is ever required to build
a shipped model. onnxruntime-web executes MatMulInteger/DynamicQuantizeLinear in
the WASM backend, so export/bench_wasm.mjs exercises the same kernel path a
browser would.

Two builds are produced because they sit at different points on the
size/agreement curve and export/verify.py judges both against SIZE_BUDGET.md:

  * ``int8_full``  - every quantizable op. Smallest.
  * ``int8_embed`` - the embedding Gather only; every MatMul stays fp32.

export/quant_sensitivity.py is what established that those are the two points
worth keeping.
"""

from __future__ import annotations

import json
import time

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic

from common import ARTIFACTS, ONNX_FP32, ONNX_INT8, sha256_file, write_json

FP32_SINGLE = ARTIFACTS / "onnx" / "ledger_scorer_fp32_single.onnx"
ONNX_INT8_EMBED = ARTIFACTS / "onnx" / "ledger_scorer_int8embed.onnx"


def main() -> None:
    t0 = time.time()

    # One self-contained fp32 file, so the size comparison is like-for-like and
    # the browser never has to fetch a sidecar .onnx.data blob.
    model = onnx.load(str(ONNX_FP32), load_external_data=True)
    onnx.save_model(model, str(FP32_SINGLE), save_as_external_data=False)

    quantize_dynamic(
        model_input=str(FP32_SINGLE), model_output=str(ONNX_INT8),
        weight_type=QuantType.QInt8, per_channel=True, reduce_range=False,
        extra_options={"MatMulConstBOnly": True},
    )
    quantize_dynamic(
        model_input=str(FP32_SINGLE), model_output=str(ONNX_INT8_EMBED),
        weight_type=QuantType.QInt8, per_channel=True,
        op_types_to_quantize=["Gather"],
    )

    fp32_bytes = FP32_SINGLE.stat().st_size
    out = {
        "builds": {
            "fp32": {"path": FP32_SINGLE.name, "bytes": fp32_bytes,
                     "sha256": sha256_file(FP32_SINGLE), "scope": "reference build"},
            "int8_full": {"path": ONNX_INT8.name, "bytes": ONNX_INT8.stat().st_size,
                          "sha256": sha256_file(ONNX_INT8),
                          "scope": "all quantizable ops (MatMul const-B + embedding Gather)",
                          "compression_vs_fp32": round(fp32_bytes / ONNX_INT8.stat().st_size, 3)},
            "int8_embed": {"path": ONNX_INT8_EMBED.name, "bytes": ONNX_INT8_EMBED.stat().st_size,
                           "sha256": sha256_file(ONNX_INT8_EMBED),
                           "scope": "embedding Gather only; every MatMul left fp32",
                           "compression_vs_fp32": round(fp32_bytes / ONNX_INT8_EMBED.stat().st_size, 3)},
        },
        "method": "onnxruntime.quantization.quantize_dynamic, QInt8, per_channel=True",
        "activations": "left in float32 (dynamic quantization) - no calibration data used",
        "elapsed_s": round(time.time() - t0, 2),
    }
    write_json(ARTIFACTS / "onnx" / "quantize_report.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
