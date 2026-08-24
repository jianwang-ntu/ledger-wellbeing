#!/usr/bin/env bash
# The whole export pipeline, one command, from a clean checkout.
#
#   bash export/run_all.sh
#
# Exits non-zero if no candidate build clears every ceiling in SIZE_BUDGET.md.
# That is the intended behaviour: the pipeline is a test, not a description.
set -uo pipefail
cd "$(dirname "$0")/.."

PY="${LEDGER_PYTHON:-../.venv-export/bin/python}"
# The shared anaconda tree ships a pyarrow built against a newer libstdc++ than
# /usr/lib provides; transformers imports it transitively through sklearn.
export LD_LIBRARY_PATH="${LEDGER_LIBSTDCXX:-/data/wj/anaconda/lib}:${LD_LIBRARY_PATH:-}"

RUNS="../audit/runs"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUNS" artifacts/wasm

step () { echo; echo "=== $* ==="; }

step "1/6 build encoder + anchor head"      && "$PY" export/build_model.py       2>&1 | tee "$RUNS/export_1_build_$STAMP.txt"
step "2/6 export ONNX (fp32)"               && "$PY" export/export_onnx.py       2>&1 | tee "$RUNS/export_2_onnx_$STAMP.txt"
step "3/6 quantize"                         && "$PY" export/quantize.py          2>&1 | tee "$RUNS/export_3_quantize_$STAMP.txt"
step "4/6 quantization sensitivity"         && "$PY" export/quant_sensitivity.py 2>&1 | tee "$RUNS/export_4_sensitivity_$STAMP.txt"
step "5/6 WASM benchmark (CEIL-4)"          && "$PY" export/make_bench_input.py  2>&1 | tee "$RUNS/export_5_benchinput_$STAMP.txt"
for pair in "ledger_scorer_fp32_single.onnx:fp32" \
            "ledger_scorer_int8.onnx:int8_full" \
            "ledger_scorer_int8embed.onnx:int8_embed"; do
  f="${pair%%:*}"; n="${pair##*:}"
  node web/bench_wasm.mjs "artifacts/onnx/$f" "$n" | tee "artifacts/wasm/bench_$n.json"
done
cp artifacts/wasm/bench_*.json "$RUNS/" 2>/dev/null

step "6/6 verify against SIZE_BUDGET.md"
"$PY" export/verify.py 2>&1 | tee "$RUNS/export_6_verify_$STAMP.txt"
rc=${PIPESTATUS[0]}
echo
echo "verify.py exit code: $rc  (0 = a build clears every ceiling, 1 = none does)"
exit "$rc"
