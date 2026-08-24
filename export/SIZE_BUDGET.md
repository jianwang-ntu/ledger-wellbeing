# SIZE_BUDGET.md — the in-browser ceiling

Closes plan.md risk **R-4** ("In-browser inference is a real engineering risk —
model size, WASM/WebGPU support, cold-start latency. *Size ceiling fixed on day 1
and the export pipeline built before the UI.*").

These ceilings were **fixed before the model was exported or measured**, so that
the measurement in `../audit/runs/export_verify_*.json` is a test the pipeline
can fail, not a description of whatever came out. Written 2026-08-24, build
increment 2.

## The one number we do not control

`onnxruntime-web` is the runtime. Its cold payload is fixed by the package, not
by us. Measured from `web/node_modules/onnxruntime-web/dist` at version
`1.23.0`:

| Artifact | Bytes | Note |
|---|---:|---|
| `ort.wasm.min.mjs` | 49,856 | wasm-only ESM entry point (not the `ort.all` bundle) |
| `ort-wasm-simd-threaded.wasm` | 11,815,498 | SIMD + threads, **non-JSEP** |
| **Runtime floor** | **11,865,354** | **11.32 MiB** |

The JSEP build (`ort-wasm-simd-threaded.jsep.wasm`, 23,696,346 B) doubles the
floor to buy WebGPU. A 22M-parameter encoder over a 256-token journal entry does
not need a GPU, so the plain SIMD+threads build is chosen and WebGPU is dropped.
That decision costs 11.3 MiB of budget we would otherwise have spent for nothing.

## The ceilings

| ID | Ceiling | Value | Why this number |
|---|---|---:|---|
| **CEIL-1** | Quantized model weights (`.onnx`, int8) | **≤ 32 MiB** | Leaves ≥ 20 MiB of headroom under CEIL-3 after the fixed runtime floor, so the tokenizer, the app shell and a future larger head all fit without reopening this decision. |
| **CEIL-2** | Tokenizer + vocabulary | **≤ 2 MiB** | A WordPiece vocab of 30k entries is ~230 KiB of text; 2 MiB is generous and forces a rejection if someone swaps in a 250k-entry multilingual vocab without saying so. |
| **CEIL-3** | Total cold first-load payload | **≤ 64 MiB** | At a deliberately pessimistic 10 Mbit/s effective downlink this is ~54 s once, and the Cache API makes every later visit a local read. At 50 Mbit/s it is ~11 s. Stated as an assumption, not a measurement: we have not measured a user's connection and will not claim to have. |
| **CEIL-4** | p95 inference latency, 256-token entry, single-threaded WASM | **≤ 500 ms** | A journal entry is scored on submit, not per keystroke. 500 ms is under the 1 s limit at which a user's flow of thought stays uninterrupted (Nielsen, *Usability Engineering*, 1993, §5.5). Single-threaded is the pessimistic case: `crossOriginIsolated` is false without COOP/COEP headers, and a static host may not send them. |
| **CEIL-5** | int8-vs-fp32 agreement, per dimension, over the fixed probe set | **Pearson r ≥ 0.99 AND max abs difference ≤ 0.02 on the 0–1 normalised score** | Quantization that moves a displayed score by more than 2 points out of 100 would change what a clinician reads. This is the tolerance at which the int8 build may be substituted for the fp32 build without qualification. |

## What happens if a ceiling fails

Per plan.md R-4: *"If the web target fails, fall back to a local desktop app; the
zero-egress claim survives, the 'in-browser' claim is dropped rather than
fudged."* A CEIL-1/2/3 failure means a smaller encoder. A CEIL-4 failure means
shorter chunking or a smaller encoder. A **CEIL-5 failure means the int8 build is
not shipped** and the fp32 build is measured against CEIL-1 instead — the
accuracy claim is not weakened to keep the size claim.

No ceiling is to be edited to match a measurement. If one is genuinely wrong,
it is superseded in a new row with the reason, and the old row stays.
