// CEIL-4: p95 inference latency for a 256-token entry, single-threaded WASM.
//
// This runs onnxruntime-web's WASM backend under node. It is the same wasm
// binary and the same kernels a browser loads; what it does not include is the
// browser's own overhead (fetch, Cache API, main-thread contention). That
// difference is stated rather than papered over - see the caveat field in the
// emitted report.
//
// Usage: node web/bench_wasm.mjs <model.onnx> <build-name>

import * as ort from 'onnxruntime-web';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '..');   // project/ - this file lives in project/web/
const WARMUP = 5, TIMED = 30;

const modelPath = process.argv[2];
const buildName = process.argv[3] ?? path.basename(modelPath);

ort.env.wasm.wasmPaths = path.join(HERE, 'node_modules', 'onnxruntime-web', 'dist') + path.sep;
ort.env.wasm.numThreads = 1;   // pessimistic: no crossOriginIsolated, no COOP/COEP
ort.env.wasm.simd = true;
ort.env.logLevel = 'error';

const bench = JSON.parse(fs.readFileSync(path.join(ROOT, 'artifacts', 'bench_input.json'), 'utf8'));
const seq = bench.sequence_length;
const feeds = {
  input_ids: new ort.Tensor('int64', BigInt64Array.from(bench.input_ids.map(BigInt)), [1, seq]),
  attention_mask: new ort.Tensor('int64', BigInt64Array.from(bench.attention_mask.map(BigInt)), [1, seq]),
};

const loadStart = performance.now();
const session = await ort.InferenceSession.create(modelPath, { executionProviders: ['wasm'] });
const loadMs = performance.now() - loadStart;

for (let i = 0; i < WARMUP; i++) await session.run(feeds);

const samples = [];
let last;
for (let i = 0; i < TIMED; i++) {
  const t = performance.now();
  last = await session.run(feeds);
  samples.push(performance.now() - t);
}
samples.sort((a, b) => a - b);
const pct = (p) => samples[Math.round(p * (samples.length - 1))];

// Cross-runtime agreement: does WASM produce what native onnxruntime produced?
const wasmLogits = Array.from(last.logits.data).map(Number);
const nativeLogits = bench.expected_logits_native_ort[buildName] ?? null;
const maxAbsDiff = nativeLogits
  ? Math.max(...wasmLogits.map((v, i) => Math.abs(v - nativeLogits[i])))
  : null;

// The identity the whole explainability claim rests on, re-checked in the
// runtime that will actually serve users.
const attr = last.token_attr;
const [, T, K] = attr.dims;
const summed = new Array(K).fill(0);
for (let t = 0; t < T; t++) for (let k = 0; k < K; k++) summed[k] += Number(attr.data[t * K + k]);
const identityResidual = Math.max(
  ...summed.map((s, k) => Math.abs(wasmLogits[k] - (s + bench.head_bias[k]))));

// DEFECT-INC6-001: a bench report carried no identity of the model it timed, so
// export/verify.py read a stale bench_*.json from a previous encoder as if it
// described the current build. The sha is emitted here and checked there.
const modelSha = crypto.createHash('sha256').update(fs.readFileSync(modelPath)).digest('hex');

console.log(JSON.stringify({
  build: buildName,
  model_path: path.relative(ROOT, modelPath),
  model_sha256: modelSha,
  runtime: 'onnxruntime-web WASM (node host)',
  onnxruntime_web_version: JSON.parse(fs.readFileSync(
    path.join(HERE, 'node_modules', 'onnxruntime-web', 'package.json'), 'utf8')).version,
  node_version: process.version,
  threads: ort.env.wasm.numThreads,
  simd: ort.env.wasm.simd,
  sequence_length: seq,
  real_tokens: bench.real_tokens,
  session_load_ms: Number(loadMs.toFixed(2)),
  runs: TIMED,
  p50_ms: Number(pct(0.5).toFixed(3)),
  p95_ms: Number(pct(0.95).toFixed(3)),
  min_ms: Number(samples[0].toFixed(3)),
  max_ms: Number(samples[samples.length - 1].toFixed(3)),
  wasm_vs_native_ort_max_abs_logit_diff: maxAbsDiff === null ? null : Number(maxAbsDiff.toExponential(4)),
  logits_wasm: wasmLogits.map((v) => Number(v.toFixed(6))),
  attribution_sum_without_bias: summed.map((v) => Number(v.toFixed(6))),
  attribution_identity_residual: Number(identityResidual.toExponential(4)),
  attribution_identity_note: 'max_k | logit_k - (sum_i token_attr[i,k] + bias_k) |, measured in the WASM runtime',
  caveat: 'Node host, not a browser. Same wasm binary and kernels; excludes browser fetch, Cache API and main-thread contention.',
}, null, 1));
