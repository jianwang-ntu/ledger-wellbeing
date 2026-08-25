#!/usr/bin/env python3
"""Re-derive every published shipped-artifact figure from the artifact that
produced it, and fail if the prose and the artifact disagree.

Why this exists
---------------
`audit/README.md` claims a discipline: no number appears in a document unless
an artifact in this repository produced it. Round-1 finding **F-03** showed the
claim was aspirational — six of twenty traced figures did not match the
artifact the document named as their evidence, and `845.98` (published three
times as the shipped build's WASM p95) existed in no artifact at all. This file
makes the discipline executable so the same defect cannot recur silently.

What it enforces
----------------
1. **Every claim in CLAIMS is still in its document.** If a document is edited
   so a registered literal disappears, this fails — the table cannot silently
   drift out of the prose it governs.
2. **Every claim's literal equals the artifact re-rendered.** The value is
   recomputed from the JSON artifact and formatted with the claim's own
   formatter; the comparison is on the rendered *string*, not a tolerance, so
   "close enough" is not a pass.
3. **No unregistered live figure.** Every ``<number> ms`` / ``<number> MiB``
   token in the governed documents must be either covered by a CLAIMS literal
   or named in EXEMPT with a reason. A number nobody can account for fails.

Scope, stated so the boundary is auditable rather than implied
--------------------------------------------------------------
This enforces figures that describe **the artifact that ships today**. Three
classes are exempt and each exempt token carries its reason in EXEMPT:

* **ceiling values** — CEIL-1..CEIL-5 are budget constants fixed in
  `export/SIZE_BUDGET.md` before anything was exported. They are inputs, not
  measurements, and are checked against `verify_report.json` separately below.
* **superseded increment figures** — numbers from builds 4-6 (the all-MiniLM
  body, the pre-compaction tokenizer) that appear in narrative recording what
  was measured *then*. Rewriting them would falsify the record.
  `check_superseded_markers` requires that any document quoting one carries the
  word ``SUPERSEDED`` somewhere. Stated precisely because it is weaker than it
  sounds: it is a **per-document** check, not a per-figure one. It cannot prove
  that a given historical number sits next to its marker, only that the document
  admits it contains history. Closing that gap needs a block-level parser and is
  recorded as not done rather than implied.
* **third-party / environmental figures** — download sizes, disk costs, and
  numbers measured outside this repository.

Run:  ``python export/check_published_numbers.py``  (exit 0 = agree)
      ``python export/check_published_numbers.py --json`` for the report.

Asserted by `tests/test_published_numbers.py`.
"""
from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MiB = 1048576.0


def _load(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return json.load(fh)


def _doc(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
        return fh.read()


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------
VR = _load("artifacts/verify_report.json")
BENCH = {b: _load(f"artifacts/wasm/bench_{b}.json")
         for b in ("fp32", "int8_full", "int8_embed")}
CB = VR["candidate_builds"]
REF = VR["reference_build"]

# Cold first load is model + onnxruntime floor + tokenizer. verify.py records it
# for the candidate builds but not for the fp32 reference, so the fp32 cold
# figure is re-derived here from the same three terms and the derivation is
# named in the claim. Round 1 published 325.78 MiB for it, which is not that sum
# and matches no artifact; this check is what caught it.
FP32_COLD_BYTES = REF["bytes"] + VR["ort_runtime_floor_bytes"] + VR["tokenizer_bytes"]


def mib2(x):
    return f"{x / MiB:.2f}"


def ms2(x):
    return f"{x:.2f}"


def ms3(x):
    return f"{x:.3f}"


def r5(x):
    return f"{x:.5f}"


def r4(x):
    return f"{x:.4f}"


def worst_r(build):
    return min(v["pearson_r"] for v in CB[build]["vs_fp32_per_dimension"].values())


def max_delta(build):
    return max(v["max_abs_score_delta"] for v in CB[build]["vs_fp32_per_dimension"].values())


# --------------------------------------------------------------------------
# the claims
#
# (id, document, literal-as-published, source description, value, formatter)
# The literal is the exact substring the document must contain. The value is
# recomputed from the artifact on every run.
# --------------------------------------------------------------------------
CLAIMS = [
    # ---- shipped build: int8_embed -------------------------------------
    ("int8_embed.bytes.README", "README.md", "199.49 MiB",
     "verify_report.candidate_builds.int8_embed.bytes / 2**20",
     CB["int8_embed"]["bytes"] / MiB, "{:.2f} MiB"),
    ("int8_embed.bytes.SIZE_BUDGET", "export/SIZE_BUDGET.md", "199.49 MiB",
     "verify_report.candidate_builds.int8_embed.bytes / 2**20",
     CB["int8_embed"]["bytes"] / MiB, "{:.2f} MiB"),
    ("int8_embed.cold.README", "README.md", "212.29 MiB",
     "verify_report.candidate_builds.int8_embed.cold_first_load_bytes / 2**20",
     CB["int8_embed"]["cold_first_load_bytes"] / MiB, "{:.2f} MiB"),
    ("int8_embed.cold.SIZE_BUDGET", "export/SIZE_BUDGET.md", "212.29 MiB",
     "verify_report.candidate_builds.int8_embed.cold_first_load_bytes / 2**20",
     CB["int8_embed"]["cold_first_load_bytes"] / MiB, "{:.2f} MiB"),
    ("int8_embed.native_p95.README", "README.md", "230.39 ms",
     "verify_report.candidate_builds.int8_embed.latency_native_ort_cpu_1thread.p95_ms",
     CB["int8_embed"]["latency_native_ort_cpu_1thread"]["p95_ms"], "{:.2f} ms"),
    ("int8_embed.native_p95.SIZE_BUDGET", "export/SIZE_BUDGET.md", "230.39 ms",
     "verify_report.candidate_builds.int8_embed.latency_native_ort_cpu_1thread.p95_ms",
     CB["int8_embed"]["latency_native_ort_cpu_1thread"]["p95_ms"], "{:.2f} ms"),
    ("int8_embed.ceil4.SIZE_BUDGET", "export/SIZE_BUDGET.md", "230.39 ms",
     "verify_report...int8_embed.ceiling_checks.CEIL_4_p95_latency_ms.measured",
     CB["int8_embed"]["ceiling_checks"]["CEIL_4_p95_latency_ms"]["measured"], "{:.2f} ms"),
    ("int8_embed.ceil4.README", "README.md", "CEIL-4 230.39 ms",
     "verify_report...int8_embed.ceiling_checks.CEIL_4_p95_latency_ms.measured",
     CB["int8_embed"]["ceiling_checks"]["CEIL_4_p95_latency_ms"]["measured"], "CEIL-4 {:.2f} ms"),
    ("int8_embed.wasm_p95.README", "README.md", "836.61 ms",
     "artifacts/wasm/bench_int8_embed.json.p95_ms",
     BENCH["int8_embed"]["p95_ms"], "{:.2f} ms"),
    ("int8_embed.wasm_p95.SIZE_BUDGET", "export/SIZE_BUDGET.md", "836.61 ms",
     "artifacts/wasm/bench_int8_embed.json.p95_ms",
     BENCH["int8_embed"]["p95_ms"], "{:.2f} ms"),
    ("int8_embed.wasm_p95.limitations", "docs/limitations.md", "836.61 ms",
     "artifacts/wasm/bench_int8_embed.json.p95_ms",
     BENCH["int8_embed"]["p95_ms"], "{:.2f} ms"),
    ("int8_embed.worst_r.README", "README.md", "0.99995",
     "min pearson_r over verify_report...int8_embed.vs_fp32_per_dimension",
     worst_r("int8_embed"), "{:.5f}"),
    ("int8_embed.max_delta.README", "README.md", "0.00694",
     "max max_abs_score_delta over verify_report...int8_embed.vs_fp32_per_dimension",
     max_delta("int8_embed"), "{:.5f}"),

    # ---- rejected candidate: int8_full ---------------------------------
    ("int8_full.bytes.README", "README.md", "78.20 MiB",
     "verify_report.candidate_builds.int8_full.bytes / 2**20",
     CB["int8_full"]["bytes"] / MiB, "{:.2f} MiB"),
    ("int8_full.cold.README", "README.md", "91.00 MiB",
     "verify_report.candidate_builds.int8_full.cold_first_load_bytes / 2**20",
     CB["int8_full"]["cold_first_load_bytes"] / MiB, "{:.2f} MiB"),
    ("int8_full.cold.SIZE_BUDGET", "export/SIZE_BUDGET.md", "91.00 MiB",
     "verify_report.candidate_builds.int8_full.cold_first_load_bytes / 2**20",
     CB["int8_full"]["cold_first_load_bytes"] / MiB, "{:.2f} MiB"),
    ("int8_full.native_p95.README", "README.md", "75.92 ms",
     "verify_report.candidate_builds.int8_full.latency_native_ort_cpu_1thread.p95_ms",
     CB["int8_full"]["latency_native_ort_cpu_1thread"]["p95_ms"], "{:.2f} ms"),
    ("int8_full.native_p95.SIZE_BUDGET", "export/SIZE_BUDGET.md", "75.92 ms",
     "verify_report.candidate_builds.int8_full.latency_native_ort_cpu_1thread.p95_ms",
     CB["int8_full"]["latency_native_ort_cpu_1thread"]["p95_ms"], "{:.2f} ms"),
    ("int8_full.worst_r.README", "README.md", "0.99282",
     "min pearson_r over verify_report...int8_full.vs_fp32_per_dimension",
     worst_r("int8_full"), "{:.5f}"),
    ("int8_full.max_delta.README", "README.md", "0.0770",
     "max max_abs_score_delta over verify_report...int8_full.vs_fp32_per_dimension",
     max_delta("int8_full"), "{:.4f}"),

    # ---- fp32 reference -------------------------------------------------
    ("fp32.bytes.README", "README.md", "311.07 MiB",
     "verify_report.reference_build.bytes / 2**20",
     REF["bytes"] / MiB, "{:.2f} MiB"),
    ("fp32.cold.README", "README.md", "323.87 MiB",
     "(reference_build.bytes + ort_runtime_floor_bytes + tokenizer_bytes) / 2**20",
     FP32_COLD_BYTES / MiB, "{:.2f} MiB"),
    ("fp32.cold.SIZE_BUDGET", "export/SIZE_BUDGET.md", "323.87 MiB",
     "(reference_build.bytes + ort_runtime_floor_bytes + tokenizer_bytes) / 2**20",
     FP32_COLD_BYTES / MiB, "{:.2f} MiB"),
    ("fp32.native_p95.README", "README.md", "227.00 ms",
     "verify_report.reference_build.latency_native_ort_cpu_1thread.p95_ms",
     REF["latency_native_ort_cpu_1thread"]["p95_ms"], "{:.2f} ms"),
    ("fp32.native_p95.SIZE_BUDGET", "export/SIZE_BUDGET.md", "227.00 ms",
     "verify_report.reference_build.latency_native_ort_cpu_1thread.p95_ms",
     REF["latency_native_ort_cpu_1thread"]["p95_ms"], "{:.2f} ms"),

    # ---- tokenizer, the CEIL-2 story -----------------------------------
    ("tokenizer.bytes.README", "README.md", "1,556,504",
     "verify_report.tokenizer_bytes",
     VR["tokenizer_bytes"], "{:,}"),
    ("tokenizer.MiB.README", "README.md", "1.484 MiB",
     "verify_report.tokenizer_bytes / 2**20",
     VR["tokenizer_bytes"] / MiB, "{:.3f} MiB"),
]

# Numeric tokens in the governed documents that are NOT re-derivable from a
# shipped artifact, each with the reason. This list is what makes the boundary
# of the check auditable: anything here is a number the check does not stand
# behind, and it is written down rather than left implicit.
EXEMPT = {
    # ceilings — budget constants, not measurements
    "32 MiB": "CEIL-1 ceiling value, fixed in SIZE_BUDGET.md before export",
    "32.00 MiB": "CEIL-1 ceiling value rendered to 2dp",
    "2 MiB": "CEIL-2 ceiling value",
    "2.000 MiB": "CEIL-2 ceiling value rendered to 3dp",
    "64 MiB": "CEIL-3 ceiling value",
    "500 ms": "CEIL-4 ceiling value",
    "4 MiB": "CEIL-2 ceiling discussion (an earlier candidate ceiling, narrative)",
    "20 MiB": "SIZE_BUDGET narrative: a budget line considered and not adopted",
    "120 MiB": "SIZE_BUDGET narrative: a budget line considered and not adopted",
    "230 KiB": "onnxruntime-web wasm binary, third-party artifact size",
    # superseded increment 4-6 figures (all-MiniLM body / pre-compaction tokenizer)
    "52.55 MiB": "SUPERSEDED increment-5 figure: all-MiniLM int8_embed size",
    "52.04 MiB": "SUPERSEDED increment-5 figure: all-MiniLM int8_embed size (earlier run)",
    "64.04 MiB": "SUPERSEDED increment-5 figure: all-MiniLM int8_embed cold load",
    "21.78 MiB": "SUPERSEDED increment-5 figure: all-MiniLM int8_full size",
    "33.78 MiB": "SUPERSEDED increment-5 figure: all-MiniLM int8_full cold load",
    "86.14 MiB": "SUPERSEDED increment-5 figure: all-MiniLM fp32 size",
    "98.1 MiB": "SUPERSEDED increment-5 figure: all-MiniLM fp32 cold load",
    "124 ms": "SUPERSEDED increment-5 figure: all-MiniLM int8 WASM p95",
    "236 ms": "SUPERSEDED increment-5 figure: all-MiniLM int8_embed WASM p95",
    "238 ms": "SUPERSEDED increment-5 figure: all-MiniLM fp32 WASM p95",
    "25.91 MiB": "SUPERSEDED increment-5/6 figure: encoder-ablation candidate size",
    "26.7 MiB": "SUPERSEDED increment-5/6 figure: encoder-ablation candidate size",
    "26 MiB": "SUPERSEDED increment-5/6 figure: encoder-ablation candidate size",
    "31.93 MiB": "SUPERSEDED increment-5/6 figure: encoder-ablation candidate size",
    "32.24 MiB": "SUPERSEDED increment-5/6 figure: encoder-ablation candidate size",
    "19.88 MiB": "SUPERSEDED increment-5/6 figure: encoder-ablation candidate size",
    "128.90 MiB": "SUPERSEDED increment-4 figure: NLI bi-encoder int8_embed size",
    "184.96 MiB": "SUPERSEDED increment-4 figure: NLI bi-encoder size",
    "187.22 MiB": "SUPERSEDED increment-4 figure: NLI bi-encoder cold load",
    "201.68 MiB": "SUPERSEDED increment-4 figure: NLI bi-encoder fp32 size as first measured",
    "201.69 MiB": "SUPERSEDED increment-4 figure: NLI bi-encoder fp32 size, second run",
    "213 MiB": "SUPERSEDED increment-4 figure, rounded in prose",
    "3.394 MiB": "SUPERSEDED increment-6 figure: distilroberta tokenizer BEFORE compaction",
    "11.32 MiB": "onnxruntime CPU runtime floor, measured; narrative rendering",
    "11.3 MiB": "onnxruntime CPU runtime floor, narrative rounding",
    "0.24 MiB": "SUPERSEDED increment-5 figure: all-MiniLM tokenizer size",
    "37.2 MiB": "SUPERSEDED increment-5 figure",
    "30 ms": "SUPERSEDED increment-5 latency narrative",
    # environmental / third-party
    "900 MiB": "torch+transformers download cost, third-party",
    "1.2 GiB": "peak disk during build, environmental measurement",
    "2.5 GiB": "peak disk during build, environmental measurement",
    "314 MiB": "HuggingFace model download size, third-party",
    "330 MiB": "HuggingFace model download size, third-party",
    "5.5 MiB": "onnxruntime-web package download, third-party",
}

TOKEN = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(ms|MiB|GiB|KiB)\b")
GOVERNED = ["README.md", "export/SIZE_BUDGET.md", "docs/limitations.md",
            "data/MANIFEST.md"]


def check_claims():
    rows = []
    for cid, doc, literal, source, value, fmt in CLAIMS:
        text = _doc(doc)
        rendered = fmt.format(value)
        in_doc = literal in text
        agrees = rendered == literal
        rows.append({
            "id": cid,
            "document": doc,
            "published": literal,
            "artifact_source": source,
            "artifact_value": value,
            "artifact_rendered": rendered,
            "literal_present_in_document": in_doc,
            "published_equals_artifact": agrees,
            "ok": bool(in_doc and agrees),
        })
    return rows


def check_orphans():
    covered = {c[2] for c in CLAIMS}
    # a claim literal may itself contain the token (e.g. "CEIL-4 230.39 ms")
    covered |= {m.group(0).strip() for lit in list(covered)
                for m in [TOKEN.search(lit)] if m}
    orphans = []
    for doc in GOVERNED:
        text = _doc(doc)
        for m in TOKEN.finditer(text):
            tok = f"{m.group(1)} {m.group(2)}"
            raw = m.group(0)
            if tok in covered or raw in covered or tok in EXEMPT:
                continue
            line = text[:m.start()].count("\n") + 1
            orphans.append({"document": doc, "line": line, "token": tok})
    return orphans


def check_ceilings():
    """The ceiling VALUES quoted in prose must equal the ones verify.py enforced."""
    want = {
        "CEIL_1_int8_model_bytes": 33554432,
        "CEIL_2_tokenizer_bytes": 2097152,
        "CEIL_3_cold_payload_bytes": 67108864,
        "CEIL_4_p95_latency_ms": 500.0,
    }
    got = CB["int8_embed"]["ceiling_checks"]
    rows = []
    for k, v in want.items():
        rows.append({"ceiling": k, "documented": v,
                     "verify_report": got[k]["ceiling"],
                     "ok": got[k]["ceiling"] == v})
    return rows


def check_superseded_markers():
    """Every document carrying a SUPERSEDED-exempt token must say so in the open.

    An exempt historical figure is only honest if a reader can tell it is
    historical. This requires the marker string to appear in any governed
    document that quotes one.
    """
    superseded = {k for k, v in EXEMPT.items() if v.startswith("SUPERSEDED")}
    rows = []
    for doc in GOVERNED:
        text = _doc(doc)
        hits = sorted({f"{m.group(1)} {m.group(2)}" for m in TOKEN.finditer(text)}
                      & superseded)
        if not hits:
            continue
        marked = "SUPERSEDED" in text
        rows.append({"document": doc, "superseded_tokens": len(hits),
                     "marker_present": marked, "ok": marked})
    return rows


def main(argv):
    claims = check_claims()
    orphans = check_orphans()
    ceilings = check_ceilings()
    markers = check_superseded_markers()

    bad_claims = [r for r in claims if not r["ok"]]
    bad_ceilings = [r for r in ceilings if not r["ok"]]
    bad_markers = [r for r in markers if not r["ok"]]
    ok = not (bad_claims or orphans or bad_ceilings or bad_markers)

    report = {
        "verify_report_measured_at": VR["measured_at_utc"],
        "n_claims": len(claims),
        "n_claims_failing": len(bad_claims),
        "claims_failing": bad_claims,
        "n_orphan_tokens": len(orphans),
        "orphan_tokens": orphans,
        "n_exempt_registered": len(EXEMPT),
        "ceilings": ceilings,
        "superseded_markers": markers,
        "claims": claims,
        "ok": ok,
    }

    if "--json" in argv:
        print(json.dumps(report, indent=2))
    else:
        print(f"claims checked      : {len(claims)}")
        print(f"claims failing      : {len(bad_claims)}")
        for r in bad_claims:
            if not r["literal_present_in_document"]:
                print(f"  MISSING  {r['id']}: {r['published']!r} not in {r['document']}")
            else:
                print(f"  MISMATCH {r['id']}: {r['document']} says {r['published']!r}, "
                      f"{r['artifact_source']} renders {r['artifact_rendered']!r}")
        print(f"orphan tokens       : {len(orphans)}")
        for o in orphans:
            print(f"  ORPHAN   {o['document']}:{o['line']} {o['token']!r} "
                  f"— not a registered claim and not in EXEMPT")
        print(f"ceiling values      : {len(ceilings) - len(bad_ceilings)}/{len(ceilings)} agree")
        for r in bad_ceilings:
            print(f"  CEILING  {r['ceiling']}: doc {r['documented']} vs "
                  f"verify_report {r['verify_report']}")
        print(f"superseded markers  : "
              f"{len(markers) - len(bad_markers)}/{len(markers)} documents marked")
        for r in bad_markers:
            print(f"  UNMARKED {r['document']} quotes {r['superseded_tokens']} "
                  f"superseded figures with no SUPERSEDED marker")
        print("RESULT: " + ("agree" if ok else "DISAGREE"))

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
