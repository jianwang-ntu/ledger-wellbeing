"""Increment 7: re-serialize the tokenizer compactly, and prove it is the same tokenizer.

Pre-registered in export/INCREMENT_7_PREREGISTRATION.md. The rules R7-1..R7-3 are
decided here; R7-5 is decided by export/verify.py afterwards.

`tokenizers` writes tokenizer.json pretty-printed. JSON whitespace is not
semantic, so re-dumping the *same document* compactly is a way to MEET CEIL-2
rather than argue it away -- but only if the rewritten file drives the tokenizer
to identical output. That is established here by re-tokenizing, never by
trusting a file size.

The rewrite is staged and only committed to disk once every identity rule has
passed. A failure restores the original bytes exactly.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

from common import (CEILINGS, MAX_LENGTH, TOKENIZER_DIR, dir_bytes, probe_entries,
                    sha256_file, write_json)

TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
BACKUP = TOKENIZER_DIR / "tokenizer.json.orig"
REPORT = TOKENIZER_DIR.parent / "compact_tokenizer.json"

EXPECTED_VOCAB_SIZE = 50265   # pre-registered in R7-3


def anchor_sentences() -> list[str]:
    """Every anchor sentence, in a fixed order. R7-2(b)."""
    from ledger.model.dimensions import ANCHORS
    out = []
    for dim in sorted(ANCHORS):
        for pole in sorted(ANCHORS[dim]):
            out.extend(ANCHORS[dim][pole])
    return out


def fingerprint(tok, texts: list[str], *, padded: bool) -> dict:
    """Everything R7-2 compares. Arrays are returned raw for exact comparison."""
    kw = dict(return_offsets_mapping=True)
    if padded:
        kw.update(padding="max_length", truncation=True, max_length=MAX_LENGTH)
    enc = tok(texts, **kw)
    ids = [list(map(int, row)) for row in enc["input_ids"]]
    return {
        "input_ids": ids,
        "attention_mask": [list(map(int, row)) for row in enc["attention_mask"]],
        "offset_mapping": [[list(map(int, p)) for p in row] for row in enc["offset_mapping"]],
        "decoded": [tok.decode(row, skip_special_tokens=True) for row in ids],
    }


def compare(before: dict, after: dict) -> list[str]:
    """Exact elementwise comparison. Returns a list of mismatch descriptions."""
    bad = []
    for field in ("input_ids", "attention_mask", "offset_mapping", "decoded"):
        b, a = before[field], after[field]
        if len(b) != len(a):
            bad.append(f"{field}: row count {len(b)} -> {len(a)}")
            continue
        for i, (rb, ra) in enumerate(zip(b, a)):
            if rb != ra:
                bad.append(f"{field}[{i}] differs")
    return bad


def main() -> int:
    original_bytes = TOKENIZER_JSON.read_bytes()
    original_doc = json.loads(original_bytes)
    before_dir_bytes = dir_bytes(TOKENIZER_DIR)
    before_sha = sha256_file(TOKENIZER_JSON)

    texts_probe = probe_entries()
    texts_anchor = anchor_sentences()

    tok_before = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    fp_before = {
        "probe": fingerprint(tok_before, texts_probe, padded=True),
        "anchor": fingerprint(tok_before, texts_anchor, padded=False),
    }
    vocab_before = tok_before.vocab_size
    merges_before = len(original_doc["model"].get("merges", []))

    # --- stage the rewrite -------------------------------------------------
    shutil.copy2(TOKENIZER_JSON, BACKUP)
    compact = json.dumps(original_doc, separators=(",", ":"), ensure_ascii=False)
    TOKENIZER_JSON.write_text(compact, encoding="utf-8")

    failures: list[str] = []

    # R7-3 document identity ------------------------------------------------
    rewritten_bytes = TOKENIZER_JSON.read_bytes()
    rewritten_doc = json.loads(rewritten_bytes)
    if rewritten_doc != original_doc:
        failures.append("R7-3: parsed JSON document is not equal to the original")
    merges_after = len(rewritten_doc["model"].get("merges", []))
    if merges_after != merges_before:
        failures.append(f"R7-3: merges table {merges_before} -> {merges_after}")

    # R7-2 encode identity --------------------------------------------------
    tok_after = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    vocab_after = tok_after.vocab_size
    if vocab_after != EXPECTED_VOCAB_SIZE:
        failures.append(f"R7-3: vocab_size {vocab_after} != pre-registered {EXPECTED_VOCAB_SIZE}")
    if vocab_after != vocab_before:
        failures.append(f"R7-3: vocab_size {vocab_before} -> {vocab_after}")

    fp_after = {
        "probe": fingerprint(tok_after, texts_probe, padded=True),
        "anchor": fingerprint(tok_after, texts_anchor, padded=False),
    }
    mismatches = {k: compare(fp_before[k], fp_after[k]) for k in fp_before}
    for setname, bad in mismatches.items():
        for line in bad:
            failures.append(f"R7-2 [{setname}] {line}")

    # R7-1 size -------------------------------------------------------------
    after_dir_bytes = dir_bytes(TOKENIZER_DIR) - BACKUP.stat().st_size
    ceil2 = CEILINGS["CEIL_2_tokenizer_bytes"]
    ceil2_pass = after_dir_bytes <= ceil2
    if not ceil2_pass:
        failures.append(f"R7-1: {after_dir_bytes} B > CEIL-2 {ceil2} B")

    adopted = not failures
    if adopted:
        BACKUP.unlink()
    else:
        shutil.copy2(BACKUP, TOKENIZER_JSON)
        BACKUP.unlink()
        assert TOKENIZER_JSON.read_bytes() == original_bytes, "restore failed"

    report = {
        "increment": 7,
        "preregistration": "export/INCREMENT_7_PREREGISTRATION.md",
        "question": "is there a serialization of this tokenizer, same vocabulary and "
                    "identical encode() output, that fits CEIL-2's 2 MiB?",
        "adopted": adopted,
        "R7_1_size": {
            "before_dir_bytes": before_dir_bytes,
            "after_dir_bytes": after_dir_bytes,
            "ceiling_bytes": ceil2,
            "before_MiB": round(before_dir_bytes / 1048576, 3),
            "after_MiB": round(after_dir_bytes / 1048576, 3),
            "pass": ceil2_pass,
            "blind": False,
            "note": "probed before the pre-registration was written; disclosed there",
        },
        "R7_2_encode_identity": {
            "probe_entries": len(texts_probe),
            "probe_max_length": MAX_LENGTH,
            "anchor_sentences": len(texts_anchor),
            "fields_compared": ["input_ids", "attention_mask", "offset_mapping", "decoded"],
            "mismatches": mismatches,
            "pass": all(not v for v in mismatches.values()),
            "blind": True,
        },
        "R7_3_document_identity": {
            "parsed_documents_equal": rewritten_doc == original_doc,
            "vocab_size_before": vocab_before,
            "vocab_size_after": vocab_after,
            "vocab_size_expected": EXPECTED_VOCAB_SIZE,
            "merges_before": merges_before,
            "merges_after": merges_after,
            "pass": rewritten_doc == original_doc and vocab_after == EXPECTED_VOCAB_SIZE,
            "blind": True,
        },
        "sha256": {
            "tokenizer_json_before": before_sha,
            "tokenizer_json_after": sha256_file(TOKENIZER_JSON),
        },
        "failures": failures,
        "what_this_does_not_retire": (
            "The vocabulary really did grow 30,522 (WordPiece) -> 50,265 (byte-level BPE), "
            "1.65x, and CEIL-2's day-1 tripwire fired on a real change. Meeting the ceiling "
            "by compact serialization satisfies it as written -- bytes on disk -- and does "
            "NOT retire that signal. CEIL-1, CEIL-3 and CEIL-4-on-web remain failed."
        ),
    }
    write_json(REPORT, report)

    print(json.dumps({k: v for k, v in report.items() if k != "R7_2_encode_identity"}, indent=1))
    r2 = report["R7_2_encode_identity"]
    print(f"\nR7-2 encode identity: probe={r2['probe_entries']} entries @ {MAX_LENGTH} tok, "
          f"anchors={r2['anchor_sentences']} sentences, "
          f"fields={','.join(r2['fields_compared'])} -> "
          f"{'IDENTICAL' if r2['pass'] else 'MISMATCH'}")
    print(f"\ntokenizer dir: {before_dir_bytes} B -> {after_dir_bytes} B "
          f"(CEIL-2 {ceil2} B) : {'PASS' if ceil2_pass else 'FAIL'}")
    print(f"ADOPTED: {adopted}")
    for f in failures:
        print("  FAILURE:", f)
    return 0 if adopted else 1


if __name__ == "__main__":
    sys.exit(main())
