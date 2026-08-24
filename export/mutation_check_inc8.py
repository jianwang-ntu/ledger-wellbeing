"""Do increment 8's guards actually catch anything?

A test suite that has never failed is a description, not a check. This script
breaks each property increment 8 claims — in the source or in a measured
artifact — runs the guard that is supposed to notice, and records whether it
did. Every mutation is reverted afterwards, whether it was caught or not.

The mutations are chosen to be the *plausible* regressions: the shortcut a
future author takes because it is simpler, or the number a future author edits
because it is in the way. "Delete the whole file" would be caught by anything
and proves nothing.

Run: ``python3 export/mutation_check_inc8.py``
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (label, file, old, new, test selector)
MUTATIONS: list[tuple[str, str, str, str, str]] = [
    # --- the safety layer -------------------------------------------------
    ("crisis routing stops blocking the model",
     "ledger/safety/crisis_router.py",
     'return self.severity == "acute"', "return False",
     "tests/test_engine.py::TestTheModelIsNotInTheCrisisPath"),
    ("an acute entry is scored anyway",
     "ledger/app/engine.py",
     "if decision.blocks_model_output:", "if False:",
     "tests/test_engine.py::TestTheModelIsNotInTheCrisisPath"),
    ("the crisis path returns no helplines",
     "ledger/safety/resources.py",
     "return local + directory", "return ()",
     "tests/test_engine.py::TestTheModelIsNotInTheCrisisPath::"
     "test_an_acute_entry_still_returns_reachable_help"),

    # --- span aggregation, R8-1 -------------------------------------------
    ("structural tokens are silently dropped instead of bucketed",
     "ledger/app/spans.py",
     "            structural.append(token_index)\n            continue",
     "            continue",
     "tests/test_spans.py::TestPartition"),
    ("a token is counted in every span it overlaps",
     "ledger/app/spans.py",
     "                buckets[span_index].append(token_index)\n                break",
     "                buckets[span_index].append(token_index)",
     "tests/test_spans.py::TestPartition"),
    ("padding tokens are attributed",
     "ledger/app/spans.py",
     "        if not int(mask):\n            continue",
     "        if False:\n            continue",
     "tests/test_spans.py::TestPartition"),

    # --- the store, R8-5 and R8-6 -----------------------------------------
    ("scrypt cost is lowered to something a laptop can brute-force",
     "ledger/store/crypto.py", "SCRYPT_N = 1 << 15", "SCRYPT_N = 1 << 10",
     "tests/test_store.py::TestTheKdfIsNotQuietlyWeakened"),
    ("a record that fails authentication is skipped rather than raising",
     "ledger/store/crypto.py",
     "            raise CorruptStore(f\"record {index} failed authentication\") from exc",
     "            index += 1\n            continue",
     "tests/test_store.py::TestTampering"),
    ("record position is no longer bound into the AAD",
     "ledger/store/crypto.py",
     "    return header + LENGTH_PREFIX.pack(index)", "    return header",
     "tests/test_store.py::TestTampering::"
     "test_a_record_cannot_be_moved_to_another_position"),
    ("wipe unlinks without overwriting",
     "ledger/store/journal.py",
     "            os.write(fd, os.urandom(size))", "            os.write(fd, b\"\")",
     "tests/test_store.py::TestWipe"),
    ("the store file is created world-readable",
     "ledger/store/journal.py",
     "os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600",
     "os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644",
     "tests/test_store.py::TestRoundTrip::test_file_is_owner_only"),
    ("entries are appended as plaintext json",
     "ledger/store/journal.py",
     "        record = seal(key, header, index, entry.to_json())",
     "        record = entry.to_json()",
     "tests/test_store.py::TestNoPlaintextOnDisk"),

    # --- what the product claims, R8-7 and R8-8 ---------------------------
    ("the usable-AUC floor is lowered until activation passes",
     "ledger/app/evidence.py", "\nUSABLE_HELD_OUT_AUC = 0.70", "\nUSABLE_HELD_OUT_AUC = 0.50",
     "tests/test_engine.py::TestWhatTheEngineClaims"),
    ("the report stops flagging an unestablished dimension",
     "ledger/app/report.py",
     'mark = "" if evidence[dim]["established"] else "   [NOT ESTABLISHED]"',
     'mark = ""',
     "tests/test_report.py::TestUnestablishedDimensionsAreLabelled"),
    ("the report drops the non-diagnostic contract",
     "ledger/app/report.py",
     "    for chunk in _wrap(CONTRACT, 62):\n        lines.append(chunk)",
     "    pass",
     "tests/test_report.py::TestTheContract"),
    ("the report starts naming a condition",
     "ledger/app/report.py",
     '"Each number is a contrast between two ways of writing, not a quantity of "',
     '"Depression symptom severity is improving. Each number is a contrast, "',
     "tests/test_report.py::TestNoClinicalLanguage"),
    ("the report claims the three listed spans are the whole score",
     "ledger/app/report.py",
     '"anything. Only the largest few contributions are listed above. The "',
     '"anything. The contributions listed above are the score, and "',
     "tests/test_report.py::TestWhatTheReportShows::"
     "test_it_does_not_claim_the_LISTED_contributions_are_the_whole_sum"),
    ("the head starts describing itself as trained",
     "ledger/app/engine.py", '"head_is_trained": False,', '"head_is_trained": True,',
     "tests/test_engine.py::TestWhatTheEngineClaims::"
     "test_the_head_is_never_described_as_trained"),

    # --- the dependency fix, DEFECT-INC8-001 ------------------------------
    ("the application imports transformers again",
     "ledger/app/engine.py", "import onnxruntime as ort",
     "import onnxruntime as ort\nimport transformers",
     "tests/test_engine.py::TestTheApplicationRunsWithoutTransformers"),
    ("the tokenizer stops padding to the measured length",
     "ledger/app/local_tokenizer.py",
     "    tokenizer.enable_padding(length=max_length, pad_id=pad_id, pad_token=pad_token)",
     "    pass",
     "tests/test_local_tokenizer.py::TestTheTokenizerItself"),

    # --- the measurements themselves --------------------------------------
    ("the egress instrument calls every destination loopback",
     "export/egress_audit.py",
     "def _is_loopback(host) -> bool:",
     "def _is_loopback(host) -> bool:\n    return True",
     "tests/test_egress.py::TestTheInstrumentCanFail"),
    ("the additivity tolerance is widened in code",
     "ledger/app/engine.py", "\nADDITIVITY_MAX_RESIDUAL = 1e-4",
     "\nADDITIVITY_MAX_RESIDUAL = 1e-1",
     "tests/test_spans.py::TestTheToleranceIsOneNumber"),
    ("the app and the export pipeline disagree about the tolerance",
     "export/verify.py", "\nADDITIVITY_MAX_RESIDUAL = 1e-4",
     "\nADDITIVITY_MAX_RESIDUAL = 1e-3",
     "tests/test_spans.py::TestTheToleranceIsOneNumber"),
]

# (label, artifact, json mutation, test selector)
ARTIFACT_MUTATIONS = [
    ("the egress report hides a non-loopback call",
     "artifacts/egress_audit.json",
     lambda d: d.update(non_loopback_calls=[{"kind": "connect",
                                             "target": "('huggingface.co', 443)",
                                             "loopback": False, "stack": []}]),
     "tests/test_egress.py::TestTheMeasurementPassed::"
     "test_no_non_loopback_call_was_recorded"),
    ("the egress report passes without exercising the app",
     "artifacts/egress_audit.json",
     lambda d: d.update(application_exercise={}),
     "tests/test_egress.py::TestTheMeasurementPassed::"
     "test_the_application_was_actually_exercised"),
    ("the egress report describes itself as a packet capture",
     "artifacts/egress_audit.json",
     lambda d: d.update(scope="packet-level capture", not_in_scope=[]),
     "tests/test_egress.py::TestTheMeasurementPassed::"
     "test_the_report_states_what_it_does_not_cover"),
    ("the span measurement hides a partition failure",
     "artifacts/span_additivity.json",
     lambda d: d["R8_1_partition"].update(failures=[{"entry": 3}]),
     "tests/test_spans.py::TestMeasuredAdditivity::"
     "test_the_partition_held_on_every_probe_entry"),
    ("the span measurement raises its recorded tolerance",
     "artifacts/span_additivity.json",
     lambda d: d.update(tolerance=1e-2),
     "tests/test_spans.py::TestMeasuredAdditivity"),
    ("tokenizer parity hides a mismatch",
     "artifacts/tokenizer_parity.json",
     lambda d: d["encodings"].update(mismatches=[{"field": "input_ids"}]),
     "tests/test_local_tokenizer.py::TestParityWasMeasured::test_not_one_token_differs"),
    ("tokenizer parity compares a handful of strings and calls it done",
     "artifacts/tokenizer_parity.json",
     lambda d: d["encodings"].update(texts_compared=3),
     "tests/test_local_tokenizer.py::TestParityWasMeasured::"
     "test_the_comparison_covered_the_probe_set_and_every_anchor"),
]


def guard_fails(selector: str) -> bool:
    """True when the named guard fails, which is what a caught mutation means."""
    result = subprocess.run(
        [sys.executable, "-m", "pytest", selector, "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=1200,
    )
    return result.returncode != 0


def main() -> int:
    results = []

    for label, relative, old, new, selector in MUTATIONS:
        path = ROOT / relative
        original = path.read_text()
        # An ambiguous target is worse than a missing one: replacing the first of
        # several occurrences can silently mutate a docstring instead of the code,
        # and then report MISSED against a guard that was never given anything to
        # catch. That happened on the first run of this script.
        occurrences = original.count(old)
        if occurrences != 1:
            results.append({"mutation": label, "file": relative, "guard": selector,
                            "caught": False,
                            "error": f"mutation target occurs {occurrences} times; "
                                     "it must occur exactly once"})
            print(f"INVALID {label}  ->  target occurs {occurrences} times")
            continue
        try:
            path.write_text(original.replace(old, new, 1))
            caught = guard_fails(selector)
        finally:
            path.write_text(original)
        results.append({"mutation": label, "file": relative, "guard": selector,
                        "caught": caught})
        print(("CAUGHT  " if caught else "MISSED  ") + f"{label}  ->  {selector}")

    for label, relative, mutate, selector in ARTIFACT_MUTATIONS:
        path = ROOT / relative
        if not path.exists():
            results.append({"mutation": label, "file": relative, "guard": selector,
                            "caught": False, "error": "artifact absent"})
            continue
        original = path.read_text()
        try:
            document = json.loads(original)
            mutate(document)
            path.write_text(json.dumps(document, indent=1) + "\n")
            caught = guard_fails(selector)
        finally:
            path.write_text(original)
        results.append({"mutation": label, "file": relative, "guard": selector,
                        "caught": caught})
        print(("CAUGHT  " if caught else "MISSED  ") + f"{label}  ->  {selector}")

    caught = sum(1 for r in results if r["caught"])
    print(f"\n{caught}/{len(results)} mutations caught")

    out = ROOT.parent / "audit" / "runs" / "inc8_mutations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"caught": caught, "total": len(results),
                               "results": results}, indent=1) + "\n")
    return 0 if caught == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
