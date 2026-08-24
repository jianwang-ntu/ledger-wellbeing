"""The export pipeline's ceilings are checked, not described.

These tests re-derive every pass/fail flag in artifacts/verify_report.json from
the measured numbers in the same file, so a hand-edited verdict fails here. They
skip - rather than pass - when the artifacts have not been built, so a green run
on a clean checkout can never be mistaken for a green run on real measurements.
"""

import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
VERIFY = ROOT / "artifacts" / "verify_report.json"
BUDGET = ROOT / "export" / "SIZE_BUDGET.md"


def load(path):
    if not path.exists():
        raise unittest.SkipTest(f"{path.relative_to(ROOT)} not built - run export/run_all.sh")
    return json.loads(path.read_text())


class TestCeilingsAreHonoured(unittest.TestCase):
    def setUp(self):
        self.report = load(VERIFY)

    def test_declared_ceilings_match_the_budget_document(self):
        """common.CEILINGS is what the pipeline enforces; SIZE_BUDGET.md is what we published."""
        text = BUDGET.read_text()
        for needle in ("32 MiB", "2 MiB", "64 MiB", "500 ms", "0.99", "0.02"):
            self.assertIn(needle, text, f"{needle} missing from SIZE_BUDGET.md")

    def test_pass_flags_are_recomputable_from_the_measurements(self):
        for build, data in self.report["candidate_builds"].items():
            for name, check in data["ceiling_checks"].items():
                measured, ceiling = check["measured"], check["ceiling"]
                if measured is None:
                    self.assertIsNone(check["pass"], f"{build}/{name}: unmeasured but flagged")
                    continue
                expected = measured >= ceiling if "min_pearson" in name else measured <= ceiling
                self.assertEqual(check["pass"], expected,
                                 f"{build}/{name}: flag {check['pass']} contradicts "
                                 f"measured={measured} vs ceiling={ceiling}")

    def test_shippable_list_agrees_with_the_flags(self):
        derived = sorted(b for b, d in self.report["candidate_builds"].items()
                         if all(c["pass"] is True for c in d["ceiling_checks"].values()))
        self.assertEqual(sorted(self.report["shippable_builds"]), derived)

    def test_no_build_is_claimed_shippable_while_breaching_a_ceiling(self):
        for build in self.report["shippable_builds"]:
            checks = self.report["candidate_builds"][build]["ceiling_checks"]
            failed = [k for k, c in checks.items() if c["pass"] is not True]
            self.assertEqual(failed, [], f"{build} is listed shippable but fails {failed}")


class TestAttributionIdentitySurvivesExport(unittest.TestCase):
    """logit_k must equal sum_i token_attr[i,k] + bias_k in every build, or the
    explanation is a decoration rather than the score."""

    TOL = 1e-4

    def test_native_runtime(self):
        report = load(VERIFY)
        residuals = {"fp32": report["reference_build"]["attribution_identity_max_residual"]}
        for build, data in report["candidate_builds"].items():
            residuals[build] = data["attribution_identity_max_residual"]
        for build, residual in residuals.items():
            self.assertLess(residual, self.TOL, f"{build}: identity residual {residual}")

    def test_wasm_runtime(self):
        wasm_dir = ROOT / "artifacts" / "wasm"
        files = sorted(wasm_dir.glob("bench_*.json")) if wasm_dir.exists() else []
        if not files:
            raise unittest.SkipTest("no WASM benchmark output - run export/run_all.sh")
        for path in files:
            rec = json.loads(path.read_text())
            self.assertLess(rec["attribution_identity_residual"], self.TOL,
                            f"{rec['build']}: WASM identity residual "
                            f"{rec['attribution_identity_residual']}")


class TestNothingClaimsToBeTrained(unittest.TestCase):
    """The head is a zero-shot anchor construction. Until that changes in
    build_report.json, nothing in the repository may say 'fine-tuned'."""

    def test_build_report_says_untrained(self):
        report = load(ROOT / "artifacts" / "torch" / "build_report.json")
        self.assertFalse(report["head_is_trained"])
        self.assertEqual(report["head_version"], "anchor_v0")

    def test_readme_does_not_claim_a_finetune(self):
        readme = (ROOT / "README.md").read_text().lower()
        report_path = ROOT / "artifacts" / "torch" / "build_report.json"
        if report_path.exists() and json.loads(report_path.read_text())["head_is_trained"]:
            raise unittest.SkipTest("head is trained; this guard no longer applies")
        for phrase in ("we fine-tuned", "fine-tuned model", "our fine-tune"):
            self.assertNotIn(phrase, readme, f"README claims {phrase!r} but the head is untrained")


if __name__ == "__main__":
    unittest.main()
