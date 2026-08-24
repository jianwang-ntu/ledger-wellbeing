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

    # ------------------------------------------------------------------
    # DEFECT-INC7-001. The two tests below used to read
    #
    #     all(c["pass"] for c in checks.values())
    #
    # i.e. "shippable => passes all six ceilings". That was the correct rule
    # until increment 6 scoped enforcement to a delivery target, and it was
    # never updated - because it could not fail. shippable_builds was empty in
    # increments 3, 4, 5 and 6, so both tests compared [] against [] and stayed
    # green through the single most self-serving change in this repository.
    # Increment 7 is the first run that produces a shippable build, and it is
    # the run that exposed them.
    #
    # They are replaced rather than relaxed. The new pair asserts the enforced
    # rule AND the bookkeeping that keeps the dropped claim visible, and
    # TestTheWebTargetIsLostNotForgotten below re-asserts the original all-six
    # rule for the target it was written for.
    # ------------------------------------------------------------------

    def _enforced(self, build):
        checks = self.report["candidate_builds"][build]["ceiling_checks"]
        return {k: c for k, c in checks.items() if c["enforced_on_this_target"]}

    def test_the_report_enforces_exactly_the_targets_ceilings(self):
        """The report may not invent its own enforcement set."""
        import sys, pathlib as _pl
        sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1] / "export"))
        import common
        self.assertEqual(sorted(self.report["enforced_ceilings"]),
                         sorted(common.ENFORCED_BY_TARGET[self.report["delivery_target"]]))
        for build, data in self.report["candidate_builds"].items():
            flagged = sorted(k for k, c in data["ceiling_checks"].items()
                             if c["enforced_on_this_target"])
            self.assertEqual(flagged, sorted(self.report["enforced_ceilings"]), build)

    def test_shippable_list_agrees_with_the_flags(self):
        derived = sorted(b for b in self.report["candidate_builds"]
                         if all(c["pass"] is True for c in self._enforced(b).values()))
        self.assertEqual(sorted(self.report["shippable_builds"]), derived)

    def test_no_build_is_claimed_shippable_while_breaching_an_enforced_ceiling(self):
        for build in self.report["shippable_builds"]:
            failed = [k for k, c in self._enforced(build).items() if c["pass"] is not True]
            self.assertEqual(failed, [], f"{build} is listed shippable but fails {failed}")

    def test_a_shippable_build_still_publishes_every_ceiling_it_breaches(self):
        """The half of the old rule that must not be lost: an unenforced ceiling
        a shipping build fails has to appear in would_fail_web_target_on, so the
        dropped claim is in the artifact rather than absent from it."""
        for build in self.report["shippable_builds"]:
            data = self.report["candidate_builds"][build]
            breached = {k for k, c in data["ceiling_checks"].items() if c["pass"] is False}
            disclosed = set(data["would_fail_web_target_on"])
            self.assertEqual(breached - disclosed, set(),
                             f"{build} breaches {sorted(breached - disclosed)} without disclosing it")
            self.assertEqual(disclosed - breached, set(),
                             f"{build} discloses {sorted(disclosed - breached)} it does not breach")


class TestTheWebTargetIsLostNotForgotten(unittest.TestCase):
    """The original all-six rule, kept for the target it was written for.

    If DELIVERY_TARGET ever returns to `web`, the pre-increment-6 rule applies
    again with no edit to this file.
    """

    def setUp(self):
        self.report = load(VERIFY)

    def test_on_a_web_target_every_ceiling_gates_again(self):
        if self.report["delivery_target"] != "web":
            raise unittest.SkipTest("target is not web; the desktop rule is tested above")
        for build in self.report["shippable_builds"]:
            checks = self.report["candidate_builds"][build]["ceiling_checks"]
            failed = [k for k, c in checks.items() if c["pass"] is not True]
            self.assertEqual(failed, [], f"{build} shippable on web but fails {failed}")

    def test_no_build_here_would_have_shipped_to_a_browser(self):
        """Increment 7 met CEIL-2 and produced the first shippable build. That
        must not be allowed to read as 'the browser target came back'."""
        for name, data in self.report["candidate_builds"].items():
            self.assertNotEqual(data["would_fail_web_target_on"], [],
                                f"{name} claims no web-target failure; if that is real, "
                                "delete this test on purpose with the measurement")

    def test_the_selected_build_misses_the_web_latency_ceiling_too(self):
        """The browser was not lost on size alone (SIZE_BUDGET.md increment 6)."""
        selected = self.report.get("selected_build")
        if selected is None:
            raise unittest.SkipTest("nothing selected")
        wasm = self.report["candidate_builds"][selected].get("latency_wasm_1thread")
        if not isinstance(wasm, dict):
            raise unittest.SkipTest("no accepted WASM bench for the selected build")
        self.assertGreater(wasm["p95_ms"], 500.0,
                           "the selected build now fits the web latency ceiling - "
                           "re-open the delivery target on purpose")


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
