"""Increment 5's conclusion, re-derived from the artifact's own numbers.

Same discipline as tests/test_scorer_ablation.py. The specific temptations this
increment creates, each of which is a guard below:

* **Reading a row whose positive control failed.** Four of seven rows here have
  a control under the floor. `nli_minilm_l6_mnli_unlicensed` shows exactly why
  that matters: it posts the second-best held-out number in the table (0.728) on
  a control of 0.516, which means the protocol cannot read it and the number is
  not evidence of anything.
* **Moving a ceiling by 0.24 MiB.** `nli_paraphrase_minilm_l3` projects to 32.24
  MiB against CEIL-1's 32.00. Editing CEIL-1 to 33 would "close" R4-CEIL-001 on
  a model that is at chance anyway.
* **Adopting the shippable row because it ships.** `nli_xtremedistil_h256` fits
  and is at 0.504 - the incumbent's number - on an unreadable control. Shipping
  it would be a size fix presented as a scorer fix.
* **Quietly dropping R-4 after invoking it.** A closed option B has to be written
  where a reader will see it, not left in a JSON file.

These skip - not pass - when the artifact has not been built.
"""

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPORT = ROOT / "artifacts" / "size_feasible_scorer.json"
SCORER4 = ROOT / "artifacts" / "scorer_ablation.json"
ENCODER3 = ROOT / "artifacts" / "encoder_ablation.json"
MANIFEST = ROOT / "data" / "MANIFEST.md"
LIMITATIONS = ROOT / "docs" / "limitations.md"
BUDGET = ROOT / "export" / "SIZE_BUDGET.md"


def load():
    if not REPORT.exists():
        raise unittest.SkipTest("artifacts/size_feasible_scorer.json not built - "
                                "run export/size_feasible_scorer.py")
    return json.loads(REPORT.read_text())


def variant(report, name):
    return next(v for v in report["variants"] if v["variant"] == name)


class TestTheThreeRunsAgree(unittest.TestCase):
    """Increment 5 recomputes both increment 3's incumbent and increment 4's
    selection inside its own run. A disagreement means one of the runs is wrong
    and none of the three may be quoted."""

    def test_baseline_reproduces_increments_3_and_4(self):
        report = load()
        self.assertAlmostEqual(variant(report, "incumbent_centroid")["macro_held_out_auc"],
                               0.504, places=4)
        if SCORER4.exists():
            four = json.loads(SCORER4.read_text())
            prev = next(v for v in four["variants"] if v["variant"] == "incumbent_centroid")
            self.assertAlmostEqual(variant(report, "incumbent_centroid")["macro_held_out_auc"],
                                   prev["macro_held_out_auc"], places=4)

    def test_upper_reference_reproduces_increment_4s_selection(self):
        report = load()
        if not SCORER4.exists():
            raise unittest.SkipTest("artifacts/scorer_ablation.json absent")
        four = json.loads(SCORER4.read_text())
        prev = next(v for v in four["variants"] if v["variant"] == "nli_sbert_centroid")
        here = variant(report, "nli_sbert_768_reference")
        self.assertEqual(here["model"], prev["model"])
        self.assertAlmostEqual(here["macro_held_out_auc"], prev["macro_held_out_auc"], places=4)
        self.assertEqual(here["int8embed_projected_bytes"], prev["int8embed_projected_bytes"])

    def test_thresholds_are_the_ones_increment_3_fixed(self):
        report = load()
        if not ENCODER3.exists():
            raise unittest.SkipTest("artifacts/encoder_ablation.json absent")
        enc = json.loads(ENCODER3.read_text())
        t = report["thresholds_inherited"]
        self.assertEqual(t["usable_held_out_auc"], enc["usable_held_out_auc_threshold"])
        self.assertEqual(t["positive_control_floor"], enc["positive_control_floor"])


class TestNoCeilingMoved(unittest.TestCase):
    """CEIL-1 and CEIL-3 were fixed in SIZE_BUDGET.md before increment 2 exported
    anything. The 32.24 MiB row is 0.24 MiB over. That is the edit this guards."""

    def test_ceilings_in_the_artifact_are_the_ones_in_common(self):
        report = load()
        sys.path.insert(0, str(ROOT / "export"))
        from common import CEILINGS, ORT_RUNTIME_FLOOR_BYTES
        t = report["thresholds_inherited"]
        self.assertEqual(t["ceil_1_int8_model_bytes"], CEILINGS["CEIL_1_int8_model_bytes"])
        self.assertEqual(t["ceil_3_cold_payload_bytes"], CEILINGS["CEIL_3_cold_payload_bytes"])
        self.assertEqual(t["ort_runtime_floor_bytes"], ORT_RUNTIME_FLOOR_BYTES)

    def test_ceil_1_is_still_32_mib(self):
        sys.path.insert(0, str(ROOT / "export"))
        from common import CEILINGS
        self.assertEqual(CEILINGS["CEIL_1_int8_model_bytes"], 32 * 1024 * 1024,
                         "CEIL-1 moved. SIZE_BUDGET.md forbids editing a ceiling to match a "
                         "measurement; a genuinely wrong ceiling is superseded in a new row.")
        self.assertEqual(CEILINGS["CEIL_3_cold_payload_bytes"], 64 * 1024 * 1024)

    def test_shippability_is_recomputable_for_every_row(self):
        report = load()
        sys.path.insert(0, str(ROOT / "export"))
        from common import CEILINGS, ORT_RUNTIME_FLOOR_BYTES
        for v in report["variants"]:
            b = v["int8embed_projected_bytes"]
            expected = (b <= CEILINGS["CEIL_1_int8_model_bytes"]
                        and ORT_RUNTIME_FLOOR_BYTES + b <= CEILINGS["CEIL_3_cold_payload_bytes"])
            self.assertEqual(v["shippable"], expected,
                             f"{v['variant']} records shippable={v['shippable']} but the "
                             f"ceilings give {expected}")
            self.assertEqual(v["projected_cold_payload_bytes"], ORT_RUNTIME_FLOOR_BYTES + b)


class TestAnUnreadableRowIsNotReadThrough(unittest.TestCase):
    def test_readable_set_matches_the_controls(self):
        report = load()
        floor = report["thresholds_inherited"]["positive_control_floor"]
        derived = sorted(v["variant"] for v in report["variants"]
                         if v["selection_candidate"] and v["positive_control_macro"] >= floor)
        self.assertEqual(sorted(report["attrition"]["readable"]), derived)

    def test_no_unreadable_variant_is_selected(self):
        report = load()
        if report["selected"] is None:
            raise unittest.SkipTest("nothing selected")
        floor = report["thresholds_inherited"]["positive_control_floor"]
        self.assertGreaterEqual(variant(report, report["selected"])["positive_control_macro"],
                                floor)

    def test_the_best_held_out_number_in_the_table_is_not_quoted_as_a_result(self):
        """The highest held-out AUC among the non-reference rows belongs to a row
        whose control failed. It must not appear in the verdict as a finding."""
        report = load()
        floor = report["thresholds_inherited"]["positive_control_floor"]
        unreadable = [v for v in report["variants"] if v["positive_control_macro"] < floor]
        if not unreadable:
            raise unittest.SkipTest("every row is readable")
        best_unreadable = max(unreadable, key=lambda v: v["macro_held_out_auc"])
        self.assertNotIn(best_unreadable["variant"], report["verdict"],
                         "the verdict names a variant whose positive control is below the floor")


class TestSelectionRequiresBothHalves(unittest.TestCase):
    """The whole point of increment 5: separation and size are one criterion now."""

    def test_selection_rule_is_a_tightening_not_a_loosening(self):
        report = load()
        self.assertIn("SHIPPABLE", report["selection_rule"])
        self.assertIn("tightening", report["selection_rule"].lower())

    def test_a_selection_must_satisfy_every_criterion(self):
        report = load()
        if report["selected"] is None:
            raise unittest.SkipTest("nothing selected")
        chosen = variant(report, report["selected"])
        t = report["thresholds_inherited"]
        self.assertTrue(chosen["selection_candidate"])
        self.assertTrue(chosen["shippable"])
        self.assertTrue(chosen["additivity_preserved"])
        self.assertGreaterEqual(chosen["macro_held_out_auc"], t["usable_held_out_auc"])
        self.assertGreaterEqual(chosen["positive_control_macro"], t["positive_control_floor"])

    def test_a_selection_is_the_highest_usable_shippable_auc(self):
        report = load()
        if report["selected"] is None:
            raise unittest.SkipTest("nothing selected")
        t = report["thresholds_inherited"]
        pool = [v for v in report["variants"] if v["selection_candidate"] and v["shippable"]
                and v["additivity_preserved"]
                and v["macro_held_out_auc"] >= t["usable_held_out_auc"]
                and v["positive_control_macro"] >= t["positive_control_floor"]]
        self.assertEqual(report["selected"],
                         max(pool, key=lambda v: v["macro_held_out_auc"])["variant"])

    def test_a_shippable_row_that_does_not_separate_is_not_adopted(self):
        """nli_xtremedistil_h256 fits. That is not a reason to ship it."""
        report = load()
        t = report["thresholds_inherited"]
        for v in report["variants"]:
            if v["shippable"] and v["macro_held_out_auc"] < t["usable_held_out_auc"]:
                self.assertNotEqual(report["selected"], v["variant"],
                                    f"{v['variant']} was selected on size alone")

    def test_nothing_selected_means_nothing_adopted(self):
        report = load()
        self.assertEqual(report["adopted"], report["selected"] is not None)


class TestLicenceIsAGateNotAPreference(unittest.TestCase):
    def test_every_measured_model_records_a_licence_field(self):
        report = load()
        for v in report["variants"]:
            self.assertIn("licence", v, f"{v['variant']} has no licence field")

    def test_an_undeclared_licence_is_never_a_selection_candidate(self):
        report = load()
        for v in report["variants"]:
            if v["licence"] is None:
                self.assertFalse(v["selection_candidate"],
                                 f"{v['model']} declares no licence and is a selection "
                                 "candidate; data/MANIFEST.md requires a cleared licence")
                self.assertIn(v["variant"], report["excluded_from_selection"])

    def test_the_licence_exclusion_is_recorded_with_its_reason(self):
        report = load()
        for name, reason in report["excluded_from_selection"].items():
            self.assertTrue(reason and len(reason) > 20,
                            f"{name} is excluded without a stated reason")

    def test_every_model_this_run_downloads_is_attributed(self):
        report = load()
        if not MANIFEST.exists():
            raise unittest.SkipTest("data/MANIFEST.md absent")
        text = MANIFEST.read_text()
        for v in report["variants"]:
            self.assertIn(v["model"], text,
                          f"{v['model']} was downloaded and measured but is not attributed "
                          "in data/MANIFEST.md")
            self.assertIn(v["revision"], text,
                          f"{v['model']} is attributed without its pinned revision")


class TestTheCandidateListWasClosedBeforeTheRun(unittest.TestCase):
    def test_the_artifact_matches_the_script(self):
        report = load()
        sys.path.insert(0, str(ROOT / "export"))
        from size_feasible_scorer import CANDIDATES
        self.assertEqual([c["variant"] for c in CANDIDATES],
                         [v["variant"] for v in report["variants"]],
                         "the artifact's rows differ from the committed candidate list")
        for c in CANDIDATES:
            v = variant(report, c["variant"])
            self.assertEqual(v["model"], c["model"])
            self.assertEqual(v["revision"], c["revision"],
                             f"{c['variant']} was measured at a different revision than the "
                             "one committed")

    def test_expectations_were_recorded_before_the_measurement(self):
        """A pre-registered expectation that turns out wrong is information.
        One did: paraphrase-MiniLM-L3 was expected to fit and does not."""
        report = load()
        self.assertEqual(len(report["expectation_vs_outcome"]), len(report["variants"]))
        wrong = [k for k, v in report["expectation_vs_outcome"].items()
                 if v["expected_shippable"] != v["measured_shippable"]]
        for v in report["variants"]:
            self.assertEqual(v["expected_shippable_before_measurement"],
                             report["expectation_vs_outcome"][v["variant"]]["expected_shippable"])
        self.assertIsInstance(wrong, list)


class TestTheOutcomeIsHonouredInTheRepository(unittest.TestCase):
    def test_a_null_selection_leaves_common_on_the_incumbent(self):
        report = load()
        if report["selected"] is not None:
            raise unittest.SkipTest("a scorer was selected")
        sys.path.insert(0, str(ROOT / "export"))
        from common import BASE_MODEL, BASE_REVISION
        incumbent = variant(report, "incumbent_centroid")
        self.assertEqual(BASE_MODEL, incumbent["model"])
        self.assertEqual(BASE_REVISION, incumbent["revision"])

    def test_a_null_selection_invokes_r4_by_name(self):
        report = load()
        if report["selected"] is not None:
            raise unittest.SkipTest("a scorer was selected")
        self.assertIn("R-4", report["verdict"])
        self.assertIn("CLOSED", report["verdict"])

    def test_a_closed_option_b_is_written_where_a_reader_will_see_it(self):
        report = load()
        if report["selected"] is not None:
            raise unittest.SkipTest("a scorer was selected")
        self.assertTrue(LIMITATIONS.exists())
        text = LIMITATIONS.read_text()
        self.assertIn("R-4", text,
                      "docs/limitations.md does not record that the R-4 fallback fired")
        shippable = [v for v in report["variants"] if v["shippable"]]
        for v in shippable:
            self.assertIn(v["model"], text,
                          f"{v['model']} is the row that fits and does not separate; "
                          "docs/limitations.md must say so")

    def test_the_budget_records_the_increment(self):
        report = load()
        if report["selected"] is not None:
            raise unittest.SkipTest("a scorer was selected")
        self.assertIn("increment 5", BUDGET.read_text().lower(),
                      "export/SIZE_BUDGET.md does not record what increment 5 measured")

    def test_readme_makes_no_in_browser_claim_the_measurements_do_not_support(self):
        report = load()
        if report["selected"] is not None:
            raise unittest.SkipTest("a scorer was selected")
        readme = (ROOT / "README.md").read_text().lower()
        for phrase in ("runs entirely in your browser today",
                       "ships in the browser",
                       "a scorer that fits the browser budget"):
            self.assertNotIn(phrase, readme)


if __name__ == "__main__":
    unittest.main()
