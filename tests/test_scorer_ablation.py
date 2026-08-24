"""Increment 4's conclusions, re-derived from the artifact's own numbers.

Same discipline as tests/test_encoder_ablation.py. The specific thing these
guards are here to stop is the temptation this increment creates: option B was
supposed to rescue the product, so a variant that scores well on a *broken*
control, or a cross-encoder adopted while the repository keeps advertising the
exactly-additive attribution head, are both failures that would read as
successes. Each of those is a test below.

These skip - not pass - when the artifact has not been built.
"""

import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCORER = ROOT / "artifacts" / "scorer_ablation.json"
ENCODER = ROOT / "artifacts" / "encoder_ablation.json"
MANIFEST = ROOT / "data" / "MANIFEST.md"
LIMITATIONS = ROOT / "docs" / "limitations.md"


def load():
    if not SCORER.exists():
        raise unittest.SkipTest("artifacts/scorer_ablation.json not built - "
                                "run export/scorer_ablation.py")
    return json.loads(SCORER.read_text())


def variant(report, name):
    return next(v for v in report["variants"] if v["variant"] == name)


class TestTheTwoRunsAgree(unittest.TestCase):
    """Increment 4 recomputes increment 3's incumbent inside its own run. If the
    two disagree, one of the two measurements is wrong and neither may be quoted."""

    def test_baseline_reproduces_the_increment_3_incumbent(self):
        report = load()
        if not ENCODER.exists():
            raise unittest.SkipTest("artifacts/encoder_ablation.json absent")
        enc = json.loads(ENCODER.read_text())
        incumbent = next(c for c in enc["candidates"] if c["role"].startswith("incumbent"))
        self.assertAlmostEqual(
            variant(report, "incumbent_centroid")["macro_held_out_auc"],
            incumbent["anchor_separation"]["macro_held_out_auc"], places=4,
            msg="increment 4's baseline does not reproduce increment 3's incumbent")

    def test_baseline_reference_block_is_not_hand_written(self):
        report = load()
        self.assertAlmostEqual(report["baseline_reference"]["macro_held_out_auc"],
                               variant(report, "incumbent_centroid")["macro_held_out_auc"],
                               places=4)

    def test_thresholds_are_the_ones_increment_3_fixed(self):
        report = load()
        if not ENCODER.exists():
            raise unittest.SkipTest("artifacts/encoder_ablation.json absent")
        enc = json.loads(ENCODER.read_text())
        inherited = report["thresholds_inherited_from_increment_3"]
        self.assertEqual(inherited["usable_held_out_auc"], enc["usable_held_out_auc_threshold"])
        self.assertEqual(inherited["positive_control_floor"], enc["positive_control_floor"])


class TestAControlThatFailsIsNotReadThrough(unittest.TestCase):
    """A variant whose positive control is below the floor is measuring its own
    bias. Its held-out number may not be used for anything, however good it looks."""

    def test_readable_set_matches_the_controls(self):
        report = load()
        floor = report["thresholds_inherited_from_increment_3"]["positive_control_floor"]
        derived = sorted(v["variant"] for v in report["variants"]
                         if v["selection_candidate"] and v["positive_control_macro"] >= floor)
        self.assertEqual(sorted(report["readable_variants"]), derived)

    def test_no_unreadable_variant_is_selected(self):
        report = load()
        if report["selected"] is None:
            return
        self.assertIn(report["selected"], report["readable_variants"],
                      f"{report['selected']} was selected with a failing positive control")

    def test_option_c_basis_comes_from_a_readable_variant(self):
        report = load()
        self.assertIn(report["option_c_basis"]["measured_under"],
                      report["readable_variants"] + ["incumbent_centroid"])

    def test_a_variant_that_beats_the_threshold_on_a_broken_control_is_not_usable(self):
        """The concrete trap: a scorer can look like it separates poles while being
        unable to tell two dimensions apart at all."""
        report = load()
        for v in report["variants"]:
            if v["positive_control_macro"] < report["thresholds_inherited_from_increment_3"][
                    "positive_control_floor"]:
                self.assertNotIn(v["variant"], report["usable_variants"])


class TestSelectionFollowsTheStatedRule(unittest.TestCase):
    def test_selection_is_recomputable(self):
        report = load()
        t = report["thresholds_inherited_from_increment_3"]
        readable = [v for v in report["variants"] if v["selection_candidate"]
                    and v["positive_control_macro"] >= t["positive_control_floor"]]
        usable = [v for v in readable if v["macro_held_out_auc"] >= t["usable_held_out_auc"]]
        additive = [v for v in usable if v["additivity_preserved"]]
        pool = additive or usable
        expected = max(pool, key=lambda v: v["macro_held_out_auc"])["variant"] if pool else None
        self.assertEqual(report["selected"], expected,
                         f"selected={report['selected']} but the stated rule yields {expected}")

    def test_usable_set_matches_the_measurements(self):
        report = load()
        t = report["thresholds_inherited_from_increment_3"]
        derived = sorted(v["variant"] for v in report["variants"]
                         if v["selection_candidate"]
                         and v["positive_control_macro"] >= t["positive_control_floor"]
                         and v["macro_held_out_auc"] >= t["usable_held_out_auc"])
        self.assertEqual(sorted(report["usable_variants"]), derived)

    def test_the_diagnostic_is_never_selected(self):
        report = load()
        diagnostic = [v["variant"] for v in report["variants"] if not v["selection_candidate"]]
        self.assertIn("sst2_global_polarity", diagnostic)
        self.assertNotIn(report["selected"], diagnostic)

    def test_an_additive_variant_is_preferred_whenever_one_is_usable(self):
        report = load()
        usable = [v for v in report["variants"] if v["variant"] in report["usable_variants"]]
        if any(v["additivity_preserved"] for v in usable) and report["selected"] is not None:
            self.assertTrue(report["selection_preserves_additivity"],
                            "an additivity-preserving variant was usable and a non-additive one "
                            "was selected anyway")


class TestTheRepositoryHonoursTheOutcome(unittest.TestCase):
    def test_a_null_selection_leaves_the_scorer_alone(self):
        """Option B closed means the head does not change on the strength of it."""
        report = load()
        if report["selected"] is not None:
            raise unittest.SkipTest("a scorer was selected; this guard does not apply")
        import sys
        sys.path.insert(0, str(ROOT))
        from ledger.model.dimensions import HEAD_VERSION
        self.assertEqual(HEAD_VERSION, "anchor_v0",
                         "scorer_ablation.json selected nothing, so the head version must not "
                         "have moved")

    def test_c4_additivity_is_not_advertised_after_a_non_additive_selection(self):
        report = load()
        if report["selected"] is None or report["selection_preserves_additivity"]:
            raise unittest.SkipTest("no non-additive scorer was adopted")
        readme = (ROOT / "README.md").read_text()
        self.assertNotIn("holds *exactly*", readme,
                         "a cross-encoder was selected but the README still claims the exact "
                         "attribution identity")

    def test_every_model_this_ablation_downloads_is_attributed(self):
        report = load()
        if not MANIFEST.exists():
            raise unittest.SkipTest("data/MANIFEST.md absent")
        text = MANIFEST.read_text()
        for v in report["variants"]:
            self.assertIn(v["model"], text,
                          f"{v['model']} was downloaded and measured but is not attributed "
                          f"in data/MANIFEST.md")
            self.assertIn(v["revision"], text,
                          f"{v['model']} is attributed without its pinned revision")


class TestASelectionIsNotAnAdoption(unittest.TestCase):
    """The trap this increment walks into: a scorer that wins on separation and
    is six times the size budget. Selecting it and quietly leaving the repository
    on the incumbent - or quietly switching to it and breaking a ceiling fixed
    before increment 2 - are both silent failures. Neither is allowed to be."""

    def test_adoption_blockers_are_recomputable_from_the_ceilings(self):
        report = load()
        if report["selected"] is None:
            raise unittest.SkipTest("nothing selected")
        import sys
        sys.path.insert(0, str(ROOT / "export"))
        from common import CEILINGS, ORT_RUNTIME_FLOOR_BYTES
        chosen = variant(report, report["selected"])
        b = chosen["int8embed_projected_bytes"]
        expected = (b > CEILINGS["CEIL_1_int8_model_bytes"],
                    ORT_RUNTIME_FLOOR_BYTES + b > CEILINGS["CEIL_3_cold_payload_bytes"])
        recorded = report["adoption"]["adoption_blockers"]
        self.assertEqual(any(expected), bool(recorded),
                         f"adoption_blockers={recorded} but the ceilings give {expected}")
        self.assertEqual(report["adoption"]["adopted"], not recorded)

    def test_a_blocked_adoption_leaves_common_on_the_incumbent(self):
        report = load()
        if report["selected"] is None or report["adoption"]["adopted"]:
            raise unittest.SkipTest("no blocked adoption")
        import sys
        sys.path.insert(0, str(ROOT / "export"))
        from common import BASE_MODEL, BASE_REVISION
        incumbent = variant(report, "incumbent_centroid")
        self.assertEqual(BASE_MODEL, incumbent["model"],
                         "the selected scorer breaches a size ceiling, so export/common.py must "
                         "still hold the incumbent")
        self.assertEqual(BASE_REVISION, incumbent["revision"])

    def test_an_unblocked_adoption_is_actually_applied(self):
        report = load()
        if report["selected"] is None or not report["adoption"]["adopted"]:
            raise unittest.SkipTest("no clean adoption to check")
        import sys
        sys.path.insert(0, str(ROOT / "export"))
        from common import BASE_MODEL, BASE_REVISION
        self.assertEqual(BASE_MODEL, report["selected_model"])
        self.assertEqual(BASE_REVISION, report["selected_revision"])

    def test_a_blocked_adoption_is_recorded_where_a_reader_will_see_it(self):
        report = load()
        if report["selected"] is None or report["adoption"]["adopted"]:
            raise unittest.SkipTest("no blocked adoption")
        self.assertTrue(LIMITATIONS.exists())
        text = LIMITATIONS.read_text()
        chosen = variant(report, report["selected"])
        self.assertIn(chosen["model"], text,
                      "docs/limitations.md does not name the scorer that was selected and not "
                      "adopted")
        self.assertIn(str(chosen["int8embed_projected_MiB"]), text,
                      "docs/limitations.md does not quote the size that blocks the adoption")

    def test_a_dimension_below_the_usable_threshold_is_not_hidden_by_the_macro(self):
        """Macro AUC can clear 0.70 with a dimension underneath it that does not.
        Whatever ships, that dimension may not be presented as working."""
        report = load()
        if report["selected"] is None:
            raise unittest.SkipTest("nothing selected")
        t = report["thresholds_inherited_from_increment_3"]["usable_held_out_auc"]
        chosen = variant(report, report["selected"])
        weak = {d: v["held_out_auc"] for d, v in chosen["per_dimension"].items()
                if v["held_out_auc"] < t}
        if not weak:
            raise unittest.SkipTest("every dimension clears the threshold")
        text = LIMITATIONS.read_text()
        for dim in weak:
            self.assertIn(dim, text,
                          f"{dim} is at {weak[dim]} under the selected scorer, below the "
                          f"{t} usable threshold, and docs/limitations.md does not say so")


class TestTheFindingIsWhereAReaderWillSeeIt(unittest.TestCase):
    def test_limitations_records_a_closed_option_b(self):
        report = load()
        if report["selected"] is not None:
            raise unittest.SkipTest("option B is open; this guard does not apply")
        self.assertTrue(LIMITATIONS.exists())
        text = LIMITATIONS.read_text().lower()
        self.assertIn("option b", text)
        best = report["option_c_basis"]["measured_under"]
        self.assertIn(str(variant(report, best)["macro_held_out_auc"]), text,
                      "docs/limitations.md does not quote the best readable variant's number")

    def test_readme_does_not_claim_a_polarity_fix_that_was_not_selected(self):
        report = load()
        if report["selected"] is not None:
            raise unittest.SkipTest("option B is open; this guard does not apply")
        readme = (ROOT / "README.md").read_text().lower()
        for phrase in ("polarity-aware scoring works", "solved the generalisation",
                       "the head now generalises", "nli scoring fixes"):
            self.assertNotIn(phrase, readme)

    def test_the_hypotheses_are_recorded_with_the_numbers(self):
        """The cross-encoder scores depend on the hypothesis wording. Recording it
        in the artifact is what makes 'we did not tune it' checkable."""
        report = load()
        self.assertEqual(len(report["hypotheses"]), 5)
        import sys
        sys.path.insert(0, str(ROOT / "export"))
        from scorer_ablation import HYPOTHESES
        self.assertEqual(report["hypotheses"], HYPOTHESES,
                         "the artifact's hypotheses differ from the ones in the script")


if __name__ == "__main__":
    unittest.main()
