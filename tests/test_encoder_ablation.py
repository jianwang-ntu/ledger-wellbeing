"""The encoder ablation's conclusions are re-derived here from its own numbers.

Same discipline as tests/test_export_pipeline.py: a hand-edited verdict, a
selection that the stated rule would not have made, or a repository that has
quietly switched encoders on a null selection all fail here rather than passing
review. These tests skip - not pass - when the artifact has not been built.
"""

import json
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ABLATION = ROOT / "artifacts" / "encoder_ablation.json"
MANIFEST = ROOT / "data" / "MANIFEST.md"


def load():
    if not ABLATION.exists():
        raise unittest.SkipTest("artifacts/encoder_ablation.json not built - "
                                "run export/encoder_ablation.py")
    return json.loads(ABLATION.read_text())


class TestTheProtocolValidatesItself(unittest.TestCase):
    """A separation test that cannot separate two obviously different topics is
    measuring its own bias, and nothing else in the report may be read."""

    def test_positive_control_holds_for_every_candidate(self):
        report = load()
        floor = report["positive_control_floor"]
        for cand in report["candidates"]:
            control = cand["anchor_separation"]["positive_control_macro"]
            self.assertGreaterEqual(
                control, floor,
                f"{cand['model']}: positive control {control} < {floor}; the held-out "
                f"numbers in this report are not interpretable")

    def test_trustworthy_flag_matches_the_measurements(self):
        report = load()
        derived = all(c["anchor_separation"]["positive_control_macro"]
                      >= report["positive_control_floor"] for c in report["candidates"])
        self.assertEqual(report["protocol_trustworthy"], derived)

    def test_in_sample_is_reported_and_is_not_mistaken_for_a_result(self):
        """In-sample AUC on a 5-vs-5 centroid split is ~1.0 for any encoder. Its
        presence next to the held-out number is what stops it being quoted alone."""
        for cand in load()["candidates"]:
            sep = cand["anchor_separation"]
            self.assertIn("macro_in_sample_auc", sep)
            self.assertIn("macro_held_out_auc", sep)


class TestSelectionFollowsTheStatedRule(unittest.TestCase):
    def test_selection_is_recomputable(self):
        report = load()
        threshold = report["usable_held_out_auc_threshold"]
        eligible = [c for c in report["candidates"]
                    if c["anchor_separation"]["macro_held_out_auc"] >= threshold
                    and c["int8embed_size"]["clears_CEIL_1"]]
        expected = (max(eligible, key=lambda c: c["anchor_separation"]["macro_held_out_auc"])["model"]
                    if eligible else None)
        self.assertEqual(report["selected"], expected,
                         f"selected={report['selected']} but the stated rule yields {expected}")

    def test_no_candidate_is_selected_while_failing_a_gate_of_the_rule(self):
        report = load()
        if report["selected"] is None:
            return
        chosen = next(c for c in report["candidates"] if c["model"] == report["selected"])
        self.assertTrue(chosen["int8embed_size"]["clears_CEIL_1"])
        self.assertGreaterEqual(chosen["anchor_separation"]["macro_held_out_auc"],
                                report["usable_held_out_auc_threshold"])


class TestTheRepositoryHonoursTheSelection(unittest.TestCase):
    """A null selection means the encoder does not change. This is the guard
    against shipping a smaller model on the strength of a metric that is at
    chance for every candidate."""

    def test_base_model_matches_the_selection(self):
        report = load()
        import sys
        sys.path.insert(0, str(ROOT / "export"))
        sys.path.insert(0, str(ROOT / "tests"))
        from common import BASE_MODEL, BASE_REVISION

        if report["selected"] is None:
            incumbent = next(c for c in report["candidates"]
                             if c["role"].startswith("incumbent"))
            if BASE_MODEL != incumbent["model"]:
                # Increment 6 moved the pin under plan.md R-4. THIS ablation still
                # selected nothing, so it may not be the thing that moved it: the
                # move has to be legal under the shared invariant, which requires
                # some ablation to have selected the new pin and every blocker it
                # recorded to be unenforced on the current target. See
                # tests/pin_invariant.py for why this is stricter, not looser.
                from pin_invariant import pin_is_legal
                legal, why = pin_is_legal()
                self.assertTrue(legal, "encoder_ablation.json selected nothing and the pin "
                                       f"moved anyway: {why}")
                return
            self.assertEqual(BASE_REVISION, incumbent["revision"])
        else:
            self.assertEqual(BASE_MODEL, report["selected"])
            self.assertEqual(BASE_REVISION, report["selected_revision"])

    def test_manifest_attributes_the_encoder_actually_in_use(self):
        """Hackathon rule 2 is attribution. A stale MANIFEST is a rule problem,
        not a documentation problem."""
        if not MANIFEST.exists():
            raise unittest.SkipTest("data/MANIFEST.md absent")
        import sys
        sys.path.insert(0, str(ROOT / "export"))
        sys.path.insert(0, str(ROOT / "tests"))
        from common import BASE_MODEL, BASE_REVISION

        text = MANIFEST.read_text()
        self.assertIn(BASE_MODEL, text)
        self.assertIn(BASE_REVISION, text)

    def test_every_model_downloaded_by_the_ablation_is_attributed(self):
        report = load()
        if not MANIFEST.exists():
            raise unittest.SkipTest("data/MANIFEST.md absent")
        text = MANIFEST.read_text()
        for cand in report["candidates"]:
            self.assertIn(cand["model"], text,
                          f"{cand['model']} was downloaded and measured but is not attributed "
                          f"in data/MANIFEST.md")


class TestFindingsAreNotSilentlyDropped(unittest.TestCase):
    """When held-out separation is below the usable threshold, the repository has
    to say so where a reader will find it."""

    def test_limitations_records_a_failing_head(self):
        report = load()
        incumbent = next(c for c in report["candidates"] if c["role"].startswith("incumbent"))
        if incumbent["anchor_separation"]["macro_held_out_auc"] >= report["usable_held_out_auc_threshold"]:
            raise unittest.SkipTest("head clears the usable threshold; this guard does not apply")
        limitations = ROOT / "docs" / "limitations.md"
        self.assertTrue(limitations.exists(),
                        "held-out separation is below the usable threshold and docs/limitations.md "
                        "does not exist")
        text = limitations.read_text().lower()
        self.assertIn("held-out", text)
        self.assertIn(str(incumbent["anchor_separation"]["macro_held_out_auc"]), text)

    def test_readme_does_not_advertise_a_score_the_measurement_does_not_support(self):
        report = load()
        incumbent = next(c for c in report["candidates"] if c["role"].startswith("incumbent"))
        if incumbent["anchor_separation"]["macro_held_out_auc"] >= report["usable_held_out_auc_threshold"]:
            raise unittest.SkipTest("head clears the usable threshold; this guard does not apply")
        readme = (ROOT / "README.md").read_text().lower()
        for phrase in ("accurately scores", "validated score", "detects depression",
                       "measures anxiety", "clinically validated", "accurate signal"):
            self.assertNotIn(phrase, readme,
                             f"README claims {phrase!r} while held-out separation is "
                             f"{incumbent['anchor_separation']['macro_held_out_auc']}")


if __name__ == "__main__":
    unittest.main()
