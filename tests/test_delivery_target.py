"""Build increment 6 guards: the target may move, the ceilings may not.

Increment 6 stopped enforcing CEIL-1 and CEIL-3 because the artifact stopped
being an HTTP download. That is the single most self-serving-looking move in this
repository, so it gets the most guards. Each test here fails if the relaxation is
ever widened past what `export/INCREMENT_6_PREREGISTRATION.md` fixed *before* any
increment-6 number existed.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
sys.path.insert(0, str(ROOT))

import common  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
VERIFY_REPORT = ARTIFACTS / "verify_report.json"


def _report():
    if not VERIFY_REPORT.exists():
        pytest.skip("no verify_report.json - run export/verify.py first")
    return json.loads(VERIFY_REPORT.read_text())


class TestNoCeilingValueMoved:
    """The day-1 numbers. Increment 5 already guarded CEIL-1; this covers the rest."""

    def test_every_ceiling_holds_its_day_one_value(self):
        assert common.CEILINGS == {
            "CEIL_1_int8_model_bytes": 32 * 1024 * 1024,
            "CEIL_2_tokenizer_bytes": 2 * 1024 * 1024,
            "CEIL_3_cold_payload_bytes": 64 * 1024 * 1024,
            "CEIL_4_p95_latency_ms": 500.0,
            "CEIL_5_min_pearson_r": 0.99,
            "CEIL_5_max_abs_score_delta": 0.02,
        }

    def test_ort_runtime_floor_is_not_quietly_reduced(self):
        assert common.ORT_RUNTIME_FLOOR_BYTES == 49_856 + 11_815_498


class TestTheRelaxationIsBounded:
    """Exactly two ceilings stopped gating, and only the two that bound a download."""

    def test_web_target_still_enforces_all_six(self):
        assert set(common.ENFORCED_BY_TARGET["web"]) == set(common.CEILINGS)

    def test_desktop_drops_only_ceil_1_and_ceil_3(self):
        dropped = set(common.ENFORCED_BY_TARGET["web"]) - set(common.ENFORCED_BY_TARGET["desktop"])
        assert dropped == {"CEIL_1_int8_model_bytes", "CEIL_3_cold_payload_bytes"}

    def test_ceil_2_ceil_4_ceil_5_still_gate_the_desktop_target(self):
        # CEIL-2 is the one that actually failed in increment 6. If a later
        # increment wants it re-scoped, that has to be an explicit decision that
        # breaks this test - not a quiet edit to the map.
        for key in ("CEIL_2_tokenizer_bytes", "CEIL_4_p95_latency_ms",
                    "CEIL_5_min_pearson_r", "CEIL_5_max_abs_score_delta"):
            assert key in common.ENFORCED_BY_TARGET["desktop"], key

    def test_no_third_target_can_be_added_without_touching_this_test(self):
        assert set(common.ENFORCED_BY_TARGET) == {"web", "desktop"}

    def test_ceil_4_basis_per_target(self):
        assert common.CEIL_4_RUNTIME_BY_TARGET == {
            "web": "wasm_1thread", "desktop": "native_ort_cpu_1thread"}


class TestTheDroppedClaimStaysVisible:
    """A ceiling that stopped gating must still be measured and still be reported."""

    def test_every_build_reports_all_six_ceilings(self):
        for name, build in _report()["candidate_builds"].items():
            assert set(build["ceiling_checks"]) == set(common.CEILINGS), name

    def test_every_build_carries_its_web_target_failures(self):
        for name, build in _report()["candidate_builds"].items():
            assert "would_fail_web_target_on" in build, name

    def test_the_web_failure_is_recorded_not_hidden(self):
        # This is the substance of the relaxation. If a future build genuinely
        # fits a browser, this test is the one that should be deleted - on
        # purpose, with the measurement that justifies it.
        for name, build in _report()["candidate_builds"].items():
            assert "CEIL_1_int8_model_bytes" in build["would_fail_web_target_on"], name


class TestPreRegistrationIsNotRewritten:
    """The rule was fixed before the measurement. Keep it that way."""

    PRE = ROOT / "export" / "INCREMENT_6_PREREGISTRATION.md"

    def test_the_preregistration_exists(self):
        assert self.PRE.exists()

    def test_it_names_every_adoption_rule(self):
        text = self.PRE.read_text()
        for rule in ("R6-1", "R6-2", "R6-3", "R6-4", "R6-5", "R6-6"):
            assert rule in text, rule

    def test_it_states_the_ceil_4_basis_change_as_a_relaxation(self):
        text = self.PRE.read_text().lower()
        assert "relaxation" in text
        assert "native" in text and "wasm" in text


class TestAdditivitySurvivedTheBodySwap:
    """R6-1. If the explanation stops being the score, C4's claim is dropped."""

    def test_residual_is_within_the_pre_registered_rule(self):
        for name, build in _report()["candidate_builds"].items():
            assert build["attribution_identity_max_residual"] <= 1e-4, name
            assert build["attribution_identity_holds"] is True, name


class TestStaleBenchesAreRejected:
    """DEFECT-INC6-001. A bench with no model identity is not evidence."""

    def test_verify_reports_a_rejection_channel(self):
        assert "wasm_benches_rejected_as_stale" in _report()

    def test_accepted_benches_match_the_model_on_disk(self):
        import hashlib
        for path in sorted((ARTIFACTS / "wasm").glob("bench_*.json")):
            rec = json.loads(path.read_text())
            sha = rec.get("model_sha256")
            if sha is None:
                continue                      # a stale file; verify.py rejects it
            model = ROOT / rec["model_path"]
            if not model.exists():
                continue
            h = hashlib.sha256()
            with open(model, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            assert sha == h.hexdigest(), path.name

    def test_bench_script_emits_the_sha(self):
        assert "model_sha256" in (ROOT / "web" / "bench_wasm.mjs").read_text()


class TestTheHeadIsStillNotTrained:
    """R6-5. Swapping the body trains nothing. Nothing here is a fine-tune."""

    def test_build_report_says_so(self):
        path = ARTIFACTS / "torch" / "build_report.json"
        if not path.exists():
            pytest.skip("no build_report.json")
        assert json.loads(path.read_text())["head_is_trained"] is False

    #: Claims that would be false while head_is_trained is False. A whitelist of
    #: "allowed" wordings around the word fine-tune proved too clever to be a
    #: guard, so this checks for the specific false statements instead.
    FORBIDDEN = (
        "our model is fine-tuned",
        "we fine-tuned",
        "we fine-tune",
        "fine-tuned on our",
        "a fine-tune of ledger",
        "ledger is fine-tuned",
        "the head is fine-tuned",
        "the head is trained",
        "trained on a mental-health corpus",
    )

    def test_no_artifact_claims_a_fine_tune_of_ours(self):
        for path in (ROOT / "README.md", ROOT / "data" / "MANIFEST.md",
                     ROOT / "export" / "SIZE_BUDGET.md", ROOT.parent / "plan.md"):
            if not path.exists():
                continue
            low = path.read_text().lower()
            for claim in self.FORBIDDEN:
                assert claim not in low, f"{path.name}: {claim!r}"


class TestTheArtifactAndThePlanAgree:
    """The target moved. Anything still claiming a browser is now false."""

    def test_common_py_pins_the_selected_body(self):
        assert common.BASE_MODEL == "sentence-transformers/nli-distilroberta-base-v2"
        assert common.BASE_REVISION == "cc35a0bfb6251228a6fb8c797bca5fef0ece3c1d"

    def test_delivery_target_is_desktop(self):
        assert common.DELIVERY_TARGET == "desktop"

    def test_readme_does_not_promise_in_browser_inference(self):
        text = (ROOT / "README.md").read_text().lower()
        for claim in ("runs entirely in the user's browser",
                      "runs in your browser",
                      "inference runs under `onnxruntime-web`"):
            assert claim not in text, claim
