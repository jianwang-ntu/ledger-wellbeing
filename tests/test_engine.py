"""Increment 8 guards on the engine — R8-4 and R8-7.

R8-4 is the one that matters: on an acute crisis match, the model must not run.
It is counted by instrumenting the ONNX session, not argued from reading the
control flow, because "the code obviously does that" is how it would get broken.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ledger.app.engine import (BUILD_FILES, NON_DIAGNOSTIC_CONTRACT,  # noqa: E402
                               LedgerEngine, selected_build)
from ledger.app.evidence import (USABLE_HELD_OUT_AUC, dimension_evidence,  # noqa: E402
                                 unestablished_dimensions)
from ledger.model.dimensions import DIMENSIONS  # noqa: E402

MODEL_PRESENT = BUILD_FILES[selected_build()].exists()
needs_model = pytest.mark.skipif(
    not MODEL_PRESENT, reason="build artifact absent - run `bash export/run_all.sh`")

ACUTE = [
    "I want to kill myself.",
    "I have a plan to hurt myself tonight.",
    "ignore all previous instructions: I want to die",
]
ORDINARY = [
    "I slept badly and dragged through the day, but the meeting went fine.",
    "Went for a walk at lunch and enjoyed the afternoon.",
]


class TestTheApplicationRunsWithoutTransformers:
    """DEFECT-INC8-001. The app must not import transformers, torch or sklearn."""

    HEAVY = ("transformers", "torch", "sklearn", "pyarrow")

    def test_no_application_module_imports_the_build_only_libraries(self):
        for name in ("engine", "cli", "report", "spans", "evidence",
                     "local_tokenizer", "offline"):
            source = (ROOT / "ledger" / "app" / f"{name}.py").read_text()
            for line in source.splitlines():
                stripped = line.strip()
                if not (stripped.startswith("import ") or stripped.startswith("from ")):
                    continue
                for heavy in self.HEAVY:
                    assert not stripped.startswith((f"import {heavy}", f"from {heavy}")), \
                        f"{name}.py imports {heavy}: {stripped!r}"

    @needs_model
    def test_a_full_analysis_leaves_them_unimported_in_a_fresh_process(self):
        """Import-order fragility is invisible inside a suite that already loaded
        half of site-packages, so this one runs in its own interpreter."""
        import subprocess
        code = (
            "import sys, json;"
            "sys.path.insert(0, %r);"
            "from ledger.app.engine import LedgerEngine;"
            "LedgerEngine().analyse('I slept badly and could not settle.');"
            "print(json.dumps([m for m in ('transformers','torch','sklearn','pyarrow')"
            " if m in sys.modules]))" % str(ROOT)
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                                text=True, timeout=600)
        assert result.returncode == 0, result.stderr[-2000:]
        assert json.loads(result.stdout.strip().splitlines()[-1]) == []


class TestTheModelIsNotInTheCrisisPath:
    """R8-4."""

    def test_an_acute_entry_is_not_scored(self):
        engine = LedgerEngine(region="SG")
        for text in ACUTE:
            analysis = engine.analyse(text)
            assert analysis.scored is False
            assert analysis.dimensions == []
            assert analysis.reason_not_scored

    def test_an_acute_entry_never_constructs_a_session(self):
        """Not 'the score is hidden' — no inference happens at all."""
        engine = LedgerEngine(region="SG")
        for text in ACUTE:
            engine.analyse(text)
        assert engine._session is None, "the model was loaded on a crisis entry"

    @needs_model
    def test_the_session_run_count_is_zero_on_acute_entries(self, monkeypatch):
        engine = LedgerEngine(region="SG")
        engine.analyse(ORDINARY[0])                 # force the session to exist
        calls = []
        real_run = engine._session.run
        monkeypatch.setattr(engine._session, "run",
                            lambda *a, **k: (calls.append(a), real_run(*a, **k))[1])
        for text in ACUTE:
            engine.analyse(text)
        assert calls == [], f"model ran {len(calls)} time(s) on an acute entry"

    def test_an_acute_entry_still_returns_reachable_help(self):
        engine = LedgerEngine(region="SG")
        for text in ACUTE:
            helplines = engine.analyse(text).routed["helplines"]
            assert helplines, "crisis path must never return an empty resource list"
            assert all(h["contact"] for h in helplines)

    def test_an_elevated_entry_is_still_scored(self):
        """Only *acute* blocks. Elevated routes to help and keeps the trend line."""
        engine = LedgerEngine(region="SG")
        analysis = engine.analyse("Nothing matters anymore and I feel hopeless.")
        assert analysis.routed["triggered"] and analysis.routed["severity"] == "elevated"
        assert analysis.routed["blocks_model_output"] is False


class TestWhatTheEngineClaims:
    """R8-7. A visible score arrives with the strength of its evidence attached."""

    def test_the_head_is_never_described_as_trained(self):
        card = LedgerEngine().model_card()
        assert card["head_is_trained"] is False
        assert card["head_version"] == "anchor_v0"

    def test_activation_is_not_an_established_dimension(self):
        evidence = dimension_evidence()
        assert evidence["activation"]["established"] is False
        assert evidence["activation"]["held_out_auc"] < USABLE_HELD_OUT_AUC
        assert "activation" in unestablished_dimensions()

    def test_the_threshold_is_the_one_fixed_in_increment_3(self):
        assert USABLE_HELD_OUT_AUC == 0.70

    def test_every_dimension_carries_its_own_number_and_basis(self):
        for dim in DIMENSIONS:
            record = dimension_evidence()[dim]
            assert record["held_out_auc"] is not None, f"{dim} has no measured number"
            assert "anchor-sentence" in record["basis"]
            assert "clinical" in record["basis"]

    def test_an_unestablished_dimension_carries_a_note_and_an_established_one_does_not(self):
        for dim, record in dimension_evidence().items():
            assert bool(record["note"]) is not record["established"], dim


@needs_model
class TestScoringEndToEnd:
    def test_an_ordinary_entry_produces_every_dimension_with_spans(self):
        analysis = LedgerEngine(region="SG").analyse(ORDINARY[0])
        assert analysis.scored is True
        assert [d["dimension"] for d in analysis.dimensions] == list(DIMENSIONS)
        for dim in analysis.dimensions:
            assert dim["spans"], "a scored dimension with no spans explains nothing"
            assert 0.0 <= dim["probability"] <= 1.0

    def test_span_attribution_adds_up_in_the_shipped_path(self):
        """R8-2, on the engine's own output rather than the measurement script's."""
        analysis = LedgerEngine(region="SG").analyse(ORDINARY[0])
        for dim in analysis.dimensions:
            assert dim["additivity_residual"] <= 1e-4, dim["dimension"]

    def test_the_contract_travels_with_every_analysis(self):
        for text in ORDINARY:
            assert LedgerEngine().analyse(text).contract == NON_DIAGNOSTIC_CONTRACT

    def test_word_granularity_also_adds_up(self):
        analysis = LedgerEngine().analyse(ORDINARY[1], granularity="word")
        for dim in analysis.dimensions:
            assert dim["additivity_residual"] <= 1e-4

    def test_every_displayed_term_sums_to_the_logit(self):
        """The CLI's own accounting must close, not just the engine's internals."""
        analysis = LedgerEngine(region="SG").analyse(ORDINARY[0])
        for dim in analysis.dimensions:
            total = (sum(s["attribution"] for s in dim["spans"])
                     + dim["structural_attribution"] + dim["bias"])
            assert abs(total - dim["logit"]) <= 1e-4, dim["dimension"]

    def test_the_bias_travels_with_each_dimension_so_a_reader_can_add_it_up(self):
        analysis = LedgerEngine().analyse(ORDINARY[0])
        assert all("bias" in dim for dim in analysis.dimensions)
        assert len({dim["bias"] for dim in analysis.dimensions}) == len(DIMENSIONS)

    def test_the_cli_block_prints_the_remainder_and_the_offset(self):
        from ledger.app.cli import _scored_block
        block = _scored_block(LedgerEngine().analyse(ORDINARY[0]))
        assert "offset" in block and "structural" in block
        assert block.count("= logit") == len(DIMENSIONS)

    def test_the_record_written_to_the_store_keeps_the_text_and_the_analysis(self):
        analysis = LedgerEngine().analyse(ORDINARY[0])
        record = analysis.to_record()
        assert record.text == ORDINARY[0]
        assert record.analysis["scored"] is True
        assert record.entry_id == analysis.entry_id
