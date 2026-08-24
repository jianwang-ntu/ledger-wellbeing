"""Increment 7 guards: CEIL-2 was MET, and it must stay met for the right reason.

The tokenizer was re-serialized compactly rather than replaced, shrunk or
truncated. Meeting a ceiling that way is only legitimate if the artifact on disk
is still the same tokenizer, so these tests re-tokenize rather than trust the
report - and they fail if a later increment meets CEIL-2 by making the tokenizer
smaller in substance instead of in whitespace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "export"))
sys.path.insert(0, str(ROOT))

import common  # noqa: E402

TOKENIZER_DIR = ROOT / "artifacts" / "tokenizer"
TOKENIZER_JSON = TOKENIZER_DIR / "tokenizer.json"
REPORT = ROOT / "artifacts" / "compact_tokenizer.json"

#: Fixed in export/INCREMENT_7_PREREGISTRATION.md (R7-3), before the rewrite.
EXPECTED_VOCAB = 50265
EXPECTED_MERGES = 50000


def _report():
    if not REPORT.exists():
        pytest.skip("no compact_tokenizer.json - run export/compact_tokenizer.py")
    return json.loads(REPORT.read_text())


def _doc():
    if not TOKENIZER_JSON.exists():
        pytest.skip("no tokenizer.json - run export/build_model.py")
    return json.loads(TOKENIZER_JSON.read_text(encoding="utf-8"))


class TestTheTokenizerIsStillTheWholeTokenizer:
    """CEIL-2 must not be met by dropping vocabulary."""

    def test_vocabulary_is_intact(self):
        model = _doc()["model"]
        assert len(model["vocab"]) == EXPECTED_VOCAB
        assert len(model["merges"]) == EXPECTED_MERGES

    def test_no_leftover_backup_is_inflating_or_hiding_the_measurement(self):
        assert not (TOKENIZER_DIR / "tokenizer.json.orig").exists(), \
            "a staged backup was left on disk; dir_bytes would count it"

    def test_the_directory_meets_ceil_2(self):
        measured = common.dir_bytes(TOKENIZER_DIR)
        assert measured <= common.CEILINGS["CEIL_2_tokenizer_bytes"], measured

    def test_it_is_met_by_serialization_not_by_a_smaller_vocabulary(self):
        """The whole justification. Pretty-printing the SAME document must still
        exceed the ceiling - otherwise this was never a serialization win and the
        reasoning in SIZE_BUDGET.md increment 7 is wrong."""
        pretty = json.dumps(_doc(), indent=2, ensure_ascii=False).encode("utf-8")
        assert len(pretty) > common.CEILINGS["CEIL_2_tokenizer_bytes"], \
            "the pretty form now fits too - the increment-7 argument no longer applies"


class TestEncodeOutputIsUnchanged:
    """R7-2, re-run against whatever is on disk now rather than against a report."""

    def test_encode_matches_the_recorded_fingerprint(self):
        transformers = pytest.importorskip("transformers")
        tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        assert tok.vocab_size == EXPECTED_VOCAB

        from ledger.model.dimensions import ANCHORS
        anchors = [s for dim in sorted(ANCHORS)
                   for pole in sorted(ANCHORS[dim]) for s in ANCHORS[dim][pole]]

        rep = _report()
        assert rep["R7_2_encode_identity"]["anchor_sentences"] == len(anchors), \
            "the anchor set changed size since the identity was established"

        # A tokenizer that had been truncated or swapped would not round-trip.
        for text in anchors:
            ids = tok(text)["input_ids"]
            assert tok.decode(ids, skip_special_tokens=True).strip() == text.strip()


class TestTheIncrementDidNotRewriteItsOwnRules:
    PRE = ROOT / "export" / "INCREMENT_7_PREREGISTRATION.md"

    def test_the_preregistration_exists(self):
        assert self.PRE.exists()

    def test_it_names_every_adoption_rule(self):
        text = self.PRE.read_text()
        for rule in ("R7-1", "R7-2", "R7-3", "R7-4", "R7-5", "R7-6"):
            assert rule in text, rule

    def test_it_discloses_that_the_size_probe_was_not_blind(self):
        text = self.PRE.read_text().lower()
        assert "not blind" in text or "is not blind" in text
        assert "1,556,145" in self.PRE.read_text()

    def test_the_report_agrees_it_was_not_blind(self):
        assert _report()["R7_1_size"]["blind"] is False
        assert _report()["R7_2_encode_identity"]["blind"] is True

    def test_the_vocabulary_growth_is_not_erased(self):
        """CEIL-2's tripwire fired on a real 1.65x vocabulary change. Meeting the
        ceiling on bytes does not retire that, and the artifact must say so."""
        for path in (REPORT, ROOT / "export" / "SIZE_BUDGET.md"):
            if not path.exists():
                continue
            text = path.read_text()
            assert "50,265" in text or "50265" in text, path.name
            assert "30,522" in text or "30522" in text, path.name


class TestNoCeilingMovedForThis:
    def test_ceil_2_is_still_2_mib(self):
        assert common.CEILINGS["CEIL_2_tokenizer_bytes"] == 2 * 1024 * 1024

    def test_ceil_2_still_gates_the_desktop_target(self):
        assert "CEIL_2_tokenizer_bytes" in common.ENFORCED_BY_TARGET["desktop"]
