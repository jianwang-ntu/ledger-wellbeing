"""Increment 8 guards on span aggregation — R8-1 and R8-2.

The partition property is checked here directly on synthetic offsets, so it
holds without a 200 MB model on disk. The measured additivity result is read
from `artifacts/span_additivity.json`, which `export/span_additivity.py`
produces; a guard that recomputed it would be measuring itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ledger.app.spans import (_assign, attribute_spans, sentence_spans,  # noqa: E402
                              word_spans)

REPORT = ROOT / "artifacts" / "span_additivity.json"


def _report():
    if not REPORT.exists():
        pytest.skip("no span_additivity.json - run export/span_additivity.py first")
    return json.loads(REPORT.read_text())


class TestSplitters:
    def test_sentence_spans_cover_the_sentences(self):
        text = "I slept badly. The meeting went fine! Did it? Yes."
        spans = sentence_spans(text)
        assert [text[s.start:s.end] for s in spans] == [
            "I slept badly.", "The meeting went fine!", "Did it?", "Yes."]

    def test_sentence_spans_survive_text_with_no_terminator(self):
        assert [s.text("no full stop here")
                for s in sentence_spans("no full stop here")] == ["no full stop here"]

    def test_newlines_split_and_blank_lines_do_not_produce_empty_spans(self):
        text = "first line\n\n\nsecond line\n"
        assert [s.text(text) for s in sentence_spans(text)] == ["first line", "second line"]

    def test_word_spans_exclude_whitespace(self):
        text = "  two   words  "
        assert [s.text(text) for s in word_spans(text)] == ["two", "words"]

    def test_empty_text_yields_no_spans(self):
        assert sentence_spans("") == [] and word_spans("") == []


class TestPartition:
    """R8-1. Every live token in exactly one bucket, always."""

    TEXT = "I slept badly. I could not settle."

    @staticmethod
    def _fake_encoding(text, n_structural=2, n_pad=3):
        """Offsets shaped like a real tokenizer's: specials at (0,0), then words."""
        offsets = [(0, 0)] * (n_structural - 1)
        for span in word_spans(text):
            offsets.append((span.start, span.end))
        offsets.append((0, 0))                       # closing special
        mask = [1] * len(offsets) + [0] * n_pad
        offsets += [(0, 0)] * n_pad                  # padding
        return offsets, mask

    @pytest.mark.parametrize("splitter", [sentence_spans, word_spans])
    def test_every_live_token_is_assigned_exactly_once(self, splitter):
        offsets, mask = self._fake_encoding(self.TEXT)
        buckets, structural = _assign(offsets, mask, splitter(self.TEXT))
        assigned = [i for bucket in buckets for i in bucket] + structural
        assert len(assigned) == sum(mask)
        assert len(set(assigned)) == len(assigned)

    def test_padding_tokens_are_never_assigned(self):
        offsets, mask = self._fake_encoding(self.TEXT, n_pad=5)
        buckets, structural = _assign(offsets, mask, word_spans(self.TEXT))
        assigned = {i for bucket in buckets for i in bucket} | set(structural)
        assert all(i not in assigned for i, m in enumerate(mask) if not m)

    def test_structural_tokens_are_reported_not_dropped(self):
        offsets, mask = self._fake_encoding(self.TEXT)
        _, structural = _assign(offsets, mask, word_spans(self.TEXT))
        assert len(structural) == 2, "the two specials must land in the structural bucket"

    def test_a_token_outside_every_span_falls_to_structural_not_to_nowhere(self):
        """Trailing punctuation that no span covers must still be counted."""
        text = "hello"
        offsets = [(0, 0), (0, 5), (20, 25), (0, 0)]     # (20,25) is past the text
        mask = [1, 1, 1, 1]
        buckets, structural = _assign(offsets, mask, word_spans(text))
        assert len([i for b in buckets for i in b] + structural) == 4
        assert 2 in structural

    def test_attribution_sums_match_the_buckets_exactly(self):
        offsets, mask = self._fake_encoding(self.TEXT)
        attr = [float(i + 1) for i in range(len(offsets))]
        spans = word_spans(self.TEXT)
        out, structural, n_structural = attribute_spans(self.TEXT, offsets, mask, attr, spans)
        live_total = sum(a for a, m in zip(attr, mask) if m)
        assert abs(sum(s.attribution for s in out) + structural - live_total) < 1e-9
        assert n_structural == 2
        assert sum(s.n_tokens for s in out) + n_structural == sum(mask)


class TestMeasuredAdditivity:
    """R8-2, read from the measurement rather than recomputed."""

    def test_the_measurement_passed(self):
        assert _report()["verdict"] == "PASS"

    def test_the_partition_held_on_every_probe_entry(self):
        report = _report()
        assert report["R8_1_partition"]["failures"] == []
        assert report["R8_1_partition"]["checks"] >= 128     # 64 entries x 2 granularities

    @pytest.mark.parametrize("granularity", ["sentence", "word"])
    def test_residual_is_within_the_token_level_tolerance(self, granularity):
        report = _report()
        result = report["R8_2_additivity"][granularity]
        assert result["max_residual"] <= report["tolerance"]
        assert result["checks"] == 64 * 5

    def test_the_tolerance_was_not_widened_for_the_aggregation_step(self):
        """R6-1's number, unchanged. A looser one here would hide the failure."""
        assert _report()["tolerance"] == 1e-4

    def test_the_measurement_names_the_build_it_measured(self):
        report = _report()
        assert report["build"] in {"int8_embed", "int8_full", "fp32"}
        assert len(report["model_sha256"]) == 64
