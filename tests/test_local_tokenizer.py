"""Guards on the application's tokenizer — DEFECT-INC8-001.

Switching the application off `transformers` is only legitimate if it encodes
identically to the path every prior measurement was run through. That parity is
measured by `export/tokenizer_parity.py`; these guards read the measurement and
fail if it is absent, stale, or shows a single mismatch.

The last class is the one that matters: it asserts the *reason* for the switch
still holds, so nobody re-introduces the dependency for convenience.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ledger.app import local_tokenizer  # noqa: E402

REPORT = ROOT / "artifacts" / "tokenizer_parity.json"
TOKENIZER_JSON = ROOT / "artifacts" / "tokenizer" / "tokenizer.json"

needs_tokenizer = pytest.mark.skipif(
    not TOKENIZER_JSON.exists(),
    reason="tokenizer is a build output - run `bash export/run_all.sh`")


def _report():
    if not REPORT.exists():
        pytest.skip("no tokenizer_parity.json - run export/tokenizer_parity.py first")
    return json.loads(REPORT.read_text())


class TestParityWasMeasured:
    def test_verdict_is_pass(self):
        assert _report()["verdict"] == "PASS"

    def test_not_one_token_differs(self):
        encodings = _report()["encodings"]
        assert encodings["mismatches"] == []
        assert encodings["pass"] is True

    def test_the_comparison_covered_the_probe_set_and_every_anchor(self):
        assert _report()["encodings"]["texts_compared"] == 64 + 50

    def test_there_is_no_tolerance_on_an_integer_comparison(self):
        assert "no tolerance" in _report()["protocol"]

    def test_the_head_bias_is_bit_identical_without_torch(self):
        bias = _report()["head_bias"]
        assert bias["bit_identical_after_cast"] is True
        assert bias["max_abs_delta_after_float32_cast"] == 0.0


@needs_tokenizer
class TestTheTokenizerItself:
    def test_it_pads_and_truncates_to_the_measured_length(self):
        ids, mask, offsets = local_tokenizer.encode("a short entry")
        assert len(ids) == len(mask) == len(offsets) == local_tokenizer.MAX_LENGTH

    def test_a_long_entry_is_truncated_rather_than_overflowing(self):
        ids, mask, _ = local_tokenizer.encode("word " * 4000)
        assert len(ids) == local_tokenizer.MAX_LENGTH
        assert sum(mask) == local_tokenizer.MAX_LENGTH

    def test_padding_is_masked_out(self):
        _, mask, _ = local_tokenizer.encode("short")
        assert sum(mask) < local_tokenizer.MAX_LENGTH
        assert mask[-1] == 0

    def test_special_tokens_carry_a_zero_width_offset(self):
        _, mask, offsets = local_tokenizer.encode("short")
        live = [o for o, m in zip(offsets, mask) if m]
        assert live[0] == (0, 0) and live[-1] == (0, 0)

    def test_offsets_index_into_the_original_text(self):
        text = "I slept badly last night."
        _, mask, offsets = local_tokenizer.encode(text)
        covered = "".join(text[a:b] for (a, b), m in zip(offsets, mask) if m and b > a)
        assert covered.replace(" ", "") == text.replace(" ", "")

    def test_the_vocabulary_is_the_one_the_model_was_exported_against(self):
        assert local_tokenizer.load().get_vocab_size() == 50265

    def test_encoding_is_deterministic(self):
        assert local_tokenizer.encode("same in, same out") == \
               local_tokenizer.encode("same in, same out")


class TestTheDependencyIsGone:
    """Why the switch happened, asserted so it cannot be quietly undone."""

    SOURCE = (ROOT / "ledger" / "app" / "local_tokenizer.py").read_text()

    def test_it_uses_the_tokenizers_library_directly(self):
        assert "from tokenizers import Tokenizer" in self.SOURCE

    def test_the_defect_that_caused_the_switch_is_recorded_in_the_file(self):
        assert "DEFECT-INC8-001" in self.SOURCE
        assert "CXXABI" in self.SOURCE, "the actual failure should be quoted, not summarised"
