"""Increment 8 guards on the report — R8-7 and R8-8.

The banned-vocabulary check runs over the **rendered output**, not the source.
A test that greps `report.py` for the word "depression" passes the moment the
word arrives via an f-string, which is exactly how it would arrive.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ledger.app.report import CONTRACT, render  # noqa: E402
from ledger.model.dimensions import DIMENSION_LABELS  # noqa: E402


def flat(text: str) -> str:
    """Collapse the renderer's line wrapping, so assertions are about words.

    A phrase assertion that breaks when a paragraph rewraps is testing the
    column width, which nothing depends on.
    """
    return " ".join(text.split())

#: Words that would turn an observation into a clinical statement. Deliberately
#: broad: this list should be annoying to a future author, because a report that
#: reaches for any of them has stopped being an instrument.
BANNED = [
    "diagnos",          # diagnosis, diagnostic, diagnosed
    "depression", "depressive", "depressed",
    "anxiety disorder", "gad", "phq",
    "insomnia", "bipolar", "psychosis", "disorder",
    "symptom", "severity", "clinically", "clinical significance",
    "treatment", "medication", "prescri", "therapy plan",
    "you should", "we recommend", "advice", "at risk of",
    "improving", "worsening", "deteriorat", "recovery trend",
]


def _entry(entry_id, day, text, *, scored=True, triggered=False, probability=0.5,
           spans=None, dimension="low_mood"):
    analysis = {
        "scored": scored,
        "routed": {"triggered": triggered, "severity": "acute" if triggered else None},
        "dimensions": [] if not scored else [{
            "dimension": dimension,
            "label": DIMENSION_LABELS[dimension],
            "probability": probability,
            "spans": spans or [{"start": 0, "end": len(text), "text": text,
                                "attribution": 0.4, "n_tokens": 5}],
        }],
    }
    return {"entry_id": entry_id, "written_at": f"2026-08-{day:02d}T09:00:00Z",
            "text": text, "analysis": analysis}


ENTRIES = [
    _entry("a", 10, "I slept badly and could not settle.", probability=0.72),
    _entry("b", 11, "Better today, I went for a walk.", probability=0.31),
    _entry("c", 12, "I want to kill myself.", scored=False, triggered=True),
]


class TestNoClinicalLanguage:
    """R8-8, over the rendered text."""

    @staticmethod
    def _body(entries) -> str:
        """The report minus the disclaimer.

        The contract has to be able to say the word "diagnosis" in order to deny
        making one. Exempting exactly that block — and nothing else — keeps the
        ban meaningful: any other line reaching for clinical vocabulary still
        fails. The exemption is by exact text, so it cannot be widened by
        accident.
        """
        return flat(render(entries)).replace(flat(CONTRACT), "").lower()

    def test_the_report_uses_no_banned_vocabulary(self):
        found = [word for word in BANNED if word in self._body(ENTRIES)]
        assert found == [], f"clinical vocabulary in the report: {found}"

    def test_an_empty_report_also_uses_none(self):
        assert [w for w in BANNED if w in self._body([])] == []

    def test_the_exemption_covers_the_contract_and_nothing_more(self):
        """If the disclaimer were dropped, the ban would have nothing to exempt."""
        assert "diagnos" in flat(CONTRACT).lower()
        assert "diagnos" not in self._body(ENTRIES)

    def test_the_dimension_labels_themselves_are_about_language_not_conditions(self):
        for label in DIMENSION_LABELS.values():
            assert "language" in label.lower()


class TestTheContract:
    def test_every_report_carries_it(self):
        for entries in (ENTRIES, [], ENTRIES[:1]):
            assert flat(CONTRACT) in flat(render(entries))

    def test_it_appears_before_any_number(self):
        rendered = flat(render(ENTRIES))
        assert rendered.index("not a diagnosis") < rendered.index("Entries in this store")

    def test_it_says_nothing_was_sent_anywhere(self):
        assert "Nothing in it was sent anywhere" in flat(render(ENTRIES))


class TestUnestablishedDimensionsAreLabelled:
    """R8-7, in the surface the user and any clinician actually reads."""

    def test_activation_is_marked_not_established_wherever_it_appears(self):
        """"Wherever" means every section, not at least one of them.

        The first mutation run flagged this: blanking the trajectory marker left
        the "what moved each score" marker in place and the guard stayed green.
        A reader who looks only at the trajectory would have seen no warning.
        """
        entries = [_entry("a", 10, "Busy day, lots done.", dimension="activation",
                          probability=0.8)]
        rendered = render(entries)
        sections = rendered.split("WHAT MOVED EACH SCORE")
        assert len(sections) == 2, "the report lost a section this guard depends on"
        for name, section in zip(("trajectory", "what moved each score"), sections):
            assert "[NOT ESTABLISHED]" in section, f"no warning in the {name} section"
        assert "0.60" in rendered
        assert "has not been shown to work" in flat(rendered)

    def test_an_established_dimension_is_not_flagged(self):
        rendered = render([_entry("a", 10, "Heavy day.", dimension="low_mood")])
        assert "NOT ESTABLISHED" not in rendered

    def test_the_evaluation_basis_is_stated(self):
        rendered = flat(render(ENTRIES))
        assert "anchor-sentence pairs" in rendered
        assert "Not clinical data" in rendered

    def test_the_head_is_described_as_zero_shot_not_trained(self):
        assert "zero-shot anchor head, not a trained model" in flat(render(ENTRIES))


class TestWhatTheReportShows:
    def test_crisis_routed_entries_are_counted_but_not_scored(self):
        rendered = render(ENTRIES)
        assert "Entries scored        : 2" in rendered
        assert "Entries routed to help: 1" in rendered

    def test_the_safety_section_says_the_rule_cannot_be_overridden(self):
        assert "cannot be overridden" in flat(render(ENTRIES))

    def test_the_safety_count_agrees_with_itself_grammatically(self):
        one = render(ENTRIES)
        assert "1 entry matched a crisis rule and was routed" in flat(one)
        two = render(ENTRIES + [_entry("d", 13, "I want to die.", scored=False,
                                       triggered=True)])
        assert "2 entries matched a crisis rule and were routed" in flat(two)

    def test_it_says_the_contributions_are_the_score(self):
        assert "adds up to the score exactly" in flat(render(ENTRIES))

    def test_it_does_not_claim_the_LISTED_contributions_are_the_whole_sum(self):
        """Found by the first end-to-end run, not by a test.

        A dimension came out at 0.60 with all three displayed spans negative,
        because the offset and the unshown spans carry the rest. Text that says
        "the contributions listed above are the score" is false whenever more
        than TOP_SPANS spans exist, which is almost always.
        """
        rendered = flat(render(ENTRIES))
        assert "The contributions listed above are the score" not in rendered
        assert "Only the largest few contributions are listed" in rendered
        for term in ("every span of the entry", "structural tokens",
                     "one fixed offset per dimension"):
            assert term in rendered, term

    def test_the_section_heading_says_it_is_showing_only_the_largest(self):
        assert "WHAT MOVED EACH SCORE  (largest 3" in render(ENTRIES)

    def test_an_empty_store_says_so_rather_than_drawing_an_empty_chart(self):
        assert "no trajectory to show" in flat(render([]))

    def test_the_sparkline_scale_is_fixed_not_autoscaled(self):
        """Two entries a hair apart must not render as a dramatic swing."""
        near = [_entry("a", 10, "x", probability=0.50),
                _entry("b", 11, "y", probability=0.51)]
        line = [ln for ln in render(near).splitlines() if "latest" in ln][0]
        bars = line.strip().split()[0]
        assert len(set(bars)) == 1, f"near-identical values rendered as a swing: {bars!r}"

    def test_rendering_is_deterministic(self):
        assert render(ENTRIES) == render(ENTRIES)
