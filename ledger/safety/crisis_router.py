"""Deterministic crisis routing.

The design rule this file exists to enforce: **the model is never in the crisis
path.** Everything here is rule-based, pure, and offline. A language model
cannot suppress this router's decision, because the router never consults one —
it runs on the raw text before any inference happens, and its output is not a
suggestion the rest of the pipeline may override.

Why that matters for this project's rubric: guardrails implemented as a prompt
or as a model's own judgement can be talked out of firing. This cannot, and the
test suite in tests/test_crisis_router.py is what shows it.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .resources import Helpline, for_region

# --- normalisation ---------------------------------------------------------
# Obfuscation seen in real self-harm text is mostly character substitution and
# separator insertion, not clever paraphrase. Normalising for it is cheap and
# is done BEFORE matching, so "k i l l  m y s e l f" and "k1ll myself" are the
# same string to the matcher.

# Glyphs that stand for exactly one letter are folded here, during
# normalisation, because there is nothing to decide about them.
_LEET_UNAMBIGUOUS = str.maketrans({"0": "o", "3": "e", "4": "a", "5": "s",
                                   "7": "t", "@": "a", "$": "s"})

#: Glyphs that could stand for MORE THAN ONE letter, and the letters they could
#: stand for. These are deliberately **not** folded during normalisation.
#:
#: Folding them to a single letter is the defect an independent audit found in
#: round 1 (F-01): the old table mapped ``1 -> i`` only, so ``1`` standing in
#: for ``l`` was never folded and every phrase in RULES containing an ``l`` -
#: "kill myself", "end my life", "self harm", "suicidal" - could be written
#: straight past the router. ``k1ll myse1f`` normalised to ``kill myseif`` and
#: missed. Ten of the auditor's twelve probes evaded that way.
#:
#: A single-valued fold has to guess. The matcher below does not: it expands the
#: PHRASE instead, so one glyph in the text can satisfy either letter and both
#: readings of ``k1ll myse1f`` are caught by the same rule.
_AMBIGUOUS_GLYPHS: dict[str, str] = {"1": "il", "!": "il", "|": "il"}

#: Inverted: letter -> the glyphs that may appear in its place.
_GLYPHS_FOR_LETTER: dict[str, str] = {}
for _glyph, _letters in _AMBIGUOUS_GLYPHS.items():
    for _letter in _letters:
        _GLYPHS_FOR_LETTER[_letter] = _GLYPHS_FOR_LETTER.get(_letter, "") + _glyph

# NFKD decomposes accents but NOT confusables: the dotless i (U+0131), the
# Cyrillic lookalikes and the Latin stroked letters all survive it unchanged and
# would then be stripped as non-ASCII, silently splitting the phrase they sit
# in. Found by tests/test_crisis_router.py, which routed "sui\u0131c\u0131de"
# straight past an earlier version of this normaliser.
_CONFUSABLES = str.maketrans({
    "\u0131": "i", "\u0142": "l", "\u00f8": "o", "\u0111": "d",
    "\u0138": "k", "\u017f": "s", "\u0127": "h", "\u01bf": "w",
    # Cyrillic homoglyphs
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p",
    "\u0441": "c", "\u0445": "x", "\u0443": "y", "\u0456": "i",
    "\u043a": "k", "\u043c": "m", "\u0442": "t", "\u0432": "b",
    # Greek homoglyphs
    "\u03bf": "o", "\u03b1": "a", "\u03b5": "e", "\u03c1": "p",
    "\u03c5": "y", "\u03ba": "k", "\u03c4": "t", "\u03b9": "i",
})
# "|" is genuinely two things at once: a separator in "kill|myself" and a
# stand-in for "l" in "k|ll". One normalisation has to pick, so the router does
# not pick - it matches against both readings. See normalise_variants().
_SEPARATORS = re.compile(r"[\s._\-*+|/\\]+")
_SEPARATORS_KEEPING_PIPE = re.compile(r"[\s._\-*+/\\]+")
#: Ambiguous glyphs must survive this or the matcher below never sees them.
_NON_ALNUM = re.compile(r"[^a-z0-9 !|]+")


def normalise(text: str, *, pipe_as_letter: bool = False) -> str:
    """Fold text to a matchable form. Lossy on purpose, never shown to the user.

    Ambiguous glyphs (``1``, ``!``, ``|``) are left in place; resolving them is
    the matcher's job, not the normaliser's.
    """
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.lower().translate(_CONFUSABLES).translate(_LEET_UNAMBIGUOUS)
    folded = _NON_ALNUM.sub(" ", folded)
    separators = _SEPARATORS_KEEPING_PIPE if pipe_as_letter else _SEPARATORS
    return separators.sub(" ", folded).strip()


def normalise_variants(text: str) -> tuple[str, ...]:
    """Every reading of the text the matcher must be satisfied by none of.

    Two, because "|" is a separator in ``kill|myself`` and a letter in
    ``k|ll``. Reading it one way loses the other, so both are matched.
    """
    both = (normalise(text), normalise(text, pipe_as_letter=True))
    return both if both[0] != both[1] else both[:1]


def _despace(text: str) -> str:
    """Collapse every space, so letter-spaced evasion collapses onto the phrase."""
    return text.replace(" ", "")


# --- rules -----------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    id: str
    phrases: tuple[str, ...]
    severity: str  # "acute" | "elevated"


#: Phrases are matched on normalised text and again on de-spaced normalised
#: text. They are intentionally literal: a rule you can read is a rule you can
#: audit, and this list is meant to be reviewed by a clinician, not tuned.
RULES: tuple[Rule, ...] = (
    Rule("R-ACUTE-SUICIDE", (
        "kill myself", "killing myself", "end my life", "ending my life",
        "take my own life", "taking my own life", "suicide", "suicidal",
        "want to die", "wanna die", "better off dead", "not want to be alive",
        "dont want to be alive", "do not want to be alive",
    ), "acute"),
    Rule("R-ACUTE-SELF-HARM", (
        "hurt myself", "hurting myself", "harm myself", "harming myself",
        "cut myself", "cutting myself", "self harm",
    ), "acute"),
    Rule("R-ACUTE-PLAN", (
        "have a plan to", "wrote a note", "goodbye note", "overdose",
    ), "acute"),
    Rule("R-ELEVATED-HOPELESS", (
        "no reason to go on", "no point in going on", "cant go on",
        "cannot go on", "nothing matters anymore", "hopeless",
    ), "elevated"),
)


def _pattern(phrase: str) -> re.Pattern[str]:
    """Compile one rule phrase so an ambiguous glyph satisfies the letter it mimics.

    "kill myself" becomes ``k[i1!|][l1!|][l1!|] mysel...`` - so ``k1ll``,
    ``ki11``, ``kil1`` and ``k!ll`` all match the one rule, and no reading of
    the glyph has to be guessed at normalisation time.
    """
    out = []
    for ch in phrase:
        glyphs = _GLYPHS_FOR_LETTER.get(ch)
        out.append(f"[{re.escape(ch + glyphs)}]" if glyphs else re.escape(ch))
    return re.compile("".join(out))


#: (spaced, de-spaced) matcher per phrase, compiled once at import. Pure data:
#: the phrase list stays the reviewable thing, and this is derived from it, so a
#: clinician editing RULES cannot forget to update the matcher.
_PHRASE_PATTERNS: dict[str, tuple[re.Pattern[str], re.Pattern[str]]] = {
    phrase: (_pattern(phrase), _pattern(_despace(phrase)))
    for rule in RULES for phrase in rule.phrases
}


@dataclass(frozen=True)
class Decision:
    triggered: bool
    severity: str | None
    rule_ids: tuple[str, ...]
    matched_phrases: tuple[str, ...]
    helplines: tuple[Helpline, ...] = field(default_factory=tuple)

    @property
    def blocks_model_output(self) -> bool:
        """True when the entry must not be sent for analysis or commentary.

        Acute matches route to resources and stop. The instrument's job at that
        point is to get a real human in front of the user, not to produce a
        trend line about them.
        """
        return self.severity == "acute"


def route(text: str, region: str | None = None) -> Decision:
    """Classify one journal entry. Pure: no I/O, no model, no network."""
    forms = [f for variant in normalise_variants(text)
             for f in (variant, _despace(variant))]

    hits: list[tuple[Rule, str]] = []
    for rule in RULES:
        for phrase in rule.phrases:
            spaced, dense = _PHRASE_PATTERNS[phrase]
            if any(spaced.search(f) or dense.search(f) for f in forms):
                hits.append((rule, phrase))
                break

    if not hits:
        return Decision(False, None, (), ())

    severity = "acute" if any(r.severity == "acute" for r, _ in hits) else "elevated"
    return Decision(
        triggered=True,
        severity=severity,
        rule_ids=tuple(r.id for r, _ in hits),
        matched_phrases=tuple(p for _, p in hits),
        helplines=for_region(region),
    )
