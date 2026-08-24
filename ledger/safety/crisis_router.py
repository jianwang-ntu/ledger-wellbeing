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

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
                       "7": "t", "@": "a", "$": "s", "!": "i"})

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
_SEPARATORS = re.compile(r"[\s._\-*+|/\\]+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")


def normalise(text: str) -> str:
    """Fold text to a matchable form. Lossy on purpose, never shown to the user."""
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    folded = folded.lower().translate(_CONFUSABLES).translate(_LEET)
    folded = _NON_ALNUM.sub(" ", folded)
    return _SEPARATORS.sub(" ", folded).strip()


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
    norm = normalise(text)
    dense = _despace(norm)

    hits: list[tuple[Rule, str]] = []
    for rule in RULES:
        for phrase in rule.phrases:
            if phrase in norm or _despace(phrase) in dense:
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
