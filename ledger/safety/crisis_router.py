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
#:
#: ``9`` was added by the round-2 pass: it stands for ``g`` in ordinary leet
#: ("no reason to 9o on") and for ``q`` about as often, so it goes here with the
#: other glyphs that are not resolved rather than into the single-valued table
#: above. Guessing ``g`` would have been right this time and wrong later.
_AMBIGUOUS_GLYPHS: dict[str, str] = {"1": "il", "!": "il", "|": "il", "9": "gq"}

#: Inverted: letter -> the glyphs that may appear in its place.
_GLYPHS_FOR_LETTER: dict[str, str] = {}
for _glyph, _letters in _AMBIGUOUS_GLYPHS.items():
    for _letter in _letters:
        _GLYPHS_FOR_LETTER[_letter] = _GLYPHS_FOR_LETTER.get(_letter, "") + _glyph

#: Round-2 audit finding **AUDR2-F-008 (a)**. Capital ``I`` is the commonest
#: stand-in for lowercase ``l`` there is, and it was invisible to the table above
#: because ``normalise`` lowercases first: ``kiII myself`` became ``kiii myself``
#: and missed, as did ``suicidaI``, ``seIf harm`` and ``end my Iife``.
#:
#: It is folded to ``!`` — an existing ambiguous glyph that already means
#: "``i`` or ``l``" — *before* the lowercase, because that is the only point at
#: which the case still exists to be read. Nothing new is needed in the matcher.
_UPPERCASE_AMBIGUOUS = str.maketrans({"I": "!"})

#: Round-2 audit finding **AUDR2-F-008 (b)**. ``_CONFUSABLES`` is a
#: hand-maintained enumeration, and anything not in it was deleted as non-ASCII —
#: which SPLITS the phrase it sits in, the exact failure the table was written to
#: stop. U+0455 turned ``ѕuicide`` into ``uicide``; U+04CF turned ``kiӏӏ myself``
#: into ``ki myself``.
#:
#: So an unmapped non-ASCII letter is no longer deleted. It becomes this
#: wildcard, which any letter position in a rule phrase may cross. The
#: enumeration above is kept because an exact letter beats a wildcard, but it is
#: no longer the only thing standing between a homoglyph and a missed phrase.
_WILDCARD = "�"

#: How much of a match may be wildcard before it stops being evidence.
#:
#: Without this, a run of any unmapped script is a run of wildcards, and seven
#: consecutive CJK characters would satisfy "suicide" — every rule would fire on
#: any sentence of Chinese, Japanese or Russian. A homoglyph evasion substitutes
#: one or two characters into an otherwise-Latin word; a foreign-script sentence
#: substitutes all of them. The cap is what tells those apart.
_MAX_WILDCARD_SHARE = 0.4

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
#: Ambiguous glyphs and the wildcard must survive this or the matcher below
#: never sees them.
_NON_ALNUM = re.compile(r"[^a-z0-9 !|" + _WILDCARD + r"]+")

#: Two or more of the same character in a row. AUDR2-F-008 (c).
_RUNS = re.compile(r"(.)\1+")


def normalise(text: str, *, pipe_as_letter: bool = False) -> str:
    """Fold text to a matchable form. Lossy on purpose, never shown to the user.

    Ambiguous glyphs (``1``, ``!``, ``|``) are left in place; resolving them is
    the matcher's job, not the normaliser's. An unmapped non-ASCII letter is left
    in place too, as ``_WILDCARD``, for the same reason: deleting it would decide
    that it meant nothing, and it usually means the letter it looks like.
    """
    folded = unicodedata.normalize("NFKD", text or "")
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    # Before the lowercase, because after it the capital I is gone. F-008 (a).
    folded = folded.translate(_UPPERCASE_AMBIGUOUS)
    folded = folded.lower().translate(_CONFUSABLES).translate(_LEET_UNAMBIGUOUS)
    # F-008 (b): what the enumeration did not name is now carried, not dropped.
    folded = "".join(_WILDCARD if (not c.isascii() and c.isalpha()) else c
                     for c in folded)
    folded = _NON_ALNUM.sub(" ", folded)
    separators = _SEPARATORS_KEEPING_PIPE if pipe_as_letter else _SEPARATORS
    return separators.sub(" ", folded).strip()


def collapse_runs(text: str) -> str:
    """Fold ``killl`` onto ``kil`` — and the rule phrase onto ``kil`` with it.

    AUDR2-F-008 (c): elongation walked straight past the matcher. Collapsing only
    the text would break every phrase containing a double letter, so both sides
    are collapsed and the collapsed forms are matched against each other.
    """
    return _RUNS.sub(r"\1", text)


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
    # AUDR2-F-008 (d). The list above reads like 2015: an acute router that does
    # not know "unalive" in 2026 does not know the word people actually use,
    # because the platforms that moderate the plain words are where the
    # euphemisms were forced into existence.
    #
    # PROVENANCE, stated because it differs from every rule above: these were
    # added by the round-2 remediation pass from the auditor's probe list. They
    # have NOT been clinician-reviewed. They are here because shipping a router
    # that misses them is worse than shipping them unreviewed, and the review is
    # named as outstanding in docs/limitations.md §7.8 rather than implied.
    Rule("R-ACUTE-EUPHEMISM", (
        "unalive", "unaliving", "sewerslide", "off myself", "offing myself",
        "delete myself", "end it all", "ending it all",
    ), "acute"),
    # "kms" is acute in meaning and ambiguous in form: it is also the plural of
    # km, and "ran 10 kms" is an ordinary sentence in a wellbeing journal. The
    # digit that would disambiguate it does not survive normalisation - the leet
    # table folds "5" to "s" and "0" to "o" - so a boundary rule cannot be
    # written honestly here.
    #
    # Elevated rather than acute is the deliberate choice: elevated surfaces the
    # helplines but does NOT set blocks_model_output, so a real "kms" is met with
    # resources and a runner's "10 kms" still gets its trend line. Ambiguity is
    # answered with a lower severity, not with silence. Flagged for the same
    # clinician pass.
    Rule("R-ELEVATED-EUPHEMISM", ("kms",), "elevated"),
)


def _pattern(phrase: str) -> re.Pattern[str]:
    """Compile one rule phrase so an ambiguous glyph satisfies the letter it mimics.

    "kill myself" becomes ``k[i1!|][l1!|][l1!|] mysel...`` - so ``k1ll``,
    ``ki11``, ``kil1`` and ``k!ll`` all match the one rule, and no reading of
    the glyph has to be guessed at normalisation time.

    Every letter position also admits ``_WILDCARD``, so an unmapped homoglyph
    crosses it (F-008 b). ``_hit`` then bounds how much of a match may be
    wildcard, because a position that admits anything is only useful while the
    positions around it do not.
    """
    out = []
    for ch in phrase:
        glyphs = _GLYPHS_FOR_LETTER.get(ch, "")
        admits = ch + glyphs + (_WILDCARD if ch.isalpha() else "")
        out.append(f"[{re.escape(admits)}]" if len(admits) > 1 else re.escape(ch))
    return re.compile("".join(out))


#: Matchers per phrase, compiled once at import. Pure data: the phrase list stays
#: the reviewable thing, and this is derived from it, so a clinician editing
#: RULES cannot forget to update the matcher.
#:
#: Two families, matched against text folded the same two ways — spaced and
#: de-spaced as before, and each of those collapsed for elongation (F-008 c).
#: Collapsed patterns are matched only against collapsed text: ``kill`` collapses
#: to ``kil``, so the two sides have to be folded together or neither.
_PHRASE_PATTERNS: dict[str, tuple[tuple[re.Pattern[str], ...],
                                  tuple[re.Pattern[str], ...]]] = {
    phrase: ((_pattern(phrase), _pattern(_despace(phrase))),
             (_pattern(collapse_runs(phrase)),
              _pattern(collapse_runs(_despace(phrase)))))
    for rule in RULES for phrase in rule.phrases
}


def _hit(pattern: re.Pattern[str], form: str) -> bool:
    """Search, then refuse a match that is mostly wildcard. See _MAX_WILDCARD_SHARE."""
    found = pattern.search(form)
    if found is None:
        return False
    matched = found.group()
    if not matched:
        return False
    return matched.count(_WILDCARD) / len(matched) <= _MAX_WILDCARD_SHARE


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
    plain = [f for variant in normalise_variants(text)
             for f in (variant, _despace(variant))]
    collapsed = [collapse_runs(f) for f in plain]

    hits: list[tuple[Rule, str]] = []
    for rule in RULES:
        for phrase in rule.phrases:
            patterns, collapsed_patterns = _PHRASE_PATTERNS[phrase]
            if (any(_hit(p, f) for p in patterns for f in plain)
                    or any(_hit(p, f) for p in collapsed_patterns for f in collapsed)):
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
