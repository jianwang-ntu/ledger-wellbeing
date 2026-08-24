"""Regroup per-token attribution into spans of the user's own words.

The model's explanation is exact at the token level: mean pooling followed by a
linear head makes the logit a *sum* of per-token terms, so the per-token
contributions are the score rather than an approximation of it. That identity is
asserted on every build (`export/verify.py`, residual ~3e-07).

A person does not read byte-pair fragments. The product surfaces spans — words
and sentences — and aggregation is exactly where an explanation stops adding up
if nobody checks. So this module has one hard rule, pre-registered as **R8-1**:

    aggregation is a PARTITION.

Every token with a non-zero attention mask lands in exactly one bucket. Nothing
is dropped and nothing is counted twice. Tokens that correspond to no characters
in the user's text — ``<s>``, ``</s>``, and anything else the tokenizer inserts,
all of which carry offset ``(0, 0)`` — go into a named ``structural`` bucket that
is **reported**. They carry real weight and hiding them would make the visible
spans add up to something other than the score.

The consequence, checked in ``tests/test_spans.py`` as **R8-2**::

    logit == sum(span.attribution) + structural + bias      (to <= 1e-4)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["Span", "SpanAttribution", "word_spans", "sentence_spans", "attribute_spans"]

#: Sentence break: terminal punctuation followed by whitespace, or a newline.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Span:
    """A half-open character range of the user's text."""

    start: int
    end: int

    def text(self, source: str) -> str:
        return source[self.start:self.end]


@dataclass(frozen=True)
class SpanAttribution:
    """One span's contribution to one dimension's logit, in logit units."""

    start: int
    end: int
    text: str
    attribution: float
    n_tokens: int


def word_spans(text: str) -> list[Span]:
    return [Span(m.start(), m.end()) for m in _WORD.finditer(text)]


def sentence_spans(text: str) -> list[Span]:
    """Split on terminal punctuation, keeping character offsets into `text`."""
    spans, cursor = [], 0
    for match in _SENTENCE_BREAK.finditer(text):
        if match.start() > cursor:
            spans.append(Span(cursor, match.start()))
        cursor = match.end()
    if cursor < len(text):
        spans.append(Span(cursor, len(text)))
    return [s for s in spans if text[s.start:s.end].strip()]


def _assign(offsets, attention_mask, spans: list[Span]) -> tuple[list[list[int]], list[int]]:
    """Map each live token to exactly one span index, or to the structural bucket.

    A token is assigned to the first span it overlaps. Tokenizer offsets do not
    straddle whitespace, so for word spans this is unambiguous; for sentence
    spans a token can only straddle a boundary if the tokenizer merged across
    terminal punctuation, and first-overlap keeps that deterministic rather than
    double-counting it.
    """
    buckets: list[list[int]] = [[] for _ in spans]
    structural: list[int] = []
    for token_index, (mask, (start, end)) in enumerate(zip(attention_mask, offsets)):
        if not int(mask):
            continue                      # padding: masked to exactly zero upstream
        start, end = int(start), int(end)
        if end <= start:
            structural.append(token_index)
            continue
        for span_index, span in enumerate(spans):
            if start < span.end and end > span.start:
                buckets[span_index].append(token_index)
                break
        else:
            structural.append(token_index)
    return buckets, structural


def attribute_spans(text: str, offsets, attention_mask, token_attr, spans: list[Span]):
    """Return (span attributions, structural attribution) for one dimension.

    `token_attr` is the per-token contribution column for a single dimension, in
    logit units, straight out of the graph.
    """
    buckets, structural = _assign(offsets, attention_mask, spans)
    out = [
        SpanAttribution(
            start=span.start, end=span.end, text=text[span.start:span.end],
            attribution=float(sum(token_attr[i] for i in bucket)),
            n_tokens=len(bucket),
        )
        for span, bucket in zip(spans, buckets)
    ]
    return out, float(sum(token_attr[i] for i in structural)), len(structural)
