# Increment 7 — pre-registration

Written and committed **before** the decisive measurement, per the discipline
established in `INCREMENT_6_PREREGISTRATION.md`. Increment 6 ended with exactly
one enforced ceiling unmet, so this increment has exactly one question.

## The question

Increment 6 swapped the body to `sentence-transformers/nli-distilroberta-base-v2`
and adopted nothing, because **CEIL-2 failed at 3,559,258 B against 2,097,152 B**
— 70% over. Every other enforced ceiling passed: CEIL-4 at 224.98 ms native
against 500 ms, CEIL-5 at r 0.99995 / max Δ 0.00694 against 0.99 / 0.02.

> **Is there a serialization of this tokenizer — same 50,265-entry vocabulary,
> identical `encode()` output — that fits CEIL-2's 2 MiB?**

`artifacts/tokenizer/tokenizer.json` is written by `tokenizers` in a
pretty-printed form with two-space indentation. JSON whitespace carries no
semantics. Re-serializing the *same document* compactly is a way to **meet** the
ceiling on its own terms, not to argue it away — but that only holds if the
re-serialized file drives the tokenizer to identical output, and that has to be
established by re-tokenizing, not by trusting a file size.

## Disclosure: one half of this is not blind

Honesty about the order things happened in, because a pre-registration that
overstates its own blindness is worth less than none.

**A size probe was already run before this file was written.** Parsing
`tokenizer.json` and re-dumping it with `separators=(',',':')` produced
**1,556,145 B** (`ensure_ascii=False`) and **1,842,761 B** (`ensure_ascii=True`).
So R7-1 below is a *confirmation*, not a prediction, and is labelled as such.

What was **not** run, and is genuinely pre-registered here, is every rule that
decides adoption: R7-2 (encode identity), R7-3 (document identity), R7-5
(verify.py), R7-6 (the test suite). The size number alone adopts nothing.

## Adoption rules, fixed now

| | Rule | Blind? |
|---|---|---|
| **R7-1** | `dir_bytes(artifacts/tokenizer)` ≤ 2,097,152 B after the rewrite | **no** — probed at 1,556,504 B, disclosed above |
| **R7-2** | Re-tokenizing after the rewrite returns `input_ids` and `attention_mask` **exactly equal** to the pre-rewrite tokenizer, elementwise, on (a) the 64 probe entries at `max_length=256` and (b) every anchor sentence in `ledger/model/dimensions.py`. Also identical `offset_mapping` and identical `decode()` round-trip. **A single mismatch anywhere fails this rule.** | **yes** |
| **R7-3** | The parsed JSON **object** after the rewrite compares equal to the parsed object before it, and `vocab_size == 50265`, and the merges table length is unchanged. Same document, different whitespace — or it is not a re-serialization. | **yes** |
| **R7-4** | No value in `CEILINGS` edited and no entry in `ENFORCED_BY_TARGET` changed. Already guarded by `tests/test_delivery_target.py::TestNoCeilingValueMoved` and `tests/test_size_feasible_scorer.py::TestNoCeilingMoved`. | **yes** |
| **R7-5** | If R7-1..R7-4 all pass: `export/verify.py` exits **0** with `int8_embed` in `shippable_builds`. If it exits 1 for any *other* reason, that reason is published and **nothing is adopted**. | **yes** |
| **R7-6** | Unchanged from increment 6: the head stays untrained behind plan.md R-1; `activation` stays 0.600 and is never presented as a working dimension. | **yes** |

**If R7-2 or R7-3 fails, the rewrite is discarded outright** and the choice
between re-scoping CEIL-2 to the desktop target and abandoning this body is made
explicitly, as its own decision, with 3.394 MiB already published. It is not to
be folded into an engineering commit.

## Predictions

1. **R7-2 holds.** JSON whitespace is not semantic to `tokenizers`. *High.*
2. **R7-3 holds.** *High.*
3. **R7-5 gives exit 0 with `int8_embed`.** CEIL-2 was measured in increment 6 as
   the sole failing enforced ceiling. *Medium-high* — the other ceilings are
   re-measured in the same run and could drift.
4. **The 87 existing tests still pass.** *Medium.* No test reads the tokenizer's
   byte count, but the export pipeline is re-run and the suite is broad.

## The thing this increment must not be allowed to erase

CEIL-2's stated purpose on day 1 was to *"force a rejection if someone swaps in a
250k-entry multilingual vocab without saying so"*. **A 50,265-entry byte-level
BPE vocabulary really was swapped in for a 30,522-entry WordPiece one — 1.65×
larger — and the tripwire fired on a real change.** Meeting the ceiling by
compact serialization satisfies the ceiling *as written*, which bounds bytes on
disk. It does **not** retire the signal, and it must not be reported as though
the vocabulary had not grown.

Two limits of this route, stated in advance so they are not discovered as
excuses later:

- Compact serialization removes indentation, which is close to what any
  compressed transport would have removed anyway. The win is real against a
  ceiling denominated in raw bytes on disk, and thin against one denominated in
  transfer size. CEIL-2 is the former.
- It buys nothing on CEIL-1/CEIL-3/CEIL-4-on-web. The web target stays lost on
  all four, exactly as increment 6 published it.

Both go into `SIZE_BUDGET.md` and `docs/limitations.md` whichever way the
measurement lands.
