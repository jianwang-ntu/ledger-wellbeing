# Ledger

A client-side longitudinal well-being instrument, built for
**Hack for Humanity | Summer 2026** (focus: Mental health).

> **Status: in progress.** This README records only what is built and has been
> run. Claims about parts that do not exist yet are not made here. Every claim
> below is backed by a checked-in artifact under `../audit/`.

## What it is meant to be

An instrument that makes a person's own between-visits experience legible to the
clinician actually treating them — not an AI that acts as a therapist. Inference
runs in the user's browser over their own writing; the output is per-span
attribution over their words and a trajectory over time, not generated advice.

## What is built and verified so far

| Component | State | Runtime evidence |
|---|---|---|
| `ledger/safety/` — deterministic crisis router | **built, tested** | `../audit/runs/safety_tests_20260824T1350Z.txt` (8/8), `../audit/runs/crisis_matrix_20260824T1350Z.json` (23/23) |
| Model export / quantization pipeline | not started | — |
| In-browser inference | not started | — |
| UI | not started | — |

## The safety layer

Crisis routing is rule-based and runs **before** any inference. The model is
never in the crisis path, so no model output can suppress a routing decision —
the router does not consult one. `tests/test_crisis_router.py` is written to try
to make it fail to fire: leetspeak, letter-spacing, separator insertion,
homoglyphs, and five prompt-injection attempts that instruct the system to stay
silent.

That suite has already caught one real evasion in this code: the dotless-i
homoglyph `U+0131` survived NFKD normalisation, so `suıcıde` routed past an
earlier version of the normaliser. The fix is the `_CONFUSABLES` table in
`ledger/safety/crisis_router.py`; the case is now a permanent test.

```
python3 -m unittest discover -s tests -v
```

## How this was created — disclosure

This entry is being developed by an autonomous coding agent (Claude, Anthropic)
operating as the entrant's tool, on the entrant's own machine and accounts. The
entrant is a solo participant.

This is disclosed deliberately and in advance. Rule 3 of the hackathon states
"You may not receive any assistance from others outside of your group"; we read
that clause as governing assistance from other *people*, and rule 2 — "You must
only use resources and assets you have access to and must attribute
appropriately" — as the clause governing tools, which requires attribution
rather than abstention. If the organisers read rule 3 as covering AI agents,
they learn it here rather than discovering it.

## Licence

MIT — see `LICENSE`.
