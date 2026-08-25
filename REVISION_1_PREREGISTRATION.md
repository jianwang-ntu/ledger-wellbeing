# Revision round 1 — pre-registration

Written and committed **before** any of the code it governs, per the discipline
of `export/INCREMENT_6/7/8/9_PREREGISTRATION.md`. Round 1 of the 2 budgeted by
`criteria`; round-1 audit scored **65.90 / 92.0**, `DO_NOT_SUBMIT`.

The audit is `audit/round1/verdict.json` in the orchestration workspace. It
raised 9 findings (2 HIGH, 3 MEDIUM, 4 LOW) and 4 **known gate FAILs**:
`G3_artifacts`, `G4_functional`, `G5_claims`, `G6_security`.

## What this round does NOT claim

Recorded first, because it is the fact that governs the round:

* **G3 cannot be closed here.** The organiser requires a ≤4-minute demonstration
  video (`overview.txt:282`). No video exists. It must not be faked, and this
  agent cannot record one. `R8-VIDEO-001` stays `NEEDS_HUMAN` and blocks
  submission at *any* audit score.
* The auditor put this artifact's realistic ceiling at **~81, not 92**, because
  the organiser band scale caps the official judge score near 60 for an
  instrument evaluated only on sentences written for it. Closing every finding
  below is therefore **not** predicted to reach the gate. It is done because the
  findings are true, and because a safety defect in a mental-health tool is
  worth fixing whether or not it is ever submitted.

## Acceptance rules, fixed now

Each rule states the instrument and the number **before** it is run. A rule that
is not met is reported as not met; no rule is rewritten after seeing a result.

### R1-1 — F-01 / G6: the crisis router's leetspeak fold is single-valued

`_LEET` maps `1 -> i` only, so `1` standing for `l` is never folded and every
`RULES` phrase containing an `l` can be written past the router.

**Closes when all four hold:**

1. The auditor's own probe `probe_crisis_leet_class.py`, **run unmodified**
   against the revised router, reports `fired 12 / 12`.
2. The auditor's own probe `probe_crisis_router.py`, run unmodified, reports
   `fired 24 / 24`.
3. A new in-repo matrix covers **every** `RULES` phrase containing `i` or `l`,
   in **both** readings of each ambiguous glyph (`1`, `!`, `|`), and fires on
   100% of them.
4. The false-positive control does not move: `MustNotFire.BENIGN` stays at 0
   triggers, and a new ≥40-entry ordinary-language control corpus triggers 0
   times. *Predicted before running: 0. If it is not 0, the widened matcher is
   reported as too broad and the count is published either way.*

### R1-2 — F-02 / G4: the published repository does not run

`artifacts/tokenizer/`, `artifacts/onnx/` and `artifacts/torch/` are gitignored
build outputs, so a clone fails 4 tests and cannot tokenize. README claims "All
tests pass offline", which is false of the clone.

**Closes when all four hold:**

1. `git clone` of the pushed repository into an empty directory, then
   `python -m pytest -q` from that clone, exits **0**.
2. The 1,556,504-byte compacted tokenizer ships in the repository (it already
   meets `CEIL-2`), so a clone can tokenize without a 900 MiB build.
3. Every test that genuinely needs the ONNX build **skips with a reason naming
   the command that produces it** — it does not pass vacuously.
4. README carries an install/build section stating the prerequisite and its
   **true cost** (bytes downloaded, wall-clock, peak disk), and the sentence
   "All tests pass offline" is replaced by one that is true of a clone.

### R1-3 — F-03 / G5: published numbers contradict the artifacts cited for them

6 of 20 traced figures do not match the artifact named as their evidence;
`845.98` appears in no artifact at all.

**Closes when both hold:**

1. The auditor's own `trace_claims.py`, run unmodified, reports
   `n_mismatch 0` of `n_checked 20`.
2. `export/check_published_numbers.py` exists, re-derives **every** published
   figure from the artifact that produced it, exits non-zero on any mismatch,
   and is asserted by a test — the discipline `audit/README.md` already claims
   is in force, made executable.

### R1-4 — F-04 / G6+G5: an unlocked journal is readable by any local client

Once one browser unlocks, the server keeps the key, `GET /` is unauthenticated,
the token is in the page body, and loopback TCP carries no UID check. A local
client that never supplies the passphrase reads plaintext.

**Closes when both hold:**

1. The auditor's own `probe_session_unlock.py`, run unmodified, no longer
   returns plaintext entry text to a client that supplied no passphrase.
2. `docs/limitations.md` 7.6 is rewritten: its current argument is wrong on both
   halves (`journal.enc` is ciphertext without the passphrase, and this host
   runs `yama ptrace_scope=1`).

### R1-5 — F-05: `docs/limitations.md` is stale at its own headline

Header says increment 5; repository is at increment 9. Section 1 reports the
superseded 0.504 body; the shipped body is at 0.880. `data/MANIFEST.md` still
describes the pre-increment-7 tokenizer.

**Closes when** sections 1, 3 and 6, the "not claimed anywhere" list, the
`Last updated` line and the `MANIFEST.md` tokenizer row all describe the
artifact that actually ships, verified by diffing each against
`export/common.py`, `artifacts/torch/build_report.json` and
`artifacts/verify_report.json`.

### R1-6 — F-06: the additivity residual is near-tautological as presented

`LedgerScorer.forward` computes `logits = token_attr.sum(dim=1) + bias`, so the
residual measures summation order, not a property that could have failed.

**Closes when** every artifact and document describing the residual calls it an
export/quantization regression guard rather than evidence for additivity, and
the contextual-embedding caveat (attributions are over contextual embeddings, so
exact additivity over the pooling layer is not encoder-level faithfulness) is
stated in `docs/limitations.md`. The architectural claim itself is **correct**
and is not withdrawn.

### R1-7 — F-07: the axe-core headline omits its mode

**Closes when** README states the mode beside the number and records the
forced-colours incomplete count with the one-line reason it is not a failure.

### R1-8 — F-08: `scorer.py` points at a test file that does not exist

**Closes when** the docstring names the tests that actually assert the identity
and a grep confirms no other dangling test reference in shipped source.

### R1-9 — F-09: the live ledger is not an entry in the freeze manifest

**Closes when** `audit/evidence.jsonl` is itself a manifest entry with frozen
bytes and digest, so a later round can distinguish an append from an edit.

## Ordering rule

R1-1 first, because it is the only finding that can hurt a user. Then R1-2 and
R1-3, the other two closable gate FAILs. Then R1-4. Then the documentation
findings. If the round runs out of time, the remainder are reported **open**,
not deferred silently.

## The rule that binds the round

Any rule above that is not met is written down as not met, with the measured
number. No acceptance threshold in this file may be changed after a measurement
that bears on it. If a fix widens the crisis matcher enough to produce a false
positive on ordinary language, that is published and the fix is reconsidered —
a guardrail that fires on everything is not a guardrail.
