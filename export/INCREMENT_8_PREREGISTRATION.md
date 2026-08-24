# Increment 8 — pre-registration

Written and committed **before** the code it governs, per the discipline
established in `INCREMENT_6_PREREGISTRATION.md` and `INCREMENT_7_PREREGISTRATION.md`.

Increments 2–7 were all *model* increments. Increment 7 ended with `verify.py`
exiting 0 and `int8_embed` selected, so the model half now clears its own gate.
That gate is about **size and fidelity only**. It says the artifact is shippable;
it says nothing about whether there is anything to ship it inside.

## The gap this increment closes

`rules_canonical/overview.txt` → "What to Submit" asks for a GitHub repository
with **functioning source code**. What this repository contains is an export
pipeline, a safety module and a set of measurements. There is no application:

- no way for a person to enter anything,
- no storage, encrypted or otherwise,
- no attribution surface over their own words,
- no report.

So `plan.md` **C6** has zero artifact behind it, and **C4**'s two named
mechanisms — "local storage encrypted with a user-held passphrase, and a
one-click wipe" and "per-span attribution UI" — are unbuilt. An unshippable-but-
measured model plus no application cannot be submitted at all.

## Scope, fixed now

Increment 8 builds the **smallest honest end-to-end path**:

```
text in → deterministic crisis route → score → per-span attribution
        → encrypted local store → trajectory report
```

as a library (`ledger/store`, `ledger/app`) plus one working front-end
(`ledger/app/cli.py`).

**Explicitly NOT in increment 8**, so the hole is deliberate and not discovered
later as a shortfall:

| Not built | Why | Owed to |
|---|---|---|
| Graphical UI, visual design, keyboard/focus states | A CLI is a front-end, not a *visual design*. C6's band language is about visual quality and navigation. | increment 9 |
| `a11y/axe_report.json` | Needs a rendered DOM. There is no DOM until increment 9. | increment 9 |
| The 4-minute video | Requires a human. Must not be faked. | NEEDS_HUMAN |
| Any training | plan.md R-1 uncleared. | blocked |

A CLI satisfies "functioning source code". It does **not** satisfy C6, and this
increment may not be reported as if it did.

## The two decisive measurements

Most of this increment is ordinary engineering. Two parts are claims, and claims
get measured before they are made.

### Zero egress

`plan.md` differentiator 1 is *"Zero egress, provable"*. Until now that has been
true only by the accident of nothing existing that could send anything. Once
there is an application with a store and a report, it becomes a property that has
to be measured on the running system.

The measurement is `export/egress_audit.py`, committed before it is run.

### Span-level additivity

The exactly-additive identity (`R6-1`, residual ≤ 1e-4) is currently asserted at
the **token** level. The product surfaces *spans* — the user's words, not
byte-pair fragments. Aggregation is where an explanation quietly stops being the
score. If span attributions do not still sum to the logit, the explainability
claim is decoration, and `plan.md` C4 says explicitly that it must not be.

## Adoption rules, fixed now

| | Rule | Blind? |
|---|---|---|
| **R8-1** | Span aggregation is a **partition**: every non-padding token contributes to exactly one bucket, no token dropped, no token counted twice. Special/structural tokens (offset `(0,0)`) go to a named `structural` bucket that is *reported*, never silently discarded. Checked per entry, per dimension. | **yes** |
| **R8-2** | Span-level additivity: `abs(logit − (Σ span_attr + structural_attr + bias)) ≤ 1e-4` on all 64 probe entries × 5 dimensions, in the shipped `int8_embed` build. Same threshold as R6-1; no new tolerance is invented for the aggregation step. | **yes** |
| **R8-3** | **Zero egress, measured.** A full exercise of the application — init store, add N entries incl. crisis-routing ones, score, attribute, persist, report, wipe — with `socket.socket.connect` / `connect_ex` / `sendto` / `sendmsg` and `socket.getaddrinfo` / `gethostbyname` instrumented, records **zero** calls to any non-loopback address, and zero DNS resolutions. A single non-loopback attempt fails this rule, and the claim is **dropped, not softened**. | **yes** |
| **R8-4** | **The model is never in the crisis path.** For every entry the router marks `acute`, the ONNX session's `run()` is called **zero** times. Instrumented and counted, not argued from reading the code. | **yes** |
| **R8-5** | **Store confidentiality.** For every entry written: no contiguous 12-character substring of the entry text appears anywhere in the store file's bytes; opening with a wrong passphrase raises and returns no plaintext; a tampered byte anywhere in a record is rejected rather than decrypted. Key derivation is `hashlib.scrypt` and record encryption is AES-256-GCM from `cryptography` — **no hand-rolled cipher, no hand-rolled KDF, no hand-rolled MAC.** | **yes** |
| **R8-6** | **Wipe leaves nothing.** After wipe, the store path does not exist, and the bytes that were the store are overwritten before unlink. The limits of that guarantee on journalled and wear-levelled filesystems are stated in `docs/limitations.md` rather than left implied. | **yes** |
| **R8-7** | **No new claims about the model.** The head stays untrained (`head_is_trained: false`). `activation` — held-out AUC 0.600 against the 0.700 floor fixed in increment 3 — is **never presented as a working dimension** in any application output. Anywhere the application shows a dimension, it shows whether that dimension is established, and it names the evaluation as 25 withheld anchor-sentence pairs from this repository, not clinical data. | **yes** |
| **R8-8** | **No diagnosis, ever.** No application output states or implies a condition, a severity, a diagnosis or a treatment. Every report carries the non-diagnostic contract. Guarded by a banned-vocabulary test over the actual rendered output, not over the source. | **yes** |
| **R8-9** | No value in `CEILINGS` edited, no entry in `ENFORCED_BY_TARGET` changed, `verify.py` still exits 0 with `int8_embed` selected. | **yes** |

**If R8-3 fails**, `plan.md` differentiator 1 and the C4 "no remote component"
line are rewritten to say what was measured, in the same tick, before anything
else is reported.

**If R8-2 fails**, the product surfaces token-level attribution and says so, or
it surfaces nothing. It does not surface a span view that does not add up.

## A claim that is being narrowed on purpose, before it is measured

`plan.md` C4 says *"Client-side-only architecture with **no server component at
all**"*. That was written for a browser app. This increment ships a CLI, which
opens no socket of any kind, so the sentence holds today.

It will **not** survive increment 9 if the UI is served over loopback. Recording
the fork now, before the measurement that would be embarrassing to explain later:

> The claim this project can defend indefinitely is **"nothing leaves the
> device"**, measured. The claim "no server component at all" is an
> implementation detail that happens to be true today. If increment 9 binds a
> loopback listener, the second sentence is **dropped and replaced by the
> measured one** — the same way "in the browser" was dropped in increment 6 —
> and it is not quietly reinterpreted to mean "no *remote* server".

## Predictions

1. **R8-2 holds.** Aggregation is a regrouping of a sum, and float addition is
   associative to ~1e-7 at these magnitudes. *High.*
2. **R8-3 holds.** `onnxruntime` reads a local file; `transformers` is pointed at
   a local tokenizer directory. *Medium-high* — HF libraries have been known to
   phone home for revision checks unless `HF_HUB_OFFLINE` is set, and that is
   exactly the kind of thing this measurement exists to catch.
3. **R8-4 holds.** *High* — it is a control-flow property, but it has never been
   asserted, and "obviously true" is how it would get broken.
4. **R8-5/R8-6 hold.** *High.*
5. **The 103 existing tests still pass.** *High* — nothing in `export/` or
   `ledger/model/` is touched.

## What this increment must not be allowed to erase

The application makes the model *visible* for the first time. A visible score is
persuasive in a way a JSON file is not, and four of five dimensions clearing a
threshold on **withheld anchor sentences written for this repository** is not the
same kind of evidence as clearing one on real journal entries. The application's
own output has to carry that distinction — `R8-7` — because the video and the
README will not be where a judge forms their impression of what was established.
