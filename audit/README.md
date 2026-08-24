# audit/

The raw output of every measurement this project's claims rest on, and the
evidence ledger that indexes them.

Nothing here is written by hand. `runs/` holds the stdout of the scripts in
`export/` and of the test suite, captured at the moment each was run; `evidence.jsonl`
is one row per material claim, recording what was claimed, what was measured,
which artifact holds the number, and by what method.

The rule this directory exists to enforce is `plan.md`'s: **a README assertion is
not evidence.** If a number appears in `README.md` or `docs/limitations.md`, it
appears here first, in the output of something anyone can re-run.

| | |
|---|---|
| `evidence.jsonl` | 84 rows, `EV-H4H-001` .. `EV-H4H-084`. Append-only. |
| `runs/` | Captured stdout, named `<what>_<UTC timestamp>.txt`. |
| `runs/inc8_mutations.json` | Which guards catch which deliberate regressions. |

Re-derive everything:

```
bash export/run_all.sh              # build, quantize, verify against SIZE_BUDGET.md
python3 export/span_additivity.py   # R8-1, R8-2
python3 export/egress_audit.py      # R8-3
python3 export/tokenizer_parity.py  # DEFECT-INC8-001
python3 -m pytest tests/            # 215 guards
python3 export/mutation_check_inc8.py   # do those guards catch anything?
```

Two rows describe writes to systems outside this repository (registration and
the publication of this repository itself). Their read-backs are recorded in the
workspace that drove them, not here, because that workspace holds credentials
and is not published.
