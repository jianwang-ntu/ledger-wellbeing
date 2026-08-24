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
| `evidence.jsonl` | `EV-H4H-001` onwards. Append-only. |
| `runs/` | Captured stdout, named `<what>_<UTC timestamp>.txt`. |
| `runs/inc8_mutations.json` | Which guards catch which deliberate regressions. |
| `runs/inc9_mutations.json` | The same, for increment 9's interface and accessibility guards. Two of them were too weak on the first run; both were tightened. |

Re-derive everything:

```
bash export/run_all.sh              # build, quantize, verify against SIZE_BUDGET.md
python3 export/span_additivity.py   # R8-1, R8-2
python3 export/egress_audit.py      # R8-3
python3 export/tokenizer_parity.py  # DEFECT-INC8-001
python3 export/egress_audit.py --ui  # R9-7, R9-8(a): the interface's listener
python3 -m pytest tests/            # 241 guards
python3 export/mutation_check_inc8.py   # do those guards catch anything?
python3 export/mutation_check_inc9.py   # and increment 9's? (19/19)
(cd a11y && npm install) && python3 a11y/audit_a11y.py   # R9-1..R9-9, needs a browser
```

`export/verify.py` and anything else that imports `onnx` must be run in the build
virtualenv (`../.venv-export/bin/python`), not the interpreter the application
uses — the application deliberately does not have `onnx`, `torch` or
`transformers` installed.

Two rows describe writes to systems outside this repository (registration and
the publication of this repository itself). Their read-backs are recorded in the
workspace that drove them, not here, because that workspace holds credentials
and is not published.
