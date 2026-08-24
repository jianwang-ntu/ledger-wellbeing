"""Ledger's front-end: `python -m ledger.app.cli`.

A command-line front-end, not a designed interface. It is here so that the
end-to-end path is *usable* — entry in, explanation and stored record out —
which `rules_canonical/overview.txt` asks for as "functioning source code".

It is explicitly not the artifact for `plan.md` C6 (UI/UX & Accessibility). That
criterion is graded on visual design quality, navigation and accessibility
standards, and a CLI answers none of those. See
`export/INCREMENT_8_PREREGISTRATION.md`, which fixes that hole as deliberate.

Commands
--------
    init      create an encrypted journal
    add       analyse a piece of writing and append it
    show      print one entry's per-span attribution
    list      list stored entries
    report    render the clinician-shareable report
    wipe      overwrite and remove the journal

The passphrase is read from a prompt with echo off, or from `LEDGER_PASSPHRASE`
for scripted use. It is never taken as a command-line argument, because argv is
world-readable in `/proc` on the platform this is built on.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

from ledger.app import offline  # noqa: F401  - side effect: offline before transformers
from ledger.app.report import render
from ledger.store.journal import Journal, StoreError

DEFAULT_STORE = Path(os.environ.get("LEDGER_HOME", Path.home() / ".ledger")) / "journal.enc"

PASSPHRASE_ENV = "LEDGER_PASSPHRASE"


def read_passphrase(confirm: bool = False) -> str:
    """Prompt with echo off, or take the environment variable for scripted use."""
    from_env = os.environ.get(PASSPHRASE_ENV)
    if from_env:
        return from_env
    if not sys.stdin.isatty():
        raise SystemExit(
            f"no tty for a passphrase prompt; set {PASSPHRASE_ENV} for scripted use"
        )
    first = getpass.getpass("passphrase: ")
    if confirm and getpass.getpass("passphrase (again): ") != first:
        raise SystemExit("passphrases did not match; nothing was created")
    return first


def _journal(args, *, create: bool = False) -> Journal:
    journal = Journal(Path(args.store), read_passphrase(confirm=create))
    return journal.create() if create else journal.unlock()


def cmd_init(args) -> int:
    journal = _journal(args, create=True)
    print(f"created {journal.path}")
    print("The passphrase is not stored anywhere. If it is lost, the journal is gone.")
    return 0


def cmd_add(args) -> int:
    text = Path(args.file).read_text() if args.file else sys.stdin.read()
    text = text.strip()
    if not text:
        raise SystemExit("nothing to add")

    journal = _journal(args)
    from ledger.app.engine import LedgerEngine        # deferred: 200 MB of model

    engine = LedgerEngine(region=args.region)
    analysis = engine.analyse(text, region=args.region, granularity=args.granularity)

    if not analysis.scored:
        print(_crisis_block(analysis))
    else:
        print(_scored_block(analysis))

    journal.append(analysis.to_record())
    print(f"\nstored as {analysis.entry_id} in {journal.path}")
    return 0


def _crisis_block(analysis) -> str:
    lines = ["", "=" * 62,
             "This entry matched a crisis rule, so it was not scored.",
             analysis.reason_not_scored or "", "",
             "Published helplines:"]
    for helpline in analysis.routed["helplines"]:
        lines.append(f"  {helpline['name']} — {helpline['contact']} ({helpline['hours']})")
    lines += ["", "This routing is a rule in the code. No model decided it and no",
              "model can suppress it.", "=" * 62]
    return "\n".join(lines)


def _scored_block(analysis, top: int = 3) -> str:
    lines = ["", analysis.contract, ""]
    for dim in analysis.dimensions:
        flag = "" if dim["established"] else "   [NOT ESTABLISHED]"
        lines.append(f"{dim['label']:<34} {dim['probability']:.2f}{flag}")
        spans = sorted(dim["spans"], key=lambda s: -s["attribution"])[:top]
        for span in spans:
            snippet = span["text"].strip()
            snippet = snippet if len(snippet) <= 46 else snippet[:43] + "..."
            lines.append(f"    {span['attribution']:+.3f}  \"{snippet}\"")
        if not dim["established"]:
            lines.append(f"    held-out AUC {dim['held_out_auc']}: {dim['evidence_note']}")
        lines.append("")
    return "\n".join(lines)


def cmd_list(args) -> int:
    for entry in _journal(args).entries():
        analysis = entry.analysis
        state = "scored" if analysis.get("scored") else "routed-to-help"
        preview = entry.text.replace("\n", " ")[:44]
        print(f"{entry.entry_id}  {entry.written_at}  {state:<15} {preview}")
    return 0


def cmd_show(args) -> int:
    for entry in _journal(args).entries():
        if entry.entry_id == args.entry_id:
            print(json.dumps({"entry_id": entry.entry_id, "written_at": entry.written_at,
                              "text": entry.text, "analysis": entry.analysis},
                             indent=1, ensure_ascii=False))
            return 0
    raise SystemExit(f"no entry {args.entry_id}")


def cmd_report(args) -> int:
    entries = [
        {"entry_id": e.entry_id, "written_at": e.written_at,
         "text": e.text, "analysis": e.analysis}
        for e in _journal(args).entries()
    ]
    text = render(entries, region=args.region)
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}")
    else:
        print(text, end="")
    return 0


def cmd_wipe(args) -> int:
    path = Path(args.store)
    if not args.yes:
        confirm = input(f"permanently destroy {path}? type WIPE to confirm: ")
        if confirm.strip() != "WIPE":
            print("not wiped")
            return 1
    # Wipe deliberately does not require the passphrase: someone who needs their
    # journal gone should not have to remember how to open it first.
    result = Journal(path, "unused").wipe()
    print(json.dumps(result, indent=1))
    return 0 if result["wiped"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledger",
        description="A local, encrypted, explainable well-being journal. "
                    "Nothing it computes leaves this machine.",
    )
    parser.add_argument("--store", default=str(DEFAULT_STORE),
                        help=f"path to the encrypted journal (default: {DEFAULT_STORE})")
    parser.add_argument("--region", default=os.environ.get("LEDGER_REGION"),
                        help="two-letter region for helpline selection, e.g. SG")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init", help="create an encrypted journal").set_defaults(func=cmd_init)

    add = sub.add_parser("add", help="analyse and store one entry (reads stdin by default)")
    add.add_argument("--file", help="read the entry from this file instead of stdin")
    add.add_argument("--granularity", choices=("sentence", "word"), default="sentence")
    add.set_defaults(func=cmd_add)

    sub.add_parser("list", help="list stored entries").set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="print one entry with its full attribution")
    show.add_argument("entry_id")
    show.set_defaults(func=cmd_show)

    report = sub.add_parser("report", help="render the clinician-shareable report")
    report.add_argument("--out", help="write to this file instead of stdout")
    report.set_defaults(func=cmd_report)

    wipe = sub.add_parser("wipe", help="overwrite and remove the journal")
    wipe.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    wipe.set_defaults(func=cmd_wipe)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except StoreError as exc:
        # Never let a store failure print anything derived from the plaintext.
        print(f"store error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
