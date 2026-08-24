"""Do increment 9's guards actually catch anything?

Same discipline as `mutation_check_inc8.py`: break each property the increment
claims, run the guard that is supposed to notice, record whether it did, revert.

Two kinds of guard are exercised, because increment 9 has two kinds of claim:

* **unit guards** (`tests/test_ui.py`) for the server's posture — bind address,
  Host pin, token, content policy, path containment;
* **the accessibility harness** (`a11y/audit_a11y.py`) for the rendered-document
  claims — focus visibility, reduced motion, live regions, contrast, axe.

The second kind is slow, about a minute per run, and it is the one that most
needs mutating. An accessibility report that has only ever been run against
correct markup is a description of that markup, not a check on it.

Run: ``python3 export/mutation_check_inc9.py``  (``--fast`` skips the harness)
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (label, file, old, new, pytest selector)
UNIT_MUTATIONS: list[tuple[str, str, str, str, str]] = [
    ("the listener binds every interface instead of loopback",
     "ledger/ui/server.py", 'BIND_HOST = "127.0.0.1"', 'BIND_HOST = "0.0.0.0"',
     "tests/test_ui.py::TestTheListenerIsLocal::test_it_binds_loopback_and_nothing_else"),
    ("the Host pin is dropped, reopening DNS rebinding",
     "ledger/ui/server.py",
     "        return host in {f\"{BIND_HOST}:{port}\", f\"localhost:{port}\"}",
     "        return True",
     "tests/test_ui.py::TestTheListenerIsLocal::test_a_foreign_host_header_is_refused"),
    ("binding resolves a name again",
     "ledger/ui/server.py",
     "        self.server_name = BIND_HOST",
     "        self.server_name = socket.getfqdn(BIND_HOST)",
     "tests/test_ui.py::TestTheListenerIsLocal::test_binding_does_not_resolve_a_name"),
    ("the API token is no longer checked",
     "ledger/ui/server.py",
     "        return secrets.compare_digest(\n"
     "            self.headers.get(\"X-Ledger-Token\") or \"\", self.state.token)",
     "        return True",
     "tests/test_ui.py::TestTheTokenGuardsTheApi"),
    ("the content policy stops pinning connections to this origin",
     "ledger/ui/server.py", "connect-src 'self'; ", "connect-src *; ",
     "tests/test_ui.py::TestTheContentPolicyForbidsLeaving"),
    ("responses become readable cross-origin",
     "ledger/ui/server.py",
     '        self.send_header("Referrer-Policy", "no-referrer")',
     '        self.send_header("Access-Control-Allow-Origin", "*")',
     "tests/test_ui.py::TestTheTokenGuardsTheApi::"
     "test_there_is_no_cors_header_on_any_response"),
    ("static serving stops containing the path",
     "ledger/ui/server.py",
     "        if STATIC not in target.parents or not target.is_file():",
     "        if not target.is_file():",
     "tests/test_ui.py::TestStaticServing::"
     "test_a_path_outside_the_static_directory_is_refused"),
    ("wipe accepts anything at all",
     "ledger/ui/server.py", 'if body.get("confirm") != "WIPE":', "if False:",
     "tests/test_ui.py::TestTheEndToEndPath::test_wipe_needs_the_exact_word"),
    ("an empty entry is stored instead of refused",
     "ledger/ui/server.py", "        if not text:", "        if False:",
     "tests/test_ui.py::TestTheEndToEndPath::test_an_empty_entry_is_refused"),
    ("the unlock response echoes the passphrase back",
     "ledger/ui/server.py",
     'return self._send_json({"unlocked": True, "created": create,',
     'return self._send_json({"unlocked": True, "created": create, "p": passphrase,',
     "tests/test_ui.py::TestTheServerHoldsNothingItShouldNot::"
     "test_the_passphrase_is_never_echoed_back"),
    ("the server starts caching entry text",
     "ledger/ui/server.py",
     "        journal.append(analysis.to_record())",
     "        journal.append(analysis.to_record())\n"
     "        self.state.last_text = analysis.text",
     "tests/test_ui.py::TestTheServerHoldsNothingItShouldNot::"
     "test_no_entry_text_is_cached_on_the_server_object"),
    ("a stylesheet is pulled from a CDN",
     "ledger/ui/static/index.html",
     '<link rel="stylesheet" href="/app.css">',
     '<link rel="stylesheet" href="https://cdn.example.com/reset.css">',
     "tests/test_ui.py::TestTheContentPolicyForbidsLeaving::"
     "test_no_shipped_asset_references_a_third_party"),
    ("the bind address becomes configurable",
     "ledger/ui/server.py",
     "def serve(*, store: Path | str = DEFAULT_STORE, region: str | None = None,\n"
     "          port: int = 0) -> LedgerUIServer:",
     "def serve(*, store: Path | str = DEFAULT_STORE, region: str | None = None,\n"
     "          port: int = 0, host: str = BIND_HOST) -> LedgerUIServer:",
     "tests/test_ui.py::TestTheListenerIsLocal::test_the_bind_address_is_not_configurable"),
]

# (label, file, old, new, the R9 verdict key that must flip to FAIL)
HARNESS_MUTATIONS: list[tuple[str, str, str, str, str]] = [
    ("the focus ring is removed from every control",
     "ledger/ui/static/app.css",
     ":focus-visible {\n  outline: 3px solid var(--focus-ring);\n"
     "  outline-offset: 2px;\n  box-shadow: 0 0 0 5px var(--focus-halo);\n"
     "  border-radius: var(--radius);\n}",
     ":focus-visible {\n  outline: none;\n}",
     "R9-2_visible_focus"),
    ("the reduced-motion block is deleted",
     "ledger/ui/static/app.css",
     "@media (prefers-reduced-motion: reduce) {\n  *, *::before, *::after {",
     "@media (prefers-reduced-motion: no-preference) {\n  *, *::before, *::after {",
     "R9-5_reduced_motion"),
    ("the status region stops being a live region",
     "ledger/ui/static/index.html",
     'id="status" role="status" aria-live="polite"', 'id="status"',
     "R9-6_announcements"),
    ("a form control loses its label",
     "ledger/ui/static/index.html",
     '<label for="entry-text">Today</label>', '<span>Today</span>',
     "R9-3_axe_zero_violations"),
    ("the muted text colour is lightened past the contrast floor",
     "ledger/ui/static/app.css", "--ink-soft: #4a5158;", "--ink-soft: #b3b8bd;",
     "R9-4_contrast_and_forced_colors"),
    ("clinical vocabulary appears in the interface copy",
     "ledger/ui/static/index.html",
     "<h2 id=\"h-history\" tabindex=\"-1\">History</h2>",
     "<h2 id=\"h-history\" tabindex=\"-1\">History</h2>\n"
     "    <p>Your symptom severity is improving.</p>",
     "R9-9_no_clinical_vocabulary_rendered"),
]


def guard_fails(selector: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", selector, "-q", "--no-header",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800,
    )
    return result.returncode != 0


def harness_verdicts() -> dict:
    """Run the accessibility harness and return its per-rule verdicts."""
    subprocess.run([sys.executable, "a11y/audit_a11y.py"], cwd=ROOT,
                   capture_output=True, text=True, timeout=3600)
    report = ROOT / "artifacts" / "a11y_report.json"
    return json.loads(report.read_text()).get("verdicts", {}) if report.exists() else {}


def apply_and_check(label, relative, old, new, check, results, kind):
    path = ROOT / relative
    original = path.read_text()
    # An ambiguous target is worse than a missing one — it can mutate a comment
    # and then report MISSED against a guard that was never given anything.
    occurrences = original.count(old)
    if occurrences != 1:
        results.append({"mutation": label, "kind": kind, "file": relative,
                        "caught": False,
                        "error": f"target occurs {occurrences} times; must be exactly 1"})
        print(f"INVALID {label}  ->  target occurs {occurrences} times")
        return
    try:
        path.write_text(original.replace(old, new, 1))
        caught, detail = check()
    finally:
        path.write_text(original)
    results.append({"mutation": label, "kind": kind, "file": relative,
                    "caught": caught, "detail": detail})
    print(("CAUGHT  " if caught else "MISSED  ") + f"{label}  ->  {detail}")


def main() -> int:
    fast = "--fast" in sys.argv
    results: list[dict] = []

    for label, relative, old, new, selector in UNIT_MUTATIONS:
        apply_and_check(label, relative, old, new,
                        lambda s=selector: (guard_fails(s), s), results, "unit")

    if not fast:
        for label, relative, old, new, key in HARNESS_MUTATIONS:
            def check(k=key):
                verdicts = harness_verdicts()
                return verdicts.get(k) == "FAIL", f"{k}={verdicts.get(k)}"
            apply_and_check(label, relative, old, new, check, results, "harness")

    caught = sum(1 for r in results if r["caught"])
    print(f"\n{caught}/{len(results)} mutations caught")
    out = ROOT / "audit" / "runs" / "inc9_mutations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "measured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "caught": caught, "total": len(results),
        "harness_mutations_run": not fast, "results": results}, indent=1) + "\n")
    return 0 if caught == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
