"""Every published shipped-artifact figure must re-derive from its artifact.

Round-1 audit finding F-03: six of twenty traced figures did not match the
artifact the document named as their evidence, and `845.98` — published three
times as the shipped build's WASM p95 — existed in no artifact at all. The
discipline `audit/README.md` claims was in force was not executable, so nothing
enforced it. This test makes it executable.

It runs `export/check_published_numbers.py`, which re-derives every registered
figure from `artifacts/verify_report.json` and `artifacts/wasm/bench_*.json`,
compares the *rendered string* rather than a tolerance, and additionally fails
on any `<n> ms` / `<n> MiB` token in the governed documents that is neither a
registered claim nor an explicitly-exempt one.

No artifact is built here: the check reads checked-in JSON, so it runs in a
clean clone with no ONNX build.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "export" / "check_published_numbers.py"


def _load():
    spec = importlib.util.spec_from_file_location("check_published_numbers", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_exists():
    assert SCRIPT.is_file(), "export/check_published_numbers.py is the enforcement"


def test_every_registered_claim_matches_its_artifact():
    mod = _load()
    failures = [r for r in mod.check_claims() if not r["ok"]]
    assert not failures, "\n".join(
        f"{r['id']}: {r['document']} publishes {r['published']!r}; "
        f"{r['artifact_source']} renders {r['artifact_rendered']!r}"
        f"{' (literal absent from document)' if not r['literal_present_in_document'] else ''}"
        for r in failures
    )


def test_no_unregistered_live_figure():
    mod = _load()
    orphans = mod.check_orphans()
    assert not orphans, "\n".join(
        f"{o['document']}:{o['line']} {o['token']!r} is neither a registered "
        f"claim nor listed in EXEMPT with a reason"
        for o in orphans
    )


def test_quoted_ceiling_values_are_the_enforced_ones():
    mod = _load()
    bad = [r for r in mod.check_ceilings() if not r["ok"]]
    assert not bad, str(bad)


def test_documents_quoting_superseded_figures_say_so():
    mod = _load()
    bad = [r for r in mod.check_superseded_markers() if not r["ok"]]
    assert not bad, "\n".join(
        f"{r['document']} quotes {r['superseded_tokens']} superseded figures "
        f"but never uses the word SUPERSEDED" for r in bad
    )


def test_check_is_not_vacuous():
    """A negative control: corrupt one artifact value and the check must fail.

    Without this, a check that silently registered zero claims would pass.
    """
    mod = _load()
    assert len(mod.CLAIMS) >= 20, "claim table is suspiciously small"
    cid, doc, literal, source, value, fmt = mod.CLAIMS[0]
    perturbed = (cid, doc, literal, source, value * 1.5, fmt)
    saved = mod.CLAIMS[0]
    mod.CLAIMS[0] = perturbed
    try:
        failures = [r for r in mod.check_claims() if not r["ok"]]
        assert failures, "perturbing an artifact value did not fail the check"
    finally:
        mod.CLAIMS[0] = saved


def test_main_exits_zero():
    mod = _load()
    assert mod.main([]) == 0, "published figures and artifacts disagree"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
