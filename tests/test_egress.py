"""Increment 8 guards on the zero-egress claim — R8-3.

`plan.md`'s first differentiator is "zero egress, provable". Two different
things have to hold for that sentence to be worth anything:

1. The **measurement passed** — read from `artifacts/egress_audit.json`, which
   `export/egress_audit.py` produces. A guard that recomputed it would be
   grading its own homework.
2. The **measurement would have caught a violation**. A pass from an instrument
   that cannot fail is worth nothing, so this file makes a deliberate outbound
   call under the same instrumentation and asserts it is recorded.

Point 2 is why this file exists at all. Two guards in this repository were once
green only because the thing they checked was empty (DEFECT-INC7-001), and that
is not going to happen to the claim the whole project rests on.
"""

from __future__ import annotations

import importlib
import json
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "export"))

REPORT = ROOT / "artifacts" / "egress_audit.json"


def _report():
    if not REPORT.exists():
        pytest.skip("no egress_audit.json - run export/egress_audit.py first")
    return json.loads(REPORT.read_text())


class TestTheMeasurementPassed:
    def test_verdict_is_pass(self):
        assert _report()["verdict"] == "PASS"

    def test_no_non_loopback_call_was_recorded(self):
        assert _report()["non_loopback_calls"] == []

    def test_the_application_was_actually_exercised(self):
        """A pass over an application that never ran is not a pass."""
        exercised = _report()["application_exercise"]
        assert exercised, "the exercise did not complete"
        assert exercised["entries_added"] >= 5
        assert exercised["entries_scored"] >= 1, "nothing was scored, so the model never ran"
        assert exercised["entries_routed_to_help"] >= 1, "the crisis path was never taken"
        assert exercised["entries_read_back"] == exercised["entries_added"]
        assert exercised["report_chars"] > 0
        assert exercised["wiped"] and exercised["store_removed"]

    def test_the_exercise_raised_nothing(self):
        assert _report()["exercise_error"] is None

    def test_the_instrumentation_covered_dns_as_well_as_connect(self):
        instrumented = set(_report()["instrumented"])
        assert {"socket.getaddrinfo", "socket.socket.connect"} <= instrumented

    def test_the_report_states_what_it_does_not_cover(self):
        """A process-level measurement described as a packet capture would be a lie."""
        report = _report()
        assert "process-level" in report["scope"]
        assert any("packet-level" in item for item in report["not_in_scope"])
        assert any("subprocess" in item for item in report["not_in_scope"])


class TestTheInstrumentCanFail:
    """The measurement's own positive control."""

    def test_a_deliberate_outbound_call_is_recorded_as_a_violation(self, monkeypatch):
        egress_audit = importlib.import_module("egress_audit")
        monkeypatch.setattr(egress_audit, "CALLS", [])
        egress_audit.instrument()
        try:
            with pytest.raises(OSError):
                # RFC 5737 TEST-NET-1: routable-looking, guaranteed not to answer.
                sock = socket.socket()
                sock.settimeout(0.05)
                sock.connect(("192.0.2.1", 80))
        finally:
            sock.close()
            importlib.reload(socket)

        violations = [c for c in egress_audit.CALLS if not c["loopback"]]
        assert violations, "the instrument did not record a real outbound connect"
        assert "192.0.2.1" in violations[0]["target"]

    def test_a_loopback_call_is_recorded_but_is_not_a_violation(self, monkeypatch):
        egress_audit = importlib.import_module("egress_audit")
        monkeypatch.setattr(egress_audit, "CALLS", [])
        egress_audit.instrument()
        try:
            sock = socket.socket()
            sock.settimeout(0.05)
            try:
                sock.connect(("127.0.0.1", 9))     # discard port, almost certainly closed
            except OSError:
                pass
        finally:
            sock.close()
            importlib.reload(socket)

        assert egress_audit.CALLS, "the instrument recorded nothing at all"
        assert all(call["loopback"] for call in egress_audit.CALLS)

    def test_loopback_classification_is_not_fooled_by_a_hostname(self):
        egress_audit = importlib.import_module("egress_audit")
        assert egress_audit._is_loopback("127.0.0.1")
        assert egress_audit._is_loopback("::1")
        assert egress_audit._is_loopback("localhost")
        assert not egress_audit._is_loopback("huggingface.co")
        assert not egress_audit._is_loopback("127.0.0.1.evil.com")
        assert not egress_audit._is_loopback("192.0.2.1")
        assert not egress_audit._is_loopback("8.8.8.8")


class TestOfflineIsEnforcedInCode:
    def test_the_offline_switches_are_set_on_import(self):
        from ledger.app import offline
        import os
        for key, value in offline.OFFLINE_ENV.items():
            assert os.environ.get(key) == value, key

    def test_the_engine_imports_offline_before_transformers(self):
        source = (ROOT / "ledger" / "app" / "engine.py").read_text()
        assert source.index("from ledger.app import offline") < source.index("transformers")

    def test_the_tokenizer_is_loaded_local_files_only(self):
        source = (ROOT / "ledger" / "app" / "engine.py").read_text()
        assert "local_files_only=True" in source
