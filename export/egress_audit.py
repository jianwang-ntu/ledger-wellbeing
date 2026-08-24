"""Measure R8-3: exercise the whole application and count what left the host.

`plan.md`'s first differentiator is "zero egress, provable", and its evidence
plan asks for "a re-runnable script, not a screenshot". This is that script.

Method
------
Every outbound primitive in the standard library's socket layer is wrapped
*before* the application is imported, and every call is recorded with its target
and a stack trace:

    socket.socket.connect / connect_ex / sendto / sendmsg
    socket.create_connection
    socket.getaddrinfo / gethostbyname / gethostbyname_ex

Then the application is driven end to end: create an encrypted journal, add
entries including ones that route to crisis resources, score and attribute the
rest, read them all back, render the report, and wipe.

A call whose destination is not a loopback address is a **failure**, and so is
any DNS resolution of a non-local name — because a resolution is an outbound UDP
packet carrying the name of what was about to be contacted, whether or not the
connection then happens.

What this measures, and what it does not
----------------------------------------
This is a **process-level** measurement. It observes what this Python process
asked the kernel to do. It does not observe a subprocess, a native library that
opens a socket without going through the `socket` module, or a kernel-level
send. `docs/limitations.md` says so plainly. A packet-level capture is strictly
stronger and needs privileges this build environment does not have; the two are
complementary and the weaker one is not described as the stronger one.

What it *is* sufficient for: `onnxruntime` reads a file, and the tokenizer is
loaded from a local directory. The realistic failure mode is a Hugging Face
library issuing a revision check on first use — which goes through
`socket.getaddrinfo` and would be caught here.
"""

from __future__ import annotations

import ipaddress
import json
import os
import socket
import sys
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CALLS: list[dict] = []

#: Names that resolve without leaving the host. Anything else is a violation.
LOCAL_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "", None}


def _is_loopback(host) -> bool:
    if host in LOCAL_NAMES:
        return True
    try:
        return ipaddress.ip_address(str(host)).is_loopback
    except ValueError:
        return False


def _record(kind: str, target, loopback: bool) -> None:
    CALLS.append({
        "kind": kind,
        "target": repr(target),
        "loopback": loopback,
        "stack": [f"{f.filename}:{f.lineno} in {f.name}"
                  for f in traceback.extract_stack()[:-2]][-6:],
    })


def instrument() -> None:
    """Wrap the socket layer. Called before the application is imported."""
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_sendto = socket.socket.sendto
    real_sendmsg = socket.socket.sendmsg
    real_create = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo
    real_gethostbyname = socket.gethostbyname
    real_gethostbyname_ex = socket.gethostbyname_ex

    def host_of(address):
        if isinstance(address, tuple) and address:
            return address[0]
        return address

    def connect(self, address):
        _record("connect", address, _is_loopback(host_of(address)))
        return real_connect(self, address)

    def connect_ex(self, address):
        _record("connect_ex", address, _is_loopback(host_of(address)))
        return real_connect_ex(self, address)

    def sendto(self, *args, **kwargs):
        address = args[-1] if args else None
        _record("sendto", address, _is_loopback(host_of(address)))
        return real_sendto(self, *args, **kwargs)

    def sendmsg(self, *args, **kwargs):
        address = args[3] if len(args) > 3 else None
        _record("sendmsg", address, address is None or _is_loopback(host_of(address)))
        return real_sendmsg(self, *args, **kwargs)

    def create_connection(address, *args, **kwargs):
        _record("create_connection", address, _is_loopback(host_of(address)))
        return real_create(address, *args, **kwargs)

    def getaddrinfo(host, *args, **kwargs):
        _record("getaddrinfo", host, _is_loopback(host))
        return real_getaddrinfo(host, *args, **kwargs)

    def gethostbyname(host):
        _record("gethostbyname", host, _is_loopback(host))
        return real_gethostbyname(host)

    def gethostbyname_ex(host):
        _record("gethostbyname_ex", host, _is_loopback(host))
        return real_gethostbyname_ex(host)

    socket.socket.connect = connect
    socket.socket.connect_ex = connect_ex
    socket.socket.sendto = sendto
    socket.socket.sendmsg = sendmsg
    socket.create_connection = create_connection
    socket.getaddrinfo = getaddrinfo
    socket.gethostbyname = gethostbyname
    socket.gethostbyname_ex = gethostbyname_ex


#: Entries chosen to exercise every branch: two ordinary, one elevated, one
#: acute (which must not be scored at all), one with obfuscated crisis text.
EXERCISE_ENTRIES = [
    "I slept badly again and dragged through the whole day. The meeting went fine "
    "but I could not settle afterwards and kept going over it.",
    "Slept right through for once. I went for a walk at lunch and actually enjoyed "
    "the afternoon.",
    "Nothing matters anymore and I cannot see the point in going on with any of it.",
    "I want to k1ll myself.",
    "Quiet day. I read for an hour and went to bed early without checking my phone.",
]


def exercise(store_path: Path) -> dict:
    """Drive the application end to end. Every step a user would take."""
    from ledger.app.engine import LedgerEngine
    from ledger.app.report import render
    from ledger.store.journal import Journal

    passphrase = "an audit passphrase that is not a user's"
    journal = Journal(store_path, passphrase).create()
    engine = LedgerEngine(region="SG")

    scored, routed = 0, 0
    for day, text in enumerate(EXERCISE_ENTRIES):
        analysis = engine.analyse(
            text, region="SG", written_at=f"2026-08-{10 + day:02d}T09:00:00Z"
        )
        scored += int(analysis.scored)
        routed += int(analysis.routed["triggered"])
        journal.append(analysis.to_record())

    reopened = Journal(store_path, passphrase).unlock()
    entries = [{"entry_id": e.entry_id, "written_at": e.written_at,
                "text": e.text, "analysis": e.analysis} for e in reopened.entries()]
    report = render(entries, region="SG")
    wiped = reopened.wipe()

    return {
        "entries_added": len(EXERCISE_ENTRIES),
        "entries_scored": scored,
        "entries_routed_to_help": routed,
        "entries_read_back": len(entries),
        "report_chars": len(report),
        "wiped": wiped["wiped"],
        "store_removed": not store_path.exists(),
    }


def main() -> int:
    instrument()
    with tempfile.TemporaryDirectory(prefix="ledger-egress-") as tmp:
        store = Path(tmp) / "journal.enc"
        try:
            exercised = exercise(store)
            error = None
        except Exception as exc:                       # measurement still gets written
            exercised, error = {}, f"{type(exc).__name__}: {exc}"

    violations = [call for call in CALLS if not call["loopback"]]
    report = {
        "measured_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rule": "R8-3, export/INCREMENT_8_PREREGISTRATION.md",
        "scope": "process-level: socket calls made by this Python process",
        "not_in_scope": [
            "subprocesses",
            "native libraries opening sockets without the socket module",
            "kernel-level sends",
            "a packet-level capture, which is strictly stronger and needs privileges "
            "this environment does not have",
        ],
        "instrumented": [
            "socket.socket.connect", "socket.socket.connect_ex", "socket.socket.sendto",
            "socket.socket.sendmsg", "socket.create_connection", "socket.getaddrinfo",
            "socket.gethostbyname", "socket.gethostbyname_ex",
        ],
        "offline_env_applied": {k: os.environ.get(k) for k in (
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")},
        "application_exercise": exercised,
        "exercise_error": error,
        "socket_calls_total": len(CALLS),
        "socket_calls_loopback": len(CALLS) - len(violations),
        "non_loopback_calls": violations,
        "verdict": "PASS" if (not violations and error is None) else "FAIL",
    }
    out = ROOT / "artifacts" / "egress_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "non_loopback_calls"}, indent=1))
    if violations:
        print(f"\n{len(violations)} NON-LOOPBACK CALL(S):", file=sys.stderr)
        print(json.dumps(violations, indent=1), file=sys.stderr)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
