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

#: Set while the harness itself is probing the network posture (R9-7). Those
#: calls are deliberately aimed at a NON-loopback address — that is the whole
#: point of the probe — so they are recorded and flagged rather than counted as
#: violations. Anything the application does outside this window is a violation.
PROBING = False

#: Every address the process asked the kernel to listen on. Added in increment 9:
#: until the UI there was no listener, and "what did it bind" is now as much a
#: part of the network posture as "what did it dial".
BINDS: list[dict] = []

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
        "probe": PROBING,
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
    real_bind = socket.socket.bind
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

    def bind(self, address):
        host = host_of(address)
        BINDS.append({"address": repr(address), "loopback": _is_loopback(host),
                      "wildcard": host in {"", "0.0.0.0", "::"}})
        return real_bind(self, address)

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
    socket.socket.bind = bind
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


def interface_addresses() -> list[dict]:
    """Every IPv4 address on this host, read from the kernel, not from a name.

    `ioctl(SIOCGIFADDR)` on each interface. No resolver, no packet, nothing that
    would itself be egress — asking DNS for the machine's own address in order to
    prove the machine sends no DNS would be a poor joke.
    """
    import fcntl
    import struct

    SIOCGIFADDR = 0x8915
    out = []
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for _, name in socket.if_nameindex():
            try:
                packed = fcntl.ioctl(sock.fileno(), SIOCGIFADDR,
                                     struct.pack("256s", name.encode()[:15]))
                address = socket.inet_ntoa(packed[20:24])
            except OSError:
                continue                       # no IPv4 on this interface
            out.append({"interface": name, "address": address,
                        "loopback": _is_loopback(address)})
    finally:
        sock.close()
    return out


def probe_reachability(port: int) -> dict:
    """R9-7. The UI's port must be refused on every non-loopback address.

    A listener bound to 127.0.0.1 is unreachable from the network. That is a
    claim about the kernel's binding, and it is checked from outside the server
    rather than read off the source line that set the bind address.
    """
    global PROBING
    results = []
    addresses = interface_addresses()
    PROBING = True
    try:
        for entry in addresses:
            if entry["loopback"]:
                continue
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            try:
                code = sock.connect_ex((entry["address"], port))
            except OSError as exc:
                code = getattr(exc, "errno", -1)
            finally:
                sock.close()
            results.append({"address": entry["address"], "interface": entry["interface"],
                            "connect_ex": code, "accepted": code == 0})
    finally:
        PROBING = False
    reachable = [r for r in results if r["accepted"]]
    return {
        "port": port,
        "interfaces": addresses,
        "non_loopback_probes": results,
        "reachable_from_non_loopback": reachable,
        "verdict": "PASS" if not reachable else "FAIL",
        "note": ("No non-loopback IPv4 address exists on this host, so the probe "
                 "had nothing to try; the bind address is still recorded below.")
        if not results else "",
    }


def exercise_ui(store_path: Path) -> dict:
    """Drive the increment-9 interface over loopback, in this same process.

    The browser half of the same claim is measured separately, in
    `a11y/audit_a11y.py`, which records every request the page issues. This half
    is about the Python process: what the server binds, and what the client
    half of the same process dials.
    """
    import urllib.error
    import urllib.request

    from ledger.ui.server import serve_in_thread

    server = serve_in_thread(store=store_path, region="SG")
    base = server.url.rstrip("/")
    token = server.state.token
    # Read off the live socket now: after `server_close()` the descriptor is gone
    # and `getsockname()` raises EBADF.
    bound_host, bound_port = server.bound_host, server.server_port
    steps = []

    def call(path, body=None):
        request = urllib.request.Request(base + path,
                                         method="POST" if body is not None else "GET")
        request.add_header("X-Ledger-Token", token)
        if body is not None:
            request.add_header("Content-Type", "application/json")
            request.data = json.dumps(body).encode()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return response.status, json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"{}")

    try:
        steps.append(("GET /", call("/api/state")[0]))
        steps.append(("unlock", call("/api/unlock", {"passphrase": "an audit passphrase"})[0]))
        for text in EXERCISE_ENTRIES:
            steps.append(("entry", call("/api/entry", {"text": text})[0]))
        steps.append(("entries", call("/api/entries")[0]))
        steps.append(("report", call("/api/report")[0]))
        reachability = probe_reachability(bound_port)
        steps.append(("wipe", call("/api/wipe", {"confirm": "WIPE"})[0]))
    finally:
        server.shutdown_now()

    return {
        "bound_host": bound_host,
        "port": bound_port,
        "steps": [{"step": name, "status": status} for name, status in steps],
        "all_ok": all(status in (200, 201) for _, status in steps),
        "reachability": reachability,
        "store_removed": not store_path.exists(),
    }


def main() -> int:
    ui_mode = "--ui" in sys.argv
    instrument()
    ui_result: dict = {}
    with tempfile.TemporaryDirectory(prefix="ledger-egress-") as tmp:
        store = Path(tmp) / "journal.enc"
        try:
            exercised = exercise(store)
            error = None
        except Exception as exc:                       # measurement still gets written
            exercised, error = {}, f"{type(exc).__name__}: {exc}"
        if ui_mode and error is None:
            try:
                ui_result = exercise_ui(Path(tmp) / "ui-journal.enc")
            except Exception as exc:                   # measurement still gets written
                ui_result, error = {}, f"UI: {type(exc).__name__}: {exc}"

    # A probe call is one this harness aimed at a non-loopback address on purpose
    # (R9-7). It is reported, and it is not a violation; nothing else is exempt.
    violations = [call for call in CALLS if not call["loopback"] and not call["probe"]]
    wildcard_binds = [b for b in BINDS if b["wildcard"] or not b["loopback"]]
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
            "socket.gethostbyname", "socket.gethostbyname_ex", "socket.socket.bind",
        ],
        "offline_env_applied": {k: os.environ.get(k) for k in (
            "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")},
        "application_exercise": exercised,
        "ui_exercise": ui_result,
        "binds": BINDS,
        "non_loopback_or_wildcard_binds": wildcard_binds,
        "exercise_error": error,
        "socket_calls_total": len(CALLS),
        "socket_calls_loopback": len(CALLS) - len(violations),
        "non_loopback_calls": violations,
        "probe_calls": [c for c in CALLS if c["probe"]],
        "verdict": "PASS" if (
            not violations and error is None and not wildcard_binds
            and (not ui_mode or (ui_result.get("all_ok")
                                 and ui_result.get("reachability", {}).get("verdict") == "PASS"))
        ) else "FAIL",
    }
    out = ROOT / "artifacts" / ("egress_audit_ui.json" if ui_mode else "egress_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "non_loopback_calls"}, indent=1))
    if violations:
        print(f"\n{len(violations)} NON-LOOPBACK CALL(S):", file=sys.stderr)
        print(json.dumps(violations, indent=1), file=sys.stderr)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
