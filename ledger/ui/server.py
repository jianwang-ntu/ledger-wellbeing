"""The loopback HTTP server behind Ledger's interface.

Everything the browser talks to is here. The rules that make a local web UI an
acceptable way to ship a private instrument are enforced in this file, not in the
page, because the page is the untrusted half:

**Loopback only.** ``BIND_HOST`` is ``127.0.0.1`` and there is no option to
change it. R9-7 measures that the same port is refused on the host's own
non-loopback address.

**No name lookup at bind time.** ``http.server.HTTPServer.server_bind`` calls
``socket.getfqdn()``, which asks the resolver. A privacy tool that performs a DNS
lookup in order to start is a bad joke, so ``server_bind`` is overridden.

**Host-header pinning.** A ``Host`` that is not ``127.0.0.1:<port>`` or
``localhost:<port>`` is rejected with 421. That is what closes DNS rebinding: a
page on the open internet can point a name at 127.0.0.1, but it cannot make the
browser send our ``Host``.

**A per-run token, served inside the page.** Every ``/api`` call must carry
``X-Ledger-Token``. The token is minted at startup and embedded in the HTML, and
no cross-origin document can read the HTML back — there is no CORS header on any
response, deliberately.

**A content policy that forbids leaving.** Every response carries
``connect-src 'self'`` and ``default-src 'self'``, so the browser itself refuses
an external request even if the page asked for one. R9-8(b) then measures that
the page never asks.

The passphrase is posted over loopback in a request body. That is stated in
``docs/limitations.md`` rather than implied: it does not leave the host, and it
is never written to disk, to a log line, or to the URL.
"""

from __future__ import annotations

import json
import os
import secrets
import socket
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ledger.app import offline  # noqa: F401  - side effect: pin the ML libraries offline
from ledger.app.evidence import EVALUATION_BASIS, USABLE_HELD_OUT_AUC, dimension_evidence
from ledger.app.report import CONTRACT as REPORT_CONTRACT, render
from ledger.model.dimensions import DIMENSION_LABELS, DIMENSIONS
from ledger.store.journal import Journal, StoreError

#: Not configurable. See the module docstring.
BIND_HOST = "127.0.0.1"

STATIC = Path(__file__).resolve().parent / "static"

DEFAULT_STORE = Path(os.environ.get("LEDGER_HOME", Path.home() / ".ledger")) / "journal.enc"

#: Sent on every response. `connect-src 'self'` is the one that matters for R9-8:
#: with it, the browser refuses an outbound request the page did not have to be
#: trusted not to make.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; "
       "img-src 'self' data:; font-src 'self'; connect-src 'self'; "
       "base-uri 'none'; form-action 'none'; frame-ancestors 'none'; "
       "object-src 'none'")

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
}

MAX_BODY_BYTES = 256 * 1024


class UIState:
    """What the server holds between requests.

    An unlocked `Journal` and nothing else. No entry text is cached here: every
    read goes back through the encrypted store, so 'what the UI is showing' and
    'what is on disk' cannot drift apart.
    """

    def __init__(self, store: Path, region: str | None):
        self.store = Path(store)
        self.region = region
        self.token = secrets.token_urlsafe(32)
        self.journal: Journal | None = None
        self._engine = None
        self._lock = threading.Lock()

    def engine(self):
        # Deferred and cached: ~200 MB of model, loaded on first score, never on
        # the crisis path (R8-4) and never merely because a page was opened.
        from ledger.app.engine import LedgerEngine
        with self._lock:
            if self._engine is None:
                self._engine = LedgerEngine(region=self.region)
            return self._engine

    def require_journal(self) -> Journal:
        if self.journal is None:
            raise StoreError("the journal is locked")
        return self.journal


class LedgerHandler(BaseHTTPRequestHandler):
    server_version = "Ledger"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    # -- plumbing ----------------------------------------------------------

    @property
    def state(self) -> UIState:
        return self.server.state          # type: ignore[attr-defined]

    def log_message(self, fmt, *args):
        """Silence by default.

        The default handler writes every request line to stderr. Request lines
        are not sensitive here, but a terminal transcript of a private journal
        session is exactly the artifact this product exists to avoid creating.
        """
        if os.environ.get("LEDGER_UI_LOG"):
            super().log_message(fmt, *args)

    def _headers(self, status: int, content_type: str, length: int) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def _send_bytes(self, status: int, content_type: str, payload: bytes) -> None:
        self._headers(status, content_type, len(payload))
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        self._send_bytes(status, "application/json; charset=utf-8",
                         json.dumps(payload).encode("utf-8"))

    def _error(self, status: int, message: str) -> None:
        self._send_json({"error": message}, status)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").strip()
        port = self.server.server_address[1]
        return host in {f"{BIND_HOST}:{port}", f"localhost:{port}"}

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            raise ValueError("request body too large")
        return json.loads(self.rfile.read(length) or b"{}")

    # -- routing -----------------------------------------------------------

    def do_GET(self) -> None:                      # noqa: N802 - stdlib naming
        if not self._host_ok():
            return self._error(HTTPStatus.MISDIRECTED_REQUEST, "unexpected Host header")
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/"):
            if not self._token_ok():
                return self._error(HTTPStatus.FORBIDDEN, "missing or wrong token")
            return self._api_get(path)
        return self._static(path)

    do_HEAD = do_GET

    def do_POST(self) -> None:                     # noqa: N802 - stdlib naming
        if not self._host_ok():
            return self._error(HTTPStatus.MISDIRECTED_REQUEST, "unexpected Host header")
        if not self.path.startswith("/api/"):
            return self._error(HTTPStatus.NOT_FOUND, "no such endpoint")
        if not self._token_ok():
            return self._error(HTTPStatus.FORBIDDEN, "missing or wrong token")
        try:
            body = self._read_json()
        except (ValueError, json.JSONDecodeError):
            return self._error(HTTPStatus.BAD_REQUEST, "malformed request")
        return self._api_post(self.path.split("?", 1)[0], body)

    def _token_ok(self) -> bool:
        return secrets.compare_digest(
            self.headers.get("X-Ledger-Token") or "", self.state.token)

    # -- static ------------------------------------------------------------

    def _static(self, path: str) -> None:
        name = "index.html" if path == "/" else path.lstrip("/")
        target = (STATIC / name).resolve()
        if STATIC not in target.parents or not target.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "no such file")
        body = target.read_bytes()
        if target.name == "index.html":
            # The token is injected rather than fetched, so the page is usable
            # from its first paint and there is no unauthenticated bootstrap call.
            body = body.replace(b"__LEDGER_TOKEN__", self.state.token.encode())
        self._send_bytes(HTTPStatus.OK,
                         CONTENT_TYPES.get(target.suffix, "application/octet-stream"), body)

    # -- api ---------------------------------------------------------------

    def _api_get(self, path: str) -> None:
        try:
            if path == "/api/state":
                return self._send_json(self._state_payload())
            if path == "/api/entries":
                journal = self.state.require_journal()
                return self._send_json({"entries": [
                    {"entry_id": e.entry_id, "written_at": e.written_at,
                     "text": e.text, "analysis": e.analysis}
                    for e in journal.entries()]})
            if path == "/api/report":
                journal = self.state.require_journal()
                entries = [{"entry_id": e.entry_id, "written_at": e.written_at,
                            "text": e.text, "analysis": e.analysis}
                           for e in journal.entries()]
                return self._send_json({
                    "report": render(entries, region=self.state.region),
                    "contract": REPORT_CONTRACT,
                    "entries": len(entries),
                })
        except StoreError as exc:
            return self._error(HTTPStatus.CONFLICT, str(exc))
        return self._error(HTTPStatus.NOT_FOUND, "no such endpoint")

    def _api_post(self, path: str, body: dict) -> None:
        try:
            if path == "/api/unlock":
                return self._unlock(body)
            if path == "/api/lock":
                self.state.journal = None
                return self._send_json({"unlocked": False})
            if path == "/api/entry":
                return self._entry(body)
            if path == "/api/wipe":
                return self._wipe(body)
        except StoreError as exc:
            return self._error(HTTPStatus.CONFLICT, str(exc))
        return self._error(HTTPStatus.NOT_FOUND, "no such endpoint")

    def _state_payload(self) -> dict:
        evidence = dimension_evidence()
        return {
            "store": str(self.state.store),
            "store_exists": self.state.store.exists(),
            "unlocked": self.state.journal is not None,
            "region": self.state.region,
            "contract": REPORT_CONTRACT,
            "evaluation_basis": EVALUATION_BASIS,
            "threshold": USABLE_HELD_OUT_AUC,
            "dimensions": [
                {"dimension": dim, "label": DIMENSION_LABELS[dim],
                 "established": evidence[dim]["established"],
                 "held_out_auc": evidence[dim]["held_out_auc"],
                 "note": evidence[dim]["note"]}
                for dim in DIMENSIONS
            ],
        }

    def _unlock(self, body: dict) -> None:
        passphrase = body.get("passphrase") or ""
        if len(passphrase) < 1:
            return self._error(HTTPStatus.BAD_REQUEST, "a passphrase is required")
        journal = Journal(self.state.store, passphrase)
        create = not self.state.store.exists()
        try:
            self.state.journal = journal.create() if create else journal.unlock()
        except StoreError as exc:
            self.state.journal = None
            # The message from the store layer never contains plaintext; it is
            # forwarded rather than replaced so the user learns which failure.
            return self._error(HTTPStatus.UNAUTHORIZED, str(exc))
        return self._send_json({"unlocked": True, "created": create,
                                "entries": self.state.journal.count()})

    def _entry(self, body: dict) -> None:
        text = (body.get("text") or "").strip()
        if not text:
            return self._error(HTTPStatus.BAD_REQUEST, "there is nothing written yet")
        journal = self.state.require_journal()
        analysis = self.state.engine().analyse(
            text, region=self.state.region,
            granularity=body.get("granularity") or "sentence")
        journal.append(analysis.to_record())
        payload = {"entry_id": analysis.entry_id, "written_at": analysis.written_at,
                   "text": analysis.text, "analysis": {
                       "routed": analysis.routed, "scored": analysis.scored,
                       "reason_not_scored": analysis.reason_not_scored,
                       "granularity": analysis.granularity,
                       "dimensions": analysis.dimensions, "model": analysis.model,
                       "contract": analysis.contract}}
        return self._send_json(payload)

    def _wipe(self, body: dict) -> None:
        if body.get("confirm") != "WIPE":
            return self._error(HTTPStatus.BAD_REQUEST,
                               "type WIPE to confirm; nothing was destroyed")
        # As in the CLI: wipe does not require the passphrase. Someone who needs
        # their journal gone should not have to open it first.
        result = Journal(self.state.store, "unused").wipe()
        self.state.journal = None
        return self._send_json(result)


class LedgerUIServer(ThreadingHTTPServer):
    """A `ThreadingHTTPServer` that binds loopback and resolves nothing."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, state: UIState, port: int = 0):
        self.state = state
        super().__init__((BIND_HOST, port), LedgerHandler)

    def server_bind(self) -> None:
        """Bind without `socket.getfqdn()`.

        `HTTPServer.server_bind` sets `server_name` from `getfqdn(host)`, which
        goes to the resolver. Nothing here needs a name, so nothing here asks for
        one — and R9-8(a) would otherwise record a lookup at startup.
        """
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        socket.socket.bind(self.socket, self.server_address)
        self.server_address = self.socket.getsockname()
        self.server_name = BIND_HOST
        self.server_port = self.server_address[1]

    @property
    def url(self) -> str:
        return f"http://{BIND_HOST}:{self.server_port}/"

    @property
    def bound_host(self) -> str:
        return self.socket.getsockname()[0]

    def shutdown_now(self) -> None:
        self.shutdown()
        self.server_close()


def serve(*, store: Path | str = DEFAULT_STORE, region: str | None = None,
          port: int = 0) -> LedgerUIServer:
    """Create and bind the server. The caller decides how to run it."""
    return LedgerUIServer(UIState(Path(store), region), port=port)


def serve_in_thread(**kwargs) -> LedgerUIServer:
    """Bind, then run in a daemon thread. Used by the tests and the a11y harness."""
    server = serve(**kwargs)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
