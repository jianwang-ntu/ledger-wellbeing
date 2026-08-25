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

**A per-unlock session token, which is never in the page.** The run token above
is a same-origin guard, not an authenticator: ``GET /`` is unauthenticated by
necessity (the browser has to be able to load the page), so *any* local process
can fetch the HTML and scrape the run token out of it. Round-1 audit finding
**F-04** demonstrated exactly that — a second local client that never supplied
the passphrase read the decrypted journal. The fix is that unlocking now mints a
second secret, ``session``, returned **only in the body of the successful
``/api/unlock`` response**, and every endpoint that can reach journal plaintext
(``/api/entries``, ``/api/report``, ``/api/entry``) requires it in
``X-Ledger-Session``. A client that did not supply the passphrase never sees that
value, because it was never written anywhere it could be read from. The store is
mode 0600 to exclude other local users; this is what stops the listener from
undoing that while unlocked.

**The session guards destruction and eviction too, not only reading.** Round-2
audit finding **AUDR2-F-001** showed that closing F-04 for confidentiality left
destruction open: ``/api/wipe`` sat outside the session-guarded set, so the same
scraped run token let an unrelated local process overwrite and unlink the
encrypted journal irrecoverably, without ever supplying the passphrase. Wipe
still does not require the journal to be *open* — someone who needs their
journal gone should not have to read it first — but it now requires proof of
ownership: either the session token, or the passphrase in the request body.
``/api/lock`` is likewise session-guarded (**AUDR2-F-002**), a failed unlock no
longer evicts the session it does not hold, failed passphrase attempts back off
exponentially, and the unauthenticated ``/api/state`` no longer discloses the
store path or whether a key is currently in memory.

**An idle key drop.** ``IDLE_LOCK_SECONDS`` (default 900) bounds how long the
derived key stays in memory after the last authenticated request. On expiry the
journal handle and the session token are both dropped and the passphrase is
required again. This is the second half of F-04: "the server retains the derived
key and never re-challenges".

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
import time
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

#: How long an unlocked journal stays unlocked without an authenticated request.
#: Configurable downward for tests; there is deliberately no way to disable it.
IDLE_LOCK_SECONDS = float(os.environ.get("LEDGER_IDLE_LOCK_SECONDS", 900))

#: How long a client that got the passphrase wrong must wait before the next
#: attempt. Doubles per consecutive failure and is capped, so an unattended local
#: guesser is bounded in rate while a person who mistyped waits a second or two.
UNLOCK_BACKOFF_SECONDS = float(os.environ.get("LEDGER_UNLOCK_BACKOFF_SECONDS", 1.0))
UNLOCK_BACKOFF_MAX_SECONDS = float(
    os.environ.get("LEDGER_UNLOCK_BACKOFF_MAX_SECONDS", 60.0))

#: Endpoints that can reach journal plaintext. These need the session token that
#: only a client which supplied the passphrase has ever seen — not just the run
#: token, which any local process can scrape out of the served page (F-04).
PLAINTEXT_PATHS = frozenset({"/api/entries", "/api/report", "/api/entry"})

#: Endpoints that act on another client's behalf without reading anything:
#: ``/api/lock`` ends the session and drops the derived key. AUDR2-F-002 (b):
#: outside this set, the scraped run token was enough to lock the legitimate
#: user out of their own journal at will.
CONTROL_PATHS = frozenset({"/api/lock"})

#: Everything the run token alone must not be enough for.
#:
#: ``/api/wipe`` is deliberately NOT in here, and that is not an oversight: it is
#: the one endpoint that accepts the passphrase *in place of* a session, so that
#: the owner who cannot or will not open their journal can still destroy it.
#: ``_wipe`` enforces the choice itself. See AUDR2-F-001.
SESSION_PATHS = PLAINTEXT_PATHS | CONTROL_PATHS


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
        #: Minted on a successful unlock and returned only to the client that
        #: supplied the passphrase. Never rendered into the page.
        self.session: str | None = None
        self.unlocked_at: float | None = None
        self.last_seen: float | None = None
        #: Consecutive wrong passphrases, and the monotonic time before which
        #: the next attempt is refused. AUDR2-F-002 (c).
        self.failed_attempts = 0
        self.blocked_until = 0.0
        self._engine = None
        self._lock = threading.Lock()

    def open_session(self) -> str:
        self.session = secrets.token_urlsafe(32)
        self.unlocked_at = time.monotonic()
        self.last_seen = self.unlocked_at
        return self.session

    def close_session(self) -> None:
        self.journal = None
        self.session = None
        self.unlocked_at = None
        self.last_seen = None

    def expire_if_idle(self) -> bool:
        """Drop the derived key if nothing authenticated has happened lately.

        Returns True if this call locked the journal.
        """
        if self.journal is None or self.last_seen is None:
            return False
        if time.monotonic() - self.last_seen < IDLE_LOCK_SECONDS:
            return False
        self.close_session()
        return True

    def session_matches(self, presented: str | None) -> bool:
        """Whether this client holds the current session — without touching the clock.

        Separate from `session_ok` because `/api/state` is polled by the page.
        Refreshing the idle timer from a poll would mean the journal never
        idle-locks while a tab is open, which is the opposite of what
        `IDLE_LOCK_SECONDS` is for.
        """
        if self.session is None:
            return False
        return secrets.compare_digest(presented or "", self.session)

    def session_ok(self, presented: str | None) -> bool:
        if not self.session_matches(presented):
            return False
        self.last_seen = time.monotonic()
        return True

    # -- passphrase attempt rate limiting (AUDR2-F-002 c) ------------------

    def throttle_remaining(self) -> float:
        """Seconds until the next passphrase attempt is allowed. 0.0 when open."""
        return max(0.0, self.blocked_until - time.monotonic())

    def record_failed_attempt(self) -> None:
        self.failed_attempts += 1
        delay = min(UNLOCK_BACKOFF_MAX_SECONDS,
                    UNLOCK_BACKOFF_SECONDS * (2 ** (self.failed_attempts - 1)))
        self.blocked_until = time.monotonic() + delay

    def clear_failed_attempts(self) -> None:
        self.failed_attempts = 0
        self.blocked_until = 0.0

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

    def _headers(self, status: int, content_type: str, length: int,
                 extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("Content-Security-Policy", CSP)
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        # Threaded through rather than sent by a caller of send_header, so that
        # every response still leaves by this one door and the policy above
        # cannot be skipped by adding a header somewhere else.
        for name, value in (extra or {}).items():
            self.send_header(name, str(value))
        self.end_headers()

    def _send_bytes(self, status: int, content_type: str, payload: bytes,
                    extra: dict | None = None) -> None:
        self._headers(status, content_type, len(payload), extra)
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _send_json(self, payload: dict, status: int = 200,
                   extra: dict | None = None) -> None:
        self._send_bytes(status, "application/json; charset=utf-8",
                         json.dumps(payload).encode("utf-8"), extra)

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
            self.state.expire_if_idle()
            if path in SESSION_PATHS and not self._session_ok():
                return self._locked()
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
        self.state.expire_if_idle()
        path = self.path.split("?", 1)[0]
        if path in SESSION_PATHS and not self._session_ok():
            return self._locked()
        try:
            body = self._read_json()
        except (ValueError, json.JSONDecodeError):
            return self._error(HTTPStatus.BAD_REQUEST, "malformed request")
        return self._api_post(path, body)

    def _token_ok(self) -> bool:
        return secrets.compare_digest(
            self.headers.get("X-Ledger-Token") or "", self.state.token)

    def _session_ok(self) -> bool:
        """The run token is not enough to reach plaintext. See F-04."""
        return self.state.session_ok(self.headers.get("X-Ledger-Session"))

    def _locked(self) -> None:
        # 401, not 403: the correct remedy is to supply the passphrase. The body
        # never says whether a journal exists or how many entries it holds.
        return self._error(HTTPStatus.UNAUTHORIZED,
                           "this journal is locked for this client; unlock it "
                           "with the passphrase")

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
                self.state.close_session()
                return self._send_json({"unlocked": False})
            if path == "/api/entry":
                return self._entry(body)
            if path == "/api/wipe":
                return self._wipe(body)
        except StoreError as exc:
            return self._error(HTTPStatus.CONFLICT, str(exc))
        return self._error(HTTPStatus.NOT_FOUND, "no such endpoint")

    def _state_payload(self) -> dict:
        # AUDR2-F-002 (the disclosure half). The absolute store path and "is a
        # key in memory right now" told an unauthenticated local process where
        # the journal lives and exactly when destroying or reading it would cost
        # the most. Both are now behind the session.
        #
        # `unlocked` is reported as "unlocked FOR YOU", which is also the truer
        # answer: a client without the session cannot read, write or lock this
        # journal, so from where it stands the journal is shut. `store_exists`
        # stays public because the page must choose between "create" and "open"
        # before anyone has unlocked anything, and it discloses only what a stat
        # of the documented default path already would.
        mine = self.state.session_matches(self.headers.get("X-Ledger-Session"))
        evidence = dimension_evidence()
        return {
            **({"store": str(self.state.store)} if mine else {}),
            "store_exists": self.state.store.exists(),
            "unlocked": mine and self.state.journal is not None,
            "idle_lock_seconds": IDLE_LOCK_SECONDS,
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
        remaining = self.state.throttle_remaining()
        if remaining > 0:
            # AUDR2-F-002 (c). Checked BEFORE the scrypt derivation, so a rate
            # limited attempt costs the machine nothing.
            return self._throttled(remaining)
        journal = Journal(self.state.store, passphrase)
        create = not self.state.store.exists()
        try:
            self.state.journal = journal.create() if create else journal.unlock()
        except StoreError as exc:
            # AUDR2-F-002 (a): do NOT close the session here. The assignment
            # above did not happen, so there is no half-open state to clean up,
            # and the session that exists belongs to a client which DID supply
            # the passphrase. Dropping it because someone else guessed wrong let
            # any local process holding the scraped run token evict the
            # legitimate user's key at will.
            self.state.record_failed_attempt()
            # The message from the store layer never contains plaintext; it is
            # forwarded rather than replaced so the user learns which failure.
            return self._error(HTTPStatus.UNAUTHORIZED, str(exc))
        self.state.clear_failed_attempts()
        # Minted here and returned here. This response body is the only place the
        # session token is ever written, which is what makes it unavailable to a
        # local client that did not supply the passphrase (F-04).
        session = self.state.open_session()
        return self._send_json({"unlocked": True, "created": create,
                                "entries": self.state.journal.count(),
                                "session": session,
                                "idle_lock_seconds": IDLE_LOCK_SECONDS})

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
        # AUDR2-F-001. Wipe still does not require the journal to be OPEN — the
        # CLI reasoning holds, and someone who needs their journal gone should
        # not have to read it first. What it now requires is proof that the
        # caller is the owner, because this endpoint destroys the file
        # irrecoverably (docs/limitations.md §7.3) and the run token alone is
        # scrapeable by any local process out of the served page.
        #
        # Two ways to prove it, and no third: hold the session minted by a
        # successful unlock, or supply the passphrase here. Refusals are
        # deliberately uniform — a caller who is neither learns nothing about
        # whether a journal exists at all.
        if not self._session_ok():
            remaining = self.state.throttle_remaining()
            if remaining > 0:
                return self._throttled(remaining)
            if not self._passphrase_ok(body.get("passphrase") or ""):
                self.state.record_failed_attempt()
                return self._error(
                    HTTPStatus.UNAUTHORIZED,
                    "destroying this journal needs its passphrase, or a client "
                    "that has already unlocked it; nothing was destroyed")
            self.state.clear_failed_attempts()
        result = Journal(self.state.store, "unused").wipe()
        self.state.close_session()
        return self._send_json(result)

    def _passphrase_ok(self, passphrase: str) -> bool:
        """Verify a passphrase against the store without unlocking the session.

        Deriving the key is what proves the caller is the owner. The handle is
        discarded immediately: this must not become a second way to leave a key
        in memory. Costs one scrypt derivation per call, which is also why the
        caller of this method rate-limits it.
        """
        if not passphrase or not self.state.store.exists():
            return False
        try:
            Journal(self.state.store, passphrase).unlock()
        except StoreError:
            return False
        return True

    def _throttled(self, remaining: float) -> None:
        # AUDR2-F-002 (c). 429 rather than 401 so the caller can tell "wrong"
        # from "too fast", and a Retry-After so a legitimate client that
        # mistyped can simply wait rather than guess.
        return self._send_json(
            {"error": f"too many failed attempts; try again in {remaining:.1f}s",
             "retry_after": round(remaining, 1)},
            HTTPStatus.TOO_MANY_REQUESTS,
            {"Retry-After": max(1, int(remaining + 0.5))})


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
