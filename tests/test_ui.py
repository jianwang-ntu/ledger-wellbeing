"""Guards for the increment-9 interface.

These are the properties that make a local web UI an acceptable shape for a
private instrument. Each one corresponds to a line in `ledger/ui/server.py` that
would be easy to relax later without noticing:

* the bind address,
* the `Host` pin that closes DNS rebinding,
* the per-run token,
* the content policy that forbids the page reaching anything else,
* and the absence of any third-party URL in the shipped assets.

The browser-side rules — R9-1 through R9-8(b) — are measured in
`a11y/audit_a11y.py` against a real Chromium, because they are properties of a
rendered document and no unit test can stand in for one.
"""

from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from pathlib import Path

import pytest

from ledger.app.engine import BUILD_FILES, selected_build
from ledger.ui import server as ui_server
from ledger.ui.server import CSP, STATIC, serve_in_thread

#: Round-1 audit F-02. The ONNX build is a 900 MiB output of
#: `bash export/run_all.sh` and is not in the repository, so the two tests that
#: score an entry end to end SKIP on a fresh clone rather than fail - and skip
#: rather than pass, so a green run never means less than it says.
needs_model = pytest.mark.skipif(
    not BUILD_FILES[selected_build()].exists(),
    reason="ONNX build absent - run `bash export/run_all.sh` (see README, Install)")

PASSPHRASE = "a test passphrase for the interface"

#: Any absolute URL that is not this machine.
THIRD_PARTY_URL = re.compile(rb"https?://(?!127\.0\.0\.1|localhost)", re.I)

#: The one exemption, by exact text. See the test that pins it.
SVG_NAMESPACE = b"http://www.w3.org/2000/svg"

ORDINARY = ("I slept badly again and dragged through the whole day. The meeting "
            "went fine but I could not settle afterwards.")
CRISIS = "I want to k1ll myself."


@pytest.fixture(scope="module")
def live(tmp_path_factory):
    store = tmp_path_factory.mktemp("ledger-ui") / "journal.enc"
    server = serve_in_thread(store=store, region="SG")
    try:
        yield server, store
    finally:
        server.shutdown_now()


def call(server, path, body=None, *, token=None, headers=None, method=None,
         session=None):
    """Speak to the server the way the shipped page does.

    Round-1 audit F-04 added a second secret: the run token is a same-origin
    guard, and the per-unlock session token is the thing that actually stands
    between a local process and journal plaintext. The real client sends both
    once it has unlocked, so this helper does too. Pass ``session=False`` to
    behave like a client that never supplied the passphrase — that is the shape
    the F-04 tests need.
    """
    url = server.url.rstrip("/") + path
    request = urllib.request.Request(url, method=method or ("POST" if body is not None else "GET"))
    request.add_header("X-Ledger-Token",
                       server.state.token if token is None else token)
    if session is not False:
        held = server.state.session if session is None else session
        if held:
            request.add_header("X-Ledger-Session", held)
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    if body is not None:
        request.add_header("Content-Type", "application/json")
        request.data = json.dumps(body).encode()
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


class TestTheListenerIsLocal:
    def test_it_binds_loopback_and_nothing_else(self, live):
        server, _ = live
        assert server.bound_host == "127.0.0.1"
        assert ui_server.BIND_HOST == "127.0.0.1"

    def test_the_bind_address_is_not_configurable(self):
        """A `host=` parameter is exactly how this would stop being local."""
        import inspect
        signature = inspect.signature(ui_server.serve)
        assert "host" not in signature.parameters
        assert "host" not in inspect.signature(ui_server.LedgerUIServer.__init__).parameters

    def test_binding_does_not_resolve_a_name(self, monkeypatch, tmp_path):
        """`HTTPServer.server_bind` calls `getfqdn`. This one must not."""
        called = []
        monkeypatch.setattr(socket, "getfqdn", lambda *a: called.append(a) or "x")
        monkeypatch.setattr(socket, "gethostbyaddr", lambda *a: called.append(a) or ("x", [], []))
        server = ui_server.serve(store=tmp_path / "j.enc")
        try:
            assert called == [], "the server resolved a name in order to start"
        finally:
            server.server_close()

    def test_a_foreign_host_header_is_refused(self, live):
        """The DNS-rebinding wall. A browser cannot forge this header."""
        server, _ = live
        status, _, _ = call(server, "/api/state", headers={"Host": "ledger.example.com"})
        assert status == HTTPStatus.MISDIRECTED_REQUEST


class TestTheTokenGuardsTheApi:
    def test_no_token_is_refused(self, live):
        server, _ = live
        status, _, _ = call(server, "/api/state", token="")
        assert status == HTTPStatus.FORBIDDEN

    def test_a_wrong_token_is_refused(self, live):
        server, _ = live
        status, _, _ = call(server, "/api/state", token="not-the-token")
        assert status == HTTPStatus.FORBIDDEN

    def test_a_wrong_token_cannot_wipe(self, live):
        server, store = live
        status, _, _ = call(server, "/api/wipe", {"confirm": "WIPE"}, token="nope")
        assert status == HTTPStatus.FORBIDDEN

    def test_the_page_carries_the_token_and_no_placeholder(self, live):
        server, _ = live
        status, body, _ = call(server, "/", token="")     # static needs no token
        assert status == 200
        assert b"__LEDGER_TOKEN__" not in body
        assert server.state.token.encode() in body

    def test_there_is_no_cors_header_on_any_response(self, live):
        server, _ = live
        for path in ("/", "/app.js", "/api/state"):
            _, _, headers = call(server, path)
            assert not any(key.lower().startswith("access-control-")
                           for key in headers), f"{path} would be readable cross-origin"


class TestTheContentPolicyForbidsLeaving:
    @pytest.mark.parametrize("path", ["/", "/app.css", "/app.js", "/icon.svg", "/api/state"])
    def test_every_response_carries_the_policy(self, live, path):
        server, _ = live
        _, _, headers = call(server, path)
        assert headers.get("Content-Security-Policy") == CSP

    def test_the_policy_pins_connections_to_this_origin(self):
        assert "connect-src 'self'" in CSP
        assert "default-src 'self'" in CSP
        assert "frame-ancestors 'none'" in CSP

    def test_no_shipped_asset_references_a_third_party(self):
        """The cheapest possible check for the thing R9-8(b) measures at runtime.

        A CDN link added later would be caught here, in the suite, before anyone
        had to open a browser to find out.
        """
        offenders = []
        for asset in sorted(STATIC.iterdir()):
            body = asset.read_bytes().replace(SVG_NAMESPACE, b"")
            hits = THIRD_PARTY_URL.findall(body)
            if hits:
                offenders.append((asset.name, len(hits)))
        assert offenders == [], f"third-party URLs in shipped assets: {offenders}"

    def test_the_only_exempt_url_is_the_svg_namespace(self):
        """The exemption is one exact string, so it cannot be widened by accident.

        `xmlns="http://www.w3.org/2000/svg"` is an identifier the SVG spec
        requires; no browser dereferences it. Exempting the literal — rather than
        loosening the pattern to "w3.org" or "anything in an svg" — keeps every
        other absolute URL in the shipped assets a failure.
        """
        assert THIRD_PARTY_URL.search(SVG_NAMESPACE), "the exemption must be a real hit"
        icon = (STATIC / "icon.svg").read_bytes()
        assert icon.count(SVG_NAMESPACE) == 1
        assert THIRD_PARTY_URL.findall(icon.replace(SVG_NAMESPACE, b"")) == []


class TestStaticServing:
    def test_a_path_outside_the_static_directory_is_refused(self, live):
        server, _ = live
        for attempt in ("/../server.py", "/..%2fserver.py", "/static/../../__main__.py"):
            status, _, _ = call(server, attempt, token="")
            assert status == HTTPStatus.NOT_FOUND, attempt

    def test_the_stylesheet_and_script_are_served_from_the_package(self, live):
        server, _ = live
        for path, needle in (("/app.css", b":focus-visible"), ("/app.js", b"X-Ledger-Token")):
            status, body, _ = call(server, path, token="")
            assert status == 200 and needle in body


class TestTheEndToEndPath:
    @needs_model
    def test_unlock_then_entry_then_report_then_wipe(self, live):
        server, store = live
        status, body, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
        assert status == 200 and json.loads(body)["created"] is True

        status, body, _ = call(server, "/api/entry", {"text": ORDINARY})
        assert status == 200
        analysis = json.loads(body)["analysis"]
        assert analysis["scored"] is True
        assert len(analysis["dimensions"]) == 5

        # R9-9: every dimension the interface can render arrives with the strength
        # of its evidence attached, so the page cannot show a bare number.
        for dimension in analysis["dimensions"]:
            assert "established" in dimension
            assert "held_out_auc" in dimension
            if not dimension["established"]:
                assert dimension["evidence_note"]

        # R8-4 through the UI path: an acute entry is never scored.
        status, body, _ = call(server, "/api/entry", {"text": CRISIS})
        assert status == 200
        crisis = json.loads(body)["analysis"]
        assert crisis["scored"] is False
        assert crisis["routed"]["triggered"] is True
        assert crisis["routed"]["helplines"]

        status, body, _ = call(server, "/api/entries")
        assert status == 200 and len(json.loads(body)["entries"]) == 2

        status, body, _ = call(server, "/api/report")
        assert status == 200 and len(json.loads(body)["report"]) > 400

        assert store.exists()

    def test_an_empty_entry_is_refused(self, live):
        """Self-sufficient, because a clean clone caught it borrowing an unlock.

        F-04 made `/api/entry` require the per-unlock session token, so this test
        stopped being about empty text on a clone: it inherited its unlock from
        the end-to-end test above, which SKIPS when the ONNX build is absent, and
        then got 401 instead of 400. That is exactly the F-02 defect class — a
        test standing on state a skippable test happened to leave behind. Found
        by cloning the published repository and running it, not on this machine.
        """
        server, store = live
        self._ensure_unlocked(server, store)
        status, _, _ = call(server, "/api/entry", {"text": "   "})
        assert status == HTTPStatus.BAD_REQUEST

    def _ensure_unlocked(self, server, store):
        """The wipe tests are about wiping, not about scoring.

        They used to inherit an unlocked journal from the end-to-end test above.
        That test now skips when the ONNX build is absent (F-02), so on a fresh
        clone these two would fail for a reason that has nothing to do with what
        they assert. Unlocking needs no model.
        """
        if server.state.journal is None:
            status, _, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            assert status == 200, status
        assert store.exists()
        assert server.state.session, "the caller needs an authenticated session"

    def test_wipe_needs_the_exact_word(self, live):
        server, store = live
        self._ensure_unlocked(server, store)
        for wrong in ("wipe", "WIPE ", "yes", ""):
            status, _, _ = call(server, "/api/wipe", {"confirm": wrong})
            assert status == HTTPStatus.BAD_REQUEST, wrong
            assert store.exists(), f"{wrong!r} destroyed the journal"

    def test_wipe_removes_the_store(self, live):
        server, store = live
        self._ensure_unlocked(server, store)
        status, body, _ = call(server, "/api/wipe", {"confirm": "WIPE"})
        assert status == 200 and json.loads(body)["wiped"] is True
        assert not store.exists()

    def test_reading_a_locked_journal_is_refused_not_leaked(self, live):
        """The status changed at revision round 1; the property did not.

        Before F-04 this returned 409 CONFLICT from the store layer, because the
        request reached the store and the store said "locked". Now the request
        is refused earlier, at 401, because a locked journal has no session token
        and no client can present one. 401 is the right code: the remedy is to
        supply the passphrase. What is asserted here is what was always asserted
        — the response says "locked" and carries no entry text.
        """
        server, _ = live
        status, body, _ = call(server, "/api/entries")
        assert status == HTTPStatus.UNAUTHORIZED
        assert b"locked" in body
        assert b"entries" not in body


class TestTheServerHoldsNothingItShouldNot:
    def test_the_passphrase_is_never_echoed_back(self, live, tmp_path):
        server = serve_in_thread(store=tmp_path / "j.enc", region="SG")
        try:
            _, body, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            assert PASSPHRASE.encode() not in body
            _, body, _ = call(server, "/api/state")
            assert PASSPHRASE.encode() not in body
        finally:
            server.shutdown_now()

    @needs_model
    def test_no_entry_text_is_cached_on_the_server_object(self, live, tmp_path):
        """Every read goes back through the encrypted store, by construction."""
        server = serve_in_thread(store=tmp_path / "k.enc", region="SG")
        try:
            call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            call(server, "/api/entry", {"text": ORDINARY})
            held = json.dumps({k: repr(v) for k, v in vars(server.state).items()})
            assert "dragged through the whole day" not in held
        finally:
            server.shutdown_now()


class TestUnlockingOneClientDoesNotUnlockTheMachine:
    """Round-1 audit finding **F-04**, closed at revision round 1.

    The auditor unlocked the journal in a Chromium context, then acted as an
    unrelated local process: `GET /` (unauthenticated by necessity — the browser
    must be able to load the page), scraped the run token out of the served
    HTML, and read the full decrypted journal with it. No passphrase at any
    point. The store is mode 0600 precisely to keep other local users out, and
    the listener was undoing that for as long as anything was unlocked.

    These tests are the shape of that attack, and the shape of the second half of
    the finding: the server used to hold the derived key indefinitely.
    """

    def test_the_page_still_serves_the_run_token_to_anyone(self, live):
        """Not fixed, and not fixable — asserted so the fix is not misread.

        `GET /` cannot require a secret; the browser has to load it. So the run
        token remains scrapeable by any local process. That is why it was
        demoted from authenticator to same-origin guard rather than defended.
        """
        server, _ = live
        status, body, _ = call(server, "/", method="GET")
        assert status == 200
        assert server.state.token.encode() in body

    def test_the_session_token_is_never_in_the_page(self, live, tmp_path):
        server = serve_in_thread(store=tmp_path / "f04a.enc", region="SG")
        try:
            status, body, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            assert status == 200
            session = json.loads(body)["session"]
            assert session and session != server.state.token

            _, page, _ = call(server, "/", method="GET")
            assert session.encode() not in page, "the session token leaked into the HTML"

            for asset in ("/app.js", "/app.css"):
                _, served, _ = call(server, asset, method="GET")
                assert session.encode() not in served
        finally:
            server.shutdown_now()

    def test_a_client_with_only_the_run_token_cannot_read_plaintext(self, live, tmp_path):
        """The auditor's attack, executed against the revised server."""
        server = serve_in_thread(store=tmp_path / "f04b.enc", region="SG")
        try:
            status, _, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            assert status == 200, "the legitimate client must be unlocked first"
            assert server.state.journal is not None

            for path in ("/api/entries", "/api/report"):
                status, body, _ = call(server, path, method="GET", session=False)
                assert status == HTTPStatus.UNAUTHORIZED, f"{path} answered {status}"
                assert b"locked" in body

            status, body, _ = call(server, "/api/entry",
                                   {"text": ORDINARY}, session=False)
            assert status == HTTPStatus.UNAUTHORIZED
        finally:
            server.shutdown_now()

    def test_the_legitimate_client_still_works(self, live, tmp_path):
        """The negative control: without it, a server that refused everything
        would pass the test above."""
        server = serve_in_thread(store=tmp_path / "f04c.enc", region="SG")
        try:
            status, body, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            session = json.loads(body)["session"]
            status, body, _ = call(server, "/api/entries", method="GET",
                                   session=session)
            assert status == 200, body
            assert "entries" in json.loads(body)
        finally:
            server.shutdown_now()

    def test_a_wrong_session_token_is_refused(self, live, tmp_path):
        server = serve_in_thread(store=tmp_path / "f04d.enc", region="SG")
        try:
            call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            status, _, _ = call(server, "/api/entries", method="GET",
                                session="x" * 43)
            assert status == HTTPStatus.UNAUTHORIZED
        finally:
            server.shutdown_now()

    def test_the_key_is_dropped_after_an_idle_period(self, live, tmp_path, monkeypatch):
        """The second half of F-04: the server never used to re-challenge."""
        monkeypatch.setattr(ui_server, "IDLE_LOCK_SECONDS", 0.05)
        server = serve_in_thread(store=tmp_path / "f04e.enc", region="SG")
        try:
            _, body, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            session = json.loads(body)["session"]
            assert server.state.journal is not None

            status, _, _ = call(server, "/api/entries", method="GET", session=session)
            assert status == 200, "should still be open before the timer fires"

            time.sleep(0.2)
            status, body, _ = call(server, "/api/entries", method="GET", session=session)
            assert status == HTTPStatus.UNAUTHORIZED
            assert server.state.journal is None, "the derived key was not dropped"
            assert server.state.session is None
        finally:
            server.shutdown_now()

    def test_locking_invalidates_the_session(self, live, tmp_path):
        server = serve_in_thread(store=tmp_path / "f04f.enc", region="SG")
        try:
            _, body, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            session = json.loads(body)["session"]
            status, _, _ = call(server, "/api/lock", {})
            assert status == 200
            status, _, _ = call(server, "/api/entries", method="GET", session=session)
            assert status == HTTPStatus.UNAUTHORIZED
        finally:
            server.shutdown_now()


class TestTheRunTokenCannotDestroyOrEvict:
    """Round-2 audit findings **AUDR2-F-001** (HIGH) and **AUDR2-F-002**.

    Revision round 1 closed F-04 for *reading* and left *destruction* open. The
    round-2 auditor planted a sentence, let a legitimate client unlock, then
    acted as an unrelated local process: `GET /` unauthenticated, scraped the
    43-character run token out of the page body, and `POST /api/wipe` with only
    `X-Ledger-Token`. The journal was overwritten and unlinked — irrecoverably,
    by design (docs/limitations.md §7.3) — with no passphrase at any point.

    The same token also evicted the legitimate session by *failing* an unlock,
    locked the user out via `/api/lock`, and read the store path and the
    unlocked flag out of `/api/state`.

    These tests are the shape of that attack. The F-04 tests above had the right
    shape already and simply did not cover these verbs, which is the reason the
    defect shipped.
    """

    @staticmethod
    def _journal_with_an_entry(path):
        """A store that holds something, so a wipe would destroy something."""
        from ledger.store.journal import Journal, JournalEntry
        journal = Journal(path, PASSPHRASE).create()
        journal.append(JournalEntry(entry_id="planted", written_at="2026-08-25T00:00:00Z",
                                    text="a sentence that must survive the attack"))
        return journal

    def test_a_client_with_only_the_run_token_cannot_wipe(self, tmp_path):
        """The auditor's attack, executed against the revised server."""
        store = tmp_path / "r2a.enc"
        self._journal_with_an_entry(store)
        server = serve_in_thread(store=store, region="SG")
        try:
            status, body, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            assert status == 200, "the legitimate client must be unlocked first"
            session = json.loads(body)["session"]

            # The attacker: run token only, no passphrase, no session.
            status, body, _ = call(server, "/api/wipe", {"confirm": "WIPE"},
                                   session=False)
            assert status == HTTPStatus.UNAUTHORIZED, f"the wipe was allowed: {body}"
            assert store.exists(), "the journal was destroyed by an unauthenticated client"

            # And the legitimate client is untouched by the attempt.
            status, body, _ = call(server, "/api/entries", method="GET", session=session)
            assert status == 200
            assert "must survive" in body.decode()
        finally:
            server.shutdown_now()

    def test_a_wrong_passphrase_does_not_wipe(self, tmp_path):
        store = tmp_path / "r2b.enc"
        self._journal_with_an_entry(store)
        server = serve_in_thread(store=store, region="SG")
        try:
            status, _, _ = call(server, "/api/wipe",
                                {"confirm": "WIPE", "passphrase": "not the passphrase"},
                                session=False)
            assert status == HTTPStatus.UNAUTHORIZED
            assert store.exists()
        finally:
            server.shutdown_now()

    def test_the_owner_can_wipe_with_the_passphrase_without_unlocking(self, tmp_path):
        """The negative control, and the reason wipe is not simply session-gated.

        Someone who needs their journal gone should not have to open it first.
        The passphrase proves ownership without putting a key in memory.
        """
        store = tmp_path / "r2c.enc"
        self._journal_with_an_entry(store)
        server = serve_in_thread(store=store, region="SG")
        try:
            status, body, _ = call(server, "/api/wipe",
                                   {"confirm": "WIPE", "passphrase": PASSPHRASE},
                                   session=False)
            assert status == 200, body
            assert json.loads(body)["wiped"] is True
            assert not store.exists()
            assert server.state.journal is None, "wipe must not leave a key in memory"
            assert server.state.session is None
        finally:
            server.shutdown_now()

    def test_the_unlocked_client_can_still_wipe(self, tmp_path):
        store = tmp_path / "r2d.enc"
        self._journal_with_an_entry(store)
        server = serve_in_thread(store=store, region="SG")
        try:
            _, body, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            session = json.loads(body)["session"]
            status, body, _ = call(server, "/api/wipe", {"confirm": "WIPE"},
                                   session=session)
            assert status == 200, body
            assert not store.exists()
        finally:
            server.shutdown_now()

    def test_an_empty_journal_still_cannot_authenticate_anyone(self, tmp_path):
        """A residual, asserted so it is not mistaken for closure.

        `Journal.unlock` verifies a passphrase by decrypting record 0, so a store
        with **no records authenticates every passphrase** — its own docstring
        says as much. The wipe gate therefore does not bind on a journal that
        holds zero entries, and neither would a session gate, because the same
        property means any passphrase can mint a session on it.

        What this costs is bounded and it is not the finding: zero entries are
        destroyed. Closing it means changing the on-disk format (a sealed
        sentinel record at creation), which is not a change to make in a
        remediation pass — revision round 1 opened AUDR2-F-001 exactly by
        reaching further than the defect. Stated in docs/limitations.md §7.7.
        """
        from ledger.store.journal import Journal
        store = tmp_path / "r2e.enc"
        Journal(store, PASSPHRASE).create()          # created, never written to
        server = serve_in_thread(store=store, region="SG")
        try:
            status, body, _ = call(server, "/api/wipe",
                                   {"confirm": "WIPE", "passphrase": "any string at all"},
                                   session=False)
            assert status == 200, "if this now refuses, the residual is closed — update §7.7"
            assert not store.exists()
        finally:
            server.shutdown_now()

    def test_a_wrong_passphrase_does_not_evict_the_legitimate_session(self, tmp_path):
        """AUDR2-F-002 (a). A failed unlock used to drop the derived key."""
        store = tmp_path / "r2f.enc"
        self._journal_with_an_entry(store)
        server = serve_in_thread(store=store, region="SG")
        try:
            _, body, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            session = json.loads(body)["session"]

            status, _, _ = call(server, "/api/unlock", {"passphrase": "wrong"},
                                session=False)
            assert status == HTTPStatus.UNAUTHORIZED

            assert server.state.session == session, "the attacker evicted the session"
            assert server.state.journal is not None, "the attacker dropped the derived key"
            status, body, _ = call(server, "/api/entries", method="GET", session=session)
            assert status == 200, f"the legitimate client was locked out: {body}"
        finally:
            server.shutdown_now()

    def test_a_client_with_only_the_run_token_cannot_lock(self, tmp_path):
        """AUDR2-F-002 (b)."""
        store = tmp_path / "r2g.enc"
        self._journal_with_an_entry(store)
        server = serve_in_thread(store=store, region="SG")
        try:
            _, body, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            session = json.loads(body)["session"]

            status, _, _ = call(server, "/api/lock", {}, session=False)
            assert status == HTTPStatus.UNAUTHORIZED

            status, _, _ = call(server, "/api/entries", method="GET", session=session)
            assert status == 200, "the attacker locked the legitimate client out"
        finally:
            server.shutdown_now()

    def test_failed_passphrase_attempts_back_off(self, tmp_path, monkeypatch):
        """AUDR2-F-002 (c). The guessing oracle was unlimited and silent."""
        monkeypatch.setattr(ui_server, "UNLOCK_BACKOFF_SECONDS", 30.0)
        store = tmp_path / "r2h.enc"
        self._journal_with_an_entry(store)
        server = serve_in_thread(store=store, region="SG")
        try:
            status, _, _ = call(server, "/api/unlock", {"passphrase": "wrong"},
                                session=False)
            assert status == HTTPStatus.UNAUTHORIZED

            status, body, headers = call(server, "/api/unlock", {"passphrase": "wrong"},
                                         session=False)
            assert status == HTTPStatus.TOO_MANY_REQUESTS, "the oracle is still unlimited"
            assert "Retry-After" in headers
            assert json.loads(body)["retry_after"] > 0

            # The wipe passphrase path is the same oracle and shares the limit.
            status, _, _ = call(server, "/api/wipe",
                                {"confirm": "WIPE", "passphrase": "wrong"},
                                session=False)
            assert status == HTTPStatus.TOO_MANY_REQUESTS
            assert store.exists()
        finally:
            server.shutdown_now()

    def test_the_backoff_doubles_and_a_correct_passphrase_clears_it(self, tmp_path,
                                                                    monkeypatch):
        monkeypatch.setattr(ui_server, "UNLOCK_BACKOFF_SECONDS", 0.05)
        store = tmp_path / "r2i.enc"
        self._journal_with_an_entry(store)
        server = serve_in_thread(store=store, region="SG")
        try:
            delays = []
            for _ in range(3):
                call(server, "/api/unlock", {"passphrase": "wrong"}, session=False)
                delays.append(server.state.throttle_remaining())
                time.sleep(server.state.throttle_remaining() + 0.01)
            assert delays[1] > delays[0] and delays[2] > delays[1], delays

            status, _, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE},
                                session=False)
            assert status == 200
            assert server.state.failed_attempts == 0
            assert server.state.throttle_remaining() == 0.0
        finally:
            server.shutdown_now()

    def test_the_backoff_is_capped(self, monkeypatch, tmp_path):
        """Bounded, because the limit is global and an attacker can trip it.

        The trade is recorded rather than hidden: a local process that knows the
        run token can hold the legitimate user at the cap by guessing wrong. A
        bounded wait for the user is the lesser cost against an unbounded
        guessing rate for the attacker, and the CLI path is unaffected.
        """
        monkeypatch.setattr(ui_server, "UNLOCK_BACKOFF_SECONDS", 1.0)
        monkeypatch.setattr(ui_server, "UNLOCK_BACKOFF_MAX_SECONDS", 4.0)
        state = ui_server.UIState(tmp_path / "r2j.enc", "SG")
        for _ in range(20):
            state.record_failed_attempt()
        assert state.throttle_remaining() <= 4.0

    def test_unauthenticated_state_discloses_neither_the_path_nor_the_key(self, tmp_path):
        """AUDR2-F-002, disclosure half."""
        store = tmp_path / "r2k.enc"
        self._journal_with_an_entry(store)
        server = serve_in_thread(store=store, region="SG")
        try:
            _, body, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            session = json.loads(body)["session"]
            assert server.state.journal is not None

            status, body, _ = call(server, "/api/state", method="GET", session=False)
            assert status == 200
            payload = json.loads(body)
            assert "store" not in payload, "the absolute store path leaked"
            assert payload["unlocked"] is False, \
                "an unauthenticated client learned a key is in memory"
            assert str(store) not in body.decode()

            # The negative control: the client that holds the session still sees both.
            _, body, _ = call(server, "/api/state", method="GET", session=session)
            payload = json.loads(body)
            assert payload["store"] == str(store)
            assert payload["unlocked"] is True
        finally:
            server.shutdown_now()

    def test_polling_state_does_not_hold_the_journal_open(self, tmp_path, monkeypatch):
        """`/api/state` must not refresh the idle clock, or a tab defeats it."""
        monkeypatch.setattr(ui_server, "IDLE_LOCK_SECONDS", 0.15)
        store = tmp_path / "r2l.enc"
        self._journal_with_an_entry(store)
        server = serve_in_thread(store=store, region="SG")
        try:
            _, body, _ = call(server, "/api/unlock", {"passphrase": PASSPHRASE})
            session = json.loads(body)["session"]
            for _ in range(6):
                time.sleep(0.05)
                call(server, "/api/state", method="GET", session=session)
            status, _, _ = call(server, "/api/entries", method="GET", session=session)
            assert status == HTTPStatus.UNAUTHORIZED, \
                "polling /api/state kept the derived key alive past the idle limit"
            assert server.state.journal is None
        finally:
            server.shutdown_now()
