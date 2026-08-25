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


def call(server, path, body=None, *, token=None, headers=None, method=None):
    url = server.url.rstrip("/") + path
    request = urllib.request.Request(url, method=method or ("POST" if body is not None else "GET"))
    request.add_header("X-Ledger-Token",
                       server.state.token if token is None else token)
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
        server, _ = live
        status, _, _ = call(server, "/api/entry", {"text": "   "})
        assert status == HTTPStatus.BAD_REQUEST

    def test_wipe_needs_the_exact_word(self, live):
        server, store = live
        for wrong in ("wipe", "WIPE ", "yes", ""):
            status, _, _ = call(server, "/api/wipe", {"confirm": wrong})
            assert status == HTTPStatus.BAD_REQUEST, wrong
            assert store.exists(), f"{wrong!r} destroyed the journal"

    def test_wipe_removes_the_store(self, live):
        server, store = live
        status, body, _ = call(server, "/api/wipe", {"confirm": "WIPE"})
        assert status == 200 and json.loads(body)["wiped"] is True
        assert not store.exists()

    def test_reading_a_locked_journal_is_a_conflict_not_a_leak(self, live):
        server, _ = live
        status, body, _ = call(server, "/api/entries")
        assert status == HTTPStatus.CONFLICT
        assert b"locked" in body


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
