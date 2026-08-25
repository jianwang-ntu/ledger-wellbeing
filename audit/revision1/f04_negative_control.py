"""Reproduce round-1 F-04 against the PRE-FIX server, then against the revised
one, with the auditor's own probe and a GET-shaped variant of the attack the
auditor described in prose.
"""
import json, os, re, shutil, subprocess, sys, tempfile, threading, time, urllib.request, urllib.error

PROJ = "/home/wj/wj_code/dl_hackathon/devpost_dirs/workspaces/hack-for-humanity-summer-26/project"
PROBE = "/home/wj/wj_code/dl_hackathon/devpost_dirs/workspaces/hack-for-humanity-summer-26/audit/round1/scratch/probe_session_unlock.py"
PASS = "a probe passphrase for F-04"
PLAINTEXT = "I lay awake until nearly four again, watching the ceiling."


def seed(store):
    sys.path.insert(0, PROJ)
    from ledger.store.journal import Journal, JournalEntry
    j = Journal(store, PASS).create()
    j.append(JournalEntry(entry_id="e1", written_at="2026-08-25T00:00:00Z",
                          text=PLAINTEXT, analysis={"routed": "ordinary", "scored": False}))
    return j.count()


def get(port, path, headers):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 headers={"Host": f"127.0.0.1:{port}", **headers})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")
        except Exception as e:
            last = e
            time.sleep(0.3)
    return -1, f"TRANSPORT_ERROR: {type(last).__name__}: {last}"


def post(port, path, body, headers):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(body).encode(),
                                 headers={"Host": f"127.0.0.1:{port}",
                                          "Content-Type": "application/json", **headers},
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def attack_get(port):
    """The attack as the auditor DESCRIBED it: GET /, scrape token, GET the reads."""
    out = {}
    s, html = get(port, "/", {})
    out["GET_/_status"] = s
    m = re.search(r'name="ledger-token"\s+content="([A-Za-z0-9_\-]{8,})"', html) \
        or re.search(r'content="([A-Za-z0-9_\-]{20,})"', html)
    tok = m.group(1) if m else None
    out["token_scraped_from_page"] = bool(tok)
    if not tok:
        return out
    H = {"X-Ledger-Token": tok}
    for path in ("/api/entries", "/api/report"):
        st, body = get(port, path, H)
        out[f"GET {path}"] = {"status": st, "body_head": body[:300]}
        out[f"plaintext_in {path}"] = PLAINTEXT in body
    out["NO_PASSPHRASE_WAS_SUPPLIED"] = True
    out["plaintext_recovered"] = any(v for k, v in out.items() if k.startswith("plaintext_in "))
    return out


def run_case(label, tree):
    """Start the server from `tree`, unlock it legitimately, then attack it."""
    env = dict(os.environ, PYTHONPATH=tree)
    store = tempfile.mkdtemp(prefix="f04-") + "/journal.enc"
    code = (
        "import json,sys,threading;"
        f"sys.path.insert(0,{tree!r});"
        "from ledger.ui.server import serve_in_thread;"
        f"s=serve_in_thread(store={store!r}, region='SG');"
        "print(json.dumps({'port':s.server_port,'token':s.state.token}),flush=True);"
        "threading.Event().wait()"
    )
    # seed the store first, with this tree's own store layer
    subprocess.run([sys.executable, "-c",
                    f"import sys;sys.path.insert(0,{tree!r});"
                    "from ledger.store.journal import Journal, JournalEntry;"
                    f"j=Journal({store!r},{PASS!r}).create();"
                    f"j.append(JournalEntry(entry_id='e1',written_at='2026-08-25T00:00:00Z',text={PLAINTEXT!r},analysis={{'routed':'ordinary','scored':False}}));"
                    "print(j.count())"], check=True, env=env, capture_output=True)
    proc = subprocess.Popen([sys.executable, "-c", code], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        info = json.loads(proc.stdout.readline())
        port, token = info["port"], info["token"]
        # the LEGITIMATE client unlocks, supplying the passphrase
        st, body = post(port, "/api/unlock", {"passphrase": PASS},
                        {"X-Ledger-Token": token})
        legit = {"unlock_status": st, "has_session": "session" in json.loads(body or "{}")}
        # the auditor's probe, run UNMODIFIED
        probe = subprocess.run([sys.executable, PROBE, str(port)],
                               capture_output=True, text=True, timeout=120)
        return {
            "tree": label,
            "legitimate_unlock": legit,
            "auditor_probe_unmodified": json.loads(probe.stdout or "{}"),
            "auditor_probe_rc": probe.returncode,
            "attack_as_described_GET": attack_get(port),
        }
    finally:
        proc.kill()


control_tree = tempfile.mkdtemp(prefix="f04-control-")
shutil.copytree(PROJ, control_tree + "/t", symlinks=True,
                ignore=shutil.ignore_patterns(".git", "artifacts", "a11y",
                                              "__pycache__", ".venv*", ".pytest_cache"))
CT = control_tree + "/t"
for f in ("ledger/ui/server.py", "ledger/ui/static/app.js"):
    pre = subprocess.run(["git", "-C", PROJ, "show", f"HEAD:{f}"],
                         capture_output=True, check=True).stdout
    open(os.path.join(CT, f), "wb").write(pre)

report = {
    "what": "F-04 negative control and fix verification, revision round 1",
    "plaintext_planted": PLAINTEXT,
    "control_is": "ledger/ui/server.py and static/app.js as at git HEAD (pre-fix), everything else current",
    "cases": [run_case("PRE-FIX (git HEAD)", CT),
              run_case("REVISED (working tree)", PROJ)],
}
print(json.dumps(report, indent=2))
