"""`python -m ledger.ui` — open the local interface.

Prints the loopback URL to the terminal and, unless `--no-browser` is given,
asks the platform to open it. The URL carries no secret: the API token is served
inside the page, which only a same-origin document can read.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ledger.ui.server import DEFAULT_STORE, serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ledger-ui",
        description="Ledger's local interface. Binds 127.0.0.1 only; nothing it "
                    "computes leaves this machine.",
    )
    parser.add_argument("--store", default=str(DEFAULT_STORE),
                        help=f"path to the encrypted journal (default: {DEFAULT_STORE})")
    parser.add_argument("--region", default=os.environ.get("LEDGER_REGION"),
                        help="two-letter region for helpline selection, e.g. SG")
    parser.add_argument("--port", type=int, default=0,
                        help="loopback port (default: an ephemeral one)")
    parser.add_argument("--no-browser", action="store_true",
                        help="print the URL instead of opening it")
    args = parser.parse_args(argv)

    server = serve(store=Path(args.store), region=args.region, port=args.port)
    print(f"Ledger is running at {server.url}")
    print("Bound to 127.0.0.1 only. Close this terminal to stop it.")
    if not args.no_browser:
        # webbrowser is stdlib and hands a loopback URL to the platform opener.
        import webbrowser
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
    finally:
        server.shutdown_now()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
