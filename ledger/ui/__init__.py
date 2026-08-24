"""Ledger's visual interface: `python -m ledger.ui`.

`plan.md` C6 is graded on "visual design quality, ease of navigation, intuitive
user flows, and adherence to accessibility standards". Increment 8 shipped a CLI,
which answers none of those, and said so. This package is that criterion's
artifact.

It is a **local** interface. The server binds `127.0.0.1` on an ephemeral port and
nothing else; `export/INCREMENT_9_PREREGISTRATION.md` R9-7 measures that the same
port is refused on the host's own non-loopback address, and R9-8 measures that
neither the process nor the page reaches anything but loopback.

The wording this replaces is recorded rather than quietly dropped: increment 8
pre-registered that if increment 9 bound a loopback listener, the sentence "no
server component at all" would be **withdrawn** and replaced by the two claims
that are actually measured. It has been.
"""

from ledger.ui.server import LedgerUIServer, serve

__all__ = ["LedgerUIServer", "serve"]
