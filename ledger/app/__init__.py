"""The application layer: entry, routing, scoring, attribution, storage, report.

`offline` is imported first and for its side effects. Everything downstream of
it assumes the ML libraries have already been pinned offline.
"""

from . import offline  # noqa: F401

__all__ = ["offline"]
