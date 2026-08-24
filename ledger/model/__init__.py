"""What Ledger measures, and the torch module that was used to build it.

`dimensions` is *data* — names, labels and anchor phrases — and the shipped
application needs it. `scorer` is a `torch.nn.Module` and is needed only by the
export pipeline, which runs on a developer's machine and never on a user's.

So `LedgerScorer` and `build_anchor_head` are resolved lazily. Importing
`ledger.model.DIMENSIONS` used to drag `torch` into the process through this
file, which is half of DEFECT-INC8-001 and is guarded by
`tests/test_engine.py::TestTheApplicationRunsWithoutTransformers`.

The public API is unchanged: `from ledger.model import LedgerScorer` still works
and still imports torch — at the moment it is asked for, not before.
"""

from .dimensions import ANCHORS, DIMENSIONS, HEAD_VERSION

__all__ = ["DIMENSIONS", "ANCHORS", "HEAD_VERSION", "LedgerScorer", "build_anchor_head"]

_LAZY = {"LedgerScorer", "build_anchor_head"}


def __getattr__(name: str):
    if name in _LAZY:
        from . import scorer
        return getattr(scorer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
