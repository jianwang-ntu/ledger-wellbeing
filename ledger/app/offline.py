"""Force every ML library this application imports into offline mode.

Imported for its side effects, before `transformers` or `huggingface_hub`, by
everything in `ledger.app`.

This is deliberately blunt. `plan.md`'s first differentiator is "zero egress,
provable", and the most likely way for this project to break it is not a
deliberate upload — it is a library that checks for a newer model revision on
first use. Pointing the tokenizer at a local directory is not sufficient on its
own, because a cache-hit path can still issue a HEAD request.

Setting the environment inside a library is normally bad manners. It is done
here because the alternative is a promise that depends on the caller remembering
to keep it. `export/egress_audit.py` measures whether this actually worked, on
the running application, and `tests/test_egress.py` fails if it stops working.
"""

from __future__ import annotations

import os

#: Every switch is set to "1" rather than appended to, so the value is not
#: dependent on what the caller's environment already contained.
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "TOKENIZERS_PARALLELISM": "false",
}


def enforce() -> dict:
    """Apply the offline environment. Returns what was set, for the evidence log."""
    for key, value in OFFLINE_ENV.items():
        os.environ[key] = value
    return dict(OFFLINE_ENV)


enforce()
