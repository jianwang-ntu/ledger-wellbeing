"""Tokenization for the shipped application, without `transformers`.

DEFECT-INC8-001, found by the first end-to-end CLI run rather than by a test.
The application originally loaded the tokenizer through `transformers`, which
imports `sklearn`, which imports `pyarrow`. On this machine that fails outright
once `onnxruntime` or `torch` has already been imported::

    ImportError: libstdc++.so.6: version `CXXABI_1.3.15' not found
                 (required by .../libarrow.so.2100)

The suite did not catch it because pytest's collection order happens to import
things in an order where it works. A product whose startup depends on import
order is broken; a product that drags a dataframe library and an ML training
framework into a journal app is also just wrong, independently of the crash.

So the application reads `tokenizer.json` with the `tokenizers` library
directly. This is not a workaround — it removes `transformers`, `sklearn`,
`pyarrow` and `torch` from everything the shipped application touches. The
runtime dependency set becomes `onnxruntime`, `numpy`, `tokenizers`,
`cryptography`. It also removes the last library in the application that has any
hub-checking code path at all, which is worth something to R8-3 on its own.

The measured pipeline (`export/verify.py`, `export/span_additivity.py`) was run
through the `transformers` path, so switching is only legitimate if the two
produce **identical** encodings. That is not assumed: `export/tokenizer_parity.py`
measures `input_ids`, `attention_mask` and `offset_mapping` elementwise over the
64 probe entries and all 50 anchor sentences, and `tests/test_local_tokenizer.py`
fails if the measurement is missing or shows a single mismatch — the same
protocol R7-2 used for the tokenizer re-serialization.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOKENIZER_DIR = ROOT / "artifacts" / "tokenizer"

#: CEIL-4 is stated at this length, so the application runs at this length.
MAX_LENGTH = 256


class TokenizerUnavailable(RuntimeError):
    """The tokenizer directory is absent. Raised rather than falling back."""


@lru_cache(maxsize=2)
def load(max_length: int = MAX_LENGTH):
    """Return a `tokenizers.Tokenizer` configured exactly as the export used it.

    Truncation and padding are set here to match
    `tokenizer(..., padding="max_length", truncation=True, max_length=256)` —
    the call the whole measured pipeline was run through.
    """
    from tokenizers import Tokenizer

    path = TOKENIZER_DIR / "tokenizer.json"
    if not path.exists():
        raise TokenizerUnavailable(
            f"{path} is absent. It is a build output; regenerate it with "
            "`bash export/run_all.sh`."
        )
    tokenizer = Tokenizer.from_file(str(path))

    config = json.loads((TOKENIZER_DIR / "tokenizer_config.json").read_text())
    pad_token = config.get("pad_token", "<pad>")
    pad_id = tokenizer.token_to_id(pad_token)
    if pad_id is None:
        raise TokenizerUnavailable(f"pad token {pad_token!r} is not in the vocabulary")

    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.enable_padding(length=max_length, pad_id=pad_id, pad_token=pad_token)
    return tokenizer


def encode(text: str, max_length: int = MAX_LENGTH):
    """Encode one entry. Returns (input_ids, attention_mask, offsets) as lists.

    Offsets are character offsets into `text`; special and padding tokens carry
    `(0, 0)`, which is what `ledger.app.spans` routes into the structural bucket.
    """
    encoding = load(max_length).encode(text)
    return list(encoding.ids), list(encoding.attention_mask), list(encoding.offsets)
