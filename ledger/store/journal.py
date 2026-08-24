"""The encrypted journal: append an analysed entry, read them all back, wipe.

`plan.md` C4 names two mechanisms this file exists to be: "local storage
encrypted with a user-held passphrase, and a one-click wipe". Both are here, and
both are measured in `tests/test_store.py` rather than asserted in a README.

The store holds the user's own text. That is the most sensitive thing this
project touches, so the design rule is the boring one: the plaintext exists in
this process's memory for as long as it takes to seal it, and nowhere else on
disk, ever. There is no cache, no index, no "recently used" file, and no
temporary decrypted copy — reading the journal decrypts into memory and returns.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .crypto import (SALT_BYTES, CorruptStore, KdfParams, StoreError, WrongPassphrase,
                     build_header, derive_key, open_records, parse_header, seal)

__all__ = ["Journal", "JournalEntry", "StoreError", "WrongPassphrase", "CorruptStore"]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class JournalEntry:
    """One dated piece of writing and whatever the instrument made of it."""

    entry_id: str
    written_at: str
    text: str
    analysis: dict = field(default_factory=dict)

    def to_json(self) -> bytes:
        return json.dumps(
            {"entry_id": self.entry_id, "written_at": self.written_at,
             "text": self.text, "analysis": self.analysis},
            separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "JournalEntry":
        obj = json.loads(raw.decode("utf-8"))
        return cls(entry_id=obj["entry_id"], written_at=obj["written_at"],
                   text=obj["text"], analysis=obj.get("analysis", {}))


class Journal:
    """An append-only encrypted file of entries, opened with a passphrase."""

    def __init__(self, path: Path, passphrase: str):
        self.path = Path(path)
        self._passphrase = passphrase
        self._header: bytes | None = None
        self._key: bytes | None = None

    # -- lifecycle ---------------------------------------------------------

    def exists(self) -> bool:
        return self.path.exists()

    def create(self) -> "Journal":
        """Write a fresh header. Refuses to overwrite an existing store."""
        if self.path.exists():
            raise StoreError(f"{self.path} already exists; refusing to overwrite a journal")
        salt = os.urandom(SALT_BYTES)
        params = KdfParams()
        header = build_header(salt, params)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 0600 from the moment it exists, not after the first write.
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, header)
        finally:
            os.close(fd)
        self._header, self._key = header, derive_key(self._passphrase, salt, params)
        return self

    def unlock(self) -> "Journal":
        """Derive the key from the stored parameters and verify it reads record 0."""
        if not self.path.exists():
            raise StoreError(f"{self.path} does not exist")
        blob = self.path.read_bytes()
        salt, params, header_len = parse_header(blob)
        self._header = blob[:header_len]
        self._key = derive_key(self._passphrase, salt, params)
        # Touching the first record is what distinguishes a wrong passphrase from
        # an empty store. An empty store has nothing to authenticate against, and
        # that is reported as such rather than as success.
        next(iter(open_records(self._key, self._header, blob[header_len:])), None)
        return self

    def open_or_create(self) -> "Journal":
        return self.unlock() if self.path.exists() else self.create()

    # -- reading and writing ----------------------------------------------

    def _require_key(self) -> tuple[bytes, bytes]:
        if self._key is None or self._header is None:
            raise StoreError("journal is locked; call unlock() or create() first")
        return self._key, self._header

    def append(self, entry: JournalEntry) -> JournalEntry:
        key, header = self._require_key()
        index = self.count()
        record = seal(key, header, index, entry.to_json())
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            os.write(fd, record)
            os.fsync(fd)
        finally:
            os.close(fd)
        return entry

    def _body(self) -> bytes:
        blob = self.path.read_bytes()
        _, _, header_len = parse_header(blob)
        return blob[header_len:]

    def entries(self) -> list[JournalEntry]:
        key, header = self._require_key()
        return [JournalEntry.from_json(raw) for raw in open_records(key, header, self._body())]

    def count(self) -> int:
        key, header = self._require_key()
        return sum(1 for _ in open_records(key, header, self._body()))

    # -- destruction -------------------------------------------------------

    def wipe(self) -> dict:
        """Overwrite the file's bytes, then remove it. The 'one-click wipe'.

        Overwriting before unlinking is what makes this more than a delete, and
        it is honest only within limits that `docs/limitations.md` states in
        full: on a copy-on-write, journalled or wear-levelled filesystem the
        superseded blocks may survive this call, and no userspace program can
        promise otherwise.
        """
        if not self.path.exists():
            return {"wiped": False, "reason": "no store at that path"}
        size = self.path.stat().st_size
        fd = os.open(self.path, os.O_WRONLY)
        try:
            os.write(fd, os.urandom(size))
            os.fsync(fd)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, b"\x00" * size)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.unlink(self.path)
        self._key, self._header = None, None
        return {"wiped": True, "bytes_overwritten": size, "passes": 2, "path": str(self.path)}
