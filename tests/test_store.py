"""Increment 8 guards on the encrypted store — R8-5 and R8-6.

These are written to try to get the plaintext *out* of the file, because that is
the failure that hurts a user. A test that writes an entry and reads it back
proves the store works; it proves nothing about whether it protects anything.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ledger.store.crypto import (KEY_BYTES, SCRYPT_N, SCRYPT_P, SCRYPT_R,  # noqa: E402
                                 CorruptStore, KdfParams, StoreError, WrongPassphrase,
                                 derive_key)
from ledger.store.journal import Journal, JournalEntry  # noqa: E402

PASSPHRASE = "correct horse battery staple, and then some"

#: Deliberately distinctive so a substring search cannot match by accident.
SECRETS = [
    "I lay awake until four thinking about the appointment on Thursday.",
    "My sister called and I did not pick up, which I regret now.",
    "Everything took more effort than it should have, again.",
]


def _journal(tmp_path: Path, entries=SECRETS) -> Journal:
    journal = Journal(tmp_path / "journal.enc", PASSPHRASE).create()
    for i, text in enumerate(entries):
        journal.append(JournalEntry(f"e{i}", f"2026-08-{10 + i:02d}T09:00:00Z", text,
                                    {"scored": True}))
    return journal


class TestRoundTrip:
    def test_entries_come_back_in_order_and_unchanged(self, tmp_path):
        self._assert_roundtrip(_journal(tmp_path))

    def test_a_reopened_journal_reads_the_same(self, tmp_path):
        _journal(tmp_path)
        self._assert_roundtrip(Journal(tmp_path / "journal.enc", PASSPHRASE).unlock())

    @staticmethod
    def _assert_roundtrip(journal):
        entries = journal.entries()
        assert [e.text for e in entries] == SECRETS
        assert [e.entry_id for e in entries] == ["e0", "e1", "e2"]

    def test_create_refuses_to_overwrite(self, tmp_path):
        _journal(tmp_path)
        with pytest.raises(StoreError):
            Journal(tmp_path / "journal.enc", PASSPHRASE).create()

    def test_file_is_owner_only(self, tmp_path):
        journal = _journal(tmp_path)
        assert oct(journal.path.stat().st_mode & 0o777) == "0o600"


class TestNoPlaintextOnDisk:
    """R8-5. The point of the whole module."""

    def test_no_twelve_character_run_of_any_entry_appears_in_the_file(self, tmp_path):
        journal = _journal(tmp_path)
        blob = journal.path.read_bytes()
        for text in SECRETS:
            for start in range(0, len(text) - 12):
                fragment = text[start:start + 12].encode("utf-8")
                assert fragment not in blob, f"plaintext fragment on disk: {fragment!r}"

    def test_json_field_names_do_not_leak_either(self, tmp_path):
        journal = _journal(tmp_path)
        blob = journal.path.read_bytes()
        for marker in (b'"text"', b'"analysis"', b'"entry_id"', b"scored"):
            assert marker not in blob

    def test_the_body_does_not_compress_like_text(self, tmp_path):
        """Ciphertext is incompressible; a plaintext body would not be."""
        import zlib
        journal = _journal(tmp_path)
        body = journal._body()
        assert len(zlib.compress(body, 9)) > 0.95 * len(body)


class TestWrongPassphrase:
    def test_wrong_passphrase_raises_and_returns_nothing(self, tmp_path):
        _journal(tmp_path)
        with pytest.raises(WrongPassphrase):
            Journal(tmp_path / "journal.enc", "not the passphrase").unlock()

    def test_empty_passphrase_is_refused_outright(self, tmp_path):
        with pytest.raises(StoreError):
            Journal(tmp_path / "journal.enc", "").create()

    def test_a_near_miss_is_still_a_miss(self, tmp_path):
        _journal(tmp_path)
        with pytest.raises(WrongPassphrase):
            Journal(tmp_path / "journal.enc", PASSPHRASE + " ").unlock()


class TestTampering:
    """A store that silently drops what it cannot authenticate can be edited."""

    @pytest.mark.parametrize("offset_from_end", [1, 40, 100])
    def test_a_flipped_byte_is_rejected_not_decrypted(self, tmp_path, offset_from_end):
        journal = _journal(tmp_path)
        blob = bytearray(journal.path.read_bytes())
        index = len(blob) - offset_from_end
        blob[index] ^= 0x01
        journal.path.write_bytes(bytes(blob))
        with pytest.raises((CorruptStore, WrongPassphrase)):
            Journal(journal.path, PASSPHRASE).unlock().entries()

    def test_a_record_cannot_be_moved_to_another_position(self, tmp_path):
        """Position is bound into the AAD, so reordering must fail."""
        journal = _journal(tmp_path)
        blob = journal.path.read_bytes()
        _, _, header_len = __import__("ledger.store.crypto", fromlist=["parse_header"]) \
            .parse_header(blob)
        header, body = blob[:header_len], blob[header_len:]

        import struct
        records, cursor = [], 0
        while cursor < len(body):
            (length,) = struct.unpack_from(">I", body, cursor)
            records.append(body[cursor:cursor + 4 + length])
            cursor += 4 + length
        assert len(records) == 3

        journal.path.write_bytes(header + records[1] + records[0] + records[2])
        with pytest.raises((CorruptStore, WrongPassphrase)):
            Journal(journal.path, PASSPHRASE).unlock().entries()

    def test_a_record_from_another_store_is_rejected(self, tmp_path):
        """AAD includes the header, and the header includes a per-store salt."""
        first = _journal(tmp_path / "a")
        second = _journal(tmp_path / "b")
        import struct
        from ledger.store.crypto import parse_header

        blob_b = second.path.read_bytes()
        _, _, header_len_b = parse_header(blob_b)
        (length,) = struct.unpack_from(">I", blob_b, header_len_b)
        foreign = blob_b[header_len_b:header_len_b + 4 + length]

        first.path.write_bytes(first.path.read_bytes() + foreign)
        with pytest.raises(CorruptStore):
            Journal(first.path, PASSPHRASE).unlock().entries()


class TestWipe:
    """R8-6."""

    def test_wipe_removes_the_file(self, tmp_path):
        journal = _journal(tmp_path)
        result = journal.wipe()
        assert result["wiped"] is True
        assert not journal.path.exists()

    def test_wipe_overwrites_before_unlinking(self, tmp_path, monkeypatch):
        journal = _journal(tmp_path)
        original = journal.path.read_bytes()
        written: list[bytes] = []
        real_write = os.write

        def spy(fd, data):
            written.append(bytes(data))
            return real_write(fd, data)

        monkeypatch.setattr(os, "write", spy)
        journal.wipe()
        assert len(written) == 2, "expected a random pass and a zero pass"
        assert all(len(chunk) == len(original) for chunk in written)
        assert written[0] != original
        assert written[1] == b"\x00" * len(original)

    def test_wipe_does_not_need_the_passphrase(self, tmp_path):
        journal = _journal(tmp_path)
        assert Journal(journal.path, "wrong on purpose").wipe()["wiped"] is True

    def test_wiping_nothing_is_not_an_error(self, tmp_path):
        assert Journal(tmp_path / "absent.enc", PASSPHRASE).wipe()["wiped"] is False


class TestTheKdfIsNotQuietlyWeakened:
    """The scrypt cost is the only thing between a weak passphrase and the file."""

    def test_parameters_hold_their_declared_values(self):
        assert (SCRYPT_N, SCRYPT_R, SCRYPT_P, KEY_BYTES) == (1 << 15, 8, 1, 32)

    def test_the_header_records_the_parameters_actually_used(self, tmp_path):
        from ledger.store.crypto import parse_header
        journal = _journal(tmp_path)
        _, params, _ = parse_header(journal.path.read_bytes())
        assert (params.n, params.r, params.p, params.dklen) == (SCRYPT_N, SCRYPT_R,
                                                                SCRYPT_P, KEY_BYTES)

    def test_derive_key_matches_the_stdlib_directly(self):
        """No custom stretching wrapped around the KDF — it is scrypt, unmodified."""
        salt = b"0123456789abcdef"
        expected = hashlib.scrypt(b"pw", salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
                                  dklen=KEY_BYTES, maxmem=64 * 1024 * 1024)
        assert derive_key("pw", salt, KdfParams()) == expected

    def test_two_stores_with_the_same_passphrase_get_different_keys(self, tmp_path):
        from ledger.store.crypto import parse_header
        a = _journal(tmp_path / "a")
        b = _journal(tmp_path / "b")
        salt_a, params, _ = parse_header(a.path.read_bytes())
        salt_b, _, _ = parse_header(b.path.read_bytes())
        assert salt_a != salt_b
        assert derive_key(PASSPHRASE, salt_a, params) != derive_key(PASSPHRASE, salt_b, params)


class TestNoHandRolledCrypto:
    """R8-5's last clause, asserted against the source rather than trusted."""

    SOURCE = (ROOT / "ledger" / "store" / "crypto.py").read_text()

    def test_the_aead_is_the_libraries(self):
        assert "from cryptography.hazmat.primitives.ciphers.aead import AESGCM" in self.SOURCE
        assert "hashlib.scrypt" in self.SOURCE

    def test_no_xor_loop_or_hand_built_mac_appears(self):
        for smell in ("^= ", "hmac.new", "def encrypt(", "def _xor"):
            assert smell not in self.SOURCE, f"hand-rolled crypto smell: {smell!r}"
