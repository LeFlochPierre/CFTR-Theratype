from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cftr_theratype import checkpoint as ck


def workspace(tmp_path: Path) -> Path:
    for rel in ck.REQUIRED_COPY:
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("training-only placeholder\n")
    source = tmp_path / "src" / "cftr_theratype" / "demo.py"
    source.parent.mkdir(parents=True)
    source.write_text("VERSION = 1\n")
    return tmp_path


def test_create_verify_and_no_overwrite(tmp_path):
    root = workspace(tmp_path)
    ck.create(root, dry_run=False)
    assert ck.verify(root) == 0
    with pytest.raises(FileExistsError):
        ck.create(root, dry_run=False)


def test_raw_not_copied_or_modified(tmp_path):
    root = workspace(tmp_path)
    original = root / "data/raw/private.xlsx"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"SENSITIVE RAW BYTES")
    ck.create(root, dry_run=False)
    snapshot = root / ck.RELATIVE_DEST
    assert original.read_bytes() == b"SENSITIVE RAW BYTES"
    assert not (snapshot / "files/data/raw").exists()


def test_verify_catches_tampering(tmp_path):
    root = workspace(tmp_path)
    ck.create(root, dry_run=False)
    copy = root / ck.RELATIVE_DEST / "files" / ck.REQUIRED_COPY[0]
    copy.write_text("changed")
    assert ck.verify(root) == 1


def test_dry_run_has_no_side_effect(tmp_path):
    root = workspace(tmp_path)
    ck.create(root, dry_run=True)
    assert not (root / ck.RELATIVE_DEST).exists()


def test_missing_essential_file_blocks_snapshot(tmp_path):
    root = workspace(tmp_path)
    (root / ck.REQUIRED_COPY[0]).unlink()
    with pytest.raises(RuntimeError, match="Missing required"):
        ck.create(root, dry_run=False)
    assert not (root / ck.RELATIVE_DEST).exists()


def test_reject_path_escape_and_symlinks(tmp_path):
    root = workspace(tmp_path)
    with pytest.raises(ValueError, match="Unsafe"):
        ck.safe_file(root, "../elsewhere")
    target = root / "rogue"
    target.symlink_to(root / ck.REQUIRED_COPY[0])
    with pytest.raises(ValueError, match="Symlink"):
        ck.safe_file(root, "rogue")
