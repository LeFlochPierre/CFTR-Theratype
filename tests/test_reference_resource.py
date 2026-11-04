"""Verify the pinned public P13569 reference works in clean worktrees."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cftr_theratype import reference


UNIPROT_MD5 = "f29b8c0a9056a0f7680f3290d259b6ac"


def test_packaged_cftr_reference_is_complete_and_verified(monkeypatch, tmp_path):
    monkeypatch.setattr(reference, "UNIPROT_FASTA", tmp_path / "missing.fasta")
    sequence = reference.canonical_cftr_sequence()

    assert len(sequence) == 1480
    assert sequence[469] == "V"
    assert hashlib.md5(sequence.encode("ascii")).hexdigest() == UNIPROT_MD5

    assay = reference.bihler_2024_assay_sequence()
    assert assay[469] == "M"
    assert sum(a != b for a, b in zip(sequence, assay)) == 1


def test_local_reference_still_supported_when_matching_pinned_sequence(monkeypatch, tmp_path):
    local = tmp_path / "data/raw/uniprot/P13569.fasta"
    local.parent.mkdir(parents=True)
    local.write_bytes(reference.PACKAGED_UNIPROT_FASTA.read_bytes())
    monkeypatch.setattr(reference, "UNIPROT_FASTA", local)

    assert reference.canonical_cftr_sequence()[469] == "V"


def test_wrong_local_cftr_reference_is_rejected(monkeypatch, tmp_path):
    local = tmp_path / "P13569.fasta"
    contents = reference.PACKAGED_UNIPROT_FASTA.read_text(encoding="utf-8")
    assert "\nMQRSP" in contents
    local.write_text(contents.replace("\nMQRSP", "\nAQRSP", 1), encoding="utf-8")
    monkeypatch.setattr(reference, "UNIPROT_FASTA", local)

    with pytest.raises(ValueError, match="canonical P13569-1"):
        reference.canonical_cftr_sequence()
