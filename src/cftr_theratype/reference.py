"""Reference sequences used for CFTR variant modelling.

Bihler et al. (2024) used a CFTR construct containing V470M
relative to canonical UniProt P13569.

Reference: Bihler et al., J Cyst Fibros 2024, Methods 2.1.
DOI: 10.1016/j.jcf.2024.02.006
"""

from pathlib import Path
import hashlib

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


ROOT = Path(__file__).resolve().parents[2]
UNIPROT_FASTA = ROOT / "data/raw/uniprot/P13569.fasta"
PACKAGED_UNIPROT_FASTA = Path(__file__).resolve().parent / "resources/P13569.fasta"
UNIPROT_SEQUENCE_MD5 = "f29b8c0a9056a0f7680f3290d259b6ac"
ASSAY_FASTA = ROOT / "data/processed/bihler_2024_assay_reference.fasta"


def canonical_cftr_sequence() -> str:
    """Canonical human CFTR protein sequence, UniProt P13569."""
    # Preserve the established local reference path, but allow fresh worktrees
    # and installed distributions to use the pinned, public package resource.
    fasta = UNIPROT_FASTA if UNIPROT_FASTA.is_file() else PACKAGED_UNIPROT_FASTA
    sequence = str(SeqIO.read(fasta, "fasta").seq)

    # Length/V470 alone would not detect other residue substitutions.
    digest = hashlib.md5(sequence.encode("ascii")).hexdigest()
    if (
        len(sequence) != 1480
        or sequence[469] != "V"
        or digest != UNIPROT_SEQUENCE_MD5
    ):
        raise ValueError(
            f"Unexpected UniProt CFTR reference at {fasta}. "
            "Expected canonical P13569-1, length 1480, V470, "
            f"MD5 {UNIPROT_SEQUENCE_MD5}."
        )

    return sequence


def bihler_2024_assay_sequence() -> str:
    """Reconstruct the protein background used in Bihler 2024."""
    canonical = canonical_cftr_sequence()

    assay = canonical[:469] + "M" + canonical[470:]

    differences = [
        i + 1
        for i, (a, b) in enumerate(zip(canonical, assay))
        if a != b
    ]

    if differences != [470]:
        raise ValueError("Unexpected experimental reference differences.")

    return assay


def write_bihler_2024_assay_fasta() -> Path:
    """Export the experimental WT protein sequence for ML features."""
    sequence = bihler_2024_assay_sequence()

    ASSAY_FASTA.parent.mkdir(parents=True, exist_ok=True)

    record = SeqRecord(
        Seq(sequence),
        id="CFTR_Bihler2024_WT_M470",
        description=(
            "Experimental CFTR protein background derived from "
            "UniProt P13569 with V470M"
        ),
    )

    SeqIO.write(record, ASSAY_FASTA, "fasta")

    return ASSAY_FASTA
