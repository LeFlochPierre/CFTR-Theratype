"""Reference sequences used for CFTR variant modelling.

Bihler et al. (2024) used a CFTR construct containing V470M
relative to canonical UniProt P13569.

Reference: Bihler et al., J Cyst Fibros 2024, Methods 2.1.
DOI: 10.1016/j.jcf.2024.02.006
"""

from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


ROOT = Path(__file__).resolve().parents[2]
UNIPROT_FASTA = ROOT / "data/raw/uniprot/P13569.fasta"
ASSAY_FASTA = ROOT / "data/processed/bihler_2024_assay_reference.fasta"


def canonical_cftr_sequence() -> str:
    """Canonical human CFTR protein sequence, UniProt P13569."""
    sequence = str(SeqIO.read(UNIPROT_FASTA, "fasta").seq)

    if len(sequence) != 1480 or sequence[469] != "V":
        raise ValueError(
            "Unexpected UniProt CFTR reference. "
            "Expected 1480 amino acids with V at position 470."
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
