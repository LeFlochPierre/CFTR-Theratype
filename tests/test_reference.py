"""Tests for reference-sequence and variant integrity."""

import pandas as pd

from cftr_theratype.reference import (
    canonical_cftr_sequence,
    bihler_2024_assay_sequence,
)

from cftr_theratype.splits import parse_substitution


def test_bihler_reference_background():
    canonical = canonical_cftr_sequence()
    assay = bihler_2024_assay_sequence()

    assert len(assay) == 1480
    assert canonical[469] == "V"
    assert assay[469] == "M"

    differences = [
        i + 1
        for i, (a, b) in enumerate(zip(canonical, assay))
        if a != b
    ]

    assert differences == [470]


def test_m470v_valid_against_experimental_reference():
    row = pd.Series({
        "variant_type": "Missense",
        "hgvs_protein": "p.Met470Val",
        "variant": "M470V",
    })

    result = parse_substitution(
        row,
        bihler_2024_assay_sequence(),
    )

    assert result == (470, "M", "V", "valid")
