"""Synthetic-only contracts for canonical Phase31 assay helpers.

No frozen VTI assay outcome files or protected partitions are opened.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cftr_theratype.datasets.assay_identifiers import norm_aa, with_ids
from cftr_theratype.features.assay_substitutions import AA, mutation_features
from cftr_theratype.models.assay_teachers import make_teacher

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase31i_auxiliary_transfer as legacy_i  # noqa: E402
import phase31i2_repository as legacy_i2  # noqa: E402


def test_identifier_normalisation_and_legacy_parity():
    for raw, expected in (
        ("p.Ala12Val", "A12V"), ("A12V", "A12V"),
        ("p.Trp1282Gly", "W1282G"), ("p.Ala12Ala", None),
        ("p.Ala1481Val", None), (None, None),
    ):
        assert norm_aa(raw) == expected
        assert legacy_i.norm_aa(raw) == expected


def test_identifier_alignment_and_protected_metadata_only():
    frame = pd.DataFrame({
        "variant_key": ["v1", "v2"],
        "hgvs_protein": ["p.Ala12Val", "p.Trp1282Gly"],
        "protein_position": [12, 1282],
        "transfer_split": ["train", "train"],
    })
    result = with_ids(frame, "synthetic", strict=True)
    pd.testing.assert_frame_equal(result, legacy_i.with_ids(frame, "synthetic", strict=True))
    assert result.aa31i.tolist() == ["A12V", "W1282G"]
    assert result.pos31i2.astype(int).tolist() == [12, 1282]
    assert not result.control31i.any()

    bad = frame.copy()
    bad.loc[1, "protein_position"] = 999
    with pytest.raises(ValueError, match="conflicting positions"):
        with_ids(bad, "synthetic", strict=True)


def test_mutation_features_are_fixed_and_assay_free():
    codes = pd.Series(["A12V", "W1282G"])
    x = mutation_features(codes)
    np.testing.assert_array_equal(x, legacy_i2.mutation_features(codes))
    assert x.shape == (2, 61)
    assert np.isfinite(x).all()
    assert x[0, AA.index("A")] == 1.0
    assert x[0, len(AA) + AA.index("V")] == 1.0
    assert x[0, -6] == pytest.approx(12.0 / 1480.0)
    with pytest.raises(ValueError, match="canonical missense"):
        mutation_features(pd.Series(["p.Gly551Asp"]))
    with pytest.raises(ValueError, match="Invalid mutation"):
        mutation_features(pd.Series(["A12A"]))


def test_teacher_model_is_exact_legacy_constructor():
    a = make_teacher(10.0)
    b = legacy_i2.make_teacher(10.0)
    assert list(a.named_steps) == ["impute", "scale", "ridge"]
    assert list(a.named_steps) == list(b.named_steps)
    for name in a.named_steps:
        assert a.named_steps[name].get_params(deep=False) == b.named_steps[name].get_params(deep=False)
    assert a.named_steps["ridge"].alpha == 10.0


def test_phase31j_no_longer_imports_numbered_scripts_for_pure_helpers():
    tree = ast.parse((ROOT / "scripts/phase31j_error_diagnostics.py").read_text())
    modules = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert "cftr_theratype.datasets.assay_identifiers" in modules
    assert "cftr_theratype.features.assay_substitutions" in modules
    assert "cftr_theratype.models.assay_teachers" in modules
    assert "phase31i_auxiliary_transfer" not in modules
