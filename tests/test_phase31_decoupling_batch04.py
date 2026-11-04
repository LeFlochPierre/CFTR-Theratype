"""Batch04 structural + synthetic parity checks (no clinical data)."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cftr_theratype.experiments import three_tracks_inputs as shared


ROOT = Path(__file__).resolve().parents[1]


def test_three_tracks_frozen_train_alignment(tmp_path, monkeypatch):
    rows = []
    for i, pos in enumerate(range(50, 218)):
        for alt in ("V", "G") if i < 42 else ("V",):
            rows.append({
                "variant_key": f"v{pos}{alt}",
                "protein_position": pos,
                "transfer_split": "train",
                "hgvs_protein": f"p.Ala{pos}{'Val' if alt == 'V' else 'Gly'}",
            })
    meta = pd.DataFrame(rows)
    base = meta[["variant_key", "protein_position", "transfer_split"]].copy()
    x = np.zeros((210, 80), float)
    assert len(base) == 210 and base.protein_position.nunique() == 168

    def fake_load(root):
        assert Path(root) == tmp_path
        return base, {"sequence_structure": x}, pd.DataFrame(), ["synthetic-fold"], "synthetic"

    def fake_parquet(path, columns=None, **kwargs):
        assert str(path).endswith("bihler_2026_transfer_splits.parquet")
        assert columns == ["variant_key", "hgvs_protein", "protein_position", "transfer_split"]
        return meta[columns].copy()

    monkeypatch.setattr(shared, "load_vti_train", fake_load)
    monkeypatch.setattr(pd, "read_parquet", fake_parquet)

    actual, mat, folds, annotated = shared.get_data(tmp_path)
    pd.testing.assert_frame_equal(actual, base)
    np.testing.assert_array_equal(mat, x)
    assert folds == ["synthetic-fold"]
    assert annotated.aa31i.iloc[0] == "A50V"
    assert annotated.aa31i.iloc[1] == "A50G"
    assert annotated.pos31i2.nunique() == 168
    assert annotated.variant_key.astype(str).tolist() == base.variant_key.tolist()


def test_esmrank_score_alignment_and_rejection(tmp_path):
    f = tmp_path / "scores.csv"
    pd.DataFrame({"ids": ["p.Ala12Val", "p.Trp1282Gly"],
                  "prediction": [2.5, -3.0]}).to_csv(f, index=False)
    score = shared.load_esm(f)
    assert score.index.tolist() == ["A12V", "W1282G"]
    assert score.loc["A12V"] == pytest.approx(2.5)

    pd.DataFrame({"ids": ["A12V", "p.Ala12Val"],
                  "prediction": [2.5, 3.0]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="unique ESMRank"):
        shared.load_esm(f)

    pd.DataFrame({"ids": ["A12V"], "prediction": ["not numeric"]}).to_csv(f, index=False)
    with pytest.raises(ValueError, match="Nonfinite ESMRank"):
        shared.load_esm(f)


def test_migrated_scripts_have_no_imports_from_numbered_phase_modules():
    for rel in (
        "scripts/phase31k_response_generalisation.py",
        "scripts/phase31m_three_tracks.py",
        "scripts/phase32a_adaptive_theratyping.py",
    ):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
        assert not any(name.startswith("phase")
                       for name in imports), (rel, imports)

    rel = "scripts/phase31i2_repository.py"
    tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    assert not any(name.startswith(("phase26d_", "phase26g_")) for name in names)
    assert not any(name.startswith("phase") for name in names)
