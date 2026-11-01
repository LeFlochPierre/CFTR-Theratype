"""Source-independent synthetic tests for canonical feature benchmark / parity guard."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cftr_theratype.experiments import feature_benchmark as b


def synthetic():
    rng = np.random.default_rng(321)
    positions = np.repeat(np.arange(20), 2)
    keys = np.array([f"TEST{i}" for i in range(40)])
    d = pd.DataFrame({"variant_key": keys, "protein_position": positions,
                      "study_cohort": ["development"] * 40,
                      "corrector_rescue_pp": rng.normal(20, 4, 40),
                      "vti_rescue_pp": rng.normal(36, 8, 40)})
    x = rng.normal(size=(40, 80))
    x[::4, 72] = np.nan
    features = {"sequence": x[:, :68],
                "sequence_structure": x,
                "sequence_structure_thermompnn": np.column_stack([x, rng.normal(size=(40, 2))])}
    test = set(keys[positions == 0]) | set(keys[positions == 1])
    folds = [(1, 1, set(keys) - test, test)]
    return d, features, folds


def test_fast_one_fold(monkeypatch):
    monkeypatch.setattr(b, "SETUPS", [
        ("ridge_80", "sequence_structure", "ridge", None),
        ("xgb_80", "sequence_structure", "xgb", None),
    ])
    d, features, folds = synthetic()
    output = b.evaluate(d, features, folds, target="total", fast=True, max_folds=1,
                        bootstraps=10)
    oof, chosen, metrics, comparisons = output
    assert len(oof) == 4 and len(chosen) == 2
    assert set(oof.protein_position) == {0, 1}
    assert set(oof.variant_key) == set(folds[0][3])
    assert np.isfinite(oof[["ridge_80", "xgb_80"]].to_numpy(float)).all()
    assert len(metrics) == 2 and len(comparisons) == 1


def test_compare_key_alignment_and_mismatch(tmp_path):
    d = pd.DataFrame({
        "repeat": [1, 1, 2, 2], "fold": [1, 2, 1, 2],
        "variant_key": ["a", "b", "a", "b"],
        "protein_position": [1, 2, 1, 2],
        "observed": [30., 40., 30., 40.],
        "study_cohort": ["x"] * 4,
        "xgb_80": [31., 42., 31., 42.],
        "ridge_80": [32., 43., 32., 43.],
    })
    p1, p2 = tmp_path / "new.csv", tmp_path / "old.csv"
    d.to_csv(p1, index=False)
    d.iloc[::-1].to_csv(p2, index=False)
    result = b.compare_oof(p1, p2, require_full=False)
    assert result["rows"] == 4
    changed = d.copy()
    changed.loc[0, "xgb_80"] += 1
    changed.to_csv(p2, index=False)
    with pytest.raises(ValueError, match="PARITY FAIL xgb_80"):
        b.compare_oof(p1, p2, require_full=False)


def test_save_no_overwrite(tmp_path):
    frames = tuple(pd.DataFrame({"x": [1]}) for _ in range(4))
    b.save_results(frames, "total", tmp_path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        b.save_results(frames, "total", tmp_path)
