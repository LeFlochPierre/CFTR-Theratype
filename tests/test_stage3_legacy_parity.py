"""Synthetic end-to-end Phase26G old-vs-canonical OOF reproducibility test.

No clinical data, no protected outcomes. Uses the real legacy script with
its loader patched to a synthetic dataset and writes only to tmp_path.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import phase26g_feature_model_ablation as legacy
from cftr_theratype.experiments.feature_benchmark import evaluate, compare_oof


def test_phase26g_full_config_synthetic_parity(monkeypatch, tmp_path):
    rng = np.random.default_rng(20261101)
    pos = np.repeat(np.arange(20), 2)
    keys = np.array([f"X{i:03}" for i in range(40)])
    d = pd.DataFrame({
        "variant_key": keys, "protein_position": pos,
        "study_cohort": ["TRAIN"] * 40,
        "corrector_rescue_pp": rng.normal(19, 9, 40),
        "vti_rescue_pp": rng.normal(35, 11, 40),
    })
    X = rng.normal(size=(40, 80))
    X[::5, 71] = np.nan
    features = {"sequence": X[:, :68], "sequence_structure": X,
                "sequence_structure_thermompnn": np.column_stack([X, rng.normal(size=(40, 2))])}
    folds = []
    for repeat in (1, 2):
        for fold in (1, 2):
            # Two partitions, repeated with alternate grouping.
            heldpos = set(range(10)) if fold == 1 else set(range(10, 20))
            if repeat == 2:
                heldpos = {i for i in range(20) if i % 2 == (fold - 1)}
            test = set(keys[np.isin(pos, list(heldpos))])
            folds.append((repeat, fold, set(keys) - test, test))
    monkeypatch.setattr(legacy.prior, "load_train", lambda: (d, features, None, folds, "synthetic"))
    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["phase26g_feature_model_ablation.py", "--target", "total", "--fast", "--bootstraps", "10"])
    legacy.main()
    new = evaluate(d, features, folds, target="total", fast=True, max_folds=4,
                   bootstraps=10)
    out = tmp_path / "new"
    out.mkdir()
    new[0].to_csv(out / "phase26g_total_oof.csv", index=False)
    compare_oof(out / "phase26g_total_oof.csv",
                tmp_path / "results/biology/phase26g/phase26g_total_oof.csv",
                require_full=False, atol=1e-7)
    a = pd.read_csv(tmp_path / "results/biology/phase26g/phase26g_total_selected.csv")
    b = new[1]
    assert a[["repeat", "fold", "model", "selected_param"]].equals(
        b[["repeat", "fold", "model", "selected_param"]])
