"""Tests for cohort-weighted, train-only CFTR robustness analysis."""
import numpy as np
import pandas as pd
import pytest
from cftr_theratype.robustness import cohort_weights, position_metrics


def test_balanced_weights_equal_cohort_totals():
    c = ["ETI, VTI"] * 2 + ["VTI"] * 6
    weights = cohort_weights(c, strength=1)
    assert weights.sum() == pytest.approx(8)
    assert weights[:2].sum() == pytest.approx(4)
    assert weights[2:].sum() == pytest.approx(4)


def test_uniform_weights_and_partial_interpolation():
    c = ["ETI, VTI"] * 2 + ["VTI"] * 6
    np.testing.assert_allclose(cohort_weights(c, 0), np.ones(8))
    midway = cohort_weights(c, 0.5)
    assert midway[:2].mean() > midway[2:].mean()
    assert midway[:2].mean() < cohort_weights(c, 1)[:2].mean()


def test_reject_missing_or_unknown_group():
    with pytest.raises(ValueError):
        cohort_weights(["VTI"] * 4, 1)
    with pytest.raises(ValueError):
        cohort_weights(["VTI", "other"], 1)
    with pytest.raises(ValueError):
        cohort_weights(["VTI", "ETI, VTI"], 1.5)


def toy_oof():
    items = []
    for repeat in (1, 2):
        for i in range(8):
            cohort = "ETI, VTI" if i < 4 else "VTI"
            truth = float(i * 2)
            for name, error in (("vti_forest", 5),
                                ("balanced_forest_half", 3),
                                ("balanced_forest_full", 4)):
                items.append({
                    "repeat": repeat, "model": name, "variant_key": f"v{i}",
                    "protein_position": i + 1, "study_cohort": cohort,
                    "observed_pp": truth, "predicted_pp": truth + error,
                })
    return pd.DataFrame(items)


def test_position_metrics_paired_improvement():
    summary, details = position_metrics(toy_oof(), n_boot=100, seed=123)
    all_half = summary.loc[(summary.cohort == "all") &
                           (summary.model == "balanced_forest_half")].iloc[0]
    assert all_half.position_mae_pp == pytest.approx(3)
    assert all_half.gain_vs_forest_pp == pytest.approx(2)
    assert set(summary.cohort) == {"all", "ETI, VTI", "VTI"}
    assert len(details) == 24


def test_position_metrics_reject_duplicates():
    toy = toy_oof()
    with pytest.raises(ValueError, match="Duplicate"):
        position_metrics(pd.concat([toy, toy.head(1)]), n_boot=100)
