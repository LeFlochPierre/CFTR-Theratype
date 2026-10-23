"""Tests for descriptive and position-safe cohort shift analysis."""
import numpy as np
import pandas as pd

from cftr_theratype.cohort_shift import (
    baseline_bins, common_support_difference, train_only_cohort_classifier,
)


def test_baseline_bins_include_negative_and_boundaries():
    result = baseline_bins(pd.Series([-1.0, 5.0, 20.0, 50.0]))
    assert list(result.astype(str)) == ["<5% WT", "5–<20% WT", "20–<50% WT", "≥50% WT"]


def test_common_support_refuses_one_cohort_only():
    df = pd.DataFrame({
        "baseline_stratum": ["<5% WT"] * 6,
        "study_cohort": ["VTI"] * 6,
        "protein_position": [1, 2, 3, 4, 5, 6],
        "vti_rescue_pp": [1., 2., 3., 4., 5., 6.],
    })
    result = common_support_difference(df, "vti_rescue_pp")
    assert result["status"] == "insufficient_common_support"


def test_common_support_descriptive_weighting():
    df = pd.DataFrame({
        "baseline_stratum": ["<5% WT"] * 10,
        "study_cohort": ["ETI, VTI"] * 5 + ["VTI"] * 5,
        "protein_position": list(range(1, 11)),
        "vti_rescue_pp": [10.] * 5 + [20.] * 5,
    })
    result = common_support_difference(df, "vti_rescue_pp")
    assert result["status"] == "descriptive_only"
    assert np.isclose(result["weighted_difference_pp"], -10.)


def test_cohort_classifier_disjoint_groups_and_finite_predictions():
    # Distinct positions, enough examples from both cohorts for five folds.
    n = 60
    rng = np.random.default_rng(17)
    features = rng.normal(size=(n, 8))
    cohort = pd.Series((["ETI, VTI", "VTI"] * 30))
    positions = pd.Series(np.arange(1, n + 1))
    result = train_only_cohort_classifier(features, cohort, positions)
    assert 0 <= result["roc_auc"] <= 1
    assert sum(x["held_variants"] for x in result["folds"]) == n
