import numpy as np
import pandas as pd
import pytest
from cftr_theratype.response_threshold import (
    validate_and_average, position_weights, classification_metrics, paired_ba_gain_bootstrap,
)


def mock():
    keys = [f"c.{x}" for x in range(12)]
    positions = [1, 1] + list(range(2, 12))
    labels = [4., 5., 7., 9., 9.9, 10., 11., 12., 18., 25., 35., 40.]
    membership = pd.DataFrame({"variant_key": keys + ["c.protected"],
        "protein_position": positions + [999],
        "study_cohort": ["ETI, VTI"] * 6 + ["VTI"] * 6 + ["VTI"],
        "transfer_split": ["train"] * 12 + ["test"]})
    rows = []
    for repeat in [1, 2]:
        for key, p, y in zip(keys, positions, labels):
            fold = p % 5 + 1
            rows.append({"repeat": repeat, "fold": fold, "variant_key": key,
                "protein_position": p, "observed_pp": y,
                "predicted_pp": y + (2 if repeat == 1 else -2), "model": "vti_forest"})
            rows.append({"repeat": repeat, "fold": fold, "variant_key": key,
                "protein_position": p, "observed_pp": y,
                "predicted_pp": 18., "model": "median"})
    return pd.DataFrame(rows), membership


def test_repeats_mean_and_threshold_and_sealed_exclusion():
    oof, m = mock()
    data = validate_and_average(oof, m, expected_variants=12, expected_positions=11)
    assert len(data) == 12
    assert "c.protected" not in data.variant_key.values
    assert data.loc[data.variant_key.eq("c.5"), "vti_forest"].iloc[0] == 10.
    assert bool(data.loc[data.variant_key.eq("c.5"), "responder"].iloc[0])
    assert data["near_threshold_observed_5pp"].sum() >= 3


def test_protected_outcome_row_rejected():
    oof, m = mock()
    extra = oof.iloc[0].copy()
    extra.variant_key = "c.protected"
    extra.protein_position = 999
    with pytest.raises(ValueError, match="missing/extra"):
        validate_and_average(pd.concat([oof, pd.DataFrame([extra])], ignore_index=True), m,
                             expected_variants=12, expected_positions=11)


def test_metadata_cannot_contain_assay_labels():
    oof, m = mock()
    m["vti_rescue_pp"] = 2.0
    with pytest.raises(ValueError, match="outcome columns"):
        validate_and_average(oof, m, expected_variants=12, expected_positions=11)


def test_repeat_and_position_integrity():
    oof, m = mock()
    with pytest.raises(ValueError, match="Duplicated"):
        validate_and_average(pd.concat([oof, oof.iloc[[0]]]), m,
                             expected_variants=12, expected_positions=11)
    bad = oof.copy()
    bad.loc[bad.variant_key.eq("c.1") & bad.repeat.eq(2), "observed_pp"] = 70.
    with pytest.raises(ValueError, match="Observed label changed"):
        validate_and_average(bad, m, expected_variants=12, expected_positions=11)


def test_position_balance_and_always_responder_baseline():
    oof, m = mock()
    data = validate_and_average(oof, m, expected_variants=12, expected_positions=11)
    w = position_weights(data)
    assert np.isclose(w.sum(), 11.)
    assert np.allclose(w[data.protein_position.eq(1)], 0.5)
    always = classification_metrics(data, None)
    assert always["balanced_accuracy"] == 0.5
    assert always["specificity_nonresponder"] == 0.
    assert always["sensitivity_responder"] == 1.
    forest = classification_metrics(data, data.vti_forest.to_numpy())
    assert forest["balanced_accuracy"] == 1.
    assert forest["variant_fp"] == 0
    assert forest["variant_fn"] == 0
    assert forest["auroc_continuous_rescue_score"] == 1.
    assert forest["auprc_nonresponder"] == 1.


def test_position_cluster_bootstrap_reproducible():
    oof, m = mock()
    data = validate_and_average(oof, m, expected_variants=12, expected_positions=11)
    a = paired_ba_gain_bootstrap(data, data.vti_forest.to_numpy(), n_boot=80, seed=42)
    b = paired_ba_gain_bootstrap(data, data.vti_forest.to_numpy(), n_boot=80, seed=42)
    assert a == b
    assert np.isclose(a["gain"], .5)
    assert a["ci95"][0] >= .49


def test_missing_median_model_rejected():
    oof, m = mock()
    with pytest.raises(ValueError, match="median"):
        validate_and_average(oof.loc[oof.model.eq("vti_forest")], m,
                             expected_variants=12, expected_positions=11)
