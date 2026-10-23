"""Leakage and aggregation tests for train-only retrospective controls."""

import numpy as np
import pandas as pd
import pytest

from cftr_theratype.cohort_shift import fold_diagnostic_controls, _position_metrics


def make_toy():
    # Two cohorts, eight distinct positions, two folds, two repeats.
    rows = []
    for idx in range(8):
        rows.append({
            "variant_key": f"c.{idx+1}A>G",
            "protein_position": idx + 1,
            "study_cohort": "ETI, VTI" if idx < 4 else "VTI",
            "baseline_fsk_pct_wt": float(idx * 5),
            "iva_treated_pct_wt": float(idx * 6),
            "vti_rescue_pp": float(10 + (idx >= 4) * 50 + idx),
        })
    train = pd.DataFrame(rows)
    oof = []
    for repeat in (1, 2):
        for fold in (1, 2):
            held = train.loc[train.protein_position.mod(2).eq(fold % 2)]
            for model in ("median", "vti_forest"):
                for row in held.itertuples(index=False):
                    oof.append({"repeat": repeat, "fold": fold,
                                "model": model, "variant_key": row.variant_key,
                                "protein_position": row.protein_position,
                                "observed_pp": row.vti_rescue_pp,
                                "predicted_pp": 30.0})
    return train, pd.DataFrame(oof)


def test_control_output_uses_every_variant_and_repeat():
    train, raw = make_toy()
    out = fold_diagnostic_controls(train, raw)
    assert set(out.model) == {
        "median", "vti_forest", "cohort_median_oracle",
        "measured_fsk_ridge_oracle", "measured_fsk_iva_ridge_oracle",
    }
    assert len(out) == 8 * 2 * 5
    assert not out.duplicated(["repeat", "model", "variant_key"]).any()
    metrics, per_variant = _position_metrics(out, draws=200)
    assert len(metrics) == 5
    assert len(per_variant) == 8 * 5
    assert all(np.isfinite(x["position_mae_pp"]) for x in metrics)


def test_held_position_target_change_cannot_change_its_prediction():
    train, raw = make_toy()
    first = fold_diagnostic_controls(train, raw)
    # Change the target for one held-out position, while keeping fold alignment.
    # Because that position is excluded from fitting its own OOF estimate,
    # its oracle predictions must not change.
    target_key = train.iloc[0].variant_key
    changed = train.copy()
    changed.loc[changed.variant_key == target_key, "vti_rescue_pp"] += 400
    changed_oof = raw.copy()
    changed_oof.loc[changed_oof.variant_key == target_key, "observed_pp"] += 400
    second = fold_diagnostic_controls(changed, changed_oof)
    for name in ("cohort_median_oracle", "measured_fsk_ridge_oracle", "measured_fsk_iva_ridge_oracle"):
        a = first.loc[(first.variant_key == target_key) & (first.model == name), "predicted_pp"].to_numpy()
        b = second.loc[(second.variant_key == target_key) & (second.model == name), "predicted_pp"].to_numpy()
        np.testing.assert_allclose(a, b)


def test_reject_duplicate_out_of_fold_rows():
    train, raw = make_toy()
    with pytest.raises(ValueError, match="Duplicated out-of-fold"):
        fold_diagnostic_controls(train, pd.concat([raw, raw.head(1)], ignore_index=True))


def test_reject_target_or_position_mismatch():
    train, raw = make_toy()
    corrupted = raw.copy()
    corrupted.loc[0, "observed_pp"] += 10
    with pytest.raises(ValueError, match="OOF targets disagree"):
        fold_diagnostic_controls(train, corrupted)
    corrupted = raw.copy()
    corrupted.loc[0, "protein_position"] = 999
    with pytest.raises(ValueError, match="OOF positions disagree"):
        fold_diagnostic_controls(train, corrupted)
