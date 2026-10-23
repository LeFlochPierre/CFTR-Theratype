import numpy as np
import pandas as pd
import pytest

from cftr_theratype.maturation_prediction import (
    paired_multitask_predictions, evaluate_position_predictions, _validate_reference,
    MODELS,
)


def test_auxiliary_train_only_predicts_two_columns():
    x = np.random.default_rng(1).normal(size=(36, 5))
    r, b = x[:, 0] * 4 + 8, x[:, 0] * 8 + 12
    one = paired_multitask_predictions(x[:30], r[:30], b[:30], x[30:], seed=13)
    same = paired_multitask_predictions(x[:30], r[:30], b[:30], x[30:], seed=13)
    assert one.shape == (6, 2)
    assert np.isfinite(one).all()
    assert np.allclose(one, same, rtol=1e-12, atol=1e-10)


def test_permutation_is_deterministic_and_changes_training():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 4))
    r, b = 8 * x[:, 0] + x[:, 1], 15 * x[:, 0] - x[:, 2]
    true = paired_multitask_predictions(x[:40], r[:40], b[:40], x[40:], seed=12)
    null1 = paired_multitask_predictions(x[:40], r[:40], b[:40], x[40:], seed=12, shuffled_aux=True)
    null2 = paired_multitask_predictions(x[:40], r[:40], b[:40], x[40:], seed=12, shuffled_aux=True)
    assert np.allclose(null1, null2, rtol=1e-12, atol=1e-10)
    assert not np.allclose(true, null1)


def test_bad_training_and_nonfinite_rejected():
    x = np.ones((18, 3))
    with pytest.raises(ValueError):
        paired_multitask_predictions(x, np.zeros(18), np.zeros(17), x[:2], seed=1)
    with pytest.raises(ValueError, match="Non-finite"):
        paired_multitask_predictions(x, np.r_[np.zeros(17), np.nan], np.zeros(18), x[:2], seed=1)


def _mock_oof():
    data = []
    for repeat in (1, 2):
        for i in range(10):
            label = 20.0 + i
            for model in MODELS:
                delta = {"vti_forest": 4., "rescue_plus_band_c": 2., "rescue_plus_shuffled_band_c": 3.}[model]
                data.append({"repeat": repeat, "fold": i % 5 + 1, "model": model,
                             "variant_key": f"c.{i}", "protein_position": i+1,
                             "study_cohort": "ETI, VTI" if i<5 else "VTI",
                             "observed_pp": label, "predicted_pp": label+delta})
    return pd.DataFrame(data)


def test_paired_position_metrics_all_and_both_cohorts():
    summary, per_variant = evaluate_position_predictions(_mock_oof(), n_boot=100, seed=4)
    assert len(summary) == 9
    assert len(per_variant) == 30
    selected = summary[(summary.cohort == "all") & (summary.model == "rescue_plus_band_c")].iloc[0]
    assert selected.gain_vs_forest_pp == 2.
    assert selected.gain_vs_shuffled_aux_pp == 1.


def test_duplicate_and_missing_repeats_rejected():
    frame = _mock_oof()
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_position_predictions(pd.concat([frame, frame.iloc[[0]]]))
    with pytest.raises(ValueError, match="Incomplete"):
        evaluate_position_predictions(frame.drop(index=0))


def test_validate_reference_checks_identity_and_label():
    train = pd.DataFrame({"variant_key": ["a", "b"], "protein_position": [3, 4], "vti_rescue_pp": [8., 9.]})
    ref = pd.DataFrame([{"repeat": i, "fold": j, "model": "vti_forest", "variant_key": k,
                         "protein_position": pos, "observed_pp": y, "predicted_pp": y + 1}
                        for i, j, k, pos, y in [(1, 1, "a", 3, 8.), (1, 2, "b", 4, 9.),
                                                  (2, 1, "a", 3, 8.), (2, 2, "b", 4, 9.)]])
    with pytest.raises(ValueError, match="10 frozen CV folds"):
        _validate_reference(ref, train)
    ref.loc[0, "observed_pp"] = 99.
    with pytest.raises(ValueError, match="10 frozen CV folds"):
        _validate_reference(ref, train)
