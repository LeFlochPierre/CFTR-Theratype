"""Unit tests for frozen-ESM grouped scoring utilities."""
import numpy as np
from cftr_theratype.plm import position_mae, bootstrap_gain, make_ridge


def test_position_mae_equal_weighting():
    y = np.array([0.0, 10.0, 2.0])
    pred = np.zeros(3)
    pos = np.array([100, 100, 200])
    assert np.isclose(position_mae(y, pred, pos), 3.5)


def test_paired_bootstrap_direction():
    y = np.array([0.0, 10.0, 2.0, 20.0])
    better = y.copy()
    worse = np.zeros_like(y)
    pos = np.array([100, 100, 200, 300])
    gain, ci = bootstrap_gain(y, better, worse, pos, n_boot=200)
    assert gain > 0
    assert ci[0] > 0


def test_ridge_pipeline_has_preprocessing_inside_fit():
    model = make_ridge(100.0)
    assert model.named_steps["standardscaler"] is not None
    assert model.named_steps["ridge"].alpha == 100.0
