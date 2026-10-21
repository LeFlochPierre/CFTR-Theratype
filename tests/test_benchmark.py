"""Tests for reproducible CFTR baseline features and metrics."""

import numpy as np
import pandas as pd
import pytest

from cftr_theratype.reference import bihler_2024_assay_sequence
from cftr_theratype.benchmark import (
    make_features,
    position_mae_and_ci,
)


def test_m470v_feature_generation():
    frame = pd.DataFrame([{
        "protein_position": 470,
        "reference_aa": "M",
        "alternate_aa": "V",
    }])

    X, names = make_features(
        frame, bihler_2024_assay_sequence()
    )

    assert X.shape == (1, len(names))
    assert np.isfinite(X).all()
    assert X[0, names.index("wt_M")] == 1
    assert X[0, names.index("alt_V")] == 1


def test_experimental_targets_cannot_change_features():
    frame = pd.DataFrame([{
        "protein_position": 470,
        "reference_aa": "M",
        "alternate_aa": "V",
        "eti_rescue_pp": 5.0,
    }])

    sequence = bihler_2024_assay_sequence()

    first, _ = make_features(frame, sequence)

    frame["eti_rescue_pp"] = 500.0
    second, _ = make_features(frame, sequence)

    np.testing.assert_array_equal(first, second)


def test_invalid_reference_rejected():
    frame = pd.DataFrame([{
        "protein_position": 470,
        "reference_aa": "V",
        "alternate_aa": "M",
    }])

    with pytest.raises(ValueError):
        make_features(
            frame, bihler_2024_assay_sequence()
        )


def test_position_weighting():
    y = np.array([0.0, 10.0, 2.0])
    predictions = np.zeros(3)
    positions = np.array([100, 100, 200])

    # Position 100: mean absolute error 5
    # Position 200: mean absolute error 2
    # Equally weighted position MAE = 3.5
    score, interval = position_mae_and_ci(
        y, predictions, positions
    )

    assert np.isclose(score, 3.5)
    assert len(interval) == 2
