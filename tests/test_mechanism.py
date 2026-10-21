"""Unit tests for position-balanced corrector evaluation."""

import numpy as np
import pandas as pd
import pytest

from cftr_theratype.mechanism import per_position_errors, cluster_gains


def test_stratum_and_position_weighting():
    rows = pd.DataFrame([
        {"model": "a", "study_cohort": "VTI", "protein_position": 10,
         "absolute_error_pp": 0.0},
        {"model": "a", "study_cohort": "VTI", "protein_position": 10,
         "absolute_error_pp": 10.0},
        {"model": "a", "study_cohort": "VTI", "protein_position": 20,
         "absolute_error_pp": 2.0},
    ])
    result = per_position_errors(rows)
    assert result.absolute_error_pp.tolist() == [5.0, 2.0]


def test_gain_sign_and_position_bootstrap():
    data = []
    for position in range(10):
        data.extend([
            {"protein_position": position, "model": "median", "absolute_error_pp": 5.0},
            {"protein_position": position, "model": "forest", "absolute_error_pp": 3.0},
        ])
    gain, low, high = cluster_gains(pd.DataFrame(data), "forest", "median")
    assert np.isclose(gain, 2.0)
    assert np.isclose(low, 2.0)
    assert np.isclose(high, 2.0)


def test_missing_models_raise():
    data = pd.DataFrame([
        {"protein_position": 1, "model": "a", "absolute_error_pp": 5.0},
    ])
    with pytest.raises(ValueError):
        cluster_gains(data, "a", "b")
