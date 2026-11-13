"""Synthetic-only parity guards for the frozen Phase26H/26D model specification.

These checks verify retained canonical factory compatibility without accessing
VTI TRAIN, VAL, or TEST outcome data.
"""
from __future__ import annotations


import numpy as np

from cftr_theratype.models.auxiliary import SPEC
from cftr_theratype.models.regressors import phase26d_model


from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_xgb_factory_matches_original_phase26h_compatibility():
    # The compatibility object from the older script now delegates to the
    # same canonical constructor; assert the complete fixed XGB parameters.
    import sys
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import phase26h_auxiliary_corrector as historical

    assert historical.SPEC == SPEC == (2, 5.0, 180)
    old = historical.previous.model(historical.SPEC)
    current = phase26d_model(SPEC)
    assert type(old) is type(current)
    assert old.get_params(deep=True) == current.get_params(deep=True)
    assert current.max_depth == 2
    assert current.n_estimators == 180
    assert current.min_child_weight == 3
    assert current.random_state == 26


def test_fixed_factory_predicts_exactly_on_synthetic_samples():
    rng = np.random.default_rng(2810)
    features = rng.normal(size=(32, 81))
    target = 20 + 3 * features[:, 0] - features[:, 1]
    first = phase26d_model(SPEC)
    second = phase26d_model(SPEC)
    first.fit(features[:26], target[:26])
    second.fit(features[:26], target[:26])
    np.testing.assert_array_equal(
        first.predict(features[26:]),
        second.predict(features[26:]),
    )
