"""Original Phase29A fixed-design model parity, using synthetic arrays only.

All reference fits below independently restate the historical Phase29A code
rather than delegating to the refactored compatibility wrappers. No protected
VTI outcome, frozen data or historical experiment files are loaded.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from cftr_theratype.models import response_capacity as canonical
from cftr_theratype.models.auxiliary import SPEC
from cftr_theratype.models.regressors import phase26d_model
from cftr_theratype.evaluation import response_capacity as evaluation


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase29a_extreme_response_capacity as legacy  # noqa: E402
import phase26h_auxiliary_corrector as old_h  # noqa: E402


def historical_ref_fit(config, x_train, y_train, x_test):
    """Frozen original Phase29A implementation, copied independently."""
    if config == "extra_trees":
        model = ExtraTreesRegressor(
            n_estimators=250, max_features=0.8,
            min_samples_leaf=2, max_depth=None, bootstrap=False,
            n_jobs=4, random_state=29,
        )
    else:
        model = old_h.previous.model(old_h.SPEC)
        if config in ("higher_capacity", "capacity_balanced"):
            model = model.set_params(
                max_depth=5, min_child_weight=1,
                n_estimators=350, learning_rate=0.04,
                reg_lambda=1.0, subsample=0.9,
                colsample_bytree=0.9, n_jobs=4, random_state=29,
            )

    w = None
    if config in ("response_balanced", "capacity_balanced"):
        groups = np.where(y_train < 10, "low",
                          np.where(y_train > 40, "high", "middle"))
        counts = {name: int((groups == name).sum())
                  for name in ("low", "middle", "high")}
        w = np.asarray([
            len(y_train) / (3 * counts[group]) for group in groups
        ], dtype=float)

    if config == "extra_trees":
        imputer = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True,
        )
        x_train = imputer.fit_transform(x_train)
        x_test = imputer.transform(x_test)

    if w is None:
        model.fit(x_train, y_train)
    else:
        model.fit(x_train, y_train, sample_weight=w)
    return (
        np.asarray(model.predict(x_train), float).reshape(-1),
        np.asarray(model.predict(x_test), float).reshape(-1),
    )


def test_module_migration_has_no_phase_script_imports_or_repository_io():
    tree = ast.parse(
        (ROOT / "scripts/phase29a_extreme_response_capacity.py")
        .read_text(encoding="utf-8")
    )
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert not any(module.startswith("phase") for module in modules)
    assert "cftr_theratype.datasets.eti_auxiliary" in modules

    for relative in (
        "src/cftr_theratype/models/response_capacity.py",
        "src/cftr_theratype/evaluation/response_capacity.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "pd.read_parquet" not in source
        assert "pd.read_csv" not in source
        assert "phase26h_auxiliary_corrector" not in source


def test_prespecified_models_and_original_phase26h_factory_are_identical():
    assert canonical.SPEC == old_h.SPEC == SPEC == (2, 5.0, 180)
    assert canonical.CONFIGS == legacy.CONFIGS == [
        "global", "higher_capacity", "response_balanced",
        "capacity_balanced", "extra_trees",
    ]

    sample_y = np.asarray([2, 5, 9, 14, 20, 35, 50, 65], float)
    for config in canonical.CONFIGS:
        model = canonical.make_model(config)
        historical = legacy.make_model(old_h, config)
        assert type(model) is type(historical)
        assert model.get_params(deep=True) == historical.get_params(deep=True)
        if config == "extra_trees":
            assert model.n_estimators == 250
            assert model.random_state == 29
        elif config in ("higher_capacity", "capacity_balanced"):
            assert model.max_depth == 5
            assert model.n_estimators == 350
            assert model.random_state == 29
        else:
            assert model.get_params() == phase26d_model(SPEC).get_params()
    w, counts = canonical.weights_by_train_regime(sample_y)
    assert counts == {"low": 3, "middle": 3, "high": 2}
    np.testing.assert_array_equal(w, legacy.weights_by_train_regime(sample_y)[0])
    assert w.mean() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="missing regime"):
        canonical.weights_by_train_regime(np.asarray([10., 15., 41.]))


def test_all_five_models_match_independent_original_predictions():
    rng = np.random.default_rng(2911)
    x = rng.normal(size=(45, 81))
    x[0, 70] = np.nan
    x[11, 75] = np.nan
    y = np.tile([5., 21., 52.], 12) + rng.normal(0, 0.8, 36)
    x_train, x_test = x[:36], x[36:]
    for config in canonical.CONFIGS:
        original = historical_ref_fit(config, x_train.copy(), y, x_test.copy())
        current = canonical.fit_predict(config, x_train.copy(), y, x_test.copy())
        wrapper = legacy.fit_predict(old_h, config, x_train.copy(), y, x_test.copy())
        for original_arr, current_arr, wrapped_arr in zip(original, current, wrapper):
            np.testing.assert_array_equal(original_arr, current_arr)
            np.testing.assert_array_equal(original_arr, wrapped_arr)
        assert current[0].shape == (36,) and current[1].shape == (9,)


def test_position_mae_cluster_bootstrap_and_legacy_wrappers():
    frame = pd.DataFrame({
        "repeat": [1, 1, 2, 2],
        "protein_position": [1, 2, 1, 2],
        "truth": [0., 10., 0., 10.],
        "old": [5., 5., 5., 5.],
        "new": [0., 10., 0., 10.],
    })
    old = evaluation.gain_bootstrap(
        frame, "new", "old", "truth", 100, 29,
    )
    assert old == legacy.gain_bootstrap(
        frame, "new", "old", "truth", 100, 29,
    )
    assert old["mae_gain_pp"] == pytest.approx(5.0)
    assert old["positions"] == 2
    assert evaluation.pos_mae(frame, "truth", "new") == 0
    assert legacy.pos_mae(frame, "truth", "old") == pytest.approx(5.0)
    with pytest.raises(ValueError, match="Expected 210 variants"):
        evaluation.averaged_one_variant(frame)
