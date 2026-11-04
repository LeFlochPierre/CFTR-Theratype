"""Synthetic-only parity guards for Phase28 and Phase30 dependency cleanup.

These checks never load VTI TRAIN, VAL, or TEST data. Historical reference
products remain untouched; real-data parity must be checked separately.
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

from cftr_theratype.models.auxiliary import SPEC
from cftr_theratype.models.regressors import phase26d_model


ROOT = Path(__file__).resolve().parents[1]


def _module_imports(path: str) -> list[str]:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return imports


def test_phase28_30_no_longer_import_phase26h_or_phase27d1():
    for path in (
        "scripts/phase28c_structure_biophysics.py",
        "scripts/phase28e5_foldx_predictive_ablation.py",
        "scripts/phase30a_target_information_ladder.py",
        "scripts/phase30c_eti_incremental_ladder.py",
    ):
        modules = _module_imports(path)
        assert "phase26h_auxiliary_corrector" not in modules, path
        assert "phase27d1_biology_ablation" not in modules, path
        tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
        # Named legacy aliases should not survive as runtime calls.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute):
                continue
            if isinstance(node.value, ast.Name):
                assert not (
                    node.value.id == "h" and node.attr in ("load_data", "previous", "SPEC")
                ), path
        assert "cftr_theratype.datasets.eti_auxiliary" in modules, path
        assert "cftr_theratype.models.regressors" in modules, path
        assert "cftr_theratype.models.auxiliary" in modules, path

    for path in (
        "scripts/phase28e5_foldx_predictive_ablation.py",
        "scripts/phase30a_target_information_ladder.py",
        "scripts/phase30c_eti_incremental_ladder.py",
    ):
        modules = _module_imports(path)
        assert not any(name.startswith("phase") for name in modules), (path, modules)


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
