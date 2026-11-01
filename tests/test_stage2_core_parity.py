"""Stage 2 pure-function parity against the five original phase scripts.

Run: python -m pytest -q tests/test_stage2_core_parity.py
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import phase26d_corrector_bottleneck as d
import phase26g_feature_model_ablation as g
import phase26h_auxiliary_corrector as h
import phase27d1_biology_ablation as bio
from cftr_theratype.evaluation.metrics import position_mae, paired_ci
from cftr_theratype.models.regressors import (
    phase26d_model, phase26d_select_spec, phase26g_model, phase26g_choose,
)
from cftr_theratype.models.auxiliary import positional_shuffle
from cftr_theratype.features.biology_annotations import annotation_features
from cftr_theratype.datasets.eti_auxiliary import parsed_position


def synthetic():
    rng = np.random.default_rng(20261101)
    X = rng.normal(size=(36, 80))
    X[::4, 70] = np.nan
    y = rng.normal(size=36) * 15 + 30
    groups = np.repeat(np.arange(12), 3)
    return X, y, groups


def test_metrics_identical():
    X, y, groups = synthetic()
    pred = y + np.arange(len(y)) * 0.1
    expected = g.position_mae(y, pred, groups)
    assert position_mae(y, pred, groups) == expected
    assert position_mae(y, pred, groups) == d.pos_mae(y, pred, groups)
    frame = pd.DataFrame({
        'repeat': np.tile([1, 2], 12),
        'protein_position': np.repeat(np.arange(6), 4),
        'observed': y[:24],
        'baseline': pred[:24],
        'candidate': (pred[:24] + .01),
    })
    a = g.paired_ci(frame, 'candidate', 'baseline', n=100)
    b = paired_ci(frame, 'candidate', 'baseline', n=100)
    assert a == b


def test_model_factories_equivalent():
    for spec in d.GRID:
        assert phase26d_model(spec).get_params() == d.model(spec).get_params()
    for kind, values in g.PARAMS.items():
        for setting in values:
            for topk in ([None, 16] if kind == 'xgb' else [None]):
                original = g.make_model(kind, setting, topk)
                canonical = phase26g_model(kind, setting, topk)
                # Same estimator classes, pipeline preprocessing and params.
                assert repr(original) == repr(canonical)



def test_fitted_model_prediction_parity():
    X, y, groups = synthetic()
    # Ensure the shared specification reproduces fitted predictions, not just
    # sklearn.get_params() or pipeline repr.
    for original, new in [
        (d.model((2, 5.0, 180)), phase26d_model((2, 5.0, 180))),
        (g.make_model('xgb', (2, 5.0, 180)), phase26g_model('xgb', (2, 5.0, 180))),
    ]:
        original.fit(X[:25], y[:25])
        new.fit(X[:25], y[:25])
        np.testing.assert_allclose(original.predict(X[25:]), new.predict(X[25:]), atol=0, rtol=0)

    from cftr_theratype.models.auxiliary import fit_joint
    original_joint = h.fit_joint(X[:12], y[:12], X[12:24], y[12:24], 0.5)
    canonical_joint = fit_joint(X[:12], y[:12], X[12:24], y[12:24], 0.5)
    np.testing.assert_allclose(original_joint.predict(X[24:]), canonical_joint.predict(X[24:]), atol=0, rtol=0)


def test_grouped_selection_parity():
    X, y, groups = synthetic()
    # Fixed 'quick' path keeps the synthetic test lightweight.
    legacy_spec, legacy_loss = d.select_spec(X, y, groups, quick=True)
    new_spec, new_loss = phase26d_select_spec(X, y, groups, quick=True)
    assert legacy_spec == new_spec and np.isnan(legacy_loss) and np.isnan(new_loss)
    aa = g.choose(X, y, groups, 'ridge', None, fast=True)
    bb = phase26g_choose(X, y, groups, 'ridge', None, fast=True)
    assert aa == bb


def test_auxiliary_shuffle_parity():
    _, y, groups = synthetic()
    np.testing.assert_array_equal(h.positional_shuffle(y, groups, 27),
                                  positional_shuffle(y, groups, 27))


def test_biology_features_parity():
    positions = np.array([508, 551, 1060, 1282])
    codes = ['F508del', 'G551D', 'K1060T', 'W1282R']
    # Phase27D1 only accepts substitutions; skip the F508del example.
    positions = positions[1:]
    codes = codes[1:]
    a = bio.annotation_features(positions, codes)
    b = annotation_features(positions, codes)
    assert set(a) == set(b)
    for key in a:
        pd.testing.assert_frame_equal(a[key], b[key], check_exact=True)


def test_parsed_positions():
    for label in ['p.Gly551Asp', 'G551D', 'p.Trp1282Ter']:
        assert parsed_position(label) == h.parsed_position(label)
