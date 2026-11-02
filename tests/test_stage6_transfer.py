"""Synthetic exact Phase26H forward prediction parity on one held-out fold."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from cftr_theratype.experiments import auxiliary_transfer as canonical
import phase26h_auxiliary_corrector as historical


def synthetic():
    r = np.random.default_rng(321)
    pos = np.repeat(np.arange(18), 2)
    keys = np.array([f'C{i}' for i in range(len(pos))])
    v = pd.DataFrame({'variant_key': keys, 'protein_position': pos,
                      'corrector_rescue_pp': r.normal(22, 8, len(pos)),
                      'vti_rescue_pp': r.normal(32, 12, len(pos)),
                      'study_cohort': ['VTI'] * len(pos)})
    ep = np.repeat(np.arange(30, 62), 2)
    e = pd.DataFrame({'protein_position': ep,
                      'corrector_rescue_pp': r.normal(25, 8, len(ep))})
    X = r.normal(size=(len(v), 80))
    E = r.normal(size=(len(e), 80))
    X[0, 72] = np.nan
    E[0, 72] = np.nan
    test = set(keys[pos <= 3])
    folds = [(1, 1, set(keys) - test, test)]
    return v, X, e, E, folds


def test_onefold_exact_negative_control_parity(monkeypatch):
    v, X, e, E, folds = synthetic()
    # Inner selection is independently covered by the Stage2 parity suite.
    # Fix the weight to speed up the full old-vs-new XGBoost forward parity.
    def fixed_weight(*args, **kwargs):
        return (0.25, 0.0)
    monkeypatch.setattr(historical, 'choose_weight', fixed_weight)
    monkeypatch.setattr(canonical, 'choose_weight', fixed_weight)
    a, a_choice = historical.benchmark(v, X, e, E, folds, 1)
    b, b_choice = canonical.evaluate(v, X, e, E, folds, shuffles=1, max_folds=1)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(a_choice, b_choice, check_exact=True)


def test_transfer_compare_key_order_and_overwrite(tmp_path):
    oof = pd.DataFrame({'repeat': [1, 1], 'fold': [1, 1], 'variant_key': ['a','b'],
                       'protein_position': [1,2], 'observed_corrector': [12., 16.],
                       'observed_total': [15., 20.], 'joint_real': [13.,14.]})
    new, old = tmp_path / 'new.csv', tmp_path / 'old.csv'
    oof.to_csv(new, index=False)
    oof.iloc[::-1].to_csv(old, index=False)
    canonical.compare_oof(new, old, full=False)
    changed = oof.copy(); changed.loc[0, 'joint_real'] += 1
    changed.to_csv(old, index=False)
    with pytest.raises(ValueError, match='PARITY FAIL'):
        canonical.compare_oof(new, old, full=False)
    s = pd.DataFrame({'repeat':[1], 'model':['joint_real']})
    canonical.save_results(oof, s, tmp_path, outdir=tmp_path / 'out')
    with pytest.raises(FileExistsError):
        canonical.save_results(oof, s, tmp_path, outdir=tmp_path / 'out')
