#!/usr/bin/env python3
"""Phase 19I: training-only acute IVA response learning-objective ablation.

Five experiments, same frozen 2 x 5 position-disjoint outer folds:
  1. Unweighted XGBoost control.
  2. Tail-weighted XGBoost (inner-select strength).
  3. High-response oversampling (inner-select repetition factor).
  4. Assay-uncertainty weighted XGBoost (inner-select strength).
  5. Two-head FSK and FSK+IVA prediction, subtract predictions.

All candidate strengths are chosen only with inner training folds. The existing
Phase 19C inner-selected XGBoost hyperparameters are reused per outer fold.
The existing Phase 19C OOF predictions are kept untouched and displayed only as
an independent reference; primary comparisons are against the new control,
which uses exactly the same estimator implementation as experiments 2-5.

Run: python scripts/phase19i_augmentation_ablation.py [--smoke]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from cftr_theratype import model_zoo as mz
from cftr_theratype.transfer import ROOT

TARGET = 'acute_iva_vti_pp'
KEYS = ['repeat', 'fold', 'model', 'variant_key']
SEEDS = 42
MODELS = (
    'control_xgb', 'tail_weighted', 'oversampled', 'reliability_weighted',
    'two_head_fsk_total',
)
STRENGTHS = {
    'tail_weighted': (1.0, 1.5, 2.0, 3.0),
    'oversampled': (1, 2, 3),
    'reliability_weighted': (0.0, 0.5, 1.0, 2.0),
}


def position_mae(y: np.ndarray, p: np.ndarray, position: np.ndarray) -> float:
    frame = pd.DataFrame({'position': position, 'err': np.abs(y - p)})
    return float(frame.groupby('position', sort=False).err.mean().mean())


def core_xgb(wrapper):
    """Extract configured XGBoost component from the existing model-zoo estimator.

    This avoids guessing the project's learning rate, regularisation, or other
    XGBoost settings; the same extracted estimator is used in every arm.
    """
    checked = set()
    pending = [(wrapper, 0)]
    while pending:
        obj, depth = pending.pop(0)
        if id(obj) in checked:
            continue
        checked.add(id(obj))
        if isinstance(obj, XGBRegressor):
            return obj
        if depth >= 4:
            continue
        if hasattr(obj, '__dict__'):
            for value in vars(obj).values():
                if isinstance(value, XGBRegressor) or (
                    hasattr(value, '__dict__')
                    and type(value).__module__.startswith('cftr_theratype')
                ):
                    pending.append((value, depth + 1))
    raise RuntimeError(
        'Cannot locate XGBRegressor within model_zoo.estimator(). '
        'Show the source of FoldRegressor/estimator before adapting the code. '
        'No approximate estimator will be silently substituted.'
    )


def make_xgb(params: dict, seed: int) -> XGBRegressor:
    wrapper = mz.estimator('xgboost', params, seed=seed)
    core = core_xgb(wrapper)
    new = clone(core)
    new.set_params(n_jobs=int(os.environ.get('CFTR_XGB_JOBS', '4')))
    return new


def train_predict(
    Xtr: np.ndarray, Ytr: np.ndarray, Xte: np.ndarray,
    params: dict, seed: int, mode: str, strength: float | int,
    sigma_train: np.ndarray | None = None,
    heads: np.ndarray | None = None,
) -> np.ndarray:
    """Train on training rows ONLY. No validation labels used here."""
    Xtr = np.asarray(Xtr, float)
    Xte = np.asarray(Xte, float)
    y = np.asarray(Ytr, float)
    if not np.isfinite(y).all() or np.isinf(Xtr).any() or np.isinf(Xte).any():
        raise ValueError('Non-finite outcomes or infinite covariates')

    if mode == 'two_head_fsk_total':
        if heads is None or heads.shape != (len(y), 2):
            raise ValueError('Missing two-head outcomes')
        if not np.isfinite(heads).all():
            raise ValueError('Invalid two-head outcomes')
        pred = []
        for index in range(2):
            model = make_xgb(params, seed)
            model.fit(Xtr, heads[:, index])
            pred.append(np.asarray(model.predict(Xte), float))
        # Head 0 = FSK+IVA activity; Head 1 = FSK activity.
        # Their difference predicts acute IVA increment without division.
        return pred[0] - pred[1]

    weights = None
    if mode == 'tail_weighted':
        weights = np.where(y > 20.0, float(strength), 1.0)
    elif mode == 'reliability_weighted':
        if sigma_train is None or not np.isfinite(sigma_train).all():
            raise ValueError('Assay uncertainty missing or non-finite')
        s = np.maximum(np.asarray(sigma_train, float), 0.0)
        # Robust regularization of low SEM: prevent low-SEM rows dominating.
        typical = max(float(np.median(s)), 1.0)
        raw = (typical / np.sqrt(s * s + typical * typical)) ** float(strength)
        weights = np.clip(raw / max(float(np.mean(raw)), 1e-12), 0.5, 2.0)
    elif mode == 'oversampled':
        factor = int(strength)
        if factor < 1:
            raise ValueError('Oversampling factor must be >= 1')
        indices = np.arange(len(y))
        extra = np.repeat(indices[y > 20.0], factor - 1)
        indices = np.concatenate([indices, extra])
        # Repeating rows is NOT new data. Shuffle deterministically.
        rng = np.random.default_rng(seed)
        indices = rng.permutation(indices)
        Xtr, y = Xtr[indices], y[indices]
    elif mode != 'control_xgb':
        raise ValueError(f'Unknown experiment {mode}')

    model = make_xgb(params, seed)
    if weights is None:
        model.fit(Xtr, y)
    else:
        model.fit(Xtr, y, sample_weight=weights)
    return np.asarray(model.predict(Xte), float)


def load_inputs():
    train, mats, _, outer = mz.prepare(ROOT)
    assert len(train) == 210 and train.protein_position.nunique() == 168
    assert train.variant_key.is_unique and len(outer) == 10
    X = np.asarray(mats['sequence_structure'], float)
    assert X.shape[0] == 210 and not np.isinf(X).any()
    # Read only training outcomes from the already audited Phase 19B file.
    label_path = ROOT / 'results/boltz/phase19b/vti_train_acute_iva_targets.csv'
    labels = pd.read_csv(label_path, dtype={'variant_key': str})
    if len(labels) != 210 or labels.variant_key.duplicated().any():
        raise ValueError('Phase19B targets do not match frozen train cardinality')
    assert set(labels.variant_key) == set(train.variant_key)
    cols = [
        'variant_key', 'protein_position', TARGET,
        'vti_treated_pct_wt', 'vnz_tez_fsk_pct_wt',
        'fsk_sem_vnz_tez', 'fsk_iva_sem_vnz_tez',
    ]
    aligned = train[['variant_key', 'protein_position']].merge(
        labels[cols], on='variant_key', sort=False, validate='one_to_one',
        suffixes=('_frozen', '_target'),
    )
    if not np.array_equal(
        aligned.protein_position_frozen.to_numpy(int),
        aligned.protein_position_target.to_numpy(int),
    ) or aligned.variant_key.tolist() != train.variant_key.tolist():
        raise ValueError('Frozen TRAIN identity/position alignment mismatch')
    y = aligned[TARGET].to_numpy(float)
    heads = aligned[['vti_treated_pct_wt', 'vnz_tez_fsk_pct_wt']].to_numpy(float)
    sigma = np.sqrt(
        aligned.fsk_sem_vnz_tez.to_numpy(float) ** 2 +
        aligned.fsk_iva_sem_vnz_tez.to_numpy(float) ** 2
    )
    assert np.isfinite(y).all() and np.isfinite(heads).all() and np.isfinite(sigma).all()
    assert np.allclose(heads[:, 0] - heads[:, 1], y, rtol=0, atol=1e-8)

    selection_path = ROOT / 'results/boltz/phase19c/acute_iva_vti_pp_inner_selection.csv'
    choices = pd.read_csv(selection_path)
    choices = choices.loc[choices.model.eq('sequence_structure__xgboost')].copy()
    assert len(choices) == 10, ('Expected 10 original nested selections', len(choices))
    assert not choices.duplicated(['repeat', 'fold']).any()
    choices = {(int(r.repeat), int(r.fold)): json.loads(r.hyperparams)
               for r in choices.itertuples(index=False)}
    for rep, fold, fit, held in outer:
        assert (int(rep), int(fold)) in choices
        assert len(set(train.protein_position.iloc[fit]) & set(train.protein_position.iloc[held])) == 0
    return train, X, y, heads, sigma, outer, choices, label_path, selection_path


def tune_strength(X, y, heads, sigma, pos, params, seed, mode, strengths):
    # GroupKFold ensures no protein position overlaps between inner train/valid.
    inner = list(GroupKFold(n_splits=3).split(X, y, groups=pos))
    evaluations = []
    for strength in strengths:
        residual = np.full(len(y), np.nan, dtype=float)
        for j, (tr, val) in enumerate(inner):
            assert not (set(pos[tr]) & set(pos[val]))
            pred = train_predict(
                X[tr], y[tr], X[val], params=params,
                seed=seed + j, mode=mode, strength=strength,
                sigma_train=sigma[tr], heads=heads[tr],
            )
            assert np.isfinite(pred).all() and len(pred) == len(val)
            residual[val] = np.abs(y[val] - pred)
        assert np.isfinite(residual).all()
        mean_mae = float(pd.DataFrame({'pos': pos, 'ae': residual})
                         .groupby('pos').ae.mean().mean())
        evaluations.append((mean_mae, strength))
    # Stable tie-break: prefer the first (generally least aggressive) setting.
    best_index = min(range(len(evaluations)), key=lambda k: (evaluations[k][0], k))
    return evaluations[best_index][1], evaluations[best_index][0], evaluations


def summarize(oof: pd.DataFrame, train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert len(oof) == 210 * 2 * len(MODELS)
    assert not oof.duplicated(['repeat', 'model', 'variant_key']).any()
    averaged = (oof.groupby(['model', 'variant_key'], as_index=False)
                .agg(protein_position=('protein_position', 'first'),
                     observed_pp=('observed_pp', 'first'),
                     predicted_pp=('predicted_pp', 'mean'),
                     n_repeats=('repeat', 'nunique')))
    assert averaged.n_repeats.eq(2).all()
    assert len(averaged) == 210 * len(MODELS)
    measures = []
    for model, g in averaged.groupby('model'):
        y = g.observed_pp.to_numpy(float)
        p = g.predicted_pp.to_numpy(float)
        pos = g.protein_position.to_numpy(int)
        err = y - p
        hi = y > 20
        vhi = y > 40
        lo = y <= 2
        measures.append(dict(
            model=model,
            position_mae_pp=position_mae(y, p, pos),
            variant_mae_pp=float(np.abs(err).mean()),
            rmse_pp=float(np.sqrt(np.mean(err ** 2))),
            bias_observed_minus_pred_pp=float(err.mean()),
            high20_mae_pp=float(np.abs(err[hi]).mean()),
            high40_mae_pp=float(np.abs(err[vhi]).mean()),
            low2_mae_pp=float(np.abs(err[lo]).mean()),
            high20_detection_recall=float(np.mean(p[hi] > 20.0)),
            high20_prediction_count=int(np.sum(p > 20.0)),
            max_predicted_pp=float(np.max(p)),
        ))
    summary = pd.DataFrame(measures)
    control_mae = float(summary.loc[summary.model.eq('control_xgb'), 'position_mae_pp'].iloc[0])
    summary['gain_vs_control_pp'] = control_mae - summary.position_mae_pp
    return averaged, summary.sort_values('position_mae_pp').reset_index(drop=True)


def paired_bootstrap(averaged: pd.DataFrame, n_boot=20000):
    positions = sorted(averaged.protein_position.unique())
    assert len(positions) == 168
    e = {}
    for name, frame in averaged.groupby('model'):
        errors = (frame.assign(ae=(frame.observed_pp - frame.predicted_pp).abs())
                  .groupby('protein_position').ae.mean().reindex(positions))
        assert errors.notna().all()
        e[name] = errors.to_numpy(float)
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(positions), size=(n_boot, len(positions)))
    rows = []
    for mode in MODELS:
        if mode == 'control_xgb':
            continue
        gains = e['control_xgb'] - e[mode]
        boot = gains[indices].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        rows.append(dict(
            model=mode,
            gain_pp=float(np.mean(gains)),
            ci95_lo=float(lo), ci95_hi=float(hi),
            n_positions_improved=int(np.sum(gains > 0)),
            fraction_bootstrap_positive=float(np.mean(boot > 0)),
        ))
    return pd.DataFrame(rows)


def main(smoke: bool = False):
    train, X, y, heads, sigma, outer, params_by_fold, labels_file, selection_file = load_inputs()
    root_out = ROOT / 'results/boltz/phase19i'
    print('=' * 108, flush=True)
    print('PHASE 19I | FIVE LEARNING-OBJECTIVE EXPERIMENTS — TRAIN ONLY', flush=True)
    print('=' * 108, flush=True)
    print('210 frozen VTI train variants / 168 positions; same 2 x 5 outer folds', flush=True)
    print('Outer-fold XGBoost hyperparameters: reused from Phase 19C INNER selections', flush=True)
    print('Tail/oversampling/reliability strengths: selected by nested GroupKFold(3),', flush=True)
    print('  targeting unweighted position-macro MAE on inner held-out positions', flush=True)
    print('Two-head target: treated FSK+IVA and treated FSK, predicted difference', flush=True)
    print('Assay SEM is used for training weights only, never as a prediction feature', flush=True)
    print('Validation/test labels: NEVER OPENED', flush=True)
    print('Exploratory post-hoc subgroup diagnostics: <=2, >20, >40 pp', flush=True)

    # Compare an unweighted raw-XGB model to project's wrapper on first held-out fold.
    # If they differ, all five experiments still use identical raw-XGB preprocessing,
    # so within-Phase-19I comparisons remain valid, but results are NOT parity claims.
    rep0, fold0, fit0, held0 = outer[0]
    p0 = params_by_fold[(int(rep0), int(fold0))]
    seed0 = int(mz.SEED + rep0 * 100 + fold0)
    wrapped = mz.estimator('xgboost', p0, seed=seed0)
    wrapped.fit(X[fit0], y[fit0])
    reference_pred = np.asarray(wrapped.predict(X[held0]), float)
    new_pred = train_predict(X[fit0], y[fit0], X[held0], p0, seed0,
                             'control_xgb', 1)
    parity = float(np.max(np.abs(reference_pred - new_pred)))
    print(f'\nESTIMATOR AUDIT | project wrapper vs shared weighted-capable XGB core:', flush=True)
    print(f'  max prediction difference, first outer fold = {parity:.8f} pp', flush=True)
    if parity > 1e-4:
        print('  NOTE: wrapper preprocessing differs. Primary effects are evaluated', flush=True)
        print('  against NEW internally matched control_xgb, not historical 7.5817.', flush=True)
    else:
        print('  PASS: estimator parity on first fold.', flush=True)

    results, selection = [], []
    for ix, (rep, fold, fit, held) in enumerate(outer[:1] if smoke else outer):
        rep, fold = int(rep), int(fold)
        params = params_by_fold[(rep, fold)]
        seed = int(mz.SEED + rep * 100 + fold)
        pf = train.protein_position.to_numpy(int)
        print(f'\n[OUTER] repeat={rep}/2 fold={fold}/5 | fit={len(fit)} held={len(held)}', flush=True)
        assert not (set(pf[fit]) & set(pf[held]))
        for mode in MODELS:
            t0 = perf_counter()
            if mode in STRENGTHS:
                strength, inner_mae, vals = tune_strength(
                    X[fit], y[fit], heads[fit], sigma[fit], pf[fit], params,
                    seed, mode, STRENGTHS[mode],
                )
                choices = ', '.join(f'{v:g}:{m:.3f}' for m, v in vals)
            else:
                strength, inner_mae, choices = 1, float('nan'), '(no strength tuning)'
            pred = train_predict(
                X[fit], y[fit], X[held], params, seed,
                mode, strength, sigma_train=sigma[fit], heads=heads[fit],
            )
            assert len(pred) == len(held) and np.isfinite(pred).all()
            held_mae = position_mae(y[held], pred, pf[held])
            print(f'  {mode:<21s} | selected={strength:g} | '
                  f'outer pos-MAE={held_mae:8.3f} | inner={inner_mae:8.3f} | '
                  f'{perf_counter()-t0:.1f}s', flush=True)
            if mode in STRENGTHS:
                print(f'    inner options (strength:MAE): {choices}', flush=True)
            selection.append(dict(repeat=rep, fold=fold, model=mode,
                                  strength=float(strength), inner_position_mae_pp=inner_mae,
                                  outer_position_mae_pp=held_mae,
                                  hyperparams=json.dumps(params, sort_keys=True),
                                  inner_choices=choices))
            for index, value in zip(held, pred):
                results.append(dict(repeat=rep, fold=fold, model=mode,
                                    variant_key=str(train.variant_key.iloc[index]),
                                    protein_position=int(pf[index]),
                                    observed_pp=float(y[index]), predicted_pp=float(value)))
    if smoke:
        print('\nPASS: Smoke test. No outputs written.', flush=True)
        return

    oof = pd.DataFrame(results)
    averaged, summary = summarize(oof, train)
    boot = paired_bootstrap(averaged)

    print('\n' + '=' * 108, flush=True)
    print('PHASE 19I | REPEATED OOF RESULTS | POSITION-MACRO PRIMARY', flush=True)
    print('=' * 108, flush=True)
    print(summary[['model','position_mae_pp','gain_vs_control_pp','rmse_pp',
                   'high20_mae_pp','high40_mae_pp','low2_mae_pp',
                   'high20_detection_recall','max_predicted_pp']]
          .round(3).to_string(index=False), flush=True)
    print('\nPaired position bootstrap (168 positions; 20,000 resamples)', flush=True)
    print(boot.round(4).to_string(index=False), flush=True)

    old_path = ROOT / 'results/boltz/phase19c/acute_iva_vti_pp_averaged.csv'
    old = pd.read_csv(old_path, dtype={'variant_key': str})
    old = old.loc[old.model.eq('sequence_structure__xgboost')].copy()
    assert len(old) == 210
    old = old.set_index('variant_key').loc[train.variant_key]
    old_mae = position_mae(old.observed_pp.to_numpy(float),
                           old.predicted_pp.to_numpy(float),
                           old.protein_position.to_numpy(int))
    assert np.allclose(old.observed_pp.to_numpy(float), y, atol=1e-8)
    print(f'\nHISTORICAL Phase 19C sequence+structure XGB: {old_mae:.4f} pp', flush=True)
    print('NOTE: Historical reference is NOT used to select strengths or tune models.', flush=True)
    print('NOTE: Outer-OOF comparison is exploratory; CI not corrected for method selection.', flush=True)
    print('NOTE: n=9 variants above 40 pp; high-response subgroup results are unstable.', flush=True)
    print('NOTE: Oversampling repeats labels; it does not create biological data.', flush=True)
    root_out.mkdir(parents=True, exist_ok=True)
    files = {
        'oof': root_out / 'phase19i_oof.csv',
        'averaged': root_out / 'phase19i_averaged.csv',
        'summary': root_out / 'phase19i_summary.csv',
        'bootstrap': root_out / 'phase19i_position_bootstrap.csv',
        'selection': root_out / 'phase19i_inner_strength_selection.csv',
        'report': root_out / 'phase19i_report.json',
    }
    oof.to_csv(files['oof'], index=False)
    averaged.to_csv(files['averaged'], index=False)
    summary.to_csv(files['summary'], index=False)
    boot.to_csv(files['bootstrap'], index=False)
    pd.DataFrame(selection).to_csv(files['selection'], index=False)
    report = {
        'phase': '19I_learning_objective_augmentation_ablation',
        'status': 'train_only_exploratory',
        'train_variants': 210, 'train_positions': 168,
        'outer_folds': 'unchanged Phase17A/19C 2x5 position-disjoint',
        'inner_folds': '3fold GroupKFold only on each outer train',
        'frozen_xgb_params': str(selection_file.relative_to(ROOT)),
        'training_labels': str(labels_file.relative_to(ROOT)),
        'historical_mae_pp': old_mae,
        'first_fold_wrapper_raw_xgb_max_abs_diff_pp': parity,
        'strengths': STRENGTHS,
        'environment': {'python': platform.python_version()},
        'no_heldout_val_or_test_labels_opened': True,
        'interpretation': 'Posthoc method comparison; external eval required before final claims',
    }
    files['report'].write_text(json.dumps(report, indent=2) + '\n')
    print('\nOUTPUTS:', flush=True)
    for name, p in files.items():
        print(f'  {name}: {p.relative_to(ROOT)}', flush=True)
    print('\nPASS: Phase19I completed; training-only, no protected outcomes read', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--smoke', action='store_true', help='First outer fold only; no outputs written')
    args = ap.parse_args()
    main(smoke=args.smoke)
