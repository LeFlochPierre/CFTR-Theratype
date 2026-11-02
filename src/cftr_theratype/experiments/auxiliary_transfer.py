"""Canonical TRAIN-only reimplementation of Phase26H ETI->VTI auxiliary transfer.

Frozen protein-position held-out folds and globally protected positions are
inherited from canonical loaders; no VAL/TEST outcomes loaded. Original phase
script is unchanged so parity can be checked rather than assumed.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from cftr_theratype.models.auxiliary import choose_weight, fit_joint, positional_shuffle
from cftr_theratype.evaluation.metrics import position_mae

KEYS = ('repeat', 'fold', 'variant_key', 'protein_position')


def evaluate(vti, vti80, eti, eti80, folds, *, shuffles=2, max_folds=None):
    """Produce Phase26H-compatible core OOF predictions and inner selections."""
    if not 0 <= shuffles <= 10:
        raise ValueError('shuffles must be in 0..10')
    if max_folds is not None and max_folds < 1:
        raise ValueError('max_folds must be positive')
    vti80, eti80 = np.asarray(vti80, float), np.asarray(eti80, float)
    if vti80.shape != (len(vti), 80) or eti80.shape != (len(eti), 80):
        raise ValueError('Expected matching 80-feature VTI/ETI arrays')
    keys = vti.variant_key.astype(str).to_numpy()
    pos = vti.protein_position.to_numpy(int)
    y = vti.corrector_rescue_pp.to_numpy(float)
    aux_y = eti.corrector_rescue_pp.to_numpy(float)
    aux_pos = eti.protein_position.to_numpy(int)
    if len(set(keys)) != len(keys) or not np.isfinite(y).all() or not np.isfinite(aux_y).all():
        raise ValueError('Invalid TRAIN labels or duplicate variant keys')
    # Last feature is the actual assay identity (VTI=1; ETI=0).
    vx = np.column_stack([vti80, np.ones(len(vti80))])
    ex = np.column_stack([eti80, np.zeros(len(eti80))])
    rows, choices = [], []
    selected = folds if max_folds is None else folds[:max_folds]
    for f, (repeat, fold, trkeys, tekeys) in enumerate(selected, 1):
        it = np.flatnonzero(np.isin(keys, list(trkeys)))
        iv = np.flatnonzero(np.isin(keys, list(tekeys)))
        if (not len(it) or not len(iv) or len(it) + len(iv) != len(keys)
                or set(pos[it]) & set(pos[iv])):
            raise ValueError(f'Invalid frozen fold {repeat}/{fold}')
        heldout = set(pos[iv])
        eligible = ~np.isin(aux_pos, sorted(heldout))
        if not eligible.any() or set(aux_pos[eligible]) & heldout:
            raise ValueError('Outer-fold ETI protected-position leakage or no auxiliary data')
        weight, score = choose_weight(vx[it], y[it], pos[it], ex[eligible],
                                      aux_y[eligible], aux_pos[eligible])
        predictions = {
            'joint_real': fit_joint(vx[it], y[it], ex[eligible], aux_y[eligible], weight).predict(vx[iv])
        }
        choices.append({
            'repeat': repeat, 'fold': fold, 'model': 'joint_real', 'eti_weight': weight,
            'inner_pos_mae_pp': score, 'outer_train_vti_positions': len(np.unique(pos[it])),
            'outer_test_positions': len(heldout), 'outer_eti_variants': int(eligible.sum()),
        })
        for k in range(shuffles):
            sham_seed = 2608 + 100 * f + k
            sham_labels = positional_shuffle(aux_y[eligible], aux_pos[eligible], sham_seed)
            sham_weight, sham_score = choose_weight(
                vx[it], y[it], pos[it], ex[eligible], aux_y[eligible], aux_pos[eligible],
                shuffle_seed=sham_seed,
            )
            name = f'joint_sham{k + 1}'
            predictions[name] = fit_joint(
                vx[it], y[it], ex[eligible], sham_labels, sham_weight,
            ).predict(vx[iv])
            choices.append({
                'repeat': repeat, 'fold': fold, 'model': name, 'eti_weight': sham_weight,
                'inner_pos_mae_pp': sham_score, 'outer_train_vti_positions': len(np.unique(pos[it])),
                'outer_test_positions': len(heldout), 'outer_eti_variants': int(eligible.sum()),
            })
        for j, i in enumerate(iv):
            row = {'repeat': repeat, 'fold': fold, 'variant_key': keys[i],
                   'protein_position': int(pos[i]), 'study_cohort': vti.study_cohort.iloc[i],
                   'observed_corrector': float(y[i]), 'observed_total': float(vti.vti_rescue_pp.iloc[i])}
            row.update({name: float(pred[j]) for name, pred in predictions.items()})
            rows.append(row)
        print(f'completed {f}/{len(selected)}: repeat={repeat}, fold={fold}, '
              f'test_positions={len(heldout)}, ETI_train={int(eligible.sum())}, '
              f'selected_weight={weight}', flush=True)
    result = pd.DataFrame(rows)
    if result.duplicated(list(KEYS[:3])).any():
        raise ValueError('Duplicate held-out OOF keys')
    if max_folds is None:
        if len(selected) != 10 or result.groupby('repeat').variant_key.nunique().to_dict() != {1: 210, 2: 210}:
            raise ValueError('Incomplete frozen 2x5 TRAIN OOF')
    return result, pd.DataFrame(choices)


def compare_oof(new_path, historical_path, *, atol=1e-6, full=True):
    """Compare core Phase26H predictions even if legacy file has extra report columns."""
    new = pd.read_csv(new_path, dtype={'variant_key': str})
    old = pd.read_csv(historical_path, dtype={'variant_key': str})
    required = set(KEYS) | {'observed_corrector', 'observed_total', 'joint_real'}
    for name, frame in (('new', new), ('historical', old)):
        if required - set(frame) or frame.duplicated(list(KEYS[:3])).any():
            raise ValueError(f'Invalid {name} Phase26H OOF schema or duplicate keys')
        if full and (len(frame) != 420 or frame.groupby('repeat').variant_key.nunique().to_dict() != {1: 210, 2: 210}):
            raise ValueError(f'Incomplete {name} OOF rows')
    cols = ['observed_corrector', 'observed_total', 'joint_real']
    cols += sorted(c for c in new if c.startswith('joint_sham') and c in old)
    for c in new:
        if c.startswith('joint_sham') and c not in old:
            raise ValueError(f'Historical OOF missing negative control {c}')
    a = new[list(KEYS) + cols].merge(old[list(KEYS) + cols], on=list(KEYS),
                                     suffixes=('_new', '_old'), how='outer',
                                     indicator=True, validate='one_to_one')
    if len(a) != len(new) or len(a) != len(old) or not a._merge.eq('both').all():
        raise ValueError('Historical fold/variant membership mismatch')
    differences = {}
    for col in cols:
        x = a[col + '_new'].to_numpy(float)
        y = a[col + '_old'].to_numpy(float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f'Nonfinite {col}')
        delta = float(np.max(np.abs(x - y)))
        differences[col] = delta
        if delta > atol:
            raise ValueError(f'PARITY FAIL {col}: max_abs_diff={delta:g} > {atol:g}')
    print(f'PASS: Phase26H OOF parity — {len(a)} rows, {len(cols)} numeric outputs')
    for col, delta in differences.items():
        print(f'  {col}: max_abs_diff={delta:.9g}')
    return differences


def save_results(oof, selected, root, *, outdir=None):
    root = Path(root)
    folder = Path(outdir) if outdir else root / 'results/maintenance/refactor_audit/stage6/transfer'
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / 'phase26h_oof.csv', folder / 'phase26h_inner_selections.csv']
    if any(p.exists() for p in paths):
        raise FileExistsError(f'Refusing to overwrite existing transfer results: {paths}')
    oof.to_csv(paths[0], index=False)
    selected.to_csv(paths[1], index=False)
    summary = pd.DataFrame([
        {'repeat': int(rep), 'model': model,
         'position_mae_pp': position_mae(g.observed_corrector.to_numpy(float),
                                       g[model].to_numpy(float),
                                       g.protein_position.to_numpy(int))}
        for rep, g in oof.groupby('repeat')
        for model in ('joint_real',) + tuple(c for c in oof if c.startswith('joint_sham'))
    ])
    path = folder / 'phase26h_simple_metrics.csv'
    if path.exists():
        raise FileExistsError(f'Refusing overwrite {path}')
    summary.to_csv(path, index=False)
    print('Saved TRAIN-only ETI->VTI transfer to:', folder)
    return folder


def run(root, *, shuffles=2, max_folds=None, outdir=None):
    from cftr_theratype.datasets.eti_auxiliary import load_data
    root = Path(root).resolve()
    vti, vti80, eti, eti80, folds = load_data(root)
    oof, selected = evaluate(vti, vti80, eti, eti80, folds,
                             shuffles=shuffles, max_folds=max_folds)
    save_results(oof, selected, root, outdir=outdir)
    return oof, selected
