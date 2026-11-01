#!/usr/bin/env python3
"""Phase 21D: Does *predictable* CFTR maturation help acute IVA prediction?

Controlled train-only experiment on the exact Phase 19C/20B frozen 2x5
protein-position-disjoint outer folds. In each outer-fit set, maturation
features are GROUP-CROSS-FITTED: no sample's own position may supervise
its stage-1 prediction. Outer-held features are predicted using models
trained only on the outer fit positions. No measured maturation marker is
used as an acute-IVA inference-time input.

Compares: control XGB, predicted band-C, predicted four maturation features,
and a negative control with shuffled band-C labels. Previous reliability-
weighted XGB is included as a read-only post-hoc benchmark if available.

Intentional separation: Tedman 2025 is NOT trained in this phase; the external
expression targets are not directly comparable to Bihler band C, and only 14
exact substitutions overlap the VTI train set. This isolates whether
maturation has *mutation-predictable* signal before external transfer.

Usage:
    python scripts/phase21d_crossfit_maturation_acute.py --smoke
    python scripts/phase21d_crossfit_maturation_acute.py

Requires CFTR_ENV and completed Phases 19B, 19C, 19I, 21A.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cftr_theratype import model_zoo as mz
from cftr_theratype.transfer import ROOT

OUT = ROOT / 'results/biology/phase21d'
LABELS = ROOT / 'results/boltz/phase19b/vti_train_acute_iva_targets.csv'
MAT = ROOT / 'results/biology/phase21a/vti_train_maturation_assays.csv'
SELECTION = ROOT / 'results/boltz/phase19c/acute_iva_vti_pp_inner_selection.csv'
HIST = ROOT / 'results/boltz/phase19c/acute_iva_vti_pp_oof.csv'
PRIOR = ROOT / 'results/boltz/phase19i/phase19i_averaged.csv'
TARGET = 'acute_iva_vti_pp'
AUX = ['band C, %WT', 'band C, %Total', 'C-B ratio, %WT', 'Total protein, %WT']
MODES = ('control_xgb', 'pred_band_c', 'pred_four_maturation', 'shuffled_band_c')


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def pos_mae(y, pred, pos):
    return float(pd.DataFrame({'position': pos, 'ae': np.abs(np.asarray(y) - np.asarray(pred))})
                 .groupby('position').ae.mean().mean())


def load_inputs():
    train, mats, _, outer = mz.prepare(ROOT)
    assert len(train) == 210 and train.protein_position.nunique() == 168 and len(outer) == 10
    assert train.variant_key.is_unique
    X = np.asarray(mats['sequence_structure'], dtype=float)
    assert X.shape == (210, 80) and not np.isinf(X).any()
    pos = train.protein_position.to_numpy(int)

    labels = pd.read_csv(LABELS, dtype={'variant_key': str})
    assert len(labels) == 210 and labels.variant_key.is_unique
    assert set(labels.variant_key) == set(train.variant_key)
    joined = train[['variant_key']].merge(labels[['variant_key', TARGET]],
                                        on='variant_key', validate='one_to_one', sort=False)
    assert joined.variant_key.tolist() == train.variant_key.tolist()
    y = joined[TARGET].to_numpy(float)
    assert np.isfinite(y).all()

    mat = pd.read_csv(MAT, dtype={'variant_key': str})
    assert len(mat) == 420 and not mat.duplicated(['variant_key', 'Treatment']).any()
    assert set(mat.variant_key) == set(train.variant_key)
    mat = mat.loc[mat.Treatment.eq('VX121+VX661'), ['variant_key'] + AUX]
    assert len(mat) == 210
    aux_df = train[['variant_key']].merge(mat, on='variant_key', sort=False, validate='one_to_one')
    assert aux_df.variant_key.tolist() == train.variant_key.tolist()
    a = aux_df[AUX].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    assert a.shape == (210, 4) and np.isfinite(a).all()

    choices = pd.read_csv(SELECTION)
    choices = choices.loc[choices.model.eq('sequence_structure__xgboost')]
    assert len(choices) == 10 and not choices.duplicated(['repeat', 'fold']).any()
    params = {(int(r.repeat), int(r.fold)): json.loads(r.hyperparams)
              for r in choices.itertuples(index=False)}
    for rep, fold, fit, held in outer:
        assert (int(rep), int(fold)) in params
        assert not np.intersect1d(pos[fit], pos[held]).size
    return train, X, y, a, pos, outer, params


def predict_stage1(Xtr, ytr, Xvalid, params, seed):
    """Separate small XGB models for the auxiliary *training* targets only."""
    pred = np.empty((len(Xvalid), ytr.shape[1]), dtype=float)
    for k in range(ytr.shape[1]):
        model = mz.estimator('xgboost', params, seed + k)
        model.fit(Xtr, ytr[:, k])
        pred[:, k] = np.asarray(model.predict(Xvalid), float)
    assert np.isfinite(pred).all()
    return pred


def group_crossfit_features(Xtr, aux_tr, pos_tr, Xheld, params, seed, *, shuffle=False):
    """All fit-set maturation features are predicted OUT OF GROUP.

    Outer-held features are predicted from full outer fit. Shuffling affects
    only outer-fit auxiliary labels and never any acute-IVA labels.
    """
    Xtr, Xheld = np.asarray(Xtr, float), np.asarray(Xheld, float)
    aux_tr = np.asarray(aux_tr, float).copy()
    pos_tr = np.asarray(pos_tr, int)
    assert not np.isinf(Xtr).any() and not np.isinf(Xheld).any()
    assert np.isfinite(aux_tr).all()
    if shuffle:
        # Destabilize the biological mapping without changing the marginal
        # distribution or ever using outer-held maturation measurements.
        perm = np.random.default_rng(seed + 1907).permutation(len(aux_tr))
        aux_tr = aux_tr[perm]
    fit_pred = np.full_like(aux_tr, np.nan, dtype=float)
    cv = GroupKFold(n_splits=3)
    for j, (subfit, subheld) in enumerate(cv.split(Xtr, groups=pos_tr)):
        assert not np.intersect1d(pos_tr[subfit], pos_tr[subheld]).size
        fit_pred[subheld] = predict_stage1(
            Xtr[subfit], aux_tr[subfit], Xtr[subheld], params, seed + 200 + j * 10)
    assert np.isfinite(fit_pred).all()
    held_pred = predict_stage1(Xtr, aux_tr, Xheld, params, seed + 800)
    return fit_pred, held_pred


def acute_predict(Xfit, yfit, Xheld, params, seed):
    reg = mz.estimator('xgboost', params, seed)
    reg.fit(Xfit, yfit)
    p = np.asarray(reg.predict(Xheld), float)
    assert p.shape == (len(Xheld),) and np.isfinite(p).all()
    return p


def summarize(frame):
    expected = len(MODES) * 210 * 2
    assert len(frame) == expected and not frame.duplicated(['repeat', 'model', 'variant_key']).any()
    avg = (frame.groupby(['model', 'variant_key', 'protein_position'], as_index=False)
           .agg(observed_pp=('observed_pp', 'first'), predicted_pp=('predicted_pp', 'mean'),
                n_repeats=('repeat', 'nunique')))
    assert avg.groupby('model').size().eq(210).all() and avg.n_repeats.eq(2).all()
    rows = []
    for mode, a in avg.groupby('model'):
        y = a.observed_pp.to_numpy(float)
        p = a.predicted_pp.to_numpy(float)
        pos = a.protein_position.to_numpy(int)
        high20 = y > 20; high40 = y > 40; low2 = y <= 2
        rows.append(dict(model=mode, position_mae_pp=pos_mae(y, p, pos),
                         variant_mae_pp=float(np.mean(np.abs(y - p))),
                         rmse_pp=float(np.sqrt(np.mean((y - p)**2))),
                         high20_mae_pp=float(np.mean(np.abs(y[high20] - p[high20]))),
                         high40_mae_pp=float(np.mean(np.abs(y[high40] - p[high40]))),
                         low2_mae_pp=float(np.mean(np.abs(y[low2] - p[low2]))),
                         high20_recall=float(np.mean(p[high20] > 20)),
                         max_predicted_pp=float(np.max(p))))
    summary = pd.DataFrame(rows).sort_values('position_mae_pp')
    return avg, summary


def benchmark_prior(avg, prior_path=PRIOR):
    if not prior_path.is_file():
        return avg, False
    prior = pd.read_csv(prior_path, dtype={'variant_key': str})
    prior = prior.loc[prior.model.eq('reliability_weighted')].copy()
    assert len(prior) == 210 and prior.variant_key.is_unique
    ref = avg.loc[avg.model.eq('control_xgb'), ['variant_key', 'protein_position', 'observed_pp']]
    chk = ref.merge(prior[['variant_key', 'protein_position', 'observed_pp']],
                    on='variant_key', validate='one_to_one', suffixes=('_new', '_prior'))
    assert len(chk) == 210
    assert (chk.protein_position_new == chk.protein_position_prior).all()
    assert np.allclose(chk.observed_pp_new, chk.observed_pp_prior, rtol=0, atol=1e-8)
    prior['model'] = 'historical_reliability_19i'
    prior = prior[['model', 'variant_key', 'protein_position', 'observed_pp', 'predicted_pp']]
    return pd.concat([avg, prior], ignore_index=True), True


def bootstrap(avg, modes, n_boot=20000):
    out = []
    table = (avg.assign(ae=(avg.observed_pp - avg.predicted_pp).abs())
             .groupby(['protein_position', 'model']).ae.mean().unstack())
    assert len(table) == 168 and table.notna().all().all()
    rng = np.random.default_rng(42)
    draws = rng.integers(0, len(table), size=(n_boot, len(table)))
    for mode in modes:
        if mode == 'control_xgb':
            continue
        refs = ['control_xgb'] + (['historical_reliability_19i']
                                    if 'historical_reliability_19i' in table else [])
        if mode == 'pred_band_c':
            refs.append('shuffled_band_c')
        for ref in refs:
            if mode == ref:
                continue
            d = table[ref].to_numpy(float) - table[mode].to_numpy(float)
            bs = d[draws].mean(axis=1)
            lo, hi = np.quantile(bs, [0.025, 0.975])
            out.append(dict(model=mode, comparator=ref, gain_pp=float(np.mean(d)),
                            ci95_lo=float(lo), ci95_hi=float(hi),
                            positions_improved=int((d > 0).sum()),
                            frac_boot_positive=float((bs > 0).mean())))
    return pd.DataFrame(out)


def main(smoke=False):
    train, X, y, aux, pos, outer, params = load_inputs()
    print('=' * 104, flush=True)
    print('PHASE 21D | POSITION-SAFE PREDICTED CFTR MATURATION -> ACUTE IVA', flush=True)
    print('=' * 104, flush=True)
    print('VTI TRAIN ONLY: 210 variants / 168 protein positions', flush=True)
    print('Frozen 2 x 5 outer position folds; Phase 19C per-fold INNER-selected XGB parameters', flush=True)
    print('Stage1 training rows: 3-fold GroupKFold maturation predictions (NO own-position supervision)', flush=True)
    print('Stage1 held rows: model trained only on outer fit positions', flush=True)
    print('Stage2 inputs: mutation sequence+structure, plus only PREDICTED maturation', flush=True)
    print('Negative control: band-C labels shuffled within outer training set', flush=True)
    print('Tedman targets not yet incorporated; no VTI/ETI protected outcomes accessed', flush=True)
    print('Mode:', 'SMOKE' if smoke else 'FULL', flush=True)

    out_rows = []
    for rep, fold, fit, held in (outer[:1] if smoke else outer):
        t0 = perf_counter()
        rep, fold = int(rep), int(fold)
        seed = mz.SEED + rep * 100 + fold
        prm = params[(rep, fold)]
        assert not np.intersect1d(pos[fit], pos[held]).size
        predictions = {}
        predictions['control_xgb'] = acute_predict(X[fit], y[fit], X[held], prm, seed)

        b_fit, b_held = group_crossfit_features(
            X[fit], aux[fit, :1], pos[fit], X[held], prm, seed + 1000)
        predictions['pred_band_c'] = acute_predict(
            np.column_stack([X[fit], b_fit]), y[fit],
            np.column_stack([X[held], b_held]), prm, seed)

        all_fit, all_held = group_crossfit_features(
            X[fit], aux[fit], pos[fit], X[held], prm, seed + 2000)
        predictions['pred_four_maturation'] = acute_predict(
            np.column_stack([X[fit], all_fit]), y[fit],
            np.column_stack([X[held], all_held]), prm, seed)

        s_fit, s_held = group_crossfit_features(
            X[fit], aux[fit, :1], pos[fit], X[held], prm, seed + 1000,
            shuffle=True)
        predictions['shuffled_band_c'] = acute_predict(
            np.column_stack([X[fit], s_fit]), y[fit],
            np.column_stack([X[held], s_held]), prm, seed)

        print(f'\n[OUTER] repeat={rep} fold={fold}, fit={len(fit)} held={len(held)} '
              f'({perf_counter()-t0:.1f}s)', flush=True)
        for mode, pred in predictions.items():
            print(f'  {mode:23s} position-MAE={pos_mae(y[held], pred, pos[held]):.3f}', flush=True)
            for idx, p in zip(held, pred):
                out_rows.append(dict(repeat=rep, fold=fold, model=mode,
                                     variant_key=str(train.variant_key.iloc[idx]),
                                     protein_position=int(pos[idx]),
                                     observed_pp=float(y[idx]), predicted_pp=float(p)))
    if smoke:
        print('\nPASS: Phase21D smoke test; no files written', flush=True)
        return

    frame = pd.DataFrame(out_rows)
    hist = pd.read_csv(HIST, dtype={'variant_key': str})
    hist = hist.loc[hist.model.eq('sequence_structure__xgboost')]
    check = frame.loc[frame.model.eq('control_xgb')].merge(
        hist[['repeat', 'fold', 'variant_key', 'observed_pp', 'predicted_pp']],
        on=['repeat', 'fold', 'variant_key'], how='outer', validate='one_to_one',
        suffixes=('_new', '_hist'), indicator=True)
    assert len(check) == 420 and check._merge.eq('both').all(), 'Frozen Phase19C key parity failed'
    d = float(np.max(np.abs(check.predicted_pp_new - check.predicted_pp_hist)))
    assert np.allclose(check.observed_pp_new, check.observed_pp_hist, rtol=0, atol=1e-8)
    if d > 1e-3:
        raise ValueError(f'Phase19C baseline parity failed (maximum difference {d:.6g} pp)')
    print(f'\nPASS: Phase19C baseline parity, max absolute prediction difference={d:.8f} pp', flush=True)

    avg, summary = summarize(frame)
    comparators, has_prior = benchmark_prior(avg)
    boot = bootstrap(comparators, MODES)
    if has_prior:
        h = comparators.loc[comparators.model.eq('historical_reliability_19i')]
        print(f'Historical reliability-weighted MAE: '
              f'{pos_mae(h.observed_pp, h.predicted_pp, h.protein_position):.4f} pp', flush=True)
    print('\n' + '=' * 104, flush=True)
    print('PHASE 21D | POSITION-MACRO REPEATED OOF', flush=True)
    print('=' * 104, flush=True)
    print(summary.round(4).to_string(index=False), flush=True)
    print('\nPAIRED POSITION BOOTSTRAP (168 positions; exploratory, unadjusted)', flush=True)
    print(boot.round(4).to_string(index=False), flush=True)
    print('\nNOTE: Strong internal band-C correlation is not proof that predicted band-C helps.', flush=True)
    print('NOTE: Positive improvement must also beat the shuffled auxiliary control.', flush=True)
    print('NOTE: Model selection across many experiments requires independent validation.', flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    files = {'oof': OUT / 'phase21d_oof.csv',
             'averaged': OUT / 'phase21d_averaged.csv',
             'summary': OUT / 'phase21d_summary.csv',
             'bootstrap': OUT / 'phase21d_bootstrap.csv',
             'report': OUT / 'phase21d_report.json'}
    frame.to_csv(files['oof'], index=False)
    avg.to_csv(files['averaged'], index=False)
    summary.to_csv(files['summary'], index=False)
    boot.to_csv(files['bootstrap'], index=False)
    report = {'phase': '21D_group_crossfitted_predicted_maturation',
              'n_variants': 210, 'n_positions': 168,
              'n_repeats': 2, 'n_outer_folds': 10, 'stage1_inner_group_folds': 3,
              'acute_target': TARGET, 'maturation_targets': AUX,
              'baseline_parity_max_pp': d, 'historical_reliability_available': has_prior,
              'inputs_sha256': {str(f.relative_to(ROOT)): sha256(f)
                                for f in [LABELS, MAT, SELECTION, HIST]},
              'no_protected_outcomes_read': True,
              'no_observed_maturation_used_for_heldout_prediction': True,
              'external_study_not_yet_trained': True,
              'limitations': ['Exploratory repeated TRAIN CV',
                              'No external Tedman auxiliary labels used',
                              'No direct experimental band-C feature at inference',
                              'Stage1/2 stack not inner-tuned for augmented regression']}
    files['report'].write_text(json.dumps(report, indent=2) + '\n')
    print('\nOUTPUTS:', flush=True)
    for k, v in files.items():
        print(f'  {k}: {v.relative_to(ROOT)}', flush=True)
    print('PASS: Phase21D completed; train-only; outer position-disjoint', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true',
                        help='Run first frozen outer fold and write no outputs')
    main(smoke=parser.parse_args().smoke)
