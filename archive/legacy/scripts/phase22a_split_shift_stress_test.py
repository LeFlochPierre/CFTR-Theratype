#!/usr/bin/env python3
"""Phase 22A: CFTR acute IVA label/split shift audit and grouped re-split stress test.

Does NOT read protected VTI/ETI validation or test assay outcomes; protected
partitions are inspected using identity/position/cohort metadata only.

Runs:
  python scripts/phase22a_split_shift_stress_test.py --self-test
  python scripts/phase22a_split_shift_stress_test.py                 # audit only
  python scripts/phase22a_split_shift_stress_test.py --resplits 2    # pilot
  python scripts/phase22a_split_shift_stress_test.py --resplits 8    # full

Random folds are diagnostics INSIDE the frozen 210 VTI training variants;
they NEVER replace frozen val/test membership or a final validation estimate.
Each random outer fold independently chooses XGB depth with 3-fold position-
disjoint nested CV and foldwise imputation via model_zoo's estimator.
No seed is selected post-hoc for headline reporting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

from cftr_theratype import model_zoo as mz
from cftr_theratype.transfer import ROOT

TARGET_FILE = ROOT / 'results/boltz/phase19b/vti_train_acute_iva_targets.csv'
OOF_FILE = ROOT / 'results/boltz/phase19c/acute_iva_vti_pp_oof.csv'
SPLIT_FILE = ROOT / 'data/processed/bihler_2026_transfer_splits.parquet'
OUT = ROOT / 'results/biology/phase22a'
SEEDS = (101, 211, 307, 401, 503, 601, 701, 809, 907, 1009, 1103, 1201)
BINS = (-np.inf, 2, 5, 10, 20, 40, np.inf)
BIN_NAMES = ('<=2', '2-5', '5-10', '10-20', '20-40', '>40')


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def position_mae(y, p, pos):
    frame = pd.DataFrame({'pos': np.asarray(pos), 'ae': np.abs(np.asarray(y) - np.asarray(p))})
    if not len(frame) or not np.isfinite(frame.ae).all():
        raise ValueError('Invalid predictions')
    return float(frame.groupby('pos').ae.mean().mean())


def summary_metrics(frame, *, name):
    y, p = frame.observed_pp.to_numpy(float), frame.predicted_pp.to_numpy(float)
    pos = frame.protein_position.to_numpy(int)
    h20, h40, lo2 = y > 20, y > 40, y <= 2
    return {'model': name, 'position_mae_pp': position_mae(y, p, pos),
            'variant_mae_pp': float(np.mean(np.abs(y - p))),
            'rmse_pp': float(np.sqrt(np.mean((y - p) ** 2))),
            'high20_mae_pp': float(np.mean(np.abs(y[h20] - p[h20]))) if h20.any() else np.nan,
            'high40_mae_pp': float(np.mean(np.abs(y[h40] - p[h40]))) if h40.any() else np.nan,
            'low2_mae_pp': float(np.mean(np.abs(y[lo2] - p[lo2]))) if lo2.any() else np.nan,
            'high20_recall': float(np.mean(p[h20] > 20)) if h20.any() else np.nan,
            'max_predicted_pp': float(np.max(p))}


def load_data():
    train, mats, _, original = mz.prepare(ROOT)
    if len(train) != 210 or train.protein_position.nunique() != 168 or len(original) != 10:
        raise ValueError('Unexpected frozen VTI TRAIN cardinalities')
    lab = pd.read_csv(TARGET_FILE, dtype={'variant_key': str})
    if len(lab) != 210 or not lab.variant_key.is_unique:
        raise ValueError('Acute target file must have 210 unique variant keys')
    assert set(train.variant_key) == set(lab.variant_key), 'Frozen train/target mismatch'
    label = train[['variant_key']].merge(lab[['variant_key', 'acute_iva_vti_pp']],
                                       on='variant_key', validate='one_to_one', sort=False)
    assert label.variant_key.tolist() == train.variant_key.tolist()
    y = label.acute_iva_vti_pp.to_numpy(float)
    pos = train.protein_position.to_numpy(int)
    X = np.asarray(mats['sequence_structure'], float)
    assert X.shape == (210, 80) and not np.isinf(X).any() and np.isfinite(y).all()
    metadata = pd.read_parquet(SPLIT_FILE,
                               columns=['variant_key', 'protein_position', 'transfer_split',
                                        'study_cohort', 'variant_type', 'hgvs_protein'])
    assert metadata.variant_key.is_unique
    assert set(metadata.loc[metadata.transfer_split.eq('train'), 'variant_key']) == set(train.variant_key)
    for _, _, fit, held in original:
        assert not set(pos[fit]) & set(pos[held])
    return train, X, y, pos, original, metadata


def metadata_audit(metadata):
    print('\n' + '=' * 94)
    print('A. FROZEN PARTITIONS | METADATA ONLY (NO PROTECTED LABELS)')
    print('=' * 94)
    tab = (metadata.groupby('transfer_split', dropna=False)
           .agg(variants=('variant_key', 'nunique'), positions=('protein_position', 'nunique'),
                median_protein_position=('protein_position', 'median')))
    print(tab.to_string())
    for partition in ('train', 'val', 'test', 'excluded'):
        m = metadata.loc[metadata.transfer_split.eq(partition)]
        if m.empty:
            continue
        print(f'\n{partition.upper()} | {len(m)} variants / {m.protein_position.nunique()} positions')
        print('Study cohort:', m.study_cohort.fillna('NA').value_counts().to_dict())
        print('Variant type:', m.variant_type.fillna('NA').value_counts().to_dict())
        positions = m.protein_position.dropna().astype(int).drop_duplicates()
        region = pd.cut(positions, bins=[0, 300, 600, 900, 1200, 1480],
                        labels=['1-300', '301-600', '601-900', '901-1200', '1201-1480'])
        print('Position bands (unique positions):', region.value_counts(sort=False).to_dict())
    groups = {k: set(g.protein_position.dropna().astype(int))
              for k, g in metadata.groupby('transfer_split')}
    train_pos = groups.get('train', set())
    for part in ('val', 'test', 'excluded'):
        overlap = train_pos & groups.get(part, set())
        print(f'TRAIN vs {part.upper()} position overlap:', len(overlap))
        if part in ('val', 'test') and overlap:
            raise ValueError(f'Protected position overlap: {part}, {sorted(overlap)[:12]}')
        a, b = sorted(train_pos), sorted(groups.get(part, set()))
        if a and b:
            stat = ks_2samp(a, b).statistic
            print(f'  Exploratory position KS statistic vs TRAIN: {stat:.3f} (metadata only)')
    print('CAUTION: Distributional differences in position/cohort do not prove outcome shift.')


def load_frozen_oof(train, pos, y, original):
    df = pd.read_csv(OOF_FILE, dtype={'variant_key': str})
    df = df.loc[df.model.eq('sequence_structure__xgboost')].copy()
    assert len(df) == 420 and not df.duplicated(['repeat', 'variant_key']).any()
    match = train[['variant_key', 'protein_position']].copy()
    match['target_from_train'] = y
    checked = df.merge(match, on='variant_key', validate='many_to_one', suffixes=('', '_freeze'))
    assert len(checked) == 420
    assert np.array_equal(checked.protein_position.to_numpy(int), checked.protein_position_freeze.to_numpy(int))
    assert np.allclose(checked.observed_pp, checked.target_from_train, rtol=0, atol=1e-7)
    for rep, fold, _fit, held in original:
        a = set(df.loc[df.repeat.eq(rep) & df.fold.eq(fold), 'variant_key'].astype(str))
        b = set(train.variant_key.iloc[held].astype(str))
        assert a == b, f'Frozen OOF fold mismatch {rep}, {fold}'
    return df


def sample_difficulty(X, pos, folds):
    """Distance from a held variant to nearest different-position *fit* variant.
    Scaling and imputation are learned exclusively from each fit fold.
    This is descriptive feature-space novelty, not a predictor/label.
    """
    novelty_rows = []
    for rep, fold, fit, held in folds:
        imp = SimpleImputer(strategy='median', keep_empty_features=True)
        Zfit = imp.fit_transform(X[fit]); Zheld = imp.transform(X[held])
        scaler = StandardScaler()
        Zfit = scaler.fit_transform(Zfit); Zheld = scaler.transform(Zheld)
        dist = cdist(Zheld, Zfit, metric='euclidean') / np.sqrt(Zfit.shape[1])
        assert np.isfinite(dist).all()
        closest = np.min(dist, axis=1)
        for ix, value in zip(held, closest):
            novelty_rows.append({'repeat': rep, 'fold': fold, 'index': int(ix),
                                 'nn_scaled_distance': float(value)})
    return pd.DataFrame(novelty_rows)


def frozen_audit(train, X, y, pos, original, oof):
    print('\n' + '=' * 94)
    print('B. TRAIN-ONLY OUTCOME DISTRIBUTION')
    print('=' * 94)
    print('Variants:', len(y), '| protein positions:', len(set(pos)))
    print('Range / median / mean / SD:',
          f'{y.min():.3f}, {np.median(y):.3f}, {y.mean():.3f}, {y.std(ddof=1):.3f} pp')
    print('Training response bins (inclusive right endpoint):')
    bins = pd.cut(y, bins=BINS, labels=BIN_NAMES)
    d = pd.DataFrame({'bin': bins, 'position': pos})
    print(d.groupby('bin', observed=False).agg(variants=('position','size'),
                                               positions=('position','nunique')).to_string())
    print('Strong >20 response positions:', np.unique(pos[y > 20]).size)
    print('Extreme >40 response positions:', np.unique(pos[y > 40]).size,
          ' | variants:', int((y > 40).sum()))
    print('Cohort-by-response counts (TRAIN only):')
    co = pd.DataFrame({'cohort': train.study_cohort.fillna('NA'), 'response_bin': bins})
    print(pd.crosstab(co.cohort, co.response_bin).to_string())

    print('\n' + '=' * 94)
    print('C. ORIGINAL FROZEN POSITION-DISJOINT OUTER FOLDS')
    print('=' * 94)
    rows = []
    for rep, fold, fit, held in original:
        p = oof.loc[oof.repeat.eq(rep) & oof.fold.eq(fold)].set_index('variant_key')
        pred = p.loc[train.variant_key.iloc[held], 'predicted_pp'].to_numpy(float)
        held_y = y[held]
        stat = ks_2samp(held_y, y[fit]).statistic
        rows.append({'repeat': int(rep), 'fold': int(fold),
                     'fit_positions': len(np.unique(pos[fit])),
                     'held_positions': len(np.unique(pos[held])),
                     'held_variants': len(held),
                     'held_mean_pp': np.mean(held_y),
                     'fit_mean_pp': np.mean(y[fit]),
                     'mean_shift_pp': np.mean(held_y) - np.mean(y[fit]),
                     'held_gt20': int((held_y > 20).sum()),
                     'held_gt40': int((held_y > 40).sum()),
                     'label_KS_vs_fit': stat,
                     'fold_position_mae_pp': position_mae(held_y, pred, pos[held]),
                     'fold_variant_mae_pp': float(np.mean(np.abs(held_y-pred)))})
    folds = pd.DataFrame(rows)
    print(folds.round(3).to_string(index=False))
    print(f'Folds with no >40 responders: {int((folds.held_gt40 == 0).sum())}/10')
    print('Frozen per-repeat position-macro MAE:')
    for rep, frame in oof.groupby('repeat'):
        print(f'  repeat={int(rep)}: {position_mae(frame.observed_pp, frame.predicted_pp, frame.protein_position):.4f} pp')
    avg = oof.groupby('variant_key', as_index=False).agg(
        protein_position=('protein_position','first'), observed_pp=('observed_pp','first'),
        predicted_pp=('predicted_pp','mean'))
    print('Frozen 2-repeat averaged position MAE:',
          f'{position_mae(avg.observed_pp, avg.predicted_pp, avg.protein_position):.4f} pp')

    print('\n' + '=' * 94)
    print('D. FEATURE-SPACE NOVELTY AND ERRORS (DESCRIPTIVE ONLY)')
    print('=' * 94)
    novelty = sample_difficulty(X, pos, original)
    merged = oof.merge(train[['variant_key']], on='variant_key', validate='many_to_one')
    idx_lookup = dict(zip(train.variant_key, range(len(train))))
    merged['index'] = merged.variant_key.map(idx_lookup)
    merged = merged.merge(novelty, on=['repeat','fold','index'], validate='one_to_one')
    merged['absolute_error_pp'] = abs(merged.observed_pp - merged.predicted_pp)
    per_variant = merged.groupby('variant_key', as_index=False).agg(
        protein_position=('protein_position','first'), observed_pp=('observed_pp','first'),
        absolute_error_pp=('absolute_error_pp','mean'),
        nn_scaled_distance=('nn_scaled_distance','mean'))
    rho, _ = spearmanr(per_variant.absolute_error_pp, per_variant.nn_scaled_distance)
    print(f'Spearman (error vs closest-fit mutation-feature distance): rho={rho:+.3f} (n=210)')
    print('Median novelty by observed response:')
    per_variant['response_bin'] = pd.cut(per_variant.observed_pp, bins=BINS, labels=BIN_NAMES)
    print(per_variant.groupby('response_bin',observed=False).agg(
        n=('variant_key','size'), median_novelty=('nn_scaled_distance','median'),
        mean_abs_error_pp=('absolute_error_pp','mean')).round(4).to_string())
    print('\nTen highest-error TRAIN-only variants (never protected labels):')
    print(per_variant.sort_values('absolute_error_pp', ascending=False).head(10)[
        ['variant_key','protein_position','observed_pp','absolute_error_pp','nn_scaled_distance']
    ].round(3).to_string(index=False))
    return folds, per_variant


def new_position_folds(positions, seed, n_splits=5):
    """Shuffle UNIQUE protein positions; variant keys sharing a position stay together."""
    pos = np.asarray(positions, int)
    unique = np.unique(pos)
    if len(unique) != 168:
        raise ValueError('Expected exactly 168 unique TRAIN protein positions')
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=int(seed))
    all_folds = []
    counts = np.zeros(len(pos), int)
    for fold, (_fitpos, heldpos) in enumerate(kfold.split(unique), start=1):
        held_group = unique[heldpos]
        held = np.flatnonzero(np.isin(pos, held_group))
        fit = np.flatnonzero(~np.isin(pos, held_group))
        assert len(fit) and len(held) and not set(pos[fit]) & set(pos[held])
        counts[held] += 1
        all_folds.append((fold, fit, held))
    assert np.all(counts == 1)
    return all_folds


def self_test():
    pos = np.arange(168).repeat(2)
    for seed in SEEDS[:3]:
        folds = []
        for fid, fit, held in new_position_folds(pos, seed):
            assert len(set(pos[held])) in (33,34)
            folds.append((fit,held))
        mz.verify_splits(folds, pos)
    df = pd.DataFrame({'observed_pp':[0.,10.,20.,40.],
                       'predicted_pp':[1.,9.,21.,39.], 'protein_position':[1,2,2,3]})
    assert abs(position_mae(df.observed_pp, df.predicted_pp, df.protein_position) - 1) < 1e-12
    print('PASS: Phase22A synthetic group-disjoint folds, coverage, macro metric')


def random_resplit_experiment(train, X, y, pos, n_repeats):
    print('\n' + '=' * 94, flush=True)
    print(f'E. RANDOM POSITION-SPLIT SENSITIVITY | {n_repeats} PREDECLARED 5-FOLD REPEATS',flush=True)
    print('=' * 94,flush=True)
    print('Nested: 3-fold GroupKFold selects XGB depth from frozen model-zoo candidates.',flush=True)
    print('This is TRAIN-only diagnostic resampling; frozen val/test stay unchanged.',flush=True)
    rows, selections, fold_summary = [], [], []
    for repeat, seed in enumerate(SEEDS[:n_repeats], start=1):
        begin = perf_counter()
        for fold, fit, held in new_position_folds(pos, seed):
            prm, inner_mae = mz.inner_select(X[fit], y[fit], pos[fit], 'xgboost',
                                             seed=mz.SEED + 100000 + seed*10 + fold)
            reg = mz.estimator('xgboost', prm, seed=mz.SEED + 100000 + seed*10 + fold)
            reg.fit(X[fit], y[fit])
            p = np.asarray(reg.predict(X[held]), float)
            baseline = np.full(len(held), float(np.median(y[fit])))
            selections.append({'repeat': repeat,'split_seed': seed,'fold':fold,
                               'max_depth': prm['max_depth'], 'inner_position_mae_pp':inner_mae,
                               'n_fit':len(fit),'n_held':len(held)})
            fold_summary.append({'repeat':repeat, 'split_seed':seed,'fold':fold,
                                 'held_positions':len(np.unique(pos[held])),
                                 'held_variants':len(held),
                                 'held_gt20':int((y[held]>20).sum()),
                                 'held_gt40':int((y[held]>40).sum()),
                                 'held_mean_pp':float(np.mean(y[held])),
                                 'fit_mean_pp':float(np.mean(y[fit])),
                                 'position_mae_pp':position_mae(y[held],p,pos[held])})
            for idx,pred,med in zip(held,p,baseline):
                for model, pp in [('nested_xgb',float(pred)),('median_baseline',float(med))]:
                    rows.append({'repeat':repeat,'split_seed':seed,'fold':fold,
                                 'model':model,'variant_key':str(train.variant_key.iloc[idx]),
                                 'protein_position':int(pos[idx]),'observed_pp':float(y[idx]),
                                 'predicted_pp':pp})
        df = pd.DataFrame(rows)
        m = df.loc[df.repeat.eq(repeat)]
        print(f'  repeat {repeat:02d} seed={seed:4d} '
              f'XGB pos-MAE={position_mae(m[m.model.eq("nested_xgb")].observed_pp, m[m.model.eq("nested_xgb")].predicted_pp, m[m.model.eq("nested_xgb")].protein_position):.3f} '
              f'median pos-MAE={position_mae(m[m.model.eq("median_baseline")].observed_pp, m[m.model.eq("median_baseline")].predicted_pp, m[m.model.eq("median_baseline")].protein_position):.3f} '
              f'elapsed={perf_counter()-begin:.1f}s', flush=True)

    predictions = pd.DataFrame(rows)
    assert len(predictions) == n_repeats*210*2
    assert not predictions.duplicated(['repeat','model','variant_key']).any()
    repeat_results = pd.DataFrame([
        {'repeat':rep, 'split_seed':seed, **summary_metrics(m, name='model_' + name)}
        for (rep,seed,name),m in predictions.groupby(['repeat','split_seed','model'])
    ])
    agg_rows = []
    for model, mm in predictions.groupby('model'):
        for count in sorted(set([2,n_repeats])):
            if n_repeats < count:
                continue
            a = (mm.loc[mm.repeat.le(count)].groupby('variant_key',as_index=False)
                 .agg(protein_position=('protein_position','first'),
                      observed_pp=('observed_pp','first'), predicted_pp=('predicted_pp','mean')))
            assert len(a) == 210
            agg_rows.append({'model':model,'n_repeats_averaged':count,
                             **{k:v for k,v in summary_metrics(a, name=model).items() if k!='model'}})
    aggregated = pd.DataFrame(agg_rows)
    print('\nRANDOM SPLIT REPEAT-WISE PERFORMANCE (NEVER PICK THE BEST SEED)',flush=True)
    print(repeat_results[['repeat','split_seed','model','position_mae_pp',
                          'high20_mae_pp','high40_mae_pp','high20_recall']]
          .sort_values(['model','repeat']).round(4).to_string(index=False),flush=True)
    print('\nREPEAT SENSITIVITY (descriptive variability across assigned folds)',flush=True)
    s = (repeat_results.groupby('model').position_mae_pp
         .agg(['mean','std','min','max']))
    print(s.round(4).to_string(),flush=True)
    print('\nPREDECLARED REPEAT-AVERAGING, FOR CONTEXT ONLY',flush=True)
    print(aggregated.round(4).to_string(index=False),flush=True)
    print('Note: differences among seeds are not confidence intervals, not independent test sets.',flush=True)
    print('Note: easier-looking random seeds do not justify changing the frozen test partition.',flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    paths = {'resplit_oof':OUT/'phase22a_resplit_oof.csv',
             'resplit_fold_summary':OUT/'phase22a_resplit_fold_composition.csv',
             'resplit_inner_selection':OUT/'phase22a_resplit_inner_selection.csv',
             'repeat_summary':OUT/'phase22a_repeat_results.csv',
             'averaged_summary':OUT/'phase22a_repeat_averaged_summary.csv'}
    predictions.to_csv(paths['resplit_oof'], index=False)
    pd.DataFrame(fold_summary).to_csv(paths['resplit_fold_summary'],index=False)
    pd.DataFrame(selections).to_csv(paths['resplit_inner_selection'],index=False)
    repeat_results.to_csv(paths['repeat_summary'],index=False)
    aggregated.to_csv(paths['averaged_summary'],index=False)
    print('\nRESPLIT OUTPUTS:',flush=True)
    for name,p in paths.items(): print(' ',name+':',p.relative_to(ROOT),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--resplits',type=int,default=0,
                        help='Number of prespecified 5-fold random-position repeats (0-12). 0 audits only')
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    if not 0 <= args.resplits <= len(SEEDS):
        raise ValueError(f'--resplits must be 0..{len(SEEDS)}')
    train,X,y,pos,outer,metadata=load_data()
    oof=load_frozen_oof(train,pos,y,outer)
    print('=' * 94)
    print('PHASE 22A | IS SPLIT COMPOSITION LIMITING ACUTE IVA PREDICTION?')
    print('=' * 94)
    print('VTI TRAIN outcomes only: 210 variants / 168 positions')
    print('Protected VAL/TEST/EXCLUDED: identity/position/cohort metadata ONLY; no outcomes')
    print('Primary validation/test membership and historical scores NEVER rewritten')
    metadata_audit(metadata)
    folds,novelty=frozen_audit(train,X,y,pos,outer,oof)
    OUT.mkdir(parents=True,exist_ok=True)
    paths = {'frozen_folds':OUT/'phase22a_original_fold_skew.csv',
             'novelty':OUT/'phase22a_train_difficulty_novelty.csv',
             'report':OUT/'phase22a_report.json'}
    folds.to_csv(paths['frozen_folds'],index=False)
    novelty.to_csv(paths['novelty'],index=False)
    report={'phase':'22A_frozen_train_split_shift_stress_test',
            'n_train':210,'n_positions':168,
            'random_group_seeds':list(SEEDS[:args.resplits]),
            'protected_outcomes_read':False,
            'metadata_source_sha256':sha256(SPLIT_FILE),
            'acute_train_label_sha256':sha256(TARGET_FILE),
            'historical_oof_sha256':sha256(OOF_FILE),
            'note':'Never report a cherry-picked seed as improved generalisation. Frozen val/test unchanged.'}
    paths['report'].write_text(json.dumps(report,indent=2)+'\n')
    print('\nAUDIT OUTPUTS:')
    for name,p in paths.items(): print(' ', name+':', p.relative_to(ROOT))
    if args.resplits:
        random_resplit_experiment(train,X,y,pos,args.resplits)
    print('\nPASS: TRAIN-only outcome audit and optional random grouped resplit diagnostic',flush=True)

if __name__=='__main__':
    main()
