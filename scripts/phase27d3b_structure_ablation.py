#!/usr/bin/env python3
"""Phase 27D-3B: frozen-fold 3D drug-pocket feature ablation, VTI TRAIN only.

Uses Phase 27D-3A coordinates and the previously saved Phase 27D-1 OOF
predictions. It does not fit a structural model for individual variants.
3D geometry comes from deposited CFTR conformations, not mutant structures;
8EIG contains F508del and VX-445 (proxy, not VX-121 binding coordinates).
Do not use these exploratory results to select a model on prior VAL/TEST.
"""
from __future__ import annotations
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))

OUT = ROOT / 'results/biology/phase27d'
GEO = OUT / 'phase27d3_3d_distances_by_position.csv'
PROVENANCE = OUT / 'phase27d3_structure_provenance.csv'
FROZEN = OUT / 'phase27d1_repeated_oof.csv'
KEY = ['task', 'repeat', 'fold', 'variant_key', 'protein_position']

SPECS = {
    '8EIG': {
        'dmin': '8eig_elexacaftor_VX445_min_heavy_A',
        'dca': '8eig_elexacaftor_VX445_ca_A',
        'obs': '8eig_observed',
    },
    '7SV7': {
        'dmin': '7sv7_tezacaftor_VX661_min_heavy_A',
        'dca': '7sv7_tezacaftor_VX661_ca_A',
        'obs': '7sv7_observed',
    }
}


def verify_provenance():
    if not PROVENANCE.is_file():
        raise FileNotFoundError(f'Missing Phase 27D-3A structure provenance: {PROVENANCE}')
    info = pd.read_csv(PROVENANCE)
    if info.pdb.duplicated().any() or set(info.pdb) != set(SPECS):
        raise ValueError('Unexpected PDB provenance')
    for row in info.itertuples(index=False):
        if row.canonical_agreement < .99:
            raise ValueError(f'Insufficient sequence agreement for {row.pdb}')
        if row.mismatches_canonical != 1 or '(1371,' not in str(row.first_mismatches):
            raise ValueError(f'Unexpected mismatch set for {row.pdb}; inspect mapping')
        path = ROOT / 'data/external/cftr_structures' / f'{row.pdb}.pdb'
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row.file_sha256:
            raise ValueError(f'Structure file missing or changed since Phase 27D-3A: {path}')
    return info


def load_geometry():
    if not GEO.is_file():
        raise FileNotFoundError(f'Missing Phase 27D-3A geometry: {GEO}')
    raw = pd.read_csv(GEO)
    if len(raw) != 1480 or raw.position.duplicated().any() or set(raw.position) != set(range(1, 1481)):
        raise ValueError('Expected one distance row for all 1480 positions')
    for pdb, names in SPECS.items():
        obs = raw[names['obs']].to_numpy(int)
        heavy = raw[names['dmin']].to_numpy(float)
        ca = raw[names['dca']].to_numpy(float)
        if not np.isin(obs, [0, 1]).all():
            raise ValueError(f'Non-binary coverage flags: {pdb}')
        if ((obs == 1) & (~np.isfinite(heavy) | (heavy < 0))).any():
            raise ValueError(f'Invalid heavy distances at modelled sites: {pdb}')
        if ((obs == 1) & (~np.isfinite(ca) | (ca < 0))).any():
            raise ValueError(f'Invalid CA distances at modelled sites: {pdb}')
        if ((obs == 0) & np.isfinite(heavy)).any():
            raise ValueError(f'Unmodelled residues have numeric distances: {pdb}')
    return raw.set_index('position')


def feature_sets(geometry, positions, seq_proxy, chemistry):
    """Missing distance gets zero transformed proximity plus observed=0 flag.

    Zero is not treated as a zero-Angstrom distance. Structural mismatch
    residue 1371 is withheld from BOTH structures' geometry features.
    """
    p = np.asarray(positions, int)
    if (p < 1).any() or (p > 1480).any():
        raise ValueError('Invalid canonical position')
    indexed = geometry.loc[p]
    arrays = {'heavy': [], 'ca': [], 'contacts': []}
    coverage = []
    for pdb, names in SPECS.items():
        obs = indexed[names['obs']].to_numpy(int).astype(bool) & (p != 1371)
        heavy = indexed[names['dmin']].to_numpy(float)
        ca = indexed[names['dca']].to_numpy(float)
        if ((obs) & (~np.isfinite(heavy) | ~np.isfinite(ca))).any():
            raise ValueError(f'Inconsistent distance/coverage: {pdb}')
        # 0 transformed proximity means far OR missing, disambiguated by flag.
        near_heavy = np.where(obs, np.exp(-np.nan_to_num(heavy, nan=1e5) / 10.0), 0.0)
        near_ca = np.where(obs, np.exp(-np.nan_to_num(ca, nan=1e5) / 10.0), 0.0)
        arrays['heavy'].append(near_heavy)
        arrays['ca'].append(near_ca)
        for threshold in (5., 10., 20.):
            arrays['contacts'].append((obs & (heavy < threshold)).astype(float))
        arrays['heavy'].append(obs.astype(float))
        arrays['ca'].append(obs.astype(float))
        arrays['contacts'].append(obs.astype(float))
        coverage.append(obs.astype(float))
    out = {
        '3d_heavy': np.column_stack(arrays['heavy']),
        '3d_ca': np.column_stack(arrays['ca']),
        '3d_contacts': np.column_stack(arrays['contacts']),
    }
    out['seq_plus_3d'] = np.column_stack([seq_proxy, out['3d_heavy']])
    out['chem_plus_3d'] = np.column_stack([chemistry, out['3d_heavy']])
    for name, value in out.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite encoded 3D features: {name}')
    return out, np.column_stack(coverage)


def position_region(pos, domains):
    for name, (a, b) in domains.items():
        if a <= pos <= b:
            return name
    return 'other'


def region_matched_decoy(feature, positions, domains, rng):
    """Shuffle 3D feature VECTORS by protein position within broad regions.

    Keeps same-position variants together and preserves correlations among
    8EIG/7SV7 signals, including structure-coverage indicators.
    """
    p = np.asarray(positions, int)
    if len(p) != len(feature):
        raise ValueError('Feature/position length mismatch')
    unique = np.unique(p)
    first = np.array([np.flatnonzero(p == x)[0] for x in unique])
    raw = feature[first]
    shuffled = raw.copy()
    groups = np.array([position_region(x, domains) for x in unique])
    for region in np.unique(groups):
        which = np.flatnonzero(groups == region)
        if len(which) > 1:
            shuffled[which] = raw[rng.permutation(which)]
    mapping = {int(pos): shuffled[i] for i, pos in enumerate(unique)}
    out = np.vstack([mapping[int(pos)] for pos in p])
    if out.shape != feature.shape or not np.isfinite(out).all():
        raise ValueError('Malformed structural decoy')
    # Equal positions always have equal decoy descriptors.
    for pos in unique:
        ix = np.flatnonzero(p == pos)
        if not np.allclose(out[ix], out[ix[0]]):
            raise ValueError('Decoy breaks same-position grouping')
    return out


def evaluation(df):
    if len(df) != 210 or df.variant_key.nunique() != 210 or not np.isfinite(df.score).all():
        raise ValueError('Expected complete two-repeat OOF score table')
    y = df.label.to_numpy(int)
    s = df.score.to_numpy(float)
    if np.unique(y).size != 2:
        raise ValueError('One class in pooled OOF')
    top = np.argsort(-s, kind='stable')[:42]
    return {
        'ap': float(average_precision_score(y, s)),
        'auroc': float(roc_auc_score(y, s)),
        'top42_hits': int(y[top].sum()),
        'positives': int(y.sum())
    }


def self_test():
    dummy = pd.DataFrame({'position': np.arange(1, 6)})
    for names in SPECS.values():
        dummy[names['obs']] = [1, 0, 1, 1, 0]
        dummy[names['dmin']] = [3., np.nan, 15., 40., np.nan]
        dummy[names['dca']] = [6., np.nan, 19., 45., np.nan]
    dummy = dummy.set_index('position')
    pp = np.array([1, 1, 2, 3, 4, 5])
    ff, coverage = feature_sets(dummy, pp, np.zeros((6, 6)), np.zeros((6, 11)))
    assert ff['3d_heavy'].shape == (6, 4)
    assert ff['3d_contacts'].shape == (6, 8)
    assert ff['seq_plus_3d'].shape == (6, 10)
    assert np.allclose(ff['3d_heavy'][0], ff['3d_heavy'][1])
    assert np.allclose(ff['3d_heavy'][2], [0., 0., 0., 0.])
    assert np.array_equal(coverage[2], [0, 0])
    dec = region_matched_decoy(ff['3d_heavy'], pp,
                               {'first': (1, 3), 'second': (4, 5)},
                               np.random.default_rng(27))
    assert dec.shape == ff['3d_heavy'].shape
    assert np.allclose(dec[0], dec[1])
    print('PASS synthetic: missing geometry, proximity transforms, grouped region decoys')


def run(n_decoys, nboot):
    from cftr_theratype.datasets.eti_auxiliary import load_data as load_eti_train
    import phase27b1_extreme_classifiers as b
    import phase27d1_biology_ablation as d1

    verify_provenance()
    geometry = load_geometry()
    if not FROZEN.is_file():
        raise FileNotFoundError(f'Run Phase 27D-1 first: {FROZEN}')
    v, x80, _, _, folds = load_eti_train(ROOT)
    x80 = np.asarray(x80, float)
    pp = v.protein_position.to_numpy(int)
    keys = v.variant_key.astype(str).to_numpy()
    yy = v.corrector_rescue_pp.to_numpy(float)
    if x80.shape != (210, 80) or len(np.unique(pp)) != 168 or len(folds) != 10:
        raise ValueError('Unexpected VTI TRAIN or frozen folds')
    if not np.isfinite(x80[:, :68]).all():
        raise ValueError('Nonfinite sequence descriptors')
    code = d1.extract_codes(v, pp)
    annotation = d1.annotation_features(pp, code)
    feat, cover = feature_sets(geometry, pp,
                               annotation['pocket_proxy'].to_numpy(float),
                               annotation['chemistry'].to_numpy(float))
    frozen = pd.read_csv(FROZEN, dtype={'variant_key': str})
    frozen = frozen[frozen.config.isin(['baseline','pocket_proxy','chemistry'])].copy()
    if len(frozen) != 3 * 2 * 420 or frozen.duplicated(KEY + ['config']).any():
        raise ValueError('Incomplete frozen Phase 27D-1 reference OOF')
    if not np.isfinite(frozen.score).all():
        raise ValueError('Nonfinite frozen OOF')

    # True coordinates and decoys are independent of response labels.
    rng = np.random.default_rng(273003)
    controls = {f'decoy_{k+1:03d}': region_matched_decoy(
        feat['3d_heavy'], pp, d1.DOMAINS, rng) for k in range(n_decoys)}
    tasks = [
        ('corrector_high', (yy > 40).astype(int), x80[:, :68], 'logistic_seq68',
         ['3d_heavy', '3d_ca', '3d_contacts', 'seq_plus_3d'], controls),
        ('corrector_low', (yy < 10).astype(int), x80, 'xgb_seq80',
         ['3d_heavy', '3d_contacts', 'chem_plus_3d'], {}),
    ]
    print('=' * 102)
    print('PHASE 27D-3B | EXPERIMENTAL 3D DRUG-POCKET FEATURE ABLATION')
    print('=' * 102)
    print('TRAIN only: 210 mutations / 168 positions, frozen original 2x5 grouped folds')
    print('High-responder PRIMARY: 3d_heavy vs published sequence pocket_proxy and baseline')
    print('Other comparisons and matched 3D decoys are exploratory controls')
    print('8EIG = elexacaftor / F508del structure; NOT a VX-121 or variant-specific structure')
    print('7SV7 = tezacaftor structure; geometries are from deposited conformations')
    print('Residue 1371: both structures have Q rather than canonical E; geometry masked')
    print('Unmodelled residues: proximity encoded as 0 WITH observed flag = 0 (not distance 0 A)')
    print('Protected VAL / TEST response outcomes unopened')
    print('\nA. STRUCTURAL COVERAGE IN VTI TRAIN')
    for j, pdb in enumerate(SPECS):
        print(f'{pdb}: observed variants={int(cover[:,j].sum())}/210; '
              f'observed positions={len(np.unique(pp[cover[:,j] == 1]))}/168; '
              f'contacts<5A={int((feat["3d_contacts"][:, 4*j] == 1).sum())}')
    print('Modelled in both: ', int((cover.sum(axis=1)==2).sum()), '/210')
    print('Unmodelled in both:', int((cover.sum(axis=1)==0).sum()), '/210')

    rows = []
    expected = {}
    for ti, (task, labels, base, method, main_sets, ctrl) in enumerate(tasks):
        expected[task] = set(main_sets) | set(ctrl)
        ref = frozen.loc[frozen.task == task]
        check = ref[ref.config == 'baseline'].set_index(['repeat','fold','variant_key'])
        if labels.sum() != (29 if task == 'corrector_high' else 52):
            raise ValueError('Responder counts changed')
        for fold_id, (repeat, fold, train_keys, test_keys) in enumerate(folds, 1):
            tr = np.flatnonzero(np.isin(keys, list(train_keys)))
            te = np.flatnonzero(np.isin(keys, list(test_keys)))
            if len(tr) + len(te) != 210 or set(pp[tr]) & set(pp[te]):
                raise ValueError(f'Protein-position leakage in fold {fold_id}')
            old = check.loc[[(repeat,fold,key) for key in keys[te]],:]
            if not np.array_equal(old.label.to_numpy(int), labels[te]):
                raise ValueError('Frozen OOF label alignment failure')
            if len(np.unique(labels[tr])) != 2:
                raise ValueError('One-class training fold')
            for name in sorted(expected[task]):
                f = feat[name] if name in feat else ctrl[name]
                XX = np.column_stack([base, f])
                if task == 'corrector_high' and not np.isfinite(XX).all():
                    raise ValueError('Nonfinite logistic input features')
                mdl = b.make_model(method, labels[tr],
                                   27000 + (100 if task == 'corrector_high' else 0) + fold_id)
                mdl.fit(XX[tr], labels[tr])
                score = mdl.predict_proba(XX[te])[:, 1]
                if not np.isfinite(score).all():
                    raise ValueError(f'Nonfinite OOF scores: {name}')
                rows.extend({
                    'task':task, 'repeat':int(repeat), 'fold':int(fold),
                    'variant_key':keys[ix], 'protein_position':int(pp[ix]),
                    'label':int(labels[ix]), 'config':name, 'score':float(score[j])
                } for j,ix in enumerate(te))
            print(f'{task}: completed fold {fold_id}/10 ({len(expected[task])} feature sets)', flush=True)
    pred = pd.DataFrame(rows)
    if pred.duplicated(KEY + ['config']).any():
        raise ValueError('Duplicate OOF prediction')
    for task, names in expected.items():
        for name in names:
            if len(pred[(pred.task == task) & (pred.config == name)]) != 420:
                raise ValueError(f'Incomplete OOF: {task}, {name}')
    combined = pd.concat([frozen, pred], ignore_index=True)
    if combined.duplicated(KEY + ['config']).any():
        raise ValueError('Duplicate combined frozen OOF key')
    avg = combined.groupby(['task', 'config', 'variant_key', 'protein_position'],
                           as_index=False).agg(label=('label','first'), score=('score','mean'))
    metrics = pd.DataFrame([{'task': task, 'config': cfg, **evaluation(g)}
                            for (task,cfg),g in avg.groupby(['task','config'])])
    metrics = metrics.sort_values(['task','ap'], ascending=[True,False])
    print('\nB. SAME-FOLD AVERAGED OOF CLASSIFIER COMPARISON')
    print(metrics[~metrics.config.str.startswith('decoy_')].round(4).to_string(index=False))
    
    contrasts = []
    for task in ['corrector_high','corrector_low']:
        part = avg.loc[avg.task == task]
        wide = part.pivot(index=['variant_key','protein_position','label'],
                          columns='config',values='score').reset_index()
        if len(wide) != 210 or wide.isna().any().any():
            raise ValueError('Incomplete paired OOF table')
        if task == 'corrector_high':
            comparisons = [('3d_heavy','pocket_proxy'),('3d_heavy','baseline'),
                           ('seq_plus_3d','pocket_proxy'),('3d_contacts','baseline')]
        else:
            comparisons = [('3d_heavy','baseline'),('chem_plus_3d','chemistry'),
                           ('3d_contacts','baseline')]
        for j,(candidate, reference) in enumerate(comparisons):
            if candidate not in wide.columns:
                raise ValueError(f'Missing candidate {candidate}')
            res = b.ap_gain_bootstrap(wide, reference, candidate, nboot,
                                       seed=273600 + (100 if task=='corrector_high' else 0) + j)
            contrasts.append({'task':task, **res})
    gains = pd.DataFrame(contrasts)
    print('\nC. PAIRED POSITION-CLUSTER AP DIFFERENCES (positive favours candidate)')
    print(gains.round(4).to_string(index=False))
    high_metrics = metrics[metrics.task=='corrector_high'].set_index('config')
    null = high_metrics.loc[[n for n in controls]].ap
    real = float(high_metrics.loc['3d_heavy','ap'])
    print('\nD. REGION-MATCHED STRUCTURAL DECOYS (descriptive, NOT a p-value)')
    print(f'Decoys n={len(null)} median AP={null.median():.4f}, '
          f'p05={null.quantile(.05):.4f}, p95={null.quantile(.95):.4f}, '
          f'max={null.max():.4f}')
    print(f'Real 3D heavy proximity AP={real:.4f}; decoys >= real: {(null >= real).sum()}/{len(null)}')

    OUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT / 'phase27d3b_repeated_oof.csv',index=False)
    avg.to_csv(OUT / 'phase27d3b_averaged_oof.csv',index=False)
    metrics.to_csv(OUT / 'phase27d3b_metrics.csv',index=False)
    gains.to_csv(OUT / 'phase27d3b_ap_gains.csv',index=False)
    coverage_output = pd.DataFrame({'variant_key':keys, 'protein_position':pp,
                                    'observed_8EIG':cover[:,0].astype(int),
                                    'observed_7SV7':cover[:,1].astype(int)})
    coverage_output.to_csv(OUT / 'phase27d3b_structural_coverage_train.csv',index=False)
    print('\nPASS: Phase 27D-3B complete; VTI TRAIN only, unchanged outer folds.')
    print('These are exploratory comparisons on repeatedly reused development folds.')
    print('Saved:', OUT)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--decoys',type=int,default=20)
    p.add_argument('--bootstraps',type=int,default=2000)
    args=p.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.decoys < 5 or args.bootstraps < 100:
            p.error('Require --decoys >= 5 and --bootstraps >= 100')
        run(args.decoys,args.bootstraps)
