#!/usr/bin/env python3
"""Phase 27D-4B: ESM-2 ablation for CFTR responder ranking AND rescue MAE.

VTI TRAIN only. Uses the original two-repeat, five-fold position-disjoint splits.
ESM-2 is frozen. All PCA/scaling of embeddings is fitted WITHIN each outer
training fold. The previous continuous and classifier OOF results are retained
as references; no validation or final-test outcome is accessed.

These are exploratory development results on reused folds, not independent
confirmation. 8EIG contains F508del and is a structural proxy, not mutant-specific.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'results/biology/phase27d'
ESM_SCALAR = OUT / 'phase27d4a_esm2_scalar_features.csv'
ESM_DELTA = OUT / 'phase27d4a_esm2_delta_embeddings.npz'
REF_CLASS = ROOT / 'results/biology/phase27b/phase27b1_repeated_oof.csv'
REF_HIGH = OUT / 'phase27d3b_repeated_oof.csv'
REF_LOW = OUT / 'phase27d1_repeated_oof.csv'
REF_REGR = ROOT / 'results/biology/phase26h2/phase26h2_oof.csv'
KEY = ['repeat', 'fold', 'variant_key', 'protein_position']
SCALARS = ['esm2_llr', 'esm2_mask_entropy', 'esm2_delta_l2', 'esm2_site_cosine']
CONFIGS = ['esm_only', 'baseline_scalar', 'baseline_pca8',
           'bio_scalar', 'bio_pca8']
REG_CONFIGS = ['xgb80', 'xgb80_scalar', 'xgb80_pca8']


def position_mae(d: pd.DataFrame, actual: str, pred: str) -> float:
    t = d[['protein_position', actual, pred]].copy()
    t['ae'] = (t[actual] - t[pred]).abs()
    return float(t.groupby('protein_position').ae.mean().mean())


def ap(y, score):
    return float(average_precision_score(np.asarray(y, int), np.asarray(score, float)))


def rank_metrics(d):
    y = d.label.to_numpy(int)
    s = d.score.to_numpy(float)
    if len(d) != 210 or np.unique(y).size != 2 or not np.isfinite(s).all():
        raise ValueError('Invalid pooled rank evaluation')
    hits = int(y[np.argsort(-s, kind='stable')[:42]].sum())
    return {'ap': ap(y, s), 'auroc': float(roc_auc_score(y, s)),
            'top42_hits': hits, 'positives': int(y.sum())}


def boot_ap_gain(d: pd.DataFrame, candidate: str, reference: str,
                 nboot: int, seed: int):
    q = d.pivot(index=['variant_key', 'protein_position', 'label'],
                columns='config', values='score').reset_index()
    if len(q) != 210 or q[[candidate, reference]].isna().any().any():
        raise ValueError('Incomplete AP comparison')
    pos = q.protein_position.to_numpy(int)
    groups = [np.flatnonzero(pos == p) for p in np.unique(pos)]
    y = q.label.to_numpy(int)
    a, b = q[candidate].to_numpy(float), q[reference].to_numpy(float)
    observed = ap(y, a) - ap(y, b)
    rng = np.random.default_rng(seed)
    draw = []
    for _ in range(nboot):
        ii = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        if len(np.unique(y[ii])) == 2:
            draw.append(ap(y[ii], a[ii]) - ap(y[ii], b[ii]))
    if len(draw) < .95 * nboot:
        raise ValueError('Too many single-class bootstrap samples')
    return {'candidate': candidate, 'reference': reference, 'ap_gain': observed,
            'ci95_low': float(np.quantile(draw, .025)),
            'ci95_high': float(np.quantile(draw, .975)),
            'positions': len(groups)}


def boot_mae_gain(d: pd.DataFrame, new: str, old: str,
                  label: str, nboot: int, seed: int):
    # Aggregate repeat-wise absolute errors within each position, then average repeats.
    z = d[['repeat', 'protein_position', new, old, label]].copy()
    z['gain'] = (z[old]-z[label]).abs() - (z[new]-z[label]).abs()
    v = (z.groupby(['repeat', 'protein_position']).gain.mean()
         .groupby('protein_position').mean().to_numpy(float))
    rng = np.random.default_rng(seed)
    ix = rng.integers(len(v), size=(nboot, len(v)))
    vals = v[ix].mean(axis=1)
    return {'candidate': new, 'reference': old, 'target': label,
            'mae_gain_pp': float(v.mean()), 'ci95_low': float(np.quantile(vals, .025)),
            'ci95_high': float(np.quantile(vals, .975)), 'positions': len(v)}


def load_esm(mutations: list[str]):
    if not ESM_SCALAR.is_file() or not ESM_DELTA.is_file():
        raise FileNotFoundError('Run Phase 27D-4A before the ESM benchmark.')
    scalar = pd.read_csv(ESM_SCALAR, dtype={'mutation': str})
    if scalar.mutation.duplicated().any():
        raise ValueError('Duplicate scalar mutation keys')
    if not set(SCALARS).issubset(scalar):
        raise ValueError('Missing ESM scalar columns')
    with np.load(ESM_DELTA, allow_pickle=False) as archive:
        codes = [str(x) for x in archive['mutation']]
        vectors = np.asarray(archive['delta'], float)
    if vectors.shape != (len(codes), 480) or len(set(codes)) != len(codes):
        raise ValueError('Expected unique ESM-2 480-dim mutation embeddings')
    if set(scalar.mutation) != set(codes) or set(mutations) != set(codes):
        raise ValueError('ESM mutation codes do not match exactly the TRAIN cohort')
    if len(codes) != 206:
        raise ValueError('Unexpected ESM unique substitution count')
    idx = scalar.set_index('mutation').loc[mutations]
    scalars = idx[SCALARS].to_numpy(float)
    by_mut = {c: v for c, v in zip(codes, vectors)}
    delta = np.vstack([by_mut[m] for m in mutations])
    if delta.shape != (210, 480) or not np.isfinite(scalars).all() or not np.isfinite(delta).all():
        raise ValueError('Incomplete or nonfinite ESM rows')
    return scalars, delta


def fold_pca(delta: np.ndarray, tr: np.ndarray, te: np.ndarray, k: int=8):
    # No unsupervised preprocessing on held-out protein positions.
    scale = StandardScaler().fit(delta[tr])
    ztr = scale.transform(delta[tr])
    zte = scale.transform(delta[te])
    pca = PCA(n_components=k, svd_solver='randomized', random_state=27).fit(ztr)
    atr, ate = pca.transform(ztr), pca.transform(zte)
    if not np.isfinite(atr).all() or not np.isfinite(ate).all():
        raise ValueError('Bad fold-local PCA')
    return atr, ate, float(np.sum(pca.explained_variance_ratio_))


def load_reference(path, filters, expected_count):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    q = pd.read_csv(path, dtype={'variant_key':str})
    for col, val in filters.items():
        if col not in q:
            raise ValueError(f'Missing reference column {col}: {path}')
        q = q[q[col].eq(val)]
    if len(q) != expected_count or q.duplicated(KEY).any():
        raise ValueError(f'Unexpected reference OOF cardinality: {path}: {len(q)}')
    return q


def self_test():
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(27)
    pos = np.repeat(np.arange(20), 2)
    X = rng.normal(size=(40, 480))
    tr, te = next(GroupKFold(4).split(X, groups=pos))
    a, b, explained = fold_pca(X, tr, te)
    assert a.shape == (len(tr), 8) and b.shape == (len(te), 8)
    assert 0 < explained < 1
    y = np.array([0,1]*20)
    q = pd.DataFrame({'variant_key': [str(i) for i in range(40)],
                      'protein_position': pos, 'label':y,
                      'config': ['a']*40, 'score':rng.random(40)})
    assert np.isfinite(q.score).all()
    dd = pd.DataFrame({'repeat':[1,1,2,2], 'protein_position':[1,2,1,2],
                       'y':[1,4,1,4], 'base':[2,3,2,3], 'new':[1,4,1,4]})
    result=boot_mae_gain(dd, 'new','base','y',200,27)
    assert np.isclose(result['mae_gain_pp'], 1.0)
    print('PASS synthetic: fold-only PCA, mutation mapping shape, paired position bootstrap')


def run(nboot):
    import phase26h_auxiliary_corrector as h
    import phase27b1_extreme_classifiers as b
    import phase27d1_biology_ablation as d1
    import phase27d3b_structure_ablation as d3

    v, x80, _, _, folds = h.load_data()
    x80 = np.asarray(x80, float)
    pp = v.protein_position.to_numpy(int)
    keys = v.variant_key.astype(str).to_numpy()
    yc = v.corrector_rescue_pp.to_numpy(float)
    yt = v.vti_rescue_pp.to_numpy(float)
    if x80.shape != (210, 80) or len(np.unique(pp)) != 168 or len(folds) != 10 or len(set(keys)) != 210:
        raise ValueError('Unexpected TRAIN cohort or CV folds')
    if not np.isfinite(x80[:, :68]).all() or not np.isfinite(yc).all() or not np.isfinite(yt).all():
        raise ValueError('Nonfinite sequence features or TRAIN outcomes')
    mutations = d1.extract_codes(v, pp)
    esm_scalar, esm_delta = load_esm(mutations)
    annotations = d1.annotation_features(pp, mutations)
    pocket = annotations['pocket_proxy'].to_numpy(float)
    chemistry = annotations['chemistry'].to_numpy(float)
    d3.verify_provenance()
    bio, _ = d3.feature_sets(d3.load_geometry(), pp, pocket, chemistry)

    original = load_reference(REF_CLASS, {'task':'corrector_high'}, 420)
    original_low = load_reference(REF_CLASS, {'task':'corrector_low'}, 420)
    high_bio = load_reference(REF_HIGH, {'task':'corrector_high','config':'seq_plus_3d'}, 420)
    low_bio = load_reference(REF_LOW, {'task':'corrector_low','config':'chemistry'}, 420)
    ref_reg = load_reference(REF_REGR, {}, 420)
    required = {'vti_fixed','vti_fixed_total','observed_corrector','observed_total'}
    if not required.issubset(ref_reg):
        raise ValueError(f'Missing Phase26H2 regression columns: {sorted(required - set(ref_reg))}')

    # Check all saved sources share EXACTLY the same fold-membership, labels and positions.
    membership = pd.MultiIndex.from_frame(ref_reg[KEY])
    for frame, name in [(original,'27B1 high'),(original_low,'27B1 low'),
                        (high_bio,'27D3B high'),(low_bio,'27D1 low')]:
        if set(pd.MultiIndex.from_frame(frame[KEY])) != set(membership):
            raise ValueError(f'Frozen fold membership mismatch: {name}')
    previous = ref_reg.set_index(KEY)
    prev_classifier = {
        'corrector_high': original.set_index(KEY),
        'corrector_low': original_low.set_index(KEY),
    }
    prior_bio = {
        'corrector_high': high_bio.set_index(KEY),
        'corrector_low': low_bio.set_index(KEY),
    }

    print('=' * 110)
    print('PHASE 27D-4B | FROZEN ESM-2: RESPONDER AP AND CONTINUOUS RESCUE MAE')
    print('=' * 110)
    print('VTI TRAIN 210 variants / 168 positions / 206 unique protein substitutions; 2x5 original held-out-position folds')
    print('ESM scalars=4, delta embeddings=480, PCA=8 FIT WITHIN outer training fold only')
    print('Classifiers: high logistic (seq68); low XGB (seq80); best prior bio controls also retained')
    print('Regression: original fixed-spec XGB corrector; total reconstructed with the frozen acute component')
    print('No VAL or TEST labels read; no selection using protected data')

    classification = []
    regression = []
    qc = []
    for fi, (repeat, fold, train_keys, test_keys) in enumerate(folds,1):
        tr = np.flatnonzero(np.isin(keys, list(train_keys)))
        te = np.flatnonzero(np.isin(keys, list(test_keys)))
        if len(tr)+len(te) != 210 or set(pp[tr]) & set(pp[te]):
            raise ValueError('Outer fold position leakage')
        ixs = [(int(repeat), int(fold), keys[j], int(pp[j])) for j in te]
        reference = previous.loc[ixs]
        if not np.allclose(reference.observed_corrector.to_numpy(float), yc[te], atol=1e-6):
            raise ValueError('Saved regression label mismatch')
        if not np.allclose(reference.observed_total.to_numpy(float), yt[te], atol=1e-6):
            raise ValueError('Saved total label mismatch')
        pca_tr, pca_te, fraction_var = fold_pca(esm_delta, tr, te, 8)
        z_base = x80[:, :68]
        # Python arrays below are fold-scoped; no PCA fitted on held-out variants.
        for task, y, base_all, bio_all, method in [
            ('corrector_high', (yc > 40).astype(int), z_base, bio['seq_plus_3d'], 'logistic_seq68'),
            ('corrector_low', (yc < 10).astype(int), x80, chemistry, 'xgb_seq80'),
        ]:
            old = prev_classifier[task].loc[ixs]
            oldbio = prior_bio[task].loc[ixs]
            if not np.array_equal(old.label.to_numpy(int), y[te]) or not np.array_equal(oldbio.label.to_numpy(int), y[te]):
                raise ValueError('Saved classification label mismatch')
            old_name = 'logistic_seq68' if task=='corrector_high' else 'xgb_seq80'
            scores = {
                'baseline': old[old_name].to_numpy(float),
                'best_bio': oldbio.score.to_numpy(float),
            }
            configs = {
                'esm_only': (np.column_stack([esm_scalar[tr], pca_tr]),
                             np.column_stack([esm_scalar[te], pca_te])),
                'baseline_scalar': (np.column_stack([base_all[tr], esm_scalar[tr]]),
                                    np.column_stack([base_all[te], esm_scalar[te]])),
                'baseline_pca8': (np.column_stack([base_all[tr], esm_scalar[tr], pca_tr]),
                                  np.column_stack([base_all[te], esm_scalar[te], pca_te])),
                'bio_scalar': (np.column_stack([base_all[tr], bio_all[tr], esm_scalar[tr]]),
                               np.column_stack([base_all[te], bio_all[te], esm_scalar[te]])),
                'bio_pca8': (np.column_stack([base_all[tr], bio_all[tr], esm_scalar[tr], pca_tr]),
                             np.column_stack([base_all[te], bio_all[te], esm_scalar[te], pca_te])),
            }
            for k, (atr, ate) in configs.items():
                if task=='corrector_high' and not np.isfinite(atr).all():
                    raise ValueError('Nonfinite logistic features')
                mdl = b.make_model(method, y[tr], 27400 + fi*10 + (0 if task=='corrector_high' else 1))
                mdl.fit(atr, y[tr])
                scores[k] = mdl.predict_proba(ate)[:,1].astype(float)
            for j, idx in enumerate(te):
                for config, score in scores.items():
                    classification.append({'repeat':int(repeat),'fold':int(fold),
                         'variant_key':keys[idx],'protein_position':int(pp[idx]),
                         'task':task, 'label':int(y[idx]), 'config':config,'score':float(score[j])})

        # Regression: compare ESM additions to the SAME newly-fitted XGB baseline.
        # The original Phase26H2 regression is separately reported as frozen reference.
        A_base_tr = np.column_stack([x80[tr], np.ones(len(tr))])
        A_base_te = np.column_stack([x80[te], np.ones(len(te))])
        A_scalar_tr = np.column_stack([x80[tr], esm_scalar[tr], np.ones(len(tr))])
        A_scalar_te = np.column_stack([x80[te], esm_scalar[te], np.ones(len(te))])
        A_pca_tr = np.column_stack([x80[tr], esm_scalar[tr], pca_tr, np.ones(len(tr))])
        A_pca_te = np.column_stack([x80[te], esm_scalar[te], pca_te, np.ones(len(te))])
        Xreg = {'xgb80':(A_base_tr,A_base_te),
                'xgb80_scalar':(A_scalar_tr,A_scalar_te),
                'xgb80_pca8':(A_pca_tr,A_pca_te)}
        reg_pred = {}
        for name, (atr,ate) in Xreg.items():
            mdl = h.previous.model(h.SPEC)
            mdl.fit(atr, yc[tr])
            reg_pred[name] = np.asarray(mdl.predict(ate), float)
            if len(reg_pred[name]) != len(te) or not np.isfinite(reg_pred[name]).all():
                raise ValueError(f'Nonfinite regression predictions: {name}')
        frozen_corrector = reference.vti_fixed.to_numpy(float)
        frozen_total = reference.vti_fixed_total.to_numpy(float)
        frozen_acute = frozen_total - frozen_corrector
        for j, idx in enumerate(te):
            row = {'repeat':int(repeat),'fold':int(fold), 'variant_key':keys[idx],
                   'protein_position':int(pp[idx]), 'observed_corrector':float(yc[idx]),
                   'observed_total':float(yt[idx]),
                   'corrector_frozen':float(frozen_corrector[j]),
                   'total_frozen':float(frozen_total[j])}
            for name, pred in reg_pred.items():
                row['corrector_'+name] = float(pred[j])
                row['total_'+name] = float(pred[j] + frozen_acute[j])
            regression.append(row)
        qc.append({'repeat':int(repeat),'fold':int(fold), 'n_train':len(tr),
                   'n_test':len(te), 'train_positions':len(np.unique(pp[tr])),
                   'test_positions':len(np.unique(pp[te])),
                   'fold_pca_variance_fraction_8':fraction_var,
                   'refit_vs_frozen_corrector_max_abs':float(np.max(np.abs(reg_pred['xgb80'] - frozen_corrector)))})
        print(f'completed {fi}/10 repeat={repeat} fold={fold} held-out positions={len(np.unique(pp[te]))} PCA8_var={fraction_var:.3f}', flush=True)

    cd = pd.DataFrame(classification)
    rd = pd.DataFrame(regression)
    if len(cd) != 2*420*7 or cd.duplicated(KEY+['task','config']).any():
        raise ValueError('Incomplete classification OOF')
    if len(rd) != 420 or rd.duplicated(KEY).any():
        raise ValueError('Incomplete regression OOF')
    # Averages across two repeat predictions for ranking. Errors for regression
    # are averaged per position and repeat, not by pretending 420 independent rows.
    avg = cd.groupby(['task','config','variant_key','protein_position','label'],as_index=False).score.mean()
    ms = pd.DataFrame([{'task':task,'config':config,**rank_metrics(part)}
                       for (task,config),part in avg.groupby(['task','config'])]).sort_values(['task','ap'],ascending=[True,False])
    gain_ap = []
    for ti,task in enumerate(['corrector_high','corrector_low']):
        z=avg.loc[avg.task==task]
        for i, (name, ref) in enumerate([('baseline_scalar','baseline'),('baseline_pca8','baseline'),
                                         ('bio_scalar','best_bio'),('bio_pca8','best_bio')]):
            gain_ap.append({'task':task,**boot_ap_gain(z,name,ref,nboot,seed=27401+ti*20+i)})
    mae=[]
    for target in ['corrector','total']:
        actual='observed_'+target
        for name in ['frozen']+REG_CONFIGS:
            pred=target+'_'+name
            by_rep=[position_mae(z,actual,pred) for _,z in rd.groupby('repeat')]
            bias=float(np.mean(rd[pred]-rd[actual]))
            mae.append({'target':target,'config':name,'position_mae_pp':float(np.mean(by_rep)),
                        'variant_mae_pp':float(np.mean(np.abs(rd[pred]-rd[actual]))),
                        'bias_pp':bias,'rmse_pp':float(np.sqrt(np.mean((rd[pred]-rd[actual])**2)))})
    gains_mae=[]
    for i, target in enumerate(['corrector','total']):
        actual='observed_'+target
        for j, (name, ref) in enumerate([('xgb80_scalar','xgb80'),('xgb80_pca8','xgb80'),
                                          ('xgb80_scalar','frozen'),('xgb80_pca8','frozen')]):
            gains_mae.append(boot_mae_gain(rd,target+'_'+name,target+'_'+ref,actual,
                                          nboot,seed=27430+i*10+j))

    OUT.mkdir(parents=True,exist_ok=True)
    cd.to_csv(OUT/'phase27d4b_classifier_oof.csv',index=False)
    avg.to_csv(OUT/'phase27d4b_classifier_averaged_oof.csv',index=False)
    ms.to_csv(OUT/'phase27d4b_classifier_metrics.csv',index=False)
    pd.DataFrame(gain_ap).to_csv(OUT/'phase27d4b_classifier_ap_gains.csv',index=False)
    rd.to_csv(OUT/'phase27d4b_regression_oof.csv',index=False)
    pd.DataFrame(mae).to_csv(OUT/'phase27d4b_regression_mae.csv',index=False)
    pd.DataFrame(gains_mae).to_csv(OUT/'phase27d4b_regression_mae_gains.csv',index=False)
    pd.DataFrame(qc).to_csv(OUT/'phase27d4b_fold_qc.csv',index=False)

    print('\nA. ESM-2 RESPONDER RANKING: TWO-REPEAT AVERAGED OOF')
    print(ms.round(4).to_string(index=False))
    print('\nB. CLASSIFIER AP GAINS: POSITION-CLUSTER BOOTSTRAP (positive = improvement)')
    print(pd.DataFrame(gain_ap).round(4).to_string(index=False))
    print('\nC. CONTINUOUS RESCUE MAE: PAIRED OUTER-FOLD EVALUATION')
    print(pd.DataFrame(mae).round(4).to_string(index=False))
    print('\nD. MAE GAINS: POSITION-CLUSTER BOOTSTRAP (positive = LOWER MAE)')
    print(pd.DataFrame(gains_mae).round(4).to_string(index=False))
    print('\nE. REGRESSION REFIT / PCA QC')
    print(pd.DataFrame(qc).round(4).to_string(index=False))
    print('\nINTERPRETATION: ESM features cannot be assumed to help. Refit baseline differs from frozen baseline')
    print('if the underlying implementation/seed differs. All comparisons are exploratory TRAIN development only.')
    print('Total = corrector prediction + FROZEN acute prediction, so any change is isolated to corrector features.')
    print('Never select or re-fit models using the previously evaluated final test labels.')
    print('PASS: Phase27D-4B complete | TRAIN only | Results:',OUT)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--bootstraps',type=int,default=2000)
    args=parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.bootstraps<100:
            parser.error('--bootstraps must be >=100')
        run(args.bootstraps)
