#!/usr/bin/env python3
"""Phase 27D-1: fixed-fold biological feature ablation, VTI TRAIN only.

Published UniProt CFTR domain and transmembrane annotations are sequence intervals.
Published corrector residues are used for sequence-distance proxy features only:
NOT physical/3D distance, NOT confirmed VX-121 contact sites.
No ETI/VTI VAL/TEST outcomes or labels are loaded; no external downloads.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
OUT = ROOT / 'results/biology/phase27d'
PRIOR = ROOT / 'results/biology/phase27b/phase27b1_repeated_oof.csv'

# UniProt P13569: https://www.uniprot.org/uniprotkb/P13569/entry
DOMAINS = {
    'uniprot_TMD1': (81, 365),
    'uniprot_NBD1_core': (423, 646),
    'uniprot_R_region': (654, 831),
    'uniprot_TMD2': (859, 1155),
    'uniprot_NBD2_core': (1210, 1443),
}
TM_HELICES = [
    (78, 98), (123, 146), (196, 216), (223, 243),
    (299, 319), (340, 358), (859, 879), (919, 939),
    (991, 1011), (1014, 1034), (1096, 1116), (1131, 1151),
]
# Published tezacaftor/lumacaftor TMD1 site and elexacaftor contacts.
# These are SEQUENCE-DISTANCE PROXIES, not structural Euclidean distances.
CONTACT_SETS = {
    'typeI_corrector_seq': [68, 74, 195, 198, 364],
    'VX445_corrector_seq': [18, 21, 1098, 1102],
}

# Standard simple residue descriptors. No new experimental labels.
HYDROPATHY = dict(zip(
    'ACDEFGHIKLMNPQRSTVWY',
    [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
     1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]
))
CHARGE = {aa: (1 if aa in 'KR' else -1 if aa in 'DE' else 0)
          for aa in HYDROPATHY}
VOLUME = dict(zip(
    'ACDEFGHIKLMNPQRSTVWY',
    [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7,
     168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0,
     116.1, 140.0, 227.8, 193.6]
))


def annotation_features(positions, codes):
    from cftr_theratype.features.biology_annotations import annotation_features as shared
    return shared(positions, codes)

def extract_codes(frame, pp):
    """Legacy API: shared strict metadata/position source selection."""
    from cftr_theratype.features.mutation_codes import extract_codes as shared
    return shared(frame, pp)


def evaluation(y, scores):
    y = np.asarray(y, int)
    s = np.asarray(scores, float)
    if len(y) != 210 or len(np.unique(y)) != 2 or not np.isfinite(s).all():
        raise ValueError('Invalid OOF scores/labels')
    n = 42
    top = np.argsort(-s, kind='stable')[:n]
    return {
        'ap': float(average_precision_score(y, s)),
        'auroc': float(roc_auc_score(y, s)),
        'top20_hits': int(y[top].sum()),
        'positives': int(y.sum()),
    }


def run(nboot):
    from cftr_theratype.datasets.eti_auxiliary import load_data
    from cftr_theratype.models.responder_classifiers import make_model
    from cftr_theratype.evaluation.responder_classification import ap_gain_bootstrap
    v, X80, _, _, folds = load_data(ROOT)
    X80 = np.asarray(X80, float)
    pp = v.protein_position.to_numpy(int)
    keys = v.variant_key.astype(str).to_numpy()
    yc = v.corrector_rescue_pp.to_numpy(float)
    if X80.shape != (210, 80) or len(np.unique(pp)) != 168:
        raise ValueError('Unexpected frozen VTI train data')
    if not np.isfinite(X80[:, :68]).all():
        raise ValueError('Nonfinite baseline sequence features')
    codes = extract_codes(v, pp)
    groups = annotation_features(pp, codes)
    if not PRIOR.exists():
        raise FileNotFoundError(f'Missing frozen specialist OOF: {PRIOR}')
    previous = pd.read_csv(PRIOR, dtype={'variant_key': str})
    assert len(previous) == 1680
    print('=' * 108)
    print('PHASE 27D-1 | MECHANISTIC FEATURES VS FROZEN RESPONDER SPECIALISTS')
    print('=' * 108)
    print('VTI TRAIN only: 210 variants / 168 positions; 2x5 original outer folds')
    print('No hyperparameter tuning or protected VAL/TEST label access')
    print('Annotations: UniProt P13569 domains + TM helices; published corrector site sequence proxies')
    print('Contact features are SEQUENCE DISTANCES, NOT 3D distances')
    print('Engineered feature counts:', {k: z.shape[1] for k, z in groups.items()})
    print('High (>40) / low (<10): original fixed logistic/XGB model families')

    rows = []
    all_configs = ['baseline', 'domains', 'pocket_proxy', 'chemistry', 'all']
    for task_idx, (task, threshold, side, ref_col, algorithm) in enumerate([
        ('corrector_low', 10, 'low', 'xgb_seq80', 'xgb_seq80'),
        ('corrector_high', 40, 'high', 'logistic_seq68', 'logistic_seq68'),
    ]):
        yy = (yc < threshold).astype(int) if side == 'low' else (yc > threshold).astype(int)
        ref = previous.loc[previous.task.eq(task)].copy()
        if len(ref) != 420:
            raise ValueError('Missing frozen OOF for ' + task)
        ref = ref.set_index(['repeat', 'fold', 'variant_key'])
        for fold_id, (repeat, fold, tr_keys, te_keys) in enumerate(folds, 1):
            tr = np.flatnonzero(np.isin(keys, list(tr_keys)))
            te = np.flatnonzero(np.isin(keys, list(te_keys)))
            if len(tr) + len(te) != 210 or set(pp[tr]) & set(pp[te]):
                raise ValueError('Outer-fold membership/leakage')
            old = ref.loc[[(repeat, fold, key) for key in keys[te]], :]
            if (old.label.to_numpy(int) != yy[te]).any():
                raise ValueError('Mismatch with frozen task labels')
            baseline = old[ref_col].to_numpy(float)
            ytr = yy[tr]
            if np.unique(ytr).size != 2:
                raise ValueError('Single class in training fold')
            base_matrix = X80 if side == 'low' else X80[:, :68]
            for config in all_configs:
                if config == 'baseline':
                    pred = baseline
                else:
                    XX = np.column_stack([base_matrix, groups[config].to_numpy(float)])
                    fitted = make_model(algorithm, ytr,
                                          27000 + task_idx * 100 + fold_id)
                    fitted.fit(XX[tr], ytr)
                    pred = fitted.predict_proba(XX[te])[:, 1]
                for row_idx, real_idx in enumerate(te):
                    rows.append({
                        'task': task, 'repeat': int(repeat), 'fold': int(fold),
                        'variant_key': keys[real_idx], 'protein_position': int(pp[real_idx]),
                        'label': int(yy[real_idx]), 'config': config,
                        'score': float(pred[row_idx]),
                    })
            print(f'{task} fold {fold_id}/{len(folds)} done', flush=True)

    out = pd.DataFrame(rows)
    if len(out) != 420 * 2 * len(all_configs):
        raise ValueError('Incomplete prediction table')
    if out.duplicated(['task','repeat','fold','variant_key','config']).any():
        raise ValueError('Duplicate predictions')
    if not np.isfinite(out.score).all():
        raise ValueError('Nonfinite predictions')
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / 'phase27d1_repeated_oof.csv', index=False)
    avg = out.groupby(['task', 'config', 'variant_key', 'protein_position'],
                      as_index=False).agg(label=('label', 'first'), score=('score','mean'))
    if len(avg) != 2 * len(all_configs) * 210:
        raise ValueError('Incomplete averaged OOF')
    avg.to_csv(OUT / 'phase27d1_averaged_oof.csv', index=False)
    metrics = []
    comparisons = []
    for task in ('corrector_low','corrector_high'):
        d = avg.loc[avg.task.eq(task)]
        wide = d.pivot(index=['variant_key','protein_position','label'],
                       columns='config',values='score').reset_index()
        if len(wide) != 210 or wide.isna().any().any():
            raise ValueError('Misaligned averaged predictions')
        print('\n' + task.upper())
        for config in all_configs:
            m = evaluation(wide.label, wide[config])
            metrics.append({'task':task,'config':config,**m})
            print(f"  {config:13s}: AP={m['ap']:.4f} AUROC={m['auroc']:.4f} "
                  f"top42={m['top20_hits']}/{m['positives']}")
            if config != 'baseline':
                bres = ap_gain_bootstrap(wide, 'baseline', config, nboot,
                                           seed=27800 + len(comparisons))
                comparisons.append({'task':task,'config':config,**bres})
    metrics = pd.DataFrame(metrics)
    gains = pd.DataFrame(comparisons)
    metrics.to_csv(OUT / 'phase27d1_metrics.csv', index=False)
    gains.to_csv(OUT / 'phase27d1_ap_gain_bootstrap.csv', index=False)
    print('\nA. TWO-REPEAT AVERAGED OOF RESULTS')
    print(metrics.round(4).to_string(index=False))
    print('\nB. POSITION-CLUSTER BOOTSTRAP VS FROZEN SPECIALISTS')
    print(gains.round(4).to_string(index=False))
    print('\nC. ANNOTATION COVERAGE')
    for name, (lo, hi) in DOMAINS.items():
        mask = (pp >= lo) & (pp <= hi)
        print(f'{name:21s} n={int(mask.sum()):3d} '
              f'low={int((yc[mask] < 10).sum()):2d} '
              f'high={int((yc[mask] > 40).sum()):2d}')
    print('\nPASS: Phase27D-1 completed. TRAIN only; all original folds retained.')
    print('These feature groups were compared repeatedly; gains are exploratory.')
    print('Outputs:', OUT)


def self_test():
    g = annotation_features([74, 195, 500, 750, 1098, 1300],
                            ['R74W','L195P','F500L','A750V','W1098C','N1300K'])
    assert len(g['all']) == 6
    assert g['domains']['uniprot_NBD1_core'].iloc[2] == 1
    assert g['pocket_proxy']['VX445_corrector_seq_exp30'].iloc[4] == 1
    assert g['chemistry']['gain_proline'].iloc[1] == 1
    assert np.isfinite(g['all'].to_numpy()).all()
    assert np.isclose(evaluation(np.array([0,1]*105),
                                  np.array([.1,.9]*105))['ap'],1.)
    print('PASS synthetic: UniProt intervals, residues, feature shapes and AP')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--bootstraps', type=int, default=2000)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.bootstraps < 100:
            parser.error('Bootstraps must be >=100')
        run(args.bootstraps)
