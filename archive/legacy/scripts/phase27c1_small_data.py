#!/usr/bin/env python3
"""Phase 27C-1: architecture + within-fold resampling for CFTR response.

Only VTI TRAIN; original 2x5 protein-position-disjoint splits.
Benchmark against frozen 27B1 OOF predictions without refitting baselines.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path.cwd()
OUT = ROOT / 'results/biology/phase27c'
BASE = ROOT / 'results/biology/phase27b/phase27b1_repeated_oof.csv'
KEY = ['repeat', 'fold', 'variant_key', 'protein_position']

NEW = ['weighted_logistic', 'oversampled_logistic', 'pca_logistic', 'rbf_svm', 'extra_trees']


def scores(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    if not np.isfinite(s).all() or len(np.unique(y)) != 2:
        raise ValueError('Nonfinite scores or one-class target')
    n = int(np.ceil(len(y) * .2))
    ix = np.argsort(-s, kind='stable')[:n]
    return dict(ap=float(average_precision_score(y, s)),
                auc=float(roc_auc_score(y, s)),
                top20_hits=int(y[ix].sum()), top20_size=n,
                positives=int(y.sum()))


def cluster_bootstrap(q, candidate, baseline, nboot, seed):
    q = q.copy()
    pos = q.protein_position.to_numpy(int)
    y = q.label.to_numpy(int)
    a = q[candidate].to_numpy(float)
    b = q[baseline].to_numpy(float)
    unique = np.unique(pos)
    blocks = [np.where(pos == z)[0] for z in unique]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(nboot):
        draw = rng.integers(len(blocks), size=len(blocks))
        ix = np.concatenate([blocks[j] for j in draw])
        if len(np.unique(y[ix])) < 2:
            continue
        diffs.append(average_precision_score(y[ix], a[ix]) - average_precision_score(y[ix], b[ix]))
    if len(diffs) < .95 * nboot:
        raise ValueError('Bootstrap samples lack a class too often')
    return [float(average_precision_score(y, a) - average_precision_score(y, b)),
            *np.quantile(diffs, [.025, .975]).tolist()]


def fit_predict(method, x68tr, x68te, x80tr, x80te, ytr, seed):
    if len(np.unique(ytr)) != 2:
        raise ValueError('One-class training fold')
    if method == 'weighted_logistic':
        m = make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, solver='liblinear', class_weight='balanced', max_iter=3000))
        m.fit(x68tr, ytr)
        result = m.predict_proba(x68te)[:, 1]
    elif method == 'oversampled_logistic':
        # Random oversampling adds NO independent biological observations.
        rng = np.random.default_rng(seed)
        a, b = np.flatnonzero(ytr == 0), np.flatnonzero(ytr == 1)
        if len(a) < len(b):
            extra = rng.choice(a, size=len(b) - len(a), replace=True)
        else:
            extra = rng.choice(b, size=len(a) - len(b), replace=True)
        idx = np.concatenate([np.arange(len(ytr)), extra])
        m = make_pipeline(StandardScaler(), LogisticRegression(
            C=.1, solver='liblinear', max_iter=3000))
        m.fit(x68tr[idx], ytr[idx])
        result = m.predict_proba(x68te)[:, 1]
    elif method == 'pca_logistic':
        m = make_pipeline(StandardScaler(), PCA(n_components=12, random_state=seed),
                          LogisticRegression(C=.1, solver='liblinear', class_weight='balanced', max_iter=3000))
        m.fit(x68tr, ytr)
        result = m.predict_proba(x68te)[:, 1]
    elif method == 'rbf_svm':
        m = make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf',
                                                gamma='scale', class_weight='balanced'))
        m.fit(x68tr, ytr)
        # Decision scores rank outcomes; do NOT call these probabilities.
        result = m.decision_function(x68te)
    elif method == 'extra_trees':
        m = make_pipeline(SimpleImputer(strategy='median', add_indicator=True,
                                        keep_empty_features=True),
                          ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3,
                                               max_features=.7,
                                               class_weight='balanced_subsample',
                                               n_jobs=4, random_state=seed))
        m.fit(x80tr, ytr)
        result = m.predict_proba(x80te)[:, 1]
    elif method == 'tabpfn':
        try:
            from tabpfn import TabPFNClassifier
        except ImportError as exc:
            raise SystemExit('Install optional TabPFN with: python -m pip install tabpfn') from exc
        model = TabPFNClassifier()
        model.fit(x68tr, ytr)
        result = model.predict_proba(x68te)[:, list(model.classes_).index(1)]
    else:
        raise ValueError(method)
    result = np.asarray(result, float)
    if result.shape != (len(x68te),) or not np.isfinite(result).all():
        raise ValueError(f'Bad output for {method}: {result.shape}')
    return result


def self_test():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(70, 80))
    x[:12, 68:76] = np.nan
    y = np.array([0] * 50 + [1] * 20)
    for name in NEW:
        p = fit_predict(name, x[:60, :68], x[60:, :68], x[:60], x[60:], y[:60], 27)
        assert p.shape == (10,) and np.isfinite(p).all(), name
    q = pd.DataFrame({'protein_position':np.arange(10), 'label':[0,1]*5,
                      'base':np.arange(10), 'new':np.arange(10)})
    assert np.allclose(cluster_bootstrap(q, 'new', 'base', 100, 1), [0,0,0])
    print('PASS synthetic: five architectures, resampling, NaN imputation, cluster bootstrap')


def run(nboot, do_tabpfn):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import phase26h_auxiliary_corrector as h
    if not BASE.exists():
        raise FileNotFoundError(f'Prior frozen OOF missing: {BASE}')
    v, x, _, _, folds = h.load_data()
    x = np.asarray(x, float)
    vk = v.variant_key.astype(str).to_numpy()
    pp = v.protein_position.to_numpy(int)
    yc = v.corrector_rescue_pp.to_numpy(float)
    if x.shape != (210, 80) or len(np.unique(pp)) != 168 or len(set(vk)) != 210:
        raise ValueError('Unexpected VTI TRAIN shape or identities')
    if not np.isfinite(x[:, :68]).all() or not np.isfinite(yc).all():
        raise ValueError('Missing sequence features or target')
    frozen = pd.read_csv(BASE, dtype={'variant_key':str})
    if len(frozen) != 1680 or frozen.duplicated(KEY+['task']).any():
        raise ValueError('Frozen 27B1 OOF invalid')
    tasks = {
        'corrector_low':(yc < 10).astype(int),
        'corrector_high':(yc > 40).astype(int)
    }
    expected = {'corrector_low':52, 'corrector_high':29}
    originals = {'corrector_low':('xgb_seq80', 'original_low'),
                 'corrector_high':('logistic_seq68', 'original_high')}
    methods = NEW + (['tabpfn'] if do_tabpfn else [])
    records = []
    print('='*108)
    print('PHASE 27C-1 | SMALL-DATA ARCHITECTURES AND CLASS BALANCING | VTI TRAIN ONLY')
    print('='*108)
    print('210 variants / 168 protein positions; original 2x5 grouped folds')
    print('Tasks: corrector <10 (n=52), corrector >40 (n=29)')
    print('Methods:', ', '.join(methods))
    print('No VAL/TEST labels, no synthetic CFTR variants, no model selection on protected labels')
    for t, (repeat, fold, train_keys, test_keys) in enumerate(folds, 1):
        tr = np.flatnonzero(np.isin(vk, list(train_keys)))
        te = np.flatnonzero(np.isin(vk, list(test_keys)))
        if len(tr) + len(te) != 210 or set(pp[tr]) & set(pp[te]):
            raise ValueError('Outer fold leakage or missing samples')
        for task_idx, (task, y) in enumerate(tasks.items()):
            previous = frozen.loc[frozen.task.eq(task) & frozen.repeat.eq(repeat) & frozen.fold.eq(fold)]
            if len(previous) != len(te) or set(previous.variant_key) != set(vk[te]):
                raise ValueError('Frozen fold membership mismatch')
            previous = previous.set_index('variant_key').loc[vk[te]]
            if not np.array_equal(previous.label.to_numpy(int), y[te]):
                raise ValueError('Frozen label mismatch')
            source_col, baseline = originals[task]
            all_scores = {baseline:previous[source_col].to_numpy(float)}
            for method_idx, method in enumerate(methods):
                all_scores[method] = fit_predict(
                    method, x[tr, :68], x[te, :68], x[tr], x[te], y[tr],
                    seed=28000 + 100*t + task_idx*20 + method_idx)
            for j, ix in enumerate(te):
                row = dict(repeat=int(repeat),fold=int(fold),variant_key=vk[ix],
                           protein_position=int(pp[ix]),task=task,label=int(y[ix]))
                row.update({name:float(value[j]) for name,value in all_scores.items()})
                records.append(row)
        print(f'completed {t}/{len(folds)} repeat={repeat} fold={fold} held_positions={len(np.unique(pp[te]))}', flush=True)
    df = pd.DataFrame(records)
    if len(df) != 840 or df.duplicated(KEY+['task']).any():
        raise ValueError('Incomplete outer OOF')
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT/'phase27c1_repeated_oof.csv', index=False)
    group_cols = ['task','variant_key','protein_position','label']
    mean = df.groupby(group_cols,as_index=False).mean(numeric_only=True)
    # Remove meaningless averaged fold/repeat columns.
    mean = mean.drop(columns=['fold','repeat'], errors='ignore')
    if len(mean) != 420:
        raise ValueError('Unexpected averaged OOF variants')
    mean.to_csv(OUT/'phase27c1_averaged_oof.csv',index=False)
    metric_rows=[]; ci_rows=[]
    for task, q in mean.groupby('task'):
        y = q.label.to_numpy(int)
        if int(y.sum()) != expected[task]:
            raise ValueError('Responder prevalence mismatch')
        baseline = originals[task][1]
        for name in [baseline] + methods:
            metric_rows.append(dict(task=task,method=name,**scores(y,q[name])))
        for j, name in enumerate(methods):
            gain, lo, hi = cluster_bootstrap(q, name, baseline, nboot,
                                              seed=29000 + 100*list(tasks).index(task)+j)
            ci_rows.append(dict(task=task,method=name,reference=baseline,
                                ap_gain=gain,ci95_low=lo,ci95_high=hi,positions=len(np.unique(q.protein_position))))
    metrics = pd.DataFrame(metric_rows)
    comparisons = pd.DataFrame(ci_rows)
    metrics.to_csv(OUT/'phase27c1_metrics.csv',index=False)
    comparisons.to_csv(OUT/'phase27c1_ap_bootstrap.csv',index=False)
    print('\nA. HELD-OUT-POSITION CLASSIFIER RANKING (two-repeat averaged OOF)')
    print(metrics.round(4).to_string(index=False))
    print('\nB. AP GAIN VS FROZEN SPECIALIST; POSITION-CLUSTER BOOTSTRAP')
    print(comparisons.round(4).to_string(index=False))
    print('\nINTERPRETATION: OOF discovery only. These models are not independently confirmed.')
    print('Random oversampling duplicates TRAIN observations; it does not create biological variants.')
    print('All methods are compared on exactly the same mutations/outer splits.')
    print('PASS: Phase27C-1 complete. Saved:',OUT)


if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--bootstraps',type=int,default=2000)
    ap.add_argument('--tabpfn',action='store_true',help='Optional pretrained tabular foundation model; requires separate install/license')
    a=ap.parse_args()
    if a.self_test:
        self_test()
    else:
        if a.bootstraps < 100:
            ap.error('Need at least 100 bootstrap draws')
        run(a.bootstraps, a.tabpfn)
