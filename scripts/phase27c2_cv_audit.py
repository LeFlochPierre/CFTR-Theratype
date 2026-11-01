#!/usr/bin/env python3
"""Phase 27C-2: responder fold balance and position-grouped stratification.

TRAIN only. Compare pre-existing grouped 5-fold OOF with stratified
position-grouped 5-fold and 10-fold using identical classifier architectures.
Do not interpret a favourable split design as independently verified improvement.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))

OUT = ROOT / 'results/biology/phase27c'
OLD = ROOT / 'results/biology/phase27b/phase27b1_repeated_oof.csv'
TASKS = (('corrector_low', 'xgb_seq80'), ('corrector_high', 'logistic_seq68'))
PROTOCOLS = (('stratified_group5', 5, (271005, 271006)),
             ('stratified_group10', 10, (271010, 271011)))


def categories(y):
    y = np.asarray(y, float)
    return np.where(y < 10, 0, np.where(y > 40, 2, 1)).astype(int)


def summarise(q):
    y = q.label.to_numpy(int)
    s = q.score.to_numpy(float)
    if len(set(y)) != 2 or not np.isfinite(s).all():
        raise ValueError('Invalid full-OOF labels or scores')
    n = int(np.ceil(len(q) * .2))
    hits = int(y[np.argsort(-s, kind='stable')[:n]].sum())
    return dict(n=len(q), positives=int(y.sum()), ap=float(average_precision_score(y, s)),
                auroc=float(roc_auc_score(y, s)), top20_hits=hits,
                top20_precision=hits/n, top20_recall=hits/int(y.sum()))


def validate_fold(groups, tr, te, n):
    tr = np.asarray(tr, int)
    te = np.asarray(te, int)
    if len(tr) + len(te) != n or len(set(tr) & set(te)):
        raise ValueError('Nonpartition or repeated indices')
    if set(groups[tr]) & set(groups[te]):
        raise ValueError('Leakage: protein position in both train/test')


def fold_audit(protocol, rep, fold, tr, te, y, groups):
    c = categories(y)
    counts_train = np.bincount(c[tr], minlength=3)
    counts_test = np.bincount(c[te], minlength=3)
    return {'protocol': protocol, 'repeat': int(rep), 'fold': int(fold),
            'train_variants': len(tr), 'test_variants': len(te),
            'train_positions': len(set(groups[tr])), 'test_positions': len(set(groups[te])),
            'train_low': int(counts_train[0]), 'train_middle': int(counts_train[1]),
            'train_high': int(counts_train[2]), 'test_low': int(counts_test[0]),
            'test_middle': int(counts_test[1]), 'test_high': int(counts_test[2])}


def bootstrap_ap_gain(avg, new_protocol, task, nboot=2000, seed=27):
    base = avg[(avg.protocol == 'original_group5') & (avg.task == task)]
    new = avg[(avg.protocol == new_protocol) & (avg.task == task)]
    q = base[['variant_key', 'protein_position', 'label', 'score']].merge(
        new[['variant_key', 'protein_position', 'label', 'score']],
        on=['variant_key', 'protein_position', 'label'],
        how='inner', validate='one_to_one', suffixes=('_old', '_new'))
    if len(q) != 210 or not q.variant_key.is_unique:
        raise ValueError('Paired OOF identities do not match')
    positions = q.protein_position.unique()
    cluster_ix = [np.flatnonzero(q.protein_position.to_numpy() == p) for p in positions]
    y = q.label.to_numpy(int)
    old, new = q.score_old.to_numpy(float), q.score_new.to_numpy(float)
    observed = average_precision_score(y, new) - average_precision_score(y, old)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(nboot):
        chosen = rng.integers(0, len(positions), size=len(positions))
        ix = np.concatenate([cluster_ix[k] for k in chosen])
        if np.unique(y[ix]).size != 2:
            continue
        draws.append(average_precision_score(y[ix], new[ix]) -
                     average_precision_score(y[ix], old[ix]))
    if len(draws) < .95*nboot:
        raise ValueError('Bootstrap too many single-class draws')
    return dict(task=task, protocol=new_protocol, reference='original_group5',
                ap_gain=float(observed), ci95_low=float(np.quantile(draws, .025)),
                ci95_high=float(np.quantile(draws, .975)), positions=len(positions))


def self_test():
    y = np.array([5, 10, 30, 40, 41, 51], float)
    assert np.array_equal(categories(y), [0, 1, 1, 1, 2, 2])
    q = pd.DataFrame({'label': [0, 0, 1, 1], 'score': [.1, .2, .8, .9]})
    assert np.isclose(summarise(q)['ap'], 1)
    g = np.repeat(np.arange(60), 2)
    cl = np.repeat(np.array([0]*15+[1]*33+[2]*12), 2)
    for nfold in (5, 10):
        splitter = StratifiedGroupKFold(n_splits=nfold, shuffle=True, random_state=270)
        seen = []
        for train, test in splitter.split(np.zeros((len(g), 1)), cl, g):
            validate_fold(g, train, test, len(g))
            seen.extend(test)
        assert sorted(seen) == list(range(len(g)))
    print('PASS synthetic: labels, ranking, grouped stratified 5/10 folds')


def run(nboot):
    import phase26h_auxiliary_corrector as h
    import phase27b1_extreme_classifiers as b

    OUT.mkdir(parents=True, exist_ok=True)
    if not OLD.is_file():
        raise FileNotFoundError(f'Expected existing OOF: {OLD}')
    v, X, _, _, old_folds = h.load_data()
    X = np.asarray(X, float)
    keys = v.variant_key.astype(str).to_numpy()
    groups = v.protein_position.to_numpy(int)
    y = v.corrector_rescue_pp.to_numpy(float)
    cls = categories(y)
    if len(y) != 210 or X.shape != (210, 80) or len(set(groups)) != 168:
        raise ValueError('Unexpected original VTI TRAIN dataset shape')
    if not np.array_equal(np.bincount(cls, minlength=3), [52, 129, 29]):
        raise ValueError('Unexpected responder prevalence')
    if not np.isfinite(y).all() or not np.isfinite(X[:, :68]).all():
        raise ValueError('Nonfinite labels or sequence features')
    if len(old_folds) != 10:
        raise ValueError('Expected original 2x5 splits')

    print('='*102)
    print('PHASE 27C-2 | ORIGINAL vs STRATIFIED GROUP 5/10-FOLD | TRAIN ONLY')
    print('='*102)
    print('210 VTI TRAIN variants, 168 positions | low=52, middle=129, high=29')
    print('Models unchanged: low XGBoost seq80, high logistic seq68')
    print('Original Group5 uses frozen Phase27B1 OOF; stratified models newly cross-fitted')
    print('Held-out validation/test outcomes unopened')

    old = pd.read_csv(OLD, dtype={'variant_key': str})
    if len(old) != 1680:
        raise ValueError('Phase27B1 predictions expected 1680 records')
    known = dict(zip(keys, zip(groups, y)))
    repeated = []
    for task, column in TASKS:
        q = old.loc[old.task.eq(task)].copy()
        if len(q) != 420 or q.duplicated(['repeat', 'variant_key']).any():
            raise ValueError(f'Old predictions corrupted for {task}')
        for rec in q.itertuples(index=False):
            if rec.variant_key not in known or int(rec.protein_position) != int(known[rec.variant_key][0]):
                raise ValueError('Old OOF identity mismatch')
            actual = known[rec.variant_key][1]
            if int(rec.label) != int(actual < 10 if task == 'corrector_low' else actual > 40):
                raise ValueError('Old OOF label mismatch')
        repeated.extend(dict(protocol='original_group5', repeat=int(r.repeat), fold=int(r.fold),
                             variant_key=r.variant_key, protein_position=int(r.protein_position),
                             task=task, label=int(r.label), score=float(getattr(r, column)))
                        for r in q.itertuples(index=False))

    audit = []
    for rep, fold, train_keys, test_keys in old_folds:
        tr = np.flatnonzero(np.isin(keys, list(train_keys)))
        te = np.flatnonzero(np.isin(keys, list(test_keys)))
        validate_fold(groups, tr, te, len(y))
        audit.append(fold_audit('original_group5', rep, fold, tr, te, y, groups))

    for protocol, nfold, seeds in PROTOCOLS:
        for rep, seed in enumerate(seeds, 1):
            sgkf = StratifiedGroupKFold(n_splits=nfold, shuffle=True, random_state=seed)
            visited = []
            for fold, (tr, te) in enumerate(sgkf.split(X[:, :68], cls, groups), 1):
                validate_fold(groups, tr, te, len(y))
                visited.extend(te.tolist())
                audit.append(fold_audit(protocol, rep, fold, tr, te, y, groups))
                for task, method in TASKS:
                    labels = (y < 10).astype(int) if task == 'corrector_low' else (y > 40).astype(int)
                    if len(np.unique(labels[tr])) != 2:
                        raise ValueError(f'No positives or negatives in training: {protocol}, {rep}, {fold}, {task}')
                    model = b.make_model(method, labels[tr], int(seed + fold + (1000 if task == 'corrector_high' else 0)))
                    data = X[:, :68] if method == 'logistic_seq68' else X
                    model.fit(data[tr], labels[tr])
                    prob = model.predict_proba(data[te])[:, 1]
                    if not np.isfinite(prob).all():
                        raise ValueError('Nonfinite predictions')
                    repeated.extend({'protocol': protocol, 'repeat': rep, 'fold': fold,
                                     'variant_key': keys[j], 'protein_position': int(groups[j]),
                                     'task': task, 'label': int(labels[j]), 'score': float(prob[k])}
                                    for k, j in enumerate(te))
                print(f'finished {protocol} repeat {rep}/{len(seeds)}, fold {fold}/{nfold}: '
                      f'test high={int(sum(cls[te] == 2))}, train high={int(sum(cls[tr] == 2))}, '
                      f'test low={int(sum(cls[te] == 0))}', flush=True)
            if sorted(visited) != list(range(len(y))):
                raise ValueError(f'Incomplete or overlapping OOF for {protocol}, repeat {rep}')

    d = pd.DataFrame(repeated)
    if d.duplicated(['protocol', 'repeat', 'task', 'variant_key']).any():
        raise ValueError('Duplicated OOF membership')
    for protocol in ['original_group5'] + [p[0] for p in PROTOCOLS]:
        for rep in (1, 2):
            for task, _ in TASKS:
                q = d[(d.protocol == protocol) & (d.repeat == rep) & (d.task == task)]
                if len(q) != 210 or q.variant_key.nunique() != 210:
                    raise ValueError(f'Incomplete protocol {protocol}, rep {rep}, task {task}')

    per_repeat = pd.DataFrame([dict(protocol=protocol, repeat=int(rep), task=task, **summarise(q))
                               for (protocol, rep, task), q in d.groupby(['protocol', 'repeat', 'task'])])
    averaged = d.groupby(['protocol', 'task', 'variant_key', 'protein_position', 'label'], as_index=False).score.mean()
    mean_metrics = pd.DataFrame([dict(protocol=protocol, task=task, **summarise(q))
                                 for (protocol, task), q in averaged.groupby(['protocol', 'task'])])
    gains = pd.DataFrame([bootstrap_ap_gain(averaged, protocol, task, nboot=nboot, seed=275500+i)
                          for i, (protocol, task) in enumerate((p[0], t[0]) for p in PROTOCOLS for t in TASKS)])
    fold_df = pd.DataFrame(audit)
    d.to_csv(OUT/'phase27c2_repeated_oof.csv', index=False)
    averaged.to_csv(OUT/'phase27c2_averaged_oof.csv', index=False)
    per_repeat.to_csv(OUT/'phase27c2_per_repeat_metrics.csv', index=False)
    mean_metrics.to_csv(OUT/'phase27c2_averaged_metrics.csv', index=False)
    gains.to_csv(OUT/'phase27c2_ap_bootstrap.csv', index=False)
    fold_df.to_csv(OUT/'phase27c2_fold_balance.csv', index=False)

    print('\nA. RESPONDER DISTRIBUTION ACROSS FOLDS')
    for protocol, q in fold_df.groupby('protocol', sort=False):
        print(f'\n{protocol} | total folds={len(q)}')
        print(q[['test_high','train_high','test_low','train_low','test_positions']].agg(['min','median','max']).to_string())
        print('Held-out folds with <2 high responders:', int((q.test_high < 2).sum()))

    print('\nB. POOLED TWO-REPEAT AVERAGED OOF (PRIMARY COMPARISON)')
    print(mean_metrics[['task','protocol','ap','auroc','top20_hits','top20_recall','positives']]
          .sort_values(['task','protocol']).round(4).to_string(index=False))
    print('\nC. EACH REPEAT SEPARATELY')
    print(per_repeat[['task','protocol','repeat','ap','auroc','top20_hits']]
          .sort_values(['task','protocol','repeat']).round(4).to_string(index=False))
    print('\nD. POSITION-CLUSTER BOOTSTRAP: AP GAIN VS ORIGINAL GROUP5')
    print(gains.round(4).to_string(index=False))
    print('\nInterpretation: CV scheme affects training fraction and held-out composition.')
    print('Do not choose the most favourable scheme as evidence of generalisation improvement.')
    print('5-fold vs 10-fold is a protocol comparison, NOT a new held-out evaluation.')
    print('PASS: Phase27C-2 complete | TRAIN only | Saved:', OUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--bootstraps', type=int, default=2000)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        if args.bootstraps < 100:
            parser.error('bootstraps must be >=100')
        run(args.bootstraps)
