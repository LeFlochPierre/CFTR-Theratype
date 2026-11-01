#!/usr/bin/env python3
"""Phase 23B — position-disjoint ETI->VTI functional transfer pilot.

Purpose: Test whether Bihler-2024 ETI-corrected *acute* IVA increments supply
useful supervision for Bihler-2026 VTI-corrected *acute* IVA prediction.

Design:
  - Frozen 2026 VTI TRAIN only: 210 missense variants / 168 positions.
  - Eligible Bihler-2024 TRAIN-only labels from Phase 23A: 383 rows / 276 positions.
  - Sequence-only 68-feature pilot, with a matched VTI-only control. DO NOT
    compare a cross-study sequence-only result to the 80-feature control as
    an ablation; the historical 80-feature result is context only.
  - Joint XGBoost regressor with an explicit treatment/task indicator and
    treatment-specific gated features; no ETI row is given a VTI task tag.
  - Four arms: VTI control, ETI all, ETI novel-to-VTI, shuffled ETI labels.
  - Per outer fold: drop all ETI rows at held-out VTI protein positions.
    Per inner fold: drop all ETI rows at inner-held VTI positions too.
  - Select ETI mixture weight + XGB depth using 3-fold *grouped inner* VTI
    position-MAE, never outer targets.
  - No use of 2024/2026 protected outcomes or VTI validation/test outcomes.

Caution: two nested CV repeats are for development, not independent external
validation. This is a pretest of functional transfer, NOT a clinically validated
prediction method. >20 pp is an exploratory screen, not a clinical cutoff.

Run:
  python scripts/phase23b_eti_vti_functional_transfer.py --self-test
  python scripts/phase23b_eti_vti_functional_transfer.py --smoke
  python scripts/phase23b_eti_vti_functional_transfer.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from cftr_theratype import model_zoo as mz
from cftr_theratype.transfer import ROOT

SEED = 23023
DEPTHS = (2, 3, 4)
MIX_WEIGHTS = (0.0, 0.25, 0.5, 1.0)  # ETI total effective weight / VTI total effective weight
ARMS = ('vti_control', 'eti_all', 'eti_novel', 'eti_shuffled')
TARGET = 'acute_iva_vti_pp'
MIN_POSITIONS = 10


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def insist(flag, message):
    if not bool(flag):
        raise AssertionError(message)


def verify_disjoint(learn_positions, held_positions, label=''):
    overlap = np.intersect1d(np.asarray(list(learn_positions), dtype=int),
                             np.asarray(list(held_positions), dtype=int))
    insist(not len(overlap), f'{label}: protein-position leakage: {overlap[:15].tolist()}')


def group_position_mae(y, pred, pos):
    z = pd.DataFrame({'pos':pos, 'loss':np.abs(np.asarray(y)-np.asarray(pred))})
    return float(z.groupby('pos', sort=False).loss.mean().mean())


def task_features(X, eti: bool):
    """Shared sequence descriptors plus explicit treatment-specific pathway.

    The task bit is *input context*, never inferred as a VTI response label.
    XGB can learn a shared mapping and treatment-specific deviations.
    """
    X = np.asarray(X, float)
    insist(X.ndim == 2 and X.shape[1] == 68, 'Expected 68 sequence descriptors')
    task = np.full((len(X), 1), float(eti))
    return np.column_stack((X, task, X * task))


def new_xgb(depth: int, seed: int) -> XGBRegressor:
    # Exactly the core parameters used in cftr_theratype.model_zoo.estimator,
    # apart from varying max_depth with the declared inner-CV search.
    return XGBRegressor(
        n_estimators=250, max_depth=int(depth), learning_rate=0.035,
        min_child_weight=5, reg_lambda=12.0, subsample=0.85,
        colsample_bytree=0.85, tree_method='hist', objective='reg:squarederror',
        random_state=int(seed), n_jobs=int(os.environ.get('CFTR_XGB_JOBS', '1')),
        verbosity=0,
    )


def predict_fit(Xv, yv, pv, Xt, Xe, ye, pe, *, depth, mix, seed):
    """Fit one treatment-conditioned model and predict ONLY VTI target rows.

    Xe and ye must have been filtered for the intended evaluation positions by
    the caller. Additional assertion for all VTI query positions.
    """
    Xv = np.asarray(Xv, float); yv = np.asarray(yv, float)
    Xt = np.asarray(Xt, float); Xe = np.asarray(Xe, float)
    ye = np.asarray(ye, float)
    pv = np.asarray(pv, int); pe = np.asarray(pe, int)
    insist(Xv.shape[1] == Xt.shape[1] == Xe.shape[1] == 68, 'Wrong feature dims')
    insist(len(Xv) == len(yv) == len(pv) and len(Xe) == len(ye) == len(pe),
           'Misaligned train inputs')
    insist(np.isfinite(yv).all() and np.isfinite(ye).all(), 'Nonfinite supervised outcomes')
    insist(not np.isinf(Xv).any() and not np.isinf(Xe).any(), 'Inf features')
    if float(mix) > 0 and len(Xe) == 0:
        raise ValueError('No auxiliary rows for positive ETI mixture')
    if float(mix) == 0:
        # Matched task-conditional architecture even when no ETI data added.
        Zfit = task_features(Xv, False)
        weight = np.ones(len(yv), float)
        yy = yv
    else:
        Zfit = np.concatenate((task_features(Xv, False), task_features(Xe, True)))
        yy = np.concatenate((yv, ye))
        # Cross-task balancing by *total* weight, not raw 383/210 sample counts.
        weight = np.concatenate((np.ones(len(yv)),
                                 np.full(len(ye), float(mix)*len(yv)/len(ye))))
    Zte = task_features(Xt, False)
    # Fit every missing-value replacement inside this model's training subset.
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    Zfit = imputer.fit_transform(Zfit)
    Zte = imputer.transform(Zte)
    model = new_xgb(depth, seed)
    # Omitting weights for zero-mixture exactly matches an unweighted fit.
    if float(mix) == 0:
        model.fit(Zfit, yy)
    else:
        model.fit(Zfit, yy, sample_weight=weight)
    pred = np.asarray(model.predict(Zte), float)
    insist(np.isfinite(pred).all(), 'Nonfinite held-out predictions')
    return pred


def subset_aux(aux, Xet, held_positions, mode):
    pos = aux.protein_position.to_numpy(int)
    keep = ~np.isin(pos, np.asarray(sorted(set(map(int, held_positions))), int))
    if mode == 'eti_novel':
        keep &= aux.position_relation_to_vti.to_numpy(str) == 'novel_to_vti'
    assert mode in ('eti_all', 'eti_novel', 'eti_shuffled')
    return Xet[keep], aux.loc[keep].copy()


def load_data(root: Path):
    from cftr_theratype.benchmark import make_features
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype import structure_benchmark as structural

    train, mats, _, outer = mz.prepare(root)
    insist(len(train) == 210 and train.protein_position.nunique() == 168 and len(outer) == 10,
           'Frozen VTI training/folds changed')
    seq = bihler_2024_assay_sequence()
    Xv = np.asarray(mats['sequence'], float)
    # End-to-end parity: 2024+2026 features must derive from the SAME routine.
    rebuilt, feature_names = make_features(train, seq)
    insist(Xv.shape == (210, 68), 'Unexpected VTI feature dimensions')
    insist(np.allclose(np.asarray(rebuilt,float), Xv, equal_nan=True, atol=1e-10),
           'Feature implementation parity failure for VTI')

    tfile = root / 'results/boltz/phase19b/vti_train_acute_iva_targets.csv'
    t = pd.read_csv(tfile, dtype={'variant_key': str})
    insist(len(t) == 210 and t.variant_key.is_unique, 'VTI TRAIN target cardinality')
    ydf = train[['variant_key']].merge(t[['variant_key',TARGET]], on='variant_key',
                                      how='left', validate='one_to_one', sort=False)
    insist(ydf.variant_key.tolist() == train.variant_key.tolist(), 'VTI alignment failure')
    yv = ydf[TARGET].to_numpy(float)
    insist(np.isfinite(yv).all(), 'Invalid VTI TRAIN acute response')

    oldpath = root / 'data/processed/bihler_2024_missense_splits.parquet'
    vtipath = root / 'data/processed/bihler_2026_transfer_splits.parquet'
    old = pd.read_parquet(oldpath)  # metadata only
    membership = pd.read_parquet(vtipath)  # metadata only
    insist(len(old) == 586 and len(membership) == 415, 'Frozen membership changed')
    old['variant_key'] = old.variant_key.astype(str)
    old['protein_position'] = old.protein_position.astype(int)
    insist(old.variant_key.is_unique, 'Duplicate old split key')
    protected_old = set(old.loc[old.split.ne('train'), 'protein_position'].astype(int))
    protected_vti = set(membership.loc[membership.transfer_split.ne('train'),
                                       'protein_position'].dropna().astype(int))
    protected_keys = set(membership.loc[membership.transfer_split.ne('train'),
                                         'variant_key'].astype(str))
    blocked = protected_old | protected_vti

    auxfile = root / 'results/biology/phase23a/eti2024_train_auxiliary_candidates.csv'
    aux = pd.read_csv(auxfile, dtype={'variant_key': str})
    insist(len(aux) == 383 and aux.variant_key.is_unique and
           aux.protein_position.nunique() == 276, 'Unexpected Phase 23A auxiliary pool')
    aux['protein_position'] = aux.protein_position.astype(int)
    old_train = old.loc[old.split.eq('train')].copy()
    insist(set(aux.variant_key).issubset(set(old_train.variant_key)),
           'Non-2024-TRAIN ETI outcome in candidate table')
    insist(not (set(aux.variant_key) & protected_keys), 'Protected VTI key in ETI candidate table')
    insist(not (set(aux.protein_position) & blocked), 'Protected ETI position in pool')
    assert set(aux.position_relation_to_vti) == {'at_vti_train_position', 'novel_to_vti'}
    meta = aux[['variant_key', 'protein_position']].merge(
        old_train[['variant_key','protein_position','variant','hgvs_protein',
                   'reference_aa','alternate_aa']],
        on='variant_key', how='left', validate='one_to_one', suffixes=('_aux',''))
    insist((meta.protein_position_aux == meta.protein_position).all(),
           'ETI frozen position mismatch')
    meta['variant_type'] = 'Missense'
    meta['study_cohort'] = 'ETI'
    # Use precisely the same function as VTI (no label-derived descriptors).
    Xet, names_et = make_features(meta, seq)
    Xet = np.asarray(Xet, float)
    insist(Xet.shape == (383,68), 'Unexpected ETI feature dimensions')
    insist(list(feature_names) == list(names_et), 'Cross-study feature column order mismatch')
    insist(not np.isinf(Xet).any(), 'Infinite ETI descriptors')
    # Exact cDNA matches are identical mutation-derived feature vectors.
    vlookup = {str(k): i for i,k in enumerate(train.variant_key)}
    repeated = [(j,vlookup[key]) for j,key in enumerate(aux.variant_key) if key in vlookup]
    insist(len(repeated) == 66, f'Expected 66 exact training nucleotide matches, got {len(repeated)}')
    worst = max(float(np.nanmax(np.abs(Xet[j]-Xv[i])))
                for j,i in repeated)
    insist(worst < 1e-8, f'Exact mutation cross-study feature mismatch {worst}')

    eti = aux.eti_acute_iva_pp.to_numpy(float)
    insist(np.isfinite(eti).all(), 'Invalid 2024 TRAIN-only ETI labels')
    insist(int((eti>20).sum()) == 127, 'ETI high response count changed')
    insist(int((eti>40).sum()) == 29, 'ETI extreme response count changed')
    print(f'PASS: Cross-study 68-feature parity, 66 exact nucleotide variants, maximum delta={worst:.2g}', flush=True)
    print(f'PASS: 383 permitted ETI training labels, 276 positions; '
          f'additional novel positions={aux.loc[aux.position_relation_to_vti.eq("novel_to_vti"),"protein_position"].nunique()}',
          flush=True)
    provenance = {str(p.relative_to(root)): digest(p) for p in
                  (tfile, oldpath, vtipath, auxfile)}
    return train, Xv, yv, outer, aux, Xet, eti, provenance


def select_inner(Xv, yv, pv, aux, Xe, ye, outer_held_positions, arm, seed, smoke=False):
    """Nested tuning on VTI labels only, with held-out ETI labels removed anew.

    'eti_shuffled' receives labels already permuted once globally by caller;
    the held-out-position filtering is identical to eti_all.
    """
    folds = list(GroupKFold(3).split(Xv, yv, pv))
    arms = (0.0,) if arm == 'vti_control' else MIX_WEIGHTS
    depths = DEPTHS
    scores = []
    for depth in depths:
        for mix in arms:
            out = np.full(len(yv), np.nan)
            for j,(fi,hi) in enumerate(folds):
                verify_disjoint(pv[fi],pv[hi],'inner VTI')
                forbidden = set(map(int,outer_held_positions)) | set(map(int,pv[hi]))
                if mix == 0:
                    ax = Xe[:0]; ay = ye[:0]; apos = np.array([],int)
                else:
                    ax, ae = subset_aux(aux, Xe, forbidden, 'eti_novel' if arm=='eti_novel' else 'eti_all')
                    ay = ye[ae.index.to_numpy(int)]  # index preserved from aux
                    apos = ae.protein_position.to_numpy(int)
                verify_disjoint(apos, forbidden,'inner ETI')
                out[hi] = predict_fit(Xv[fi],yv[fi],pv[fi],Xv[hi],
                                      ax,ay,apos,depth=depth,mix=mix,seed=seed+100+j)
            insist(np.isfinite(out).all(),f'Inner coverage failure {arm}')
            score = group_position_mae(yv,out,pv)
            scores.append({'max_depth':depth, 'eti_weight':float(mix),'inner_mae_pp':score})
    # To avoid gratuitous transfer, prefer lower auxiliary influence on ties.
    best = min(scores,key=lambda row: (round(row['inner_mae_pp'],10),row['eti_weight'],row['max_depth']))
    return best, scores


def run_arm(Xv,yv,pv,aux,Xe,ye,fit,held,arm,seed,smoke=False):
    held_pos = set(map(int,pv[held]))
    verify_disjoint(pv[fit],pv[held],'outer VTI')
    best, trials = select_inner(Xv[fit],yv[fit],pv[fit],aux,Xe,ye,held_pos,arm,seed,smoke)
    mix = best['eti_weight']
    if mix == 0:
        ax=Xe[:0]; ay=ye[:0]; apos=np.array([],int)
    else:
        ax, ae = subset_aux(aux,Xe,held_pos,'eti_novel' if arm=='eti_novel' else 'eti_all')
        ay = ye[ae.index.to_numpy(int)]
        apos = ae.protein_position.to_numpy(int)
    verify_disjoint(apos,pv[held],'outer ETI')
    p = predict_fit(Xv[fit],yv[fit],pv[fit],Xv[held],ax,ay,apos,
                    depth=best['max_depth'],mix=mix,seed=seed+1000)
    return p, best, trials, len(ay),len(set(apos))


def rank_metrics(group):
    pos = group.groupby('protein_position').agg(
        observed_max=('observed_pp','max'), score=('predicted_pp','max'))
    positives = pos.observed_max.to_numpy(float)>20
    s = pos.score.to_numpy(float)
    k = int(np.ceil(len(pos)*.2))
    hit = int(positives[np.argsort(-s,kind='stable')[:k]].sum())
    if len(np.unique(positives))==2:
        ap=float(average_precision_score(positives,s)); auc=float(roc_auc_score(positives,s))
    else:
        ap=np.nan; auc=np.nan
    return dict(position_ap20=ap, position_roc_auc20=auc,
                position_top20_hits=hit, position_top20_precision=hit/k,
                position_top20_recall=hit/max(1,int(positives.sum())))


def subgroup_mae(y, p, selected):
    return float(np.abs(y[selected]-p[selected]).mean()) if np.any(selected) else np.nan


def metrics(df):
    y=df.observed_pp.to_numpy(float); p=df.predicted_pp.to_numpy(float)
    z=df.protein_position.to_numpy(int)
    hi=y>20; very=y>40; lo=y<=2
    return dict(position_mae_pp=group_position_mae(y,p,z),
                variant_mae_pp=float(np.abs(y-p).mean()),
                rmse_pp=float(np.sqrt(np.mean((y-p)**2))),
                high20_mae_pp=subgroup_mae(y,p,hi),
                high40_mae_pp=subgroup_mae(y,p,very),
                low2_mae_pp=subgroup_mae(y,p,lo),
                max_prediction_pp=float(np.max(p)),
                **rank_metrics(df))


def bootstrap_gain(averaged, base='vti_control', boots=5000):
    table=averaged.pivot(index='variant_key',columns='model',values='predicted_pp')
    truth=averaged.drop_duplicates('variant_key').set_index('variant_key')
    pos=truth.loc[table.index,'protein_position'].to_numpy(int)
    y=truth.loc[table.index,'observed_pp'].to_numpy(float)
    rows=[]
    for model in table.columns:
        if model==base:continue
        scores=pd.DataFrame({'position':pos,
                             'gain':np.abs(y-table[base].to_numpy(float))-
                                    np.abs(y-table[model].to_numpy(float))})
        bypos=scores.groupby('position').gain.mean().to_numpy(float)
        rng=np.random.default_rng(230000+len(model))
        draws=rng.integers(0,len(bypos),size=(boots,len(bypos)))
        z=bypos[draws].mean(axis=1)
        rows.append({'model':model,'gain_vs_vti_control_pp':float(bypos.mean()),
                     'ci95_lo':float(np.quantile(z,.025)),
                     'ci95_hi':float(np.quantile(z,.975)),
                     'n_positions_improved':int((bypos>0).sum()),
                     'fraction_boot_positive':float((z>0).mean())})
    return pd.DataFrame(rows)


def historical(root, train, averaged):
    file=root/'results/boltz/phase19c/acute_iva_vti_pp_averaged.csv'
    if not file.exists():
        return None
    hist=pd.read_csv(file,dtype={'variant_key':str})
    hist=hist.loc[hist.model.eq('sequence__xgboost')].copy()
    if len(hist)!=210 or not set(hist.variant_key)==set(train.variant_key):
        print('WARNING: historical sequence-only comparison unavailable/schema changed')
        return None
    if not {'observed_pp','predicted_pp','protein_position'}.issubset(hist):
        print('WARNING: historical sequence-only OOF columns unavailable')
        return None
    return metrics(hist)


def self_test():
    rng=np.random.default_rng(42)
    p=np.repeat(np.arange(36),2)
    X=rng.normal(size=(72,68)); y=10+4*X[:,0]+rng.normal(size=72)
    ap=np.repeat(np.arange(50),2)
    Xe=rng.normal(size=(100,68)); ye=15+4*Xe[:,0]+rng.normal(size=100)
    aux=pd.DataFrame({'protein_position':ap,
                      'position_relation_to_vti':np.where(ap<36,'at_vti_train_position','novel_to_vti')})
    fi=np.flatnonzero(p<30); hi=np.flatnonzero(p>=30)
    verify_disjoint(p[fi],p[hi])
    fullX, subset = subset_aux(aux,Xe,p[hi], 'eti_all')
    assert not set(subset.protein_position)&set(p[hi])
    assert len(fullX)==len(subset)
    # Exercise actual inner tuning and joint conditional learner on small data.
    save_depth=DEPTHS; save_mix=MIX_WEIGHTS
    best, trials=select_inner(X[fi],y[fi],p[fi],aux,Xe,ye,set(p[hi]),'eti_all',120)
    assert len(trials)==len(save_depth)*len(save_mix)
    assert best['max_depth'] in save_depth
    preds,_,_,_,_=run_arm(X,y,p,aux,Xe,ye,fi,hi,'eti_all',120)
    assert preds.shape==(len(hi),) and np.isfinite(preds).all()
    print('PASS: Synthetic joint learner, outer & inner position blocking, nesting, full fold prediction')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--bootstrap',type=int,default=5000)
    args=parser.parse_args()
    if args.self_test:
        self_test(); return
    root=Path(ROOT)
    train,Xv,yv,outer,aux,Xe,ye,prov=load_data(root)
    pv=train.protein_position.to_numpy(int)
    print('='*104,flush=True)
    print('PHASE 23B | ETI->VTI ACUTE IVA TRANSFER | FROZEN 2x5 POSITION-DISJOINT CV',flush=True)
    print('='*104,flush=True)
    print('TRAIN ONLY: VTI 210 / 168 positions; ETI aux 383 / 276 positions; 127 ETI >20 responders')
    print('Matched 68 SEQUENCE FEATURES for BOTH ETI and VTI. Treated-label assay identity preserved.')
    print('INNER: 3-fold GroupKFold position MAE selects depth + ETI influence (including 0).')
    print('Both inner and outer held-out positions excluded from ETI supervision.')
    print('20 pp is an exploratory screening definition. Protected outcomes unopened.',flush=True)
    # Single globally fixed permutation, independent of VTI labels, conserves ETI marginals.
    shuffled=np.random.default_rng(SEED).permutation(ye)
    folds=outer[:1] if args.smoke else outer
    preds=[]; choices=[]
    for rep,fold,fit,held in folds:
        start=perf_counter(); per_fold=[]
        for aidx,arm in enumerate(ARMS):
            target=shuffled if arm=='eti_shuffled' else ye
            pred,selected,trials,n_aux,n_auxpos=run_arm(
                Xv,yv,pv,aux,Xe,target,fit,held,arm,SEED+int(rep)*1000+int(fold)*100)
            for j,i in enumerate(held):
                preds.append({'repeat':int(rep),'fold':int(fold),'model':arm,
                              'variant_key':str(train.variant_key.iloc[i]),
                              'protein_position':int(pv[i]),'study_cohort':str(train.study_cohort.iloc[i]),
                              'observed_pp':float(yv[i]),'predicted_pp':float(pred[j])})
            choices.append({'repeat':int(rep),'fold':int(fold),'model':arm,
                            'max_depth':int(selected['max_depth']),
                            'eti_weight':float(selected['eti_weight']),
                            'inner_mae_pp':float(selected['inner_mae_pp']),
                            'n_aux_training_variants':n_aux,
                            'n_aux_training_positions':n_auxpos,
                            'trials_json':json.dumps(trials)})
            per_fold.append(f'{arm}:{group_position_mae(yv[held],pred,pv[held]):.3f} '
                            f'(w={selected["eti_weight"]})')
        print(f'repeat={rep} fold={fold} held_pos={len(set(pv[held]))} '
              f'{" | ".join(per_fold)} | {perf_counter()-start:.1f}s',flush=True)
    if args.smoke:
        print('PASS: One frozen held-out fold, four nested-CV arms, no outputs written.',flush=True)
        return
    oof=pd.DataFrame(preds)
    assert len(oof)==2*210*len(ARMS), ('Missing outer held-out rows',len(oof))
    assert not oof.duplicated(['repeat','model','variant_key']).any()
    avg=(oof.groupby(['model','variant_key'],as_index=False)
         .agg(protein_position=('protein_position','first'),
              study_cohort=('study_cohort','first'),
              observed_pp=('observed_pp','first'),
              predicted_pp=('predicted_pp','mean'),
              n_repeats=('repeat','nunique')))
    assert len(avg)==210*len(ARMS) and avg.n_repeats.eq(2).all()
    summaries=[]
    for arm in ARMS:
        df=avg.loc[avg.model.eq(arm)].copy()
        summaries.append({'model':arm,**metrics(df)})
    summary=pd.DataFrame(summaries).sort_values('position_mae_pp')
    paired=bootstrap_gain(avg,boots=args.bootstrap)
    print('\n'+'='*105)
    print('PHASE 23B | POSITION-MACRO MAE AND >20 pp PRIORITISATION (REPEATED OOF)')
    print('='*105)
    print(summary.round(4).to_string(index=False))
    print('\nPAIRED POSITION BOOTSTRAP OF MAE GAIN VS MATCHED VTI-ONLY CONTROL (exploratory)')
    print(paired.round(4).to_string(index=False))
    print('\nETI INFLUENCE SELECTED BY NESTED INNER CV')
    selection=pd.DataFrame(choices)
    print(pd.crosstab(selection.model,selection.eti_weight).to_string())
    h=historical(root,train,avg)
    if h is not None:
        print('\nHISTORICAL sequence-only Phase19C (context only; different CV tuning):')
        print(json.dumps(h,indent=2))
    print('\nHISTORICAL 80-feature structure+sequence Phase19C: 7.5817 pp (NOT matched ablation)')
    print('HISTORICAL 80-feature reliability-weighted Phase19I: 7.265 pp (NOT matched ablation)')
    out=root/'results/biology/phase23b'; out.mkdir(parents=True,exist_ok=True)
    save={'oof':out/'phase23b_oof.csv', 'averaged':out/'phase23b_averaged.csv',
          'summary':out/'phase23b_summary.csv', 'bootstrap':out/'phase23b_bootstrap.csv',
          'selection':out/'phase23b_inner_selection.csv', 'report':out/'phase23b_report.json'}
    oof.to_csv(save['oof'],index=False); avg.to_csv(save['averaged'],index=False)
    summary.to_csv(save['summary'],index=False); paired.to_csv(save['bootstrap'],index=False)
    selection.to_csv(save['selection'],index=False)
    save['report'].write_text(json.dumps({
        'phase':'23B_eti_vti_acute_functional_transfer',
        'outcomes':'VTI TRAIN only + permitted 2024 ETI TRAIN only',
        'protected_outcomes_accessed':False,'outer_splits':'frozen VTI 2x5 by protein position',
        'inner_splits':'3-fold GroupKFold by protein position with ETI inner and outer blocking',
        'feature_set':'identical 68 mutation-only sequence descriptors for VTI and ETI',
        'mix_grid':MIX_WEIGHTS,'depth_grid':DEPTHS,
        'negative_control':'globally permuted ETI outcomes, nested tuned',
        'source_hashes':prov, 'n_vti_train':210,'n_eti_allowed':383,
        'note':'Exploratory cross-study transfer: no locked external validation yet.',
    },indent=2))
    print('\nOUTPUTS:')
    for key,path in save.items(): print(f'  {key}: {path.relative_to(root)}')
    print('\nPASS: VTI TRAIN-only outer CV + permitted ETI TRAIN-only aux; no protected outcomes opened',flush=True)


if __name__=='__main__':
    main()
