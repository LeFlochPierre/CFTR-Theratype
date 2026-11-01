#!/usr/bin/env python3
"""Phase 23D — ETI teacher -> VTI acute ivacaftor zero-shot transfer.

WHAT IS MEASURED
---------------
Can functional targets measured in Bihler 2024 ETI (corrected acute IVA,
DMSO acute IVA, corrector-mediated FSK gain) teach a mutation-only VTI
prediction model, without knowing the ETI assay of the queried mutation?

SAFE DATA BOUNDARY
------------------
- VTI TRAIN: frozen 210 variants / 168 positions; original two-repeat,
  five-fold position-disjoint outer CV; 3-fold grouped inner CV.
- ETI auxiliary: only frozen 2024 TRAIN, previously protected-position-filtered
  383 candidate rows / 276 positions, loaded using Phase23B audited loader.
- Every ETI teacher fit excludes queried VTI protein positions, as well as
  outer-held and inner-held positions. Teacher features for VTI training rows
  are themselves position-group cross-fitted.
- ETI is NEVER used as a VTI target; ETI targets are teacher-only auxiliary labels.
- No VTI or ETI protected validation/test outcomes opened.
- Strictly mutation-derived features at deployment: XGB receives original
  80 VTI descriptors + predictions of a frozen ETI teacher from 68 sequence
  descriptors. No measured ETI labels from queried mutations are supplied.
- 20 pp is an exploratory screen, not a clinical responsiveness threshold.

ARMS
----
control80              Frozen sequence+structure XGBoost without teacher.
teacher_acute          Cross-fitted predicted ETI-corrected acute IVA feature.
teacher_multi          Predicted corrected IVA, DMSO IVA, corrector FSK gain.
teacher_multi_novel    Same, but ETI teacher uses only ETI positions novel
                       to the original VTI training set.
teacher_multi_shuffle  Same architecture, ETI labels jointly permuted across
                       allowed external rows once; a negative control.

This is development-only. Evaluating many methods on frozen VTI TRAIN OOF
still requires untouched independent validation/test to establish generality.

RUN
---
 python scripts/phase23d_eti_teacher_acute.py --self-test
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  python scripts/phase23d_eti_teacher_acute.py --smoke
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  python scripts/phase23d_eti_teacher_acute.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr

from cftr_theratype import model_zoo as mz
from cftr_theratype.transfer import ROOT
from phase23b_eti_vti_functional_transfer import load_data

SEED = 23024
DEPTHS = (2, 3, 4)
TEACHER_TARGETS = ('eti_acute_iva_pp', 'dmso_acute_iva_pp', 'eti_corrector_fsk_gain_pp')
ARMS = ('control80', 'teacher_acute', 'teacher_multi', 'teacher_multi_novel', 'teacher_multi_shuffle')


def check(flag, msg):
    if not bool(flag):
        raise AssertionError(msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def pmacro(y, pred, positions):
    e = np.abs(np.asarray(y, float) - np.asarray(pred, float))
    d = pd.DataFrame({'p': np.asarray(positions, int), 'e': e})
    return float(d.groupby('p').e.mean().mean())


def make_teacher(Xeti, Yeti, positions_eti, Xquery, forbidden_positions, seed, *, novel_mask=None, shuffle=False):
    """Predict ETI auxiliary phenotypes for queried VTI mutations.

    Teacher cannot use measured ETI labels at ANY forbidden VTI positions.
    This function has no access to VTI response labels.
    """
    Xeti = np.asarray(Xeti, float)
    Yeti = np.asarray(Yeti, float)
    pe = np.asarray(positions_eti, int)
    Xquery = np.asarray(Xquery, float)
    blocked = np.asarray(sorted(set(map(int, forbidden_positions))), int)
    keep = ~np.isin(pe, blocked)
    if novel_mask is not None:
        keep &= np.asarray(novel_mask, bool)
    check(Xeti.ndim == Xquery.ndim == Yeti.ndim == 2, 'Teacher array shape')
    check(Xeti.shape[1] == Xquery.shape[1] == 68, 'ETI teacher must use 68 sequence features')
    check(Yeti.shape[1] == 3, 'Expected three permitted ETI functional targets')
    check(len(Xeti) == len(Yeti) == len(pe) == len(keep), 'ETI alignment')
    check(np.isfinite(Yeti[keep]).all(), 'Invalid ETI labels')
    check(np.sum(keep) >= 30, 'Too few eligible ETI rows')
    check(not np.intersect1d(pe[keep], blocked).size, 'ETI position leakage')
    imp = SimpleImputer(strategy='median', keep_empty_features=True)
    ef = imp.fit_transform(Xeti[keep])
    eq = imp.transform(Xquery)
    forest = ExtraTreesRegressor(
        n_estimators=150, min_samples_leaf=4, max_features=0.75,
        n_jobs=1, random_state=int(seed)
    )
    labels = Yeti[keep]
    # Negative control: permute AFTER the held-position exclusion. Permuting all
    # ETI labels globally first would allow a held-position outcome to be
    # reassigned to a permitted position, invalidating the leakage guarantee.
    if shuffle:
        labels = labels[np.random.default_rng(seed+89011).permutation(len(labels))]
    forest.fit(ef, labels)
    out = np.asarray(forest.predict(eq), float)
    check(out.shape == (len(Xquery), 3) and np.isfinite(out).all(), 'Invalid ETI teacher output')
    return out


def make_features_for_group_split(Xseq, pos_vti, fit_idx, held_idx, Xeti, Yeti, peti,
                                  blocked_outer, seed, *, novel_mask=None, shuffle=False, n_crossfit=3):
    """Teacher pseudo-phenotypes for VTI fit (group OOF) and VTI held (unseen).

    A teacher predicting each VTI fit row never saw its own position's ETI
    labels; a teacher predicting VTI held rows never saw held-position ETI labels.
    In inner CV the caller also includes outer-held VTI positions in blocked_outer.
    """
    fit_idx = np.asarray(fit_idx, int)
    held_idx = np.asarray(held_idx, int)
    p = np.asarray(pos_vti, int)
    fitp = set(map(int, p[fit_idx]))
    heldp = set(map(int, p[held_idx]))
    outer_block = set(map(int, blocked_outer))
    check(fitp.isdisjoint(heldp), 'VTI train/held position overlap')
    check(len(fitp) >= n_crossfit, 'Too few fit positions for teacher OOF')
    check(heldp.isdisjoint(fitp), 'Bad held positions')

    fit_prediction = np.full((len(fit_idx), 3), np.nan)
    cv = GroupKFold(n_splits=n_crossfit)
    for j, (fi_local, ho_local) in enumerate(cv.split(Xseq[fit_idx], groups=p[fit_idx])):
        vti_held_pos = set(map(int, p[fit_idx[ho_local]]))
        blocked = outer_block | heldp | vti_held_pos
        pred = make_teacher(Xeti, Yeti, peti, Xseq[fit_idx[ho_local]], blocked,
                            seed + j + 1, novel_mask=novel_mask, shuffle=shuffle)
        fit_prediction[ho_local] = pred
    held_prediction = make_teacher(Xeti, Yeti, peti, Xseq[held_idx],
                                   outer_block | heldp, seed + 1234,
                                   novel_mask=novel_mask, shuffle=shuffle)
    check(np.isfinite(fit_prediction).all(), 'Incomplete cross-fitted teacher features')
    return fit_prediction, held_prediction


def join_features(X80, etifeat, arm):
    if arm == 'control80':
        return np.asarray(X80, float)
    if arm == 'teacher_acute':
        return np.column_stack([X80, etifeat[:, 0]])
    return np.column_stack([X80, etifeat])


def select_depth(X80, Xseq, y, pos, fit_idx, held_idx, Xeti, Yeti, peti,
                 arm, seed, novel_mask):
    """Nested selection via VTI-only, position-held-out inner targets."""
    inner = list(GroupKFold(3).split(X80[fit_idx], y[fit_idx], groups=pos[fit_idx]))
    cache = []
    outer_pos = set(map(int, pos[held_idx]))
    for inner_i, (fi_local, hi_local) in enumerate(inner):
        fi, hi = fit_idx[fi_local], fit_idx[hi_local]
        check(not (set(pos[fi]) & set(pos[hi])), 'Inner VTI leakage')
        if arm == 'control80':
            A, B = X80[fi], X80[hi]
        else:
            tfit, theld = make_features_for_group_split(
                Xseq, pos, fi, hi, Xeti, Yeti, peti,
                blocked_outer=outer_pos, seed=seed+3000+inner_i*30,
                novel_mask=novel_mask, shuffle=(arm=="teacher_multi_shuffle"),
            )
            A = join_features(X80[fi], tfit, arm)
            B = join_features(X80[hi], theld, arm)
        cache.append((fi, hi, A, B))
    tries = []
    for depth in DEPTHS:
        prediction = np.full(len(fit_idx), np.nan)
        for inner_i, (fi, hi, A, B) in enumerate(cache):
            reg = mz.estimator('xgboost', {'max_depth': depth}, seed+100+inner_i)
            reg.fit(A, y[fi])
            prediction[inner[inner_i][1]] = reg.predict(B)
        check(np.isfinite(prediction).all(), 'Inner CV prediction coverage failure')
        score = pmacro(y[fit_idx], prediction, pos[fit_idx])
        tries.append({'depth': depth, 'inner_mae_pp': score})
    chosen = min(tries, key=lambda z: (round(z['inner_mae_pp'], 12), z['depth']))
    return chosen, tries


def run_fold(X80, Xseq, y, pos, fit_idx, held_idx, Xeti, Yeti, peti,
             arm, seed, novel_mask):
    chosen, tries = select_depth(X80, Xseq, y, pos, fit_idx, held_idx,
                                 Xeti, Yeti, peti, arm, seed, novel_mask)
    if arm == 'control80':
        A, B = X80[fit_idx], X80[held_idx]
        t_held = None
    else:
        tfit, t_held = make_features_for_group_split(
            Xseq, pos, fit_idx, held_idx, Xeti, Yeti, peti,
            blocked_outer=set(map(int, pos[held_idx])), seed=seed+7000,
            novel_mask=novel_mask, shuffle=(arm=="teacher_multi_shuffle"),
        )
        A = join_features(X80[fit_idx], tfit, arm)
        B = join_features(X80[held_idx], t_held, arm)
    reg = mz.estimator('xgboost', {'max_depth': chosen['depth']}, seed+9000)
    reg.fit(A, y[fit_idx])
    pred = np.asarray(reg.predict(B), float)
    check(pred.shape == (len(held_idx),) and np.isfinite(pred).all(), 'Invalid held predictions')
    return pred, t_held, chosen, tries


def ranking(df):
    z = df.groupby('protein_position', sort=True).agg(
        true_max=('observed_pp', 'max'), predicted_max=('predicted_pp', 'max')
    )
    good = z.true_max.to_numpy(float) > 20.
    scores = z.predicted_max.to_numpy(float)
    k = int(np.ceil(0.20 * len(z)))
    hits = int(np.sum(good[np.argsort(-scores, kind='stable')[:k]]))
    return {
        'position_ap20': float(average_precision_score(good, scores)),
        'position_auc20': float(roc_auc_score(good, scores)),
        'position_top20_hits': hits,
        'position_top20_precision': hits/k,
        'position_top20_recall': hits/int(good.sum()),
    }


def metrics(df):
    y, pr, pos = (df[k].to_numpy(float) for k in ('observed_pp','predicted_pp','protein_position'))
    err = np.abs(y-pr)
    return {
        'position_mae_pp': pmacro(y,pr,pos),
        'variant_mae_pp': float(err.mean()),
        'rmse_pp': float(np.sqrt(np.mean((y-pr)**2))),
        'high20_mae_pp': float(err[y>20].mean()),
        'high40_mae_pp': float(err[y>40].mean()),
        'low2_mae_pp': float(err[y<=2].mean()),
        'max_prediction_pp': float(pr.max()),
        **ranking(df),
    }


def bootstrap_comparisons(averaged, baseline='control80', additional_baseline=None, boots=5000):
    pred = averaged.pivot(index='variant_key',columns='model',values='predicted_pp')
    truth = averaged.drop_duplicates('variant_key').set_index('variant_key').loc[pred.index]
    y = truth.observed_pp.to_numpy(float)
    p = truth.protein_position.to_numpy(int)
    if additional_baseline is not None:
        historic = additional_baseline.set_index('variant_key').loc[pred.index]
        pred['historical_reliability_19i'] = historic.predicted_pp.to_numpy(float)
    rows=[]
    for b in ([baseline] + (['historical_reliability_19i'] if additional_baseline is not None else [])):
        for model in ARMS:
            if model == b:
                continue
            delta = pd.DataFrame({'pos':p, 'gain':np.abs(y-pred[b].to_numpy(float))-
                                       np.abs(y-pred[model].to_numpy(float))})
            a = delta.groupby('pos').gain.mean().to_numpy(float)
            rng = np.random.default_rng(SEED+len(model)*137+len(b))
            sample = rng.integers(len(a),size=(boots,len(a)))
            vals = a[sample].mean(axis=1)
            rows.append({'model':model,'comparator':b,'gain_pp':float(a.mean()),
                         'ci95_lo':float(np.quantile(vals,.025)),
                         'ci95_hi':float(np.quantile(vals,.975)),
                         'n_positions_improved':int((a>0).sum()),
                         'fraction_boot_positive':float((vals>0).mean())})
    return pd.DataFrame(rows)


def load_historic(root, train):
    path = root / 'results/boltz/phase19i/phase19i_averaged.csv'
    if not path.is_file():
        print('NOTE: historical reliability-weighted comparator unavailable')
        return None
    x = pd.read_csv(path, dtype={'variant_key':str})
    check({'variant_key','model','predicted_pp'}.issubset(x.columns), 'Historical file schema')
    x = x.loc[x.model.eq('reliability_weighted'),['variant_key','predicted_pp']]
    check(len(x)==210 and x.variant_key.is_unique and
          set(x.variant_key)==set(train.variant_key), 'Historical comparator keys unexpected')
    return x


def self_test():
    rng=np.random.default_rng(123)
    pv=np.repeat(np.arange(24),2)
    xv=rng.normal(size=(len(pv),68))
    x80=np.column_stack([xv,rng.normal(size=(len(pv),12))])
    y=5+10*xv[:,0]+rng.normal(0,3,len(pv))
    ep=np.repeat(np.arange(35),3)
    xe=rng.normal(size=(len(ep),68))
    ey=np.column_stack([8+10*xe[:,0]+rng.normal(0,2,len(ep)),
                        2+4*xe[:,0]+rng.normal(0,2,len(ep)),
                        20+4*xe[:,1]+rng.normal(0,2,len(ep))])
    fit=np.flatnonzero(pv<20)
    held=np.flatnonzero(pv>=20)
    novel=ep>=24
    for arm in ARMS:
        Y=ey
        pred,t,sel,tries=run_fold(x80,xv,y,pv,fit,held,xe,Y,ep,arm,SEED,novel if arm=='teacher_multi_novel' else None)
        assert pred.shape==(len(held),) and np.isfinite(pred).all()
        if arm!='control80':
            assert t.shape==(len(held),3) and np.isfinite(t).all()
        assert sel['depth'] in DEPTHS and len(tries)==len(DEPTHS)
        print(f'{arm:25s} held_pos_mae={pmacro(y[held],pred,pv[held]):.4f} chosen_depth={sel["depth"]}')
    # Direct proof that auxiliary labels at requested position are not used.
    q=np.flatnonzero(pv==0)
    a=make_teacher(xe,ey,ep,xv[q],{0},SEED)
    corrupted=ey.copy();corrupted[ep==0,:]+=1000000.
    b=make_teacher(xe,corrupted,ep,xv[q],{0},SEED)
    assert np.allclose(a,b), 'Masked-position ETI labels changed teacher predictions'
    print('PASS: Five arms, grouped inner/outer VTI split and blocked ETI teacher')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--bootstrap',type=int,default=5000)
    args=parser.parse_args()
    if args.self_test:
        self_test();return
    root=Path(ROOT)
    train,xseq,y,outer,aux,xeti,eti_acute,provenance=load_data(root)
    tr2,mats2,_,outer2=mz.prepare(root)
    check(train.variant_key.tolist()==tr2.variant_key.tolist(), 'VTI feature row mismatch')
    check(len(outer)==len(outer2)==10, 'Frozen outer assignment unavailable')
    x80=np.asarray(mats2['sequence_structure'],float)
    check(x80.shape==(210,80) and xseq.shape==(210,68), 'Unexpected feature shape')
    check(np.allclose(xseq,x80[:,:68], equal_nan=True, atol=1e-8),
          '80 feature prefix differs from validated 68 sequence features')
    check(set(TEACHER_TARGETS).issubset(aux), 'Expected three Phase23A ETI targets')
    etilabels=aux[list(TEACHER_TARGETS)].to_numpy(float)
    check(np.allclose(eti_acute,etilabels[:,0]), 'ETI acute target mismatch')
    check(len(aux)==383 and len(np.unique(aux.protein_position))==276, 'ETI count changed')
    check(np.isfinite(etilabels).all(), 'Nonfinite auxiliary labels')
    peti=aux.protein_position.to_numpy(int)
    pv=train.protein_position.to_numpy(int)
    novel_mask=aux.position_relation_to_vti.to_numpy(str)=='novel_to_vti'
    check(int(np.sum(novel_mask))==258, 'Unexpected novel-position aux count')
    old_split=root/'data/processed/bihler_2024_missense_splits.parquet'
    vti_split=root/'data/processed/bihler_2026_transfer_splits.parquet'
    prov_old = pd.read_parquet(old_split, columns=['variant_key','protein_position','split'])
    prov_vti = pd.read_parquet(vti_split, columns=['variant_key','protein_position','transfer_split'])
    forbidden_old=set(prov_old.loc[prov_old.split.ne('train'),'protein_position'].astype(int))
    forbidden_vti=set(prov_vti.loc[prov_vti.transfer_split.ne('train'), 'protein_position'].dropna().astype(int))
    check(not set(peti)&(forbidden_old|forbidden_vti), 'Unprotected ETI position violation')
    check(set(aux.variant_key).issubset(set(prov_old.loc[prov_old.split.eq('train'),'variant_key'])),
          'ETI auxiliary labels outside original 2024 training set')
    print('='*108,flush=True)
    print('PHASE 23D | ZERO-SHOT ETI TEACHER -> VTI ACUTE IVA | FROZEN POSITION CV',flush=True)
    print('='*108,flush=True)
    print('VTI TRAIN 210 variants / 168 positions; ETI 2024 permitted TRAIN 383 variants / 276 positions')
    print('Teacher: 68 sequence features -> ETI acute IVA, DMSO IVA, ELX+TEZ FSK gain')
    print('Student: 80 sequence+structure features + position-crossfitted ETI teacher predictions')
    print('At INFERENCE: no measured ETI or VTI outcomes of the queried mutation')
    print('ETI labels at outer/inner held VTI positions excluded; nested 3-fold selection')
    print('Five arms including real/shuffled ETI and novel-position-only teacher')
    print('Protected study val/test outcomes unopened',flush=True)
    rows=[]; choices=[]
    for rep,fold,fi,hi in (outer[:1] if args.smoke else outer):
        t=perf_counter()
        check(not(set(pv[fi]) & set(pv[hi])), 'Original outer-fold position overlap')
        s=[]
        for k,arm in enumerate(ARMS):
            labels=etilabels
            nm=novel_mask if arm=='teacher_multi_novel' else None
            pred,tfeat,selected,tries=run_fold(
                x80,xseq,y,pv,fi,hi,xeti,labels,peti,arm,
                SEED+int(rep)*1000+int(fold)*100+k*5,nm)
            for j,i in enumerate(hi):
                r={'repeat':int(rep),'fold':int(fold),'model':arm,
                   'variant_key':str(train.variant_key.iloc[i]),
                   'protein_position':int(pv[i]),
                   'observed_pp':float(y[i]),'predicted_pp':float(pred[j])}
                if tfeat is not None:
                    for c,target in enumerate(TEACHER_TARGETS):
                        r['pred_'+target] = float(tfeat[j,c])
                rows.append(r)
            choices.append({'repeat':int(rep),'fold':int(fold),'model':arm,
                            'selected_depth':int(selected['depth']),
                            'inner_mae_pp':float(selected['inner_mae_pp']),
                            'inner_candidates_json':json.dumps(tries)})
            s.append(f'{arm}={pmacro(y[hi],pred,pv[hi]):.3f}')
        print(f'repeat={rep} fold={fold} held_variants={len(hi)} | '+
              ' | '.join(s)+f' | elapsed={perf_counter()-t:.1f}s',flush=True)
    if args.smoke:
        print('PASS: Smoke outer fold, five nested arms, no files written',flush=True)
        return
    oof=pd.DataFrame(rows)
    check(len(oof)==2*210*len(ARMS) and
          not oof.duplicated(['repeat','model','variant_key']).any(),'OOF cardinality')
    aver=(oof.groupby(['model','variant_key'],as_index=False)
          .agg(protein_position=('protein_position','first'),
               observed_pp=('observed_pp','first'),
               predicted_pp=('predicted_pp','mean'),
               n_repeats=('repeat','nunique')))
    check(len(aver)==len(ARMS)*210 and aver.n_repeats.eq(2).all(), 'Averaging integrity')
    summaries=pd.DataFrame([{'model':arm,**metrics(aver.loc[aver.model.eq(arm)])}
                            for arm in ARMS]).sort_values('position_mae_pp')
    hist=load_historic(root, train)
    boot=bootstrap_comparisons(aver,additional_baseline=hist,boots=args.bootstrap)
    selection=pd.DataFrame(choices)
    print('\n'+'='*108)
    print('PHASE 23D | REPEATED OOF POSITION-MACRO & RESPONDER RANKING')
    print('='*108)
    print(summaries.round(4).to_string(index=False))
    if hist is not None:
        h=aver.loc[aver.model.eq('control80')].copy()
        ph=hist.set_index('variant_key').loc[h.variant_key,'predicted_pp'].to_numpy(float)
        hh=h.copy();hh['predicted_pp']=ph
        print('\nHISTORICAL Phase19I reliability-weighted 80-feature reference (different protocol):')
        print(pd.DataFrame([metrics(hh)]).round(4).to_string(index=False))
    print('\nPAIRED POSITION BOOTSTRAP (exploratory; method-selection unadjusted)')
    print(boot.round(4).to_string(index=False))
    print('\nNESTED DEPTH SELECTION COUNTS')
    print(pd.crosstab(selection.model,selection.selected_depth).to_string())
    # ETI teacher ability is separately audited at 2024-TRAIN exact matches (never used as held teacher labels).
    teacher_rows=oof.loc[oof.model.eq('teacher_multi'),
                        ['repeat','variant_key','pred_eti_acute_iva_pp']]
    teacher_avg=teacher_rows.groupby('variant_key').pred_eti_acute_iva_pp.mean().rename('teacher_eti_pred_pp')
    exact=aux[['variant_key','protein_position','eti_acute_iva_pp']].merge(
        teacher_avg, on='variant_key',how='inner',validate='one_to_one')
    check(len(exact)==66, f'Unexpected exact 2024-TRAIN/2026-TRAIN overlap count: {len(exact)}')
    teacher_mae=float(np.mean(np.abs(exact.eti_acute_iva_pp-exact.teacher_eti_pred_pp)))
    teacher_rho=float(spearmanr(exact.eti_acute_iva_pp,exact.teacher_eti_pred_pp).statistic)
    print('\nTRAIN-ONLY DIAGNOSTIC: ETI teacher accuracy at 66 exact ETI/VTI substitutions')
    print(f'position-held-out teacher ETI MAE={teacher_mae:.3f} pp; Spearman rho={teacher_rho:+.3f} (n=66)')
    print('ETI labels at target positions were excluded before every teacher prediction.')
    print('Do not confuse this with Phase23C measured-ETI transfer; Phase23D is mutation-only.')
    out=root/'results/biology/phase23d';out.mkdir(parents=True,exist_ok=True)
    files={'oof':out/'phase23d_oof.csv',
           'averaged':out/'phase23d_averaged.csv',
           'summary':out/'phase23d_summary.csv',
           'bootstrap':out/'phase23d_bootstrap.csv',
           'selection':out/'phase23d_inner_selection.csv',
           'teacher_diagnostic':out/'phase23d_teacher_exact_match_diagnostic.csv',
           'report':out/'phase23d_report.json'}
    for name,df in [('oof',oof),('averaged',aver),('summary',summaries),
                    ('bootstrap',boot),('selection',selection),('teacher_diagnostic',exact)]:
        df.to_csv(files[name],index=False)
    report={'phase':'23D','VTI_train_only':True,'2024_ETI_train_only':True,
            'protected_outcomes_opened':False,'teacher_has_measured_ETI_at_query':False,
            'n_vti_variants':210,'n_vti_positions':168,'n_aux_eti':383,
            'n_exact_teacher_diagnostic':int(len(exact)),
            'teacher_diagnostic_mae':teacher_mae,'teacher_diagnostic_spearman':teacher_rho,
            'outer_fold_count':10,'inner_fold_count':3,'teacher_crossfit_groups':3,
            'teacher_targets':list(TEACHER_TARGETS),'arms':list(ARMS),
            'input_provenance_phase23b':provenance,
            'candidate_table_sha256':sha256(root/'results/biology/phase23a/eti2024_train_auxiliary_candidates.csv'),
            'interpretation':'Exploratory VTI development OOF only; no independent study validation'}
    files['report'].write_text(json.dumps(report,indent=2))
    print('\nOUTPUTS:')
    for name,path in files.items():
        print(f' {name}: {path.relative_to(root)}')
    print('PASS: Zero-shot ETI teacher OOF completed; all query positions excluded from ETI teacher fitting.')


if __name__=='__main__':
    main()
