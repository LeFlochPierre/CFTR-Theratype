#!/usr/bin/env python3
"""Phase 27E-2: fixed biological-gate transplant, VTI TRAIN OOF only.

No new model fitting. Reuse the EXACT saved outer-fold OOF scores and the
EXACT Phase 27E-1 expert predictions, then swap only low/high gating scores:
  low  = Phase 27D-1 chemistry (XGBoost)
  high = Phase 27D-3B seq_plus_3d (logistic)
No VTI VAL/TEST labels loaded. This is an exploratory audit on extensively
reused TRAIN folds; do NOT use results to select a final model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, confusion_matrix

ROOT = Path.cwd()
OLD = ROOT / 'results/biology/phase27e/phase27e1_repeated_oof.csv'
D1 = ROOT / 'results/biology/phase27d/phase27d1_repeated_oof.csv'
D3 = ROOT / 'results/biology/phase27d/phase27d3b_repeated_oof.csv'
GATES_OLD = ROOT / 'results/biology/phase27b/phase27b3_repeated_oof.csv'
OUT = ROOT / 'results/biology/phase27e'
KEY = ['repeat','fold','variant_key','protein_position']


def classes(y):
    y = np.asarray(y, float)
    return np.where(y < 10, 0, np.where(y > 40, 2, 1)).astype(int)


def couple_binary(pl, ph):
    """Reproduce the existing Phase27B-3 heuristic: NOT calibrated."""
    pl = np.clip(np.asarray(pl, float), 1e-8, 1.0-1e-8)
    ph = np.clip(np.asarray(ph, float), 1e-8, 1.0-1e-8)
    out = np.column_stack([pl*(1-ph), (1-pl)*(1-ph), ph*(1-pl)])
    out /= out.sum(axis=1, keepdims=True)
    if not np.isfinite(out).all() or not np.allclose(out.sum(axis=1), 1):
        raise ValueError('Invalid gate')
    return out


def assert_unique(d, description):
    if d.duplicated(KEY).any():
        raise ValueError('Duplicated fold/mutation identity: ' + description)
    if len(d) != 420:
        raise ValueError(f'{description}: expected 420 repeated OOF records, got {len(d)}')


def get_scores(path, task, config, result_name):
    if not path.is_file():
        raise FileNotFoundError(f'Run earlier phase first: {path}')
    d = pd.read_csv(path, dtype={'variant_key':str})
    if not {'task','config','score','label',*KEY}.issubset(d.columns):
        raise ValueError(f'Unexpected feature-OOF columns: {path}: {d.columns.tolist()}')
    q = d.loc[d.task.eq(task) & d.config.eq(config), KEY + ['score','label']].copy()
    assert_unique(q, f'{task}/{config}')
    if not np.isfinite(q.score.to_numpy(float)).all() or not q.score.between(0,1).all():
        raise ValueError(f'Invalid score: {task}/{config}')
    return q.rename(columns={'score':result_name,'label':f'label_{result_name}'})


def pos_mae(d, actual, pred):
    z=d[['repeat','protein_position']].copy()
    z['loss']=np.abs(d[pred].to_numpy(float)-d[actual].to_numpy(float))
    return float(z.groupby(['repeat','protein_position']).loss.mean().groupby('protein_position').mean().mean())


def paired_ci(d, target, cand, base, nboot, seed):
    z=d[['repeat','protein_position']].copy()
    z['gain']=(np.abs(d[base]-d[target])-np.abs(d[cand]-d[target])).to_numpy(float)
    pos=z.groupby(['repeat','protein_position']).gain.mean().groupby('protein_position').mean().to_numpy(float)
    rng=np.random.default_rng(seed)
    boot=pos[rng.integers(len(pos), size=(nboot,len(pos)))].mean(axis=1)
    return float(pos.mean()),float(np.quantile(boot,.025)),float(np.quantile(boot,.975))


def evaluate(d, nboot, save=True):
    y=classes(d.observed_corrector)
    assert len(d)==420 and d.variant_key.nunique()==210 and d.protein_position.nunique()==168
    assert set(d.repeat)=={1,2}
    for rep,q in d.groupby('repeat'):
        assert len(q)==210 and q.variant_key.nunique()==210
        assert list(np.bincount(classes(q.observed_corrector),minlength=3))==[52,129,29]
    p=couple_binary(d.low_score,d.high_score)
    ex=d[['expert_low','expert_middle','expert_high']].to_numpy(float)
    global_pred=d.corrector_global.to_numpy(float)
    acute=(d.total_global-d.corrector_global).to_numpy(float)
    if not np.isfinite(ex).all() or not np.isfinite(acute).all():
        raise ValueError('Missing or non-finite expert/global predictions')
    methods={
        'global':global_pred,
        'previous_soft':d.corrector_soft_coupled_binary.to_numpy(float),
        'previous_soft50':d.corrector_soft50_coupled_binary.to_numpy(float),
        'bio_hard':ex[np.arange(len(ex)),p.argmax(axis=1)],
        'bio_soft':np.sum(p*ex,axis=1),
    }
    methods['bio_soft50']=.5*global_pred+.5*methods['bio_soft']
    # Crucial integrity check: original corrector and total remain unchanged.
    if not np.allclose(methods['global']+acute,d.total_global):
        raise ValueError('Changed frozen acute response')
    for name,pred in methods.items():
        d[f'pred_corrector_{name}']=pred
        d[f'pred_total_{name}']=pred+acute

    # Check specialist labels, class probabilities and pairwise identities.
    if not np.array_equal(d.label_low_score.to_numpy(int), (y==0).astype(int)):
        raise ValueError('Low gate label mismatch')
    if not np.array_equal(d.label_high_score.to_numpy(int), (y==2).astype(int)):
        raise ValueError('High gate label mismatch')

    results=[]
    for target in ('corrector','total'):
        for name in methods:
            pred=f'pred_{target}_{name}'
            yy=d[f'observed_{target}']
            subset=classes(d.observed_corrector)
            rec={'target':target,'method':name,'position_mae_pp':pos_mae(d,f'observed_{target}',pred),
                 'variant_mae_pp':float(np.abs(d[pred]-yy).mean()),
                 'signed_bias_pp':float((d[pred]-yy).mean())}
            for c,label in enumerate(['low','middle','high']):
                q=d.loc[subset==c]
                rec[f'{label}_position_mae_pp']=pos_mae(q,f'observed_{target}',pred)
            results.append(rec)
    metrics=pd.DataFrame(results)

    # True-label regime gate is NOT accessed in any deployable model; this step only scores.
    averaged=d.groupby(['variant_key','protein_position'],as_index=False).agg(
        corrector=('observed_corrector','first'),
        low=('low_score','mean'),high=('high_score','mean'))
    cls=classes(averaged.corrector)
    ap=[]
    for score,name,c in [('low','low',0),('high','high',2)]:
        yy=(cls==c).astype(int)
        rr=averaged[score].to_numpy(float)
        hits=int(yy[np.argsort(-rr,kind='stable')[:42]].sum())
        ap.append({'task':name,'positives':int(yy.sum()),'ap':float(average_precision_score(yy,rr)),
                   'auroc':float(roc_auc_score(yy,rr)),'top42_hits':hits})

    gatescore=pd.DataFrame(p,columns=['p_low','p_middle','p_high'])
    predcls=p.argmax(axis=1)
    print('\nA. BIOLOGICAL GATE RANKING (averaged over two repeats)')
    print(pd.DataFrame(ap).round(4).to_string(index=False))
    print('\nB. GATE CONFUSION (record-level; each variant appears twice)')
    print('Rows: true low/middle/high; columns: predicted low/middle/high')
    print(confusion_matrix(y,predcls,labels=[0,1,2]))
    print('Mean gate mass by TRUE corrector regime:')
    print(gatescore.assign(true_regime=y).groupby('true_regime').mean().round(3).to_string())
    print('\nC. FROZEN GLOBAL VS EXPERTS, ONLY THE GATE CHANGED')
    print(metrics.round(4).to_string(index=False))

    contrasts=[]
    for i,target in enumerate(['corrector','total']):
        actual=f'observed_{target}'
        baseline=f'pred_{target}_global'
        for j,cand in enumerate(['bio_hard','bio_soft','bio_soft50','previous_soft','previous_soft50']):
            gain,lo,hi=paired_ci(d,actual,f'pred_{target}_{cand}',baseline,nboot,27000+i*100+j)
            contrasts.append({'target':target,'candidate':cand,'reference':'global','mae_gain_pp':gain,'ci95_low':lo,'ci95_high':hi})
        for j,cand in enumerate(['bio_soft','bio_soft50']):
            ref=f'pred_{target}_'+('previous_soft' if cand=='bio_soft' else 'previous_soft50')
            gain,lo,hi=paired_ci(d,actual,f'pred_{target}_{cand}',ref,nboot,27300+i*100+j)
            contrasts.append({'target':target,'candidate':cand,'reference':'previous_'+cand.removeprefix('bio_'),
                              'mae_gain_pp':gain,'ci95_low':lo,'ci95_high':hi})
    ci=pd.DataFrame(contrasts)
    print('\nD. POSITION-CLUSTER BOOTSTRAP: POSITIVE GAIN = LOWER MAE')
    print(ci.round(4).to_string(index=False))
    print('\nE. INTERPRETATION')
    print('This is a frozen OOF gate-swap diagnostic, NOT a fresh independent validation.')
    print('Gate probabilities are uncalibrated; model/feature choices were explored on these TRAIN folds.')
    print('No classifier/expert/regressor was retrained; no VAL/TEST labels opened.')
    if save:
        OUT.mkdir(parents=True,exist_ok=True)
        d.to_csv(OUT/'phase27e2_biological_gate_swap_oof.csv',index=False)
        metrics.to_csv(OUT/'phase27e2_metrics.csv',index=False)
        ci.to_csv(OUT/'phase27e2_bootstrap.csv',index=False)
        pd.DataFrame(ap).to_csv(OUT/'phase27e2_gate_ranking.csv',index=False)
        print('Saved:',OUT)
    print('PASS Phase 27E-2 completed')


def run(nboot):
    for path in [OLD,D1,D3]:
        if not path.is_file():
            raise FileNotFoundError(f'Required previous-phase OOF CSV missing: {path}')
    original=pd.read_csv(OLD,dtype={'variant_key':str})
    assert_unique(original,'phase27e1')
    keys=set(pd.MultiIndex.from_frame(original[KEY]))
    low=get_scores(D1,'corrector_low','chemistry','low_score')
    high=get_scores(D3,'corrector_high','seq_plus_3d','high_score')
    assert set(pd.MultiIndex.from_frame(low[KEY]))==keys
    assert set(pd.MultiIndex.from_frame(high[KEY]))==keys
    d=original.merge(low,on=KEY,validate='one_to_one').merge(high,on=KEY,validate='one_to_one')
    assert_unique(d,'merged frozen OOF')
    print('='*108)
    print('PHASE 27E-2 | BIOLOGICAL GATE SWAP | VTI TRAIN ONLY')
    print('='*108)
    print('Low specialist: chemistry-enhanced XGB from 27D-1')
    print('High specialist: sequence-pocket + 3D logistic from 27D-3B')
    print('Experts, frozen global predictions, acute component: unchanged from 27E-1')
    print('Original 2x5 protein-position-disjoint OOF predictions; no fit or protected labels')
    evaluate(d,nboot)


def self_test():
    rng=np.random.default_rng(27)
    x=rng.uniform(.02,.9,size=20)
    y=rng.uniform(.02,.9,size=20)
    p=couple_binary(x,y)
    assert p.shape==(20,3) and np.allclose(p.sum(axis=1),1)
    q=pd.DataFrame({'repeat':[1,1,2,2], 'protein_position':[1,2,1,2],
                    'observed':[5.,35.,5.,35.], 'baseline':[9.,30.,9.,30.],
                    'candidate':[5.,35.,5.,35.]})
    gain,lo,hi=paired_ci(q,'observed','candidate','baseline',100,7)
    assert gain==4.5 and lo>=0 and hi>=0
    print('PASS synthetic: gate coupling, repeated-position MAE, bootstrap')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--bootstraps',type=int,default=2000)
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        if args.bootstraps<100:parser.error('Use at least 100 bootstraps')
        run(args.bootstraps)
