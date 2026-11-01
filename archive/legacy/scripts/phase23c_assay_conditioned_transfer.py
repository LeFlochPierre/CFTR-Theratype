#!/usr/bin/env python3
"""Phase 23C: ETI-informed VTI acute-IVA transfer, *conditional on a measured ETI assay*.

Purpose: The 66 exact nucleotide overlaps in Bihler 2024 TRAIN and VTI 2026 TRAIN
have genuinely paired ETI and VTI acute-IVA increments. Does a known ETI response
predict the VTI response for an unseen protein position?

IMPORTANT: This is NOT sequence-only zero-shot theratyping. At prediction time
an ETI functional assay is assumed KNOWN for that exact variant. No ETI label
from 2024 validation/test is accessed. No VTI validation/test outcomes accessed.
The directly supplied ETI measurement at held-out positions is a legitimate
conditional input ONLY for this distinct use case, and must never be used as a
feature in a claimed sequence-only or entirely unassayed-variant benchmark.

Inputs:
  results/biology/phase23a/eti_vti_exact_train_matches.csv
  results/boltz/phase19c/acute_iva_vti_pp_averaged.csv  (optional comparator)
  results/boltz/phase19i/phase19i_averaged.csv  (optional comparator)

Run: python scripts/phase23c_assay_conditioned_transfer.py --self-test
     python scripts/phase23c_assay_conditioned_transfer.py | tee logs/phase23c.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EXPECTED = 66
MIN_MATCHED_POSITIONS = 56
TARGET = 'vti_acute_iva_pp'
ETI = 'eti_acute_iva_pp'
DMSO = 'dmso_acute_iva_pp'
CORRECTOR = 'eti_corrector_fsk_gain_pp'
SEED = 23023


def require(ok, msg):
    if not bool(ok):
        raise AssertionError(msg)


def sha(path):
    h=hashlib.sha256()
    with open(path, 'rb') as f:
        for blk in iter(lambda:f.read(1<<20), b''):
            h.update(blk)
    return h.hexdigest()


def pmacro_mae(y, p, positions):
    return float(pd.DataFrame({'pos':positions,'err':np.abs(np.asarray(y)-np.asarray(p))})
                 .groupby('pos').err.mean().mean())


def position_rank_metrics(y, p, positions, cutoff=20., top_fraction=.2):
    d = pd.DataFrame({'pos': positions,'obs':y,'score':p})
    z = d.groupby('pos').agg(obs=('obs','max'),score=('score','max'))
    label=z.obs.to_numpy()>cutoff; score=z.score.to_numpy()
    k=int(np.ceil(len(z)*top_fraction))
    hits=int(label[np.argsort(-score,kind='stable')[:k]].sum())
    ap=float(average_precision_score(label,score)) if 0<label.sum()<len(label) else float('nan')
    auc=float(roc_auc_score(label,score)) if 0<label.sum()<len(label) else float('nan')
    return dict(n_positive_positions=int(label.sum()), position_ap20=ap, position_auc20=auc,
                top20_screened_positions=k, top20_hits=hits,
                top20_precision=hits/k,top20_recall=hits/max(1,int(label.sum())))


def summarize(y,p,pos):
    y=np.asarray(y,float);p=np.asarray(p,float);pos=np.asarray(pos,int)
    err=np.abs(y-p)
    z={'position_mae_pp':pmacro_mae(y,p,pos),
       'variant_mae_pp':float(err.mean()),
       'rmse_pp':float(np.sqrt(np.mean((y-p)**2))),
       'mean_bias_pred_minus_obs_pp':float(np.mean(p-y)),
       'high20_mae_pp':float(err[y>20].mean()) if (y>20).any() else float('nan'),
       'low2_mae_pp':float(err[y<=2].mean()) if (y<=2).any() else float('nan')}
    return {**z,**position_rank_metrics(y,p,pos)}


def cv_predict(d, n_splits=5):
    """Group-held-out VTI calibration, ETI measurements known at deployment."""
    y=d[TARGET].to_numpy(float)
    eti=d[ETI].to_numpy(float)
    other=d[[ETI,DMSO,CORRECTOR]].to_numpy(float)
    pos=d.protein_position.to_numpy(int)
    models=['direct_eti', 'offset_eti', 'ridge_eti', 'ridge_eti_dmso_corrector',
            'huber_eti']
    preds={k:np.full(len(d),np.nan) for k in models}
    foldassign=np.full(len(d),-1,int)
    folds=list(GroupKFold(n_splits=n_splits).split(other,y,groups=pos))
    for fold,(fi,hi) in enumerate(folds,1):
        fitpos=set(pos[fi]);heldpos=set(pos[hi]);
        require(fitpos.isdisjoint(heldpos),'Held-out protein position present in calibration fit')
        foldassign[hi]=fold
        preds['direct_eti'][hi]=eti[hi]
        shift=np.median(y[fi]-eti[fi])
        preds['offset_eti'][hi]=eti[hi]+shift
        mods={
            'ridge_eti': (make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
                          eti[fi,None],eti[hi,None]),
            'ridge_eti_dmso_corrector': (
                 make_pipeline(StandardScaler(),Ridge(alpha=10.0)), other[fi],other[hi]),
            'huber_eti': (make_pipeline(StandardScaler(),HuberRegressor(alpha=1.0,
                                                                      epsilon=1.35,
                                                                      max_iter=1000)),
                          eti[fi,None],eti[hi,None]),
        }
        for key,(model,xfit,xtest) in mods.items():
            model.fit(xfit,y[fi]); preds[key][hi]=model.predict(xtest)
    require(np.all(foldassign>=0),'CV coverage not complete')
    for model,pr in preds.items():
        require(np.isfinite(pr).all(),f'Nonfinite predictions: {model}')
    return preds,foldassign


def add_historical_comparators(root, d, preds):
    """Restrict pre-existing true VTI OOF predictions to paired TRAIN subset."""
    sources=[
      ('zero_shot_80feature_xgb',
       root/'results/boltz/phase19c/acute_iva_vti_pp_averaged.csv',
       'sequence_structure__xgboost'),
      ('zero_shot_reliability_weighted',
       root/'results/boltz/phase19i/phase19i_averaged.csv',
       'reliability_weighted')]
    for name,path,model in sources:
        if not path.is_file():
            print('NOTE: comparator absent:',path.relative_to(root)); continue
        a=pd.read_csv(path,dtype={'variant_key':str})
        if not {'variant_key','model','predicted_pp'}.issubset(a.columns):
            print('WARNING: unrecognised comparator schema:',path); continue
        a=a.loc[a.model.eq(model), ['variant_key','predicted_pp']]
        if a.variant_key.duplicated().any():
            print('WARNING: historical averaged file contains duplicate keys:',path);continue
        merged=d[['variant_key']].merge(a,how='left',on='variant_key',validate='one_to_one')
        if len(merged)!=len(d) or merged.predicted_pp.isna().any():
            print('WARNING: historical comparator unavailable for all paired variants:',name);continue
        preds[name]=merged.predicted_pp.to_numpy(float)
        print('Loaded historical held-out zero-shot comparator:',name)


def bootstrap_paired(y,pos,pred_a,pred_b,nboot):
    deltas=pd.DataFrame({'pos':pos,
                         'd':np.abs(y-pred_b)-np.abs(y-pred_a)})\
              .groupby('pos').d.mean().to_numpy(float)
    rng=np.random.default_rng(SEED+147)
    draws=rng.integers(0,len(deltas),size=(nboot,len(deltas)))
    m=deltas[draws].mean(axis=1)
    return {'gain_vs_direct_eti_pp':float(deltas.mean()),
            'ci95_lo':float(np.quantile(m,.025)),
            'ci95_hi':float(np.quantile(m,.975)),
            'fraction_boot_positive':float(np.mean(m>0))}


def self_test():
    rng=np.random.default_rng(77)
    d=pd.DataFrame({'protein_position':np.repeat(np.arange(40),2),
                    ETI:rng.uniform(-1,50,size=80),
                    DMSO:rng.uniform(-1,20,size=80),
                    CORRECTOR:rng.uniform(1,65,size=80)})
    d[TARGET]=.8*d[ETI]+.2*d[DMSO]+rng.normal(0,3,size=len(d))
    d['variant_key']=[f'v{i}' for i in range(len(d))]
    preds,f=cv_predict(d)
    for k,v in preds.items():
        assert len(v)==80 and np.isfinite(v).all()
    assert (f>=1).all() and len(np.unique(f))==5
    print('PASS: Grouped position-disjoint calibration, all models and screening metrics')
    print(pd.DataFrame([{'model':k,**summarize(d[TARGET],v,d.protein_position)}
                        for k,v in preds.items()]).round(3).to_string(index=False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--bootstrap',type=int,default=10000)
    args=ap.parse_args()
    if args.self_test:
        self_test();return
    from cftr_theratype.transfer import ROOT
    root=Path(ROOT)
    source=root/'results/biology/phase23a/eti_vti_exact_train_matches.csv'
    require(source.is_file(),f'Cannot find {source}')
    d=pd.read_csv(source,dtype={'variant_key':str})
    required={'variant_key','protein_position',TARGET,ETI,DMSO,CORRECTOR}
    require(required.issubset(d),f'Missing source columns: {required-set(d.columns)}')
    require(len(d)==EXPECTED and d.variant_key.is_unique,'Expected 66 exact paired TRAIN variants')
    require(d.protein_position.nunique()==MIN_MATCHED_POSITIONS,
            'Expected 56 paired TRAIN protein positions')
    require(np.isfinite(d[[TARGET,ETI,DMSO,CORRECTOR]].to_numpy(float)).all(),
            'Paired outcomes contain NaNs or infinities')
    print('='*108)
    print('PHASE 23C | ETI-ASSAY-CONDITIONED VTI ACUTE IVA TRANSFER | 66 MATCHED TRAIN VARIANTS')
    print('='*108)
    print('Deployment assumption: a real ETI assay response for each target variant is ALREADY available.')
    print('This is NOT zero-shot mutation-only prediction. ETI input may be known for CV held-out positions.')
    print('2024 TRAIN and 2026 VTI TRAIN ONLY; no protected outcomes opened; all VTI calibrators position-held-out.')
    print('Match sample: 66 exact nucleotide variants / 56 positions; few strong responders.')
    print('Uncalibrated ETI response is the key no-training comparator.')
    y=d[TARGET].to_numpy(float);pos=d.protein_position.to_numpy(int)
    eti=d[ETI].to_numpy(float)
    print('\nRAW ETI/VTI AGREEMENT')
    print(f'Spearman: {spearmanr(eti,y).statistic:+.3f}')
    print(f'Median absolute paired difference: {np.median(np.abs(y-eti)):.3f} pp')
    print(f'ETI >20 count={(eti>20).sum()}, VTI >20 count={(y>20).sum()}')
    print(pd.crosstab(eti>20,y>20,rownames=['ETI >20'],colnames=['VTI >20']).to_string())
    preds,foldassign=cv_predict(d)
    add_historical_comparators(root,d,preds)
    print('\n'+'='*108)
    print('MATCHED-SUBSET METRICS (do NOT compare as if this were the full 210-variant dataset)')
    print('='*108)
    rows=[{'model':name,**summarize(y,p,pos)} for name,p in preds.items()]
    s=pd.DataFrame(rows).sort_values('position_mae_pp')
    print(s.round(4).to_string(index=False))
    print('\nPAIRED POSITION BOOTSTRAP VS UNCALIBRATED ETI MEASUREMENT')
    bs=pd.DataFrame([{'model':name,**bootstrap_paired(y,pos,p,preds['direct_eti'],args.bootstrap)}
                    for name,p in preds.items() if name!='direct_eti'])
    print(bs.round(4).to_string(index=False))
    print('\nLARGEST DIRECT ETI-VTI DISAGREEMENTS (TRAIN ONLY)')
    x=d[['variant_key','protein_position',TARGET,ETI,DMSO,CORRECTOR]].copy()
    x['eti_minus_vti_pp']=x[ETI]-x[TARGET]
    print(x.reindex(x.eti_minus_vti_pp.abs().sort_values(ascending=False).index)
          .head(12).round(3).to_string(index=False))
    out=root/'results/biology/phase23c';out.mkdir(parents=True,exist_ok=True)
    predfile=out/'phase23c_paired_cv_predictions.csv'
    resultfile=out/'phase23c_paired_cv_summary.csv'
    bootfile=out/'phase23c_bootstrap.csv'
    o=d[['variant_key','protein_position',TARGET,ETI,DMSO,CORRECTOR]].copy()
    o['cv_group_fold']=foldassign
    for name,p in preds.items(): o['pred_'+name]=p
    o.to_csv(predfile,index=False);s.to_csv(resultfile,index=False);bs.to_csv(bootfile,index=False)
    (out/'phase23c_report.json').write_text(json.dumps({
        'source':str(source.relative_to(root)),'sha256':sha(source),
        'n_variants':len(d),'n_positions':d.protein_position.nunique(),
        'n_vti_high20':int((y>20).sum()),'n_eti_high20':int((eti>20).sum()),
        'n_group_folds':5,'assay_conditioned':True,
        '2024_validation_test_outcomes_opened':False,
        '2026_validation_test_outcomes_opened':False,
        'caveat':'Known ETI response at prediction; NOT mutation-only zero-shot evaluation; 66 overlap variants selected non-randomly',
        'bootstrap':'Exploratory, model comparisons after examining training data'
    },indent=2))
    print('\nOUTPUTS:')
    for path in (predfile,resultfile,bootfile,out/'phase23c_report.json'):
        print(' ',path.relative_to(root))
    print('PASS: Measured-ETI conditional transfer audit, 56 position-held-out calibration groups, no protected outcomes opened')

if __name__=='__main__':
    main()
