#!/usr/bin/env python3
"""Phase 30A | Target definition + baseline/IVA assay information ladder.

Exploratory DEVELOPMENT-only diagnostic for CFTR-Theratype.
  * Exact historical 2x5 protein-position-disjoint VTI TRAIN folds, 210 records.
  * Reads assay parquet with pyarrow predicate filters using TRAIN variant keys
    only. Protected VAL/TEST values are never loaded.
  * Evaluates untreated baseline, corrector rescue, total VTI rescue, and
    absolute VTI-treated function with matched fold-conditioned baselines.
  * Compares mutation-only versus assay-conditioned scenarios; the latter
    REQUIRE additional laboratory measurements and are NOT mutation-only models.
  * Makes subtraction coupling explicit and includes absolute treated endpoint.
  * Reads existing frozen corrector OOF for exact regression parity.
  * Prior ETI-labelled data are deliberately NOT loaded: provenance, coverage
    and treatment comparability need a separate audit before an ETI ladder.

From repo root:
  python scripts/phase30a_target_information_ladder.py --self-test
  python -u scripts/phase30a_target_information_ladder.py --bootstraps 2000
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path.cwd()
PARQUET=ROOT/'data/processed/bihler_2026_vti.parquet'
REFERENCE=ROOT/'results/biology/phase26h2/phase26h2_oof.csv'
OUT=ROOT/'results/biology/phase30a'
KEY=['repeat','fold','variant_key','protein_position']
BASE='baseline_fsk_pct_wt'
CORRECTOR_TREAT='vnz_tez_fsk_pct_wt'
IVA='iva_treated_pct_wt'
VTI_TREAT='vti_treated_pct_wt'
CORRECTOR='corrector_rescue_pp'
TOTAL='vti_rescue_pp'
TARGETS={
    'baseline':BASE,
    'corrector':CORRECTOR,
    'total_rescue':TOTAL,
    'treated_absolute':VTI_TREAT,
}
PREDICTION_CONFIGS={
    'baseline':['train_median','mutation_only'],
    'corrector':['train_median','mutation_only','baseline_only','mutation_plus_baseline','baseline_treated_anchor'],
    'total_rescue':['train_median','mutation_only','baseline_only','mutation_plus_baseline','mutation_plus_baseline_iva','baseline_treated_anchor'],
    'treated_absolute':['train_median','mutation_only','baseline_only','mutation_plus_baseline','mutation_plus_baseline_iva'],
}


def sha(path):
    d=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):d.update(b)
    return d.hexdigest()


def safe_spearman(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if np.std(x)<1e-12 or np.std(y)<1e-12:return np.nan
    return float(pd.Series(x).corr(pd.Series(y),method='spearman'))


def pos_mae(df,target,pred):
    a=df[['protein_position',target,pred]].copy()
    a['e']=(a[target]-a[pred]).abs()
    return float(a.groupby('protein_position').e.mean().mean())


def pos_mean(df,column):
    return float(df.groupby('protein_position')[column].mean().mean())


def compare_pairs(df, target, cand, ref, nboot, rng):
    a=df[['repeat','protein_position',target,cand,ref]].copy()
    a['gain']=(a[ref]-a[target]).abs()-(a[cand]-a[target]).abs()
    positions=a.groupby(['repeat','protein_position']).gain.mean().groupby('protein_position').mean().to_numpy(float)
    if not len(positions) or not np.isfinite(positions).all():raise ValueError('Invalid position gains')
    ix=rng.integers(0,len(positions),size=(nboot,len(positions)))
    samples=positions[ix].mean(axis=1)
    return {'target':target,'candidate':cand,'reference':ref,'gain_pp':float(positions.mean()),
            'ci95_low':float(np.percentile(samples,2.5)),
            'ci95_high':float(np.percentile(samples,97.5)),
            'positions':len(positions)}


def ridge_baseline(xtr,ytr,xte):
    model=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
    model.fit(xtr.reshape(-1,1),ytr)
    return np.asarray(model.predict(xte.reshape(-1,1)),float)


def fit_xgb(xtr,ytr,xte,model_factory):
    # The constant bias column is part of the previously frozen implementation.
    mdl=model_factory()
    ztr=np.column_stack([xtr,np.ones(len(xtr))])
    zte=np.column_stack([xte,np.ones(len(xte))])
    mdl.fit(ztr,ytr)
    p=np.asarray(mdl.predict(zte),float)
    if len(p)!=len(xte) or not np.isfinite(p).all():raise ValueError('Bad model prediction')
    return p


def audit_assay(d):
    for c in [BASE,IVA,CORRECTOR_TREAT,VTI_TREAT,CORRECTOR,TOTAL]:
        if c not in d or not np.isfinite(d[c].to_numpy(float)).all():
            raise ValueError(f'Missing/nonfinite assay outcome: {c}')
    identities={
        'corrector=vnztez_minus_baseline':np.max(np.abs(d[CORRECTOR_TREAT]-d[BASE]-d[CORRECTOR])),
        'total=vti_treated_minus_baseline':np.max(np.abs(d[VTI_TREAT]-d[BASE]-d[TOTAL])),
        'vti_minus_vnztez_is_acute_component':np.max(np.abs(d[TOTAL]-d[CORRECTOR]-(d[VTI_TREAT]-d[CORRECTOR_TREAT]))),
    }
    if max(identities.values())>1e-5:raise ValueError(f'Invalid assay identities: {identities}')
    return {k:float(v) for k,v in identities.items()}


def eval_fold(x,tr,te,df,model_factory):
    """Pure fold-level modelling. No observed held-out outcome is used at fit time."""
    btr=df[BASE].to_numpy(float)[tr]
    bte=df[BASE].to_numpy(float)[te]
    iva=df[IVA].to_numpy(float)
    result={}
    # Direct absolute treated model is used as an anchored prediction for rescue.
    treated_from_baseline=fit_xgb(np.column_stack([x[tr],btr]),
                                  df[VTI_TREAT].to_numpy(float)[tr],
                                  np.column_stack([x[te],bte]),model_factory)
    corr_treated_from_baseline=fit_xgb(np.column_stack([x[tr],btr]),
                                       df[CORRECTOR_TREAT].to_numpy(float)[tr],
                                       np.column_stack([x[te],bte]),model_factory)
    for name,col in TARGETS.items():
        y=df[col].to_numpy(float)
        p={'train_median':np.full(len(te),float(np.median(y[tr]))),
           'mutation_only':fit_xgb(x[tr],y[tr],x[te],model_factory)}
        if name!='baseline':
            p['baseline_only']=ridge_baseline(btr,y[tr],bte)
            if name=='treated_absolute':
                p['mutation_plus_baseline']=treated_from_baseline
            else:
                p['mutation_plus_baseline']=fit_xgb(np.column_stack([x[tr],btr]),y[tr],
                                                  np.column_stack([x[te],bte]),model_factory)
        if name in ['treated_absolute','total_rescue']:
            pred=fit_xgb(np.column_stack([x[tr],btr,iva[tr]]),y[tr],
                         np.column_stack([x[te],bte,iva[te]]),model_factory)
            p['mutation_plus_baseline_iva']=pred
        if name=='corrector':
            p['baseline_treated_anchor']=corr_treated_from_baseline-bte
        if name=='total_rescue':
            p['baseline_treated_anchor']=treated_from_baseline-bte
        if set(p)!=set(PREDICTION_CONFIGS[name]):
            raise ValueError(f'Unexpected configs {name}: {set(p)}')
        for cfg,values in p.items():
            if len(values)!=len(te) or not np.isfinite(values).all():
                raise ValueError(f'Bad predictions {name}/{cfg}')
        result[name]=p
    return result


def evaluate(df,x,folds,reference,model_factory,boot=1000):
    """All model fits happen inside outer folds. No tuning from held-out labels."""
    keys=df.variant_key.to_numpy(str); positions=df.protein_position.to_numpy(int)
    if x.shape!=(len(df),80):raise ValueError('Expected 80 mutation-only descriptors')
    if not np.isfinite(x[:,:68]).all() or np.isinf(x).any():raise ValueError('Invalid input descriptors')
    allrows=[];fqc=[]
    for i,(rep,fold,trkeys,tekeys) in enumerate(folds,1):
        tr=np.flatnonzero(np.isin(keys,list(trkeys)))
        te=np.flatnonzero(np.isin(keys,list(tekeys)))
        if len(tr)+len(te)!=len(df) or set(positions[tr])&set(positions[te]):
            raise ValueError('Protein-position fold leakage')
        if len(tr)==0 or len(te)==0:raise ValueError('Empty fold')
        sel=reference[(reference.repeat==rep)&(reference.fold==fold)].set_index('variant_key')
        if set(sel.index)!=set(keys[te]):raise ValueError('Historical fold membership mismatch')
        ref=sel.loc[keys[te]]
        if not np.allclose(ref.observed_corrector.to_numpy(float),df[CORRECTOR].to_numpy(float)[te],atol=1e-8,rtol=0):
            raise ValueError('Historical corrector outcomes differ')
        if not np.allclose(ref.observed_total.to_numpy(float),df[TOTAL].to_numpy(float)[te],atol=1e-8,rtol=0):
            raise ValueError('Historical total outcomes differ')
        preds=eval_fold(x,tr,te,df,model_factory)
        parity=np.max(np.abs(preds['corrector']['mutation_only']-ref.vti_fixed.to_numpy(float)))
        if parity>1e-5:raise ValueError(f'Original XGB prediction parity FAIL: {parity}')
        for j,r in enumerate(te):
            row={'repeat':rep,'fold':fold,'variant_key':keys[r],
                 'protein_position':int(positions[r])}
            for name,col in TARGETS.items():
                row['observed_'+name]=float(df[col].iloc[r])
                for cfg,p in preds[name].items():row['pred_'+name+'__'+cfg]=float(p[j])
            row['frozen_total_reconstructed']=float(ref.vti_fixed_total.iloc[j])
            allrows.append(row)
        fqc.append({'repeat':rep,'fold':fold,'train_records':len(tr),'test_records':len(te),
                    'train_positions':len(set(positions[tr])),
                    'test_positions':len(set(positions[te])),
                    'xgb_corrector_frozen_parity_max_abs':float(parity)})
        print(f'  Fold {i:02d}/{len(folds)} repeat={rep} fold={fold}: corrector parity={parity:.2e}',flush=True)
    out=pd.DataFrame(allrows)
    if len(out)!=2*len(df) or out.duplicated(KEY).any():raise ValueError('OOF rows malformed')
    if any(len(z)!=len(df) or set(z.variant_key)!=set(keys) for _,z in out.groupby('repeat')):
        raise ValueError('Each repeat must cover every TRAIN record exactly once')
    metrics=[];paired=[]
    for name,cs in PREDICTION_CONFIGS.items():
        truth='observed_'+name
        y=out.groupby('variant_key')[truth].first().to_numpy(float)
        for cfg in cs:
            p='pred_'+name+'__'+cfg
            metrics.append({'target':name,'information':cfg,
                            'position_mae_pp':float(np.mean([pos_mae(s,truth,p) for _,s in out.groupby('repeat')])),
                            'variant_mae_pp':float(np.abs(out[p]-out[truth]).mean()),
                            'signed_bias_pp':float((out[p]-out[truth]).mean()),
                            'pred_sd_pp':float(out.groupby('variant_key')[p].mean().std(ddof=0)),
                            'target_sd_pp':float(y.std(ddof=0))})
            if cfg not in ('train_median','mutation_only'):
                paired.append(compare_pairs(out,truth,p,'pred_'+name+'__mutation_only',boot,
                                            np.random.default_rng(3000+len(paired))))
    # Mutation-only absolute treated and total rescue can differ by target construction.
    m=pd.DataFrame(metrics)
    c=pd.DataFrame(paired)
    return out,m,c,pd.DataFrame(fqc)


def strict_load():
    import pyarrow.parquet as pq
    from cftr_theratype.datasets.eti_auxiliary import load_data
    v,x,_,_,folds=load_data(ROOT)
    if len(v)!=210 or v.variant_key.nunique()!=210 or v.protein_position.nunique()!=168 or len(folds)!=10:
        raise ValueError('Frozen TRAIN cohort changed')
    keys=v.variant_key.astype(str).tolist()
    schema=set(pq.ParquetFile(PARQUET).schema.names)
    needed=['variant_key',BASE,IVA,CORRECTOR_TREAT,VTI_TREAT,CORRECTOR,TOTAL]
    if not set(needed).issubset(schema):
        raise ValueError(f'Assay parquet missing required columns: {set(needed)-schema}')
    raw=pd.read_parquet(PARQUET,columns=needed,filters=[('variant_key','in',keys)])
    raw.variant_key=raw.variant_key.astype(str)
    if len(raw)!=210 or raw.variant_key.duplicated().any() or set(raw.variant_key)!=set(keys):
        raise ValueError('TRAIN predicate returned missing/extra/duplicated records: protected split risk')
    df=v[['variant_key','protein_position','corrector_rescue_pp','vti_rescue_pp']].copy()
    df.variant_key=df.variant_key.astype(str)
    df=df.merge(raw,on='variant_key',validate='one_to_one',suffixes=('_frozen',''))
    if len(df)!=210 or df[needed[1:]].isna().any().any():raise ValueError('Missing data on TRAIN')
    for col in needed[1:]:df[col]=pd.to_numeric(df[col],errors='raise')
    if not np.allclose(df[CORRECTOR],df[CORRECTOR+'_frozen'],rtol=0,atol=1e-6):
        raise ValueError('Frozen corrector target parity')
    if not np.allclose(df[TOTAL],df[TOTAL+'_frozen'],rtol=0,atol=1e-6):
        raise ValueError('Frozen total target parity')
    audit_assay(df)
    ref=pd.read_csv(REFERENCE,dtype={'variant_key':str})
    if len(ref)!=420 or ref.duplicated(KEY).any() or not set(KEY+['vti_fixed','vti_fixed_total','observed_corrector','observed_total']).issubset(ref):
        raise ValueError('Frozen OOF missing/invalid required columns')
    if set(ref.variant_key)!=set(keys) or set(ref.repeat.unique())!={1,2}:
        raise ValueError('Frozen OOF variant/repeat mismatch')
    return df,np.asarray(x,float),folds,ref


def association_table(df):
    vals={k:df[col].to_numpy(float) for k,col in TARGETS.items()}
    vals['vnz_tez_absolute']=df[CORRECTOR_TREAT].to_numpy(float)
    vals['iva_absolute']=df[IVA].to_numpy(float)
    rows=[]
    for a in ['baseline','corrector','total_rescue','treated_absolute']:
        for b in ['baseline','corrector','total_rescue','treated_absolute','vnz_tez_absolute','iva_absolute']:
            if a==b:continue
            rows.append({'outcome_a':a,'outcome_b':b,
                         'spearman':safe_spearman(vals[a],vals[b]),
                         'pearson':float(np.corrcoef(vals[a],vals[b])[0,1]),
                         'arithmetic_overlap':bool((a=='baseline' and b in ('corrector','total_rescue','vnz_tez_absolute')) or
                                                  (b=='baseline' and a in ('corrector','total_rescue')) or
                                                  (a=='corrector' and b=='vnz_tez_absolute') or
                                                  (a=='total_rescue' and b=='treated_absolute'))})
    return pd.DataFrame(rows)


def run(args):
    from cftr_theratype.models.regressors import phase26d_model
    from cftr_theratype.models.auxiliary import SPEC as FROZEN_XGB_SPEC
    df,x,folds,ref=strict_load()
    identities=audit_assay(df)
    print('='*113)
    print('PHASE 30A | VTI TARGET DEFINITIONS / BIOLOGICAL INFORMATION LADDER | TRAIN ONLY')
    print('='*113)
    print('210 TRAIN records, 168 protein positions; original 2x5 outer position-disjoint folds.')
    print('Assay data filtered by TRAIN IDs BEFORE materialization. No VAL/TEST outcomes; no ETI labels.')
    print('ASSAY-INPUT MODELS ARE NOT MUTATION-ONLY. Baseline is part of subtraction outcomes: coupling warning.')
    print('IVA-only FSK+IVA is measured in the same experimental source; NOT a prior ETI assay.')
    print('\nA. EXACT TARGET ARITHMETIC')
    for k,v in identities.items():print(f'  {k}: max absolute discrepancy={v:.3g} pp')
    print('\nB. OUTCOME DISTRIBUTIONS')
    distributions=[]
    for name,col in TARGETS.items():
        y=df[col].to_numpy(float)
        distributions.append({'target':name,'mean_pp':float(np.mean(y)),'median_pp':float(np.median(y)),
                              'sd_pp':float(np.std(y,ddof=0)),'p10_pp':float(np.quantile(y,.1)),
                              'p90_pp':float(np.quantile(y,.9))})
    desc=pd.DataFrame(distributions)
    print(desc.round(3).to_string(index=False))
    print('\nC. SELECTED DESCRIPTIVE ASSOCIATIONS (SHARED ASSAY TERMS NOT CAUSAL)')
    asc=association_table(df)
    wanted=asc.loc[(asc.outcome_a=='baseline') & asc.outcome_b.isin(['corrector','total_rescue','treated_absolute']) |
                   (asc.outcome_a=='corrector') & asc.outcome_b.isin(['total_rescue','treated_absolute'])]
    print(wanted.round(3).to_string(index=False))
    print('\nD. SAME-FOLD INFORMATION LADDER — fitting locked algorithm parameters')
    out,metrics,paired,qc=evaluate(df,x,folds,ref,lambda:phase26d_model(FROZEN_XGB_SPEC),boot=args.bootstraps)
    print('\nE. GROUPED OOF MAE BY TARGET AND INFORMATION STAGE')
    print(metrics.round(4).to_string(index=False))
    print('\nF. ADDITIONAL ASSAY INFORMATION VS MUTATION-ONLY (positive gain=better)')
    print(paired.round(4).to_string(index=False))
    print('\nG. POSITION GROUPING / PARITY')
    print(qc.to_string(index=False))
    print('\nINTERPRETATION: Cross-target MAEs are NOT directly comparable because outcome distributions differ.')
    print('Baseline-corrector and baseline-rescue associations have mathematical coupling.')
    print('Total and treated-absolute outcomes must be interpreted together for baseline-informed models.')
    print('Prior ETI: not included; requires a separate overlap and prior-treatment provenance audit.')
    print('Repeatedly reused TRAIN folds = exploratory development evidence only.')
    OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in {'outcome_distributions':desc,'associations':asc,'oof':out,'metrics':metrics,
                       'paired_gains':paired,'fold_qc':qc}.items():
        frame.to_csv(OUT/f'phase30a_{name}.csv',index=False)
    (OUT/'phase30a_provenance.json').write_text(json.dumps({
        'assay_parquet_sha256':sha(PARQUET),'frozen_oof_sha256':sha(REFERENCE),
        'script_sha256':sha(Path(__file__)),'n_train':210,'positions':168,'folds':10,
        'label_identities_max_abs':identities,'configurations':PREDICTION_CONFIGS,
        'status':'Exploratory TRAIN only; no protected labels; no tuning; ETI not used',
        'cautions':['Assay-conditioned vs mutation-only applicability differs',
                    'Baseline part of subtraction targets; arithmetic coupling',
                    'IVA assay not prior ETI; cross-target MAE scales not directly comparable']
    },indent=2)+'\n')
    print('PASS Phase 30A; outputs:',OUT)


def self_test():
    # Identity check catches wrong component definitions.
    d=pd.DataFrame({BASE:[2.,3.,4.],IVA:[4.,5.,8.],CORRECTOR_TREAT:[7.,10.,9.],
                    VTI_TREAT:[9.,12.,15.],CORRECTOR:[5.,7.,5.],TOTAL:[7.,9.,11.]})
    assert all(v==0 for v in audit_assay(d).values())
    bad=d.copy();bad.loc[1,TOTAL]=5
    try:audit_assay(bad)
    except ValueError:pass
    else:raise AssertionError('Target arithmetic mismatch not caught')
    class MeanModel:
        def fit(self,x,y):self.ymean=float(np.mean(y));return self
        def predict(self,x):return np.full(len(x),self.ymean)
    rng=np.random.default_rng(30)
    n=80;x=rng.normal(size=(n,80));x[:,68:]=np.nan
    ydf=pd.DataFrame({BASE:rng.uniform(0,15,n),IVA:rng.uniform(0,20,n),
                      'variant_key':[f'syn{i}' for i in range(n)],
                      'protein_position':np.repeat(np.arange(40),2)})
    ydf[CORRECTOR_TREAT]=ydf[BASE]+rng.uniform(0,50,n)
    ydf[VTI_TREAT]=ydf[CORRECTOR_TREAT]+rng.uniform(0,10,n)
    ydf[CORRECTOR]=ydf[CORRECTOR_TREAT]-ydf[BASE]
    ydf[TOTAL]=ydf[VTI_TREAT]-ydf[BASE]
    tr=np.flatnonzero(ydf.protein_position<25)
    te=np.flatnonzero(ydf.protein_position>=25)
    p=eval_fold(x,tr,te,ydf,MeanModel)
    assert {k:set(v)==set(PREDICTION_CONFIGS[k]) for k,v in p.items()}=={k:True for k in TARGETS}
    assert len(p['corrector']['mutation_only'])==len(te)
    assert np.isclose(p['total_rescue']['baseline_treated_anchor'].mean(),ydf[VTI_TREAT].iloc[tr].mean()-ydf[BASE].iloc[te].mean())
    tmp=pd.DataFrame({'repeat':[1,2,1,2],'protein_position':[1,1,2,2],
        'obs':[1.,1.,3.,3.],'c':[1.,1.,3.,3.],'r':[2.,2.,2.,2.]})
    gain=compare_pairs(tmp,'obs','c','r',100,np.random.default_rng(30))
    assert np.isclose(gain['gain_pp'],1.0)
    print('PASS synthetic: target arithmetic, leakage-resistant fold inputs, four endpoints, bootstrap')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--bootstraps',type=int,default=2000)
    a=ap.parse_args()
    if a.self_test:self_test()
    else:
        if a.bootstraps<100:ap.error('--bootstraps must be >=100')
        self_test();run(a)
