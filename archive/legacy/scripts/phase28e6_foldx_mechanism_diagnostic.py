#!/usr/bin/env python3
"""Phase 28E-6 | TRAIN-only, descriptive FoldX stability/response diagnostic.

Read existing Phase 28E-5 outputs. NO model fitting, no VAL/TEST reads,
no feature selection. The output is exploratory, not causal inference.
Run from ~/projects/CFTR-Theratype.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT=Path.cwd()
SRC=ROOT/'results/biology/phase28e/phase28e5'
OUT=ROOT/'results/biology/phase28e/phase28e6'
DOMAINS=[('TMD1',81,365),('NBD1',423,646),('R',654,831),('TMD2',859,1155),('NBD2',1210,1443)]

def domain(p):
    return next((s for s,a,b in DOMAINS if a<=int(p)<=b),'other')

def hashfile(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''):h.update(chunk)
    return h.hexdigest()

def verify_and_merge(oof,feat):
    ocols={'repeat','fold','variant_key','protein_position','observed_corrector',
           'corrector_global','corrector_foldx_consensus'}
    fcols={'variant_key','protein_position','mutation','ddg_consensus','ddg_7sv7',
           'ddg_8eig','ddg_mean_repeat_sd','ddg_disagreement','n_templates'}
    if not ocols.issubset(oof):raise ValueError(f'OOF missing: {ocols-set(oof)}')
    if not fcols.issubset(feat):raise ValueError(f'FoldX features missing: {fcols-set(feat)}')
    oof=oof.copy();feat=feat.copy()
    for df in [oof,feat]:
        df['variant_key']=df['variant_key'].astype(str)
        df['protein_position']=pd.to_numeric(df['protein_position'],errors='raise').astype(int)
    if len(oof)!=420 or len(feat)!=210 or oof.variant_key.nunique()!=210 or feat.variant_key.nunique()!=210:
        raise ValueError('Unexpected TRAIN cardinality (expected 420 OOF, 210 features, 210 unique variant keys)')
    if feat.variant_key.duplicated().any() or oof.duplicated(['repeat','variant_key']).any():
        raise ValueError('Duplicate variant features or repeat OOF rows')
    if set(oof.repeat)!={1,2} or len(oof.groupby(['repeat','fold']))!=10:
        raise ValueError('Expected 2x5 frozen grouped OOF folds')
    if any(len(s)!=210 for _,s in oof.groupby('repeat')) or oof.protein_position.nunique()!=168:
        raise ValueError('Repeated OOF membership/position count mismatch')
    if oof.groupby(['repeat','protein_position']).fold.nunique().max()!=1:
        raise ValueError('Protein position leakage across folds')
    if oof.groupby('variant_key').protein_position.nunique().max()!=1 or oof.groupby('variant_key').observed_corrector.nunique().max()!=1:
        raise ValueError('Repeated outcome/position inconsistency')
    for c in ['observed_corrector','corrector_global','corrector_foldx_consensus']:
        if not np.isfinite(oof[c].astype(float)).all(): raise ValueError(f'Non-finite OOF value: {c}')
    keep=['observed_corrector','corrector_global','corrector_foldx_consensus']
    wide=oof.groupby(['variant_key','protein_position'],as_index=False)[keep].mean()
    wide=wide.merge(feat[list(fcols)],on=['variant_key','protein_position'],how='left',validate='one_to_one',indicator=True)
    if (wide['_merge']!='both').any():raise ValueError('Missing FoldX feature variant')
    wide=wide.drop(columns='_merge')
    if not wide.n_templates.isin([0,1,2]).all():raise ValueError('Invalid FoldX structure counts')
    if ((wide.n_templates>0)&(~np.isfinite(wide.ddg_consensus))).any() or ((wide.n_templates==0)&wide.ddg_consensus.notna()).any():
        raise ValueError('FoldX missingness/availability inconsistency')
    if ((wide.n_templates==2)&(wide.ddg_disagreement.isna())).any():raise ValueError('Missing paired disagreement')
    wide['regime']=np.select([wide.observed_corrector<10,wide.observed_corrector>40],['low','high'],default='middle')
    if wide.regime.value_counts().to_dict()!={'middle':129,'low':52,'high':29}:raise ValueError('Unexpected regime counts')
    wide['domain']=wide.protein_position.map(domain)
    wide['error_global']=wide.observed_corrector-wide.corrector_global
    wide['mae_gain']=np.abs(wide.corrector_global-wide.observed_corrector)-np.abs(wide.corrector_foldx_consensus-wide.observed_corrector)
    wide['qc_reliable']=(wide.n_templates>0)&(wide.ddg_mean_repeat_sd<=2)&(
       (wide.n_templates==1)|(wide.ddg_disagreement<=5))&(wide.ddg_consensus.abs()<=20)
    return wide

def corr(x,y):
    a=np.asarray(x,float);b=np.asarray(y,float)
    keep=np.isfinite(a)&np.isfinite(b)
    if keep.sum()<5 or np.unique(a[keep]).size<3 or np.unique(b[keep]).size<3:return np.nan
    return float(spearmanr(a[keep],b[keep]).statistic)

def pos_avg(g,col):
    if g.empty:return np.nan
    return float(g.groupby('protein_position')[col].mean().mean())

def bootstrap_diff(g, reps=2000, seed=42):
    """High - low mean FoldX ddG, cluster bootstrap of positions within each group.
    Descriptive: overlapping protein positions may occur in both groups.
    """
    low=g[g.regime=='low'].groupby('protein_position').ddg_consensus.mean().to_numpy(float)
    high=g[g.regime=='high'].groupby('protein_position').ddg_consensus.mean().to_numpy(float)
    if not len(low) or not len(high):return (np.nan,np.nan,np.nan)
    rng=np.random.default_rng(seed)
    diffs=high[rng.integers(len(high),size=(reps,len(high)))].mean(axis=1)-\
          low[rng.integers(len(low),size=(reps,len(low)))].mean(axis=1)
    return (float(high.mean()-low.mean()),float(np.percentile(diffs,2.5)),float(np.percentile(diffs,97.5)))

def summarize(wide,reps):
    observed=wide[wide.n_templates>0].copy()
    metrics=[]
    for (name,g) in [('all',observed),('qc_reliable',observed[observed.qc_reliable])]+\
                  [(r,observed[observed.regime==r]) for r in ('low','middle','high')]:
        metrics.append({'group':name,'n_variants':len(g),'n_positions':g.protein_position.nunique(),
                        'rho_ddg_vs_corrector':corr(g.ddg_consensus,g.observed_corrector),
                        'rho_ddg_vs_global_residual':corr(g.ddg_consensus,g.error_global),
                        'rho_ddg_vs_regression_gain':corr(g.ddg_consensus,g.mae_gain)})
    reg=[]
    for name,g in wide.groupby('regime'):
        q=g[g.n_templates>0]
        reg.append({'regime':name,'n_records':len(g),'n_covered':len(q),
                    'n_positions_covered':q.protein_position.nunique(),
                    'mean_ddg_positionweighted':pos_avg(q,'ddg_consensus'),
                    'median_ddg':float(q.ddg_consensus.median()),
                    'mean_foldx_mae_gain_pp':pos_avg(g,'mae_gain'),
                    'n_qc_reliable':int(g.qc_reliable.sum())})
    domains=[]
    for name,g in observed.groupby('domain'):
        low=g[g.regime=='low']; high=g[g.regime=='high']
        if len(low)>=3 and len(high)>=3:
            domains.append({'domain':name,'n_low':len(low),'n_high':len(high),
                            'mean_ddg_high_minus_low':pos_avg(high,'ddg_consensus')-pos_avg(low,'ddg_consensus')})
    for name,g in [('all_covered',observed),('QC_reliable',observed[observed.qc_reliable])]:
        d,l,u=bootstrap_diff(g,reps,2856)
        print(f'High-minus-low position-weighted ΔΔG [{name}]: {d:.3f} kcal/mol; exploratory 95% bootstrap [{l:.3f}, {u:.3f}]')
    return pd.DataFrame(metrics),pd.DataFrame(reg),pd.DataFrame(domains)

def self_test():
    assert domain(500)=='NBD1' and domain(950)=='TMD2' and domain(10)=='other'
    assert np.isnan(corr([1,np.nan],[2,3]))
    a=pd.DataFrame({'regime':['low','low','high','high','middle'],
                    'protein_position':[1,2,3,4,5], 'ddg_consensus':[2.,3.,5.,6.,4.]})
    diff,lo,hi=bootstrap_diff(a,300,42)
    assert abs(diff-3)<1e-10 and lo<=diff<=hi
    print('PASS: domain mapping, safe correlations and position-cluster bootstrap')

def main(args):
    f1=SRC/'phase28e5_repeated_oof.csv';f2=SRC/'phase28e5_train_foldx_features.csv'
    if not f1.exists() or not f2.exists():raise FileNotFoundError(f'Expected Phase28E5 outputs: {f1}, {f2}')
    oof=pd.read_csv(f1,dtype={'variant_key':str}); feats=pd.read_csv(f2,dtype={'variant_key':str})
    v=verify_and_merge(oof,feats)
    c,r,d=summarize(v,args.bootstraps)
    print('='*106)
    print('PHASE 28E-6 | FOLDX STABILITY/RESPONSE DIAGNOSTIC | TRAIN ONLY | NO MODEL FIT')
    print('='*106)
    print(f'Inputs SHA256: OOF={hashfile(f1)[:16]}  FoldX features={hashfile(f2)[:16]}')
    print('No validation or test labels accessed. All associations descriptive, post hoc, and non-causal.')
    print('\nA. ASSAY COVERAGE / STABILITY BY CORRECTOR REGIME')
    print(r.round(4).to_string(index=False))
    print('\nB. SPEARMAN ASSOCIATIONS: ΔΔG VS VTI RESCUE/RESIDUAL AND FOLDX GAIN')
    print(c.round(4).to_string(index=False))
    print('\nC. DOMAIN-CONDITIONAL HIGH VS LOW ΔΔG (small strata; descriptive)')
    print(d.round(4).to_string(index=False) if not d.empty else 'Insufficient domain strata')
    print('\nD. LARGEST REGRESSION ERRORS WITH ΔΔG')
    cols=['mutation','domain','regime','observed_corrector','corrector_global','error_global',
          'ddg_7sv7','ddg_8eig','ddg_consensus','ddg_disagreement','qc_reliable','mae_gain']
    worst=v.assign(abs_error=v.error_global.abs()).sort_values('abs_error',ascending=False).head(args.top)
    print(worst[cols].round(3).to_string(index=False))
    print('\nE. ENERGY QC')
    print('Available energies:', int((v.n_templates>0).sum()), '/210; QC-reliable:', int(v.qc_reliable.sum()))
    print('High template disagreement (>5 kcal/mol):',int((v.ddg_disagreement>5).sum()))
    print('High replicate SD (>2 kcal/mol):',int((v.ddg_mean_repeat_sd>2).sum()))
    OUT.mkdir(parents=True,exist_ok=True)
    v.to_csv(OUT/'phase28e6_mutation_diagnostics_train.csv',index=False)
    c.to_csv(OUT/'phase28e6_correlations.csv',index=False)
    r.to_csv(OUT/'phase28e6_regime_summary.csv',index=False)
    d.to_csv(OUT/'phase28e6_domain_summary.csv',index=False)
    print('\nPASS: descriptive FoldX diagnostic; no new model or VAL/TEST labels.')
    print('Saved:',OUT)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--bootstraps',type=int,default=2000)
    p.add_argument('--top',type=int,default=15)
    a=p.parse_args()
    if a.self_test:self_test()
    else:
        if a.bootstraps<100 or not 1<=a.top<=210:p.error('--bootstraps >=100; --top 1..210')
        self_test();main(a)
