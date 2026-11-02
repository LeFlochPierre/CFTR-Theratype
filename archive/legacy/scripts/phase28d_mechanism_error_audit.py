#!/usr/bin/env python3
"""Phase 28D: descriptive, TRAIN-only mechanism/error audit of frozen 28C models.

No model fitting or selection. No VTI VAL/TEST read. Reuses Phase 28C's
2 x 5 position-disjoint OOF predictions and mutation-specific physical proxies.

Usage from CFTR-Theratype root:
  python scripts/phase28d_mechanism_error_audit.py --self-test
  python scripts/phase28d_mechanism_error_audit.py

Structural proxies from drug-bound CFTR structures are NOT mutation-specific
folding energies, not physiological apo-state measurements, and should not be
interpreted as causal markers of responsiveness. All comparisons descriptive.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

DOMAIN_RANGES=[('TMD1',81,365),('NBD1',423,646),('R',654,831),('TMD2',859,1155),('NBD2',1210,1443)]
REQUIRED_OOF={'repeat','fold','variant_key','protein_position','observed_corrector','observed_total','corrector_global','corrector_context','corrector_context_interaction','total_global','total_context','total_context_interaction','regime'}
REQUIRED_FEAT={'variant_key','protein_position','mutation'}


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def domain(p):
    return next((name for name,lo,hi in DOMAIN_RANGES if lo<=int(p)<=hi), 'other')


def pwmae(df, actual, prediction):
    if df.empty:return float('nan')
    return float((df[prediction]-df[actual]).abs().groupby(df.protein_position).mean().mean())


def mean_by_position(df, value):
    if df.empty:return float('nan')
    return float(df.groupby('protein_position')[value].mean().mean())


def audited_data(oof,feat):
    if not REQUIRED_OOF.issubset(oof.columns):
        raise ValueError(f'OOF missing: {sorted(REQUIRED_OOF-set(oof.columns))}')
    if not REQUIRED_FEAT.issubset(feat.columns):
        raise ValueError(f'Features missing: {sorted(REQUIRED_FEAT-set(feat.columns))}')
    oof=oof.copy(); feat=feat.copy()
    for d in (oof,feat):
        d['variant_key']=d.variant_key.astype(str)
        d['protein_position']=pd.to_numeric(d.protein_position,errors='raise').astype(int)
    if len(oof)!=420 or len(feat)!=210 or oof.duplicated(['repeat','fold','variant_key']).any():
        raise ValueError('Unexpected 2x5 outer-fold sizes or duplicate OOF rows')
    if feat.variant_key.duplicated().any() or len(set(feat.variant_key))!=210:
        raise ValueError('Feature cardinality not 210 unique variants')
    if set(oof.repeat.unique())!={1,2} or len(oof.groupby(['repeat','fold']))!=10:
        raise ValueError('Expected 2 repeats and 5 folds per repeat')
    if any(len(g)!=210 or g.variant_key.nunique()!=210 for _,g in oof.groupby('repeat')):
        raise ValueError('Each repeat must have one OOF prediction per TRAIN variant')
    if oof.protein_position.nunique()!=168 or feat.protein_position.nunique()!=168:
        raise ValueError('Expected 168 training positions')
    if not set(oof.variant_key)==set(feat.variant_key):
        raise ValueError('OOF and structural feature variant identities differ')
    if (oof.groupby('variant_key').protein_position.nunique()!=1).any():
        raise ValueError('Inconsistent OOF position per mutation')
    if (oof.groupby('variant_key').observed_corrector.nunique()!=1).any():
        raise ValueError('Inconsistent corrector labels across repeats')
    if (oof.groupby('variant_key').observed_total.nunique()!=1).any():
        raise ValueError('Inconsistent total labels across repeats')
    if oof.groupby('variant_key').regime.nunique().max()!=1:
        raise ValueError('Regime inconsistent across repeats')
    if oof.isna().any().any() or np.isinf(oof.select_dtypes(include='number').to_numpy(float)).any():
        raise ValueError('NaN or Inf in frozen OOF')
    if oof.groupby(['repeat','protein_position']).fold.nunique().max()>1:
        raise ValueError('Outer folds separate different variants of the same protein position')
    target=oof.observed_corrector.to_numpy(float)
    expected=np.select([target<10,target>40],['low','high'],default='middle')
    if not np.array_equal(oof.regime.astype(str).to_numpy(),expected):
        raise ValueError('Response-regime rules do not match true TRAIN labels')
    if (oof[['variant_key','protein_position']].drop_duplicates().set_index('variant_key').protein_position
        .sort_index().to_numpy()!=feat.set_index('variant_key').protein_position.sort_index().to_numpy()).any():
        raise ValueError('OOF/feature position mismatch')
    # Average the two OOF predictions for each of the 210 variants; do not duplicate outcomes.
    numeric=['observed_corrector','observed_total']+[
        f'{target}_{config}' for target in ['corrector','total'] for config in ['global','context','context_interaction']]
    if not set(numeric).issubset(oof.columns):raise ValueError('Required prediction column missing')
    v=oof.groupby(['variant_key','protein_position'],as_index=False)[numeric].mean()
    v=v.merge(feat,on=['variant_key','protein_position'],how='left',validate='one_to_one',indicator=True)
    if not (v['_merge']=='both').all():raise ValueError('Feature missing for one or more variant')
    v=v.drop(columns='_merge')
    v['regime']=np.select([v.observed_corrector<10,v.observed_corrector>40],['low','high'],default='middle')
    v['domain']=v.protein_position.map(domain)
    for model in ['global','context','context_interaction']:
        v[f'error_{model}']=v[f'corrector_{model}']-v.observed_corrector
        v[f'ae_{model}']=v[f'error_{model}'].abs()
        v[f'gain_{model}']=v.ae_global-v[f'ae_{model}']
    obscols=[f'ctx_{p}_has_ca' for p in ['8eig','7sv7']]
    if not set(obscols).issubset(v):raise ValueError('Missing model coverage masks')
    status=v[obscols].astype(float).sum(axis=1)
    if (~status.isin([0.,1.,2.])).any():raise ValueError('Invalid PDB observation flag')
    v['structural_coverage']=np.select([status==0,status==1],['neither','one'],default='both')
    for feature in ['relative_sasa','nonlocal_ca8','interdomain_ca8','longrange_ca8']:
        cols=[f'ctx_{p}_{feature}' for p in ['8eig','7sv7']]
        if not set(cols).issubset(v):raise ValueError(f'Missing structure feature {feature}')
        arr=v[cols].to_numpy(float)
        obs=v[obscols].to_numpy(float)
        if not np.isfinite(arr).all():raise ValueError(f'Nonfinite structure feature {feature}')
        v[f'mean_{feature}']=np.divide((arr*obs).sum(axis=1),obs.sum(axis=1),out=np.full(len(v),np.nan),where=obs.sum(axis=1)>0)
    # Simple bins for EXPLORATION, not prediction. Do not silently label missing sites as buried.
    v['exposure_bin']=pd.cut(v.mean_relative_sasa,[-np.inf,0.15,0.35,np.inf],labels=['buried','partly_exposed','exposed']).astype(object)
    v.loc[v.structural_coverage=='neither','exposure_bin']='unobserved'
    return v


def summarize(v):
    counts=[]; domain_perf=[]; regime_perf=[]; exposure=[]; features=[]
    for name,g in v.groupby('domain',sort=True):
        counts.append({'domain':name,'variants':len(g),'positions':g.protein_position.nunique(),
                       'low':int((g.regime=='low').sum()),'middle':int((g.regime=='middle').sum()),
                       'high':int((g.regime=='high').sum()),
                       'mean_corrector_pp':mean_by_position(g,'observed_corrector'),
                       'global_mae_pp':pwmae(g,'observed_corrector','corrector_global'),
                       'new_mae_pp':pwmae(g,'observed_corrector','corrector_context_interaction'),
                       'gain_pp':pwmae(g,'observed_corrector','corrector_global')-pwmae(g,'observed_corrector','corrector_context_interaction')})
        for subset,sg in g.groupby('regime',sort=True):
            domain_perf.append({'domain':name,'regime':subset,'n':len(sg),'positions':sg.protein_position.nunique(),
                                'global_mae_pp':pwmae(sg,'observed_corrector','corrector_global'),
                                'new_mae_pp':pwmae(sg,'observed_corrector','corrector_context_interaction'),
                                'global_bias_pp':mean_by_position(sg,'error_global')})
    for name,g in v.groupby('regime',sort=True):
        regime_perf.append({'regime':name,'n':len(g),'positions':g.protein_position.nunique(),
                            'mean_truth_pp':mean_by_position(g,'observed_corrector'),
                            'mean_prediction_pp':mean_by_position(g,'corrector_global'),
                            'global_mae_pp':pwmae(g,'observed_corrector','corrector_global'),
                            'bio_mae_pp':pwmae(g,'observed_corrector','corrector_context_interaction'),
                            'global_bias_pp':mean_by_position(g,'error_global')})
        for f in ['mean_relative_sasa','mean_nonlocal_ca8','mean_interdomain_ca8','mean_longrange_ca8']:
            sf=g.loc[g.structural_coverage!='neither']
            features.append({'regime':name,'feature':f,'covered_variants':len(sf),
                             'covered_positions':sf.protein_position.nunique(),
                             'position_weighted_mean':mean_by_position(sf,f)})
    for (region,coverage),g in v.groupby(['regime','exposure_bin'],sort=True,dropna=False):
        exposure.append({'regime':region,'exposure_bin':coverage,'n':len(g),
                         'positions':g.protein_position.nunique(),
                         'global_mae_pp':pwmae(g,'observed_corrector','corrector_global'),
                         'new_mae_pp':pwmae(g,'observed_corrector','corrector_context_interaction')})
    return (pd.DataFrame(counts),pd.DataFrame(domain_perf),pd.DataFrame(regime_perf),
            pd.DataFrame(exposure),pd.DataFrame(features))


def self_test():
    o=[]; f=[]
    for i in range(210):
        p=i+10 if i<168 else i-168+10
        # shared position for the final 42 variants; 168 independent protein positions
        truth=[3.,23.,54.][i%3]
        regime='low' if truth<10 else 'high' if truth>40 else 'middle'
        key=f'm{i:03d}'
        f.append({'variant_key':key,'protein_position':p,'mutation':f'A{p}V',
                  'ctx_8eig_has_ca':int(i%11!=0),'ctx_7sv7_has_ca':int(i%13!=0),
                  **{f'ctx_{st}_{metric}':(0.2 if metric=='relative_sasa' else 5.)
                     for st in ['8eig','7sv7'] for metric in ['relative_sasa','nonlocal_ca8','interdomain_ca8','longrange_ca8']}})
        for rep in [1,2]:
            row={'repeat':rep,'fold':1+(p%5),'variant_key':key,'protein_position':p,
                 'observed_corrector':truth,'observed_total':truth+5.,'regime':regime}
            for target in ['corrector','total']:
                base=truth+(5. if target=='total' else 0.)
                row[target+'_global']=base+10
                row[target+'_context']=base+9
                row[target+'_context_interaction']=base+8
            o.append(row)
    v=audited_data(pd.DataFrame(o),pd.DataFrame(f))
    assert len(v)==210 and v.protein_position.nunique()==168
    assert np.isclose(pwmae(v,'observed_corrector','corrector_global'),10.)
    assert np.isclose(pwmae(v,'observed_corrector','corrector_context_interaction'),8.)
    assert v.structural_coverage.isin(['both','one','neither']).all()
    out=summarize(v)
    assert len(out)==5 and out[0].variants.sum()==210
    print('PASS synthetic: two-repeat OOF joins, protein positions, domain/regime/exposure audit, no training')


def run(args):
    oofpath=Path(args.oof); featpath=Path(args.features); out=Path(args.out)
    for p in [oofpath,featpath]:
        if not p.is_file():raise FileNotFoundError(f'Missing Phase 28C input: {p}')
    v=audited_data(pd.read_csv(oofpath,dtype={'variant_key':str}),
                   pd.read_csv(featpath,dtype={'variant_key':str}))
    bydomain,bydomainregime,byregime,byexposure,features=summarize(v)
    out.mkdir(parents=True,exist_ok=True)
    v.to_csv(out/'phase28d_variant_oof_error_audit.csv',index=False)
    bydomain.to_csv(out/'phase28d_domain_errors.csv',index=False)
    bydomainregime.to_csv(out/'phase28d_domain_by_regime_errors.csv',index=False)
    byregime.to_csv(out/'phase28d_regime_errors.csv',index=False)
    byexposure.to_csv(out/'phase28d_exposure_errors.csv',index=False)
    features.to_csv(out/'phase28d_feature_summary.csv',index=False)
    worst=v.nlargest(args.top,'ae_global')[['mutation','protein_position','domain','regime','observed_corrector','corrector_global','corrector_context_interaction','ae_global','gain_context_interaction']]
    worst.to_csv(out/'phase28d_largest_errors.csv',index=False)
    manifest={'train_only':True,'n_variants':len(v),'n_positions':v.protein_position.nunique(),
              'oof_sha256':sha(oofpath),'features_sha256':sha(featpath),
              'warning':'Descriptive repeated TRAIN OOF; no independent validation, no causal claims; static drug-bound conformations are not mutant folding energies'}
    (out/'phase28d_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('='*114)
    print('PHASE 28D | CFTR ERROR MECHANISMS — TRAIN OOF ONLY; NO FIT, NO VAL/TEST')
    print('='*114)
    print(f'Audited {len(v)} unique mutations at {v.protein_position.nunique()} protein positions.')
    print('OOF predictions averaged over repeats; no double-counted variants. Model not refitted.')
    print('\nA. BY DOMAIN: prevalence, true corrector mean, and global versus new MAE')
    print(bydomain.round(3).to_string(index=False))
    print('\nB. BY RESPONSE REGIME: signed error (positive means OVERprediction)')
    print(byregime.round(3).to_string(index=False))
    print('\nC. DOMAIN x RESPONSE REGIME ERRORS (small groups are descriptive only)')
    print(bydomainregime.sort_values(['domain','regime']).round(3).to_string(index=False))
    print('\nD. STRUCTURAL EXPOSURE COVERAGE AND ERROR')
    print(byexposure.round(3).to_string(index=False))
    print('\nE. STRUCTURAL PROXY AVERAGES BY RESPONSE REGIME (NOT mutation-specific ΔΔG)')
    print(features.pivot(index='feature',columns='regime',values='position_weighted_mean').round(3).to_string())
    print('\nF. LARGEST GLOBAL REGRESSION ERRORS; DEVELOPMENT DIAGNOSTIC ONLY')
    print(worst.round(3).to_string(index=False))
    print('\nG. STRUCTURAL COVERAGE')
    print(pd.crosstab(v.regime,v.structural_coverage).to_string())
    print('\nPASS: descriptive audit only. No new predictive model and no protected labels.')
    print('Outputs:',out.resolve())


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--oof',default='results/biology/phase28c/phase28c_repeated_oof.csv')
    p.add_argument('--features',default='results/biology/phase28c/phase28c_train_biophysics_features.csv')
    p.add_argument('--out',default='results/biology/phase28d')
    p.add_argument('--top',type=int,default=15)
    args=p.parse_args()
    if args.self_test:self_test()
    else:run(args)
