#!/usr/bin/env python3
"""PHASE29C | Read-only (for source data) TRAIN-only CFTR assay SEM / label audit.

- Loads the frozen VTI TRAIN 210 mutation list and exact VTI labels.
- Reads ONLY those IDs from assay parquet via filtered parquet read.
- Audits availability and arithmetic of baseline / VX121+TEZ FSK measurements.
- Audits candidate measurement SEM fields WITHOUT claiming their units/replicate provenance.
- When SEM units are confirmed by the user, interprets propagation under independent
  conditions as a sensitivity analysis, NOT a noise ceiling or variance lower bound.
- Reads frozen OOF from Phase29B; does not refit or tune any models.
- Never reads protected VTI VAL or TEST labels.

Requirements: run from CFTR-Theratype project root, installed CFTR_ENV.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path.cwd()
sys.path.insert(0, str(ROOT/'scripts'))
DATA=ROOT/'data/processed/bihler_2026_vti.parquet'
OOF=ROOT/'results/biology/phase29b/phase29b_oof.csv'
OUT=ROOT/'results/biology/phase29c'
BASE='baseline_fsk_pct_wt'
TREAT='vnz_tez_fsk_pct_wt'
Y='corrector_rescue_pp'
SEM_BASE='fsk_sem_dmso'
SEM_TREAT='fsk_sem_vnz_tez'


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def describe(s):
    a=pd.to_numeric(s,errors='coerce').to_numpy(float)
    a=a[np.isfinite(a)]
    if len(a)==0: return {'n':0,'median':None,'p90':None,'max':None}
    return {'n':int(len(a)),'median':float(np.median(a)),
            'p90':float(np.quantile(a,.90)),'max':float(a.max())}


def analyse_sem(frame):
    """Pure function for synthetic tests. Unknown SEM units -> no noise floor."""
    f=frame.copy()
    for c in [BASE,TREAT,Y,SEM_BASE,SEM_TREAT]:
        if c in f:f[c]=pd.to_numeric(f[c],errors='coerce')
    f['regime']=np.select([f[Y]<10,f[Y]>40],['low','high'],default='middle')
    f['nearest_class_cutoff_distance']=np.minimum(abs(f[Y]-10),abs(f[Y]-40))
    f['corrector_target_check_abs']=abs(f[TREAT]-f[BASE]-f[Y])
    have=(f[SEM_BASE].notna()&f[SEM_TREAT].notna()&
          np.isfinite(f[SEM_BASE])&np.isfinite(f[SEM_TREAT])&
          (f[SEM_BASE]>=0)&(f[SEM_TREAT]>=0))
    f['sem_pair_available']=have
    # Same-units, same-target interpretation must be independently verified.
    f['hyp_se_independent']=np.nan
    f['hyp_se_min_covariance']=np.nan
    f['hyp_se_max_covariance']=np.nan
    f.loc[have,'hyp_se_independent']=np.hypot(f.loc[have,SEM_BASE],f.loc[have,SEM_TREAT])
    f.loc[have,'hyp_se_min_covariance']=abs(f.loc[have,SEM_BASE]-f.loc[have,SEM_TREAT])
    f.loc[have,'hyp_se_max_covariance']=f.loc[have,SEM_BASE]+f.loc[have,SEM_TREAT]
    return f


def summarize(df):
    res=[]
    for k,g in df.groupby('regime'):
        avail=g[g.sem_pair_available]
        res.append({'regime':k,'records':int(len(g)), 'sem_pairs':int(len(avail)),
                    'sem_coverage':len(avail)/len(g),
                    'median_abs_global_error_pp':float(g.abs_global_error_pp.median()),
                    'median_cutoff_distance_pp':float(g.nearest_class_cutoff_distance.median()),
                    'median_hyp_independent_se_unverified_units':
                      float(avail.hyp_se_independent.median()) if len(avail) else np.nan,
                    'median_hyp_covariance_max_se_unverified_units':
                      float(avail.hyp_se_max_covariance.median()) if len(avail) else np.nan})
    return pd.DataFrame(res).sort_values('regime')


def run(args):
    import pyarrow.parquet as pq
    import phase26h_auxiliary_corrector as h
    from phase27d1_biology_ablation import extract_codes
    v,_,_,_,folds=h.load_data()
    if len(v)!=210 or v.variant_key.nunique()!=210 or v.protein_position.nunique()!=168 or len(folds)!=10:
        raise ValueError('Frozen VTI TRAIN structure differs')
    keys=v.variant_key.astype(str).tolist()
    schema=set(pq.ParquetFile(DATA).schema.names)
    names=[c for c in [SEM_BASE,SEM_TREAT,BASE,TREAT,Y,'variant_key'] if c in schema]
    for c in [BASE,TREAT,Y,'variant_key']:
        if c not in names: raise ValueError(f'Assay input column missing: {c}')
    assay=pd.read_parquet(DATA,columns=names,filters=[('variant_key','in',keys)])
    assay.variant_key=assay.variant_key.astype(str)
    if len(assay)!=210 or assay.variant_key.duplicated().any() or set(assay.variant_key)!=set(keys):
        raise ValueError('Assay filtering failed: refusing potential protected data')
    f=v[['variant_key','protein_position',Y]].copy().drop(columns=[Y])
    f.variant_key=f.variant_key.astype(str)
    f=f.merge(assay,on='variant_key',validate='one_to_one',how='left')
    if f[[BASE,TREAT,Y]].isna().any().any():raise ValueError('Missing train assay value')
    ytrain=v.set_index('variant_key').loc[f.variant_key,Y].to_numpy(float)
    if not np.allclose(f[Y].to_numpy(float),ytrain,rtol=0,atol=1e-5):
        raise ValueError('TRAIN label mismatch')
    if not all(c in f for c in [SEM_BASE,SEM_TREAT]):
        print('WARNING: needed SEM fields absent from processed parquet. Available fields:',sorted(schema))
        for c in [SEM_BASE,SEM_TREAT]:
            if c not in f:f[c]=np.nan
    df=analyse_sem(f)
    if df.corrector_target_check_abs.max()>1e-5:raise ValueError('Corrector target arithmetic mismatch')
    # Exact frozen TRAIN out-of-fold predictions, averaged across two repeats per record.
    oof=pd.read_csv(OOF,dtype={'variant_key':str})
    required={'variant_key','repeat','fold','protein_position','corrector_global','observed_corrector'}
    if not required.issubset(oof):raise ValueError(f'Phase29B OOF columns missing: {required-set(oof)}')
    if len(oof)!=420 or oof.duplicated(['repeat','variant_key']).any():
        raise ValueError('OOF repeated predictions malformed')
    if set(oof.variant_key)!=set(keys) or set(oof.repeat)!={1,2}:raise ValueError('OOF keys/repeats mismatch')
    mean=oof.groupby('variant_key',as_index=False).agg(pred_global=('corrector_global','mean'),
                            n_repeats=('repeat','nunique'),truth_oof=('observed_corrector','first'),
                            oof_position=('protein_position','first'))
    if not (mean.n_repeats==2).all():raise ValueError('OOF repeats incomplete')
    df=df.merge(mean,on='variant_key',validate='one_to_one')
    if not np.allclose(df[Y],df.truth_oof,atol=1e-5,rtol=0):raise ValueError('OOF labels differ')
    if not np.array_equal(df.protein_position.to_numpy(int),df.oof_position.to_numpy(int)):
        raise ValueError('OOF position mismatch')
    df['abs_global_error_pp']=(df.pred_global-df[Y]).abs()
    df['signed_global_error_pp']=df.pred_global-df[Y]
    codes=np.asarray(extract_codes(v,v.protein_position.to_numpy(int)),dtype=str)
    ix=pd.Index(v.variant_key.astype(str)).get_indexer(df.variant_key)
    if np.any(ix<0):raise ValueError('Missing mutation-code mapping')
    df['mutation']=codes[ix]
    # Read-only data sources, output summaries as CSV in phase29c only.
    summary=summarize(df)
    cand=[c for c in schema if any(token in c.lower() for token in ['sem','sd','replicate','sample_size','n_reps'])]
    mismatches=[]
    for code,g in df.groupby('mutation'):
        if len(g)>1:
            vals=g.sort_values('variant_key')
            mismatches.append({'mutation':code,'n':len(vals), 'variant_keys':' | '.join(vals.variant_key),
               'response_range_pp':float(vals[Y].max()-vals[Y].min()),
               'se_pair_available_all':bool(vals.sem_pair_available.all()),
               'median_hyp_independent_se_unverified_units':float(vals.hyp_se_independent.median()) if vals.sem_pair_available.any() else np.nan})
    pairs=pd.DataFrame(mismatches)
    print('='*108)
    print('PHASE 29C | EXPERIMENTAL SEM / LABEL PRECISION AUDIT — TRAIN ONLY, NO MODEL FIT')
    print('='*108)
    print('TRAIN 210 records / 168 positions, OOF averaged two repeats. No VAL/TEST labels.')
    print('Assay parquet:',DATA,'SHA256:',sha256(DATA)[:16])
    print('Observed SEM-like fields in dataset schema:',cand)
    print('WARNING: SEM provenance, units, independence and technical replicate counts NOT VERIFIED.')
    print('Therefore, no empirical measurement-noise floor or irreducible-error ceiling will be estimated.')
    print('\nA. Label arithmetic and SEM column coverage')
    print('corrector_rescue = FSK(VNZ+TEZ) - FSK(DMSO); maximum arithmetic mismatch:',
          round(float(df.corrector_target_check_abs.max()),8))
    print('Baseline SEM summary:',describe(df[SEM_BASE]))
    print('VNZ+TEZ SEM summary:',describe(df[SEM_TREAT]))
    print('Matched nonnegative finite SEM pairs:',int(df.sem_pair_available.sum()),'of 210')
    print('\nB. Regime summary (hypothetical SEM calculations, units unverified)')
    print(summary.round(3).to_string(index=False))
    print('\nC. Same-protein-substitution but different nucleotide variants (NOT replicate noise)')
    print(pairs.round(3).to_string(index=False) if len(pairs) else '(none)')
    print('\nD. Largest global prediction errors: assay uncertainty metadata')
    fields=['variant_key','mutation','protein_position','regime',Y,'pred_global',
            'abs_global_error_pp',SEM_BASE,SEM_TREAT,'hyp_se_independent','nearest_class_cutoff_distance']
    print(df.sort_values('abs_global_error_pp',ascending=False)[fields].head(args.top).round(3).to_string(index=False))
    print('\nE. QC')
    print('No VTI protected labels read; frozen predictions only, no refits or target-based model selection.')
    print('The independent-condition SE formula assumes aligned units and zero covariance.')
    print('The min/max covariance bounds assume SEMs estimate uncertainties of the same outcome units.')
    print('Interpret SEM-derived numeric quantities only after checking raw Bihler 2026 assay metadata.')
    OUT.mkdir(parents=True,exist_ok=True)
    summary.to_csv(OUT/'phase29c_regime_summary.csv',index=False)
    df[fields+['sem_pair_available','hyp_se_min_covariance','hyp_se_max_covariance']].to_csv(OUT/'phase29c_train_assay_uncertainty.csv',index=False)
    pairs.to_csv(OUT/'phase29c_duplicate_substitutions.csv',index=False)
    (OUT/'phase29c_provenance.json').write_text(json.dumps({
        'train_records':210,'positions':168,'source_sha256':sha256(DATA),
        'oof_sha256':sha256(OOF), 'sem_columns_in_schema':{k:k in schema for k in [SEM_BASE,SEM_TREAT]},
        'qualifications':['SEM unit/replicate provenance unknown','not a noise floor','development TRAIN only']},indent=2)+'\n')
    print('PASS Phase29C; saved:',OUT)


def selftest():
    test=pd.DataFrame({BASE:[10.,12.,2.],TREAT:[15.,41.,50.],Y:[5.,29.,48.],
         SEM_BASE:[2.,np.nan,3.],SEM_TREAT:[4.,4.,4.]})
    f=analyse_sem(test)
    assert np.allclose(f.corrector_target_check_abs,0)
    assert f.sem_pair_available.tolist()==[True,False,True]
    assert abs(f.hyp_se_independent.iloc[0]-np.sqrt(20))<1e-12
    assert abs(f.hyp_se_min_covariance.iloc[2]-1)<1e-12
    assert abs(f.hyp_se_max_covariance.iloc[0]-6)<1e-12
    print('PASS synthetic SEM propagation, label arithmetic, unknown SEM handling')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--top',type=int,default=12)
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:selftest()
    else:
        if a.top<1:ap.error('top must be >=1')
        selftest();run(a)
