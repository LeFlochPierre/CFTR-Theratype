#!/usr/bin/env python3
"""PHASE 29B: TRAIN-only CFTR predictability / measurement / feature-ambiguity audit.

- Frozen Phase26H2 2x5 position-disjoint OOF; XGBoost predictions are READ, NOT refit.
- Fold-conditioned naive mean, median, fixed-shrinkage domain mean, Ridge, NN1/NN5.
- Protein-position groups never cross train/test; distance scaling/Ridge fit OUTER TRAIN only.
- Identical protein substitutions may be separate nucleotide variants: NOT technical replicates.
- The measurement-noise/Bayes ceiling cannot be inferred from aggregated unique outcomes.
- No VTI validation/test data or labels. Reused TRAIN CV: exploratory only.

Run from CFTR-Theratype repo root. Saves CSVs and structured provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'scripts'))
FROZEN=ROOT/'results/biology/phase26h2/phase26h2_oof.csv'
OUT=ROOT/'results/biology/phase29b'
KEY=['repeat','fold','variant_key','protein_position']
DOMAINS=[('TMD1',81,365),('NBD1',423,646),('R',654,831),('TMD2',859,1155),('NBD2',1210,1443)]
CONFIGS=['global','train_mean','train_median','domain_shrink8','ridge_seq68','nn1_seq68','nn5_seq68']


def hashfile(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''): h.update(part)
    return h.hexdigest()


def domain(p):
    return next((s for s,a,b in DOMAINS if a<=int(p)<=b),'other')


def position_mean(y, positions):
    return float(pd.DataFrame({'p':positions,'y':y}).groupby('p').y.mean().mean())


def position_median(y, positions):
    return float(np.median(pd.DataFrame({'p':positions,'y':y}).groupby('p').y.mean().to_numpy()))


def position_mae(frame, pred, truth='observed_corrector'):
    z=frame[['protein_position',pred,truth]].copy()
    z['ae']=(z[pred]-z[truth]).abs()
    return float(z.groupby('protein_position').ae.mean().mean())


def grouped_bootstrap(oof, pred, reference='corrector_global', nboot=2000, seed=29):
    df=oof[['repeat','protein_position',pred,reference,'observed_corrector']].copy()
    df['gain']=(df[reference]-df.observed_corrector).abs()-(df[pred]-df.observed_corrector).abs()
    # Per-position average within repeat; then across two repeats. Never treat repeats as new data.
    perpos=(df.groupby(['repeat','protein_position']).gain.mean()
              .groupby('protein_position').mean().to_numpy(float))
    if len(perpos)!=168: raise ValueError('Expected 168 positions for bootstrap')
    rng=np.random.default_rng(seed)
    vals=np.array([perpos[rng.integers(0,len(perpos),size=len(perpos))].mean()
                   for _ in range(nboot)])
    return {'candidate':pred,'reference':reference,'mae_gain_pp':float(perpos.mean()),
            'ci95_low':float(np.quantile(vals,0.025)),
            'ci95_high':float(np.quantile(vals,0.975)),
            'n_positions':len(perpos)}


def fold_predictions(xtr, ytr, gtr, xte, gte):
    """All transformations and fitted parameters use ONLY outer training rows."""
    xtr=np.asarray(xtr,float); xte=np.asarray(xte,float)
    ytr=np.asarray(ytr,float);gtr=np.asarray(gtr,int);gte=np.asarray(gte,int)
    if set(gtr)&set(gte):raise ValueError('Leaking protein position between train and test')
    if not np.isfinite(xtr).all() or not np.isfinite(xte).all():
        raise ValueError('Sequence features must be fully finite')
    # Position-weighting reduces domination by multiple variants at the same position.
    counts=pd.Series(gtr).value_counts()
    sample_weight=np.asarray([1./counts[p] for p in gtr],float)
    sample_weight/=sample_weight.mean()
    tr_pos_means=pd.DataFrame({'position':gtr,'y':ytr}).groupby('position').y.mean()
    mean=float(tr_pos_means.mean()); median=float(tr_pos_means.median())
    # Shrink domain-level position means towards global position-weighted mean.
    shrink=8.0
    dtr=np.array([domain(p) for p in gtr]); dte=np.array([domain(p) for p in gte])
    dmeans={}
    for d in set(dtr):
        a=pd.DataFrame({'position':gtr[dtr==d],'y':ytr[dtr==d]}).groupby('position').y.mean()
        dmeans[d]=(len(a)*float(a.mean())+shrink*mean)/(len(a)+shrink)
    domain_pred=np.asarray([dmeans.get(d,mean) for d in dte],float)
    scaler=StandardScaler().fit(xtr)
    ztr=scaler.transform(xtr);zte=scaler.transform(xte)
    active=(np.var(ztr,axis=0)>1e-10)
    if active.sum()<2: raise ValueError('Too few nonconstant features')
    ztr= ztr[:,active]; zte=zte[:,active]
    model=Ridge(alpha=100.,fit_intercept=True)
    model.fit(ztr,ytr,sample_weight=sample_weight)
    ridge=np.asarray(model.predict(zte),float)
    # Find nearest training mutations at DISTINCT protein positions.
    # Group-based splits already ensure test position excluded; also unique train
    # positions so variants at a frequent residue don't crowd out other neighbors.
    dist=cdist(zte,ztr,metric='euclidean')/np.sqrt(ztr.shape[1])
    nn1=np.empty(len(gte));nn5=np.empty(len(gte)); nn_gap=np.empty(len(gte))
    nn_dist=np.empty(len(gte));nn_pos=np.empty(len(gte),int); nn_idx=np.empty(len(gte),int)
    for j in range(len(gte)):
        idx=np.argsort(dist[j],kind='stable')
        chosen=[];seen=set()
        for k in idx:
            k=int(k)
            if gtr[k] in seen:continue
            seen.add(int(gtr[k]));chosen.append(k)
            if len(chosen)==5:break
        if len(chosen)<5:raise ValueError('Too few distinct neighbor positions')
        nn1[j]=ytr[chosen[0]]
        nn5[j]=np.mean(ytr[chosen])  # Equal weight; no label-derived tuning.
        nn_dist[j]=dist[j,chosen[0]];nn_pos[j]=gtr[chosen[0]]
        nn_idx[j]=chosen[0]
    preds={'train_mean':np.full(len(gte),mean),'train_median':np.full(len(gte),median),
           'domain_shrink8':domain_pred,'ridge_seq68':ridge,
           'nn1_seq68':nn1,'nn5_seq68':nn5}
    if any(not np.isfinite(a).all() for a in preds.values()):raise ValueError('Nonfinite predictions')
    return preds,{'neighbor_distance':nn_dist,'neighbor_position':nn_pos,
                  'neighbor_train_index':nn_idx,
                  'active_sequence_features':int(active.sum())}


def load_frozen(v, folds):
    oof=pd.read_csv(FROZEN,dtype={'variant_key':str})
    if len(oof)!=420 or oof.duplicated(KEY).any(): raise ValueError('Historical OOF malformed')
    required=set(KEY+['vti_fixed','observed_corrector'])
    if not required.issubset(oof):raise ValueError(f'Missing OOF columns: {required-set(oof)}')
    if set(oof.repeat.unique())!={1,2} or len(oof.groupby(['repeat','fold']))!=10:
        raise ValueError('Unexpected repeat/fold set')
    keys=set(v.variant_key.astype(str))
    if any(set(t.variant_key)!=keys for _,t in oof.groupby('repeat')):
        raise ValueError('Frozen variant membership mismatch')
    if (oof.groupby(['repeat','protein_position']).fold.nunique()>1).any():
        raise ValueError('Protein-position leakage in frozen OOF')
    return oof


def assay_audit(v, keys, codes, y):
    """TRAIN-only primary assay measurements and amino-acid-substitution duplicates.

    Repeated aggregated rows / different DNA variants are NOT technical replicates.
    """
    out={}
    available=['variant_key','baseline_fsk_pct_wt','vnz_tez_fsk_pct_wt',
               'iva_treated_pct_wt','vti_treated_pct_wt','study_cohort',
               'corrector_rescue_pp']
    parquet=ROOT/'data/processed/bihler_2026_vti.parquet'
    if parquet.exists():
        # Read schema without loading protected rows or labels.
        import pyarrow.parquet as pq
        schema=set(pq.ParquetFile(parquet).schema.names)
        cols=[z for z in available if z in schema]
        if 'variant_key' not in cols:raise ValueError('Assay parquet missing variant_key')
        d=pd.read_parquet(parquet, columns=cols,filters=[('variant_key','in',list(keys))])
        d['variant_key']=d.variant_key.astype(str)
        if len(d)!=210 or d.variant_key.duplicated().any() or set(d.variant_key)!=set(keys):
            raise ValueError('TRAIN-filtered assay incomplete or contains non-TRAIN rows')
        d['mutation']=d.variant_key.map(dict(zip(keys,codes)))
        if 'corrector_rescue_pp' in d:
            if np.max(np.abs(d.set_index('variant_key').loc[list(keys),'corrector_rescue_pp'].to_numpy(float)-y))>1e-5:
                raise ValueError('Assay parquet TRAIN labels differ from frozen load_data')
        if {'baseline_fsk_pct_wt','vnz_tez_fsk_pct_wt','corrector_rescue_pp'}.issubset(d):
            diff=(d.vnz_tez_fsk_pct_wt-d.baseline_fsk_pct_wt-d.corrector_rescue_pp).abs()
            out['corrector_arithmetic_max_abs_pp']=float(diff.max())
            if diff.max()>1e-4:raise ValueError('Assay target reconstruction mismatch')
        # Column headings alone do NOT prove uncertainty is known.
        out['available_processed_assay_columns']=cols
        out['possible_replicate_or_uncertainty_columns']=[c for c in schema if re.search(r'(^|_)(sd|sem|stderr|std|replicate|replicates|n_reps|cv)(_|$)',c,re.I)]
        bycohort=pd.DataFrame({'study_cohort':d.study_cohort if 'study_cohort' in d else 'unknown',
                               'mutation':d.mutation}).groupby('study_cohort',dropna=False).agg(
                                    records=('mutation','size'),substitutions=('mutation','nunique'))
    else:
        d=pd.DataFrame({'variant_key':keys,'mutation':codes})
        out['warning']='TRAIN parquet unavailable; duplicate substitution audit uses load_data labels only'
        bycohort=pd.DataFrame()
    duplicates=pd.DataFrame({'variant_key':keys,'mutation':codes,'observed_corrector':y})
    if 'study_cohort' in d:
        duplicates=duplicates.merge(d[['variant_key','study_cohort']],on='variant_key',validate='one_to_one')
    groups=[]
    for mutation,g in duplicates.groupby('mutation'):
        if len(g)>1:
            groups.append({'mutation':mutation,'records':len(g),
                           'positions':1,'variant_keys':' | '.join(g.variant_key),
                           'cohorts':' | '.join(map(str,g.study_cohort)) if 'study_cohort' in g else 'unknown',
                           'min_corrector_pp':float(g.observed_corrector.min()),
                           'max_corrector_pp':float(g.observed_corrector.max()),
                           'range_corrector_pp':float(g.observed_corrector.max()-g.observed_corrector.min())})
    duplicates=pd.DataFrame(groups,columns=['mutation','records','positions','variant_keys','cohorts',
                                            'min_corrector_pp','max_corrector_pp','range_corrector_pp'])
    out['n_aa_substitutions']=len(set(codes))
    out['n_duplicate_substitution_groups']=len(duplicates)
    out['n_duplicate_records']=int(duplicates.records.sum()) if len(duplicates) else 0
    out['measurement_noise_floor_estimated']=False
    out['reason_no_noise_floor']='No verified assay-level technical replicate measurements/SDs; duplicate protein substitutions may arise from different DNA variants or assay observations.'
    return out,duplicates,bycohort


def inspect_nearest(oof, n_top):
    """Summarise nearest TRAIN neighbors; preserve exact repeat-specific pairs."""
    pairs=oof.copy()
    pairs['nearest_response_gap_pp']=(pairs.neighbor_response-pairs.observed_corrector).abs()
    q=pairs.groupby(['variant_key','protein_position'],as_index=False).agg(
       observed_corrector=('observed_corrector','first'),
       nearest_train_distance=('neighbor_distance','mean'),
       nearest_response_gap_pp=('nearest_response_gap_pp','mean'),
       global_oof=('corrector_global','mean'),
       nn1_oof=('corrector_nn1_seq68','mean'))
    # The nearest position can differ between two folds/repeats. Never attach
    # a repeat-1 neighbor ID to an average of two different neighbor responses.
    first=pairs[pairs.repeat==1][['variant_key','neighbor_position','neighbor_mutation',
           'neighbor_response','neighbor_distance','nearest_response_gap_pp']].rename(columns={
             'neighbor_position':'repeat1_neighbor_position',
             'neighbor_mutation':'repeat1_neighbor_mutation',
             'neighbor_response':'repeat1_neighbor_response',
             'neighbor_distance':'repeat1_neighbor_distance',
             'nearest_response_gap_pp':'repeat1_response_gap_pp'})
    q=q.merge(first,on='variant_key',how='left',validate='one_to_one')
    q['nearest_model_gain_pp']=(q.global_oof-q.observed_corrector).abs()-(q.nn1_oof-q.observed_corrector).abs()
    q['regime']=np.where(q.observed_corrector<10,'low',np.where(q.observed_corrector>40,'high','middle'))
    qq=q.nearest_train_distance.quantile([.25,.5,.75]).to_dict()
    bins=pd.qcut(q.nearest_train_distance,4,labels=['Q1 nearest','Q2','Q3','Q4 farthest'],duplicates='drop')
    q['distance_quartile']=bins.astype(str)
    block=q.groupby('distance_quartile',sort=True).agg(
       n=('variant_key','size'),mean_feature_distance=('nearest_train_distance','mean'),
       median_response_gap_pp=('nearest_response_gap_pp','median'),
       large_gap_over_30_pp=('nearest_response_gap_pp',lambda x: int((x>30).sum())),
       fraction_large_gap=('nearest_response_gap_pp',lambda x:float((x>30).mean())))
    worst=q.sort_values('nearest_response_gap_pp',ascending=False).head(n_top)
    top_closest=q.sort_values('nearest_train_distance',ascending=True).head(n_top)
    return q,block.reset_index(),worst,top_closest,qq


def run(args):
    from cftr_theratype.datasets.eti_auxiliary import load_data as load_eti_train
    from phase27d1_biology_ablation import extract_codes
    v,x80,_,_,folds=load_eti_train(ROOT)
    x80=np.asarray(x80,float)
    keys=v.variant_key.astype(str).to_numpy()
    pos=v.protein_position.to_numpy(int)
    yc=v.corrector_rescue_pp.to_numpy(float)
    if len(v)!=210 or len(set(keys))!=210 or len(set(pos))!=168 or x80.shape!=(210,80) or len(folds)!=10:
        raise ValueError('Frozen VTI training shapes/positions/folds have changed')
    if not np.isfinite(x80[:,:68]).all():raise ValueError('Sequence features not fully finite')
    if not np.isfinite(yc).all():raise ValueError('Target contains nonfinite values')
    code=extract_codes(v,pos)
    assert len(set(code))==206, 'Expected 206 AA substitutions'
    frozen=load_frozen(v,folds)
    srcmeta,dup,cohorts=assay_audit(v,keys,code,yc)
    print('='*106)
    print('PHASE 29B | PREDICTABILITY + LABEL RELIABILITY AUDIT | VTI TRAIN ONLY')
    print('='*106)
    print('210 variants, 206 distinct AA substitutions, 168 positions; fixed 2x5 grouped folds')
    print('Frozen XGB OOF read from Phase26H2. No VAL/TEST labels. No new hyperparameter selection.')
    print('Simple comparators: position-weighted mean/median, shrinkage domain mean, seq68 Ridge, NN1/NN5.')
    rows=[];checks=[]
    for i,(rep,fold,tr_keys,te_keys) in enumerate(folds,1):
        it=np.flatnonzero(np.isin(keys,list(tr_keys)))
        iv=np.flatnonzero(np.isin(keys,list(te_keys)))
        if len(it)+len(iv)!=210 or set(pos[it])&set(pos[iv]):raise ValueError('Position split leakage')
        ref=frozen.loc[(frozen.repeat==rep)&(frozen.fold==fold)].set_index('variant_key')
        if len(ref)!=len(iv) or set(ref.index)!=set(keys[iv]):raise ValueError('Frozen fold identities differ')
        ref=ref.loc[keys[iv]]
        if not np.allclose(ref.observed_corrector.to_numpy(float),yc[iv],atol=1e-8,rtol=0):
            raise ValueError('Frozen corrector labels differ')
        preds,meta=fold_predictions(x80[it,:68],yc[it],pos[it],x80[iv,:68],pos[iv])
        for j,ix in enumerate(iv):
            row={'repeat':rep,'fold':fold,'variant_key':keys[ix],
                 'protein_position':int(pos[ix]),'mutation':code[ix],
                 'observed_corrector':yc[ix],
                 'corrector_global':float(ref.vti_fixed.iloc[j]),
                 'neighbor_distance':float(meta['neighbor_distance'][j]),
                 'neighbor_position':int(meta['neighbor_position'][j]),
                 'neighbor_response':float(yc[it[meta['neighbor_train_index'][j]]]),
                 'neighbor_mutation':code[it[meta['neighbor_train_index'][j]]]}
            for c,p in preds.items():row['corrector_'+c]=float(p[j])
            rows.append(row)
        checks.append({'repeat':rep,'fold':fold,'training_variants':len(it),
                       'heldout_variants':len(iv),'training_positions':len(set(pos[it])),
                       'heldout_positions':len(set(pos[iv])),
                       'active_seq_features':meta['active_sequence_features']})
        print(f' Fold {i:02d}/10: repeat={rep}, fold={fold} train={len(it)} heldout={len(iv)}',flush=True)
    oof=pd.DataFrame(rows)
    if len(oof)!=420 or oof.duplicated(KEY).any() or len(oof.groupby(['repeat','fold']))!=10:
        raise ValueError('OOF completeness failure')
    if any(t.variant_key.nunique()!=210 or len(t)!=210 for _,t in oof.groupby('repeat')):
        raise ValueError('Each repeat must contain exactly the same 210 variants')
    metric=[]; reg=[]; gains=[]
    for m in CONFIGS:
        col='corrector_'+m
        metric.append({'model':m,'position_mae_pp':float(np.mean([position_mae(z,col) for _,z in oof.groupby('repeat')])),
                       'variant_mae_pp':float((oof[col]-oof.observed_corrector).abs().mean()),
                       'signed_bias_pp':float((oof[col]-oof.observed_corrector).mean()),
                       'pred_sd_pp':float(oof.groupby('variant_key')[col].mean().std(ddof=0))})
        if m!='global':gains.append(grouped_bootstrap(oof,col,nboot=args.bootstraps,seed=2900+len(gains)))
        for r,mask in [('low',oof.observed_corrector<10),
                       ('middle',oof.observed_corrector.between(10,40,inclusive='both')),
                       ('high',oof.observed_corrector>40)]:
            z=oof[mask]
            reg.append({'model':m,'regime':r,'variants':z.variant_key.nunique(),
                        'positions':z.protein_position.nunique(),
                        'position_mae_pp':float(np.mean([position_mae(t,col) for _,t in z.groupby('repeat')])),
                        'mean_pred_pp':float(z[col].mean()),'mean_truth_pp':float(z.observed_corrector.mean())})
    table=pd.DataFrame(metric).sort_values('position_mae_pp')
    groups=pd.DataFrame(reg)
    q,bins,worst,closest,quartiles=inspect_nearest(oof,args.top)
    OUT.mkdir(parents=True,exist_ok=True)
    output={'oof':oof,'metrics':table,'regime_metrics':groups,'paired_gains':pd.DataFrame(gains),
            'nearest_neighbors':q,'neighbor_pairs_by_repeat':oof[['repeat','fold','variant_key','protein_position','mutation','observed_corrector','neighbor_mutation','neighbor_position','neighbor_response','neighbor_distance']],
            'distance_bins':bins,'largest_neighbor_conflicts':worst,
            'closest_neighbors':closest,'duplicate_aa_substitutions':dup,
            'fold_qc':pd.DataFrame(checks)}
    if len(cohorts):output['assay_cohort_counts']=cohorts.reset_index()
    for name,df in output.items():df.to_csv(OUT/f'phase29b_{name}.csv',index=False)
    provenance={'script_sha256':hashfile(Path(__file__)),'frozen_oof_sha256':hashfile(FROZEN),
                'experiment':'TRAIN-only original grouped folds; repeated-OFF predictions not independent',
                'noise_audit':srcmeta,'n_variants':210,'n_positions':168,
                'n_substitutions':206,'n_folds':10,
                'caution':'No measurement noise floor or Bayes ceiling can be inferred from these aggregates; TRAIN fold reuse is exploratory.',
                'nearest_distance_quartiles':{str(k):float(v) for k,v in quartiles.items()}}
    (OUT/'phase29b_provenance.json').write_text(json.dumps(provenance,indent=2,default=str)+'\n')
    print('\nA. FOLD-MATCHED PREDICTABILITY BASELINES (position MAE; lower is better)')
    print(table.round(4).to_string(index=False))
    print('\nB. PAIRED POSITION-CLUSTER BOOTSTRAP (positive means BETTER THAN GLOBAL XGB)')
    print(pd.DataFrame(gains).round(4).to_string(index=False))
    print('\nC. ERRORS BY TRUE RESPONSE REGIME (diagnostic only)')
    print(groups.pivot(index='model',columns='regime',values='position_mae_pp').round(3).to_string())
    print('\nD. EXACT AMINO-ACID-SUBSTITUTION DUPLICATES (not necessarily technical replicates)')
    print(f"Unique AA substitutions={srcmeta['n_aa_substitutions']}, duplicate groups={len(dup)}")
    print(dup.round(3).to_string(index=False) if len(dup) else '(None)')
    print('Processed assay arithmetic max difference (pp):',srcmeta.get('corrector_arithmetic_max_abs_pp','not available'))
    print('Potential replicate/uncertainty fields:',srcmeta.get('possible_replicate_or_uncertainty_columns',[]))
    print('Measurement-noise ceiling estimable from these data: NO (technical repeats/assay SEM not established)')
    print('\nE. FEATURE-SPACE NEAREST NEIGHBORS (OUTER-TRAIN ONLY; first neighbor at distinct position)')
    print(bins.round(3).to_string(index=False))
    print(f'Nearest-neighbor response gap median={q.nearest_response_gap_pp.median():.3f} pp; '
          f'>30 pp fraction={(q.nearest_response_gap_pp>30).mean():.3f}')
    print('Largest response contradictions among NN1 (development diagnostic):')
    print(worst[['variant_key','protein_position','observed_corrector',
                 'repeat1_neighbor_position','repeat1_neighbor_mutation','repeat1_neighbor_distance',
                 'repeat1_neighbor_response','repeat1_response_gap_pp',
                 'nearest_response_gap_pp']].head(args.top).round(3).to_string(index=False))
    print('\nF. CV QC / PROVENANCE')
    print(pd.DataFrame(checks).to_string(index=False))
    print(f'PASS Phase29B; no VAL/TEST labels; outputs={OUT}')


def self_test():
    rng=np.random.default_rng(29)
    groups=np.repeat(np.arange(30),2)
    X=rng.normal(size=(60,68));X[:,0]=2 # zero-variance feature
    y=10+np.arange(60)/10+rng.normal(size=60)
    ix=np.flatnonzero(groups<20);iv=np.flatnonzero(groups>=20)
    p,meta=fold_predictions(X[ix],y[ix],groups[ix],X[iv],groups[iv])
    assert set(p)==set(CONFIGS)-{'global'}
    assert all(v.shape==(len(iv),) and np.isfinite(v).all() for v in p.values())
    assert not set(meta['neighbor_position'])&set(groups[iv])
    z=pd.DataFrame({'protein_position':[0,0,1,1], 'observed_corrector':[0.,0.,20.,20.],
                    'corrector_global':[10.,10.,10.,10.],'corrector_train_mean':[0.,0.,20.,20.]})
    assert position_mae(z,'corrector_global')==10
    assert position_mae(z,'corrector_train_mean')==0
    assert position_mean([2,4,7],[0,0,1])==5
    print('PASS synthetic: position baselines, fold-local scaling/Ridge, distinct-position neighbors, MAE')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bootstraps',type=int,default=2000)
    ap.add_argument('--top',type=int,default=12)
    ap.add_argument('--self-test',action='store_true')
    a=ap.parse_args()
    if a.self_test:self_test()
    else:
        if a.bootstraps<100 or a.top<1:ap.error('Need bootstraps >=100, top>=1')
        self_test();run(a)
