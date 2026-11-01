#!/usr/bin/env python3
"""Phase 28E-5 | TRAIN-only FoldX ΔΔG predictive ablation.

Uses the original frozen 2x5 protein-position-disjoint folds, exact original
XGBoost regressor, and frozen acute predictions. Does NOT access VAL/TEST.
Unlabelled FoldX features are allowed in both train and held-out positions;
no outcome-derived preprocessing or hyperparameter selection is performed.

This is exploratory analysis on multiply reused development folds.
FoldX ΔΔG in engineered drug-bound templates is NOT measured biological
folding free energy and is NOT directly a VTI response prediction.
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
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))
FOLDX = ROOT / 'results/biology/phase28e/phase28e4/phase28e4_foldx_mutation_energies.csv'
MANIFEST = ROOT / 'results/biology/phase28e/phase28e1_foldx_mutation_manifest.csv'
FROZEN = ROOT / 'results/biology/phase26h2/phase26h2_oof.csv'
OUT = ROOT / 'results/biology/phase28e/phase28e5'
KEY = ['repeat','fold','variant_key','protein_position']
CONFIGS = ['global','foldx_consensus','foldx_templates','foldx_qc']


def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def load_foldx(mutations, positions, foldx_path=FOLDX, manifest_path=MANIFEST,
               frozen_cardinality=True):
    """Strict mutation-level join; no position-only match or imputed 'zero' ΔΔG."""
    d = pd.read_csv(foldx_path)
    mandatory={'structure','mutation','protein_position','ddg_mean_kcal_mol',
               'ddg_sd_kcal_mol','n_replicates'}
    if not mandatory.issubset(d):
        raise ValueError(f'Missing FoldX columns {mandatory-set(d)}')
    if d.duplicated(['structure','mutation']).any():
        raise ValueError('Duplicated structure/mutation energies')
    if set(d.structure)!={'7SV7','8EIG'} or (d.n_replicates!=3).any():
        raise ValueError('Unexpected structures or replicate counts')
    if d.groupby('structure').size().to_dict()!={'7SV7':191,'8EIG':188} and frozen_cardinality:
        raise ValueError('Unexpected counts: expected 191 and 188 structure-substitutions')
    if not np.isfinite(d[['ddg_mean_kcal_mol','ddg_sd_kcal_mol']].to_numpy(float)).all():
        raise ValueError('Nonfinite FoldX energy or repeat SD')
    if (d.ddg_sd_kcal_mol<0).any():
        raise ValueError('Negative replicate SD')
    p_from_code=d.mutation.astype(str).str.extract(r'^[A-Z](\d+)[A-Z]$',expand=False)
    if p_from_code.isna().any() or not np.array_equal(p_from_code.astype(int).to_numpy(),d.protein_position.to_numpy(int)):
        raise ValueError('Mutation code and residue position mismatch in FoldX outputs')

    # Verify exact correspondence with Phase 28E-1 eligibility manifest.
    man=pd.read_csv(manifest_path,dtype={'variant_key':str})
    mkeys={'pdb','mutation','protein_position','eligible'}
    if not mkeys.issubset(man): raise ValueError('Missing FoldX source manifest fields')
    eligible=(man.loc[man.eligible==1,['pdb','mutation','protein_position']]
              .drop_duplicates().rename(columns={'pdb':'structure'}))
    checked=d.merge(eligible, on=['structure','mutation','protein_position'],how='outer',indicator=True)
    if (checked._merge!='both').any():
        raise ValueError('Extracted FoldX mutation identities differ from frozen eligible manifest')

    mids=pd.DataFrame({'mutation':list(map(str,mutations)),
                       'protein_position':np.asarray(positions,int)})
    if mids.mutation.isna().any() or len(mids)!=len(positions):
        raise ValueError('Invalid VTI mutation codes')
    unique=mids.drop_duplicates()
    if (unique.groupby('mutation').protein_position.nunique()!=1).any():
        raise ValueError('Mutation maps to more than one protein position')
    if frozen_cardinality:
        if len(mids)!=210 or unique.mutation.nunique()!=206 or mids.protein_position.nunique()!=168:
            raise ValueError('Unexpected frozen VTI TRAIN cohort')
    known=man[['mutation','protein_position']].drop_duplicates()
    if set(map(tuple,unique[['mutation','protein_position']].to_numpy())) != set(map(tuple,known.to_numpy())):
        raise ValueError('FoldX source manifest and VTI TRAIN mutation identities differ')

    w=d.pivot(index='mutation',columns='structure',values='ddg_mean_kcal_mol')
    s=d.pivot(index='mutation',columns='structure',values='ddg_sd_kcal_mol')
    m=mids.mutation.tolist()
    if not set(m).issubset(w.index):
        # Ineligible mutations must still appear in the cohort: reindex instead of rejecting.
        missing=set(m)-set(w.index)
        if not missing.issubset(set(man.loc[man.eligible==0,'mutation'])):
            raise ValueError('Missing ΔΔG for a mutation not documented as ineligible')
    a=w.reindex(m).reindex(columns=['7SV7','8EIG']).to_numpy(float)
    sd=s.reindex(m).reindex(columns=['7SV7','8EIG']).to_numpy(float)
    obs=np.isfinite(a)
    nobs=obs.sum(axis=1)
    mean=np.divide(np.nansum(a,axis=1),nobs,
                   out=np.full(len(m),np.nan),where=nobs>0)
    # Keep missing as NaN, never interpret it as ΔΔG = 0.
    disagreement=np.where(nobs==2,np.abs(a[:,0]-a[:,1]),np.nan)
    mean_sd=np.divide(np.nansum(sd,axis=1),nobs,
                      out=np.full(len(m),np.nan),where=nobs>0)
    feat=pd.DataFrame({'mutation':m,'protein_position':np.asarray(positions,int),
                       'ddg_7sv7':a[:,0], 'ddg_8eig':a[:,1],
                       'has_7sv7':obs[:,0].astype(float),
                       'has_8eig':obs[:,1].astype(float),
                       'n_templates':nobs.astype(float),
                       'ddg_consensus':mean,
                       'ddg_disagreement':disagreement,
                       'ddg_mean_repeat_sd':mean_sd})
    # Type-specific NA handling: rely on native XGBoost NaN support.
    subset={'global':[],
            'foldx_consensus':['ddg_consensus','n_templates'],
            'foldx_templates':['ddg_7sv7','ddg_8eig','has_7sv7','has_8eig'],
            'foldx_qc':['ddg_7sv7','ddg_8eig','has_7sv7','has_8eig',
                        'ddg_disagreement','ddg_mean_repeat_sd']}
    if np.isinf(feat.select_dtypes(include=[np.number]).to_numpy(float)).any():
        raise ValueError('Inf in FoldX engineered features')
    return feat,subset


def position_mae(frame, truth, pred):
    z=frame[['protein_position',truth,pred]].copy()
    z['error']=(z[truth]-z[pred]).abs()
    return float(z.groupby('protein_position').error.mean().mean())


def paired_gain(frame,candidate,reference,truth,boot,seed):
    x=frame[['repeat','protein_position',candidate,reference,truth]].copy()
    x['gain']=(x[reference]-x[truth]).abs()-(x[candidate]-x[truth]).abs()
    pos=(x.groupby(['repeat','protein_position']).gain.mean()
         .groupby('protein_position').mean().to_numpy(float))
    if not np.isfinite(pos).all():raise ValueError('Invalid gains')
    rng=np.random.default_rng(seed)
    idx=rng.integers(len(pos),size=(boot,len(pos)))
    samples=pos[idx].mean(axis=1)
    return {'candidate':candidate,'reference':reference,'target':truth,'gain_pp':float(pos.mean()),
            'ci95_low':float(np.percentile(samples,2.5)),
            'ci95_high':float(np.percentile(samples,97.5)), 'positions':len(pos)}


def rank_ap(frame, config, regime):
    # One row per variant; average repeated OOF predictions before ranking.
    z=(frame.groupby(['variant_key','protein_position'],as_index=False)
       .agg(observed_corrector=('observed_corrector','first'),
            score=('corrector_'+config,'mean')))
    y=(z.observed_corrector<10).astype(int) if regime=='low' else (z.observed_corrector>40).astype(int)
    score=-z.score.to_numpy(float) if regime=='low' else z.score.to_numpy(float)
    if y.nunique()!=2:raise ValueError('Ranking must include positives and negatives')
    k=int(np.ceil(.2*len(z)))
    return {'regime':regime,'model':config,'positives':int(y.sum()),
            'ap':float(average_precision_score(y,score)),
            'auroc':float(roc_auc_score(y,score)),
            'top20_hits':int(y.iloc[np.argsort(-score,kind='stable')[:k]].sum())}


def self_test():
    # Missing sites must remain missing, even if their imputation would look stable.
    import tempfile
    with tempfile.TemporaryDirectory() as t:
        p=Path(t); f=p/'energy.csv'; mn=p/'manifest.csv'
        df=pd.DataFrame([{'structure':'7SV7','mutation':'A10V','protein_position':10,
                          'ddg_mean_kcal_mol':3.,'ddg_sd_kcal_mol':.2,'n_replicates':3},
                         {'structure':'8EIG','mutation':'A10V','protein_position':10,
                          'ddg_mean_kcal_mol':5.,'ddg_sd_kcal_mol':.4,'n_replicates':3},
                         {'structure':'7SV7','mutation':'A11P','protein_position':11,
                          'ddg_mean_kcal_mol':2.,'ddg_sd_kcal_mol':.1,'n_replicates':3}])
        df.to_csv(f,index=False)
        pd.DataFrame([{'pdb':'7SV7','mutation':'A10V','protein_position':10,'eligible':1},
                      {'pdb':'8EIG','mutation':'A10V','protein_position':10,'eligible':1},
                      {'pdb':'7SV7','mutation':'A11P','protein_position':11,'eligible':1},
                      {'pdb':'8EIG','mutation':'A11P','protein_position':11,'eligible':0},
                      {'pdb':'7SV7','mutation':'A12G','protein_position':12,'eligible':0},
                      {'pdb':'8EIG','mutation':'A12G','protein_position':12,'eligible':0}]).to_csv(mn,index=False)
        feat,subset=load_foldx(['A10V','A11P','A12G'],[10,11,12],f,mn,False)
        assert feat.ddg_consensus.iloc[0]==4 and feat.ddg_disagreement.iloc[0]==2
        assert feat.ddg_consensus.iloc[1]==2 and np.isnan(feat.ddg_disagreement.iloc[1])
        assert np.isnan(feat.ddg_consensus.iloc[2]) and feat.n_templates.iloc[2]==0
        assert len(subset)==4
    z=pd.DataFrame({'repeat':[1,1,2,2], 'protein_position':[1,2,1,2],
                    'truth':[1.,3.,1.,3.], 'base':[2.,2.,2.,2.], 'cand':[1.,3.,1.,3.]})
    assert abs(paired_gain(z,'cand','base','truth',100,42)['gain_pp']-1)<1e-12
    print('PASS synthetic: FoldX manifest mapping, missing energy flags, cross-template QC, cluster bootstrap')


def run(args):
    import phase26h_auxiliary_corrector as h
    import phase27d1_biology_ablation as d1
    v,x80,_,_,folds=h.load_data()
    keys=v.variant_key.astype(str).to_numpy()
    pp=v.protein_position.to_numpy(int)
    x=np.asarray(x80,float)
    yc=v.corrector_rescue_pp.to_numpy(float)
    yt=v.vti_rescue_pp.to_numpy(float)
    if x.shape!=(210,80) or len(set(keys))!=210 or len(set(pp))!=168 or len(folds)!=10:
        raise ValueError('Frozen cohort/folds changed')
    if not np.isfinite(x[:,:68]).all() or np.isinf(x).any():
        raise ValueError('Invalid baseline sequence features')
    muts=d1.extract_codes(v,pp)
    feat,subset=load_foldx(muts,pp)
    dmodels={name:(x if name=='global' else np.column_stack([x,feat[cols].to_numpy(float)]))
             for name,cols in subset.items()}
    frozen=pd.read_csv(FROZEN,dtype={'variant_key':str})
    if len(frozen)!=420 or frozen.duplicated(KEY).any():
        raise ValueError('Frozen reference cardinality or fold keys invalid')
    mandatory=set(KEY+['vti_fixed','vti_fixed_total','observed_corrector','observed_total'])
    if not mandatory.issubset(frozen):
        raise ValueError(f'Missing reference OOF fields: {mandatory-set(frozen)}')

    print('='*112)
    print('PHASE 28E-5 | FOLDX MUTATION ΔΔG: PREDICTIVE ABLATION | VTI TRAIN ONLY')
    print('='*112)
    print('Frozen 2×5 position-disjoint folds; unchanged XGBoost parameters and acute predictions.')
    print('NO protected VAL/TEST labels; exploratory development folds have been extensively reused.')
    print('Unlabelled FoldX energies from 7SV7/8EIG are not measured ΔΔG or VTI outcomes.')
    print('\nA. FOLDX FEATURE / QC COVERAGE')
    p=feat.drop_duplicates(['mutation'])
    print('Coverage unique protein substitutions (out of 206):',
          '7SV7=',int(p.has_7sv7.sum()),'8EIG=',int(p.has_8eig.sum()),
          'both=',int((p.n_templates==2).sum()),
          'neither=',int((p.n_templates==0).sum()))
    print('Coverage VTI records (out of 210):',
          '7SV7=',int(feat.has_7sv7.sum()),'8EIG=',int(feat.has_8eig.sum()),
          'both=',int((feat.n_templates==2).sum()),'neither=',int((feat.n_templates==0).sum()))
    q=feat.loc[feat.n_templates==2]
    print(f'Median cross-template |ΔΔG difference|={q.ddg_disagreement.median():.3f} kcal/mol')
    print('High-disagreement records (>5 kcal/mol):',int((q.ddg_disagreement>5).sum()))
    print('Replicate-SD >2 kcal/mol:',int((feat.ddg_mean_repeat_sd>2).sum()))
    print('Feature matrices:',{name:arr.shape for name,arr in dmodels.items()})

    rows=[];qc=[]
    for i,(repeat,fold,tr_keys,te_keys) in enumerate(folds,1):
        tr=np.flatnonzero(np.isin(keys,list(tr_keys)))
        te=np.flatnonzero(np.isin(keys,list(te_keys)))
        if len(tr)+len(te)!=210 or len(set(keys[te]))!=len(te) or set(pp[tr])&set(pp[te]):
            raise ValueError(f'Position-disjoint fold integrity failure repeat={repeat} fold={fold}')
        ref=frozen.loc[(frozen.repeat==repeat)&(frozen.fold==fold)].set_index('variant_key')
        if set(ref.index)!=set(keys[te]):
            raise ValueError('Held-out membership differs from frozen reference')
        ref=ref.loc[keys[te]]
        if not np.allclose(ref.observed_corrector.to_numpy(float),yc[te],rtol=0,atol=1e-8):
            raise ValueError('Frozen corrector outcome parity failed')
        if not np.allclose(ref.observed_total.to_numpy(float),yt[te],rtol=0,atol=1e-8):
            raise ValueError('Frozen total outcome parity failed')
        acute=ref.vti_fixed_total.to_numpy(float)-ref.vti_fixed.to_numpy(float)
        preds={}
        for name,arr in dmodels.items():
            a=np.column_stack([arr,np.ones(len(arr))])
            mdl=h.previous.model(h.SPEC)
            mdl.fit(a[tr],yc[tr])
            preds[name]=np.asarray(mdl.predict(a[te]),float)
        parity=float(np.max(np.abs(preds['global']-ref.vti_fixed.to_numpy(float))))
        if parity>1e-5: raise ValueError(f'Frozen global regression mismatch: {parity}, fold={i}')
        qc.append({'repeat':repeat,'fold':fold,'training_records':len(tr),
                   'heldout_records':len(te),'train_positions':len(set(pp[tr])),
                   'heldout_positions':len(set(pp[te])),'global_parity_max_abs':parity})
        for j,idx in enumerate(te):
            row={'repeat':repeat,'fold':fold,'variant_key':keys[idx],
                 'protein_position':int(pp[idx]),'mutation':muts[idx],
                 'observed_corrector':float(yc[idx]),'observed_total':float(yt[idx]),
                 'frozen_acute':float(acute[j]),
                 'regime':'low' if yc[idx]<10 else 'high' if yc[idx]>40 else 'middle',
                 'n_templates':int(feat.n_templates.iloc[idx])}
            for name,yhat in preds.items():
                row['corrector_'+name]=float(yhat[j]);
                row['total_'+name]=float(yhat[j]+acute[j])
            rows.append(row)
        print(f'  Fold {i:02d}/10 repeat={repeat} fold={fold}: parity={parity:.2e}',flush=True)
    oof=pd.DataFrame(rows)
    if len(oof)!=420 or oof.duplicated(KEY).any():raise ValueError('OOF output mismatch')
    metric=[];regime=[];gain=[];rank=[]
    for target in ('corrector','total'):
        truth='observed_'+target
        for name in CONFIGS:
            pred=target+'_'+name
            metric.append({'target':target,'config':name,
                'position_mae_pp':float(np.mean([position_mae(f,truth,pred) for _,f in oof.groupby('repeat')])),
                'variant_mae_pp':float(np.mean(np.abs(oof[pred]-oof[truth]))),
                'signed_bias_pp':float(np.mean(oof[pred]-oof[truth]))})
            if target=='corrector':
                for label in ['low','middle','high']:
                    sub=oof[oof.regime==label]
                    regime.append({'regime':label,'config':name,
                                   'n_variants':sub.variant_key.nunique(),
                                   'positions':sub.protein_position.nunique(),
                                   'position_mae_pp':float(np.mean([position_mae(f,truth,pred) for _,f in sub.groupby('repeat')])),
                                   'signed_bias_pp':float(np.mean(sub[pred]-sub[truth]))})
        for i,name in enumerate(CONFIGS[1:]):
            gain.append(paired_gain(oof,target+'_'+name,target+'_global',truth,args.bootstraps,
                                    seed=2850+(0 if target=='corrector' else 20)+i))
    for label in ('low','high'):
        for name in CONFIGS:
            rank.append(rank_ap(oof,name,label))

    OUT.mkdir(parents=True,exist_ok=True)
    feat_out=pd.concat([v[['variant_key','protein_position']].reset_index(drop=True),feat.drop(columns='protein_position')],axis=1)
    feat_out.to_csv(OUT/'phase28e5_train_foldx_features.csv',index=False)
    oof.to_csv(OUT/'phase28e5_repeated_oof.csv',index=False)
    pd.DataFrame(metric).to_csv(OUT/'phase28e5_metrics.csv',index=False)
    pd.DataFrame(regime).to_csv(OUT/'phase28e5_regime_mae.csv',index=False)
    pd.DataFrame(gain).to_csv(OUT/'phase28e5_mae_gains.csv',index=False)
    pd.DataFrame(rank).to_csv(OUT/'phase28e5_regression_ranking.csv',index=False)
    pd.DataFrame(qc).to_csv(OUT/'phase28e5_fold_qc.csv',index=False)
    report={'foldx_sha256':sha256(FOLDX),'manifest_sha256':sha256(MANIFEST),
            'frozen_oof_sha256':sha256(FROZEN),
            'feature_configs':subset,'script_sha256':sha256(Path(__file__)),
            'n_train':210,'n_positions':168,'bootstraps':args.bootstraps,
            'warning':'Exploratory TRAIN-only model comparison; not independent validation.'}
    (OUT/'phase28e5_provenance.json').write_text(json.dumps(report,indent=2)+'\n')
    print('\nB. OVERALL REGRESSION — POSITION-WEIGHTED MAE (LOWER IS BETTER)')
    print(pd.DataFrame(metric).round(4).to_string(index=False))
    print('\nC. CORRECTOR MAE BY TRUE RESPONSE REGIME — DEVELOPMENT DIAGNOSTIC')
    print(pd.DataFrame(regime).round(4).to_string(index=False))
    print('\nD. PAIRED POSITION-CLUSTERED MAE GAINS — POSITIVE = LOWER MAE')
    print(pd.DataFrame(gain).round(4).to_string(index=False))
    print('\nE. LOW/HIGH RESPONDER RANKING USING CONTINUOUS REGRESSION OUTPUT')
    print(pd.DataFrame(rank).round(4).to_string(index=False))
    print('\nF. ORIGINAL GLOBAL-REGRESSION PARITY')
    print(pd.DataFrame(qc).to_string(index=False))
    print('PASS: Phase 28E-5. VTI TRAIN only; no validation or test data read.')
    print('Outputs:',OUT)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--bootstraps',type=int,default=2000)
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        if args.bootstraps<100:parser.error('--bootstraps must be >=100')
        self_test()
        run(args)
