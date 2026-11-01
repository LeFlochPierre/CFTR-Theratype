#!/usr/bin/env python3
"""Phase 21E: leakage-safe Tedman-to-VTI acute IVA transfer (exploratory).

Features at VTI inference: original 80 sequence+structure features PLUS
predictions of 5 Tedman experimental surface-expression traits, obtained
using only fixed ESM-2 mutation embeddings. No measured Tedman or Bihler
auxiliary phenotype enters the final model at VTI inference.

Training-stage external-phenotype predictions are cross-fitted by VTI protein
position. In every stage-1 fit, Tedman rows at the held VTI positions are
excluded, including the frozen outer held positions. Thus exact matched
substitutions cannot pass their measured external labels to the corresponding
held-out target features. PCA/normalization/ridge fitting are strictly local.

Comparisons: exact historical XGB control, external all-allowed, external novel
VTI positions only, and shuffled-external-label negative control. Uses fixed,
prespecified stage-1 hyperparameters, not selected against outer OOF outcomes.

This is an external biological *transfer* test, NOT yet a neural multitask
network. The external workbook is not an acute IVA label source. Results
remain exploratory across many phases; reserve protected validation and test.

Run:
  python scripts/phase21e_external_biology_transfer.py self-test
  python scripts/phase21e_external_biology_transfer.py extract --device cuda --batch-size 4
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/phase21e_external_biology_transfer.py benchmark --smoke
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/phase21e_external_biology_transfer.py benchmark
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

MODEL_ID = 'facebook/esm2_t12_35M_UR50D'
WINDOW = 511
PCA_DIM = 32
RIDGE_ALPHA = 10.0
TARGET = 'acute_iva_vti_pp'
MODES = ('control_xgb', 'external_all', 'external_novel_only', 'external_shuffled')
TRAITS = (
    'log1p_parent_apo',
    'log1p_parent_vx445_vx661',
    'log_ratio_parent_vx445_vx661_vs_apo',
    'log_ratio_canxko_parent_apo',
    'log_ratio_canxko_parent_vx445_vx661',
)


def project_root():
    from cftr_theratype.transfer import ROOT
    return ROOT


def paths(root):
    out = root / 'results/biology/phase21e'
    return {
        'out': out,
        'external': root / 'results/biology/phase21c/tedman2025_candidate_biology_targets.csv',
        'external_cache': out / 'tedman2025_esm2_162.npz',
        'vti_cache': root / 'results/boltz/phase20/esm2_vti_train_210.npz',
        'acute': root / 'results/boltz/phase19b/vti_train_acute_iva_targets.csv',
        'selections': root / 'results/boltz/phase19c/acute_iva_vti_pp_inner_selection.csv',
        'hist': root / 'results/boltz/phase19c/acute_iva_vti_pp_oof.csv',
        'reliability': root / 'results/boltz/phase19i/phase19i_averaged.csv',
        'membership': root / 'data/processed/bihler_2026_transfer_splits.parquet',
    }


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1_048_576), b''):
            h.update(block)
    return h.hexdigest()


def external_safe(source, member, ref):
    e = pd.read_csv(source)
    m = pd.read_parquet(member)
    assert len(m) == 415 and len(e) == 162, (len(m),len(e))
    assert e.variant_short.is_unique and len(e.variant_short.dropna()) == 162
    assert e.protein_position.nunique() == 138
    assert {'protein_position','transfer_split'}.issubset(m)
    blocked = set(m.loc[m.transfer_split.ne('train'), 'protein_position'].dropna().astype(int))
    if 'protected_2024_validation_or_test_position' in m:
        flag = m.protected_2024_validation_or_test_position.fillna(False).astype(str).str.lower().isin(('true','1','yes','t'))
        blocked |= set(m.loc[flag,'protein_position'].dropna().astype(int))
    assert not blocked.intersection(e.protein_position.astype(int)), 'Protected VTI/2024 position in external cache'
    assert set(TRAITS).issubset(e), f'Missing Tedman targets: {set(TRAITS)-set(e)}'
    for r in e.itertuples(index=False):
        p = int(r.protein_position)
        short = str(r.variant_short)
        import re
        mm = re.fullmatch(r'([ACDEFGHIKLMNPQRSTVWY])([1-9]\d*)([ACDEFGHIKLMNPQRSTVWY])',short)
        assert mm and int(mm.group(2)) == p and 1 <= p <= len(ref) and ref[p-1] == mm.group(1), short
    yy = e[list(TRAITS)].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    assert np.isfinite(yy).all(), 'Missing/nonfinite external biological targets'
    return e, yy


def fixed_window(ref, pos):
    k = pos-1
    start = max(0, min(k-WINDOW//2, len(ref)-WINDOW))
    window = ref[start:start+WINDOW]
    offset = k-start
    assert len(window) == WINDOW and window[offset] == ref[k]
    return window,offset


def extract(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from cftr_theratype.reference import bihler_2024_assay_sequence
    root = project_root(); p=paths(root)
    ref = bihler_2024_assay_sequence().upper()
    e,_ = external_safe(p['external'],p['membership'],ref)
    p['out'].mkdir(parents=True,exist_ok=True)
    if p['external_cache'].exists() and not args.overwrite:
        raise FileExistsError(f'Cache exists: {p["external_cache"]}; do not overwrite unintentionally')
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; use --device cpu')
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(args.device).eval()
    assert int(model.config.hidden_size) == 480
    jobs=[]
    for pos in sorted(set(e.protein_position.astype(int))):
        w,idx=fixed_window(ref,pos); jobs.append(('wt',pos,w,idx))
    for r in e.itertuples(index=False):
        w,idx=fixed_window(ref,int(r.protein_position))
        alt=str(r.variant_short)[-1]
        jobs.append(('mut',r.variant_short,w[:idx]+alt+w[idx+1:],idx))
    print('='*100,flush=True)
    print('PHASE 21E-A | FROZEN ESM-2 FEATURES FOR SAFE TEDMAN SUBSTITUTIONS',flush=True)
    print('='*100,flush=True)
    print(f'External variants={len(e)}, positions={e.protein_position.nunique()}, unique sequences={len(jobs)}',flush=True)
    print('No protected VTI/ETI experimental outcomes loaded. Protein model remains frozen.',flush=True)
    wt={}; mut={}
    for start in range(0,len(jobs),args.batch_size):
        batch=jobs[start:start+args.batch_size]
        enc=tok([r[2] for r in batch],return_tensors='pt',padding=True,add_special_tokens=True,truncation=False)
        assert enc['input_ids'].shape[1] == WINDOW+2
        enc={k:v.to(args.device) for k,v in enc.items()}
        with torch.inference_mode():
            h=model(**enc).last_hidden_state.detach().float().cpu().numpy()
        for j,(kind,key,_,off) in enumerate(batch):
            v=np.asarray(h[j,off+1],np.float32).copy()
            assert v.shape==(480,) and np.isfinite(v).all()
            (wt if kind=='wt' else mut)[key]=v
        if start==0 or (start//args.batch_size+1)%20==0 or start+args.batch_size>=len(jobs):
            print(f'ESM progress {min(len(jobs),start+args.batch_size)}/{len(jobs)}',flush=True)
    vals=[]
    for r in e.itertuples(index=False):
        w=wt[int(r.protein_position)]; difference=mut[r.variant_short]-w
        vals.append(np.concatenate([w,difference]))
    arr=np.asarray(vals,np.float32)
    assert arr.shape==(162,960) and np.isfinite(arr).all()
    info={
        'model_id':MODEL_ID,'window_aa':WINDOW,
        'source_sha256':digest(p['external']),
        'reference_sha256':hashlib.sha256(ref.encode()).hexdigest(),
        'n_external':len(e),'n_positions':int(e.protein_position.nunique()),
        'feature':'site_wt_delta','no_protected_outcomes':True,
        'torch_version':str(torch.__version__),
        'model_revision':getattr(model.config,'_commit_hash',None),
    }
    tmp=p['out']/'tedman2025_esm2_162.tmp.npz'
    np.savez_compressed(tmp,keys=e.variant_short.to_numpy(dtype=str),
                        positions=e.protein_position.to_numpy(int),
                        site_wt_delta=arr,metadata=json.dumps(info,sort_keys=True))
    with np.load(tmp,allow_pickle=False) as test:
        assert np.array_equal(test['keys'].astype(str),e.variant_short.to_numpy(dtype=str))
    tmp.replace(p['external_cache'])
    print('PASS: Saved',p['external_cache'].relative_to(root),flush=True)
    print('SHA256:',digest(p['external_cache']),flush=True)


def load_training():
    from cftr_theratype import model_zoo as mz
    from cftr_theratype.reference import bihler_2024_assay_sequence
    root=project_root(); p=paths(root)
    train,mats,_,outer=mz.prepare(root)
    assert len(train)==210 and train.protein_position.nunique()==168 and len(outer)==10
    assert train.variant_key.is_unique
    ref=bihler_2024_assay_sequence().upper()
    e,labels=external_safe(p['external'],p['membership'],ref)
    with np.load(p['vti_cache'],allow_pickle=False) as z:
        keys=z['keys'].astype(str)
        pos=z['positions'].astype(int)
        vv=np.asarray(z['site_wt_delta'],float)
        meta=json.loads(str(z['metadata'].item()))
    assert np.array_equal(keys,train.variant_key.to_numpy(dtype=str))
    assert np.array_equal(pos,train.protein_position.to_numpy(int))
    assert vv.shape==(210,960) and np.isfinite(vv).all()
    assert meta.get('model_id')==MODEL_ID and meta.get('window_aa')==WINDOW
    ref_hash=hashlib.sha256(ref.encode()).hexdigest()
    assert meta.get('assay_reference_sha256')==ref_hash
    with np.load(p['external_cache'],allow_pickle=False) as z:
        ek=z['keys'].astype(str)
        ep=z['positions'].astype(int)
        ev=np.asarray(z['site_wt_delta'],float)
        emeta=json.loads(str(z['metadata'].item()))
    assert np.array_equal(ek,e.variant_short.to_numpy(dtype=str))
    assert np.array_equal(ep,e.protein_position.to_numpy(int))
    assert ev.shape==(162,960) and np.isfinite(ev).all()
    assert emeta.get('source_sha256')==digest(p['external'])
    assert emeta.get('reference_sha256')==ref_hash
    assert emeta.get('model_id')==MODEL_ID and emeta.get('window_aa')==WINDOW
    oldrev=meta.get('model_revision_hash')
    newrev=emeta.get('model_revision')
    if oldrev is not None and newrev is not None:
        assert oldrev==newrev, 'ESM pretrained checkpoint revision changed since Phase20'
    # Exact protein substitutions across experiments must give the same frozen
    # sequence embedding. This verifies reference/window/feature consistency.
    shorts=(train.reference_aa.astype(str).str.upper()
            +train.protein_position.astype(int).astype(str)
            +train.alternate_aa.astype(str).str.upper())
    ext_by_short={k:j for j,k in enumerate(ek)}
    matched=[(i,ext_by_short[key]) for i,key in enumerate(shorts) if key in ext_by_short]
    assert len(matched)==14, f'Expected 14 exact matched substitutions, got {len(matched)}'
    delta=max(float(np.max(np.abs(vv[i]-ev[j]))) for i,j in matched)
    if delta>0.01:
        raise ValueError(f'Cross-study ESM feature identity parity failure, max abs diff={delta:.6g}')
    print(f'PASS: {len(matched)} exact VTI/Tedman sequence representations matched; max diff={delta:.7f}',flush=True)
    X=np.asarray(mats['sequence_structure'],float)
    assert X.shape==(210,80) and not np.isinf(X).any()
    ydf=pd.read_csv(p['acute'],dtype={'variant_key':str})
    assert len(ydf)==210 and ydf.variant_key.is_unique
    yy=train[['variant_key']].merge(ydf[['variant_key',TARGET]],on='variant_key',validate='one_to_one',sort=False)
    assert yy.variant_key.tolist()==train.variant_key.tolist()
    y=yy[TARGET].to_numpy(float)
    assert np.isfinite(y).all()
    sel=pd.read_csv(p['selections'])
    sel=sel.loc[sel.model.eq('sequence_structure__xgboost')]
    assert len(sel)==10 and not sel.duplicated(['repeat','fold']).any()
    prm={(int(r.repeat),int(r.fold)):json.loads(r.hyperparams) for r in sel.itertuples(index=False)}
    return train,X,vv,y,outer,mz,e,ev,labels,prm,p


def fit_external_model(Xext,Yext,idx,eval_x):
    """Fixed regularized multivariate assay predictor, fit only on permitted ext rows."""
    a=np.asarray(Xext[idx],float); b=np.asarray(eval_x,float)
    yy=np.asarray(Yext[idx],float)
    assert a.ndim==2 and a.shape[1]==960 and yy.shape==(len(a),len(TRAITS))
    assert len(a)>=45 and np.isfinite(a).all() and np.isfinite(yy).all() and np.isfinite(b).all()
    scaler=StandardScaler().fit(a)
    a=scaler.transform(a); b=scaler.transform(b)
    k=min(PCA_DIM,len(a)-1,a.shape[1])
    reducer=PCA(n_components=k,svd_solver='randomized',random_state=42)
    a=reducer.fit_transform(a); b=reducer.transform(b)
    loc=yy.mean(axis=0); sd=yy.std(axis=0)
    sd=np.maximum(sd,1e-6)
    reg=Ridge(alpha=RIDGE_ALPHA)
    reg.fit(a,(yy-loc)/sd)
    pred=reg.predict(b)*sd+loc
    assert pred.shape==(len(eval_x),len(TRAITS)) and np.isfinite(pred).all()
    return pred


def crossfit_external(vti_fit,vti_held,fit_positions,held_positions,
                      Xext,Yext,ext_positions,seed):
    """Predict external phenotypes with disjoint stage-1 external labels.

    External at OUTER held positions is excluded for EVERY stage-1 model.
    Training VTI predictions are GROUP-cross-fitted; external at the INNER
    held VTI group is ALSO removed, preventing exact-variant leakage.
    """
    fit_positions=np.asarray(fit_positions,int)
    held_positions=np.asarray(held_positions,int)
    ext_positions=np.asarray(ext_positions,int)
    assert not np.intersect1d(fit_positions,held_positions).size
    assert Xext.shape[0]==Yext.shape[0]==len(ext_positions)
    outer_safe=~np.isin(ext_positions,held_positions)
    assert outer_safe.sum()>=45
    result=np.full((len(vti_fit),len(TRAITS)),np.nan)
    cv=GroupKFold(n_splits=3)
    for subfit,subheld in cv.split(vti_fit,groups=fit_positions):
        assert not np.intersect1d(fit_positions[subfit],fit_positions[subheld]).size
        safe=outer_safe & ~np.isin(ext_positions,fit_positions[subheld])
        assert not np.isin(ext_positions[safe],held_positions).any()
        assert not np.isin(ext_positions[safe],fit_positions[subheld]).any()
        result[subheld]=fit_external_model(Xext,Yext,np.where(safe)[0],vti_fit[subheld])
    assert np.isfinite(result).all()
    out=fit_external_model(Xext,Yext,np.where(outer_safe)[0],vti_held)
    return result,out,int(outer_safe.sum())


def pos_mae(y,p,pos):
    d=pd.DataFrame({'position':np.asarray(pos,int), 'ae':np.abs(np.asarray(y)-np.asarray(p))})
    return float(d.groupby('position').ae.mean().mean())


def performance(avg):
    out=[]
    for mode,g in avg.groupby('model'):
        y=g.observed_pp.to_numpy(float);pred=g.predicted_pp.to_numpy(float)
        pos=g.protein_position.to_numpy(int)
        hi20=y>20;hi40=y>40;lo2=y<=2
        out.append(dict(model=mode,position_mae_pp=pos_mae(y,pred,pos),
            variant_mae_pp=float(np.mean(np.abs(y-pred))),
            rmse_pp=float(np.sqrt(np.mean((y-pred)**2))),
            high20_mae_pp=float(np.mean(np.abs(y[hi20]-pred[hi20]))),
            high40_mae_pp=float(np.mean(np.abs(y[hi40]-pred[hi40]))),
            low2_mae_pp=float(np.mean(np.abs(y[lo2]-pred[lo2]))),
            high20_recall=float(np.mean(pred[hi20]>20)),max_predicted_pp=float(np.max(pred))))
    return pd.DataFrame(out).sort_values('position_mae_pp')


def bootstrap(avg):
    g=avg.assign(ae=np.abs(avg.observed_pp-avg.predicted_pp))
    wide=g.groupby(['protein_position','model']).ae.mean().unstack()
    assert len(wide)==168 and wide.notna().all().all()
    rng=np.random.default_rng(42)
    draws=rng.integers(0,len(wide),(20000,len(wide)))
    out=[]
    for mode in ('external_all','external_novel_only','external_shuffled'):
        refs=['control_xgb']
        if 'historical_reliability_19i' in wide:refs.append('historical_reliability_19i')
        if mode in ('external_all','external_novel_only'):refs.append('external_shuffled')
        if mode=='external_novel_only':refs.append('external_all')
        for ref in refs:
            delta=(wide[ref]-wide[mode]).to_numpy(float)
            bs=delta[draws].mean(axis=1)
            low,high=np.quantile(bs,[.025,.975])
            out.append(dict(model=mode,comparator=ref,gain_pp=float(delta.mean()),
                ci95_lo=float(low),ci95_hi=float(high),positions_improved=int((delta>0).sum()),
                frac_boot_positive=float((bs>0).mean())))
    return pd.DataFrame(out)


def benchmark(args):
    train,X,V,y,outer,mz,ext,EV,Y,params,p=load_training()
    pos=train.protein_position.to_numpy(int)
    ext_pos=ext.protein_position.to_numpy(int)
    train_positions=set(pos.tolist())
    novel=~np.isin(ext_pos,list(train_positions))
    assert novel.sum()==123 and len(set(ext_pos[novel]))==106
    print('='*110,flush=True)
    print('PHASE 21E-B | TEDMAN EXPERIMENTAL BIOLOGY -> ACUTE IVA, POSITION-DISJOINT',flush=True)
    print('='*110,flush=True)
    print('VTI TRAIN 210 variants/168 positions; 2x5 frozen group outer folds',flush=True)
    print('External 162 substitutions/138 positions: 39 within VTI TRAIN positions + 123 novel',flush=True)
    print('Stage-1 ESM 960 -> 32 PCA -> fixed multioutput Ridge(alpha=10); 5 Tedman targets',flush=True)
    print('In every stage1 fit: exclude VTI outer-held AND stage1 crossfit-held positions',flush=True)
    print('No measured external phenotype available to the VTI acute model at inference.',flush=True)
    print('XGB parameters frozen from Phase 19C. No strength or model selection on outer OOF.',flush=True)
    print('No protected VTI / ETI experimental outcomes accessed.',flush=True)
    print('Run:', 'SMOKE fold' if args.smoke else 'FULL 2x5',flush=True)
    rows=[]; details=[]
    for rep,fold,fit,held in (outer[:1] if args.smoke else outer):
        rep=int(rep);fold=int(fold);seed=mz.SEED+rep*100+fold
        assert not np.intersect1d(pos[fit],pos[held]).size
        t0=time.monotonic()
        core=mz.estimator('xgboost',params[(rep,fold)],seed)
        core.fit(X[fit],y[fit]); base=np.asarray(core.predict(X[held]),float)
        predictions={'control_xgb':base}
        cfgs={
            'external_all':(EV,Y,ext_pos),
            'external_novel_only':(EV[novel],Y[novel],ext_pos[novel]),
        }
        allowed=~np.isin(ext_pos,pos[held])
        assert allowed.sum()>=45
        rng=np.random.default_rng(20261027+rep*100+fold)
        # Shuffle biological labels while retaining external mutation features.
        shuffled=Y.copy()
        shuffled[allowed]=Y[allowed][rng.permutation(int(allowed.sum()))]
        cfgs['external_shuffled']=(EV,shuffled,ext_pos)
        for mode,(xext,yext,pext) in cfgs.items():
            xf,xh,nallowed=crossfit_external(V[fit],V[held],pos[fit],pos[held],
                                           xext,yext,pext,seed)
            reg=mz.estimator('xgboost',params[(rep,fold)],seed)
            reg.fit(np.column_stack([X[fit],xf]),y[fit])
            pred=np.asarray(reg.predict(np.column_stack([X[held],xh])),float)
            assert pred.shape==base.shape and np.isfinite(pred).all()
            predictions[mode]=pred
            details.append(dict(repeat=rep,fold=fold,mode=mode,
                n_external_allowed=nallowed,n_external_positions_allowed=len(set(pext)-set(pos[held])),
                n_external_removed_outer=int(len(pext)-nallowed)))
        print(f'\nOuter repeat={rep} fold={fold}: train={len(fit)} held={len(held)}',flush=True)
        for mode,pred in predictions.items():
            print(f'  {mode:24s} position_MAE={pos_mae(y[held],pred,pos[held]):.3f}',flush=True)
            for ix,pred_val in zip(held,pred):
                rows.append(dict(repeat=rep,fold=fold,model=mode,
                     variant_key=str(train.variant_key.iloc[ix]),protein_position=int(pos[ix]),
                     observed_pp=float(y[ix]),predicted_pp=float(pred_val)))
        print(f'  elapsed {time.monotonic()-t0:.1f}s',flush=True)
    if args.smoke:
        print('\nPASS: Phase 21E smoke fold; no result files written',flush=True)
        return
    oof=pd.DataFrame(rows)
    assert len(oof)==len(MODES)*210*2 and not oof.duplicated(['repeat','model','variant_key']).any()
    hist=pd.read_csv(p['hist'],dtype={'variant_key':str})
    hist=hist.loc[hist.model.eq('sequence_structure__xgboost')]
    chk=oof.loc[oof.model.eq('control_xgb')].merge(
        hist[['repeat','fold','variant_key','observed_pp','predicted_pp']],
        on=['repeat','fold','variant_key'],how='outer',validate='one_to_one',
        suffixes=('_new','_hist'),indicator=True)
    assert len(chk)==420 and chk._merge.eq('both').all(), 'Phase19C historical parity key mismatch'
    d=float(np.max(np.abs(chk.predicted_pp_new-chk.predicted_pp_hist)))
    assert np.allclose(chk.observed_pp_new,chk.observed_pp_hist,rtol=0,atol=1e-8)
    assert d<1e-3, f'Historical XGBoost parity failed: max {d} pp'
    print(f'\nPASS: historical XGB frozen-OOF parity (max diff {d:.9f} pp)',flush=True)
    avg=oof.groupby(['model','variant_key','protein_position'],as_index=False).agg(
        observed_pp=('observed_pp','first'),predicted_pp=('predicted_pp','mean'),
        n_repeats=('repeat','nunique'))
    assert avg.groupby('model').size().eq(210).all() and avg.n_repeats.eq(2).all()
    prior_added=False
    if p['reliability'].is_file():
        prior=pd.read_csv(p['reliability'],dtype={'variant_key':str})
        prior=prior.loc[prior.model.eq('reliability_weighted')].copy()
        assert len(prior)==210 and prior.variant_key.is_unique
        j=avg.loc[avg.model.eq('control_xgb')].merge(
            prior[['variant_key','protein_position','observed_pp']],on='variant_key',
            suffixes=('_new','_old'),validate='one_to_one')
        assert len(j)==210 and (j.protein_position_new==j.protein_position_old).all()
        assert np.allclose(j.observed_pp_new,j.observed_pp_old,atol=1e-8,rtol=0)
        prior['model']='historical_reliability_19i'
        cols=['model','variant_key','protein_position','observed_pp','predicted_pp']
        avg_bench=pd.concat([avg[cols],prior[cols]],ignore_index=True)
        prior_added=True
    else:
        avg_bench=avg
    scores=performance(avg)
    boot=bootstrap(avg_bench)
    print('\n'+'='*110,flush=True)
    print('PHASE 21E | REPEATED OOF POSITION-MACRO RESULTS',flush=True)
    print('='*110,flush=True)
    print(scores.round(4).to_string(index=False),flush=True)
    if prior_added:
        a=avg_bench.loc[avg_bench.model.eq('historical_reliability_19i')]
        print('Historical reliability-weighted position-MAE: '
              f'{pos_mae(a.observed_pp,a.predicted_pp,a.protein_position):.4f} pp',flush=True)
    print('\nPAIRED POSITION BOOTSTRAP — 168 protein positions, 20k draws (unadjusted)',flush=True)
    print(boot.round(4).to_string(index=False),flush=True)
    print('\nExternal supervised sample counts by outer fold:',flush=True)
    print(pd.DataFrame(details).to_string(index=False),flush=True)
    print('\nCAVEATS: external label semantics differ from acute IVA; our fixed transfer may fail.',flush=True)
    print('Novel-only comparator tests independent POSITION expansion, not necessarily independent studies.',flush=True)
    print('Study-wide multiple testing across Phases 19-21 requires protected validation.',flush=True)
    p['out'].mkdir(parents=True,exist_ok=True)
    files={
       'oof':p['out']/'phase21e_oof.csv',
       'averaged':p['out']/'phase21e_averaged.csv',
       'summary':p['out']/'phase21e_summary.csv',
       'bootstrap':p['out']/'phase21e_bootstrap.csv',
       'audit':p['out']/'phase21e_fold_filter_audit.csv',
       'report':p['out']/'phase21e_report.json',
    }
    oof.to_csv(files['oof'],index=False)
    avg.to_csv(files['averaged'],index=False)
    scores.to_csv(files['summary'],index=False)
    boot.to_csv(files['bootstrap'],index=False)
    pd.DataFrame(details).to_csv(files['audit'],index=False)
    report={
       'phase':'21E_external_biology_transfer',
       'n_vti_train':210,'n_vti_positions':168,'n_external_safe':162,
       'n_external_new_positions':106,'outer_folds':10,
       'stage1_group_crossfit':3,'stage1_pca_dimensions':PCA_DIM,
       'stage1_ridge_alpha':RIDGE_ALPHA,'auxiliary_traits':list(TRAITS),
       'models':list(MODES),'historical_xgb_parity_max_pp':d,
       'historical_reliability_loaded':prior_added,
       'protected_validation_test_outcomes_loaded':False,
       'data_source_hashes':{
           str(v.relative_to(project_root())):digest(v)
           for v in [p['external'],p['external_cache'],p['vti_cache'],p['acute'],p['membership']]
       },
       'note':'Exploratory fixed-parameter transfer, not multitask NN or independent validation',
    }
    files['report'].write_text(json.dumps(report,indent=2)+'\n')
    print('\nOUTPUTS',flush=True)
    for k,f in files.items():print(f'  {k}: {f.relative_to(project_root())}',flush=True)
    print('PASS: Phase21E complete; no protected outcomes read',flush=True)


def self_test():
    rng=np.random.default_rng(2507)
    # Group-pure synthetic toy: stage1 external labels carry mutation signal.
    X=rng.normal(size=(44,960));pos=np.repeat(np.arange(22),2)
    E=rng.normal(size=(70,960));Epos=np.repeat(np.arange(35),2)
    Y=np.column_stack([0.7*E[:,0]+rng.normal(size=70)*0.1+j for j in range(len(TRAITS))])
    fit=np.arange(32);held=np.arange(32,44)
    f,h,n=crossfit_external(X[fit],X[held],pos[fit],pos[held],E,Y,Epos,42)
    assert f.shape==(32,len(TRAITS)) and h.shape==(12,len(TRAITS))
    assert np.isfinite(f).all() and np.isfinite(h).all() and n<=len(E)
    # Leak test: modify external labels at outer-held protein positions;
    # prediction must be exactly unchanged (those labels are excluded).
    Ybad=Y.copy();Ybad[np.isin(Epos,pos[held])]+=1e7
    f2,h2,n2=crossfit_external(X[fit],X[held],pos[fit],pos[held],E,Ybad,Epos,42)
    assert np.allclose(f,f2,atol=1e-6,rtol=0)
    assert np.allclose(h,h2,atol=1e-6,rtol=0)
    assert n==n2
    print('PASS: Phase21E synthetic crossfit and outer-held-aux-label leakage invariance',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    ex=sub.add_parser('extract')
    ex.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ex.add_argument('--batch-size',type=int,default=4)
    ex.add_argument('--overwrite',action='store_true')
    be=sub.add_parser('benchmark')
    be.add_argument('--smoke',action='store_true')
    sub.add_parser('self-test')
    args=parser.parse_args()
    if args.action=='extract':
        if args.batch_size<1:raise ValueError('batch-size must be positive')
        extract(args)
    elif args.action=='benchmark':benchmark(args)
    else:self_test()


if __name__=='__main__':main()
