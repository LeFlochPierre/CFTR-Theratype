#!/usr/bin/env python3
"""CFTR-Theratype Phase 31M: ESMRank, direct ETI/VTI learning, target reformulation.

Run from CFTR-Theratype repository root with this file in scripts/.
Only loads the 210 VTI TRAIN outcomes via Phase26D's filtered loader.
VTI validation/test outcomes are never read. All splits protein-position grouped.
"""
from __future__ import annotations
import argparse
import ast
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, balanced_accuracy_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))
import phase26d_corrector_bottleneck as p26d
import phase26g_feature_model_ablation as p26g
import phase31i2_repository as p31i
import phase31i_auxiliary_transfer as aux

OUT = ROOT / 'results/biology/phase31m'
TARGETS = {'corrector': 'corrector_rescue_pp', 'total': 'vti_rescue_pp'}
MODEL_SELECTION = ROOT / 'results/biology/phase26g'


def get_data():
    d, feat, _, folds, source = p26d.load_train()
    X = np.asarray(feat['sequence_structure'], dtype=float)
    if len(d) != 210 or d.protein_position.nunique() != 168 or X.shape != (210, 80):
        raise AssertionError('Unexpected frozen VTI TRAIN or feature representation')
    if not d.transfer_split.eq('train').all() or np.isinf(X).any():
        raise AssertionError('Protected partitions or invalid features')
    keys = d.variant_key.astype(str).tolist()
    metadata = pd.read_parquet(ROOT/'data/processed/bihler_2026_transfer_splits.parquet',
                               columns=['variant_key','hgvs_protein','protein_position','transfer_split'])
    metadata = metadata[metadata.transfer_split.eq('train')]
    v = d[['variant_key','protein_position']].merge(
        metadata[['variant_key','hgvs_protein','protein_position']], on='variant_key',
        how='left', validate='one_to_one',suffixes=('','_meta'))
    if v.hgvs_protein.isna().any() or not v.protein_position.eq(v.protein_position_meta).all():
        raise AssertionError('VTI metadata/position mismatch')
    v = aux.with_ids(v, 'VTI TRAIN', strict=True)
    if not v.aa31i.str.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]').all():
        raise AssertionError('Non-missense VTI variant: ESMRank and multitask need separate handling')
    if list(v.variant_key.astype(str)) != keys:
        raise AssertionError('Variant order mismatch')
    return d, X, folds, v


def load_eti():
    """ETI targets from frozen 2024 TRAIN only; filter all protected VTI 2026 positions."""
    b = pd.read_parquet(ROOT/'data/processed/bihler_2024_theratyping.parquet',
                        columns=['variant_key','hgvs_protein','corrector_rescue_pp','eti_rescue_pp'])
    splits = pd.read_parquet(ROOT/'data/processed/bihler_2024_missense_splits.parquet',
                             columns=['variant_key','protein_position','split'])
    vmeta = pd.read_parquet(ROOT/'data/processed/bihler_2026_transfer_splits.parquet',
                            columns=['protein_position','transfer_split'])
    protected = set(pd.to_numeric(splits.loc[splits.split.isin(['val','test']),'protein_position'],
                                      errors='coerce').dropna().astype(int))
    protected |= set(pd.to_numeric(vmeta.loc[vmeta.transfer_split.isin(['val','test']),'protein_position'],
                                        errors='coerce').dropna().astype(int))
    b = b.merge(splits,on='variant_key',how='inner',validate='one_to_one')
    b = b[b.split.eq('train') & ~b.protein_position.isin(protected)].copy()
    b = aux.with_ids(b, 'ETI protected-filtered TRAIN')
    b = b[b.aa31i.notna() & b.pos31i2.notna() & ~b.control31i].copy()
    if not b.protein_position.eq(b.pos31i2.astype(int)).all():
        raise AssertionError('ETI position mismatch')
    b = b[b.aa31i.str.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]')].copy()
    for c in ('corrector_rescue_pp','eti_rescue_pp'):
        b[c] = pd.to_numeric(b[c],errors='coerce')
    b = b.dropna(subset=['corrector_rescue_pp','eti_rescue_pp'])
    # Multiple DNA changes can encode one substitution. Do not double-count them.
    b = b.groupby(['aa31i','protein_position'],as_index=False)[
        ['corrector_rescue_pp','eti_rescue_pp']].mean()
    if not len(b) or set(b.protein_position.astype(int)) & protected:
        raise AssertionError('ETI training empty or protected-position leakage')
    return b, protected


def frozen_predictions(d,target):
    z = pd.read_csv(MODEL_SELECTION/f'phase26g_{target}_oof.csv',
                    usecols=['repeat','fold','variant_key','protein_position','observed','xgb_80'])
    z = z.rename(columns={'xgb_80':'prediction'})
    n = len(d)
    for r,g in z.groupby('repeat'):
        if len(g)!=n or set(g.variant_key)!=set(d.variant_key) or g.variant_key.duplicated().any():
            raise AssertionError('Incomplete Phase26G M0 reference')
    lookup=d.set_index('variant_key')[TARGETS[target]]
    if not np.allclose(z.observed.to_numpy(),z.variant_key.map(lookup).to_numpy(float),atol=1e-6):
        raise AssertionError('Historical M0 target mismatch')
    z['target']=target
    z['model']='M0_frozen'
    return z[['repeat','fold','variant_key','protein_position','observed','prediction','target','model']]


def chosen_specs(target):
    z = pd.read_csv(MODEL_SELECTION/f'phase26g_{target}_selected.csv')
    z=z[z.model.eq('xgb_80')]
    spec = {}
    for row in z.itertuples(index=False):
        p = ast.literal_eval(str(row.selected_param))
        if p not in p26g.PARAMS['xgb']:
            raise ValueError('Unexpected historical XGB spec')
        spec[int(row.repeat),int(row.fold)] = p
    if len(spec)!=10: raise AssertionError('Expected 10 frozen XGB specifications')
    return spec


def add(rows,base,target,model,rep,fold,idx,pred):
    for j,p in zip(idx,np.asarray(pred,float)):
        rows.append(dict(target=target,model=model,repeat=int(rep),fold=int(fold),
                         variant_key=str(base.variant_key.iloc[j]),
                         protein_position=int(base.protein_position.iloc[j]),
                         observed=float(base[TARGETS[target]].iloc[j]),prediction=float(p)))


def report(rows,mode,quick):
    oof=pd.concat([frozen_predictions(rows['base'],t) for t in TARGETS],ignore_index=True)
    oof=pd.concat([oof,pd.DataFrame(rows['new'])],ignore_index=True)
    out=OUT/f'phase31m_{mode}_{"quick" if quick else "oof"}.csv'
    oof.to_csv(out,index=False)
    if quick:
        print('PASS one-fold smoke test; no pooled performance interpretation. Saved:',out)
        return
    metrics=[]
    for (target,mod,rep),z in oof.groupby(['target','model','repeat']):
        if len(z)!=len(rows['base']) or z.variant_key.duplicated().any():
            raise AssertionError(f'Incomplete OOF: {target} {mod} repeat={rep}')
        err=np.abs(z.observed.to_numpy()-z.prediction.to_numpy())
        mae=float(pd.DataFrame({'pos':z.protein_position,'ae':err}).groupby('pos').ae.mean().mean())
        metrics.append(dict(target=target,model=mod,repeat=rep,position_mae_pp=mae,
                            spearman=float(spearmanr(z.observed,z.prediction).statistic)))
    metrics=pd.DataFrame(metrics)
    summary=metrics.groupby(['target','model'],as_index=False)[['position_mae_pp','spearman']].mean()
    summary.to_csv(OUT/f'phase31m_{mode}_summary.csv',index=False)
    print('\nMEAN ACROSS FROZEN REPEATS (lower MAE is better):')
    print(summary.sort_values(['target','position_mae_pp']).round(4).to_string(index=False))
    gains=[]
    for target,db in oof.groupby('target'):
        ref=db[db.model.eq('M0_frozen')][['repeat','variant_key','protein_position','observed','prediction']].rename(columns={'prediction':'baseline'})
        for name in sorted(set(db.model)-{'M0_frozen'}):
            alt=db[db.model.eq(name)][['repeat','variant_key','prediction']].rename(columns={'prediction':'candidate'})
            joined=ref.merge(alt,on=['repeat','variant_key'],validate='one_to_one')
            if len(joined)!=len(ref): raise AssertionError('Bootstrap OOF pairing failure')
            joined['gain']=abs(joined.observed-joined.baseline)-abs(joined.observed-joined.candidate)
            vals=joined.groupby(['repeat','protein_position']).gain.mean().groupby('protein_position').mean().to_numpy()
            if len(vals)!=168: raise AssertionError('Expected 168 independent positions')
            rng=np.random.default_rng(31113)
            boot=np.array([vals[rng.integers(len(vals),size=len(vals))].mean() for _ in range(2000)])
            gains.append(dict(target=target,model=name,gain_vs_M0_pp=vals.mean(),
                              ci95_low=np.quantile(boot,.025),ci95_high=np.quantile(boot,.975)))
    gains=pd.DataFrame(gains)
    gains.to_csv(OUT/f'phase31m_{mode}_paired_bootstrap.csv',index=False)
    print('\nPAIRED POSITION BOOTSTRAP (positive = better than M0):')
    print(gains.round(4).to_string(index=False))
    print('PASS TRAIN-only; no VTI VAL/TEST outcomes accessed. Saved:',out)


def read_wt(path):
    s=''.join(x.strip() for x in Path(path).read_text().splitlines() if not x.startswith('>')).upper()
    if len(s)!=1480 or set(s)-set('ACDEFGHIKLMNPQRSTVWY'):
        raise ValueError('Canonical CFTR (UniProt P13569) FASTA expected, 1480 amino acids')
    return s


def prepare_esm(args):
    # Metadata-only scoring preparation: VTI response values are never loaded.
    meta=pd.read_parquet(ROOT/'data/processed/bihler_2026_transfer_splits.parquet',
                         columns=['variant_key','hgvs_protein','protein_position','transfer_split'])
    meta=meta[meta.transfer_split.eq('train')]
    if len(meta)!=210 or meta.variant_key.duplicated().any() or meta.protein_position.nunique()!=168:
        raise AssertionError('Frozen 2026 VTI TRAIN metadata changed')
    v=aux.with_ids(meta[['variant_key','hgvs_protein','protein_position']], 'VTI TRAIN', strict=True)
    codes=sorted(v.aa31i.unique())
    if not all(re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]\d+[ACDEFGHIKLMNPQRSTVWY]', a) for a in codes):
        raise ValueError('Noncanonical missense variant in VTI TRAIN metadata')
    wt=read_wt(args.fasta)
    r=[]
    for a in codes:
        match=re.fullmatch(r'([A-Z])(\d+)([A-Z])',a)
        w,pos,mut=match.group(1),int(match.group(2)),match.group(3)
        if wt[pos-1]!=w:
            raise ValueError(f'WT reference mismatch: {a} vs {wt[pos-1]} at {pos}')
        r.append({'hgvsp':'p.'+a,'seq_wt':wt,'seq_mu':wt[:pos-1]+mut+wt[pos:]})
    out=OUT/'phase31m_esmrank_input.csv'
    pd.DataFrame(r).to_csv(out,index=False)
    pd.DataFrame({'aa_code':codes}).to_csv(OUT/'phase31m_esmrank_expected_aa.csv',index=False)
    print(f'PASS: {len(r)} unique canonical missense substitutions; input saved: {out}')
    print('Score externally with: esmrank run --input',out,'-csv --outdir',OUT/'esmrank_output','--batch-size 8')
    print('Scores are genotype-only; no VTI TEST/VAL labels used.')


def load_esm(path):
    z=pd.read_csv(path)
    if not {'ids','prediction'}.issubset(z): raise ValueError('Expected ESMRank predictions columns: ids,prediction')
    z=z[['ids','prediction']].copy()
    z['aa31i']=z.ids.astype(str).map(aux.norm_aa)
    if z.aa31i.isna().any() or z.aa31i.duplicated().any():
        raise ValueError('Cannot map unique ESMRank ids to substitutions; inspect output')
    z.prediction=pd.to_numeric(z.prediction,errors='coerce')
    if not np.isfinite(z.prediction.to_numpy(float)).all():raise ValueError('Nonfinite ESMRank score')
    return z.set_index('aa31i').prediction


def esm_eval(args):
    d,X,folds,v=get_data()
    score=load_esm(args.scores)
    if not set(v.aa31i).issubset(set(score.index)):
        miss=sorted(set(v.aa31i)-set(score.index))
        raise ValueError(f'ESMRank missing {len(miss)} VTI TRAIN substitutions: {miss[:5]}')
    xscore=v.aa31i.map(score).to_numpy(float)
    print('ESMRank VTI TRAIN score coverage:',len(xscore),'/',len(d))
    for t,col in TARGETS.items():
        y=d[col].to_numpy(float)
        print('Zero-shot',t,'variant rho=',round(float(spearmanr(xscore,y).statistic),4))
        pos=pd.DataFrame({'pos':d.protein_position,'s':xscore,'y':y}).groupby('pos').mean()
        print('Zero-shot',t,'position rho=',round(float(spearmanr(pos.s,pos.y).statistic),4))
    specs={t:chosen_specs(t) for t in TARGETS}
    keys=d.variant_key.astype(str).to_numpy()
    pos=d.protein_position.to_numpy(int)
    rows=[]
    for fi,(rep,fold,trkeys,tekeys) in enumerate(folds[:1] if args.quick else folds,1):
        it=np.flatnonzero(np.isin(keys,list(trkeys))); iv=np.flatnonzero(np.isin(keys,list(tekeys)))
        assert set(pos[it]).isdisjoint(pos[iv])
        for t,col in TARGETS.items():
            y=d[col].to_numpy(float)
            scalar=make_pipeline(StandardScaler(),Ridge(alpha=10.0))
            scalar.fit(xscore[it,None],y[it])
            add(rows,d,t,'esmrank_only_ridge',rep,fold,iv,scalar.predict(xscore[iv,None]))
            matrix=np.column_stack((X,xscore))
            est=p26g.make_model('xgb',specs[t][int(rep),int(fold)])
            est.fit(matrix[it],y[it])
            add(rows,d,t,'esmrank_plus_xgb80',rep,fold,iv,est.predict(matrix[iv]))
        print(f'ESM outer {fi}/{1 if args.quick else len(folds)} complete',flush=True)
    report({'base':d,'new':rows},'esm',args.quick)


def design(x,eti_flag):
    """Shared effect + regularised ETI-specific deviation, with VTI as reference head."""
    flag=np.full((len(x),1),float(eti_flag))
    return np.column_stack([x,x*flag,flag])


def transfer(args):
    d,X,folds,v=get_data()
    eti,protected=load_eti()
    print(f'ETI eligible unique substitutions={len(eti)}, positions={eti.protein_position.nunique()}')
    etix=p31i.mutation_features(eti.aa31i)
    vtix=p31i.mutation_features(v.aa31i)
    pos=d.protein_position.to_numpy(int); keys=d.variant_key.astype(str).to_numpy()
    epi=eti.protein_position.to_numpy(int)
    rows=[]
    for fi,(rep,fold,trkeys,tekeys) in enumerate(folds[:1] if args.quick else folds,1):
        it=np.flatnonzero(np.isin(keys,list(trkeys)));iv=np.flatnonzero(np.isin(keys,list(tekeys)))
        held=set(pos[iv]); keep=~np.isin(epi,list(held))
        if held & set(epi[keep]) or held & set(pos[it]) or held & protected:
            raise AssertionError('Protected or held-out ETI/VTI protein positions leaked')
        if not keep.any(): raise ValueError('No ETI train positions available')
        print(f'Outer {fi}: VTI train positions={len(set(pos[it]))}, ETI available={len(set(epi[keep]))}, ETI held removed={(~keep).sum()}',flush=True)
        # Standardisation uses only outer-training genotype features; no held-out variants.
        scale=StandardScaler().fit(np.concatenate([vtix[it],etix[keep]],axis=0))
        vt=scale.transform(vtix);e=scale.transform(etix[keep])
        for t,col in TARGETS.items():
            y=d[col].to_numpy(float)
            ey=eti['corrector_rescue_pp' if t=='corrector' else 'eti_rescue_pp'].to_numpy(float)[keep]
            single=Ridge(alpha=100.0).fit(vt[it],y[it])
            add(rows,d,t,'vti_only_ridge61',rep,fold,iv,single.predict(vt[iv]))
            for wt in (0.5,2.0):
                xx=np.vstack([design(vt[it],False),design(e,True)])
                yy=np.concatenate([y[it],ey])
                # Equal total VTI weight, ETI block contributes wt of that amount.
                sw=np.r_[np.ones(len(it)),np.full(len(ey),wt*len(it)/len(ey))]
                joint=Ridge(alpha=100.0).fit(xx,yy,sample_weight=sw)
                add(rows,d,t,f'joint_ETI_weight_{wt:g}',rep,fold,iv,
                    joint.predict(design(vt[iv],False)))
        print(f'Transfer outer {fi} complete',flush=True)
    report({'base':d,'new':rows},'transfer',args.quick)


def targets(args):
    d,X,folds,v=get_data()
    keys=d.variant_key.astype(str).to_numpy();pos=d.protein_position.to_numpy(int)
    specs={t:chosen_specs(t) for t in TARGETS}
    rows=[]; clas=[]
    for fi,(rep,fold,trkeys,tekeys) in enumerate(folds[:1] if args.quick else folds,1):
        it=np.flatnonzero(np.isin(keys,list(trkeys)));iv=np.flatnonzero(np.isin(keys,list(tekeys)))
        assert not set(pos[it]) & set(pos[iv])
        for t,col in TARGETS.items():
            y=d[col].to_numpy(float)
            base_col='baseline_fsk_pct_wt'
            treated_col='vnz_tez_fsk_pct_wt' if t=='corrector' else 'vti_treated_pct_wt'
            if not np.allclose(d[treated_col]-d[base_col],y,atol=1e-6):
                raise AssertionError('Incorrect target reconstruction')
            spec=specs[t][int(rep),int(fold)]
            preds={}
            for field in [base_col,treated_col]:
                model=p26g.make_model('xgb',spec)
                model.fit(X[it],d[field].to_numpy(float)[it])
                preds[field]=model.predict(X[iv])
            pred=preds[treated_col]-preds[base_col]
            add(rows,d,t,'pred_treated_minus_pred_baseline',rep,fold,iv,pred)
            # Classification: threshold from the published in-vitro response criterion.
            label=(y>=10).astype(int)
            prevalence=float(label[it].mean())
            if 0<label[it].sum()<len(it):
                logistic=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                                      StandardScaler(),LogisticRegression(C=0.1,max_iter=3000))
                logistic.fit(X[it],label[it]);prob=logistic.predict_proba(X[iv])[:,1]
            else:
                prob=np.full(len(iv),prevalence)
            for j,p in zip(iv,prob):
                clas.append(dict(target=t,repeat=int(rep),fold=int(fold),variant_key=keys[j],
                                 protein_position=pos[j],observed=int(label[j]),probability=float(p),
                                 train_prevalence=prevalence,model='logistic80'))
        print(f'Target reformulation outer {fi} complete',flush=True)
    report({'base':d,'new':rows},'targets',args.quick)
    q=pd.DataFrame(clas)
    q.to_csv(OUT/f'phase31m_classification_{"quick" if args.quick else "oof"}.csv',index=False)
    if not args.quick:
        rows=[]
        for (t,rep),z in q.groupby(['target','repeat']):
            y=z.observed.to_numpy(int);p=z.probability.to_numpy(float)
            rows.append(dict(target=t,repeat=rep,prevalence=float(y.mean()),
                             auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
                             average_precision=float(average_precision_score(y,p)),
                             brier=float(brier_score_loss(y,p)),
                             baseline_brier=float(brier_score_loss(y,z.train_prevalence)),
                             balanced_accuracy=float(balanced_accuracy_score(y,p>=0.5)),
                             majority_accuracy=float(max(y.mean(),1-y.mean()))))
        result=pd.DataFrame(rows)
        result.to_csv(OUT/'phase31m_classification_metrics.csv',index=False)
        print('\nVTI in-vitro responder classification (threshold >=10 pp; exploratory):')
        print(result.round(4).to_string(index=False))


def main():
    a=argparse.ArgumentParser()
    a.add_argument('command',choices=['prepare-esm','eval-esm','transfer','targets'])
    a.add_argument('--fasta',default='data/reference/P13569.fasta')
    a.add_argument('--scores',default='results/biology/phase31m/esmrank_output/phase31m_esmrank_input/predictions.csv')
    a.add_argument('--quick',action='store_true')
    args=a.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    print('PHASE 31M | TRAIN only | external VTI VAL/TEST outcomes protected')
    if args.command=='prepare-esm':prepare_esm(args)
    elif args.command=='eval-esm':esm_eval(args)
    elif args.command=='transfer':transfer(args)
    else:targets(args)

if __name__=='__main__':main()
