#!/usr/bin/env python3
"""Phase 25A: train-only, position-held-out, ETI-assay-conditioned VTI prediction.
Hypothesis-generating: NBD1 region and ETI functional components, selected from prior training analyses.
No VTI validation/test labels are read; ETI assay must already exist at deployment.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

DATA=Path('results/biology/phase24g/phase24g_threshold_robustness.csv')
OUT=Path('results/biology/phase25a')
FEATURES={
    'residual_eti': ['eti_corrector_pp','eti_acute_pp','eti_baseline_fsk'],
    'residual_domain': ['eti_corrector_pp','eti_acute_pp','eti_baseline_fsk','nbd1'],
    'residual_nbd1_gate': ['eti_corrector_pp','eti_acute_pp','eti_baseline_fsk','nbd1','eti_low','nbd1_low'],
}
ALPHAS=(10.,100.,1000.)

def pos_mae(y,p,g):
    return float(pd.DataFrame({'g':g,'e':np.abs(y-p)}).groupby('g').e.mean().mean())

def fit_residual(x,y,eti,alpha):
    model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
    model.fit(x,y-eti)
    return model

def choose_alpha(x,y,eti,groups):
    best=None
    for alpha in ALPHAS:
        pred=np.full(len(y),np.nan)
        for tr,te in GroupKFold(n_splits=3).split(x,y,groups):
            est=fit_residual(x[tr],y[tr],eti[tr],alpha)
            pred[te]=eti[te]+est.predict(x[te])
        err=pos_mae(y,pred,groups)
        if best is None or err < best[0]-1e-10:
            best=(err,alpha)
    return best[1]

def choose_C(x,y,groups):
    best=None
    for C in (.01,.1,1.):
        preds=np.full(len(y),np.nan)
        for tr,te in GroupKFold(n_splits=3).split(x,y,groups):
            if len(np.unique(y[tr]))<2:
                preds[te]=np.mean(y[tr])
            else:
                model=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000))
                model.fit(x[tr],y[tr])
                preds[te]=model.predict_proba(x[te])[:,1]
        err=float(brier_score_loss(y,preds))
        if best is None or err < best[0]-1e-10:
            best=(err,C)
    return best[1]

def screening(truth,score,pos):
    d=pd.DataFrame({'pos':pos,'positive':truth,'score':score})
    gp=d.groupby('pos').agg(positive=('positive','max'),score=('score','max'))
    y=gp.positive.to_numpy(int); p=gp.score.to_numpy(float)
    k=int(np.ceil(.2*len(gp)))
    hits=int(y[np.argsort(-p,kind='stable')[:k]].sum())
    return {
        'positions':len(gp),'positive_positions':int(y.sum()),
        'prevalence':float(y.mean()),
        'AP':float(average_precision_score(y,p)) if 0<y.sum()<len(y) else float('nan'),
        'AUROC':float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else float('nan'),
        'top20_positions':k,'top20_hits':hits,
        'top20_precision':hits/k,
        'top20_recall':hits/y.sum() if y.sum() else float('nan'),
    }

def run(d, folds=5):
    df=d.sort_values(['protein_position','variant_key']).reset_index(drop=True).copy()
    df['nbd1']=df.protein_position.between(390,670).astype(int)
    df['eti_low']=(df.eti_total_pp<10).astype(int)
    df['nbd1_low']=df.nbd1*df.eti_low
    y=df.vti_total_pp.to_numpy(float)
    eti=df.eti_total_pp.to_numpy(float)
    groups=df.protein_position.to_numpy(int)
    preds={n:np.full(len(df),np.nan) for n in ['direct_eti','huber_eti',*FEATURES]}
    probs={n:np.full(len(df),np.nan) for n in ['prevalence','eti_logit','eti_nbd1_logit','nbd1_only_logit']}
    params=[]
    for fold,(tr,te) in enumerate(GroupKFold(n_splits=folds).split(df,y,groups),1):
        assert set(groups[tr]).isdisjoint(groups[te])
        preds['direct_eti'][te]=eti[te]
        huber=make_pipeline(StandardScaler(),HuberRegressor(alpha=1.,epsilon=1.35,max_iter=2000))
        huber.fit(eti[tr,None],y[tr]);preds['huber_eti'][te]=huber.predict(eti[te,None])
        for name,cols in FEATURES.items():
            x=df[cols].to_numpy(float)
            alpha=choose_alpha(x[tr],y[tr],eti[tr],groups[tr])
            model=fit_residual(x[tr],y[tr],eti[tr],alpha)
            preds[name][te]=eti[te]+model.predict(x[te])
            params.append({'fold':fold,'model':name,'alpha':alpha})
        trlow=tr[eti[tr]<10]
        telow=te[eti[te]<10]
        lowlabel=(y[trlow]>=10).astype(int)
        if len(telow):
            probs['prevalence'][telow]=float(lowlabel.mean())
            for name,cols in {
                'eti_logit':['eti_total_pp','eti_corrector_pp','eti_acute_pp','eti_baseline_fsk'],
                'eti_nbd1_logit':['eti_total_pp','eti_corrector_pp','eti_acute_pp','eti_baseline_fsk','nbd1'],
                'nbd1_only_logit':['nbd1'],
            }.items():
                x=df[cols].to_numpy(float)
                if len(np.unique(lowlabel))<2:
                    probs[name][telow]=float(lowlabel.mean())
                    continue
                C=choose_C(x[trlow],lowlabel,groups[trlow])
                clf=make_pipeline(StandardScaler(),LogisticRegression(C=C,max_iter=2000))
                clf.fit(x[trlow],lowlabel)
                probs[name][telow]=clf.predict_proba(x[telow])[:,1]
                params.append({'fold':fold,'model':name,'C':C})
    result=df[['variant_key','protein_position','eti_total_pp','vti_total_pp','nbd1','eti_low']].copy()
    for name,pred in preds.items():
        assert np.isfinite(pred).all(),name
        result['pred_'+name]=pred
    for name,prob in probs.items():
        assert np.isfinite(prob[eti<10]).all(),name
        result['prob_'+name]=prob
    regression=[]
    for name,pred in preds.items():
        regression.append({'model':name,'position_mae_pp':pos_mae(y,pred,groups),
                           'variant_mae_pp':float(np.mean(np.abs(y-pred))),
                           'rmse_pp':float(np.sqrt(np.mean((y-pred)**2))),
                           'under10_mae_pp':float(np.mean(np.abs(y[y<10]-pred[y<10]))),
                           'gt40_mae_pp':float(np.mean(np.abs(y[y>40]-pred[y>40]))),
                           'n_positions':len(np.unique(groups))})
    reg=pd.DataFrame(regression).sort_values('position_mae_pp')
    low=result[result.eti_low==1]
    classification=[]
    for name,prob in probs.items():
        q=prob[eti<10]
        classification.append({'model':name,'brier':float(brier_score_loss((y[eti<10]>=10).astype(int),q)),
                              **screening((low.vti_total_pp>=10).astype(int),q,low.protein_position)})
    cls=pd.DataFrame(classification).sort_values('AP',ascending=False)
    return result,reg,cls,pd.DataFrame(params)

def bootstrap(d,a,b,n=5000):
    # A positive gain is better MAE of model a vs model b.
    t=d.copy();t['gain']=np.abs(t.vti_total_pp-t['pred_'+b])-np.abs(t.vti_total_pp-t['pred_'+a])
    values=t.groupby('protein_position').gain.mean().to_numpy()
    rng=np.random.default_rng(2501)
    sims=np.mean(rng.choice(values,size=(n,len(values)),replace=True),axis=1)
    return {'model':a,'comparator':b,'gain_pp':float(values.mean()),
            'ci95_low':float(np.quantile(sims,.025)),'ci95_high':float(np.quantile(sims,.975)),
            'fraction_positive':float((sims>0).mean())}

def self_test():
    rng=np.random.default_rng(77)
    positions=np.repeat(np.arange(45),2)
    n=len(positions)
    x=pd.DataFrame({'protein_position':positions,'variant_key':[f'test{i}' for i in range(n)]})
    x['eti_corrector_pp']=rng.normal(15,8,n)
    x['eti_acute_pp']=rng.normal(5,4,n)
    x['eti_baseline_fsk']=rng.normal(3,1,n)
    x['eti_total_pp']=x.eti_corrector_pp+x.eti_acute_pp
    x['vti_total_pp']=x.eti_total_pp+5*x.protein_position.between(5,15)+rng.normal(0,4,n)
    results,reg,cls,selection=run(x)
    assert len(results)==n and reg.shape[0]==5 and cls.shape[0]==4
    assert np.isfinite(reg.position_mae_pp).all()
    print('PASS: Synthetic grouped outer + inner CV, regression, classification, and feature processing')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--bootstraps',type=int,default=5000)
    args=parser.parse_args()
    if args.self_test: return self_test()
    d=pd.read_csv(DATA,dtype={'variant_key':str})
    assert len(d)==66 and d.variant_key.is_unique and d.protein_position.nunique()==56
    old=pd.read_parquet('data/processed/bihler_2024_missense_splits.parquet',columns=['variant_key','split'])
    new=pd.read_parquet('data/processed/bihler_2026_transfer_splits.parquet',columns=['variant_key','transfer_split'])
    keys=set(d.variant_key)
    assert keys.issubset(set(old.loc[old.split.eq('train'),'variant_key']))
    assert keys.issubset(set(new.loc[new.transfer_split.eq('train'),'variant_key']))
    for col in ['eti_total_pp','vti_total_pp','eti_corrector_pp','eti_acute_pp','eti_baseline_fsk','protein_position']:
        assert np.isfinite(pd.to_numeric(d[col])).all(),col
    OUT.mkdir(parents=True,exist_ok=True)
    p,r,c,s=run(d)
    print('='*100)
    print('PHASE 25A | MECHANISM-GUIDED ASSAY-CONDITIONED ML | TRAIN ONLY')
    print('='*100)
    print('66 exact matched variants / 56 positions; ETI assay measured, VTI held out by position')
    print('NBD1 definition 390-670 and 10 pp threshold were discovered in prior training analysis: EXPLORATORY')
    print('No protected 2024 or VTI outcomes accessed')
    print('\nA. POSITION-HELD-OUT TOTAL VTI RESCUE REGRESSION')
    print(r.round(4).to_string(index=False))
    print('\nB. ETI-LOW (<10 pp) -> VTI HIGH (>=10 pp) SCREENING')
    print(c.round(4).to_string(index=False))
    print('\nC. PAIRED POSITION BOOTSTRAP')
    rows=[]
    for name in ['huber_eti',*FEATURES]:
        rows.append(bootstrap(p,name,'direct_eti',n=args.bootstraps))
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print('\nD. SELECTED REGULARISATION (INNER-CV ONLY)')
    print(s.to_string(index=False))
    for fname,table in [('phase25a_oof.csv',p),('phase25a_regression.csv',r),
                        ('phase25a_screening.csv',c),('phase25a_selections.csv',s),
                        ('phase25a_bootstrap.csv',pd.DataFrame(rows))]:
        table.to_csv(OUT/fname,index=False)
    print('\nPASS: Phase25A completed; all VTI outer positions held out; protected outcomes unopened')
    print('Saved:',OUT)

if __name__=='__main__':main()
