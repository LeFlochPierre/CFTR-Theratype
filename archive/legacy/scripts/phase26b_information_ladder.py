#!/usr/bin/env python3
"""Phase 26B: train-only information-ladder VTI-response benchmark.

Predicts absolute VTI-treated in-vitro CFTR function, from information available
at successive assay stages. No VTI validation/test or protected ETI rows read.
These are *assay-conditioned* models, not mutation-only zero-shot models.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path('results/biology/phase26b')
Y = 'vti_treated_pct_wt'
STAGES = {
    'baseline_only': (['baseline_fsk_pct_wt'], 'baseline_fsk_pct_wt'),
    'iva_assay': (['baseline_fsk_pct_wt', 'iva_treated_pct_wt'], 'iva_treated_pct_wt'),
    'corrector_assay': (['baseline_fsk_pct_wt', 'iva_treated_pct_wt', 'vnz_tez_fsk_pct_wt'], 'vnz_tez_fsk_pct_wt'),
}
ALPHAS = (0.1, 1., 10., 100., 1000.)

def pos_mae(y, p, g):
    t = pd.DataFrame({'g':g, 'e':np.abs(y-p)})
    return float(t.groupby('g').e.mean().mean())

def fit_regressor(X, y, anchor, method='ridge', alpha=10.):
    if method == 'ridge':
        reg = Ridge(alpha=alpha)
    elif method == 'huber':
        reg = HuberRegressor(alpha=1.0, epsilon=1.35, max_iter=2000)
    else:
        raise ValueError(method)
    fit = make_pipeline(StandardScaler(), reg)
    fit.fit(X, y - anchor)
    return fit

def choose_alpha(X, y, a, groups):
    cv = GroupKFold(n_splits=min(3,len(np.unique(groups))))
    scores=[]
    for alpha in ALPHAS:
        pred=np.full(len(y),np.nan)
        for it,iv in cv.split(X,y,groups):
            fit=fit_regressor(X[it],y[it],a[it],alpha=alpha)
            pred[iv]=a[iv]+fit.predict(X[iv])
        scores.append(pos_mae(y,pred,groups))
    return ALPHAS[int(np.argmin(scores))]

def evaluate(df):
    df=df.sort_values(['protein_position','variant_key']).reset_index(drop=True).copy()
    g=df.protein_position.to_numpy(dtype=int)
    y=df[Y].to_numpy(float)
    n=len(df)
    models=['train_median','direct_baseline','direct_iva','direct_corrector']
    for s in STAGES:
        models += [s+'_ridge',s+'_huber']
    pred={m:np.full(n,np.nan) for m in models}
    selections=[]
    folds=GroupKFold(n_splits=5)
    for fold,(it,iv) in enumerate(folds.split(df,y,g),1):
        assert set(g[it]).isdisjoint(set(g[iv]))
        pred['train_median'][iv]=np.median(y[it])
        for s,(cols,anchor) in STAGES.items():
            pred['direct_'+{'baseline_only':'baseline','iva_assay':'iva','corrector_assay':'corrector'}[s]][iv]=df.iloc[iv][anchor].to_numpy(float)
            X=df[cols].to_numpy(float)
            a=df[anchor].to_numpy(float)
            alpha=choose_alpha(X[it],y[it],a[it],g[it])
            fit=fit_regressor(X[it],y[it],a[it],alpha=alpha)
            pred[s+'_ridge'][iv]=a[iv]+fit.predict(X[iv])
            fit=fit_regressor(X[it],y[it],a[it],method='huber')
            pred[s+'_huber'][iv]=a[iv]+fit.predict(X[iv])
            selections.append({'fold':fold,'stage':s,'ridge_alpha':alpha,
                               'n_train_positions':len(np.unique(g[it])),
                               'n_test_positions':len(np.unique(g[iv]))})
    output=df[['variant_key','protein_position','study_cohort',Y,
               'baseline_fsk_pct_wt','iva_treated_pct_wt','vnz_tez_fsk_pct_wt']].copy()
    metric=[]
    for m in models:
        p=pred[m]
        assert np.isfinite(p).all(),m
        output['pred_'+m]=p
        metric.append({'model':m,'position_mae_pp':pos_mae(y,p,g),
                       'variant_mae_pp':float(np.abs(y-p).mean()),
                       'rmse_pp':float(np.sqrt(np.mean((y-p)**2))),
                       'n_positions':len(np.unique(g))})
    met=pd.DataFrame(metric).sort_values('position_mae_pp')
    bycohort=[]
    for cohort,t in output.groupby('study_cohort',dropna=False):
        for m in models:
            bycohort.append({'cohort':str(cohort),'model':m,'n_variants':len(t),
                'n_positions':t.protein_position.nunique(),
                'variant_mae_pp':float(np.mean(np.abs(t[Y]-t['pred_'+m])))})
    return output,met,pd.DataFrame(bycohort),pd.DataFrame(selections)

def read_train():
    s=pd.read_parquet('data/processed/bihler_2026_transfer_splits.parquet',
                      columns=['variant_key','transfer_split','protein_position'])
    s=s.loc[s.transfer_split.eq('train')].copy()
    assert s.variant_key.is_unique
    # Filter by allowed keys *at parquet scan time*: never materialise protected outcomes.
    allowed_keys=s.variant_key.tolist()
    d=pd.read_parquet('data/processed/bihler_2026_vti.parquet',
        filters=[('variant_key','in',allowed_keys)],
        columns=['variant_key','study_cohort',Y,'vti_rescue_pp',
                 'vti_increment_over_iva_pp','baseline_fsk_pct_wt',
                 'iva_treated_pct_wt','vnz_tez_fsk_pct_wt'])
    assert set(d.variant_key)==set(allowed_keys), 'Filtered VTI keys do not match train keys'
    d=d.merge(s,on='variant_key',how='inner',validate='one_to_one')
    assert len(d)==210 and d.protein_position.nunique()==168, (len(d),d.protein_position.nunique())
    cols=[Y,'vti_rescue_pp','vti_increment_over_iva_pp',
          'baseline_fsk_pct_wt','iva_treated_pct_wt','vnz_tez_fsk_pct_wt','protein_position']
    assert d[cols].apply(pd.to_numeric,errors='coerce').notna().all().all(), 'Missing/non-numeric data'
    a=np.max(np.abs(d.vti_rescue_pp-(d[Y]-d.baseline_fsk_pct_wt)))
    b=np.max(np.abs(d.vti_increment_over_iva_pp-(d[Y]-d.iva_treated_pct_wt)))
    print(f'Identity checks: max rescue discrepancy {a:.3g} pp; max increment discrepancy {b:.3g} pp')
    assert a<1e-5 and b<1e-5, 'Inspect response target definitions before modelling'
    return d

def synthetic():
    rng=np.random.default_rng(26)
    n_pos=60; repeats=2
    g=np.repeat(np.arange(n_pos),repeats)
    n=len(g)
    d=pd.DataFrame({'variant_key':[f'syn{i}' for i in range(n)],
        'protein_position':g,'study_cohort':['synthetic']*n})
    d['baseline_fsk_pct_wt']=rng.uniform(1,15,n)
    d['iva_treated_pct_wt']=d.baseline_fsk_pct_wt+rng.uniform(1,12,n)
    d['vnz_tez_fsk_pct_wt']=d.baseline_fsk_pct_wt+rng.uniform(6,20,n)
    d[Y]=d.vnz_tez_fsk_pct_wt+0.65*(d.iva_treated_pct_wt-d.baseline_fsk_pct_wt)+rng.normal(0,2,n)
    p,m,c,s=evaluate(d)
    assert len(p)==n and len(m)==10 and np.isfinite(m.position_mae_pp).all()
    assert m.position_mae_pp.min()<m.loc[m.model=='train_median','position_mae_pp'].iloc[0]
    print('PASS synthetic: grouped nested CV, all 10 baselines, cohort summaries, no missing OOF predictions')
    print(m.round(3).to_string(index=False))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        synthetic();return
    d=read_train()
    p,m,c,s=evaluate(d)
    OUT.mkdir(parents=True,exist_ok=True)
    p.to_csv(OUT/'phase26b_oof.csv',index=False)
    m.to_csv(OUT/'phase26b_metrics.csv',index=False)
    c.to_csv(OUT/'phase26b_by_cohort.csv',index=False)
    s.to_csv(OUT/'phase26b_selections.csv',index=False)
    print('='*92)
    print('PHASE 26B | INFORMATION LADDER FOR CONTINUOUS VTI-TREATED CFTR FUNCTION | TRAIN ONLY')
    print('='*92)
    print(f'{len(d)} variants / {d.protein_position.nunique()} protein positions; grouped nested CV')
    print('Goal: predict absolute VTI-treated FSK+IVA %WT; lower error is better.')
    print('Inputs progressively require untreated, IVA-only or VTI-corrector assays.')
    print('IMPORTANT: assay-informed results cannot be compared directly with mutation-only zero-shot results.')
    print('No VTI val/test outcomes or protected 2024 outcomes read; cohort used ONLY for diagnostics.')
    print('\nA. PREDICTIVE PERFORMANCE')
    print(m.round(4).to_string(index=False))
    print('\nB. STUDY COHORT VARIANT COUNTS')
    print(d.groupby('study_cohort',dropna=False).agg(variants=('variant_key','size'),positions=('protein_position','nunique')).to_string())
    print('\nC. COHORT-SPECIFIC MAE FOR KEY COMPARATORS')
    c=c[c.model.isin(['train_median','baseline_only_ridge','iva_assay_ridge','corrector_assay_ridge'])]
    print(c.round(4).to_string(index=False))
    print('\nD. SELECTED ALPHAS (INNER CV ONLY)')
    print(s.to_string(index=False))
    print('\nPASS Phase26B. Saved:',OUT)

if __name__=='__main__':main()
