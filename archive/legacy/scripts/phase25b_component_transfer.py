#!/usr/bin/env python3
"""Phase 25B: mechanistic, component-wise ETI-assay -> VTI prediction.

TRAIN ONLY, protein-position-disjoint nested CV. Primary target is continuous
VTI total in-vitro functional rescue. This is not clinical treatment efficacy.
Only ETI functional measurements and protein position are supplied as features.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCE = Path('results/biology/phase24g/phase24g_threshold_robustness.csv')
COMP_SOURCE = Path('results/biology/phase24c/phase24c_mechanistic_decomposition.csv')
OUT = Path('results/biology/phase25b')
ALPHAS = (10., 100., 1000.)
CORR_FEATURES = ('eti_corrector_pp', 'eti_baseline_fsk')
IVA_FEATURES = ('eti_acute_pp', 'eti_corrector_pp')
TOTAL_FEATURES = ('eti_corrector_pp', 'eti_acute_pp', 'eti_baseline_fsk')


def pos_mae(y, p, g):
    df = pd.DataFrame({'pos':g, 'ae':np.abs(y-p)})
    return float(df.groupby('pos').ae.mean().mean())


def add_vti_components(d):
    """Use directly present train-only component labels, or join Phase24C by key."""
    aliases = {
        'vti_acute_pp': ('vti_acute_pp', 'vti_iva_increment_pp',
                         'vti_acute_iva_pp', 'vti_iva_pp',
                         'vti_acute_increment_pp', 'vti_iva_increment',
                         'vti_acute_iva_increment_pp'),
        'vti_corrector_pp': ('vti_corrector_pp', 'vti_corrector_gain_pp',
                             'vti_correction_pp', 'vti_corrector_gain'),
    }
    def choose(table, choices):
        bylower = {str(c).lower(): c for c in table.columns}
        for c in choices:
            if c.lower() in bylower:
                return bylower[c.lower()]
        return None

    d = d.copy()
    for component, options in aliases.items():
        found = choose(d, options)
        if found is not None:
            d[component] = pd.to_numeric(d[found], errors='coerce')

    if ('vti_acute_pp' not in d.columns and
        'vti_corrector_pp' not in d.columns and COMP_SOURCE.exists()):
        extra = pd.read_csv(COMP_SOURCE, dtype={'variant_key':str})
        if 'variant_key' not in extra or extra.variant_key.duplicated().any():
            raise ValueError('Phase24C must have unique variant_key rows')
        found = {name:choose(extra, options) for name,options in aliases.items()}
        available = {name:src for name,src in found.items() if src is not None}
        if available:
            right = extra[['variant_key',*dict.fromkeys(available.values())]].copy()
            right = right.rename(columns={src:name for name,src in available.items()})
            d = d.merge(right, on='variant_key', how='left', validate='one_to_one')
            print('Phase24C backfill:', available)

    if 'vti_acute_pp' not in d and 'vti_corrector_pp' in d:
        d['vti_acute_pp'] = d.vti_total_pp - d.vti_corrector_pp
    if 'vti_corrector_pp' not in d and 'vti_acute_pp' in d:
        d['vti_corrector_pp'] = d.vti_total_pp - d.vti_acute_pp
    if not {'vti_acute_pp','vti_corrector_pp'}.issubset(d.columns):
        print('Phase24G candidate columns:',
              [c for c in d.columns if any(s in c.lower() for s in ('vti','acute','iva','correct'))])
        if COMP_SOURCE.exists():
            extra = pd.read_csv(COMP_SOURCE,nrows=0)
            print('Phase24C candidate columns:',
                  [c for c in extra.columns if any(s in c.lower() for s in ('vti','acute','iva','correct'))])
        raise ValueError('Missing VTI acute/corrector component label. Inspect the printed column names and add an alias.')
    return d


def prepare_training_data():
    d = pd.read_csv(SOURCE, dtype={'variant_key':str})
    assert len(d)==66 and d.variant_key.is_unique and d.protein_position.nunique()==56
    old = pd.read_parquet('data/processed/bihler_2024_missense_splits.parquet', columns=['variant_key','split'])
    new = pd.read_parquet('data/processed/bihler_2026_transfer_splits.parquet', columns=['variant_key','transfer_split'])
    assert set(d.variant_key).issubset(set(old.loc[old.split.eq('train'),'variant_key']))
    assert set(d.variant_key).issubset(set(new.loc[new.transfer_split.eq('train'),'variant_key']))
    d = add_vti_components(d)
    numeric = ['eti_total_pp','vti_total_pp','eti_corrector_pp','eti_acute_pp',
               'eti_baseline_fsk','vti_corrector_pp','vti_acute_pp','protein_position']
    for c in numeric:
        assert pd.to_numeric(d[c],errors='coerce').notna().all(), c
        assert np.isfinite(d[c].to_numpy(float)).all(), c
    discrepancy = np.abs(d.vti_total_pp-d.vti_corrector_pp-d.vti_acute_pp)
    eti_discrepancy = np.abs(d.eti_total_pp-d.eti_corrector_pp-d.eti_acute_pp)
    print('Max |VTI total - (corrector + acute)|:',float(discrepancy.max()))
    print('Max |ETI total - (corrector + acute)|:',float(eti_discrepancy.max()))
    if discrepancy.max() > .25 or eti_discrepancy.max() > .25:
        raise ValueError('Decomposition fails identity QC (>0.25 pp): inspect source definitions')
    d['nbd1'] = d.protein_position.between(390,670).astype(float)
    return d.sort_values(['protein_position','variant_key']).reset_index(drop=True)


def xmat(df, cols):
    return df[list(cols)].to_numpy(float)


def ridge_pipeline(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def huber_pipeline():
    return make_pipeline(StandardScaler(), HuberRegressor(alpha=1.,epsilon=1.35,max_iter=2000))


def fit_predict(df, tr, te, method, alpha=None):
    a,b = df.iloc[tr],df.iloc[te]
    if method=='direct_eti':
        return b.eti_total_pp.to_numpy(float), None, None
    if method=='huber_total':
        m=huber_pipeline()
        m.fit(xmat(a,['eti_total_pp']),a.vti_total_pp.to_numpy(float))
        return m.predict(xmat(b,['eti_total_pp'])),None,None
    if method=='ridge_total':
        m=ridge_pipeline(alpha)
        m.fit(xmat(a,TOTAL_FEATURES), a.vti_total_pp.to_numpy(float)-a.eti_total_pp.to_numpy(float))
        return b.eti_total_pp.to_numpy(float)+m.predict(xmat(b,TOTAL_FEATURES)),None,None

    if method.startswith('component_'):
        if method=='component_huber':
            mc,mi=huber_pipeline(),huber_pipeline()
            yc = a.vti_corrector_pp.to_numpy(float)
            yi = a.vti_acute_pp.to_numpy(float)
        else:
            mc,mi=ridge_pipeline(alpha),ridge_pipeline(alpha)
            yc=a.vti_corrector_pp.to_numpy(float)-a.eti_corrector_pp.to_numpy(float)
            yi=a.vti_acute_pp.to_numpy(float)-a.eti_acute_pp.to_numpy(float)
        xc = CORR_FEATURES + (('nbd1',) if method=='component_ridge_nbd1' else ())
        xi = IVA_FEATURES
        mc.fit(xmat(a,xc),yc)
        mi.fit(xmat(a,xi),yi)
        pc=mc.predict(xmat(b,xc))
        pi=mi.predict(xmat(b,xi))
        if method != 'component_huber':
            pc=pc+b.eti_corrector_pp.to_numpy(float)
            pi=pi+b.eti_acute_pp.to_numpy(float)
        return pc+pi,pc,pi
    raise ValueError(method)


def choose_alpha(df, tr, method):
    sub = df.iloc[tr].reset_index(drop=True)
    g=sub.protein_position.to_numpy()
    y=sub.vti_total_pp.to_numpy(float)
    inner=GroupKFold(n_splits=3)
    choices=[]
    for alpha in ALPHAS:
        p=np.full(len(sub),np.nan)
        for ti,vi in inner.split(sub,y,g):
            p[vi]=fit_predict(sub,ti,vi,method,alpha)[0]
        choices.append((pos_mae(y,p,g),alpha))
    return min(choices)[1]


def evaluate(d, folds=5):
    methods=('direct_eti','huber_total','ridge_total','component_huber',
             'component_ridge','component_ridge_nbd1')
    y=d.vti_total_pp.to_numpy(float)
    g=d.protein_position.to_numpy(int)
    predictions={name:np.full(len(d),np.nan) for name in methods}
    comp={name:{'c':np.full(len(d),np.nan),'i':np.full(len(d),np.nan)}
          for name in methods if name.startswith('component_')}
    parameters=[]
    for fold,(tr,te) in enumerate(GroupKFold(n_splits=folds).split(d,y,g),1):
        assert set(g[tr]).isdisjoint(g[te])
        for method in methods:
            alpha=choose_alpha(d,tr,method) if method in (
                'ridge_total','component_ridge','component_ridge_nbd1') else None
            p,pc,pi=fit_predict(d,tr,te,method,alpha)
            predictions[method][te]=p
            if pc is not None:
                comp[method]['c'][te]=pc
                comp[method]['i'][te]=pi
            parameters.append({'fold':fold,'model':method,'alpha':alpha,'n_train_positions':len(set(g[tr])),'n_test_positions':len(set(g[te]))})
    output=d[['variant_key','protein_position','eti_total_pp','vti_total_pp',
              'eti_corrector_pp','eti_acute_pp','vti_corrector_pp','vti_acute_pp','nbd1']].copy()
    metrics=[]
    for method,p in predictions.items():
        assert np.isfinite(p).all(),method
        output['pred_'+method]=p
        low=y<10
        high=y>40
        metrics.append({'model':method,'position_mae_pp':pos_mae(y,p,g),
                        'variant_mae_pp':float(np.abs(y-p).mean()),
                        'rmse_pp':float(np.sqrt(np.mean((y-p)**2))),
                        'under10_mae_pp':float(np.abs(y[low]-p[low]).mean()) if low.sum() else np.nan,
                        'gt40_mae_pp':float(np.abs(y[high]-p[high]).mean()) if high.sum() else np.nan,
                        'n_under10_variants':int(low.sum()),'n_gt40_variants':int(high.sum())})
    component_metrics=[]
    for key,truth,col in (('c',d.vti_corrector_pp.to_numpy(float),'eti_corrector_pp'),
                          ('i',d.vti_acute_pp.to_numpy(float),'eti_acute_pp')):
        p=d[col].to_numpy(float)
        component_metrics.append({'model':'direct_eti_components',
                                  'target':'corrector' if key=='c' else 'acute_iva',
                                  'position_mae_pp':pos_mae(truth,p,g),
                                  'variant_mae_pp':float(np.abs(truth-p).mean())})
    for method,heads in comp.items():
        for key,truth in (('c',d.vti_corrector_pp.to_numpy(float)),
                          ('i',d.vti_acute_pp.to_numpy(float))):
            p=heads[key]
            assert np.isfinite(p).all(),(method,key)
            output[f'pred_{method}_{key}']=p
            component_metrics.append({'model':method,'target':'corrector' if key=='c' else 'acute_iva',
                                     'position_mae_pp':pos_mae(truth,p,g),
                                     'variant_mae_pp':float(np.abs(truth-p).mean())})
    return (output,pd.DataFrame(metrics).sort_values('position_mae_pp'),
            pd.DataFrame(component_metrics),pd.DataFrame(parameters))


def bootstraps(pred, models, n=10000):
    rng=np.random.default_rng(2512)
    out=[]
    for model in models:
        if model=='direct_eti':continue
        for baseline in ('direct_eti','huber_total'):
            if model==baseline:continue
            gain=np.abs(pred.vti_total_pp-pred['pred_'+baseline])-np.abs(pred.vti_total_pp-pred['pred_'+model])
            v=pd.DataFrame({'g':pred.protein_position,'gain':gain}).groupby('g').gain.mean().to_numpy()
            sim=rng.choice(v,(n,len(v)),replace=True).mean(axis=1)
            out.append({'model':model,'vs':baseline,'gain_mae_pp':float(v.mean()),
                        'ci_low':float(np.quantile(sim,.025)),'ci_high':float(np.quantile(sim,.975)),
                        'bootstrap_pr_gain_positive':float((sim>0).mean())})
    return pd.DataFrame(out)


def synthetic_test():
    rng=np.random.default_rng(2512)
    groups=np.r_[np.arange(56),np.arange(10)]
    n=len(groups)
    c=rng.uniform(2,60,n); i=rng.uniform(1,25,n)
    d=pd.DataFrame({'protein_position':groups,
                    'variant_key':[f'v{k}' for k in range(n)],
                    'eti_corrector_pp':c,'eti_acute_pp':i,
                    'eti_baseline_fsk':rng.uniform(0,9,n)})
    d['eti_total_pp']=c+i
    d['vti_corrector_pp']=c+5+.15*c+rng.normal(0,4,n)
    d['vti_acute_pp']=i+2-.04*c+rng.normal(0,2,n)
    d['vti_total_pp']=d.vti_corrector_pp+d.vti_acute_pp
    d['nbd1']=d.protein_position.between(10,40).astype(float)
    p,m,cm,sel=evaluate(d)
    assert len(p)==66 and len(m)==6 and len(cm)==8 and len(sel)==30
    assert np.isfinite(m.position_mae_pp).all()
    print('PASS synthetic: grouped nested CV, component reconstruction, evaluation, bootstraps')
    print(m.to_string(index=False))
    print(bootstraps(p,['component_huber'],n=100).to_string(index=False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--bootstraps',type=int,default=10000)
    args=ap.parse_args()
    if args.self_test:
        return synthetic_test()
    d=prepare_training_data()
    pred,metrics,components,params=evaluate(d)
    boot=bootstraps(pred,metrics.model.tolist(),n=args.bootstraps)
    print('='*100)
    print('PHASE 25B | COMPONENT-SPECIFIC ETI -> VTI FUNCTIONAL RESCUE | TRAIN ONLY')
    print('='*100)
    print(f'Variants: {len(d)} | positions: {d.protein_position.nunique()} | VTI held out by position')
    print('Inputs: measured ETI assay + position; target: continuous in-vitro VTI rescue')
    print('Mechanistic models learn separate corrector and acute-IVA targets (unlike Phase 25A).')
    print('NBD1 390-670 is POST HOC: exploratory only. No protected outcomes opened.')
    print('\nA. CONTINUOUS TOTAL-RESCUE TEST METRICS (GROUPED OOF)')
    print(metrics.round(4).to_string(index=False))
    print('\nB. BIOLOGICAL COMPONENT PREDICTION DIAGNOSTICS')
    print(components.round(4).to_string(index=False))
    print('\nC. PAIRED BOOTSTRAP BY PROTEIN POSITION')
    print(boot.round(4).to_string(index=False))
    print('\nD. SELECTED ALPHAS (INNER GROUPED CV)')
    print(params.to_string(index=False))
    OUT.mkdir(parents=True,exist_ok=True)
    for filename,table in (
        ('phase25b_oof.csv',pred),('phase25b_metrics.csv',metrics),
        ('phase25b_components.csv',components),('phase25b_bootstrap.csv',boot),
        ('phase25b_parameters.csv',params)):
        table.to_csv(OUT/filename,index=False)
    print('\nPASS Phase25B; saved:',OUT)


if __name__=='__main__':
    main()
