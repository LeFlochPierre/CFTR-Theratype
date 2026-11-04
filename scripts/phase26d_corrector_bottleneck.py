#!/usr/bin/env python3
"""Phase 26D: train-only correction bottleneck study.

Hypothesis: can mutation-derived data predict VNZ/TEZ correction, and does
component modelling improve zero-shot total VTI functional rescue?

Main models use mutation-only sequence/structure features; assay-conditioned
comparisons are explicitly separate. Existing frozen OOF assignments determine
outer held-out positions. No VTI validation/test outcomes are materialised.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

OUT = Path('results/biology/phase26d')
ROOT = Path.cwd()
TASKS = {
    'zero_total_seq68': ('total', 'sequence', ()),
    'zero_total_seq80': ('total', 'sequence_structure', ()),
    'zero_corrector_seq80': ('corrector', 'sequence_structure', ()),
    'zero_acute_seq80': ('acute', 'sequence_structure', ()),
    'baseline_total_seq80': ('total', 'sequence_structure', ('baseline_fsk_pct_wt',)),
    'baseline_corrector_seq80': ('corrector', 'sequence_structure', ('baseline_fsk_pct_wt',)),
    'iva_total_seq80': ('total', 'sequence_structure', ('baseline_fsk_pct_wt','iva_treated_pct_wt')),
}
GRID = [(2, 5.0, 180), (3, 8.0, 250)]  # small, declared before observing results


def pos_mae(y, pred, group):
    from cftr_theratype.evaluation.metrics import position_mae
    return position_mae(y, pred, group)

def model(spec, seed=26):
    from cftr_theratype.models.regressors import phase26d_model
    return phase26d_model(spec, seed=seed)

def select_spec(X, y, group, quick=False):
    from cftr_theratype.models.regressors import phase26d_select_spec
    return phase26d_select_spec(X, y, group, quick=quick)

def folds_from_historical_oof(oof, expected_keys, position_by_key):
    from cftr_theratype.evaluation.folds import from_historical_oof
    return from_historical_oof(oof, expected_keys, position_by_key)

def load_train():
    from cftr_theratype.datasets.frozen_vti import load_train as canonical_load_train
    return canonical_load_train(ROOT)

def evaluate(d,features,splits,quick=False):
    n=len(d)
    keys=d.variant_key.to_numpy(str)
    pos=d.protein_position.to_numpy(int)
    target={
        'total':d.vti_rescue_pp.to_numpy(float),
        'corrector':d.corrector_rescue_pp.to_numpy(float),
        'acute':d.vti_acute_after_corrector_pp.to_numpy(float),
    }
    matrices={}
    for name,(yname,fname,extra) in TASKS.items():
        x=np.asarray(features[fname],float)
        if extra:
            x=np.column_stack([x,d[list(extra)].to_numpy(float)])
        matrices[name]=x
    oof=[]; selections=[]
    for idx,(repeat,fold,train_keys,test_keys) in enumerate(splits,1):
        it=np.flatnonzero(np.isin(keys,list(train_keys)))
        iv=np.flatnonzero(np.isin(keys,list(test_keys)))
        assert len(it)+len(iv)==n
        assert set(pos[it]).isdisjoint(set(pos[iv]))
        preds={'train_median':np.full(len(iv),np.median(target['total'][it]))}
        for name,(yname,_,_) in TASKS.items():
            X=matrices[name]; y=target[yname]
            spec,score=select_spec(X[it],y[it],pos[it],quick=quick)
            est=model(spec)
            est.fit(X[it],y[it])
            preds[name]=est.predict(X[iv]).astype(float)
            selections.append({'repeat':repeat,'fold':fold,'model':name,
                'max_depth':spec[0],'reg_lambda':spec[1],'n_estimators':spec[2],
                'inner_position_mae':score,'train_positions':len(np.unique(pos[it])),
                'test_positions':len(np.unique(pos[iv]))})
        preds['zero_components_seq80']=preds['zero_corrector_seq80']+preds['zero_acute_seq80']
        preds['baseline_components_seq80']=preds['baseline_corrector_seq80']+preds['zero_acute_seq80']
        # These two are ORACLE analyses, because a VTI component must be measured.
        preds['oracle_measured_corrector_plus_pred_acute']=target['corrector'][iv]+preds['zero_acute_seq80']
        preds['oracle_pred_corrector_plus_measured_acute']=preds['zero_corrector_seq80']+target['acute'][iv]
        for j,i in enumerate(iv):
            row={'repeat':repeat,'fold':fold,'variant_key':keys[i],
                 'protein_position':int(pos[i]),'study_cohort':d.study_cohort.iloc[i],
                 'observed_total':target['total'][i],
                 'observed_corrector':target['corrector'][i],
                 'observed_acute':target['acute'][i]}
            row.update({f'pred_{name}':float(v[j]) for name,v in preds.items()})
            oof.append(row)
        print(f'  completed outer split {idx}/{len(splits)}: repeat={repeat}, fold={fold}, held-out positions={len(np.unique(pos[iv]))}',flush=True)
    out=pd.DataFrame(oof)
    for repeat,t in out.groupby('repeat'):
        assert len(t)==n and t.variant_key.nunique()==n
    return out,pd.DataFrame(selections)


def summaries(out,historical):
    models=[c for c in out if c.startswith('pred_')]
    rows=[]
    for (rep), d in out.groupby('repeat'):
        p=d.protein_position.to_numpy(int)
        for col in models:
            name=col[5:]
            yname='observed_corrector' if name=='zero_corrector_seq80' or name=='baseline_corrector_seq80' else ('observed_acute' if name=='zero_acute_seq80' else 'observed_total')
            y=d[yname].to_numpy(float); pred=d[col].to_numpy(float)
            rows.append({'repeat':rep,'model':name,'target':yname,'position_mae_pp':pos_mae(y,pred,p),
                         'variant_mae_pp':float(np.mean(np.abs(y-pred))),
                         'rmse_pp':float(np.sqrt(np.mean((y-pred)**2))),
                         'n_positions':len(np.unique(p))})
    met=pd.DataFrame(rows)
    # Position and repeat are paired for comparisons; do not treat repeat folds as extra independent patients.
    comparisons=[]
    for candidate in ('zero_components_seq80','baseline_components_seq80','baseline_total_seq80','iva_total_seq80'):
        t=out.copy()
        t['gain']=np.abs(t.observed_total-t.pred_zero_total_seq80)-np.abs(t.observed_total-t[f'pred_{candidate}'])
        paired=t.groupby(['repeat','protein_position']).gain.mean().groupby('protein_position').mean().to_numpy()
        rng=np.random.default_rng(2604)
        sims=np.mean(rng.choice(paired,size=(5000,len(paired)),replace=True),axis=1)
        comparisons.append({'candidate':candidate,'comparator':'zero_total_seq80',
                'gain_pp':float(paired.mean()),'ci95_low':float(np.quantile(sims,.025)),
                'ci95_high':float(np.quantile(sims,.975)), 'independent_positions':len(paired)})
    comp=pd.DataFrame(comparisons)
    # Historical OOF metrics provided only if observed_pp matches total-rescue target.
    historic=[]
    valid=historical.merge(out[['repeat','variant_key','observed_total']].drop_duplicates(),
                           on=['repeat','variant_key'],how='inner')
    if len(valid) and np.allclose(valid.observed_pp,valid.observed_total,atol=1e-5):
        for (rep,name), t in valid.groupby(['repeat','model']):
            if t.variant_key.nunique()!=out.variant_key.nunique():continue
            historic.append({'repeat':rep,'model':str(name),
                             'position_mae_pp':pos_mae(t.observed_pp.to_numpy(float),t.predicted_pp.to_numpy(float),t.protein_position.to_numpy(int))})
    elif len(valid):
        print('NOTE: historical observed_pp differs from total rescue: no historical comparison made')
    return met,comp,pd.DataFrame(historic)


def synthetic():
    rng=np.random.default_rng(26)
    n=60
    pos=np.repeat(np.arange(30),2)
    d=pd.DataFrame({'variant_key':[f'syn{i}' for i in range(n)],'protein_position':pos,
                    'study_cohort':['synthetic']*n})
    a=rng.normal(size=(n,68)); b=np.column_stack([a,rng.normal(size=(n,12))])
    d['baseline_fsk_pct_wt']=rng.uniform(0,5,n)
    d['iva_treated_pct_wt']=d.baseline_fsk_pct_wt+rng.uniform(1,8,n)
    d['corrector_rescue_pp']=10+7*b[:,0]+rng.normal(0,3,n)
    d['vti_acute_after_corrector_pp']=4+3*b[:,1]+rng.normal(0,2,n)
    d['vti_rescue_pp']=d.corrector_rescue_pp+d.vti_acute_after_corrector_pp
    d['vnz_tez_fsk_pct_wt']=d.baseline_fsk_pct_wt+d.corrector_rescue_pp
    d['vti_treated_pct_wt']=d.baseline_fsk_pct_wt+d.vti_rescue_pp
    folds=[]
    g=GroupKFold(3)
    for fold,(it,iv) in enumerate(g.split(d,groups=pos),1):
        folds.append((1,fold,set(d.variant_key.iloc[it]),set(d.variant_key.iloc[iv])))
    out,sel=evaluate(d,{'sequence':a,'sequence_structure':b},folds,quick=True)
    met,comp,_=summaries(out,pd.DataFrame(columns=['repeat','model','variant_key','observed_pp','predicted_pp','protein_position']))
    assert len(out)==n and not out.isna().any().any()
    assert set(met.model).issuperset({'zero_total_seq80','zero_components_seq80','baseline_total_seq80'})
    assert len(comp)==4
    print('PASS synthetic: grouped CV, continuous total/corrector/acute targets, fold-local preprocessing, paired bootstrap')
    print(met[['model','position_mae_pp']].round(3).to_string(index=False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--quick',action='store_true',help='Use a single fixed XGBoost specification (preflight only)')
    args=ap.parse_args()
    if args.self_test:
        synthetic();return
    d,feats,historical,splits,source=load_train()
    print('='*106)
    print('PHASE 26D | ZERO-SHOT CORRECTOR RESCUE BOTTLENECK | TRAIN ONLY')
    print('='*106)
    print('All 210 train variants / 168 independent positions; frozen historical position folds')
    print('Primary estimand: VTI total in-vitro rescue (%WT percentage points). This is NOT clinical efficacy.')
    print('Mutation-only models are separate from baseline-/IVA-assay-informed models and oracle checks.')
    print('NBD1 results from earlier training analyses are exploratory; no extra NBD1 features here.')
    print('No VTI validation/test outcomes or protected ETI outcomes loaded.')
    print('Frozen fold reference:',source)
    print('Train columns:',[x for x in ('baseline_fsk_pct_wt','iva_treated_pct_wt','corrector_rescue_pp','vti_acute_after_corrector_pp','vti_rescue_pp')])
    out,sel=evaluate(d,feats,splits,quick=args.quick)
    met,comp,hist=summaries(out,historical)
    OUT.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT/'phase26d_oof.csv',index=False)
    sel.to_csv(OUT/'phase26d_selections.csv',index=False)
    met.to_csv(OUT/'phase26d_metrics_by_repeat.csv',index=False)
    comp.to_csv(OUT/'phase26d_paired_bootstrap.csv',index=False)
    hist.to_csv(OUT/'phase26d_historical_reference.csv',index=False)
    print('\nA. MODEL PERFORMANCE BY TARGET (AVERAGED ACROSS REPEATS)')
    print(met.groupby(['target','model'],as_index=False)[['position_mae_pp','rmse_pp']].mean().sort_values(['target','position_mae_pp']).round(4).to_string(index=False))
    print('\nB. PAIRED POSITION BOOTSTRAP VS RE-FIT MUTATION-ONLY TOTAL MODEL')
    print(comp.round(4).to_string(index=False))
    print('\nC. ORIGINAL STORED ZERO-SHOT REFERENCES, WHEN TARGETS ALIGN')
    print(hist.groupby('model',as_index=False).position_mae_pp.mean().sort_values('position_mae_pp').round(4).to_string(index=False) if len(hist) else 'No matching historical OOF reference.')
    print('\nD. COHORT AND RESPONSE TAIL DIAGNOSTICS (DIRECT ZERO-SHOT MODEL)')
    for label,sub in [('ALL',out),('VTI',out[out.study_cohort=='VTI']),('ETI, VTI',out[out.study_cohort=='ETI, VTI']),('true_rescue_gt40',out[out.observed_total>40])]:
        if len(sub):
            ae=np.abs(sub.observed_total-sub.pred_zero_total_seq80)
            ce=np.abs(sub.observed_corrector-sub.pred_zero_corrector_seq80)
            print(f'{label:>17} variants_per_repeat={len(sub)//sub.repeat.nunique():3} total_MAE={ae.mean():7.3f} corrector_MAE={ce.mean():7.3f}')
    print('\nE. INNER-CV SELECTION COUNTS')
    print(sel.groupby(['model','max_depth','reg_lambda','n_estimators']).size().rename('folds').to_string())
    print('\nPASS Phase26D. Outputs:',OUT)

if __name__=='__main__':main()
