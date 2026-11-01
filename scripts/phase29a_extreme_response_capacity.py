#!/usr/bin/env python3
"""Phase 29A: Can model capacity/target weighting defeat CFTR corrector mean regression?

Exploratory VTI TRAIN-only experiment. Uses frozen 2x5 protein-position grouped
splits, original 80 features, original regression model, frozen heldout acute
predictions. NO validation or final TEST labels, NO mutation-specific fitting to
outer heldout outcomes. No architecture selected from protected data.

Main distinction: more variable predictions are only useful if accurate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT/'scripts'))
REFERENCE = ROOT/'results/biology/phase26h2/phase26h2_oof.csv'
OUT = ROOT/'results/biology/phase29a'
KEY = ['repeat','fold','variant_key','protein_position']
CONFIGS = ['global','higher_capacity','response_balanced',
           'capacity_balanced','extra_trees']


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def regime(y):
    y=np.asarray(y, float)
    return np.where(y < 10, 'low', np.where(y > 40, 'high','middle'))


def weights_by_train_regime(y):
    """Equal total TRAIN weight per regime, never uses held-out y."""
    groups=regime(y)
    counts={name:int((groups==name).sum()) for name in ('low','middle','high')}
    if any(v==0 for v in counts.values()):
        raise ValueError(f'Outer training fold missing regime: {counts}')
    w=np.array([len(y)/(3*counts[g]) for g in groups],float)
    if not np.isfinite(w).all() or not np.isclose(w.mean(),1):
        raise ValueError('Invalid training weights')
    return w,counts


def cap_model(base):
    """Increase capacity while preserving the original feature and estimator interface."""
    params=base.get_params()
    needed={'max_depth','min_child_weight','n_estimators','learning_rate',
            'reg_lambda','subsample','colsample_bytree'}
    # XGBRegressor exposes these even if original parameters are unset (None).
    if not needed.issubset(params):
        raise TypeError(f'Expected XGBRegressor-like model; missing {needed-set(params)}')
    return base.set_params(max_depth=5, min_child_weight=1,
                           n_estimators=350, learning_rate=0.04,
                           reg_lambda=1.0, subsample=0.9,
                           colsample_bytree=0.9, n_jobs=4,
                           random_state=29)


def make_model(h, config):
    if config=='extra_trees':
        return ExtraTreesRegressor(n_estimators=250, max_features=0.8,
                                   min_samples_leaf=2, max_depth=None,
                                   bootstrap=False, n_jobs=4, random_state=29)
    model=h.previous.model(h.SPEC)
    if config in ('higher_capacity','capacity_balanced'):
        model=cap_model(model)
    return model


def fit_predict(h, config, xtr, ytr, xte):
    model=make_model(h,config)
    sample_weight=None
    if config in ('response_balanced','capacity_balanced'):
        sample_weight,_=weights_by_train_regime(ytr)
    if config=='extra_trees':
        imp=SimpleImputer(strategy='median',add_indicator=True,
                          keep_empty_features=True)
        xtr=imp.fit_transform(xtr)
        xte=imp.transform(xte)
    if sample_weight is None:
        model.fit(xtr,ytr)
    else:
        model.fit(xtr,ytr,sample_weight=sample_weight)
    train_pred=np.asarray(model.predict(xtr),float).reshape(-1)
    test_pred=np.asarray(model.predict(xte),float).reshape(-1)
    if len(train_pred)!=len(ytr) or len(test_pred)!=len(xte):
        raise ValueError('Model prediction cardinality mismatch')
    if not np.isfinite(train_pred).all() or not np.isfinite(test_pred).all():
        raise ValueError('Nonfinite prediction')
    return train_pred,test_pred


def pos_mae(d, label, pred):
    q=d[['protein_position',label,pred]].copy()
    q['ae']=(q[label]-q[pred]).abs()
    return float(q.groupby('protein_position').ae.mean().mean())


def gain_bootstrap(frame, candidate, reference, truth, nboot, seed):
    q=frame[['repeat','protein_position',candidate,reference,truth]].copy()
    q['delta']=(q[reference]-q[truth]).abs()-(q[candidate]-q[truth]).abs()
    perpos=(q.groupby(['repeat','protein_position']).delta.mean()
            .groupby('protein_position').mean().to_numpy(float))
    rng=np.random.default_rng(seed)
    draws=rng.integers(0,len(perpos),size=(nboot,len(perpos)))
    dist=perpos[draws].mean(axis=1)
    return {'target':truth,'candidate':candidate,'reference':reference,
            'mae_gain_pp':float(perpos.mean()),
            'ci95_low':float(np.quantile(dist,.025)),
            'ci95_high':float(np.quantile(dist,.975)),
            'positions':len(perpos)}


def averaged_one_variant(oof):
    first=oof.groupby(['variant_key','protein_position'],as_index=False).agg(
        observed_corrector=('observed_corrector','first'),
        observed_total=('observed_total','first'))
    preds=oof.groupby('variant_key',as_index=False)[[f'corrector_{k}' for k in CONFIGS]].mean()
    out=first.merge(preds,on='variant_key',validate='one_to_one')
    if len(out)!=210: raise ValueError('Expected 210 variants after repeat aggregation')
    return out


def self_test():
    y=np.array([1,2,15,18,22,42,55],float)
    w,counts=weights_by_train_regime(y)
    assert counts=={'low':2,'middle':3,'high':2}
    assert abs(w.mean()-1)<1e-12
    assert np.isclose(w[:2].sum(),w[2:5].sum()) and np.isclose(w[:2].sum(),w[5:].sum())
    q=pd.DataFrame({'repeat':[1,1,2,2],'protein_position':[1,2,1,2],
                    'truth':[0,10,0,10],'old':[5,5,5,5],'new':[0,10,0,10]})
    assert np.isclose(gain_bootstrap(q,'new','old','truth',100,29)['mae_gain_pp'],5)
    print('PASS: training-only response weights, repeated-position bootstrap, diagnostics')


def run(args):
    import phase26h_auxiliary_corrector as h
    v,X80,_,_,folds=h.load_data()
    x=np.asarray(X80,float)
    y=v.corrector_rescue_pp.to_numpy(float)
    total=v.vti_rescue_pp.to_numpy(float)
    pos=v.protein_position.to_numpy(int)
    keys=v.variant_key.astype(str).to_numpy()
    if x.shape!=(210,80) or len(set(keys))!=210 or len(set(pos))!=168 or len(folds)!=10:
        raise ValueError('Unexpected frozen training cohort/split dimensions')
    if not np.isfinite(x[:,:68]).all() or np.isinf(x).any():
        raise ValueError('Invalid baseline features')
    # Preserve original constant feature, exactly as in previous benchmarks.
    x=np.column_stack([x,np.ones(len(x))])
    frozen=pd.read_csv(REFERENCE,dtype={'variant_key':str})
    if len(frozen)!=420 or frozen.duplicated(KEY).any():
        raise ValueError('Unexpected frozen OOF identity/cardinality')
    needed=set(KEY+['vti_fixed','vti_fixed_total','observed_corrector','observed_total'])
    if not needed.issubset(frozen):raise ValueError(f'Missing reference columns {needed-set(frozen)}')
    print('='*106)
    print('PHASE 29A | MODEL CAPACITY / EXTREME RESPONSE LEARNING | VTI TRAIN ONLY')
    print('='*106)
    print('Same 210 mutations, 168 positions, original 2x5 grouped folds; 80 original features.')
    print('Frozen global + acute parity required. Class weights use OUTER TRAIN ONLY.')
    print('All candidates fixed in advance; NO VAL/TEST labels; previously reused TRAIN folds.')
    print('Regime-balanced training optimises a different objective; evaluate both overall and balanced MAE.')
    print('Baseline estimator parameters:',h.previous.model(h.SPEC).get_params())
    rows=[];trainrows=[];checks=[]
    for i,(repeat,fold,tr_keys,te_keys) in enumerate(folds,1):
        tr=np.flatnonzero(np.isin(keys,list(tr_keys)))
        te=np.flatnonzero(np.isin(keys,list(te_keys)))
        if len(tr)+len(te)!=210 or set(pos[tr])&set(pos[te]):
            raise ValueError('Positions overlap across split')
        ref=frozen[(frozen.repeat==repeat)&(frozen.fold==fold)].set_index('variant_key')
        if len(ref)!=len(te) or set(ref.index)!=set(keys[te]):
            raise ValueError('Frozen heldout split changed')
        ref=ref.loc[keys[te]]
        if not np.allclose(ref.observed_corrector,y[te],rtol=0,atol=1e-8):
            raise ValueError('Corrector labels differ from frozen source')
        if not np.allclose(ref.observed_total,total[te],rtol=0,atol=1e-8):
            raise ValueError('Total labels differ from frozen source')
        acute=ref.vti_fixed_total.to_numpy(float)-ref.vti_fixed.to_numpy(float)
        predictions={}
        for conf in CONFIGS:
            yh_tr,yh_te=fit_predict(h,conf,x[tr],y[tr],x[te])
            predictions[conf]=yh_te
            trainrows.append({'repeat':repeat,'fold':fold,'model':conf,
                              'train_variant_mae_pp':float(np.abs(yh_tr-y[tr]).mean()),
                              'train_pred_sd':float(yh_tr.std()),
                              'train_pred_min':float(yh_tr.min()),
                              'train_pred_max':float(yh_tr.max())})
        parity=np.max(np.abs(predictions['global']-ref.vti_fixed.to_numpy(float)))
        if parity>1e-5: raise ValueError(f'Frozen parity failure {parity} repeat={repeat} fold={fold}')
        counts=weights_by_train_regime(y[tr])[1]
        checks.append({'repeat':repeat,'fold':fold,'train_n':len(tr),'test_n':len(te),
                       'train_positions':len(set(pos[tr])),'test_positions':len(set(pos[te])),
                       'global_parity_max_abs':parity,**{f'n_train_{k}':v for k,v in counts.items()}})
        for j,z in enumerate(te):
            row={'repeat':repeat,'fold':fold,'variant_key':keys[z],
                 'protein_position':int(pos[z]),'observed_corrector':float(y[z]),
                 'observed_total':float(total[z]),'frozen_acute':float(acute[j]),
                 'regime':str(regime([y[z]])[0])}
            for conf in CONFIGS:
                row['corrector_'+conf]=float(predictions[conf][j])
                row['total_'+conf]=float(predictions[conf][j]+acute[j])
            rows.append(row)
        print(f'fold {i:02d}/10 repeat={repeat} fold={fold}: ntrain={len(tr)} ntest={len(te)} '
              f'parity={parity:.2e}',flush=True)
    oof=pd.DataFrame(rows)
    if len(oof)!=420 or oof.duplicated(KEY).any():raise ValueError('OOF completeness failure')
    aggregated=averaged_one_variant(oof)
    metric=[]; subgroup=[]; dispersion=[]; rank=[]; gains=[]
    train=pd.DataFrame(trainrows)
    for target in ('corrector','total'):
        truth='observed_'+target
        for conf in CONFIGS:
            pred=f'{target}_{conf}'
            metric.append({'target':target,'model':conf,
                 'position_mae_pp':float(np.mean([pos_mae(z,truth,pred) for _,z in oof.groupby('repeat')])),
                 'variant_mae_pp':float(np.abs(oof[pred]-oof[truth]).mean()),
                 'bias_pp':float((oof[pred]-oof[truth]).mean())})
            if target=='corrector':
                for g in ('low','middle','high'):
                    q=oof[oof.regime==g]
                    subgroup.append({'model':conf,'regime':g,'n_variants':q.variant_key.nunique(),
                       'n_positions':q.protein_position.nunique(),
                       'position_mae_pp':float(np.mean([pos_mae(z,truth,pred) for _,z in q.groupby('repeat')])),
                       'mean_prediction_pp':float(q[pred].mean()),
                       'mean_truth_pp':float(q[truth].mean()),
                       'signed_bias_pp':float((q[pred]-q[truth]).mean())})
        for k,conf in enumerate(CONFIGS[1:]):
            gains.append(gain_bootstrap(oof,f'{target}_{conf}',f'{target}_global',truth,
                                       args.bootstraps,seed=2900+100*(target=='total')+k))
    for conf in CONFIGS:
        pred=aggregated[f'corrector_{conf}'].to_numpy(float)
        obs=aggregated.observed_corrector.to_numpy(float)
        dispersion.append({'model':conf,'observed_sd_pp':float(obs.std()),
          'pred_sd_pp':float(pred.std()),'pred_q05':float(np.quantile(pred,.05)),
          'pred_q50':float(np.median(pred)),'pred_q95':float(np.quantile(pred,.95)),
          'pred_min':float(pred.min()),'pred_max':float(pred.max()),
          'spearman_prediction_truth':float(pd.Series(pred).corr(pd.Series(obs),method='spearman')),
          'mean_train_mae_pp':float(train.loc[train.model==conf,'train_variant_mae_pp'].mean()),
          'mean_outer_test_variant_mae_pp':float(np.abs(oof[f'corrector_{conf}']-oof.observed_corrector).mean())})
        for g in ('low','high'):
            binary=(obs<10).astype(int) if g=='low' else (obs>40).astype(int)
            score=-pred if g=='low' else pred
            rank.append({'model':conf,'regime':g,'positives':int(binary.sum()),
                         'ap':float(average_precision_score(binary,score)),
                         'top42_hits':int(binary[np.argsort(-score,kind='stable')[:42]].sum())})
    m=pd.DataFrame(metric)
    g=pd.DataFrame(subgroup)
    balance=g.groupby('model',as_index=False).position_mae_pp.mean().rename(
        columns={'position_mae_pp':'equal_regime_mae_pp'})
    metric=m.merge(balance,on='model',how='left')
    metric.loc[metric.target!='corrector','equal_regime_mae_pp']=np.nan
    OUT.mkdir(parents=True,exist_ok=True)
    files={'repeated_oof':oof,'metrics':metric,'regimes':g,'dispersion':pd.DataFrame(dispersion),
          'ranking':pd.DataFrame(rank),'gain_bootstrap':pd.DataFrame(gains),
          'fold_qc':pd.DataFrame(checks),'train_fit':train}
    for name,df in files.items():df.to_csv(OUT/f'phase29a_{name}.csv',index=False)
    (OUT/'phase29a_provenance.json').write_text(json.dumps({
      'script_hash':sha(Path(__file__)), 'frozen_oof_hash':sha(REFERENCE),
      'n_train_variants':210,'n_train_positions':168,'n_folds':10,
      'configs':CONFIGS,'response_weighting':'train-only inverse 3-bin prevalence',
      'warning':'Multiple tested models; exploratory, reused TRAIN splits. Do not select on protected VAL or TEST.'},indent=2)+'\n')
    print('\nA. OVERALL MAE (lower is better)')
    print(metric.round(4).to_string(index=False))
    print('\nB. BY TRUE CORRECTOR REGIME (diagnostic only)')
    print(g.round(3).to_string(index=False))
    print('\nC. OUTER-OOF PREDICTION SPREAD VS IN-FOLD FIT')
    print(pd.DataFrame(dispersion).round(4).to_string(index=False))
    print('\nD. CORRECTOR RANKING LOW/HIGH (repeat-averaged OOF)')
    print(pd.DataFrame(rank).round(4).to_string(index=False))
    print('\nE. PAIRED POSITION BOOTSTRAP (positive means LOWER MAE)')
    print(pd.DataFrame(gains).round(4).to_string(index=False))
    print('\nF. BASELINE PARITY / TRAINING GROUP SIZES')
    print(pd.DataFrame(checks).to_string(index=False))
    print(f'PASS Phase29A: TRAIN only, outputs={OUT}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--bootstraps',type=int,default=2000)
    a=p.parse_args()
    if a.self_test:self_test()
    else:
        if a.bootstraps<100:p.error('At least 100 bootstrap samples required')
        self_test();run(a)
