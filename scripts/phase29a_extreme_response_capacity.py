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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path.cwd()
REFERENCE = ROOT/'results/biology/phase26h2/phase26h2_oof.csv'
OUT = ROOT/'results/biology/phase29a'
KEY = ['repeat','fold','variant_key','protein_position']
# Re-export canonical fixed configurations for the historical CLI.
from cftr_theratype.models.response_capacity import (
    CONFIGS,
    regime as _canonical_regime,
    weights_by_train_regime as _canonical_weights_by_train_regime,
    cap_model as _canonical_cap_model,
    make_model as _canonical_make_model,
    fit_predict as _canonical_fit_predict,
)
from cftr_theratype.evaluation.response_capacity import (
    pos_mae as _canonical_pos_mae,
    gain_bootstrap as _canonical_gain_bootstrap,
    averaged_one_variant as _canonical_averaged_one_variant,
)
from cftr_theratype.models.regressors import phase26d_model
from cftr_theratype.models.auxiliary import SPEC


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def regime(y):
    """Legacy public script API: canonical response regime assignment."""
    return _canonical_regime(y)


def weights_by_train_regime(y):
    """Legacy API: equal outer-TRAIN total weight per regime."""
    return _canonical_weights_by_train_regime(y)


def cap_model(base):
    """Legacy API: original increased-capacity XGB configuration."""
    return _canonical_cap_model(base)


def make_model(h, config):
    """Legacy API: `h` was the Phase26H module and is now unnecessary.

    The historical Phase26H model factory has been canonical since Batch08.
    """
    return _canonical_make_model(config)


def fit_predict(h, config, xtr, ytr, xte):
    """Legacy API retaining its original signature, without Phase26H import."""
    return _canonical_fit_predict(config, xtr, ytr, xte)


def pos_mae(d, label, pred):
    """Legacy API: same per-position-averaged absolute error."""
    return _canonical_pos_mae(d, label, pred)


def gain_bootstrap(frame, candidate, reference, truth, nboot, seed):
    """Legacy API: same deterministic position bootstrap."""
    return _canonical_gain_bootstrap(frame, candidate, reference, truth, nboot, seed)


def averaged_one_variant(oof):
    """Legacy API: same frozen two-repeat aggregation contract."""
    return _canonical_averaged_one_variant(oof)


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
    from cftr_theratype.datasets.eti_auxiliary import load_data
    v,X80,_,_,folds=load_data(ROOT)
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
    print('Baseline estimator parameters:',phase26d_model(SPEC).get_params())
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
            yh_tr,yh_te=_canonical_fit_predict(conf,x[tr],y[tr],x[te])
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
