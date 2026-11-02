#!/usr/bin/env python3
"""Synthetic standalone integration test; NOT actual CFTR efficacy data."""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase31k_response_generalisation as k
import phase31j_error_diagnostics as j
import phase26d_corrector_bottleneck as prior
import phase26g_feature_model_ablation as p26g


def test():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        for rel in ['results/biology/phase26g','results/biology/phase31i','results/biology/phase26d']:
            (root/rel).mkdir(parents=True)
        rng=np.random.default_rng(3117)
        positions=np.repeat(np.arange(40,208),[2]*42+[1]*126)
        n=len(positions)
        assert n==210 and len(set(positions))==168
        X=rng.normal(size=(n,80))
        X[::3,68:76]=np.nan
        cor=15+3*np.nan_to_num(X[:,1])+rng.normal(0,5,size=n)
        total=cor+7+2*np.nan_to_num(X[:,2])+rng.normal(0,3,n)
        d=pd.DataFrame({'variant_key':[f'v{i}' for i in range(n)],'protein_position':positions,
                        'transfer_split':['train']*n,'corrector_rescue_pp':cor,
                        'vti_rescue_pp':total})
        keys=d.variant_key.to_numpy()
        splits=[]; refs={t:[] for t in ('corrector','total')}; sel={t:[] for t in refs}
        stacked=[]; historical=[]
        for rep in (1,2):
            for fold,(tr,te) in enumerate(GroupKFold(5).split(X,groups=positions),1):
                splits.append((rep,fold,set(keys[tr]),set(keys[te])))
                for target,y in [('corrector',cor),('total',total)]:
                    spec=(2,5.,180)
                    sel[target].append(dict(repeat=rep,fold=fold,model='xgb_80',selected_param=str(spec)))
                    preds=np.mean(y[tr])+0.5*np.nan_to_num(X[te,0])+rng.normal(0,2,size=len(te))
                    for jj,pi in zip(te,preds):
                        base=dict(repeat=rep,fold=fold,variant_key=keys[jj],protein_position=positions[jj],
                                  observed=y[jj],prediction=pi)
                        refs[target].append({**base,'xgb_80':pi})
                        stacked.append({**base,'target':target,'model':'M0'})
                for jj in te:
                    historical.append(dict(repeat=rep,fold=fold,variant_key=keys[jj],protein_position=positions[jj],
                                           observed_total=total[jj],observed_corrector=cor[jj],observed_acute=total[jj]-cor[jj],
                                           pred_zero_corrector_seq80=cor[jj]+1.,pred_zero_acute_seq80=total[jj]-cor[jj]+2.,
                                           pred_zero_total_seq80=total[jj]+4.,pred_zero_components_seq80=total[jj]+3.))
        for target in refs:
            pd.DataFrame(refs[target]).drop(columns='prediction').to_csv(
                root/f'results/biology/phase26g/phase26g_{target}_oof.csv',index=False)
            pd.DataFrame(sel[target]).to_csv(root/f'results/biology/phase26g/phase26g_{target}_selected.csv',index=False)
        pd.DataFrame(stacked).to_csv(root/'results/biology/phase31i/phase31i2_repo_oof.csv',index=False)
        pd.DataFrame(historical).to_csv(root/'results/biology/phase26d/phase26d_oof.csv',index=False)
        original=(k.ROOT,k.OUT,j.ROOT,prior.ROOT,prior.load_train,p26g.make_model)
        def fake_load():return d, {'sequence_structure':X},pd.DataFrame(),splits,'synthetic-frozen'
        def fast_model(kind,spec):
            return Pipeline([('regression',XGBRegressor(n_estimators=12,max_depth=2,learning_rate=.1,
                objective='reg:squarederror',tree_method='hist',n_jobs=1,random_state=3117))])
        k.ROOT=root;k.OUT=root/'results/biology/phase31k';j.ROOT=root;prior.ROOT=root
        prior.load_train=fake_load;p26g.make_model=fast_model
        try:
            k.main(['analyze','--bootstraps','100'])
            a=pd.read_csv(root/'results/biology/phase31k/phase31k_calibration_bins.csv')
            assert len(a)==16 and set(a.bin_type)=={'predicted_quartile','observed_quartile_RETROSPECTIVE'}
            comp=json.loads((root/'results/biology/phase31k/phase31k_component_identity.json').read_text())
            assert comp['identity_absolute_difference']<1e-7
            k.main(['fit','--quick','--bootstraps','100'])
            quick=pd.read_csv(root/'results/biology/phase31k/phase31k_quick_onefold_oof.csv')
            assert len(quick)==4*2*len(splits[0][3])
            assert set(quick.model)==set(k.MODELS)
            assert quick.feature_knn_distance.notna().all()
            # Test complete evaluation with 10 folds and both repeats.
            k.main(['fit','--bootstraps','100'])
            full=pd.read_csv(root/'results/biology/phase31k/phase31k_fit_oof.csv')
            assert len(full)==210*2*2*4
            baseline=full[full.model=='M0']
            assert np.allclose(baseline.sort_values(['target','repeat','variant_key']).prediction.to_numpy(),
                               pd.DataFrame(stacked).sort_values(['target','repeat','variant_key']).prediction.to_numpy())
            met=pd.read_csv(root/'results/biology/phase31k/phase31k_fit_paired_bootstrap.csv')
            assert len(met)==6 and (met.n_independent_positions==168).all()
            tails=pd.read_csv(root/'results/biology/phase31k/phase31k_fit_paired_tail_gains.csv')
            assert len(tails)==48 and set(tails.stratum)=={'observed_bin_RETROSPECTIVE','m0_predicted_bin'}
            rv=pd.read_csv(root/'results/biology/phase31k/phase31k_reliability_metrics.csv')
            assert len(rv)==4
            print('PASS synthetic Phase31K: analysis, exact MSE identity, quick/full nested tail models, 210/168, 2x5 folds')
        finally:
            (k.ROOT,k.OUT,j.ROOT,prior.ROOT,prior.load_train,p26g.make_model)=original


if __name__=='__main__':test()
