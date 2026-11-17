"""Synthetic end-to-end contracts preserved from the three Phase31 standalone tests."""
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

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import phase31i2_repository as m
import phase31j_error_diagnostics as j
import phase31k_response_generalisation as k
from cftr_theratype.datasets import frozen_vti
from cftr_theratype.models import regressors

rng = np.random.default_rng(41)
orig_read_parquet = pd.read_parquet

def read_parquet_fake(path, columns=None, filters=None, **kw):
    frame=pd.read_csv(path)
    if columns is not None: frame=frame[columns]
    if filters is not None: raise AssertionError('No filtered parquet in synthetic test')
    return frame

def test_phase31i2_repository_end_to_end():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)
        (root/'data/processed').mkdir(parents=True)
        (root/'results/biology/phase31a').mkdir(parents=True)
        (root/'results/biology/phase26g').mkdir(parents=True)
        ids=[]
        for j,pos in enumerate(range(50,218)):
            for k in range(2 if j<42 else 1):
                aa=f'A{pos}{"V" if k==0 else "G"}'
                ids.append(dict(variant_key=f'c.{pos*3+k}A>T',hgvs_protein=f'p.{aa}',protein_position=pos,transfer_split='train'))
        assert len(ids)==210
        splits=pd.DataFrame(ids)
        for j,pos in enumerate(range(400,470)):
            splits.loc[len(splits)]={'variant_key':f's{j}','hgvs_protein':f'p.A{pos}V',
                                     'protein_position':pos,'transfer_split':'excluded'}
        allsubs=[]
        for j,pos in enumerate(range(50,470)):
            for k in range(2 if 50<=pos<92 else 1):
                aa=f'A{pos}{"V" if k==0 else "G"}'
                allsubs.append(dict(variant_key=f'b{j}_{k}',hgvs_protein=f'p.{aa}',
                                    protein_position=pos,split='train'))
        for pos in range(900,918):
            allsubs.append(dict(variant_key=f'x{pos}',hgvs_protein=f'p.A{pos}V',
                                protein_position=pos,split='val' if pos<909 else 'test'))
            splits.loc[len(splits)]={'variant_key':f'v{pos}','hgvs_protein':f'p.A{pos}V',
                                     'protein_position':pos,'transfer_split':'val' if pos<909 else 'test'}
        pd.DataFrame(allsubs).to_csv(root/'data/processed/bihler_2024_missense_splits.parquet',index=False)
        splits.to_csv(root/'data/processed/bihler_2026_transfer_splits.parquet',index=False)
        et=[];s5=[];s6=[]
        for j,r in enumerate(allsubs):
            code=r['hgvs_protein'][2:]
            pos=r['protein_position']
            val=np.sin(pos/30)*12+(5 if code.endswith('G') else 1)
            et.append(dict(variant_key=r['variant_key'],hgvs_protein=r['hgvs_protein'],eti_rescue_pp=val))
            for study in ('P289','U032'):
                s5.append(dict(aa_code=code,study=study,mutation_type='missense',eti_rescue_pp=val*(1.0 if study=='P289' else 1.8)))
                s6.append(dict(aa_code=code,study=study,mutation_type='missense',
                               baseline_mature_fraction=.3+val/1000,eti_mature_fraction=.5+val/1000))
        pd.DataFrame(et).to_csv(root/'data/processed/bihler_2024_theratyping.parquet',index=False)
        pd.DataFrame(s5).to_csv(root/'results/biology/phase31a/phase31a_castellani_s5_transport.csv',index=False)
        pd.DataFrame(s6).to_csv(root/'results/biology/phase31a/phase31a_castellani_s6_processing.csv',index=False)
        d=pd.DataFrame(ids)
        d['study_cohort']='VTI'
        x80=rng.normal(size=(210,80))
        d['corrector_rescue_pp']= 10 + 4*x80[:,0] + 1.5*np.sin(d.protein_position/30) + rng.normal(0,1.5,210)
        d['vti_rescue_pp']=d.corrector_rescue_pp+4+2*x80[:,1]+rng.normal(0,.8,210)
        keys=d.variant_key.to_numpy(str)
        pos=d.protein_position.to_numpy(int)
        frozen=[]
        for k,(tr,te) in enumerate(GroupKFold(5).split(x80,groups=pos)):
            frozen.append((1,k+1,set(keys[tr]),set(keys[te])))
        def fake_train(root=None): return d, {'sequence_structure':x80,'sequence':x80[:,:68]},pd.DataFrame(),frozen,'synthetic'
        pd.read_parquet=read_parquet_fake
        m.ROOT=root
        m.OUT=root/'results/biology/phase31i'
        old_train=m.frozen_vti.load_train
        m.frozen_vti.load_train=fake_train
        # Fast XGBoost parity reference from PRESELECTED Phase26G spec.
        for target,ycol in [('corrector','corrector_rescue_pp'),('total','vti_rescue_pp')]:
            selections=[]; past=[]
            for rep,fold,trkeys,tekeys in frozen:
                tr=np.flatnonzero(np.isin(keys,list(trkeys)))
                te=np.flatnonzero(np.isin(keys,list(tekeys)))
                spec=(2,5.0,180)
                mod=m.regressors.phase26g_model('xgb',spec)
                mod.fit(x80[tr],d[ycol].to_numpy(float)[tr])
                for i,pred in zip(te,mod.predict(x80[te])):
                    past.append(dict(repeat=rep,fold=fold,variant_key=keys[i],protein_position=pos[i],
                                     observed=d[ycol].iloc[i],xgb_80=pred))
                selections.append(dict(repeat=rep,fold=fold,model='xgb_80',selected_param=str(spec)))
            pd.DataFrame(past).to_csv(root/f'results/biology/phase26g/phase26g_{target}_oof.csv',index=False)
            pd.DataFrame(selections).to_csv(root/f'results/biology/phase26g/phase26g_{target}_selected.csv',index=False)
        try:
            m.main(['audit'])
            assert (root/'results/biology/phase31i/phase31i2_repo_audit.json').exists()
            m.main(['run','--quick','--teacher-folds','3','--bootstrap','100'])
            quick=pd.read_csv(root/'results/biology/phase31i/phase31i2_repo_quick_oof.csv')
            assert set(quick.model)==set(m.MODELS)
            m.main(['run','--teacher-folds','3','--bootstrap','100'])
            result=pd.read_csv(root/'results/biology/phase31i/phase31i2_repo_oof.csv')
            assert len(result)==210*2*5
            assert not result.duplicated(['target','model','repeat','variant_key']).any()
            assert (root/'results/biology/phase31i/phase31i2_repo_paired_bootstrap.csv').exists()
            print('TEST PASSED: true Phase26G XGB parity, 5 outer position-disjoint folds, 2 outcomes, 5 models.')
        finally:
            pd.read_parquet=orig_read_parquet
            m.frozen_vti.load_train=old_train


def test_phase31j_end_to_end():
    orig_read=pd.read_parquet
    def parquet_csv(path,columns=None,filters=None,**kw):
        if filters is not None:raise AssertionError('Unexpected parquet predicate filter')
        d=pd.read_csv(path)
        return d if columns is None else d[columns]
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        for rel in ('data/processed','results/biology/phase31a','results/biology/phase26g','results/biology/phase31i'):
            (root/rel).mkdir(parents=True,exist_ok=True)
        rng=np.random.default_rng(31)
        records=[]
        for k,p in enumerate(range(50,218)):
            for alt in (['V','G'] if k<42 else ['V']):
                records.append(dict(variant_key=f'c.{p*3}{alt}',hgvs_protein=f'p.A{p}{alt}',protein_position=p,transfer_split='train',variant_type='missense'))
        assert len(records)==210
        meta=pd.DataFrame(records)
        extr=[]
        for p in range(860,874):
            extr.append(dict(variant_key=f'ext{p}',hgvs_protein=f'p.A{p}V',protein_position=p,transfer_split='val' if p<867 else 'test',variant_type='missense'))
        pd.concat([meta,pd.DataFrame(extr)]).to_csv(root/'data/processed/bihler_2026_transfer_splits.parquet',index=False)
        d=meta.copy()
        d['study_cohort']='VTI'
        d['baseline_fsk_pct_wt']=1+9*rng.uniform(size=len(d))
        d['corrector_rescue_pp']=12+12*np.sin(d.protein_position.to_numpy()/30.)+rng.normal(size=len(d))
        d['vti_rescue_pp']=d.corrector_rescue_pp+9+3*rng.normal(size=len(d))
        keys=d.variant_key.to_numpy(str)
        pos=d.protein_position.to_numpy(int)
        splits=[]
        past={x:[] for x in ('corrector','total')}
        stacked=[]
        for repeat in (1,2):
            for fold,(tr,te) in enumerate(GroupKFold(5).split(d,groups=pos),1):
                splits.append((repeat,fold,set(keys[tr]),set(keys[te])))
                for target,col in [('corrector','corrector_rescue_pp'),('total','vti_rescue_pp')]:
                    y=d[col].to_numpy(float)
                    pred=np.mean(y[tr])+rng.normal(0,3,len(te))
                    for row,prediction in zip(te,pred):
                        entry=dict(target=target,repeat=repeat,fold=fold,variant_key=keys[row],protein_position=pos[row],observed=y[row],prediction=prediction)
                        stacked.append(dict(**entry,model='M0'))
                        for model in ('M1','M2','M3','M3_perm'):
                            stacked.append(dict(**{**entry,'prediction':prediction+rng.normal(0,1)},model=model))
                        past[target].append({k:v for k,v in entry.items() if k!='target'}|{'xgb_80':prediction})
        for target in past:
            pd.DataFrame(past[target]).drop(columns=['prediction']).to_csv(root/f'results/biology/phase26g/phase26g_{target}_oof.csv',index=False)
        pd.DataFrame(stacked).to_csv(root/'results/biology/phase31i/phase31i2_repo_oof.csv',index=False)
        s24=[];eti=[];s5=[];s6=[]
        for p in range(50,430):
            s24.append(dict(variant_key=f'x{p}',hgvs_protein=f'p.A{p}V',protein_position=p,split='train'))
            eti.append(dict(variant_key=f'x{p}',hgvs_protein=f'p.A{p}V',eti_rescue_pp=5*np.cos(p/35)+rng.normal()))
            for study in ('P289','U032'):
                v=5*np.cos(p/35)+rng.normal()
                s5.append(dict(study=study,aa_code=f'A{p}V',mutation_type='missense',eti_rescue_pp=v))
                s6.append(dict(study=study,aa_code=f'A{p}V',mutation_type='missense',baseline_mature_fraction=.3+v/100,eti_mature_fraction=.5+v/100))
        for p in range(860,874):
            s24.append(dict(variant_key=f'y{p}',hgvs_protein=f'p.A{p}V',protein_position=p,split='test'))
            eti.append(dict(variant_key=f'y{p}',hgvs_protein=f'p.A{p}V',eti_rescue_pp=1))
        pd.DataFrame(s24).to_csv(root/'data/processed/bihler_2024_missense_splits.parquet',index=False)
        pd.DataFrame(eti).to_csv(root/'data/processed/bihler_2024_theratyping.parquet',index=False)
        pd.DataFrame(s5).to_csv(root/'results/biology/phase31a/phase31a_castellani_s5_transport.csv',index=False)
        pd.DataFrame(s6).to_csv(root/'results/biology/phase31a/phase31a_castellani_s6_processing.csv',index=False)
        def fake_load(local_root=None):
            assert local_root == root
            return d,{},pd.DataFrame(),splits,'synthetic-frozen'
        old_load=frozen_vti.load_train
        old_root_j=j.ROOT
        pd.read_parquet=parquet_csv
        j.ROOT=root
        frozen_vti.load_train=fake_load
        try:
            j.main(['run','--out',str(root/'results/biology/phase31j'),'--bootstraps','100'])
            q=pd.read_csv(root/'results/biology/phase31j/phase31j_m0_annotated_oof.csv')
            assert len(q)==210*2*2
            assert q.absolute_error_pp.notna().all()
            joint=pd.read_csv(root/'results/biology/phase31j/phase31j_joint_by_position.csv')
            assert len(joint)==168
            comp=pd.read_csv(root/'results/biology/phase31j/phase31j_transfer_gain_strata.csv')
            assert set(comp.model)=={'M1','M2','M3','M3_perm'}
            j.main(['teachers','--out',str(root/'results/biology/phase31j'),'--bootstraps','100'])
            t=pd.read_csv(root/'results/biology/phase31j/phase31j_teacher_metrics.csv')
            assert len(t)==7 and (t.positions>100).all()
            print('PASS SYNTHETIC Phase31J: 210/168, two repeats, 5 folds, 7 teachers, no protected target load')
        finally:
            pd.read_parquet=orig_read
            frozen_vti.load_train=old_load
            j.ROOT=old_root_j


def test_phase31k_end_to_end():
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
        original=(k.ROOT,k.OUT,frozen_vti.load_train,regressors.phase26g_model)
        def fake_load(root=None):return d, {'sequence_structure':X},pd.DataFrame(),splits,'synthetic-frozen'
        def fast_model(kind,spec,topk=None):
            return Pipeline([('regression',XGBRegressor(n_estimators=12,max_depth=2,learning_rate=.1,
                objective='reg:squarederror',tree_method='hist',n_jobs=1,random_state=3117))])
        k.ROOT=root;k.OUT=root/'results/biology/phase31k'
        frozen_vti.load_train=fake_load;regressors.phase26g_model=fast_model
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
            (k.ROOT,k.OUT,frozen_vti.load_train,regressors.phase26g_model)=original
