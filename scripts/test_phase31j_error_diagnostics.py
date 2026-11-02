#!/usr/bin/env python3
"""Synthetic integration test, not a real-data CFTR result."""
import sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0,str(Path(__file__).resolve().parent))
import phase31j_error_diagnostics as j
import phase26d_corrector_bottleneck as prior
import phase31i2_repository as ph31
import phase31i_auxiliary_transfer as aux


def test():
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
        def fake_load():return d,{},pd.DataFrame(),splits,'synthetic-frozen'
        old_load=prior.load_train
        old_root_j,old_root_aux,old_root_ph,old_root_prior=j.ROOT,aux.ROOT,ph31.ROOT,prior.ROOT
        pd.read_parquet=parquet_csv
        j.ROOT=root;aux.ROOT=root;ph31.ROOT=root;prior.ROOT=root
        prior.load_train=fake_load
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
            prior.load_train=old_load
            j.ROOT,aux.ROOT,ph31.ROOT,prior.ROOT=old_root_j,old_root_aux,old_root_ph,old_root_prior

if __name__=='__main__':test()
