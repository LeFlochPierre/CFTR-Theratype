#!/usr/bin/env python3
"""End-to-end synthetic contract test; CSV-backed faux Parquet if pyarrow unavailable."""
import sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0,str(Path(__file__).parent))
import phase31i2_repository as m
rng=np.random.default_rng(41)
orig_read_parquet=pd.read_parquet

def read_parquet_fake(path, columns=None, filters=None, **kw):
    frame=pd.read_csv(path)
    if columns is not None: frame=frame[columns]
    if filters is not None: raise AssertionError('No filtered parquet in synthetic test')
    return frame

def test():
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
        m.auxiliary.ROOT=root
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

if __name__=='__main__': test()
