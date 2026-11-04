#!/usr/bin/env python3
"""Phase 28C: CFTR structure-environment biophysics + locked-fold TRAIN ablation.

Reads ONLY the established 210 VTI TRAIN labels, never VAL/TEST outcomes.
8EIG (F508del, elexacaftor-bound) and 7SV7 (tezacaftor-bound) are
experimental CFTR conformations, not variant-specific structures or apo states.
Biophysics features are packing/exposure/contact PROXIES, not predictions of
actual variant delta-delta-G, trafficking or efficacy. All candidates exploratory.

Run from CFTR-Theratype repository root:
  python scripts/phase28c_structure_biophysics.py --self-test
  python scripts/phase28c_structure_biophysics.py --bootstraps 2000

Requires earlier scripts 26H, 27D1, 27D3B, and 26H2 frozen OOF results.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Data.PDBData import residue_sasa_scales
from Bio.PDB import PDBParser, ShrakeRupley
from Bio.SeqIO import read as read_sequence
from Bio.SeqUtils import seq1, seq3
from scipy.spatial import cKDTree

ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'scripts'))
OUT=ROOT/'results/biology/phase28c'
PDB_ROOT=ROOT/'data/external/cftr_structures'
FROZEN=ROOT/'results/biology/phase26h2/phase26h2_oof.csv'
N=1480
SPEC={'8EIG':'8eig','7SV7':'7sv7'}
KEY=['repeat','fold','variant_key','protein_position']
AA='ACDEFGHIKLMNPQRSTVWY'
VOLUME=dict(zip(AA,[88.6,108.5,111.1,138.4,189.9,60.1,153.2,166.7,168.6,166.7,162.9,114.1,112.7,143.8,173.4,89.0,116.1,140.0,227.8,193.6]))
CHARGE={aa:(1 if aa in 'KR' else -1 if aa in 'DE' else 0) for aa in AA}
MAX_SASA=residue_sasa_scales['Wilke']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def domain_for_pos(pos):
    """Published broad sequence-domain proxy (27D1); 'other' is separate group."""
    for name,lo,hi in [('TMD1',81,365),('NBD1',423,646),('R',654,831),
                        ('TMD2',859,1155),('NBD2',1210,1443)]:
        if lo<=pos<=hi:return name
    return 'other'


def environment_from_arrays(positions,xyz,residue_names,sasa,canonical):
    """Pure numerical environment logic, enabling synthetic geometry tests."""
    pp=np.asarray(positions,int); xyz=np.asarray(xyz,float); sasa=np.asarray(sasa,float)
    assert xyz.shape==(len(pp),3) and len(residue_names)==len(pp) and sasa.shape==(len(pp),)
    if len(set(pp))!=len(pp) or not np.isfinite(xyz).all() or not np.isfinite(sasa).all():
        raise ValueError('Invalid protein residue coordinates or solvent-accessible areas')
    tree=cKDTree(xyz)
    neighbors=tree.query_ball_point(xyz,r=12.0)
    domains=np.array([domain_for_pos(int(i)) for i in pp])
    result=[]
    for i,p in enumerate(pp):
        aa=residue_names[i]
        canonical_aa=canonical[p-1]
        # Reject mismatched reference residues (e.g., deposited Q1371 vs E1371).
        if aa!=canonical_aa:continue
        js=np.array([j for j in neighbors[i] if j!=i and abs(int(pp[j])-int(p))>4],dtype=int)
        if len(js):
            d=np.linalg.norm(xyz[js]-xyz[i],axis=1)
            n8=js[d<8.0]
            n12=js[d<12.0]
        else:
            n8=np.array([],int);n12=np.array([],int)
        deep=n8[np.abs(pp[n8]-p)>=24] if len(n8) else n8
        inter=n8[domains[n8]!=domains[i]] if len(n8) else n8
        max_area=float(MAX_SASA.get(seq3(aa).upper(),np.nan))
        if not np.isfinite(max_area):
            raise ValueError(f'No SASA maximum for residue {aa}')
        result.append({'position':int(p),
                       'relative_sasa':float(min(sasa[i]/max_area,2.0)),
                       'nonlocal_ca8':int(len(n8)),
                       'nonlocal_ca12':int(len(n12)),
                       'longrange_ca8':int(len(deep)),
                       'interdomain_ca8':int(len(inter)),
                       'has_ca':1})
    return pd.DataFrame(result)


def load_one_structure(path,canonical,n_sasa_points=100):
    model=next(PDBParser(QUIET=True).get_structure(path.stem,str(path)).get_models())
    chains=sorted(((sum(r.id[0]==' ' for r in c),c) for c in model),reverse=True,key=lambda z:z[0])
    if not chains or chains[0][0]<900 or chains[0][1].id!='A':
        raise ValueError(f'{path}: missing expected chain A with >=900 protein residues')
    chain=chains[0][1]
    residues=[]; positions=[]; xyz=[]; aas=[]; mismatches=[]
    for r in chain:
        if r.id[0]!=' ' or 'CA' not in r:continue
        p=int(r.id[1])
        if not 1<=p<=N:continue
        aa=seq1(r.resname,custom_map={'UNK':'X'})
        if aa!=canonical[p-1]:mismatches.append((p,aa,canonical[p-1]))
        residues.append(r);positions.append(p);xyz.append(np.asarray(r['CA'].coord,float));aas.append(aa)
    if len(set(positions))!=len(positions) or len(positions)<1000:
        raise ValueError(f'{path}: invalid/missing sequence alignment')
    if set(x[0] for x in mismatches)-{1371}:
        raise ValueError(f'{path}: unexpected reference-residue mismatch {mismatches[:10]}')
    if sum(x==canonical[p-1] for x,p in zip(aas,positions))/len(positions)<0.99:
        raise ValueError(f'{path}: canonical agreement below 99%')
    # Compute protein-only exposure: exclude bound corrector, ATP, water, ions,
    # and other heteroatoms so the same definition applies to both structures.
    # This does NOT model the membrane, which is a major caveat for CFTR.
    import copy
    protein_only=copy.deepcopy(chain)
    for residue in list(protein_only):
        if residue.id[0]!=' ':protein_only.detach_child(residue.id)
    ShrakeRupley(n_points=n_sasa_points).compute(protein_only,level='R')
    sasa=np.asarray([float(protein_only[r.id].sasa) for r in residues])
    features=environment_from_arrays(positions,xyz,aas,sasa,canonical)
    # F508del is NOT present in 8EIG; absent coordinates always get NaN features.
    cols=['relative_sasa','nonlocal_ca8','nonlocal_ca12','longrange_ca8','interdomain_ca8','has_ca']
    res=pd.DataFrame({'position':np.arange(1,N+1)}).merge(features,on='position',how='left',validate='one_to_one')
    for col in cols:
        res.rename(columns={col:f'{path.stem.lower()}_{col}'},inplace=True)
    coverage=res[f'{path.stem.lower()}_has_ca'].notna().to_numpy()
    # Position 1371 is excluded because its deposited amino acid is noncanonical.
    if coverage[1370]:raise ValueError('Mismatched residue 1371 was not masked')
    if path.stem.upper()=='8EIG' and coverage[507]:
        raise ValueError('8EIG is F508del but reference position 508 has been retained')
    qc={'structure':path.stem.upper(),'pdb_sha256':digest(path),
        'mapped_ca':len(positions),'accepted_canonical':int(coverage.sum()),
        'first_mismatches':str(mismatches[:10]),'sasa_points':n_sasa_points,
        'site_sasa_definition':'protein-chain-only; ligands/membrane excluded; cryo-EM corrected conformation'}
    return res,qc


def mutation_feature_table(structure,codes,positions):
    """Position features and physically motivated mutation x environment terms."""
    pos=np.asarray(positions,int)
    if len(pos)!=len(codes) or pos.min()<1 or pos.max()>N:
        raise ValueError('Invalid mutation sites')
    base=structure.set_index('position')
    if not set(pos).issubset(base.index):raise ValueError('Missing structural positional rows')
    s=base.loc[pos].copy().reset_index(drop=True)
    s=s.drop(columns=['position'],errors='ignore')
    for key in ('8eig','7sv7'):
        flag=f'{key}_has_ca'
        s[flag]=s[flag].fillna(0.).astype(float)
        for f in ('relative_sasa','nonlocal_ca8','nonlocal_ca12','longrange_ca8','interdomain_ca8'):
            k=f'{key}_{f}'
            # Zero only after observed-flag supplied; missing residue is not truly buried.
            s[k]=s[k].fillna(0.).astype(float)
    context=s.copy()
    extra=[]
    for code,p in zip(codes,pos):
        m=re.fullmatch(r'([A-Z])(\d+)([A-Z])',str(code))
        if m is None or int(m.group(2))!=p or m.group(1) not in VOLUME or m.group(3) not in VOLUME:
            raise ValueError(f'Bad protein substitution: {code} / {p}')
        wt,mut=m.group(1),m.group(3)
        extra.append({'delta_volume':(VOLUME[mut]-VOLUME[wt])/100.,
                      'abs_delta_volume':abs(VOLUME[mut]-VOLUME[wt])/100.,
                      'delta_charge':CHARGE[mut]-CHARGE[wt],
                      'gain_charge':int(CHARGE[wt]==0 and CHARGE[mut]!=0),
                      'gain_proline':int(wt!='P' and mut=='P')})
    change=pd.DataFrame(extra)
    observed=np.column_stack([s[f'{k}_has_ca'].to_numpy() for k in SPEC.values()])
    denom=observed.sum(axis=1)
    # Mean over MODELLED experimental conformations only, zero + availability mask otherwise.
    for col in ('relative_sasa','nonlocal_ca8','longrange_ca8','interdomain_ca8'):
        vals=np.column_stack([s[f'{k}_{col}'].to_numpy() for k in SPEC.values()])
        s[f'mean_{col}']=np.divide(vals.sum(axis=1),denom,
                                  out=np.zeros(len(s),float),where=denom>0)
    s['any_structure_observed']=(denom>0).astype(float)
    buried=np.clip(1.0-s['mean_relative_sasa'].to_numpy(),0.,1.)
    pack=np.log1p(s['mean_nonlocal_ca8'].to_numpy())
    inter=np.log1p(s['mean_interdomain_ca8'].to_numpy())
    interactions=pd.DataFrame({
        'volchange_x_burial':change.abs_delta_volume.to_numpy()*buried*(denom>0),
        'signed_volume_x_packing':change.delta_volume.to_numpy()*pack,
        'charge_gain_x_burial':change.gain_charge.to_numpy()*buried*(denom>0),
        'charge_change_x_burial':change.delta_charge.to_numpy()*buried*(denom>0),
        'proline_gain_x_packing':change.gain_proline.to_numpy()*pack,
        'volchange_x_interdomain':change.abs_delta_volume.to_numpy()*inter,
    })
    # Base chemistry is already represented in prior feature configurations;
    # this ablation tests only physical context and context x mutation interactions.
    if not np.isfinite(context.to_numpy(float)).all() or not np.isfinite(interactions.to_numpy(float)).all():
        raise ValueError('NaN/Inf in engineered structural features')
    return context,interactions,observed


def position_mae(frame,true,pred):
    t=frame[['protein_position',true,pred]].copy()
    t['ae']=(t[true]-t[pred]).abs()
    return float(t.groupby('protein_position').ae.mean().mean())


def paired_mae_gain(frame,new,old,target,nboot,seed):
    t=frame[['repeat','protein_position',new,old,target]].copy()
    t['gain']=(t[old]-t[target]).abs()-(t[new]-t[target]).abs()
    each=(t.groupby(['repeat','protein_position']).gain.mean()
          .groupby('protein_position').mean().to_numpy(float))
    rng=np.random.default_rng(seed)
    sample=rng.choice(each,size=(nboot,len(each)),replace=True).mean(axis=1)
    return {'candidate':new,'reference':old,'target':target,'gain_pp':float(each.mean()),
            'ci95_low':float(np.percentile(sample,2.5)),
            'ci95_high':float(np.percentile(sample,97.5)),
            'positions':len(each)}


def self_test():
    # Small toy canonical protein with synthetic residue geometry.
    canonical='A'*N
    x=np.array([[0.,0.,0.],[5.,0.,0.],[7.,0.,0.],[30.,0.,0.]],float)
    pp=[10,30,55,140]
    f=environment_from_arrays(pp,x,['A']*4,[20.,30.,40.,70.],canonical)
    assert len(f)==4 and int(f.iloc[0].longrange_ca8)==1
    assert int(f.iloc[0].nonlocal_ca8)==2
    # Missing positions distinguishable from truly low exposure/packing.
    two=pd.DataFrame({'position':np.arange(1,N+1)})
    for code in SPEC.values():
        for field in ['relative_sasa','nonlocal_ca8','nonlocal_ca12','longrange_ca8','interdomain_ca8','has_ca']:
            two[f'{code}_{field}']=np.nan
        two.loc[two.position==10,f'{code}_relative_sasa']=0.25
        two.loc[two.position==10,f'{code}_nonlocal_ca8']=5
        two.loc[two.position==10,f'{code}_nonlocal_ca12']=7
        two.loc[two.position==10,f'{code}_longrange_ca8']=2
        two.loc[two.position==10,f'{code}_interdomain_ca8']=1
        two.loc[two.position==10,f'{code}_has_ca']=1
    ctx,inter,obs=mutation_feature_table(two,['A10V','A10P','A15G'],[10,10,15])
    assert ctx.shape==(3,12) and inter.shape==(3,6)
    assert (obs[2]==0).all() and ctx.iloc[2].sum()==0
    assert inter.loc[0,'signed_volume_x_packing']>0
    assert np.allclose(ctx.iloc[0],ctx.iloc[1])
    data=pd.DataFrame({'repeat':[1,1,2,2],'protein_position':[1,2,1,2],
                       'true':[1.,3.,1.,3.],'original':[2.,2.,2.,2.],
                       'candidate':[1.,3.,1.,3.]})
    gain=paired_mae_gain(data,'candidate','original','true',500,28)
    assert np.isclose(gain['gain_pp'],1.)
    print('PASS synthetic: contact geometry, missing-coverage masks, mutation interactions, clustered bootstrap')


def run(args):
    from cftr_theratype.datasets.eti_auxiliary import load_data
    from cftr_theratype.features.mutation_codes import extract_codes
    from cftr_theratype.models.regressors import phase26d_model
    from cftr_theratype.models.auxiliary import SPEC as FROZEN_XGB_SPEC
    import phase27d3b_structure_ablation as d3
    if not FROZEN.is_file():raise FileNotFoundError(f'Missing frozen OOF: {FROZEN}')
    d3.verify_provenance()
    v,x80,_,_,folds=load_data(ROOT)     # TRAIN partition only
    xx=np.asarray(x80,float); pp=v.protein_position.to_numpy(int)
    keys=v.variant_key.astype(str).to_numpy(); yc=v.corrector_rescue_pp.to_numpy(float)
    yt=v.vti_rescue_pp.to_numpy(float)
    if xx.shape!=(210,80) or len(set(keys))!=210 or len(np.unique(pp))!=168 or len(folds)!=10:
        raise ValueError('Frozen VTI TRAIN or outer-fold metadata changed')
    if not np.isfinite(xx[:,:68]).all() or np.isinf(xx).any():
        raise ValueError('Invalid original sequence/structural feature values')
    code=extract_codes(v,pp)
    canonical=str(read_sequence(str(PDB_ROOT/'P13569.fasta'),'fasta').seq).upper()
    if len(canonical)!=N:raise ValueError('Unexpected UniProt P13569 sequence length')
    for c,p in zip(code,pp):
        m=re.fullmatch(r'([A-Z])(\d+)([A-Z])',c)
        if not m or int(m.group(2))!=p or canonical[p-1]!=m.group(1):
            raise ValueError(f'Mutation/UniProt identity mismatch {c} at {p}')
    print('='*108)
    print('PHASE 28C | STRUCTURAL PACKING + EXPOSURE + BIOPHYSICAL MUTATION INTERACTIONS')
    print('='*108)
    print('TRAIN ONLY 210 records / 168 positions; 2x5 frozen position-disjoint folds.')
    print('VAL/TEST NOT READ; all new model comparisons exploratory on development folds.')
    print('Protein-only SASA and residue geometry measured in ligand-bound conformations; NOT mutation ΔΔG.')
    parts=[];infos=[]
    for pdb in SPEC:
        path=PDB_ROOT/f'{pdb}.pdb'
        if not path.is_file():raise FileNotFoundError(path)
        part,info=load_one_structure(path,canonical,n_sasa_points=args.sasa_points)
        parts.append(part);infos.append(info)
        print(f'Geometry {pdb}: mapped CA={info["mapped_ca"]}, canonical retained={info["accepted_canonical"]}, sha256={info["pdb_sha256"][:16]}...')
    struct=parts[0].merge(parts[1],on='position',validate='one_to_one')
    context,interaction,obs=mutation_feature_table(struct,code,pp)
    print('\nA. STRUCTURE COVERAGE / FEATURE QC')
    print('Both structures modelled:',int(np.logical_and(obs[:,0],obs[:,1]).sum()),
          'one only:',int((obs.sum(axis=1)==1).sum()),
          'neither:',int((obs.sum(axis=1)==0).sum()))
    print('Engineered features: context=',context.shape[1],'mutation x context=',interaction.shape[1])
    print('Context relative SASA / packing distribution (only observed):')
    for site in SPEC.values():
        f=f'{site}_has_ca'; selected=context.loc[context[f]>0]
        print(site,'n=',len(selected),'sasa median=',round(float(selected[f'{site}_relative_sasa'].median()),3),
              'CA8 contacts median=',round(float(selected[f'{site}_nonlocal_ca8'].median()),2),
              'interdomain contacts median=',round(float(selected[f'{site}_interdomain_ca8'].median()),2))
    OUT.mkdir(parents=True,exist_ok=True)
    struct.to_csv(OUT/'phase28c_position_environment.csv',index=False)
    feat=pd.concat([v[['variant_key','protein_position']].reset_index(drop=True),
                    pd.Series(code,name='mutation'),context.add_prefix('ctx_'),
                    interaction.add_prefix('int_')],axis=1)
    feat.to_csv(OUT/'phase28c_train_biophysics_features.csv',index=False)
    pd.DataFrame(infos).to_csv(OUT/'phase28c_structure_provenance.csv',index=False)
    frozen=pd.read_csv(FROZEN,dtype={'variant_key':str})
    if len(frozen)!=420 or frozen.duplicated(KEY).any():
        raise ValueError('Bad frozen Phase26H2 reference')
    needed=set(KEY+['vti_fixed','vti_fixed_total','observed_corrector','observed_total'])
    if not needed.issubset(frozen):raise ValueError('Missing frozen OOF columns: '+str(needed-set(frozen)))
    configs={'global':xx,'context':np.column_stack([xx,context.to_numpy(float)]),
             'context_interaction':np.column_stack([xx,context.to_numpy(float),interaction.to_numpy(float)])}
    rows=[];qc=[]
    for i,(repeat,fold,tr_keys,te_keys) in enumerate(folds,1):
        tr=np.flatnonzero(np.isin(keys,list(tr_keys)))
        te=np.flatnonzero(np.isin(keys,list(te_keys)))
        if len(tr)+len(te)!=210 or set(pp[tr])&set(pp[te]):
            raise ValueError(f'Position leakage in repeat {repeat} fold {fold}')
        s=frozen.loc[(frozen.repeat==repeat)&(frozen.fold==fold)].set_index('variant_key')
        if set(s.index)!=set(keys[te]):raise ValueError('Frozen fold membership mismatch')
        s=s.loc[keys[te]]
        if not np.allclose(s.observed_corrector.to_numpy(float),yc[te],atol=1e-8):
            raise ValueError('Corrector label mismatch between current TRAIN and reference')
        if not np.allclose(s.observed_total.to_numpy(float),yt[te],atol=1e-8):
            raise ValueError('Total label mismatch between current TRAIN and reference')
        acute=s.vti_fixed_total.to_numpy(float)-s.vti_fixed.to_numpy(float)
        fits={}
        for name,matrix in configs.items():
            a=np.column_stack([matrix,np.ones(len(matrix))])
            mdl=phase26d_model(FROZEN_XGB_SPEC)
            mdl.fit(a[tr],yc[tr])
            fits[name]=np.asarray(mdl.predict(a[te]),float)
        diff=float(np.max(np.abs(fits['global']-s.vti_fixed.to_numpy(float))))
        if diff>1e-5:
            raise ValueError(f'FROZEN BASELINE PARITY FAILURE repeat={repeat} fold={fold}: max error={diff}')
        qc.append({'repeat':repeat,'fold':fold,'train_records':len(tr),'test_records':len(te),
                   'train_positions':len(set(pp[tr])),'test_positions':len(set(pp[te])),
                   'global_vs_frozen_max_abs':diff})
        for j,idx in enumerate(te):
            row={'repeat':repeat,'fold':fold,'variant_key':keys[idx],
                 'protein_position':int(pp[idx]),'observed_corrector':float(yc[idx]),
                 'observed_total':float(yt[idx]),'frozen_acute':float(acute[j]),
                 'regime':'low' if yc[idx]<10 else 'high' if yc[idx]>40 else 'middle'}
            for name,yhat in fits.items():
                row['corrector_'+name]=float(yhat[j]);row['total_'+name]=float(yhat[j]+acute[j])
            rows.append(row)
        print(f'completed {i}/10 repeat={repeat} fold={fold} parity={diff:.2g}',flush=True)
    oof=pd.DataFrame(rows)
    if len(oof)!=420 or oof.duplicated(KEY).any():raise ValueError('OOF count/cardinality mismatch')
    qc=pd.DataFrame(qc)
    if qc.global_vs_frozen_max_abs.max()>1e-5:raise ValueError('Baseline parity failed')
    metrics=[];groupmetrics=[];gains=[]
    for target in ['corrector','total']:
        for name in configs:
            pred=target+'_'+name; actual='observed_'+target
            metrics.append({'target':target,'config':name,
                            'position_mae_pp':np.mean([position_mae(f,actual,pred) for _,f in oof.groupby('repeat')]),
                            'variant_mae_pp':float(np.mean(np.abs(oof[pred]-oof[actual]))),
                            'signed_bias_pp':float(np.mean(oof[pred]-oof[actual]))})
            if target=='corrector':
                for regime in ['low','middle','high']:
                    sub=oof[oof.regime==regime]
                    groupmetrics.append({'regime':regime,'config':name,'n_variants':sub.variant_key.nunique(),
                                         'positions':sub.protein_position.nunique(),
                                         'position_mae_pp':np.mean([position_mae(f,actual,pred) for _,f in sub.groupby('repeat')]),
                                         'signed_bias_pp':float(np.mean(sub[pred]-sub[actual]))})
        for k,name in enumerate(['context','context_interaction']):
            gains.append(paired_mae_gain(oof,target+'_'+name,target+'_global','observed_'+target,
                                         args.bootstraps,seed=28010+(0 if target=='corrector' else 10)+k))
    oof.to_csv(OUT/'phase28c_repeated_oof.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'phase28c_mae.csv',index=False)
    pd.DataFrame(groupmetrics).to_csv(OUT/'phase28c_regime_mae.csv',index=False)
    pd.DataFrame(gains).to_csv(OUT/'phase28c_mae_bootstrap_gains.csv',index=False)
    qc.to_csv(OUT/'phase28c_fold_qc.csv',index=False)
    manifest={'pdb_hashes':{x['structure']:x['pdb_sha256'] for x in infos},
              'script_sha256':digest(Path(__file__)),'frozen_oof_sha256':digest(FROZEN),
              'n_train':210,'unique_positions':168,'n_folds':10,
              'source_warning':'drug-bound experimental CFTR structures; not mutant or untreated structures',
              'predefined_configs':list(configs),'bootstraps':args.bootstraps}
    (OUT/'phase28c_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('\nB. GLOBAL BASELINE VS BIOPHYSICAL REGRESSORS (lower MAE is better)')
    print(pd.DataFrame(metrics).round(4).to_string(index=False))
    print('\nC. CORRECTOR MAE BY TRUE REGIME (diagnostic, TRAIN OOF only)')
    print(pd.DataFrame(groupmetrics).round(4).to_string(index=False))
    print('\nD. PAIRED POSITION-CLUSTERED MAE GAINS (positive = BETTER)')
    print(pd.DataFrame(gains).round(4).to_string(index=False))
    print('\nE. FROZEN BASELINE PARITY')
    print(qc.to_string(index=False))
    print('PASS Phase28C complete. NO VAL/TEST read; development comparisons exploratory.')
    print('Outputs:',OUT)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--sasa-points',type=int,default=100)
    ap.add_argument('--bootstraps',type=int,default=2000)
    args=ap.parse_args()
    if args.self_test:self_test()
    else:
        if args.sasa_points<50 or args.bootstraps<100:ap.error('Need >=50 SASA points and >=100 bootstraps')
        run(args)
