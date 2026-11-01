#!/usr/bin/env python3
"""Phase 31I.2: position-safe multi-assay transfer using Phase26D/26G infrastructure.

Run from the CFTR-Theratype repository root after putting this script alongside
phase26d_corrector_bottleneck.py, phase26g_feature_model_ablation.py and
phase31i_auxiliary_transfer.py in scripts/.

  python -u scripts/phase31i2_repository.py audit
  python -u scripts/phase31i2_repository.py run --quick
  python -u scripts/phase31i2_repository.py run

AUDIT: never loads VTI response values. RUN: prior.load_train() loads ONLY VTI TRAIN
responses through the predicate-filtered Phase26D loader. No validation/test
outcomes are read. Evaluates in-vitro rescue, not clinical treatment efficacy.

M0: Phase26G xgb_80 mutation-only baseline, with exact historical parity check.
M1: M0 + cross-fitted Bihler 2024 ETI teacher prediction.
M2: M1 + separately cross-fitted Castellani P289 and U032 ETI predictions.
M3: M2 + P289/U032 baseline and treated S6 maturation teacher predictions.
M3_perm: same M3 structure, but auxiliary labels permuted by protein position
         within each assay and outer fold (negative control).

Teachers use a deliberately SIMPLE mutation-derived representation available
for all amino-acid substitutions (identity, physicochemistry, position). The
VTI downstream estimator uses the unchanged frozen Phase26G 80-feature matrix.
Thus these are a controlled pilot, NOT a final test of learned PLM features.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))
import phase26d_corrector_bottleneck as prior
import phase26g_feature_model_ablation as phase26g
import phase31i_auxiliary_transfer as auxiliary

OUT = ROOT / 'results/biology/phase31i'
AA = 'ACDEFGHIKLMNPQRSTVWY'
# Scalar physicochemical descriptors; fixed a priori, independent of outcomes.
HYDRO = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
MASS = {'A':89.09,'C':121.15,'D':133.10,'E':147.13,'F':165.19,'G':75.07,
        'H':155.16,'I':131.17,'K':146.19,'L':131.17,'M':149.21,'N':132.12,
        'P':115.13,'Q':146.15,'R':174.20,'S':105.09,'T':119.12,'V':117.15,
        'W':204.23,'Y':181.19}
CHARGE = {a: (-1 if a in 'DE' else (1 if a in 'KR' else 0)) for a in AA}
AROM = {a: int(a in 'FWY') for a in AA}
PROLINE = {a: int(a == 'P') for a in AA}

TEACHERS = ('bihler','p289','u032','p289_s6_baseline','p289_s6_treated','u032_s6_baseline','u032_s6_treated')
MODELS = {'M0': (), 'M1': TEACHERS[:1], 'M2': TEACHERS[:3], 'M3': TEACHERS,
          'M3_perm': TEACHERS}


def mutation_features(aa_codes: pd.Series) -> np.ndarray:
    """Common mutation-only teacher representation for ALL auxiliary/VTI variants.

    No sequence embeddings or assay-derived values are included. Deliberately
    modest features make this a proof-of-concept, not a definitive multi-omics test.
    """
    rows = []
    for value in aa_codes.astype(str):
        m = re.fullmatch(r'([A-Z])(\d{1,4})([A-Z])', value)
        if m is None or m.group(1) not in AA or m.group(3) not in AA:
            raise ValueError(f'Only canonical missense substitutions supported: {value!r}')
        w, pos, a = m.group(1), int(m.group(2)), m.group(3)
        if not 1 <= pos <= 1480 or w == a:
            raise ValueError(f'Invalid mutation: {value!r}')
        wvec = [float(w == x) for x in AA]
        avec = [float(a == x) for x in AA]
        phys = []
        for prop in (HYDRO, MASS, CHARGE, AROM, PROLINE):
            phys.extend([prop[w], prop[a], prop[a] - prop[w]])
        rel = pos / 1480.0
        loc = [rel, rel**2, np.sin(2*np.pi*rel), np.cos(2*np.pi*rel),
               np.sin(6*np.pi*rel), np.cos(6*np.pi*rel)]
        rows.append(wvec + avec + phys + loc)
    return np.asarray(rows, dtype=float)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_sources(args):
    paths = {
        'bihler': ROOT / args.bihler,
        's5': auxiliary.resolve_file(args.s5, 'phase31a_castellani_s5_transport.csv'),
        's6': auxiliary.resolve_file(args.s6, 'phase31a_castellani_s6_processing.csv'),
        'splits2024': ROOT / args.splits2024,
        'vti_meta': ROOT / args.vti_meta,
    }
    for k, v in paths.items():
        if not v.is_file():
            raise FileNotFoundError(f'Required {k} input missing: {v}')
    return paths


def protected_and_eligible_positions(paths):
    """Partition metadata only, never VTI response values."""
    bsplit = pd.read_parquet(paths['splits2024'], columns=['variant_key','hgvs_protein','protein_position','split'])
    if not bsplit.split.astype(str).str.lower().isin(('train','val','test')).all():
        raise ValueError('Unknown Bihler 2024 split label')
    bp = pd.to_numeric(bsplit.protein_position, errors='coerce')
    bsplit = bsplit.assign(protein_position=bp)
    pos2024_train = set(bsplit.loc[bsplit.split.astype(str).str.lower().eq('train'), 'protein_position'].dropna().astype(int))
    pos2024_protected = set(bsplit.loc[bsplit.split.astype(str).str.lower().isin(['val','test']), 'protein_position'].dropna().astype(int))
    if pos2024_train & pos2024_protected:
        raise AssertionError('Bihler 2024 protein positions cross frozen split boundary')
    # The 2026 split metadata can describe several studies, but every VAL/TEST
    # protein position is conservatively barred from all auxiliary teacher fits.
    vm = pd.read_parquet(paths['vti_meta'], columns=['variant_key','protein_position','transfer_split'])
    vc = pd.to_numeric(vm.protein_position, errors='coerce')
    vm = vm.assign(protein_position=vc)
    if not vm.transfer_split.astype(str).str.lower().isin(('train','val','test','excluded')).all():
        raise ValueError('Unrecognised transfer_split in 2026 VTI metadata')
    pos2026_protected = set(vm.loc[vm.transfer_split.astype(str).str.lower().isin(['val','test']), 'protein_position'].dropna().astype(int))
    protected = pos2024_protected | pos2026_protected
    if not pos2024_train:
        raise ValueError('No Bihler 2024 TRAIN protein positions')
    print(f'Protected positions (2024 val/test union 2026 VTI val/test): {len(protected)}')
    print(f'Bihler 2024 TRAIN positions: {len(pos2024_train)}')
    return pos2024_train, protected


def load_auxiliary(paths, eligible_bihler, protected):
    bih = pd.read_parquet(paths['bihler'], columns=['variant_key','hgvs_protein','eti_rescue_pp'])
    s5 = pd.read_csv(paths['s5'])
    s6 = pd.read_csv(paths['s6'])
    tables, audits = {}, []
    for name, src, field, study in (
        ('bihler',bih,'eti_rescue_pp',None),
        ('p289',s5,'eti_rescue_pp','P289'),
        ('u032',s5,'eti_rescue_pp','U032'),
        ('p289_s6_baseline',s6,'baseline_mature_fraction','P289'),
        ('p289_s6_treated',s6,'eti_mature_fraction','P289'),
        ('u032_s6_baseline',s6,'baseline_mature_fraction','U032'),
        ('u032_s6_treated',s6,'eti_mature_fraction','U032')):
        tab, rep = auxiliary.table_aux(src, name, field, study)
        # Canonical point mutations only. Double check no external positions
        # from frozen protected partitions are in this data.
        before = len(tab)
        tab = tab.loc[~tab.pos31i2.isin(protected)].copy()
        if name == 'bihler':
            tab = tab.loc[tab.pos31i2.isin(eligible_bihler)].copy()
        tab = tab.loc[tab.aa31i.map(lambda x: bool(re.fullmatch(r'[ACDEFGHIKLMNPQRSTVWY]\d{1,4}[ACDEFGHIKLMNPQRSTVWY]',x)))].copy()
        if tab.empty:
            raise ValueError(f'No eligible auxiliary labels for {name}; see split filtering')
        if tab.aa31i.duplicated().any():
            raise AssertionError(f'Duplicate exact substitution after canonicalisation: {name}')
        assert set(tab.pos31i2.astype(int)).isdisjoint(protected)
        tables[name] = tab.reset_index(drop=True)
        audits.append(dict(assay=name, rows_source=rep['source_rows'],
                           unique_substitutions_before_filter=before,
                           substitutions_eligible=len(tab),positions_eligible=tab.pos31i2.nunique(),
                           rows_removed_protected_or_split=before-len(tab)))
    return tables, pd.DataFrame(audits)


def audit(paths, aux, coverage, eligible_bihler, protected, out):
    vt_meta = pd.read_parquet(paths['vti_meta'], columns=['variant_key','hgvs_protein','protein_position','transfer_split'])
    vt = auxiliary.with_ids(vt_meta.loc[vt_meta.transfer_split.astype(str).str.lower().eq('train')],
                            'VTI metadata (TRAIN only)')
    vt_pos = set(vt.pos31i2.dropna().astype(int))
    vt_aa = set(vt.aa31i.dropna().astype(str))
    all_bih = set(aux['bihler'].pos31i2.astype(int))
    stat = coverage.copy()
    stat['positions_in_vti_train'] = stat.assay.map({k:len(set(v.pos31i2.astype(int))&vt_pos) for k,v in aux.items()})
    stat['exact_substitutions_in_vti_train'] = stat.assay.map({k:len(set(v.aa31i)&vt_aa) for k,v in aux.items()})
    stat['additional_positions_vs_bihler_eti'] = stat.assay.map({k:len(set(v.pos31i2.astype(int))-all_bih) for k,v in aux.items()})
    agreements=[]
    for a,b in [('bihler','p289'),('bihler','u032'),('p289','u032')]:
        res,_=auxiliary.agreement(aux[a],aux[b],a,b)
        agreements.append(res)
    stat.to_csv(out/'phase31i2_repo_coverage.csv',index=False)
    pd.DataFrame(agreements).to_csv(out/'phase31i2_repo_agreement.csv',index=False)
    manifest={'phase':'31I.2','stage':'audit','vti_outcomes_loaded':False,
              'bihler_2024_eligible_positions':len(eligible_bihler),
              'protected_positions':len(protected),
              'sources':{k:{'path':str(v),'sha256':sha(v)} for k,v in paths.items()},
              'teacher_representation':'canonical missense substitution: AA one-hot, physicochemistry, protein position'}
    (out/'phase31i2_repo_audit.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('\nAUXILIARY COVERAGE AFTER FROZEN PARTITION FILTERS (not raw Phase31I.1 coverage):')
    print(stat.to_string(index=False))
    print('\nMATCHED ASSAY AGREEMENT (substitution-level):')
    print(pd.DataFrame(agreements).round(3).to_string(index=False))
    return manifest


def make_teacher(alpha):
    return Pipeline([('impute',SimpleImputer(strategy='median')),
                     ('scale',StandardScaler()),('ridge',Ridge(alpha=alpha))])


def teacher_predictions(table, train_aa, train_pos, held_aa, held_pos,
                        outer_held, alpha, negative_seed=None, inner_folds=5):
    """No teacher sees labels at outer held positions or inner cross-fit groups."""
    if set(table.pos31i2.astype(int)) & set(outer_held):
        raise AssertionError('Outer held-out position in teacher data')
    src=table.copy()
    if negative_seed is not None:
        src=auxiliary.permute_by_position(src,np.random.default_rng(negative_seed))
    src_pos=src.pos31i2.astype(int).to_numpy()
    xs=mutation_features(src.aa31i)
    yt=src.value31i.to_numpy(float)
    xp=mutation_features(pd.Series(train_aa))
    xq=mutation_features(pd.Series(held_aa))
    if len(np.unique(src_pos)) < 8:
        raise ValueError('Too few teacher source positions')
    k=min(inner_folds,len(np.unique(train_pos)))
    if k<2:
        raise ValueError('Too few VTI training positions to cross-fit')
    oof=np.full(len(train_aa),np.nan)
    for itr,iva in GroupKFold(n_splits=k).split(xp,groups=train_pos):
        forbidden=set(np.asarray(train_pos)[iva].tolist()) | set(outer_held)
        use=~np.isin(src_pos,list(forbidden))
        if len(set(src_pos[use])) < 8:
            raise ValueError('Too few eligible teacher positions after inner group exclusion')
        assert not set(src_pos[use]) & forbidden
        teacher=make_teacher(alpha)
        teacher.fit(xs[use],yt[use])
        oof[iva]=teacher.predict(xp[iva])
    if not np.isfinite(oof).all():
        raise AssertionError('Crossfit teacher left missing predictions')
    teacher=make_teacher(alpha)
    teacher.fit(xs,yt)
    pred=teacher.predict(xq)
    return oof,pred


def read_frozen_choices(target, folds, args):
    path=ROOT/f'results/biology/phase26g/phase26g_{target}_selected.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing Phase26G frozen parameter selections: {path}')
    d=pd.read_csv(path)
    d=d[d.model=='xgb_80']
    mapping={}
    for row in d.itertuples(index=False):
        key=(int(row.repeat),int(row.fold))
        if key in mapping:
            raise ValueError(f'Duplicate Phase26G selection: {key}')
        choice=ast.literal_eval(str(row.selected_param))
        if not isinstance(choice,tuple) or len(choice)!=3 or choice not in phase26g.PARAMS['xgb']:
            raise ValueError(f'Invalid Phase26G xgb_80 hyperparameters: {choice!r}')
        mapping[key]=choice
    missing=[(int(r),int(f)) for r,f,*_ in folds if (int(r),int(f)) not in mapping]
    if missing:
        raise ValueError(f'Missing matching Phase26G xgb_80 selections for {target}: {missing[:6]}')
    return mapping


def verify_historical_m0(frame, target, tol=1e-4):
    path=ROOT/f'results/biology/phase26g/phase26g_{target}_oof.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing mandatory M0 parity reference: {path}')
    ref=pd.read_csv(path,usecols=['repeat','fold','variant_key','protein_position','observed','xgb_80'])
    x=frame[frame.model=='M0'].merge(ref,on=['repeat','fold','variant_key','protein_position'],
                                     validate='one_to_one',how='left',indicator=True,suffixes=('','_historical'))
    if len(x)!=len(ref) or not x._merge.eq('both').all() or not np.allclose(x.observed,x.observed_historical,atol=1e-7):
        raise AssertionError(f'Historical Phase26G M0 OOF alignment failure ({target})')
    maxdiff=float(np.max(np.abs(x.prediction-x.xgb_80)))
    if maxdiff>tol or not np.isfinite(maxdiff):
        raise AssertionError(f'Phase26G M0 prediction parity failed for {target}: max diff={maxdiff:.7g}')
    return maxdiff


def position_mae(d):
    q=d.assign(err=np.abs(d.observed-d.prediction))
    return float(q.groupby('protein_position').err.mean().mean())


def summarize_and_bootstrap(oof, nboot, seed):
    metrics=[]
    for (t,m,r),d in oof.groupby(['target','model','repeat']):
        y=d.observed.to_numpy(float); p=d.prediction.to_numpy(float)
        rho=float(spearmanr(y,p).statistic) if np.unique(y).size>1 and np.unique(p).size>1 else np.nan
        metrics.append(dict(target=t,model=m,repeat=r,position_mae_pp=position_mae(d),
                            variant_mae_pp=float(np.mean(np.abs(y-p))),
                            rmse_pp=float(np.sqrt(np.mean((y-p)**2))),spearman=rho))
    met=pd.DataFrame(metrics)
    gains=[]
    # Group by (repeat,protein_position) then average repeats per actual position.
    # Repeat folds are NOT independent statistical units.
    for target,dd in oof.groupby('target'):
        base=dd[dd.model=='M0'][['repeat','variant_key','protein_position','observed','prediction']].rename(columns={'prediction':'p0'})
        for mod in ('M1','M2','M3','M3_perm'):
            alt=dd[dd.model==mod][['repeat','variant_key','protein_position','prediction']].rename(columns={'prediction':'px'})
            x=base.merge(alt,on=['repeat','variant_key','protein_position'],how='inner',validate='one_to_one')
            if len(x)!=len(base):
                raise AssertionError(f'Incomplete OOF coverage for {target}/{mod}')
            x['gain']=np.abs(x.observed-x.p0)-np.abs(x.observed-x.px)
            cluster=(x.groupby(['repeat','protein_position']).gain.mean()
                     .groupby('protein_position').mean().to_numpy())
            rng=np.random.default_rng(seed)
            draw=rng.integers(0,len(cluster),size=(nboot,len(cluster)))
            boot=np.mean(cluster[draw],axis=1)
            gains.append(dict(target=target,model=mod,reference='M0',
                              gain_position_mae_pp=float(cluster.mean()),
                              ci95_low=float(np.quantile(boot,.025)),
                              ci95_high=float(np.quantile(boot,.975)),
                              n_independent_positions=len(cluster)))
    return met,pd.DataFrame(gains)


def run(args, tables, manifest, out):
    # Canonical loader: enforced TRAIN-only VTI outcomes and frozen historical folds.
    d,features,historical,folds,source=prior.load_train()
    assert len(d)==210 and d.protein_position.nunique()==168
    keys=d.variant_key.astype(str).to_numpy()
    positions=d.protein_position.to_numpy(int)
    # The exact protein HGVS comes from split metadata, not an assumed column
    # on model_zoo.prepare's dataframe. Read IDENTIFIERS only.
    meta=pd.read_parquet(ROOT/'data/processed/bihler_2026_transfer_splits.parquet',
                         columns=['variant_key','hgvs_protein','protein_position','transfer_split'])
    meta=meta.loc[meta.transfer_split.astype(str).str.lower().eq('train')].copy()
    if meta.variant_key.duplicated().any():
        raise ValueError('Nonunique TRAIN variants in 2026 split metadata')
    ids=d[['variant_key','protein_position']].merge(meta[['variant_key','hgvs_protein','protein_position']],
                 on='variant_key',how='left',validate='one_to_one',suffixes=('','_meta'))
    if ids.hgvs_protein.isna().any() or not (ids.protein_position==ids.protein_position_meta).all():
        raise ValueError('Cannot align Phase26D TRAIN variants and 2026 protein HGVS metadata')
    codes=auxiliary.with_ids(ids[['variant_key','protein_position','hgvs_protein']],
                             'VTI TRAIN',strict=True).aa31i.to_numpy(str)
    x80=np.asarray(features['sequence_structure'],float)
    assert x80.shape==(210,80),f'Expected frozen 80 features, got {x80.shape}'
    if np.isinf(x80).any():
        raise ValueError('Unexpected infinity in frozen 80 features (Phase26G converts inf -> NaN); inspect')
    if set(positions) & set(manifest['protected_positions_list']):
        raise ValueError('VTI TRAIN and protected position sets overlap; inspect partitions before modelling')
    print(f'VTI TRAIN={len(keys)} variants / {len(set(positions))} positions. Frozen OOF source={source}')
    print(f'Independent teacher representation: {mutation_features(pd.Series(codes[:1])).shape[1]} mutation-only descriptors (no assays)')
    targets={'corrector':d.corrector_rescue_pp.to_numpy(float),'total':d.vti_rescue_pp.to_numpy(float)}
    selections={t:read_frozen_choices(t,folds,args) for t in targets}
    allrows=[]; teacher_audit=[]
    foldlist=folds[:1] if args.quick else folds
    for fi,(repeat,fold,training_keys,validation_keys) in enumerate(foldlist,1):
        tr=np.flatnonzero(np.isin(keys,list(training_keys)))
        te=np.flatnonzero(np.isin(keys,list(validation_keys)))
        held=set(positions[te].tolist())
        assert set(positions[tr]).isdisjoint(held)
        teacher_tr={}; teacher_te={}; perm_tr={}; perm_te={}
        for j,name in enumerate(TEACHERS):
            tab=tables[name]
            effective=tab.loc[~tab.pos31i2.isin(held)].copy()
            assert set(effective.pos31i2.astype(int)).isdisjoint(held)
            tt,tv=teacher_predictions(effective,codes[tr],positions[tr],codes[te],positions[te],held,
                                      args.teacher_alpha,inner_folds=args.teacher_folds)
            teacher_tr[name],teacher_te[name]=tt,tv
            pt,pv=teacher_predictions(effective,codes[tr],positions[tr],codes[te],positions[te],held,
                                      args.teacher_alpha,negative_seed=args.seed+1009*fi+31*j,
                                      inner_folds=args.teacher_folds)
            perm_tr[name],perm_te[name]=pt,pv
            teacher_audit.append(dict(repeat=repeat,fold=fold,teacher=name,
                                      available_rows=len(tab),eligible_rows=len(effective),
                                      eligible_positions=effective.pos31i2.nunique(),
                                      excluded_outer_rows=len(tab)-len(effective)))
        for target,y in targets.items():
            spec=selections[target][(int(repeat),int(fold))]
            for mod,required in MODELS.items():
                useperm=(mod=='M3_perm')
                a=perm_tr if useperm else teacher_tr
                b=perm_te if useperm else teacher_te
                xt=np.column_stack([x80[tr]]+[a[k] for k in required]) if required else x80[tr]
                xv=np.column_stack([x80[te]]+[b[k] for k in required]) if required else x80[te]
                estimator=phase26g.make_model('xgb',spec)
                estimator.fit(xt,y[tr])
                pred=estimator.predict(xv)
                if not np.isfinite(pred).all():
                    raise ValueError(f'Nonfinite predictions: {target}/{mod}/{repeat}/{fold}')
                for i,p in zip(te,pred):
                    allrows.append(dict(target=target,model=mod,repeat=int(repeat),fold=int(fold),
                                        variant_key=keys[i],protein_position=int(positions[i]),
                                        observed=float(y[i]),prediction=float(p)))
        print(f'Completed outer {fi}/{len(foldlist)}: repeat={repeat} fold={fold} held positions={len(held)}',flush=True)
    pred=pd.DataFrame(allrows)
    pd.DataFrame(teacher_audit).to_csv(out/'phase31i2_repo_teacher_filter.csv',index=False)
    if args.quick:
        pred.to_csv(out/'phase31i2_repo_quick_oof.csv',index=False)
        print('PASS quick preflight (only one outer fold; no pooled performance conclusions)')
        return
    for target in targets:
        selected=pred[pred.target==target]
        refkeys=set(keys)
        for mod in MODELS:
            z=selected[selected.model==mod]
            for repeat,g in z.groupby('repeat'):
                if set(g.variant_key)!=refkeys or g.variant_key.duplicated().any():
                    raise AssertionError(f'Incomplete OOF: {target}/{mod}/{repeat}')
        print(f'Phase26G M0 parity max_abs[{target}] = {verify_historical_m0(selected,target):.9g}')
    # Do not publish final OOF results before historical parity succeeds.
    pred.to_csv(out/'phase31i2_repo_oof.csv',index=False)
    met,comp=summarize_and_bootstrap(pred,args.bootstrap,args.seed)
    met.to_csv(out/'phase31i2_repo_metrics_by_repeat.csv',index=False)
    comp.to_csv(out/'phase31i2_repo_paired_bootstrap.csv',index=False)
    run_manifest={k:v for k,v in manifest.items() if k!='protected_positions_list'}
    run_manifest.update({'stage':'run','vti_train_outcomes_loaded':True,
                         'vti_val_test_outcomes_loaded':False,
                         'n_vti_train_variants':len(d),'n_vti_train_positions':len(set(positions)),
                         'n_outer_folds':len(folds),'frozen_fold_source':source,
                         'teacher_alpha':args.teacher_alpha,'teacher_inner_folds':args.teacher_folds,
                         'downstream':'Phase26G xgb_80, frozen per-target per-outer-fold choices',
                         'evaluation':'position-weighted MAE; position-block paired bootstrap averaged over repeats',
                         'negative_control':'per-assay protein-position median permuted within each outer fold',
                         'prespecified_models':{k:list(v) for k,v in MODELS.items()},
                         'bootstrap':args.bootstrap,'seed':args.seed})
    (out/'phase31i2_repo_run_provenance.json').write_text(json.dumps(run_manifest,indent=2)+'\n')
    print('\nA. Position-level MAE (mean across repeats; lower is better)')
    print(met.groupby(['target','model'],as_index=False).position_mae_pp.mean().round(4).sort_values(['target','position_mae_pp']).to_string(index=False))
    print('\nB. Position-block bootstrap gain vs matched M0 (positive = better)')
    print(comp.round(4).to_string(index=False))
    print('\nPASS: TRAIN-only exploratory ablation; external VAL and TEST labels untouched.')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage',choices=['audit','run'])
    ap.add_argument('--bihler',default='data/processed/bihler_2024_theratyping.parquet')
    ap.add_argument('--s5',default=None)
    ap.add_argument('--s6',default=None)
    ap.add_argument('--splits2024',default='data/processed/bihler_2024_missense_splits.parquet')
    ap.add_argument('--vti-meta',default='data/processed/bihler_2026_transfer_splits.parquet')
    ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--teacher-alpha',type=float,default=50.)
    ap.add_argument('--teacher-folds',type=int,default=5)
    ap.add_argument('--bootstrap',type=int,default=5000)
    ap.add_argument('--seed',type=int,default=3112)
    ap.add_argument('--quick',action='store_true',help='Train only first outer fold; no final metrics/parity')
    args=ap.parse_args(argv)
    if args.teacher_alpha<=0 or args.bootstrap<100 or args.teacher_folds<2:
        ap.error('Invalid teacher alpha, bootstrap, or teacher-fold count')
    args.out.mkdir(parents=True,exist_ok=True)
    paths=get_sources(args)
    trpos,protected=protected_and_eligible_positions(paths)
    tables,coverage=load_auxiliary(paths,trpos,protected)
    manifest=audit(paths,tables,coverage,trpos,protected,args.out)
    if args.stage=='audit':
        print('PASS Phase31I.2 audit: no VTI outcomes accessed, no models trained.')
        return
    manifest['protected_positions_list']=sorted(protected)
    run(args,tables,manifest,args.out)

if __name__=='__main__':
    try: main()
    except (ValueError, AssertionError, FileNotFoundError) as e:
        print(f'PHASE31I.2 STOP: {e}',file=sys.stderr)
        sys.exit(2)
