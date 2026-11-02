#!/usr/bin/env python3
"""Phase 31J: diagnostic audit of zero-shot VTI errors and ETI/S6 teachers.

Commands (run from CFTR-Theratype repository root):
  python -u scripts/phase31j_error_diagnostics.py run
  python -u scripts/phase31j_error_diagnostics.py teachers
  python -u scripts/phase31j_error_diagnostics.py all

RUN: loads only VTI TRAIN outcomes using Phase26D canonical loader and frozen
Phase26G/31I.2 out-of-fold predictions. NO model fit. Retrospective strata may
use observed responses but ARE NOT deployable features or prospective predictions.
TEACHERS: loads no VTI outcomes; tests seven eligible source assays with
position-held-out internal CV and fold-specific mean-reference predictions.
Does not read VTI validation/test responses, does not train a new VTI model.
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
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'results/biology/phase31j'

# APPROXIMATE exploratory structural regions, not a validated domain annotation.
# Optional --domain-map accepts source-curated intervals to replace these.
APPROX_REGIONS = [
    (1, 79, 'N-terminal'),
    (80, 380, 'MSD1~'),
    (381, 630, 'NBD1~'),
    (631, 849, 'R-domain~'),
    (850, 1176, 'MSD2~'),
    (1177, 1206, 'linker~'),
    (1207, 1440, 'NBD2~'),
    (1441, 1480, 'C-terminal'),
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def domains_from_file(path):
    if path is None:
        segments = APPROX_REGIONS
    else:
        tab = pd.read_csv(path)
        needed={'start','end','domain'}
        if not needed.issubset(tab):
            raise ValueError(f'--domain-map requires columns {needed}')
        segments = [(int(r.start),int(r.end),str(r.domain)) for r in tab.itertuples(index=False)]
    if not segments:
        raise ValueError('Empty domain map')
    check = np.zeros(1481,dtype=np.int8)
    for lo,hi,name in segments:
        if not 1<=lo<=hi<=1480 or not name or (check[lo:hi+1]>0).any():
            raise ValueError('Domain intervals overlap, contain an invalid range or have empty name')
        check[lo:hi+1] = 1
    return segments


def domain_of(p, segments):
    for lo,hi,name in segments:
        if lo <= int(p) <= hi:
            return name
    return 'Unannotated'


def summarize_group(q, by, target, model, group_name, lower_n=8):
    """Per-position error, averaged across repeats, then equal-weighted positions.

    The subset may have multiple variants at a position. Repeats never count
    as additional independent observations.
    """
    rows=[]
    for name,g in q.groupby(by,sort=False,dropna=False):
        e=g.assign(ae=(g.observed-g.prediction).abs(),signed=g.prediction-g.observed)
        per=(e.groupby(['repeat','protein_position'])[['ae','signed']].mean()
             .groupby('protein_position').mean())
        n=int(len(per)); v=per.ae.to_numpy(float)
        rows.append(dict(target=target,model=model,stratum=group_name,group=str(name),
                         independent_positions=n,variants=int(g.variant_key.nunique()),
                         position_mae_pp=float(v.mean()),mean_signed_bias_pp=float(per.signed.mean()),
                         median_position_ae_pp=float(np.median(v)),
                         small_group_warning=bool(n<lower_n)))
    return rows


def fixed_position_quartiles(values):
    # Rank positions, not variants. Deterministic tie breaking; descriptive only.
    ranks=values.rank(method='first',ascending=True)
    q=pd.qcut(ranks,q=4,labels=['Q1 lowest','Q2','Q3','Q4 highest'])
    return q.astype(str)


def assert_oof_valid(oof,d,kind):
    needed={'repeat','fold','variant_key','protein_position','observed','prediction'}
    if not needed.issubset(oof):
        raise ValueError(f'{kind}: missing columns {needed-set(oof)}')
    expected=set(d.variant_key.astype(str))
    positions=d.set_index('variant_key').protein_position.astype(int).to_dict()
    outcomes=dict(zip(d.variant_key.astype(str), d.corrector_rescue_pp if kind=='corrector' else d.vti_rescue_pp))
    if not np.isfinite(oof[['observed','prediction']].to_numpy(float)).all():
        raise AssertionError(f'{kind}: nonfinite OOF observations/predictions')
    for repeat,x in oof.groupby('repeat'):
        if len(x)!=len(d) or x.variant_key.duplicated().any() or set(x.variant_key)!=expected:
            raise AssertionError(f'{kind}: repeat {repeat} incomplete or duplicated')
        for row in x.itertuples(index=False):
            if int(row.protein_position)!=positions[str(row.variant_key)]:
                raise AssertionError(f'{kind}: position mismatch')
            if not np.isclose(float(row.observed),float(outcomes[str(row.variant_key)]),atol=1e-5):
                raise AssertionError(f'{kind}: unexpected observed TRAIN target')
    if set(oof.repeat)!= {1,2} or set(oof.fold)!={1,2,3,4,5}:
        raise AssertionError(f'{kind}: expected two historical repeats, five folds')
    if oof.duplicated(['repeat','fold','variant_key']).any():
        raise AssertionError(f'{kind}: duplicated fold membership')


def load_m0(d):
    samples=[]; inputs={}
    for target in ('corrector','total'):
        phase26=ROOT/f'results/biology/phase26g/phase26g_{target}_oof.csv'
        phase31=ROOT/'results/biology/phase31i/phase31i2_repo_oof.csv'
        for p in (phase26,phase31):
            if not p.is_file():
                raise FileNotFoundError(f'Missing frozen OOF file: {p}')
            inputs[str(p.relative_to(ROOT))]=digest(p)
        past=pd.read_csv(phase26,usecols=['repeat','fold','variant_key','protein_position','observed','xgb_80'])
        past=past.rename(columns={'xgb_80':'prediction'})
        audit=pd.read_csv(phase31,usecols=['target','model','repeat','fold','variant_key','protein_position','observed','prediction'])
        check=audit.loc[(audit.target==target)&(audit.model=='M0')].drop(columns=['target','model'])
        assert_oof_valid(past,d,target)
        assert_oof_valid(check,d,target)
        matched=past.merge(check,on=['repeat','fold','variant_key','protein_position'],how='outer',indicator=True,
                           suffixes=('_26g','_31i'),validate='one_to_one')
        if len(matched)!=len(past) or not matched._merge.eq('both').all():
            raise AssertionError('Phase26G and Phase31I M0 row misalignment')
        if not np.allclose(matched.observed_26g,matched.observed_31i,atol=1e-7):
            raise AssertionError('Phase26G and Phase31I M0 response mismatch')
        max_diff=float(np.max(np.abs(matched.prediction_26g-matched.prediction_31i)))
        if max_diff>1e-4:
            raise AssertionError(f'Historical M0 parity failed: {max_diff}')
        print(f'M0 parity [{target}]: max_abs={max_diff:.8g}; rows={len(past)}')
        past['target']=target
        past['model']='M0'
        samples.append(past)
    return pd.concat(samples,ignore_index=True), inputs


def mutation_annotations(d):
    # Identifier-only metadata, no new outcomes.
    path=ROOT/'data/processed/bihler_2026_transfer_splits.parquet'
    meta=pd.read_parquet(path,columns=['variant_key','hgvs_protein','protein_position','transfer_split','variant_type'])
    meta=meta[meta.transfer_split.astype(str).str.lower().eq('train')].copy()
    if meta.variant_key.duplicated().any():
        raise ValueError('Duplicate variant keys in VTI TRAIN metadata')
    z=d[['variant_key','protein_position']].merge(meta[['variant_key','protein_position','hgvs_protein','variant_type']],on='variant_key',validate='one_to_one',how='left',suffixes=('','_meta'))
    if z.hgvs_protein.isna().any() or not z.protein_position.eq(z.protein_position_meta).all():
        raise ValueError('VTI TRAIN metadata not aligned with frozen positions')
    from phase31i_auxiliary_transfer import with_ids
    a=with_ids(z[['variant_key','protein_position','hgvs_protein']],'Phase31J VTI TRAIN identifiers',strict=True)
    z['aa_code']=a.aa31i.to_numpy(str)
    def kind(code):
        m=re.fullmatch(r'([A-Z])(\d+)([A-Z])',str(code))
        if not m:return 'noncanonical'
        wt,alt=m.group(1),m.group(3)
        charge=lambda aa: -1 if aa in 'DE' else (1 if aa in 'KR' else 0)
        if charge(wt)!=charge(alt):return 'charge changed'
        if (wt=='P')!=(alt=='P'):return 'proline gained/lost'
        if (wt in 'FWY')!=(alt in 'FWY'):return 'aromatic changed'
        return 'other missense'
    z['substitution_category']=z.aa_code.map(kind)
    return z[['variant_key','aa_code','variant_type','substitution_category']],path


def block_ci(values, rng, nboot):
    values=np.asarray(values,dtype=float)
    if not len(values):return np.nan,np.nan
    # bounded memory even at high bootstrap count
    reps=[]
    for _ in range(0,nboot,200):
        b=min(200,nboot-len(reps))
        idx=rng.integers(0,len(values),size=(b,len(values)))
        reps.extend(values[idx].mean(axis=1).tolist())
    return float(np.quantile(reps,.025)),float(np.quantile(reps,.975))


def cross_target_analysis(oof, nboot, seed):
    corrector=oof[oof.target=='corrector'].drop(columns=['target','model']).rename(columns={'observed':'obs_cor','prediction':'pred_cor'})
    total=oof[oof.target=='total'].drop(columns=['target','model']).rename(columns={'observed':'obs_total','prediction':'pred_total'})
    j=corrector.merge(total,on=['repeat','fold','variant_key','protein_position'],validate='one_to_one',suffixes=('_c','_t'))
    if len(j)!=len(corrector):raise AssertionError('Missing corrector/total paired OOF rows')
    j['err_cor']=j.pred_cor-j.obs_cor
    j['err_total']=j.pred_total-j.obs_total
    j['obs_implied_acute']=j.obs_total-j.obs_cor
    j['pred_implied_acute']=j.pred_total-j.pred_cor
    j['err_implied_acute']=j.pred_implied_acute-j.obs_implied_acute
    for label in ('cor','total','implied_acute'):
        j['ae_'+label]=j['err_'+label].abs()
    # Mean absolute error BEFORE averaging positions (no sign cancellation).
    fields=['err_cor','err_total','err_implied_acute','ae_cor','ae_total','ae_implied_acute',
            'obs_cor','obs_total','obs_implied_acute','pred_cor','pred_total','pred_implied_acute']
    v=j.groupby(['variant_key','protein_position'],as_index=False)[fields].mean()
    p=v.groupby('protein_position')[['err_cor','err_total','err_implied_acute','ae_cor','ae_total','ae_implied_acute']].mean()
    rho=float(spearmanr(p.ae_cor,p.ae_total).statistic)
    signed=float(np.corrcoef(p.err_cor,p.err_total)[0,1])
    pc=int(np.ceil(.2*len(p)))
    bad_cor=set(p.ae_cor.nlargest(pc).index)
    bad_total=set(p.ae_total.nlargest(pc).index)
    overlap=int(len(bad_cor&bad_total))
    # algebraic decomposition e_total=e_corrector+e_implied_acute
    rem=float(np.max(np.abs((j.err_cor+j.err_implied_acute)-j.err_total)))
    if rem>1e-6:raise AssertionError('Joint error identity broken')
    res={
        'independent_positions':int(len(p)),
        'n_worst20pct_positions_each':pc,
        'worst20pct_overlap_positions':overlap,
        'worst20pct_overlap_fraction':overlap/pc,
        'spearman_abs_error_cor_vs_total':rho,
        'pearson_signed_error_cor_vs_total':signed,
        'position_mae_corrector':float(p.ae_cor.mean()),
        'position_mae_total':float(p.ae_total.mean()),
        'position_mae_implied_acute':float(p.ae_implied_acute.mean()),
        'signed_bias_corrector':float(p.err_cor.mean()),
        'signed_bias_total':float(p.err_total.mean()),
        'signed_bias_implied_acute':float(p.err_implied_acute.mean()),
        'algebraic_error_check':rem,
        'notes':'Implied acute is the difference of two independently fitted models, NOT an independently validated acute model.'
    }
    pos_err=p.copy(); pos_err['total_minus_corrector_abs_error']=p.ae_total-p.ae_cor
    lo,hi=block_ci(pos_err.total_minus_corrector_abs_error.to_numpy(),np.random.default_rng(seed),nboot)
    res['delta_abs_error_total_minus_corrector']=float(pos_err.total_minus_corrector_abs_error.mean())
    res['delta_abs_error_ci95_low']=lo;res['delta_abs_error_ci95_high']=hi
    return j,v,p.reset_index(),res


def transferred_gain_strata(oof31,base,strata,nboot,seed):
    rows=[]
    rng=np.random.default_rng(seed)
    for target in ('corrector','total'):
        b=base.loc[base.target.eq(target),['repeat','fold','variant_key','protein_position','observed','prediction']].rename(columns={'prediction':'p0'})
        for model in ('M1','M2','M3','M3_perm'):
            q=oof31.loc[oof31.target.eq(target)&oof31.model.eq(model),['repeat','fold','variant_key','protein_position','prediction']].rename(columns={'prediction':'px'})
            if len(q)!=len(b):raise AssertionError('Incomplete Phase31I OOF transfer model: '+model)
            x=b.merge(q,on=['repeat','fold','variant_key','protein_position'],validate='one_to_one')
            if len(x)!=len(b):raise AssertionError('Missing rows in transfer model: '+model)
            x=x.merge(strata,on='variant_key',validate='many_to_one')
            x['gain']=np.abs(x.observed-x.p0)-np.abs(x.observed-x.px)
            for field in ('domain','baseline_quartile','total_rescue_quartile','corrector_rescue_quartile'):
                for group,z in x.groupby(field,dropna=False):
                    per=(z.groupby(['protein_position','repeat']).gain.mean().groupby('protein_position').mean())
                    ci=block_ci(per.to_numpy(),rng,nboot)
                    rows.append(dict(target=target,model=model,stratum=field,group=str(group),
                                     positions=len(per),position_gain_pp=float(per.mean()),ci95_low=ci[0],ci95_high=ci[1]))
    return pd.DataFrame(rows)


def run_vti(args):
    import phase26d_corrector_bottleneck as prior
    d,_,_,folds,source=prior.load_train()
    if len(d)!=210 or d.protein_position.nunique()!=168 or not d.transfer_split.eq('train').all():
        raise AssertionError('Frozen VTI TRAIN contract failed')
    print('Phase 31J RUN | 210 TRAIN variants / 168 positions | frozen folds:',source)
    m0,checks=load_m0(d)
    frozen_membership={}
    for rep,fold,_,held_keys in folds:
        for key in held_keys:
            k=(int(rep),str(key))
            if k in frozen_membership:
                raise AssertionError('Repeated frozen OOF membership for '+str(k))
            frozen_membership[k]=int(fold)
    for row in m0.loc[m0.target.eq('corrector'),['repeat','fold','variant_key']].itertuples(index=False):
        if frozen_membership.get((int(row.repeat),str(row.variant_key)))!=int(row.fold):
            raise AssertionError('Historical M0 OOF does not match Phase26D frozen fold assignment')
    for target in ('corrector','total'):
        q=m0[m0.target.eq(target)]
        if q.groupby(['repeat','protein_position']).fold.nunique().max()!=1:
            raise AssertionError('Protein position split across Phase26G held-out folds')
    meta,meta_path=mutation_annotations(d)
    segments=domains_from_file(args.domain_map)
    tab=d[['variant_key','protein_position','study_cohort','baseline_fsk_pct_wt','corrector_rescue_pp','vti_rescue_pp']].merge(meta,on='variant_key',validate='one_to_one')
    pos=tab.groupby('protein_position')[['baseline_fsk_pct_wt','corrector_rescue_pp','vti_rescue_pp']].mean()
    pos['baseline_quartile']=fixed_position_quartiles(pos.baseline_fsk_pct_wt)
    pos['corrector_rescue_quartile']=fixed_position_quartiles(pos.corrector_rescue_pp)
    pos['total_rescue_quartile']=fixed_position_quartiles(pos.vti_rescue_pp)
    pos['domain']=[domain_of(i,segments) for i in pos.index]
    tab=tab.merge(pos[['domain','baseline_quartile','corrector_rescue_quartile','total_rescue_quartile']],left_on='protein_position',right_index=True,validate='many_to_one')
    annotated=m0.merge(tab,on=['variant_key','protein_position'],validate='many_to_one')
    annotated['error_pp']=annotated.prediction-annotated.observed
    annotated['absolute_error_pp']=annotated.error_pp.abs()
    annotated.to_csv(args.out/'phase31j_m0_annotated_oof.csv',index=False)
    groups=['domain','baseline_quartile','total_rescue_quartile','corrector_rescue_quartile','study_cohort','variant_type','substitution_category']
    grouped=[]
    for (target,model),g in annotated.groupby(['target','model']):
        for col in groups:
            grouped+=summarize_group(g,col,target,model,col)
        # This is especially relevant for tail underprediction.
        for label,mask in [('<0',g.observed<0),('[0,20)',(g.observed>=0)&(g.observed<20)),
                           ('[20,40]',(g.observed>=20)&(g.observed<=40)),('>40',g.observed>40)]:
            if mask.any():
                h=g.loc[mask].copy()
                h['response_bin']=label
                grouped+=summarize_group(h,'response_bin',target,model,'observed_response_bin')
    strata=pd.DataFrame(grouped)
    strata.to_csv(args.out/'phase31j_m0_error_strata.csv',index=False)
    joint,joint_var,joint_pos,summary=cross_target_analysis(m0,args.bootstraps,args.seed)
    for target,col in [('corrector','position_mae_corrector'),('total','position_mae_total')]:
        ref=(annotated.loc[annotated.target.eq(target)]
             .groupby(['repeat','protein_position']).absolute_error_pp.mean()
             .groupby('protein_position').mean().mean())
        if not np.isclose(float(ref),float(summary[col]),atol=1e-8):
            raise AssertionError(f'Joint-error position MAE does not match frozen M0 [{target}]')
    joint.to_csv(args.out/'phase31j_joint_oof.csv',index=False)
    joint_var.to_csv(args.out/'phase31j_joint_by_variant.csv',index=False)
    joint_pos.to_csv(args.out/'phase31j_joint_by_position.csv',index=False)
    (args.out/'phase31j_joint_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    # Candidate gain stratification reuses PREDICTIONS, no model fitting.
    path=ROOT/'results/biology/phase31i/phase31i2_repo_oof.csv'
    alt=pd.read_csv(path)
    expected={'target','model','repeat','fold','variant_key','protein_position','observed','prediction'}
    if not expected.issubset(alt):raise ValueError('Phase31I OOF missing columns')
    gain=transferred_gain_strata(alt,m0,tab[['variant_key','domain','baseline_quartile','total_rescue_quartile','corrector_rescue_quartile']],args.bootstraps,args.seed)
    gain.to_csv(args.out/'phase31j_transfer_gain_strata.csv',index=False)
    report={
        'stage':'run', 'train_only':True,'vti_val_test_outcomes_loaded':False,
        'm0_sources_sha256':checks,'vti_train_metadata_sha256':digest(meta_path),
        'domain_map':'approximate_exploratory_intervals' if args.domain_map is None else str(args.domain_map),
        'domain_map_intervals':segments,'n_vti_variants':len(d),'n_positions':d.protein_position.nunique(),
        'repeats':sorted(m0.repeat.unique().astype(int).tolist()),
        'note':'Observed rescue/baseline quartiles and true-response bins are retrospective error diagnostics only.',
        'joint_metrics':summary}
    (args.out/'phase31j_run_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('\nA. M0 ERROR STRATA (sample; full CSV has all strata)')
    print(strata.loc[strata.stratum.isin(['domain','total_rescue_quartile'])][['target','stratum','group','independent_positions','position_mae_pp','mean_signed_bias_pp']].round(3).to_string(index=False))
    print('\nB. JOINT CORRECTOR/TOTAL ERRORS')
    for key in ('worst20pct_overlap_positions','worst20pct_overlap_fraction','spearman_abs_error_cor_vs_total','position_mae_implied_acute','delta_abs_error_total_minus_corrector'):
        print(f'  {key}: {summary[key]}')
    print('\nC. TRANSFER GAINS IN HIGHEST-RESPONSE QUARTILE (exploratory)')
    focus=gain[(gain.target=='total')&(gain.stratum=='total_rescue_quartile')&(gain.group=='Q4 highest')]
    print(focus[['model','positions','position_gain_pp','ci95_low','ci95_high']].round(3).to_string(index=False))
    print(f'\nPASS: Phase31J RUN outputs saved to {args.out}; no new VTI models fitted.')
    return summary


def teacher_diagnostics(args):
    # Separate from Phase31J RUN: NO VTI outcomes loaded in this function.
    import phase31i2_repository as phase31i
    from phase31i_auxiliary_transfer import ROOT as helper_root
    if Path(helper_root).resolve()!=ROOT.resolve():
        raise ValueError('Phase31I helper root differs from current repository')
    opts=argparse.Namespace(bihler=args.bihler,s5=args.s5,s6=args.s6,
                            splits2024=args.splits2024,vti_meta=args.vti_meta)
    paths=phase31i.get_sources(opts)
    positions,protected=phase31i.protected_and_eligible_positions(paths)
    tables,coverage=phase31i.load_auxiliary(paths,positions,protected)
    print('Phase31J TEACHERS: position-held-out internal assay CV; no VTI labels loaded')
    from phase31i2_repository import mutation_features,make_teacher
    oof=[];summary=[];boots=[]
    rng=np.random.default_rng(args.seed)
    for name,tab in tables.items():
        x=mutation_features(tab.aa31i)
        y=tab.value31i.to_numpy(float)
        p=tab.pos31i2.to_numpy(int)
        if not np.isfinite(y).all():raise AssertionError('Non-finite auxiliary label')
        npos=len(set(p));k=min(args.teacher_folds,npos)
        if k<2:raise ValueError(f'Too few positions for teacher {name}')
        pred=np.full(len(y),np.nan);reference=np.full(len(y),np.nan);idx=np.full(len(y),-1,int)
        for fold,(tr,te) in enumerate(GroupKFold(k).split(x,y,p),1):
            if set(p[tr])&set(p[te]):raise AssertionError('Position leakage in teacher CV')
            est=make_teacher(args.teacher_alpha)
            est.fit(x[tr],y[tr]);pred[te]=est.predict(x[te]);reference[te]=y[tr].mean();idx[te]=fold
        if np.isnan(pred).any() or np.isnan(reference).any():raise AssertionError(f'Missing teacher predictions: {name}')
        z=pd.DataFrame(dict(teacher=name,aa_code=tab.aa31i,protein_position=p,fold=idx,
                            observed=y,teacher_pred=pred,fold_mean_ref=reference))
        oof.append(z)
        pos_df=(z.assign(err_teacher=(z.observed-z.teacher_pred).abs(),err_ref=(z.observed-z.fold_mean_ref).abs())
                .groupby('protein_position')[['err_teacher','err_ref']].mean())
        gains=(pos_df.err_ref-pos_df.err_teacher).to_numpy(float)
        lo,hi=block_ci(gains,rng,args.bootstraps)
        rho=float(spearmanr(y,pred).statistic) if len(np.unique(y))>1 and len(np.unique(pred))>1 else np.nan
        summary.append(dict(teacher=name,substitutions=len(y),positions=npos,
                            position_mae_teacher=float(pos_df.err_teacher.mean()),
                            position_mae_mean_reference=float(pos_df.err_ref.mean()),
                            position_mae_gain_vs_mean=float(gains.mean()),
                            position_rmse_teacher=float(np.sqrt(z.assign(sq=(z.observed-z.teacher_pred)**2).groupby('protein_position').sq.mean().mean())),
                            spearman_all_heldout_substitutions=rho,
                            bootstrap_gain_ci95_low=lo,bootstrap_gain_ci95_high=hi,
                            **{'scale_note':'Assay-native labels; not comparable across assays'}))
        print(f'  {name:>20}: n_sub={len(y):3} n_pos={npos:3} '
              f'MAE={pos_df.err_teacher.mean():8.3f} vs fold-mean={pos_df.err_ref.mean():8.3f} '
              f'gain={gains.mean():+8.3f}, rho={rho:.3f}',flush=True)
    pd.concat(oof,ignore_index=True).to_csv(args.out/'phase31j_teacher_internal_oof.csv',index=False)
    met=pd.DataFrame(summary)
    met.to_csv(args.out/'phase31j_teacher_metrics.csv',index=False)
    coverage.to_csv(args.out/'phase31j_teacher_filtered_coverage.csv',index=False)
    (args.out/'phase31j_teacher_manifest.json').write_text(json.dumps({
        'stage':'teachers','vti_outcomes_loaded':False,
        'assay_files_sha256':{str(k):digest(v) for k,v in paths.items()},
        'teacher_alpha':args.teacher_alpha,'n_cv_folds':args.teacher_folds,
        'evaluation':'one teacher per assay, position-held-out grouped CV; fold-local mean reference',
        'interpretation':'Internal teacher reliability, not validation of VTI predictive value.',
        'nboot':args.bootstraps,'seed':args.seed},indent=2)+'\n')
    print(f'PASS: Phase31J TEACHERS outputs saved to {args.out}; no VTI outcomes accessed.')
    return met


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage',choices=['run','teachers','all'])
    ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--domain-map',type=Path,default=None,help='CSV with start,end,domain. Overrides approximate regions.')
    ap.add_argument('--bootstraps',type=int,default=2000)
    ap.add_argument('--teacher-folds',type=int,default=5)
    ap.add_argument('--teacher-alpha',type=float,default=50.)
    ap.add_argument('--seed',type=int,default=3113)
    ap.add_argument('--bihler',default='data/processed/bihler_2024_theratyping.parquet')
    ap.add_argument('--s5',default=None)
    ap.add_argument('--s6',default=None)
    ap.add_argument('--splits2024',default='data/processed/bihler_2024_missense_splits.parquet')
    ap.add_argument('--vti-meta',default='data/processed/bihler_2026_transfer_splits.parquet')
    args=ap.parse_args(argv)
    if args.bootstraps<100 or args.teacher_alpha<=0 or args.teacher_folds<2:
        ap.error('Invalid bootstrap/teacher settings')
    args.out.mkdir(parents=True,exist_ok=True)
    # For 'all', teacher evaluation runs FIRST, before VTI TRAIN outcomes load.
    if args.stage in ('teachers','all'):
        teacher_diagnostics(args)
    if args.stage in ('run','all'):
        run_vti(args)

if __name__=='__main__':
    try:main()
    except (ValueError,AssertionError,FileNotFoundError) as e:
        print(f'PHASE31J STOP: {e}',file=sys.stderr)
        raise SystemExit(2)
