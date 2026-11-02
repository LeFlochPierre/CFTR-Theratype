#!/usr/bin/env python3
"""Phase 26I: diagnostic TRAIN-only error audit from frozen out-of-fold predictions.
No fitting, no VTI/ETI validation/test outcome reads. Descriptive/exploratory.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
OOF = ROOT / 'results/biology/phase26h2/phase26h2_oof.csv'
META = ROOT / 'data/processed/bihler_2026_transfer_splits.parquet'
ASSAY = ROOT / 'data/processed/bihler_2026_vti.parquet'
STRUCT = ROOT / 'results/biology/phase26e/phase26e_drugsite_features.csv'
OUT = ROOT / 'results/biology/phase26i'
COLS = {
    'corrector': {'target':'observed_corrector', 'baseline':'vti_fixed', 'real':'joint_real',
                  'other':'novel_real', 'zero':'pred_zero_corrector_seq80'},
    'total': {'target':'observed_total', 'baseline':'vti_fixed_total', 'real':'joint_real_total',
              'other':'novel_real_total', 'zero':'pred_zero_total_seq80'},
}


def load():
    d = pd.read_csv(OOF, dtype={'variant_key':str})
    required={'repeat','fold','variant_key','protein_position','observed_corrector',
              'observed_total','pred_zero_acute_seq80'} | {
        v for task in COLS.values() for v in task.values()
    }
    if not required.issubset(d.columns):
        raise ValueError(f'Missing OOF columns: {sorted(required-set(d.columns))}')
    if d.duplicated(['repeat','variant_key']).any() or d.duplicated(['repeat','fold','variant_key']).any():
        raise ValueError('Duplicate OOF variant within a repeat or fold')
    counts=d.groupby('repeat').variant_key.nunique()
    if not (len(counts)==2 and (counts==210).all() and len(d)==420):
        raise ValueError(f'Unexpected OOF coverage: {counts.to_dict()}, rows={len(d)}')
    if d.protein_position.nunique()!=168:
        raise ValueError('Expected 168 protein positions')
    if d.groupby('variant_key').protein_position.nunique().max()!=1:
        raise ValueError('Inconsistent position identity')
    if set(d.loc[d.repeat==counts.index[0],'variant_key']) != set(d.loc[d.repeat==counts.index[1],'variant_key']):
        raise ValueError('Different variant keys between repeats')
    need_num=required-{'variant_key'}
    for c in need_num:
        if not np.isfinite(pd.to_numeric(d[c],errors='coerce')).all():
            raise ValueError(f'Nonfinite OOF column: {c}')

    # Metadata from split manifest ONLY; never open protected outcome rows.
    meta=pd.read_parquet(META,columns=['variant_key','protein_position','transfer_split','study_cohort','hgvs_protein'])
    meta=meta.loc[meta.transfer_split.eq('train'),
                  ['variant_key','protein_position','study_cohort','hgvs_protein']].copy()
    meta.variant_key=meta.variant_key.astype(str)
    assert len(meta)==210 and meta.variant_key.is_unique
    d=d.merge(meta,on=['variant_key','protein_position'],validate='many_to_one',how='left',indicator=True)
    if not d._merge.eq('both').all():
        raise ValueError('OOF rows do not match frozen TRAIN metadata')
    d=d.drop(columns='_merge')

    # Read only TRAIN assay rows, never protected outcomes.
    assay=pd.read_parquet(ASSAY,columns=['variant_key','baseline_fsk_pct_wt',
                              'iva_treated_pct_wt','fsk_sem_dmso','fsk_sem_vnz_tez',
                              'fsk_iva_sem_vnz_tez','wells_vnz_tez'],
                         filters=[('variant_key','in',sorted(set(meta.variant_key)))])
    assay.variant_key=assay.variant_key.astype(str)
    if len(assay)!=210 or not assay.variant_key.is_unique:
        raise ValueError('Filtered TRAIN assays do not match 210 variants')
    d=d.merge(assay,on='variant_key',validate='many_to_one')

    if STRUCT.exists():
        s=pd.read_csv(STRUCT,dtype={'variant_key':str})
        if {'variant_key','tez_min_atom_A'}.issubset(s.columns) and s.variant_key.is_unique:
            d=d.merge(s[['variant_key','tez_min_atom_A']],on='variant_key',validate='many_to_one',how='left')
            d['has_7sv7_coords']=np.isfinite(pd.to_numeric(d.tez_min_atom_A,errors='coerce'))

    d['nbd1_exploratory']=d.protein_position.between(390,670)
    d['observed_acute']=d.observed_total-d.observed_corrector
    d['err_corrector']=d.joint_real-d.observed_corrector
    d['err_acute']=d.pred_zero_acute_seq80-d.observed_acute
    d['err_total']=d.joint_real_total-d.observed_total

    if not np.allclose(d.err_corrector+d.err_acute,d.err_total,atol=1e-6):
        raise ValueError('Model component prediction does not reconstruct total response')
    return d


def pos_mae(t,target,pred):
    z=t.assign(e=np.abs(t[target].to_numpy(float)-t[pred].to_numpy(float)))
    return float(z.groupby(['repeat','protein_position']).e.mean().groupby('protein_position').mean().mean())


def subsection(t,target,ref,model,label):
    if len(t)==0:
        return None
    y=t[target].to_numpy(float); a=t[model].to_numpy(float); b=t[ref].to_numpy(float)
    return {'stratum':str(label), 'variants':t.variant_key.nunique(),
            'positions':t.protein_position.nunique(), 'true_mean':float(y.mean()),
            'pred_mean':float(a.mean()),'bias_pred_minus_true_pp':float(np.mean(a-y)),
            'real_mae_pp':pos_mae(t,target,model), 'baseline_mae_pp':pos_mae(t,target,ref),
            'gain_pp':pos_mae(t,target,ref)-pos_mae(t,target,model),
            'real_rmse_pp':float(np.sqrt(np.mean((a-y)**2)))}


def diagnose(d):
    print('='*103)
    print('PHASE 26I | WHERE DO OUR ZERO-SHOT ETI-ASSISTED VTI PREDICTIONS FAIL? | TRAIN OOF')
    print('='*103)
    print(f'Frozen data: {d.variant_key.nunique()} variants; {d.protein_position.nunique()} positions; '
          f'{d.repeat.nunique()} repeated out-of-fold predictions')
    print('No model fitting or protected val/test outcomes; ALL subgroup comparisons are exploratory.')
    print('Note: groups defined using true response identify diagnostic failures, not deployment-time groups.')

    strata=[]
    for task,c in COLS.items():
        ycol=c['target']; y=d[ycol].to_numpy(float)
        categories={
            'ALL':np.ones(len(d),bool),
            'true <10 pp':y<10,
            'true 10-40 pp':(y>=10)&(y<=40),
            'true >40 pp':y>40,
            'true >60 pp':y>60,
        }
        for cohort in sorted(d.study_cohort.dropna().unique()):
            categories[f'cohort {cohort}']=d.study_cohort.eq(cohort).to_numpy()
        categories['NBD1 (post hoc)']=d.nbd1_exploratory.to_numpy()
        categories['outside NBD1']=~d.nbd1_exploratory.to_numpy()
        if 'has_7sv7_coords' in d:
            categories['mapped to 7SV7']=d.has_7sv7_coords.to_numpy()
            categories['missing in 7SV7']=~d.has_7sv7_coords.to_numpy()
        for name,mask in categories.items():
            v=subsection(d.loc[mask],ycol,c['baseline'],c['real'],name)
            if v: strata.append({'target':task,**v})
        print(f'\nA. {task.upper()} — ERROR BY RESPONSE AND COHORT (position-weighted MAE)')
        ss=pd.DataFrame([x for x in strata if x['target']==task])
        fields=['stratum','variants','positions','true_mean','pred_mean',
                'bias_pred_minus_true_pp','real_mae_pp','baseline_mae_pp','gain_pp']
        print(ss[fields].round(3).to_string(index=False))
    strata=pd.DataFrame(strata)

    print('\nB. ERROR DECOMPOSITION — TESTING THE CORRECTOR BOTTLENECK')
    comp=[]
    for name,col in [('corrector','err_corrector'),('acute IVA','err_acute'),('total','err_total')]:
        x=d[col].to_numpy(float)
        comp.append({'component':name, 'bias_pp':float(x.mean()),
                     'MAE_pp':float(np.mean(np.abs(x))),
                     'RMSE_pp':float(np.sqrt(np.mean(x*x)))})
    comp=pd.DataFrame(comp)
    print(comp.round(3).to_string(index=False))
    corr=float(np.corrcoef(d.err_corrector,d.err_acute)[0,1])
    cancel=float(np.mean(np.sign(d.err_corrector)!=np.sign(d.err_acute)))
    print(f'Corrector–acute error Pearson correlation: {corr:.3f}')
    print(f'Opposite-sign component errors: {cancel:.1%} of OOF observations')
    print('These quantities are descriptive; total error can reflect cancellation between component errors.')

    print('\nC. EXTREME RESPONDER RANKING (averaged prediction across 2 OOF repeats)')
    # Use repeat averaging ONLY for ranking diagnostics; not as MAE estimates.
    one=d.groupby(['variant_key','protein_position'],as_index=False).agg({
        'observed_corrector':'first','observed_total':'first',
        'joint_real':'mean','joint_real_total':'mean'})
    ranks=[]
    for name,ycol,pcol in [('corrector','observed_corrector','joint_real'),
                           ('total','observed_total','joint_real_total')]:
        for unit in ['variant','position']:
            q=one.copy()
            if unit=='position':
                q=q.groupby('protein_position',as_index=False).agg({ycol:'max',pcol:'max'})
            truth=(q[ycol]>40).astype(int).to_numpy()
            pred=q[pcol].to_numpy(float)
            k=max(1,int(np.ceil(0.20*len(q))))
            hits=int(truth[np.argsort(-pred,kind='stable')[:k]].sum())
            ranks.append({'target':name,'unit':unit,'n':len(q),'positives':int(truth.sum()),
                          'AP':float(average_precision_score(truth,pred)) if 0<truth.sum()<len(truth) else np.nan,
                          'AUROC':float(roc_auc_score(truth,pred)) if 0<truth.sum()<len(truth) else np.nan,
                          'top20_hits':hits, 'top20_selected':k,
                          'top20_precision':hits/k,
                          'top20_recall':hits/int(truth.sum()) if truth.sum() else np.nan})
    ranks=pd.DataFrame(ranks)
    print(ranks.round(3).to_string(index=False))

    print('\nD. HARDEST INDIVIDUAL MUTATIONS (mean absolute error across repeat predictions)')
    per_variant=d.copy()
    for task,c in COLS.items():
        per_variant[f'ae_{task}']=np.abs(per_variant[c['real']]-per_variant[c['target']])
        per_variant[f'gain_{task}']=np.abs(per_variant[c['baseline']]-per_variant[c['target']])-per_variant[f'ae_{task}']
    variant=per_variant.groupby('variant_key',as_index=False).agg(
        position=('protein_position','first'), hgvs=('hgvs_protein','first'),cohort=('study_cohort','first'),
        true_corrector=('observed_corrector','first'),pred_corrector=('joint_real','mean'),
        true_total=('observed_total','first'),pred_total=('joint_real_total','mean'),
        true_acute=('observed_acute','first'),pred_acute=('pred_zero_acute_seq80','mean'),
        ae_corrector=('ae_corrector','mean'),ae_total=('ae_total','mean'),
        gain_corrector=('gain_corrector','mean'),gain_total=('gain_total','mean'),
        baseline_fsk=('baseline_fsk_pct_wt','first'),
        corrected_sem=('fsk_sem_vnz_tez','first'),
        treated_sem=('fsk_iva_sem_vnz_tez','first'))
    print(variant.nlargest(15,'ae_total')[['variant_key','position','hgvs','true_total','pred_total',
          'ae_total','gain_total','true_corrector','pred_corrector','true_acute','pred_acute']]
          .round(2).to_string(index=False))

    print('\nE. WORST PROTEIN POSITIONS BY TOTAL ERROR')
    pos=variant.groupby('position',as_index=False).agg(
        variants=('variant_key','size'), mean_total_mae=('ae_total','mean'),
        mean_corrector_mae=('ae_corrector','mean'),gain_total=('gain_total','mean'),
        gain_corrector=('gain_corrector','mean'))
    print(pos.nlargest(12,'mean_total_mae').round(2).to_string(index=False))

    print('\nF. WHERE AUXILIARY ETI HELPED / HURT (position-level, exploratory)')
    print('Largest improvements:')
    print(pos.nlargest(10,'gain_corrector')[['position','variants','gain_corrector','gain_total',
                                            'mean_corrector_mae']].round(2).to_string(index=False))
    print('Largest deteriorations:')
    print(pos.nsmallest(10,'gain_corrector')[['position','variants','gain_corrector','gain_total',
                                             'mean_corrector_mae']].round(2).to_string(index=False))

    print('\nG. MEASUREMENT QUALITY AMONG THE HARDEST 25% (descriptive; SEM not error-free truth)')
    threshold=variant.ae_total.quantile(.75)
    variant['hard_total_q4']=variant.ae_total>=threshold
    quality=variant.groupby('hard_total_q4').agg(n=('variant_key','size'),
        mae_total=('ae_total','mean'),median_baseline=('baseline_fsk','median'),
        median_corrected_sem=('corrected_sem','median'),
        median_treated_sem=('treated_sem','median')).reset_index()
    print(quality.round(3).to_string(index=False))
    print('Do not interpret SEM correlations as formal statistical significance: cross-condition covariance is unknown.')
    return strata,comp,ranks,variant,pos,quality


def self_test():
    rng=np.random.default_rng(26)
    keys=[f'syn{i:03d}' for i in range(10)]
    d=pd.DataFrame({'variant_key':keys*2,'repeat':[1]*10+[2]*10,
         'protein_position':list(range(10))*2,'fold':list(range(5))*4})
    d['observed_corrector']=np.tile(rng.normal(20,8,10),2)
    d['observed_total']=d.observed_corrector+6
    d['joint_real']=d.observed_corrector+rng.normal(0,3,20)
    d['pred_zero_acute_seq80']=6+rng.normal(0,2,20)
    d['joint_real_total']=d.joint_real+d.pred_zero_acute_seq80
    d['vti_fixed']=d.observed_corrector+rng.normal(0,4,20)
    d['vti_fixed_total']=d.vti_fixed+d.pred_zero_acute_seq80
    d['study_cohort']='synthetic';d['nbd1_exploratory']=False;d['hgvs_protein']='p.Ala1Val'
    d['baseline_fsk_pct_wt']=3.;d['fsk_sem_vnz_tez']=1.;d['fsk_iva_sem_vnz_tez']=1.
    d['observed_acute']=6.;d['err_corrector']=d.joint_real-d.observed_corrector
    d['err_acute']=d.pred_zero_acute_seq80-6.;d['err_total']=d.joint_real_total-d.observed_total
    assert np.allclose(d.err_total,d.err_corrector+d.err_acute)
    assert np.isfinite(pos_mae(d,'observed_corrector','joint_real'))
    assert subsection(d,'observed_corrector','vti_fixed','joint_real','ALL')['positions']==10
    print('PASS synthetic: repeated-position scoring, error decomposition and subgroup summaries')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        self_test();return
    d=load()
    outputs=diagnose(d)
    OUT.mkdir(parents=True,exist_ok=True)
    for name,frame in zip(('subgroups','component_errors','tail_ranking','variant_errors',
                           'position_errors','quality_comparison'),outputs):
        frame.to_csv(OUT/f'phase26i_{name}.csv',index=False)
    print('\nPASS: Phase26I completed. All diagnostics based on frozen TRAIN OOF only.')
    print('Saved:',OUT)


if __name__=='__main__':
    main()
