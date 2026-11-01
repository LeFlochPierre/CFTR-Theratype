#!/usr/bin/env python3
"""Phase 24A | TRAIN-only total VTI rescue endpoint audit (NO new model fitting).

Reads frozen VTI TRAIN labels via cftr_theratype.model_zoo.prepare(), verifies
Phase17A repeated held-out predictions, reports objective-aligned benchmarks,
nonresponse ranking and an *assay-conditioned*, paired ETI/VTI diagnostic.

No validation/test/excluded outcomes opened. No adaptive test-set tuning.

Run:
 python scripts/phase24a_total_vti_rescue_audit.py --self-test
 python scripts/phase24a_total_vti_rescue_audit.py --bootstraps 5000
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

XGB = 'sequence_structure__xgboost'
FAMILIES = ('sequence_structure__xgboost', 'sequence_structure__extra_trees')
LIMIT = 10.0  # exploratory in-vitro total functional gain relative to untreated FSK
SEED = 20261028


def require(condition, explanation):
    if not bool(condition):
        raise AssertionError(explanation)


def as_float(a):
    return np.asarray(a, dtype=float)


def position_mae(y, pred, pos):
    z = pd.DataFrame({'pos': pos, 'ae': np.abs(as_float(y) - as_float(pred))})
    return float(z.groupby('pos').ae.mean().mean())


def mae_table(d):
    rows = []
    for name, g in d.groupby('model', sort=True):
        y, p = as_float(g.observed_pp), as_float(g.predicted_pp)
        rows.append({'model':name,
                     'position_mae_pp':position_mae(y,p,g.protein_position),
                     'variant_mae_pp':float(np.mean(np.abs(y-p))),
                     'rmse_pp':float(np.sqrt(np.mean((y-p)**2))),
                     'bias_pp':float(np.mean(p-y)),
                     'under10_mae_pp':float(np.mean(np.abs(y[y<LIMIT]-p[y<LIMIT])))
                         if (y<LIMIT).any() else np.nan,
                     'gt40_mae_pp':float(np.mean(np.abs(y[y>40]-p[y>40])))
                         if (y>40).any() else np.nan})
    return pd.DataFrame(rows).sort_values('position_mae_pp')


def screening_by_position(d):
    # Position is labelled as rescuable if ANY nucleotide substitution at it
    # has a total functional gain >= LIMIT. Score = max predicted at position.
    rows=[]
    for name, g in d.groupby('model', sort=True):
        z = g.groupby('protein_position', as_index=False).agg(
            observed_max=('observed_pp','max'), predicted_max=('predicted_pp','max'))
        y = z.observed_max.to_numpy(float) < LIMIT
        score = -z.predicted_max.to_numpy(float)  # higher => predicted nonresponder
        n=len(y); k=int(np.ceil(0.20*n)); sel=np.argsort(-score,kind='stable')[:k]
        rows.append({'model':name,'n_positions':n,'n_nonresponse':int(y.sum()),
                     'nonresponse_prevalence':float(y.mean()),
                     'nonresponse_pr_auc':float(average_precision_score(y,score))
                         if 0<y.sum()<n else np.nan,
                     'nonresponse_auroc':float(roc_auc_score(y,score))
                         if 0<y.sum()<n else np.nan,
                     'top20_screened':k,'top20_nonresponse_hits':int(y[sel].sum()),
                     'top20_nonresponse_precision':float(y[sel].mean()),
                     'top20_nonresponse_recall':float(y[sel].sum()/y.sum()) if y.sum() else np.nan})
    return pd.DataFrame(rows).sort_values('nonresponse_pr_auc', ascending=False)


def paired_bootstrap(d, model, comparator, n_boot):
    a=d.loc[d.model.eq(model),['variant_key','protein_position','observed_pp','predicted_pp']]
    b=d.loc[d.model.eq(comparator),['variant_key','predicted_pp']]
    require(len(a)==len(b)==210, 'Paired models must each have 210 variants')
    m=a.merge(b,on='variant_key',suffixes=('_a','_b'),validate='one_to_one')
    require(len(m)==210,'Paired models require exact key match')
    imp = np.abs(m.observed_pp.to_numpy(float)-m.predicted_pp_b.to_numpy(float)) - np.abs(
        m.observed_pp.to_numpy(float)-m.predicted_pp_a.to_numpy(float))
    z=pd.DataFrame({'pos':m.protein_position,'gain':imp}).groupby('pos').gain.mean().to_numpy(float)
    require(len(z)==168, 'Missing positions in paired bootstrap')
    rng=np.random.default_rng(SEED)
    means=np.empty(n_boot,float)
    for j in range(n_boot):
        means[j]=z[rng.integers(0,len(z),size=len(z))].mean()
    return {'model':model,'comparator':comparator,'mae_gain_pp':float(z.mean()),
            'ci95_low':float(np.quantile(means,.025)),
            'ci95_high':float(np.quantile(means,.975)),
            'bootstrap_positive_fraction':float((means>0).mean()),
            'positions_improved':int((z>0).sum())}


def self_test():
    rng=np.random.default_rng(17)
    truth = rng.normal(20,10,210)
    position=np.arange(210)%168
    frame=[]
    for model,noise in [('a',5),('b',7)]:
        for i in range(210):
            frame.append({'model':model,'variant_key':f'k{i}',
                          'protein_position':int(position[i]),
                          'observed_pp':truth[i],
                          'predicted_pp':truth[i]+rng.normal(0,noise)})
    d=pd.DataFrame(frame)
    require(len(mae_table(d))==2,'Bad MAE table')
    require(len(screening_by_position(d))==2,'Bad ranking table')
    require(np.isfinite(paired_bootstrap(d,'a','b',100)['mae_gain_pp']), 'Bad bootstrap')
    print('PASS: Synthetic 210-variant, 168-position MAE / ranking / paired bootstrap')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--bootstraps',type=int,default=5000)
    args=ap.parse_args()
    if args.self_test:
        self_test();return
    require(args.bootstraps>=100,'Choose at least 100 bootstraps')

    # This function verifies the frozen manifests, training membership, and
    # obtains VTI TRAIN-only outcomes without reading protected assay labels.
    from cftr_theratype import model_zoo as mz
    from cftr_theratype.transfer import ROOT
    root=Path(ROOT)
    train, arrays, _, outer=mz.prepare(root)
    require(len(train)==210 and train.protein_position.nunique()==168, 'Frozen development set changed')
    require(len(outer)==10,'Frozen 2x5 outer folds changed')
    require(len(arrays['sequence_structure'])==210,'Invalid model feature shape')
    require(len(set(train.variant_key))==210,'Duplicate development IDs')

    fp=root/'results/model_zoo/phase17a_train_predictions_averaged.csv'
    require(fp.is_file(),f'Phase17A OOF file not found: {fp}')
    d=pd.read_csv(fp,dtype={'variant_key':str})
    req={'variant_key','model','protein_position','observed_pp','predicted_pp'}
    require(req.issubset(d.columns),f'Phase17A schema missing {req-set(d.columns)}')
    require(not d.duplicated(['model','variant_key']).any(),'Duplicate model/variant prediction')
    truth=train.set_index('variant_key').vti_rescue_pp.sort_index()
    positions=train.set_index('variant_key').protein_position.sort_index()
    for model,g in d.groupby('model'):
        gg=g.set_index('variant_key').sort_index()
        require(len(gg)==210 and gg.index.equals(truth.index),f'Incomplete {model} variant universe')
        require(np.array_equal(gg.protein_position.to_numpy(int),positions.to_numpy(int)),
                f'Position alignment mismatch for {model}')
        require(np.allclose(gg.observed_pp.to_numpy(float),truth.to_numpy(float),atol=1e-5,rtol=0),
                f'Observed total-rescue label mismatch for {model}')
    require(XGB in set(d.model),'Expected XGBoost 80-feature model absent in Phase17A OOF')

    print('='*105)
    print('PHASE 24A | TOTAL VTI FUNCTIONAL RESCUE — TRAIN-ONLY ENDPOINT & SCREENING AUDIT')
    print('='*105)
    print('210 variants / 168 positions | frozen Phase17A OOF (2x5 position-disjoint nested CV)')
    print('Total VTI rescue = VTI FSK+IVA activity − untreated DMSO FSK activity, in pp of WT')
    print('NOT acute IVA increment; NOT identical to treated activity or clinical response')
    print('No new fitting or model selection; neither 2024 nor 2026 protected labels loaded')
    print('\nA. RESPONSE DISTRIBUTION')
    y=truth.to_numpy(float)
    print(f'Min={y.min():.3f} Median={np.median(y):.3f} Mean={y.mean():.3f} '
          f'SD={y.std(ddof=1):.3f} Max={y.max():.3f} pp')
    for t in [0,10,20,40,80,120]:
        mask=y<t
        upos=positions.to_numpy(int)[mask]
        print(f'  response <{t:3d} pp: variants={int(mask.sum()):3d} '
              f'positions={len(np.unique(upos)):3d}')
    mask=y>=LIMIT
    print(f'  response >=10 pp: variants={int(mask.sum())} '
          f'positions with any response >=10={int(train.loc[train.vti_rescue_pp>=LIMIT,"protein_position"].nunique())}')
    print('\nBY COHORT')
    print(train.groupby('study_cohort').agg(
        variants=('variant_key','size'),positions=('protein_position','nunique'),
        median_rescue_pp=('vti_rescue_pp','median'),mean_rescue_pp=('vti_rescue_pp','mean'),
        nonresponse_lt10=('vti_rescue_pp',lambda s:int((s<LIMIT).sum())),
        gt40=('vti_rescue_pp',lambda s:int((s>40).sum()))).round(3).to_string())

    summary=mae_table(d)
    ranking=screening_by_position(d)
    print('\nB. FROZEN PHASE17A MODEL PERFORMANCE (TOTAL VTI RESCUE)')
    print(summary.head(15).round(4).to_string(index=False))
    print('\nC. TOTAL-RESCUE LOW-RESPONSE SCREENING | <10 pp (IN VITRO RESEARCH MARKER)')
    print('Position labelled nonresponsive only when ALL its training variants have rescue <10 pp.')
    print('Rank positions by NEGATIVE of maximum predicted rescue per position.')
    print(ranking.head(12).round(4).to_string(index=False))
    print('\nD. TARGETED XGBOOST VS OTHER MODEL COMPARISON | EXPLORATORY')
    paired=[]
    for candidate in FAMILIES[1:]:
        if candidate in set(d.model):
            paired.append(paired_bootstrap(d,XGB,candidate,args.bootstraps))
    print(pd.DataFrame(paired).round(4).to_string(index=False) if paired else
          'No prespecified paired comparator available')

    print('\nE. ETI-ASSAY-CONDITIONED TOTAL VTI RESCUE | EXACT MATCHED TRAIN ONLY')
    cross=root/'results/biology/phase23a/eti_vti_exact_train_matches.csv'
    conditional=pd.DataFrame()
    if cross.is_file():
        source=pd.read_csv(cross,dtype={'variant_key':str})
        require({'variant_key','protein_position','eti_total_fsk_iva_rescue_pp'}.issubset(source),
                'Matched ETI/VTI TRAIN audit lacks ETI total rescue')
        require(source.variant_key.is_unique and len(source)==66, 'Unexpected matched ETI universe')
        conditional=(source[['variant_key','protein_position','eti_total_fsk_iva_rescue_pp']]
                     .merge(train[['variant_key','protein_position','vti_rescue_pp']],
                            on='variant_key',validate='one_to_one', suffixes=('_old','_new')))
        require(np.array_equal(conditional.protein_position_old,conditional.protein_position_new),
                'Position mismatch in matched 2024/2026 records')
        require(len(conditional)==66 and conditional.protein_position_old.nunique()==56,
                'Unexpected paired scope')
        eti=conditional.eti_total_fsk_iva_rescue_pp.to_numpy(float)
        vti=conditional.vti_rescue_pp.to_numpy(float)
        require(np.isfinite(eti).all() and np.isfinite(vti).all(),'Nonfinite exact-match outcomes')
        print('Matched: 66 variants / 56 protein positions; ETI measurement is KNOWN at deployment')
        print(f'  Direct ETI -> VTI: variant MAE={np.mean(np.abs(eti-vti)):.3f} pp, '
              f'position MAE={position_mae(vti,eti,conditional.protein_position_old):.3f} pp, '
              f'Spearman rho={pd.Series(eti).corr(pd.Series(vti),method="spearman"):+.3f}')
        xgb=d.loc[d.model.eq(XGB),['variant_key','predicted_pp']].rename(
            columns={'predicted_pp':'mutation_only_pred_pp'})
        conditional=conditional.merge(xgb,on='variant_key',validate='one_to_one')
        print(f'  Mutation-only XGB on SAME subset: position MAE='
              f'{position_mae(vti,conditional.mutation_only_pred_pp,conditional.protein_position_old):.3f} pp')
        conditional['absolute_cross_regimen_difference_pp']=np.abs(eti-vti)
        print('  Largest ETI/VTI total-rescue disagreements:')
        print(conditional.sort_values('absolute_cross_regimen_difference_pp',ascending=False)
              [['variant_key','protein_position_old','eti_total_fsk_iva_rescue_pp',
                'vti_rescue_pp','absolute_cross_regimen_difference_pp']]
              .head(10).round(3).to_string(index=False))
        print('  CAVEAT: conditional ETI is a measured outcome; never feed it to zero-shot model.')
    else:
        print('Phase23A matched file missing; skip optional assay-conditioned audit.')

    out=root/'results/biology/phase24a'
    out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/'phase24a_model_summary.csv',index=False)
    ranking.to_csv(out/'phase24a_total_rescue_ranking.csv',index=False)
    pd.DataFrame(paired).to_csv(out/'phase24a_model_bootstrap.csv',index=False)
    if len(conditional):
        conditional.to_csv(out/'phase24a_conditional_eti_vti_total.csv',index=False)
    report={'phase':'24A_total_rescue_endpoint_audit','n_vti_train_variants':210,
            'n_vti_train_positions':168,'n_phase17a_models':int(d.model.nunique()),
            'nonresponse_threshold_pp':LIMIT,'n_bootstraps':int(args.bootstraps),
            'validation_test_outcomes_loaded':False,'no_new_model_fitting':True,
            'endpoint':'VTI treated FSK+IVA minus untreated DMSO FSK in %WT pp',
            'limitations':['Development OOF; multiple historical models previously examined',
                          '10 pp is an exploratory in-vitro marker not clinical response',
                          'ETI/VTI conditional subset is selected, not independent validation']}
    (out/'phase24a_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('\nOUTPUTS:',out)
    print('PASS: Phase24A used only frozen VTI TRAIN outcomes and stored position-held-out predictions')


if __name__=='__main__':
    main()
