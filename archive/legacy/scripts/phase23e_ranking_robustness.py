#!/usr/bin/env python3
"""Phase 23E: training-only paired ranking robustness audit for Phase 23D.

Consumes ONLY repeated OOF predictions from frozen VTI TRAIN, plus optional
historical Phase 19I TRAIN OOF predictions. Never reads protected labels.
This is a descriptive development audit, NOT independent validation.

Run: python scripts/phase23e_ranking_robustness.py --bootstraps 10000
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_SOURCE = Path('results/biology/phase23d/phase23d_averaged.csv')
DEFAULT_HIST = Path('results/boltz/phase19i/phase19i_averaged.csv')
DEFAULT_OUT = Path('results/biology/phase23e')
REQUIRED = {'variant_key', 'model', 'protein_position', 'observed_pp', 'predicted_pp'}
CUTOFF = 20.0


def check(value, message):
    if not value:
        raise AssertionError(message)


def load_data(path: Path, hist_path: Path, include_historical: bool) -> pd.DataFrame:
    check(path.is_file(), f'Missing Phase 23D file: {path}')
    x = pd.read_csv(path, dtype={'variant_key':str})
    check(REQUIRED.issubset(x.columns), f'Missing columns: {sorted(REQUIRED-set(x.columns))}')
    check(not x.duplicated(['model', 'variant_key']).any(), 'Duplicated variant/model rows')
    check(x.model.nunique() >= 3, 'Expected multiple Phase 23D prediction models')
    control = x[x.model.eq('control80')].copy()
    check(len(control) == 210 and control.variant_key.nunique() == 210,
          'Expected 210 VTI TRAIN variants for control80')
    check(control.protein_position.nunique() == 168, 'Expected 168 protein positions')
    ref = control.set_index('variant_key')[['protein_position', 'observed_pp']].sort_index()
    for name, g in x.groupby('model'):
        check(len(g)==210 and set(g.variant_key)==set(control.variant_key),
              f'{name}: VTI TRAIN variants do not match control')
        t=g.set_index('variant_key').loc[ref.index]
        check(np.array_equal(t.protein_position.to_numpy(), ref.protein_position.to_numpy()),
              f'{name}: protein position mismatch')
        check(np.allclose(t.observed_pp.to_numpy(), ref.observed_pp.to_numpy(), rtol=0, atol=1e-9),
              f'{name}: truth mismatch')
    if include_historical and hist_path.is_file():
        h=pd.read_csv(hist_path, dtype={'variant_key':str})
        check({'variant_key','model','predicted_pp'}.issubset(h.columns), 'Historical schema mismatch')
        h=h[h.model.eq('reliability_weighted')].copy()
        check(len(h)==210 and set(h.variant_key)==set(ref.index), 'Historical score keys mismatch')
        m=control[['variant_key','protein_position','observed_pp']].merge(
            h[['variant_key','predicted_pp']], on='variant_key', how='inner',validate='one_to_one')
        m['model']='reliability_weighted_19i'
        x=pd.concat([x,m],ignore_index=True)
    check(np.isfinite(x[['observed_pp','predicted_pp']].to_numpy(float)).all(),
          'Missing or infinite outcome/predictions')
    return x


def position_table(x: pd.DataFrame):
    # Same definition used by Phase 23D: a position is positive if ANY variant at
    # the position has observed response >20; position score=max prediction.
    z=x.groupby(['model','protein_position'],as_index=False).agg(
        observed_max_pp=('observed_pp','max'), predicted_max_pp=('predicted_pp','max'),
        n_variants=('variant_key','nunique'))
    check(not z.duplicated(['model','protein_position']).any(), 'Position duplication')
    truth=z[z.model.eq('control80')].set_index('protein_position').observed_max_pp.sort_index()
    scores={}
    for name,g in z.groupby('model'):
        q=g.set_index('protein_position').sort_index()
        check(q.index.equals(truth.index), f'{name}: position universe changed')
        check(np.allclose(q.observed_max_pp.to_numpy(),truth.to_numpy(),atol=1e-9),
              f'{name}: inconsistent observed position labels')
        scores[name]=q.predicted_max_pp.to_numpy(float)
    labels=truth.to_numpy(float)>CUTOFF
    check(int(labels.sum())==32, 'Expected 32 strong-response positions')
    return truth.index.to_numpy(int), labels, scores, z


def metrics(labels, scores):
    n=len(labels); k10=int(np.ceil(.10*n)); k20=int(np.ceil(.20*n))
    order=np.argsort(-scores,kind='stable')
    hit10=int(labels[order[:k10]].sum()); hit20=int(labels[order[:k20]].sum())
    total=int(labels.sum())
    return {'n_positions':n,'n_positive':total,'prevalence':float(labels.mean()),
            'average_precision':float(average_precision_score(labels,scores)),
            'roc_auc':float(roc_auc_score(labels,scores)),
            'top10_hits':hit10,'top10_precision':hit10/k10,'top10_recall':hit10/total,
            'top20_screened':k20,'top20_hits':hit20,'top20_precision':hit20/k20,
            'top20_recall':hit20/total,'top20_enrichment':(hit20/k20)/labels.mean()}


def stratified_position_bootstrap(labels, scores, pairs, B, seed):
    # Fixed-prevalence paired bootstrap. Keeping the same resampled positions
    # across models preserves pairing and ensures both classes each draw.
    positives=np.flatnonzero(labels)
    negatives=np.flatnonzero(~labels)
    rng=np.random.default_rng(seed)
    out={pair:[] for pair in pairs}
    for _ in range(B):
        inds=np.concatenate([rng.choice(positives,len(positives),replace=True),
                             rng.choice(negatives,len(negatives),replace=True)])
        yb=labels[inds]
        calculated={}
        for name in {v for pair in pairs for v in pair}:
            calculated[name]=metrics(yb,scores[name][inds])
        for pair in pairs:
            a,b=pair
            out[pair].append((
                calculated[a]['average_precision']-calculated[b]['average_precision'],
                calculated[a]['roc_auc']-calculated[b]['roc_auc'],
                calculated[a]['top20_recall']-calculated[b]['top20_recall']))
    rows=[]
    for (a,b),data in out.items():
        arr=np.asarray(data,float)
        actual_a=metrics(labels,scores[a]); actual_b=metrics(labels,scores[b])
        for j,(name,key) in enumerate([
                ('average_precision','average_precision'),
                ('roc_auc','roc_auc'),
                ('top20_recall','top20_recall')]):
            dif=arr[:,j]
            rows.append({'model':a,'comparator':b,'metric':name,
                         'observed_gain':actual_a[key]-actual_b[key],
                         'ci95_low':float(np.quantile(dif,.025)),
                         'ci95_high':float(np.quantile(dif,.975)),
                         'fraction_boot_positive':float(np.mean(dif>0)),
                         'bootstrap_method':'stratified paired resampling of protein positions',
                         'n_bootstraps':B})
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',type=Path,default=DEFAULT_SOURCE)
    ap.add_argument('--historical',type=Path,default=DEFAULT_HIST)
    ap.add_argument('--out',type=Path,default=DEFAULT_OUT)
    ap.add_argument('--bootstraps',type=int,default=10000)
    ap.add_argument('--seed',type=int,default=232323)
    ap.add_argument('--no-historical',action='store_true')
    args=ap.parse_args()
    check(args.bootstraps>=100, 'Require >=100 bootstrap iterations')
    df=load_data(args.source,args.historical,not args.no_historical)
    positions, labels, scores, pos_df = position_table(df)
    pairs=[('teacher_multi','control80'),('teacher_multi','teacher_multi_shuffle'),
           ('teacher_acute','control80'),('teacher_multi_novel','control80')]
    if 'reliability_weighted_19i' in scores:
        pairs.append(('teacher_multi','reliability_weighted_19i'))
    for a,b in pairs:
        check(a in scores and b in scores, f'Missing model pair: {a} vs {b}')
    summary=pd.DataFrame([{'model':name,**metrics(labels,sc)} for name,sc in scores.items()])
    summary=summary.sort_values('average_precision',ascending=False)
    boot=stratified_position_bootstrap(labels,scores,pairs,args.bootstraps,args.seed)
    k=int(np.ceil(.20*len(positions)))
    base=set(positions[np.argsort(-scores['control80'],kind='stable')[:k]])
    multi=set(positions[np.argsort(-scores['teacher_multi'],kind='stable')[:k]])
    label_by_position=dict(zip(positions,labels))
    entered=sorted(multi-base)
    exited=sorted(base-multi)
    entered_good=[p for p in entered if label_by_position[p]]
    exited_good=[p for p in exited if label_by_position[p]]
    print('='*105)
    print('PHASE 23E | PAIRED RANKING ROBUSTNESS — DEVELOPMENT-SET OOF ONLY')
    print('='*105)
    print(f'{len(positions)} independent VTI-TRAIN positions; {int(labels.sum())} >20 pp; top-20% budget={k}')
    print('The >20 pp threshold is exploratory, NOT a clinical treatment criterion.')
    print('No val/test/excluded outcomes read. No model fitting or selection performed.')
    print('\nPOSITION-LEVEL RANKING')
    print(summary.round(4).to_string(index=False))
    print('\nPAIRED POSITION BOOTSTRAP — DIFFERENCES (fixed prevalence; unadjusted exploratory CI)')
    print(boot.round(4).to_string(index=False))
    print('\nTOP-20% SCREENING DISAGREEMENTS: teacher_multi vs control80')
    print(f'Overlapping selected positions: {len(multi&base)}/{k}')
    print(f'Positions entering teacher list: {entered}')
    print(f'Of those, actual >20 pp: {entered_good}')
    print(f'Positions leaving teacher list: {exited}')
    print(f'Of those, actual >20 pp: {exited_good}')
    print('\nCAVEATS')
    print('- Development outcomes have informed multiple past model choices; no significance claim.')
    print('- 2 repeated CV predictions per position are correlated; this is not external validation.')
    print('- Bootstrap fixes observed positive prevalence and does not account for modelling selection.')
    print('- Even a positive exploratory interval is NOT evidence of generalisation to held-out test.')
    args.out.mkdir(parents=True,exist_ok=True)
    p1=args.out/'phase23e_ranking_summary.csv'
    p2=args.out/'phase23e_paired_ranking_bootstrap.csv'
    p3=args.out/'phase23e_position_predictions.csv'
    summary.to_csv(p1,index=False);boot.to_csv(p2,index=False)
    # Only TRAIN outcomes, grouped to position. Wide format helps audit top-20% disagreement.
    pred=pd.DataFrame({'protein_position':positions,'strong20':labels.astype(int)})
    for name,sc in scores.items():
        pred[f'{name}_score_pp']=sc
    pred.to_csv(p3,index=False)
    print('\nOUTPUTS:',p1,p2,p3,sep='\n  ')
    print('PASS: Training-only Phase 23E completed.')


if __name__=='__main__':
    main()
