#!/usr/bin/env python3
"""Phase 22B: High-responder ranking and cohort-shift evaluation, VTI TRAIN only.

Primary: predict acute IVA >20 percentage points at unseen CFTR protein positions.
20 pp is a research screening cutoff, NOT a clinical treatment threshold.

Reuses the FROZEN 2 x 5 outer protein-position folds, checks historical OOF
predictions, and trains two new nested classifiers (logistic and XGBoost)
with grouped 3-fold inner selection. Also trains VTI-only variants while
scoring on EXACTLY the same outer-held cases, to quantify cohort effects.

All model features are from the frozen sequence+structure matrices. Neither
study cohort nor assay outcomes nor measured maturation are prediction features.
No protected val/test/excluded outcome files are opened. Nothing changes the
frozen train/val/test membership.

Usage:
    python scripts/phase22b_high_responder_detection.py --self-test
    python scripts/phase22b_high_responder_detection.py --smoke
    python scripts/phase22b_high_responder_detection.py

Main outputs: results/biology/phase22b/*.csv and report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

CUTOFF = 20.0
RANDOM_STATE = 2222
LOGISTIC_C = (0.02, 0.2, 2.0)
XGB_DEPTH = (1, 2, 3)
MODES = ('logistic_all', 'xgb_class_all', 'xgb_class_vti_only')


def get_project():
    from cftr_theratype import model_zoo as mz
    from cftr_theratype.transfer import ROOT
    return mz, Path(ROOT)


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load_training(mz, root):
    train, mats, _, outer = mz.prepare(root)
    if len(train) != 210 or train.protein_position.nunique() != 168 or len(outer) != 10:
        raise AssertionError('Frozen VTI train/fold cardinalities unexpectedly changed')
    if train.variant_key.duplicated().any():
        raise AssertionError('Duplicate variant keys')
    X = np.asarray(mats['sequence_structure'], dtype=float)
    if X.shape != (210, 80) or np.isinf(X).any():
        raise AssertionError('Expected (210,80) mutation-derived features')
    target_path = root / 'results/boltz/phase19b/vti_train_acute_iva_targets.csv'
    targets = pd.read_csv(target_path, dtype={'variant_key': str})
    if len(targets) != 210 or targets.variant_key.duplicated().any():
        raise AssertionError('Training-only target table changed')
    labels = train[['variant_key']].merge(
        targets[['variant_key', 'acute_iva_vti_pp']], on='variant_key',
        how='left', sort=False, validate='one_to_one')
    if labels.variant_key.tolist() != train.variant_key.tolist() or labels.acute_iva_vti_pp.isna().any():
        raise AssertionError('Target alignment failed')
    y = labels.acute_iva_vti_pp.to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise AssertionError('Non-finite training outcomes')
    position = train.protein_position.to_numpy(dtype=int)
    if 'study_cohort' not in train.columns:
        raise AssertionError('Expected study_cohort in frozen train')
    cohort = train.study_cohort.astype(str).to_numpy()
    if set(cohort) != {'VTI', 'ETI, VTI'}:
        raise AssertionError(f'Unexpected train cohorts: {set(cohort)}')
    if int((cohort == 'VTI').sum()) != 144:
        raise AssertionError('Unexpected VTI-only count')
    for rep, fold, fit, held in outer:
        if set(position[fit]) & set(position[held]):
            raise AssertionError(f'Outer group leakage: {rep}/{fold}')
    return train, X, y, position, cohort, outer, target_path


def variant_to_position(frame, score='score'):
    """Position is the independent unit; positive if ANY tested substitution is >20.

    The max score is used because this is a position-screening question: is any
    substitution at this position a strong responder? Mutation-specific ranking
    is also reported separately on the original variant rows.
    """
    return (frame.groupby('protein_position', as_index=False)
            .agg(y=('is_high20', 'max'), score=(score, 'max'),
                 n_variants=('variant_key', 'size')))


def safe_ap(y, score):
    y = np.asarray(y, int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, score))


def top_budget(y, score, frac=0.2):
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    n = len(y)
    k = max(1, int(np.ceil(frac * n)))
    # Deterministic tie-breaking by original row order; scores cannot contain NaN.
    indices = np.argsort(-score, kind='stable')[:k]
    found = int(y[indices].sum())
    total = int(y.sum())
    return {'budget_n': k,
            'precision': float(found / k),
            'recall': float(found / total) if total else np.nan,
            'hits': found, 'total_positive': total}


def score_metrics(frame, model, level='position'):
    if level == 'position':
        z = variant_to_position(frame)
    elif level == 'variant':
        z = frame[['is_high20', 'score']].copy()
    else:
        raise ValueError(level)
    y = z['y'].to_numpy(int) if level == 'position' else z['is_high20'].to_numpy(int)
    s = z.score.to_numpy(float)
    if not np.isfinite(s).all():
        raise AssertionError(f'Non-finite scores for {model}')
    b10 = top_budget(y, s, 0.1)
    b20 = top_budget(y, s, 0.2)
    return dict(model=model, level=level, n=len(z), n_positive=int(y.sum()),
                prevalence=float(y.mean()),
                average_precision=safe_ap(y, s),
                roc_auc=float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan,
                precision_at_top10pct=b10['precision'], recall_at_top10pct=b10['recall'],
                precision_at_top20pct=b20['precision'], recall_at_top20pct=b20['recall'],
                hits_top20pct=b20['hits'], budget_top20pct=b20['budget_n'])


def load_historical(root, train, y, position, cohort):
    """Read only existing TRAIN OOF predictions; validate identity and targets."""
    sources = (
        ('historical_xgb', root / 'results/boltz/phase19c/acute_iva_vti_pp_oof.csv',
         'sequence_structure__xgboost'),
        ('reliability_weighted', root / 'results/boltz/phase19i/phase19i_oof.csv',
         'reliability_weighted'),
    )
    frames = []
    for name, path, source_model in sources:
        if not path.exists():
            print(f'WARNING: Missing historical comparison {path}; skipped', flush=True)
            continue
        df = pd.read_csv(path, dtype={'variant_key': str})
        if not {'model', 'variant_key', 'repeat', 'predicted_pp', 'observed_pp'}.issubset(df):
            raise AssertionError(f'Unexpected OOF schema: {path}')
        df = df.loc[df.model.eq(source_model)].copy()
        if len(df) != 420 or df.duplicated(['repeat','variant_key']).any():
            raise AssertionError(f'Expected 2 x 210 historical rows: {path}')
        lookup = pd.DataFrame({'variant_key': train.variant_key.astype(str),
                               'truth': y, 'protein_position': position,
                               'cohort': cohort})
        checked = df.merge(lookup, on='variant_key', validate='many_to_one',
                           suffixes=('', '_confirmed'))
        if not np.allclose(checked.observed_pp, checked.truth, atol=1e-5, rtol=0):
            raise AssertionError(f'Historical OOF labels mismatch {name}')
        if set(checked.variant_key) != set(lookup.variant_key):
            raise AssertionError('Historical keys mismatch')
        if 'protein_position' in df and not np.array_equal(
                checked.protein_position.to_numpy(int),
                checked.protein_position_confirmed.to_numpy(int)):
            raise AssertionError('Historical positions mismatch')
        if set(checked.repeat.astype(int)) != {1, 2}:
            raise AssertionError('Expected historical 2-repeat OOF')
        averaged = checked.groupby('variant_key', as_index=False).agg(
            score=('predicted_pp', 'mean'), n_repeats=('repeat', 'nunique'))
        if len(averaged) != 210 or not averaged.n_repeats.eq(2).all():
            raise AssertionError('Historical OOF aggregation mismatch')
        joined = lookup.merge(averaged[['variant_key','score']], on='variant_key',
                              sort=False, validate='one_to_one')
        joined['is_high20'] = (joined.truth > CUTOFF).astype(int)
        joined['model'] = name
        frames.append(joined)
    if not frames:
        raise FileNotFoundError('Need Phase19C historical OOF to assess regression ranking')
    return pd.concat(frames, ignore_index=True)


def classifier(kind, strength, seed, y):
    """A priori small model grid; no validation/test-based tuning."""
    y = np.asarray(y, int)
    if len(np.unique(y)) != 2:
        raise ValueError('Training subset must contain both responder classes')
    if kind == 'logistic':
        return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
                             StandardScaler(),
                             LogisticRegression(C=float(strength), class_weight='balanced',
                                                max_iter=5000, solver='liblinear',
                                                random_state=int(seed)))
    if kind == 'xgb_class':
        pos = int(y.sum())
        neg = len(y)-pos
        return XGBClassifier(max_depth=int(strength), n_estimators=140,
                             learning_rate=0.045, min_child_weight=3,
                             subsample=0.9, colsample_bytree=0.9,
                             reg_lambda=8.0, objective='binary:logistic',
                             eval_metric='logloss', tree_method='hist',
                             scale_pos_weight=float(np.clip(neg/max(pos,1), 1., 6.)),
                             n_jobs=1, random_state=int(seed))
    raise ValueError(kind)


def grouped_inner_select(X, y, pos, kind, seed, candidates):
    """Hyperparameters chosen from INNER fold OOF position-level AUPRC only."""
    y = np.asarray(y, int)
    pos = np.asarray(pos, int)
    if len(np.unique(pos[y == 1])) < 6:
        raise ValueError('Too few positive protein positions for grouped 3-fold CV')
    folds = list(StratifiedGroupKFold(n_splits=3, shuffle=True,
                                     random_state=int(seed)).split(X, y, pos))
    predictions = []
    for candidate in candidates:
        inner_pred = np.full(len(y), np.nan)
        for f, h in folds:
            if set(pos[f]) & set(pos[h]):
                raise AssertionError('Inner leakage')
            model = classifier(kind, candidate, seed, y[f])
            model.fit(X[f], y[f])
            inner_pred[h] = model.predict_proba(X[h])[:, 1]
        if not np.isfinite(inner_pred).all():
            raise AssertionError('Missing inner OOF predictions')
        records = pd.DataFrame({'variant_key': [str(i) for i in range(len(y))],
                                'protein_position':pos, 'is_high20':y,
                                'score':inner_pred})
        position_ap = score_metrics(records, 'inner', 'position')['average_precision']
        variant_ap = safe_ap(y, inner_pred)
        predictions.append({'candidate':float(candidate), 'inner_position_ap':position_ap,
                            'inner_variant_ap':variant_ap})
    # Tie break in original candidate order, not by outer-fold performance.
    best_idx = max(range(len(predictions)),
                   key=lambda i: (predictions[i]['inner_position_ap'], -i))
    return candidates[best_idx], predictions


def train_outer(train, X, y, pos, cohort, outer, smoke=False):
    rows = []
    selections = []
    for outer_i, (repeat, fold, fit, held) in enumerate(outer):
        if smoke and outer_i >= 1:
            break
        begin = perf_counter()
        fit = np.asarray(fit, int)
        held = np.asarray(held, int)
        for mode in MODES:
            vti_only = mode.endswith('_vti_only')
            kind = 'logistic' if mode.startswith('logistic') else 'xgb_class'
            fit_used = fit[cohort[fit] == 'VTI'] if vti_only else fit
            # IMPORTANT: Same held rows, even for VTI-only training, to expose
            # domain shift. Cohort-stratified results identify target population.
            if set(pos[fit_used]) & set(pos[held]):
                raise AssertionError('Outer group leakage')
            ybin = (y[fit_used] > CUTOFF).astype(int)
            candidates = LOGISTIC_C if kind == 'logistic' else XGB_DEPTH
            strength, candidate_results = grouped_inner_select(
                X[fit_used], ybin, pos[fit_used], kind,
                seed=RANDOM_STATE + 100*int(repeat) + int(fold), candidates=candidates)
            fitted = classifier(kind, strength, RANDOM_STATE + 100*int(repeat) + int(fold), ybin)
            fitted.fit(X[fit_used], ybin)
            p = fitted.predict_proba(X[held])[:,1]
            if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
                raise AssertionError('Invalid classifier probabilities')
            selections.append({'repeat':int(repeat), 'fold':int(fold), 'model':mode,
                               'n_fit_variants':len(fit_used),
                               'n_fit_positions':len(np.unique(pos[fit_used])),
                               'n_fit_positive_variants':int(ybin.sum()),
                               'selected_strength':float(strength),
                               'inner_candidates_json':json.dumps(candidate_results)})
            for idx, prob in zip(held, p):
                rows.append({'repeat':int(repeat), 'fold':int(fold),
                             'model':mode,'variant_key':str(train.variant_key.iloc[idx]),
                             'protein_position':int(pos[idx]),'cohort':str(cohort[idx]),
                             'observed_pp':float(y[idx]),
                             'is_high20':int(y[idx] > CUTOFF), 'score':float(prob)})
        print(f'  repeat={int(repeat)} fold={int(fold)} held={len(held)} '
              f'fit_all={len(fit)} elapsed={perf_counter()-begin:.1f}s', flush=True)
    result = pd.DataFrame(rows)
    select = pd.DataFrame(selections)
    if not smoke:
        assert len(result) == 210*2*len(MODES)
        assert not result.duplicated(['model','repeat','variant_key']).any()
        assert set(result.repeat) == {1, 2}
        assert len(select) == 10*len(MODES)
    else:
        assert len(result) == len(np.asarray(outer[0][3]))*len(MODES)
    return result, select


def average_oof(df, train, y, pos, cohort, smoke=False):
    if df.empty:
        raise AssertionError('Missing prediction rows')
    agg = (df.groupby(['model','variant_key'], as_index=False)
           .agg(score=('score', 'mean'), n_repeats=('repeat', 'nunique')))
    if not smoke and (not agg.n_repeats.eq(2).all() or len(agg) != len(MODES)*210):
        raise AssertionError('Not two OOF predictions per variant')
    identity = pd.DataFrame({'variant_key':train.variant_key.astype(str),
                             'protein_position':pos, 'truth':y, 'cohort':cohort})
    aligned = agg.merge(identity,on='variant_key',validate='many_to_one')
    aligned['is_high20']=(aligned.truth > CUTOFF).astype(int)
    return aligned


def cohort_metrics(averaged):
    out=[]
    for (model,cohort),f in averaged.groupby(['model','cohort'],sort=True):
        for level in ('position','variant'):
            m=score_metrics(f,model,level)
            out.append({'cohort':cohort,**m})
    return pd.DataFrame(out)


def paired_position_bootstrap(averaged, comparisons, n_boot=5000, seed=2240):
    """Cluster paired by position. AP is evaluated at the position level."""
    rng = np.random.default_rng(seed)
    by_model = {m:variant_to_position(g).set_index('protein_position')
                for m,g in averaged.groupby('model')}
    output=[]
    for model,baseline in comparisons:
        if model not in by_model or baseline not in by_model:
            continue
        a,b=by_model[model],by_model[baseline]
        if set(a.index) != set(b.index):
            raise AssertionError('Bootstrap comparator position mismatch')
        a=a.sort_index(); b=b.sort_index()
        if not np.array_equal(a.y.to_numpy(int), b.y.to_numpy(int)):
            raise AssertionError('Bootstrap truth mismatch')
        yy=a.y.to_numpy(int)
        pa=a.score.to_numpy(float)
        pb=b.score.to_numpy(float)
        observed=safe_ap(yy,pa)-safe_ap(yy,pb)
        diffs=[]
        for _ in range(n_boot):
            ids=rng.integers(0,len(yy),len(yy))
            if yy[ids].sum() in (0,len(yy)):
                continue
            diffs.append(safe_ap(yy[ids],pa[ids])-safe_ap(yy[ids],pb[ids]))
        lo,hi=np.quantile(diffs,[0.025,0.975])
        output.append({'model':model,'baseline':baseline,
                       'position_ap_gain':observed,'ci95_low':float(lo),
                       'ci95_high':float(hi),
                       'fraction_boot_positive':float(np.mean(np.asarray(diffs)>0)),
                       'n_boot':len(diffs)})
    return pd.DataFrame(output)


def synthetic_self_test():
    rng=np.random.default_rng(101)
    groups=np.repeat(np.arange(45),2)
    y=np.repeat((rng.random(45)<0.24).astype(int),2)
    X=rng.normal(size=(90,8)); X[:,0]+=y*1.7
    best,results=grouped_inner_select(X,y,groups,'logistic',100,LOGISTIC_C)
    assert best in LOGISTIC_C and len(results)==3
    best2,results2=grouped_inner_select(X,y,groups,'xgb_class',100,(1,2))
    assert best2 in (1,2) and len(results2)==2
    temp=pd.DataFrame({'variant_key':np.arange(90).astype(str),
                       'protein_position':groups,'is_high20':y,
                       'score':np.linspace(.01,.99,90)})
    for level in ('variant','position'):
        metric=score_metrics(temp,'synthetic',level)
        assert 0<=metric['average_precision']<=1
        assert 0<=metric['recall_at_top20pct']<=1
    table= pd.concat([temp.assign(model='a'),
                      temp.assign(model='b',score=1-temp.score)])
    boot=paired_position_bootstrap(table,[('a','b')],n_boot=100)
    assert len(boot)==1 and np.isfinite(boot.ci95_low.iloc[0])
    print('PASS: Nested group folds, classifier fitting, position ranking, paired bootstrap', flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--bootstrap',type=int,default=5000)
    args=parser.parse_args()
    if args.self_test:
        synthetic_self_test();return
    mz,root=get_project()
    train,X,y,pos,cohort,outer,targets=load_training(mz,root)
    print('='*105, flush=True)
    print('PHASE 22B | HIGH-RESPONDER RANKING & COHORT-SHIFT TEST', flush=True)
    print('='*105, flush=True)
    print('Frozen VTI TRAIN:',len(y),'variants /',len(np.unique(pos)),'positions',flush=True)
    print('Acute IVA >20 pp:',int((y>20).sum()),'variants /',len(np.unique(pos[y>20])),'positions', flush=True)
    print('Research cutoff only. No protected val/test/excluded outcomes opened.',flush=True)
    print('Cohort provenance is used for stratification and training exclusion, NOT as a predictor.',flush=True)
    print('Fixed screening budget: top 20% of variants / protein positions.', flush=True)
    historical=load_historical(root,train,y,pos,cohort)
    print('\nHISTORICAL FROZEN OOF RANKING (2 REPEATS AVERAGED)',flush=True)
    for level in ('position','variant'):
        past=pd.DataFrame([score_metrics(m,model,level)
                           for model,m in historical.groupby('model')])
        print('\n'+level.upper(),flush=True)
        print(past[['model','n','n_positive','prevalence','average_precision',
                    'roc_auc','precision_at_top20pct','recall_at_top20pct']]
              .round(4).to_string(index=False),flush=True)
    print('\nNESTED GROUPED CLASSIFIER TRAINING',flush=True)
    predictions,selections=train_outer(train,X,y,pos,cohort,outer,smoke=args.smoke)
    if args.smoke:
        print('PASS: Smoke outer fold, no output files written',flush=True);return
    fresh=average_oof(predictions,train,y,pos,cohort)
    combined=pd.concat([historical,fresh],ignore_index=True,sort=False)
    if combined.duplicated(['model','variant_key']).any():
        raise AssertionError('Aggregated model/variant scores not unique')
    reports=[]
    for model,g in combined.groupby('model',sort=True):
        for level in ('position','variant'):
            reports.append(score_metrics(g,model,level))
    results=pd.DataFrame(reports)
    print('\n'+'='*105,flush=True)
    print('PHASE 22B | ALL TRAINING VARIANTS, REPEATED OUT-OF-FOLD RANKING',flush=True)
    print('='*105,flush=True)
    for level in ('position','variant'):
        print('\n'+level.upper()+' (independent positions preferred)',flush=True)
        table=results.loc[results.level.eq(level)].copy()
        print(table[['model','n','n_positive','prevalence','average_precision',
                     'roc_auc','precision_at_top10pct','recall_at_top10pct',
                     'precision_at_top20pct','recall_at_top20pct','hits_top20pct']]
              .sort_values('average_precision',ascending=False).round(4).to_string(index=False),flush=True)
    strata=cohort_metrics(combined)
    print('\nCOHORT STRATIFICATION (POSITION-LEVEL; position definition is within cohort)',flush=True)
    print(strata.loc[strata.level.eq('position'),
                     ['model','cohort','n','n_positive','prevalence',
                      'average_precision','recall_at_top20pct','precision_at_top20pct']]
          .sort_values(['cohort','model']).round(4).to_string(index=False),flush=True)
    pairs=[('logistic_all','historical_xgb'),
           ('xgb_class_all','historical_xgb'),
           ('xgb_class_vti_only','xgb_class_all'),
           ('xgb_class_all','reliability_weighted'),
           ('xgb_class_vti_only','historical_xgb')]
    boot=paired_position_bootstrap(combined,pairs,n_boot=args.bootstrap)
    print('\nPAIRED POSITION BOOTSTRAP: DIFFERENCE IN AVERAGE PRECISION (exploratory)',flush=True)
    print(boot.round(4).to_string(index=False),flush=True)
    out=root/'results/biology/phase22b';out.mkdir(parents=True,exist_ok=True)
    paths={'new_oof':out/'phase22b_classifier_oof.csv',
           'averaged_scores':out/'phase22b_all_averaged_scores.csv',
           'ranking_summary':out/'phase22b_ranking_summary.csv',
           'cohort_summary':out/'phase22b_cohort_summary.csv',
           'inner_selection':out/'phase22b_inner_selection.csv',
           'paired_bootstrap':out/'phase22b_paired_bootstrap.csv',
           'report':out/'phase22b_report.json'}
    predictions.to_csv(paths['new_oof'],index=False)
    combined.to_csv(paths['averaged_scores'],index=False)
    results.to_csv(paths['ranking_summary'],index=False)
    strata.to_csv(paths['cohort_summary'],index=False)
    selections.to_csv(paths['inner_selection'],index=False)
    boot.to_csv(paths['paired_bootstrap'],index=False)
    report={'phase':'22B_high20_screening_cohort_shift',
            'cutoff_acute_iva_pp':CUTOFF,
            'cutoff_is_clinical':False,
            'fixed_screening_budget_percent':20,
            'n_train_variants':len(y),'n_train_positions':int(len(np.unique(pos))),
            'n_positive_variants':int((y>CUTOFF).sum()),
            'n_positive_positions':int(len(np.unique(pos[y>CUTOFF]))),
            'frozen_outer_folds':len(outer),'inner_group_folds':3,
            'training_label_sha256':sha256(targets),
            'protected_outcomes_read':False,
            'note':'Exploratory repeat-OOF comparisons after multiple experiments; no independent validation or clinical threshold claim.'}
    paths['report'].write_text(json.dumps(report,indent=2)+'\n')
    print('\nOUTPUTS:',flush=True)
    for k,p in paths.items():print('  '+k+':',p.relative_to(root),flush=True)
    print('\nINTERPRETATION GUARDRAILS:',flush=True)
    print('  - Compare AUPRC to class prevalence; top-20% recall must be accompanied by precision.',flush=True)
    print('  - A classifier can rank well without predicting accurate response magnitude.',flush=True)
    print('  - Bootstrap intervals are exploratory: model families were considered after prior experiments.',flush=True)
    print('  - Cohort composition is an observed selection difference, not a demonstrated assay batch effect.',flush=True)
    print('  - Protein positions in CV are disjoint; val/test remain completely sealed.',flush=True)
    print('PASS: Phase22B completed; frozen VTI TRAIN outcomes only',flush=True)

if __name__=='__main__':
    main()
