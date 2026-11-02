#!/usr/bin/env python3
"""Phase 31K: TRAIN-only calibration, component, tail-weighting and reliability study.

Run from CFTR-Theratype repo root, with this file in scripts/:
    python -u scripts/phase31k_response_generalisation.py analyze
    python -u scripts/phase31k_response_generalisation.py fit --quick
    python -u scripts/phase31k_response_generalisation.py fit

Sources of VTI outcomes: canonical Phase26D.load_train() (TRAIN-only), and its
frozen Phase26G/31I.2 OOF files. Uses frozen two-repeat, position-grouped folds.
No external VTI VAL/TEST outcome labels are read. Observed-response subgroup
results are retrospective diagnostics, never model inputs at deployment.

The aim is diagnostic and exploratory. Repeated frozen TRAIN folds do not make
this an independent validation study; do not use the final TEST set for tuning.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / 'scripts'))
OUT = ROOT / 'results/biology/phase31k'
ALPHAS = (0.0, 0.75, 1.5)  # predeclared BEFORE looking at Phase31K results
MODELS = ('M0', 'T1_mild', 'T2_strong', 'T3_inner_selected')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_data():
    import phase26d_corrector_bottleneck as prior
    import phase31j_error_diagnostics as phase31j
    d, features, _, folds, frozen_source = prior.load_train()
    if len(d) != 210 or d.protein_position.nunique() != 168 or not d.transfer_split.eq('train').all():
        raise AssertionError('Frozen 210-variant / 168-position TRAIN contract failed')
    oof, checks = phase31j.load_m0(d)  # validates Phase26G <-> Phase31I baseline parity
    keys = d.variant_key.astype(str).to_numpy()
    pos = d.protein_position.to_numpy(int)
    index = {(int(r), int(f)): (set(tr), set(te)) for r, f, tr, te in folds}
    if len(index) != 10 or len(folds) != 10:
        raise AssertionError('Expected exactly two repeats x five frozen folds')
    for (r, f), (trkeys, tekeys) in index.items():
        if not trkeys.isdisjoint(tekeys) or trkeys | tekeys != set(keys):
            raise AssertionError(f'Incomplete/overlapping frozen fold {r}/{f}')
        tr = np.isin(keys, list(trkeys)); te = np.isin(keys, list(tekeys))
        if set(pos[tr]) & set(pos[te]):
            raise AssertionError(f'Position leakage: {r}/{f}')
        z = oof[(oof.target == 'corrector') & (oof.repeat == r) & (oof.fold == f)]
        if set(z.variant_key) != tekeys:
            raise AssertionError(f'Frozen fold {r}/{f} and OOF membership disagree')
    x = np.asarray(features['sequence_structure'], dtype=float)
    if x.shape != (210, 80) or np.isinf(x).any():
        raise AssertionError(f'Frozen 80-feature matrix missing or malformed: {x.shape}')
    return d, x, folds, oof, checks, frozen_source


def per_position_error(frame):
    """Equal-weight protein positions, averaging variants and repeats."""
    z = frame.copy()
    z['ae'] = np.abs(z.observed - z.prediction)
    z['se'] = (z.observed - z.prediction) ** 2
    z['err'] = z.prediction - z.observed
    return (z.groupby(['repeat', 'protein_position'])[['ae', 'se', 'err']]
             .mean().groupby('protein_position').mean())


def boot_ci(paired_position_values, nboot=3000, seed=3117):
    vals = np.asarray(paired_position_values, dtype=float)
    if not np.isfinite(vals).all() or len(vals) < 2:
        raise AssertionError('Invalid position bootstrap values')
    rng = np.random.default_rng(seed)
    reps = []
    for start in range(0, nboot, 128):
        n = min(128, nboot - start)
        ix = rng.integers(len(vals), size=(n, len(vals)))
        reps.extend(vals[ix].mean(axis=1))
    return float(np.quantile(reps, .025)), float(np.quantile(reps, .975))


def position_quartiles(series):
    if len(series) < 4:
        raise ValueError('At least four independent positions needed for quartiles')
    ranks = series.rank(method='first')
    return pd.qcut(ranks, 4, labels=['Q1', 'Q2', 'Q3', 'Q4']).astype(str)


def calibration(oof, nboot=3000, seed=3117):
    """Predicted-response bins are deployable, observed-response bins diagnostic.

    Calibrations use OOF predictions. Fitted lines are DESCRIPTIVE and never
    applied to the same OOF predictions as a claimed improvement.
    """
    pos_rows = []
    metrics = []
    for target, z in oof.groupby('target'):
        # Position-level means: each protein position is one unit per repeat.
        positions = (z.groupby(['repeat', 'protein_position'], as_index=False)
                     [['observed', 'prediction']].mean())
        for rep, sub in positions.groupby('repeat'):
            sub = sub.copy()
            sub['predicted_quartile'] = position_quartiles(sub.prediction)
            sub['observed_quartile'] = position_quartiles(sub.observed)
            sub['signed_error'] = sub.prediction - sub.observed
            sub['abs_error'] = (sub.prediction - sub.observed).abs()
            sub['target'] = target
            pos_rows.append(sub)
            observed, pred = sub.observed.to_numpy(), sub.prediction.to_numpy()
            slope = (float(np.cov(pred, observed, ddof=0)[0, 1] / np.var(pred))
                     if np.var(pred) > 1e-12 else np.nan)
            intercept = float(np.mean(observed) - slope * np.mean(pred)) if np.isfinite(slope) else np.nan
            metrics.append(dict(target=target, repeat=int(rep), n_positions=len(sub),
                                mean_observed=float(observed.mean()), mean_predicted=float(pred.mean()),
                                observed_sd=float(observed.std(ddof=0)),
                                predicted_sd=float(pred.std(ddof=0)),
                                sd_ratio_pred_over_obs=float(pred.std(ddof=0) / observed.std(ddof=0)) if observed.std(ddof=0) > 0 else np.nan,
                                descriptive_observed_on_pred_slope=slope,
                                descriptive_observed_on_pred_intercept=intercept,
                                spearman=float(spearmanr(observed, pred).statistic)))
    bypos = pd.concat(pos_rows, ignore_index=True)
    summary = []
    for (target, typ, group), sub in pd.concat([
            bypos.assign(bin_type='predicted_quartile', bin=bypos.predicted_quartile),
            bypos.assign(bin_type='observed_quartile_RETROSPECTIVE', bin=bypos.observed_quartile)
            ]).groupby(['target', 'bin_type', 'bin']):
        # Bin membership may change across repeats; cluster by original positions.
        per = sub.groupby('protein_position')[['signed_error', 'abs_error']].mean()
        lo, hi = boot_ci(per.signed_error, nboot, seed)
        summary.append(dict(target=target, bin_type=typ, bin=group,
                            n_position_repeat_rows=len(sub), n_independent_positions=len(per),
                            mean_observed=float(sub.observed.mean()),
                            mean_predicted=float(sub.prediction.mean()),
                            mean_signed_bias_pp=float(per.signed_error.mean()),
                            position_mae_pp=float(per.abs_error.mean()),
                            bias_ci95_low=lo, bias_ci95_high=hi))
    return bypos, pd.DataFrame(metrics), pd.DataFrame(summary)


def component_decomposition(oof):
    """Exact e_total = e_corrector + e_implied_acute for separate direct predictors.

    Implied acute is NOT an independently trained acute model.
    """
    a = oof[oof.target == 'corrector'].drop(columns=['target', 'model']).rename(
        columns={'observed': 'observed_corrector', 'prediction': 'prediction_corrector'})
    b = oof[oof.target == 'total'].drop(columns=['target', 'model']).rename(
        columns={'observed': 'observed_total', 'prediction': 'prediction_total'})
    z = a.merge(b, on=['repeat', 'fold', 'variant_key', 'protein_position'],
                how='inner', validate='one_to_one')
    if len(z) != len(a):
        raise AssertionError('Missing paired corrector/total rows')
    z['e_cor'] = z.prediction_corrector - z.observed_corrector
    z['e_total'] = z.prediction_total - z.observed_total
    z['e_implied_acute'] = z.e_total - z.e_cor
    z['sq_cor'] = z.e_cor ** 2
    z['sq_implied_acute'] = z.e_implied_acute ** 2
    z['twice_cross'] = 2 * z.e_cor * z.e_implied_acute
    z['sq_total'] = z.e_total ** 2
    if not np.allclose(z.sq_total, z.sq_cor + z.sq_implied_acute + z.twice_cross, atol=1e-8):
        raise AssertionError('MSE component identity failed')
    q = z.groupby(['repeat', 'protein_position'])[
        ['sq_cor', 'sq_implied_acute', 'twice_cross', 'sq_total',
         'e_cor', 'e_implied_acute', 'e_total']].mean()
    pieces = q.mean()
    e = q[['e_cor', 'e_implied_acute', 'e_total']]
    result = dict(independent_positions=int(z.protein_position.nunique()),
                  n_repeat_rows=int(len(z)),
                  position_weighted_mse_corrector=float(pieces.sq_cor),
                  position_weighted_mse_implied_acute=float(pieces.sq_implied_acute),
                  position_weighted_twice_cross_term=float(pieces.twice_cross),
                  position_weighted_mse_total=float(pieces.sq_total),
                  identity_absolute_difference=float(abs(pieces.sq_total - pieces.sq_cor - pieces.sq_implied_acute - pieces.twice_cross)),
                  signed_bias_corrector=float(e.e_cor.mean()),
                  signed_bias_implied_acute=float(e.e_implied_acute.mean()),
                  signed_bias_total=float(e.e_total.mean()),
                  note='Implied acute is computed from two separate direct predictors, not an independently validated acute predictor.')
    return z, result


def historical_independent_components(path, oof):
    """Optional historical Phase26D *independently modelled* acute/corrector diagnostic.

    This is NOT treated as Phase26G M0 because Phase26D may select different
    hyperparameters. Phase26D's saved rows must match frozen OOF labels/keys.
    """
    if not path.is_file():
        return None, None
    cols = ['repeat', 'fold', 'variant_key', 'protein_position', 'observed_total',
            'observed_corrector', 'observed_acute', 'pred_zero_corrector_seq80',
            'pred_zero_acute_seq80', 'pred_zero_total_seq80', 'pred_zero_components_seq80']
    a = pd.read_csv(path, usecols=cols)
    ref = oof[oof.target == 'total'][['repeat', 'fold', 'variant_key', 'protein_position', 'observed']]
    j = ref.merge(a, on=['repeat', 'fold', 'variant_key', 'protein_position'], validate='one_to_one')
    if len(j) != len(ref) or not np.allclose(j.observed, j.observed_total, atol=1e-6):
        raise AssertionError('Phase26D reference target/folds not aligned')
    if not np.allclose(j.observed_total, j.observed_corrector + j.observed_acute, atol=1e-6):
        raise AssertionError('Phase26D observed component identity failed')
    if not np.allclose(j.pred_zero_components_seq80,
                       j.pred_zero_corrector_seq80 + j.pred_zero_acute_seq80, atol=1e-5):
        raise AssertionError('Phase26D independently-modelled component identity failed')
    out = []
    for name, col, target in [
        ('independent_corrector', 'pred_zero_corrector_seq80', 'observed_corrector'),
        ('independent_acute', 'pred_zero_acute_seq80', 'observed_acute'),
        ('independent_components_sum', 'pred_zero_components_seq80', 'observed_total'),
        ('phase26d_direct_total', 'pred_zero_total_seq80', 'observed_total')]:
        z = j.copy()
        z['ae'] = (z[col] - z[target]).abs()
        metric = z.groupby(['repeat', 'protein_position']).ae.mean().mean()
        out.append(dict(model=name, target=target, position_mae_pp=float(metric),
                        note='Phase26D historical; different tuning from Phase26G M0'))
    return j, pd.DataFrame(out)


def tail_weights(y, pos, alpha):
    """Computed exclusively from the current FIT partition's labels.

    Equal-position distribution determines the median and IQR; the model's
    variant-level weights scale with distance from this FIT-only median.
    """
    if alpha < 0:
        raise ValueError('alpha must be nonnegative')
    if alpha == 0:
        return np.ones(len(y), float)
    yy = np.asarray(y, float)
    gg = np.asarray(pos)
    s = pd.DataFrame({'y': yy, 'p': gg}).groupby('p').y.mean().to_numpy()
    median = float(np.median(s))
    iqr = float(np.quantile(s, .75) - np.quantile(s, .25))
    scale = max(iqr, 1e-6)
    z = np.clip(np.abs(yy - median) / scale, 0, 2.0)
    w = 1 + alpha * z
    return w / np.mean(w)


def fit_weighted(kind, spec, x, y, pos, alpha):
    import phase26g_feature_model_ablation as p26g
    est = p26g.make_model('xgb', spec)
    if float(alpha) == 0.0:
        est.fit(x, y)  # exact Phase26G parity; do NOT pass all-ones sample_weight
    else:
        # Pipeline: sample_weight belongs to its regression step.
        est.fit(x, y, regression__sample_weight=tail_weights(y, pos, alpha))
    return est


def inner_select(x, y, pos, spec, alphas=ALPHAS, cv=3):
    """Select alpha using held-out TRAIN positions only. Minimise position MAE."""
    results = []
    for alpha in alphas:
        pred = np.full(len(y), np.nan)
        for tr, te in GroupKFold(n_splits=cv).split(x, y, pos):
            if set(pos[tr]) & set(pos[te]):
                raise AssertionError('Inner position leakage')
            model = fit_weighted('xgb', spec, x[tr], y[tr], pos[tr], alpha)
            pred[te] = model.predict(x[te])
        if not np.isfinite(pred).all():
            raise AssertionError('Incomplete inner CV predictions')
        score = float(pd.DataFrame({'p': pos, 'ae': abs(y-pred)}).groupby('p').ae.mean().mean())
        results.append((float(alpha), score))
    # Select smaller alpha on a tie; no outer held-out information enters.
    choice = min(results, key=lambda x: (x[1], x[0]))[0]
    return choice, results


def choices_from_phase26g(target, folds):
    path = ROOT / f'results/biology/phase26g/phase26g_{target}_selected.csv'
    if not path.is_file():
        raise FileNotFoundError(f'Missing frozen Phase26G selection: {path}')
    d = pd.read_csv(path)
    d = d[d.model.eq('xgb_80')]
    from phase26g_feature_model_ablation import PARAMS
    choices = {}
    for r in d.itertuples(index=False):
        key = (int(r.repeat), int(r.fold))
        spec = ast.literal_eval(str(r.selected_param))
        if key in choices or not isinstance(spec, tuple) or spec not in PARAMS['xgb']:
            raise AssertionError(f'Invalid/duplicated frozen XGB parameter selection {key}: {spec}')
        choices[key] = spec
    expected = {(int(r), int(f)) for r, f, *_ in folds}
    if set(choices) != expected:
        raise AssertionError(f'Frozen selection keys mismatch for {target}; missing={expected-set(choices)}')
    return choices, sha256(path)


def nearest_position_distances(xtrain, ptr, xheld, k=5):
    """Only FIT-fold mutation descriptors used to define preprocessing/distance.

    Returns mean distance to k *distinct* FIT positions for each held variant.
    Treat as a heuristic reliability signal, NOT predictive uncertainty.
    """
    med = np.nanmedian(np.where(np.isfinite(xtrain), xtrain, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0)
    a = np.where(np.isfinite(xtrain), xtrain, med)
    b = np.where(np.isfinite(xheld), xheld, med)
    center = np.mean(a, axis=0)
    scale = np.std(a, axis=0)
    scale = np.where(scale > 1e-9, scale, 1)
    a = (a-center)/scale; b=(b-center)/scale
    out = []
    pos = np.unique(ptr)
    for row in b:
        dist = np.sqrt(np.mean((a-row)**2, axis=1))
        best = pd.DataFrame({'p': ptr, 'dist': dist}).groupby('p').dist.min().nsmallest(k)
        out.append(float(best.mean()))
    if len(pos) < k:
        raise AssertionError('Too few training positions for k-NN distance')
    return np.asarray(out)


def summarize_fit(oof, nboot, seed):
    rows = []
    for (target, model, rep), sub in oof.groupby(['target', 'model', 'repeat']):
        pp = per_position_error(sub)
        rows.append(dict(target=target, model=model, repeat=int(rep),
                         independent_positions=len(pp),
                         position_mae_pp=float(pp.ae.mean()),
                         position_rmse_pp=float(np.sqrt(pp.se.mean())),
                         mean_signed_bias_pp=float(pp.err.mean()),
                         variant_mae_pp=float(np.mean(abs(sub.prediction-sub.observed)))))
    met = pd.DataFrame(rows)
    gains = []
    for target, sub in oof.groupby('target'):
        baseline = sub[sub.model == 'M0'][['repeat','fold','variant_key','protein_position','observed','prediction']].rename(columns={'prediction': 'baseline'})
        for model in MODELS[1:]:
            alt = sub[sub.model == model][['repeat','fold','variant_key','protein_position','prediction']]
            z = baseline.merge(alt, on=['repeat','fold','variant_key','protein_position'], validate='one_to_one')
            if len(z) != len(baseline):
                raise AssertionError(f'Incomplete paired OOF {target}/{model}')
            z['gain'] = abs(z.observed-z.baseline)-abs(z.observed-z.prediction)
            per = z.groupby(['repeat','protein_position']).gain.mean().groupby('protein_position').mean()
            lo, hi = boot_ci(per, nboot, seed)
            gains.append(dict(target=target, model=model, reference='M0',
                              position_mae_gain_pp=float(per.mean()), ci95_low=lo, ci95_high=hi,
                              n_independent_positions=len(per)))
    return met, pd.DataFrame(gains)


def strata_fit(oof):
    """High/low observed response is diagnostic, not prospectively knowable."""
    reports = []
    for (target, model), sub in oof.groupby(['target','model']):
        rows = sub.groupby(['repeat','protein_position'], as_index=False)[['observed','prediction']].mean()
        for rep, g in rows.groupby('repeat'):
            g = g.copy()
            g['predicted_quartile'] = position_quartiles(g.prediction)
            g['observed_quartile_RETROSPECTIVE'] = position_quartiles(g.observed)
            g['ae'] = abs(g.prediction-g.observed)
            g['err'] = g.prediction-g.observed
            for field in ('predicted_quartile','observed_quartile_RETROSPECTIVE'):
                for group,z in g.groupby(field):
                    reports.append(dict(target=target, model=model, repeat=int(rep),
                                        stratum=field, group=group, n_positions=len(z),
                                        position_mae_pp=float(z.ae.mean()), signed_bias_pp=float(z.err.mean())))
    return pd.DataFrame(reports)



def paired_tail_gains(oof, nboot=3000, seed=3117):
    """Paired model-vs-M0 gains within frozen M0 or observed-response quartiles.

    Model-comparison bins are constructed ONCE from M0 per-repeat/position
    values, never separately per candidate. Observed-response bins are only
    retrospective diagnostics and are not usable at inference.
    """
    result = []
    for target, tab in oof.groupby('target'):
        b = tab[tab.model == 'M0'][['repeat','fold','variant_key','protein_position','observed','prediction']]
        bp = b.groupby(['repeat','protein_position'],as_index=False)[['observed','prediction']].mean()
        bins = []
        for rep, q in bp.groupby('repeat'):
            q=q.copy()
            q['m0_predicted_bin'] = position_quartiles(q.prediction)
            q['observed_bin_RETROSPECTIVE'] = position_quartiles(q.observed)
            bins.append(q[['repeat','protein_position','m0_predicted_bin','observed_bin_RETROSPECTIVE']])
        labels = pd.concat(bins, ignore_index=True)
        for candidate in MODELS[1:]:
            c = tab[tab.model == candidate][['repeat','fold','variant_key','protein_position','prediction']]
            j = b.merge(c,on=['repeat','fold','variant_key','protein_position'],validate='one_to_one',suffixes=('_m0','_candidate'))
            if len(j) != len(b):
                raise AssertionError(f'Missing paired tail rows {target}/{candidate}')
            j=j.merge(labels,on=['repeat','protein_position'],validate='many_to_one')
            j['gain'] = abs(j.observed-j.prediction_m0)-abs(j.observed-j.prediction_candidate)
            for field in ('m0_predicted_bin','observed_bin_RETROSPECTIVE'):
                for name, g in j.groupby(field):
                    per=(g.groupby(['repeat','protein_position']).gain.mean()
                         .groupby('protein_position').mean())
                    lo,hi=boot_ci(per,nboot,seed)
                    result.append(dict(target=target, model=candidate, reference='M0',
                                       stratum=field, quartile=name,
                                       n_independent_positions=len(per),
                                       gain_position_mae_pp=float(per.mean()),
                                       ci95_low=lo,ci95_high=hi))
    return pd.DataFrame(result)

def reliability(oof):
    z=oof[oof.model=='M0'].copy()
    # kNN distance is fold-local, attached to each OOF test row by variant/repeat.
    if z.feature_knn_distance.isna().any():
        raise AssertionError('Missing fold-local distances')
    # Predictor spread (M0/T1/T2) is NOT a posterior predictive interval.
    other=oof[oof.model.isin(('T1_mild','T2_strong'))].pivot(index=['target','repeat','variant_key'],columns='model',values='prediction')
    z=z.merge(other.reset_index(),on=['target','repeat','variant_key'],validate='one_to_one')
    z['ensemble_disagreement'] = z[['prediction','T1_mild','T2_strong']].std(axis=1,ddof=0)
    z['abs_error'] = abs(z.prediction-z.observed)
    pos=z.groupby(['target','repeat','protein_position'],as_index=False)[['feature_knn_distance','ensemble_disagreement','abs_error']].mean()
    met=[]; strata=[]
    for target,g in pos.groupby('target'):
        averaged=g.groupby('protein_position')[['feature_knn_distance','ensemble_disagreement','abs_error']].mean()
        for name in ('feature_knn_distance','ensemble_disagreement'):
            rho=float(spearmanr(averaged[name],averaged.abs_error).statistic)
            met.append(dict(target=target, heuristic=name, spearman_with_abs_error=rho,
                            n_independent_positions=len(averaged)))
            temp=g.copy()
            for r,gg in temp.groupby('repeat'):
                quart=position_quartiles(gg[name])
                temp.loc[gg.index, 'risk_quartile'] = quart.to_numpy()
            for q,t in temp.groupby('risk_quartile'):
                strata.append(dict(target=target,heuristic=name,quartile=q,
                                   position_repeat_count=len(t),observed_mae_pp=float(t.abs_error.mean())))
    return z,pd.DataFrame(met),pd.DataFrame(strata)


def analyze(args):
    d, _, folds, oof, checks, frozen_source = source_data()
    args.out.mkdir(parents=True,exist_ok=True)
    cal, cal_metrics, cal_bins = calibration(oof,args.bootstraps,args.seed)
    cal.to_csv(args.out/'phase31k_calibration_position_oof.csv',index=False)
    cal_metrics.to_csv(args.out/'phase31k_calibration_metrics.csv',index=False)
    cal_bins.to_csv(args.out/'phase31k_calibration_bins.csv',index=False)
    components, decomp = component_decomposition(oof)
    components.to_csv(args.out/'phase31k_component_oof.csv',index=False)
    (args.out/'phase31k_component_identity.json').write_text(json.dumps(decomp,indent=2)+'\n')
    history_path=ROOT/'results/biology/phase26d/phase26d_oof.csv'
    history, history_metrics=historical_independent_components(history_path,oof)
    if history is not None:
        history_metrics.to_csv(args.out/'phase31k_historical_independent_components.csv',index=False)
    manifest=dict(stage='analyze',n_vti_train_variants=len(d),n_positions=d.protein_position.nunique(),
                  outer_folds=len(folds),frozen_source=frozen_source,
                  source_sha256=checks,
                  historical_phase26d_sha256=sha256(history_path) if history is not None else None,
                  vti_val_test_outcomes_loaded=False,
                  interpretation='Exploratory OOF diagnostics only. Observed-response bins are retrospective, not deployable calibration.')
    (args.out/'phase31k_analyze_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PHASE 31K-A | POSITION-LEVEL OOF CALIBRATION (descriptive; not corrected predictions)')
    print(cal_metrics.groupby('target')[['sd_ratio_pred_over_obs','descriptive_observed_on_pred_slope','spearman']].mean().round(4).to_string())
    print('\nPREDICTED RESPONSE BINS (prospectively definable)')
    print(cal_bins[cal_bins.bin_type.eq('predicted_quartile')][['target','bin','n_independent_positions','mean_observed','mean_predicted','mean_signed_bias_pp','position_mae_pp']].round(3).to_string(index=False))
    print('\nPHASE 31K-B | EXACT COMPONENT MSE DECOMPOSITION')
    for field in ('position_weighted_mse_corrector','position_weighted_mse_implied_acute','position_weighted_twice_cross_term','position_weighted_mse_total','identity_absolute_difference'):
        print(f'  {field}: {decomp[field]:.6f}')
    if history_metrics is not None:
        print('\nHISTORICAL PHASE26D INDEPENDENTLY FITTED COMPONENTS (not matched Phase26G M0)')
        print(history_metrics[['model','position_mae_pp']].round(3).to_string(index=False))
    print(f'\nPASS Phase31K analyze: TRAIN-only frozen OOF; saved to {args.out}')


def fit(args):
    d,x,folds,oof,checks,frozen_source=source_data()
    import phase26g_feature_model_ablation as p26g
    args.out.mkdir(parents=True,exist_ok=True)
    keys=d.variant_key.astype(str).to_numpy()
    pos=d.protein_position.to_numpy(int)
    ys={'corrector':d.corrector_rescue_pp.to_numpy(float), 'total':d.vti_rescue_pp.to_numpy(float)}
    selections={t: choices_from_phase26g(t,folds) for t in ys}
    outrows=[]; selrows=[]
    foldlist=folds[:1] if args.quick else folds
    for ix,(repeat,fold,trkeys,tekeys) in enumerate(foldlist,1):
        tr=np.flatnonzero(np.isin(keys,list(trkeys)))
        te=np.flatnonzero(np.isin(keys,list(tekeys)))
        assert set(pos[tr]).isdisjoint(pos[te]) and len(tr)+len(te)==len(keys)
        dist=nearest_position_distances(x[tr],pos[tr],x[te],k=5)
        for target,y in ys.items():
            spec=selections[target][0][(int(repeat),int(fold))]
            scores_choice, inner_scores=inner_select(x[tr],y[tr],pos[tr],spec)
            fit_preds={}
            for alpha,model_name in [(0.,'M0'),(.75,'T1_mild'),(1.5,'T2_strong')]:
                if alpha==0.:
                    m=oof[(oof.target==target)&(oof.repeat==int(repeat))&(oof.fold==int(fold))]
                    pred=m.set_index('variant_key').loc[keys[te],'prediction'].to_numpy(float)
                else:
                    model=fit_weighted('xgb',spec,x[tr],y[tr],pos[tr],alpha)
                    pred=model.predict(x[te]).astype(float)
                if not np.isfinite(pred).all():
                    raise AssertionError(f'Non-finite model {model_name} predictions')
                fit_preds[alpha]=pred
                for i,p,dd in zip(te,pred,dist):
                    outrows.append(dict(target=target,model=model_name,repeat=int(repeat),fold=int(fold),
                                        variant_key=keys[i],protein_position=int(pos[i]),
                                        observed=float(y[i]),prediction=float(p),
                                        feature_knn_distance=float(dd)))
            pred=fit_preds[scores_choice]
            for i,p,dd in zip(te,pred,dist):
                outrows.append(dict(target=target,model='T3_inner_selected',repeat=int(repeat),fold=int(fold),
                                    variant_key=keys[i],protein_position=int(pos[i]),
                                    observed=float(y[i]),prediction=float(p),
                                    feature_knn_distance=float(dd)))
            selrows.append(dict(target=target,repeat=int(repeat),fold=int(fold),
                                xgb_frozen_spec=str(spec),inner_selected_alpha=float(scores_choice),
                                inner_mae_no_weight=float(dict(inner_scores)[0.]),
                                inner_mae_mild=float(dict(inner_scores)[.75]),
                                inner_mae_strong=float(dict(inner_scores)[1.5]),
                                train_n=len(tr),heldout_n=len(te),train_positions=len(set(pos[tr])),
                                heldout_positions=len(set(pos[te]))))
        print(f'Finished {ix}/{len(foldlist)} frozen folds: repeat={repeat} fold={fold} held positions={len(set(pos[te]))}',flush=True)
    pred=pd.DataFrame(outrows)
    selected=pd.DataFrame(selrows)
    if args.quick:
        pred.to_csv(args.out/'phase31k_quick_onefold_oof.csv',index=False)
        selected.to_csv(args.out/'phase31k_quick_onefold_selections.csv',index=False)
        print('PASS quick preflight: one outer fold ONLY; not suitable for performance inference')
        return
    for target in ys:
        for model in MODELS:
            sub=pred[(pred.target==target)&(pred.model==model)]
            for r,g in sub.groupby('repeat'):
                if len(g)!=len(d) or g.variant_key.duplicated().any() or set(g.variant_key)!=set(keys):
                    raise AssertionError(f'Incomplete OOF {target}/{model}/{r}')
    pred.to_csv(args.out/'phase31k_fit_oof.csv',index=False)
    selected.to_csv(args.out/'phase31k_fit_inner_selections.csv',index=False)
    met, gains=summarize_fit(pred,args.bootstraps,args.seed)
    met.to_csv(args.out/'phase31k_fit_metrics_by_repeat.csv',index=False)
    gains.to_csv(args.out/'phase31k_fit_paired_bootstrap.csv',index=False)
    strata=strata_fit(pred)
    strata.to_csv(args.out/'phase31k_fit_strata.csv',index=False)
    tail_gains=paired_tail_gains(pred,args.bootstraps,args.seed)
    tail_gains.to_csv(args.out/'phase31k_fit_paired_tail_gains.csv',index=False)
    reliable,rv,rb=reliability(pred)
    reliable.to_csv(args.out/'phase31k_reliability_oof.csv',index=False)
    rv.to_csv(args.out/'phase31k_reliability_metrics.csv',index=False)
    rb.to_csv(args.out/'phase31k_reliability_quartiles.csv',index=False)
    manifest=dict(stage='fit',train_only=True,n_vti_train_variants=len(d),
                  n_independent_positions=d.protein_position.nunique(),frozen_source=frozen_source,
                  repeats=2,folds=10,alpha_candidates=list(ALPHAS),
                  selection='3-fold GroupKFold on outer TRAIN positions, position MAE, alpha tie-break toward 0',
                  outer_hyperparameters='Phase26G frozen xgb_80 chosen specifications',
                  weighting='1 + alpha * clip(abs(fit_label - position_mean_median) / position_mean_IQR,0,2), mean-normalised; FIT labels only',
                  downstream_model='Phase26G.make_model xgb; frozen mutation-only 80 features',
                  reference_model='Saved Phase26G xgb_80 OOF with Phase31I M0 parity',
                  label_based_subgroups='Retrospective only; cannot know true tail at inference',
                  feature_distance='Nearest five distinct outer-training positions in fold-local scaled 80-d features; heuristic only',
                  external_vti_val_test_outcomes_loaded=False,
                  phase26g_selections_sha256={t: s[1] for t,s in selections.items()},
                  m0_oof_sha256=checks,bootstrap=args.bootstraps,seed=args.seed)
    (args.out/'phase31k_fit_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('\nPHASE 31K-C | MEAN POSITION MAE ACROSS TWO REPEATS (lower is better)')
    print(met.groupby(['target','model'],as_index=False).position_mae_pp.mean().sort_values(['target','position_mae_pp']).round(4).to_string(index=False))
    print('\nPAIRED POSITION BOOTSTRAP GAINS VS FROZEN M0 (positive = improvement)')
    print(gains.round(4).to_string(index=False))
    print('\nRETROSPECTIVE LOW/HIGH TRUE-RESPONSE TAIL GAINS (exploratory only)')
    print(tail_gains[(tail_gains.stratum=='observed_bin_RETROSPECTIVE') & tail_gains.quartile.isin(['Q1','Q4'])][['target','model','quartile','gain_position_mae_pp','ci95_low','ci95_high']].round(3).to_string(index=False))
    print('\nINNER ALPHA SELECTIONS')
    print(selected.groupby(['target','inner_selected_alpha']).size().to_string())
    print('\nPHASE 31K-D | HEURISTIC RELIABILITY CORRELATION WITH OOF M0 ABS ERROR')
    print(rv.round(4).to_string(index=False))
    print(f'\nPASS Phase31K fit: TRAIN-only exploratory experiment; saved to {args.out}')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stage',choices=['analyze','fit','all'])
    ap.add_argument('--quick',action='store_true',help='Fit only first frozen outer fold as a smoke test')
    ap.add_argument('--bootstraps',type=int,default=3000)
    ap.add_argument('--seed',type=int,default=3117)
    ap.add_argument('--out',type=Path,default=OUT)
    args=ap.parse_args(argv)
    if args.bootstraps < 100:
        ap.error('--bootstraps must be >= 100')
    if args.stage in ('analyze','all'):
        analyze(args)
    if args.stage in ('fit','all'):
        fit(args)


if __name__=='__main__':
    try:
        main()
    except (FileNotFoundError,ValueError,AssertionError,KeyError) as exc:
        print(f'PHASE 31K STOP: {exc}',file=sys.stderr)
        raise SystemExit(2)
