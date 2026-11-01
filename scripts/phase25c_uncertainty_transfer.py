#!/usr/bin/env python3
"""Phase 25C: honest uncertainty and selective ETI->VTI response prediction.

TRAIN-ONLY. All outer folds are protein-position-disjoint. Each outer training
partition is subdivided into disjoint FIT and CALIBRATION positions. Calibration
labels are never used to fit point or scale models. Frozen validation/test
outcomes are never opened.

Inputs require a *measured* matched ETI functional assay; target is in-vitro
VTI total functional rescue, NOT a measured patient treatment benefit.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SOURCE = Path('results/biology/phase24g/phase24g_threshold_robustness.csv')
OUT = Path('results/biology/phase25c')
SCALE_FEATURES = ['eti_total_pp', 'eti_corrector_pp', 'eti_acute_pp', 'eti_baseline_fsk']
LEVELS = [0.80, 0.90]


def point_model(a):
    m = make_pipeline(StandardScaler(), HuberRegressor(alpha=1., epsilon=1.35,
                                                      max_iter=2000))
    m.fit(a[['eti_total_pp']].to_numpy(float), a.vti_total_pp.to_numpy(float))
    return m


def predict_point(m, a):
    return m.predict(a[['eti_total_pp']].to_numpy(float))


def fit_scale_model(fit):
    """Predict typical absolute point error from ETI assay only.

    The residual targets are strictly cross-fitted by protein position.
    """
    g = fit.protein_position.to_numpy(int)
    y = fit.vti_total_pp.to_numpy(float)
    residual = np.full(len(fit), np.nan)
    for it, iv in GroupKFold(n_splits=4).split(fit, y, g):
        m = point_model(fit.iloc[it])
        residual[iv] = np.abs(y[iv] - predict_point(m, fit.iloc[iv]))
    assert np.isfinite(residual).all()
    m = make_pipeline(StandardScaler(), Ridge(alpha=20.))
    m.fit(fit[SCALE_FEATURES].to_numpy(float), np.log1p(residual))
    return m


def predict_scale(m, d):
    return np.clip(np.expm1(m.predict(d[SCALE_FEATURES].to_numpy(float))), 1., 60.)


def grouped_quantile(cal, scores, level):
    # Maximum nonconformity per protein position: targets simultaneous
    # interval coverage for variants at an entirely unseen position.
    c = pd.DataFrame({'position':cal.protein_position.to_numpy(int), 'score':scores})
    a = np.sort(c.groupby('position').score.max().to_numpy(float))
    k = int(np.ceil((len(a) + 1) * level))
    return (float(a[k-1]) if k <= len(a) else float('inf')), len(a), k


def evaluate(d, seed=2520):
    d = d.sort_values(['protein_position', 'variant_key']).reset_index(drop=True)
    g = d.protein_position.to_numpy(int)
    y = d.vti_total_pp.to_numpy(float)
    output = []
    calibration_log = []
    for fold, (itr, ite) in enumerate(GroupKFold(n_splits=5).split(d, y, g), 1):
        outer_train = d.iloc[itr].reset_index(drop=True)
        test = d.iloc[ite].copy()
        splitter = GroupShuffleSplit(n_splits=1, test_size=.30, random_state=seed+fold)
        ifit, ical = next(splitter.split(outer_train, groups=outer_train.protein_position))
        fit, cal = outer_train.iloc[ifit], outer_train.iloc[ical]
        assert set(fit.protein_position).isdisjoint(cal.protein_position)
        assert set(test.protein_position).isdisjoint(outer_train.protein_position)
        pm = point_model(fit)
        sm = fit_scale_model(fit)
        pred_cal, pred_test = predict_point(pm, cal), predict_point(pm, test)
        scale_cal, scale_test = predict_scale(sm, cal), predict_scale(sm, test)
        err_cal = np.abs(cal.vti_total_pp.to_numpy(float) - pred_cal)
        err_direct_cal = np.abs(cal.vti_total_pp.to_numpy(float) - cal.eti_total_pp.to_numpy(float))
        t = test[['variant_key', 'protein_position', 'eti_total_pp', 'vti_total_pp']].copy()
        t['fold'] = fold
        t['pred_direct_eti'] = t.eti_total_pp
        t['pred_huber'] = pred_test
        t['estimated_error_pp'] = scale_test
        for level in LEVELS:
            for method, cal_scores, test_width_factor, centre in (
                ('direct_constant', err_direct_cal, np.ones(len(test)), t.eti_total_pp.to_numpy(float)),
                ('huber_constant', err_cal, np.ones(len(test)), pred_test),
                ('huber_scaled', err_cal / scale_cal, scale_test, pred_test),
            ):
                q, ncal, k = grouped_quantile(cal, cal_scores, level)
                width = q * test_width_factor
                key = f'{method}_{int(level*100)}'
                t['lo_'+key] = centre-width
                t['hi_'+key] = centre+width
                calibration_log.append({'fold':fold, 'method':method, 'nominal_coverage':level,
                    'n_fit_positions':fit.protein_position.nunique(),
                    'n_cal_positions':ncal, 'n_test_positions':test.protein_position.nunique(),
                    'quantile_rank':k, 'quantile':q})
        output.append(t)
    out = pd.concat(output, ignore_index=True)
    assert len(out)==len(d) and out.variant_key.is_unique
    return out, pd.DataFrame(calibration_log)


def position_mae(d, pred):
    e = np.abs(d.vti_total_pp.to_numpy(float) - d[pred].to_numpy(float))
    return float(pd.DataFrame({'g':d.protein_position,'e':e}).groupby('g').e.mean().mean())


def metrics(out):
    point = []
    for model in ['pred_direct_eti', 'pred_huber']:
        diff = out.vti_total_pp.to_numpy(float)-out[model].to_numpy(float)
        point.append({'model':model,'position_mae_pp':position_mae(out,model),
            'variant_mae_pp':float(np.abs(diff).mean()),
            'rmse_pp':float(np.sqrt(np.mean(diff**2))),
            'n_positions':out.protein_position.nunique()})
    intervals = []
    for level in LEVELS:
        for method in ['direct_constant', 'huber_constant', 'huber_scaled']:
            key = f'{method}_{int(level*100)}'
            inside = (out.vti_total_pp >= out['lo_'+key]) & (out.vti_total_pp <= out['hi_'+key])
            width = out['hi_'+key] - out['lo_'+key]
            pos = pd.DataFrame({'g':out.protein_position, 'inside':inside,'width':width})
            by_pos = pos.groupby('g').agg(all_covered=('inside','min'),width=('width','mean'))
            intervals.append({'method':method,'nominal':level,
                'position_all_variants_coverage':float(by_pos.all_covered.mean()),
                'variant_coverage':float(inside.mean()),
                'mean_position_width_pp':float(by_pos.width.mean()),
                'median_variant_width_pp':float(width.median()),
                'under10_coverage':float(inside[out.vti_total_pp < 10].mean()),
                'gt40_coverage':float(inside[out.vti_total_pp > 40].mean()),
                'n_under10':int((out.vti_total_pp < 10).sum()),
                'n_gt40':int((out.vti_total_pp > 40).sum())})
    return pd.DataFrame(point), pd.DataFrame(intervals)


def selection(out):
    out = out.copy()
    out['ae'] = np.abs(out.vti_total_pp-out.pred_huber)
    gp = out.groupby('protein_position').agg(
        mean_abs_error=('ae','mean'),
        predicted_risk=('estimated_error_pp','max'),
        n_variants=('variant_key','size'))
    gp = gp.sort_values(['predicted_risk'],kind='stable')
    rows = []
    for fraction in [.25,.50,.75,1.]:
        k = int(np.ceil(len(gp)*fraction))
        rows.append({'fraction_positions_kept':fraction,'n_positions':k,
                     'selected_position_mae_pp':float(gp.iloc[:k].mean_abs_error.mean()),
                     'all_position_mae_pp':float(gp.mean_abs_error.mean()),
                     'oracle_best_k_mae_pp':float(gp.nsmallest(k,'mean_abs_error').mean_abs_error.mean())})
    rho, pv = spearmanr(gp.predicted_risk,gp.mean_abs_error)
    return gp.reset_index(), pd.DataFrame(rows), float(rho), float(pv)


def prepare():
    d = pd.read_csv(SOURCE,dtype={'variant_key':str})
    assert len(d)==66 and d.variant_key.is_unique and d.protein_position.nunique()==56
    old = pd.read_parquet('data/processed/bihler_2024_missense_splits.parquet',
                          columns=['variant_key','split'])
    new = pd.read_parquet('data/processed/bihler_2026_transfer_splits.parquet',
                          columns=['variant_key','transfer_split'])
    assert set(d.variant_key).issubset(set(old.loc[old.split.eq('train'),'variant_key']))
    assert set(d.variant_key).issubset(set(new.loc[new.transfer_split.eq('train'),'variant_key']))
    for col in [*SCALE_FEATURES,'protein_position','vti_total_pp']:
        assert np.isfinite(d[col].to_numpy(float)).all(),col
    return d


def self_test():
    rng=np.random.default_rng(25)
    p=np.r_[np.arange(56),np.arange(10)]
    n=len(p)
    d=pd.DataFrame({'variant_key':[f'v{i}' for i in range(n)],'protein_position':p,
        'eti_corrector_pp':rng.uniform(0,50,n),'eti_acute_pp':rng.uniform(0,20,n),
        'eti_baseline_fsk':rng.uniform(0,15,n)})
    d['eti_total_pp']=d.eti_corrector_pp+d.eti_acute_pp
    sigma=3+np.maximum(0,d.eti_corrector_pp.to_numpy()-10)/5
    d['vti_total_pp']=d.eti_total_pp+2+rng.normal(0,sigma,n)
    out, log=evaluate(d)
    pm, im=metrics(out)
    gp, curve,rho,pval=selection(out)
    assert len(out)==n and len(log)==30 and len(gp)==56
    assert np.isfinite(out.filter(like='lo_').to_numpy(float)).all()
    assert np.isfinite(im.position_all_variants_coverage).all()
    print('PASS synthetic: outer/grouped fit-cal-test disjointness; point/scale/interval evaluation')
    print(pm.round(3).to_string(index=False))
    print(im.round(3).to_string(index=False))
    print(curve.round(3).to_string(index=False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--seed',type=int,default=2520)
    args=ap.parse_args()
    if args.self_test: return self_test()
    out, log=evaluate(prepare(), seed=args.seed)
    pt, iv=metrics(out)
    gp,curve,rho,pval=selection(out)
    print('='*98)
    print('PHASE 25C | UNCERTAINTY-AWARE CONTINUOUS ETI -> VTI PREDICTION | TRAIN ONLY')
    print('='*98)
    print('66 matched variants / 56 positions. Outer test positions disjoint from fit AND calibration.')
    print('Assay-conditioned: measured ETI response is required. No protected outcomes opened.')
    print('Intervals are exploratory grouped split-conformal; groupwise calibration uses maximum residual.')
    print('\nA. POINT PREDICTIONS (FIT SUBSET ONLY, NOT DIRECTLY COMPARABLE TO 25B FULL-TRAIN FIT)')
    print(pt.round(4).to_string(index=False))
    print('\nB. POSITION-LEVEL INTERVAL COVERAGE AND WIDTH')
    print(iv.round(4).to_string(index=False))
    print('\nC. CAN PREDICTED UNCERTAINTY IDENTIFY RELIABLE RESPONSE ESTIMATES?')
    print('Risk vs observed position-level MAE Spearman =',round(rho,4),'(descriptive p=',round(pval,4),')')
    print(curve.round(4).to_string(index=False))
    print('\nD. HELD-OUT CALIBRATION SIZES AND QUANTILE RANKS')
    print(log.round(4).to_string(index=False))
    OUT.mkdir(parents=True,exist_ok=True)
    for name,table in [('phase25c_oof.csv',out),('phase25c_point.csv',pt),
                       ('phase25c_intervals.csv',iv),('phase25c_risk_by_position.csv',gp),
                       ('phase25c_risk_curve.csv',curve),('phase25c_calibration.csv',log)]:
        table.to_csv(OUT/name,index=False)
    print('\nPASS Phase25C. Outputs:', OUT)

if __name__=='__main__': main()
