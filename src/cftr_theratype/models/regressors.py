"""Shared TRAIN-only model constructors and nested/grouped hyperparameter selection.

Exact specifications extracted from Phase26D and Phase26G.
"""
from __future__ import annotations
import warnings
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from cftr_theratype.evaluation.metrics import position_mae


PHASE26D_GRID = [(2, 5.0, 180), (3, 8.0, 250)]

PHASE26G_PARAMS = {
    'ridge': [10.0, 100.0],
    'elastic': [0.1, 1.0],
    'trees': [2, 5],
    'xgb': [(2, 5.0, 180), (3, 8.0, 250)],
}


def phase26d_model(spec, seed=26):
    depth, regularisation, trees = spec
    return XGBRegressor(
        n_estimators=trees, max_depth=depth, learning_rate=.05,
        subsample=.90, colsample_bytree=.90, min_child_weight=3,
        reg_lambda=regularisation, objective='reg:squarederror',
        tree_method='hist', n_jobs=3, random_state=seed,
    )

def phase26d_select_spec(X, y, group, quick=False):
    specs = PHASE26D_GRID[:1] if quick else PHASE26D_GRID
    if len(specs) == 1:
        return specs[0], float('nan')
    cv = GroupKFold(n_splits=3)
    scores=[]
    for spec in specs:
        preds=np.full(len(y),np.nan)
        for it,iv in cv.split(X,y,group):
            est=phase26d_model(spec)
            est.fit(X[it],y[it])
            preds[iv]=est.predict(X[iv])
        scores.append(position_mae(y,preds,group))
    ix=int(np.argmin(scores))
    return specs[ix], scores[ix]

def phase26g_model(kind, params, topk=None):
    steps = []

    # Imputation is fitted only on training positions.
    # XGBoost handles NaNs directly when no selection is used.
    if kind != "xgb" or topk is not None:
        steps.append(
            ("imputer", SimpleImputer(strategy="median"))
        )

    if kind in ("ridge", "elastic"):
        steps.append(("scale", StandardScaler()))

    if topk is not None:
        steps.append(
            ("select", SelectKBest(f_regression, k=topk))
        )

    if kind == "ridge":
        estimator = Ridge(alpha=params)

    elif kind == "elastic":
        estimator = ElasticNet(
            alpha=params,
            l1_ratio=0.5,
            max_iter=12000,
            random_state=2607,
        )

    elif kind == "trees":
        estimator = ExtraTreesRegressor(
            n_estimators=250,
            min_samples_leaf=params,
            max_features=0.8,
            n_jobs=3,
            random_state=2607,
        )

    elif kind == "xgb":
        depth, lam, trees = params
        estimator = XGBRegressor(
            n_estimators=trees,
            max_depth=depth,
            learning_rate=0.05,
            subsample=0.90,
            colsample_bytree=0.90,
            min_child_weight=3,
            reg_lambda=lam,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=3,
            random_state=26,
        )
    else:
        raise ValueError(kind)

    steps.append(("regression", estimator))
    return Pipeline(steps)

def phase26g_choose(x, y, groups, kind, topk, fast=False):
    best = None
    grid = PHASE26G_PARAMS[kind][:1] if fast else PHASE26G_PARAMS[kind]

    for params in grid:
        oof = np.full(len(y), np.nan)

        for tr, te in GroupKFold(3).split(x, y, groups):
            model = phase26g_model(kind, params, topk)

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message="Features .* are constant",
                )
                model.fit(x[tr], y[tr])

            oof[te] = model.predict(x[te])

        score = position_mae(y, oof, groups)

        if best is None or score < best[0] - 1e-12:
            best = (score, params)

    return best[1], best[0]
