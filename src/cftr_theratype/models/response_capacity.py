"""Fixed Phase29A TRAIN-only model capacity/response balancing helpers.

The original XGB baseline uses the Phase26D estimator (spec 2, 5.0, 180)
and seed 26. Capacity variants retain the experimental fixed seed 29.
This module only operates on arrays supplied by the caller. It does not
access assay labels, frozen folds or protected validation/test outcomes.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer

from cftr_theratype.models.regressors import phase26d_model
from cftr_theratype.models.auxiliary import SPEC


CONFIGS = ['global','higher_capacity','response_balanced',
           'capacity_balanced','extra_trees']


def regime(y):
    y=np.asarray(y, float)
    return np.where(y < 10, 'low', np.where(y > 40, 'high','middle'))


def weights_by_train_regime(y):
    """Equal total TRAIN weight per regime, never uses held-out y."""
    groups=regime(y)
    counts={name:int((groups==name).sum()) for name in ('low','middle','high')}
    if any(v==0 for v in counts.values()):
        raise ValueError(f'Outer training fold missing regime: {counts}')
    w=np.array([len(y)/(3*counts[g]) for g in groups],float)
    if not np.isfinite(w).all() or not np.isclose(w.mean(),1):
        raise ValueError('Invalid training weights')
    return w,counts


def cap_model(base):
    """Increase capacity while preserving the original feature and estimator interface."""
    params=base.get_params()
    needed={'max_depth','min_child_weight','n_estimators','learning_rate',
            'reg_lambda','subsample','colsample_bytree'}
    # XGBRegressor exposes these even if original parameters are unset (None).
    if not needed.issubset(params):
        raise TypeError(f'Expected XGBRegressor-like model; missing {needed-set(params)}')
    return base.set_params(max_depth=5, min_child_weight=1,
                           n_estimators=350, learning_rate=0.04,
                           reg_lambda=1.0, subsample=0.9,
                           colsample_bytree=0.9, n_jobs=4,
                           random_state=29)


def make_model(config):
    if config=='extra_trees':
        return ExtraTreesRegressor(n_estimators=250, max_features=0.8,
                                   min_samples_leaf=2, max_depth=None,
                                   bootstrap=False, n_jobs=4, random_state=29)
    model=phase26d_model(SPEC)
    if config in ('higher_capacity','capacity_balanced'):
        model=cap_model(model)
    return model


def fit_predict(config, xtr, ytr, xte):
    model=make_model(config)
    sample_weight=None
    if config in ('response_balanced','capacity_balanced'):
        sample_weight,_=weights_by_train_regime(ytr)
    if config=='extra_trees':
        imp=SimpleImputer(strategy='median',add_indicator=True,
                          keep_empty_features=True)
        xtr=imp.fit_transform(xtr)
        xte=imp.transform(xte)
    if sample_weight is None:
        model.fit(xtr,ytr)
    else:
        model.fit(xtr,ytr,sample_weight=sample_weight)
    train_pred=np.asarray(model.predict(xtr),float).reshape(-1)
    test_pred=np.asarray(model.predict(xte),float).reshape(-1)
    if len(train_pred)!=len(ytr) or len(test_pred)!=len(xte):
        raise ValueError('Model prediction cardinality mismatch')
    if not np.isfinite(train_pred).all() or not np.isfinite(test_pred).all():
        raise ValueError('Nonfinite prediction')
    return train_pred,test_pred
