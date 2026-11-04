"""Canonical Phase31I.2 grouped-CV ridge teacher constructor."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from cftr_theratype.datasets.assay_tables import permute_by_position
from cftr_theratype.features.assay_substitutions import mutation_features

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_teacher(alpha):
    return Pipeline([('impute',SimpleImputer(strategy='median')),
                     ('scale',StandardScaler()),('ridge',Ridge(alpha=alpha))])


def make_pilot_teacher(alpha: float):
    """Historical Phase31I ridge pilot, retaining keep_empty_features=True."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])


def crossfit_teacher_predictions(table, train_aa, train_pos, held_aa, held_pos,
                        outer_held, alpha, negative_seed=None, inner_folds=5):
    """No teacher sees labels at outer held positions or inner cross-fit groups."""
    if set(table.pos31i2.astype(int)) & set(outer_held):
        raise AssertionError('Outer held-out position in teacher data')
    src=table.copy()
    if negative_seed is not None:
        src=permute_by_position(src,np.random.default_rng(negative_seed))
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
