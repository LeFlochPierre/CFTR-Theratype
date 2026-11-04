"""Canonical Phase31I.2 grouped-CV ridge teacher constructor."""
from __future__ import annotations

from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def make_teacher(alpha):
    return Pipeline([('impute',SimpleImputer(strategy='median')),
                     ('scale',StandardScaler()),('ridge',Ridge(alpha=alpha))])


