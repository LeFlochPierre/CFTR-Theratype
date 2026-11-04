"""Canonical fixed-design Phase27B responder models (TRAIN-only design)."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def make_labels(y, direction, threshold):
    if direction == "low":
        return (y < threshold).astype(int)
    return (y > threshold).astype(int)


def make_model(name, y_train, seed):
    if name == "logistic_seq68":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                solver="liblinear",
                max_iter=3000,
            ),
        )

    if name == "xgb_seq80":
        pos = int(y_train.sum())
        neg = len(y_train) - pos

        return XGBClassifier(
            n_estimators=140,
            max_depth=2,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=10.0,
            scale_pos_weight=min(
                8.0, neg / max(pos, 1)
            ),
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=4,
            random_state=seed,
        )

    raise KeyError(name)
