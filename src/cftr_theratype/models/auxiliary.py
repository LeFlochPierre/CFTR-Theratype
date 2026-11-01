"""Position-aware ETI auxiliary supervision; excluded-position logic preserved.

These functions are statistical helpers. Labels and eligible positions must be
filtered separately by the TRAIN-only loader and the caller outer/inner folds.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from cftr_theratype.models.regressors import phase26d_model
from cftr_theratype.evaluation.metrics import position_mae


WEIGHTS = (0.25, 0.5, 1.0)
SPEC = (2, 5.0, 180)


def positional_shuffle(y, group, seed):
    """Shuffle position means; retain within-position offsets."""

    rng = np.random.default_rng(seed)
    positions = np.unique(group)

    means = pd.DataFrame({
        "g": group,
        "y": y
    }).groupby("g").y.mean()

    shuffled = rng.permutation(
        means.reindex(positions).to_numpy()
    )

    mapping = dict(zip(positions, shuffled))

    return np.array([
        mapping[g] + (v - means.loc[g])
        for v, g in zip(y, group)
    ], float)

def fit_joint(Vx, Vy, Ex, Ey, weight):
    if len(Vx) == 0 or len(Ex) == 0:
        raise ValueError("Empty training subset")

    x = np.vstack([Vx, Ex])
    y = np.r_[Vy, Ey]

    w = np.r_[
        np.ones(len(Vx)),
        np.full(len(Ex), weight)
    ]

    estimator = phase26d_model(SPEC)
    estimator.fit(
        x, y,
        sample_weight=w
    )

    return estimator

def choose_weight(
    Xv, yv, gv, Xe, ye, ge,
    weights=WEIGHTS,
    shuffle_seed=None
):
    scores = {}

    for weight in weights:
        preds = np.full(len(yv), np.nan)

        cv = GroupKFold(3)

        for inner_fold, (it, iv) in enumerate(
            cv.split(Xv, yv, gv), 1
        ):
            # Remove ETI observations at inner-held-out
            # VTI positions before fitting or shuffling.
            allowed = ~np.isin(
                ge,
                np.unique(gv[iv])
            )

            if not np.any(allowed):
                raise ValueError(
                    "No eligible inner-fold ETI supervision"
                )

            if shuffle_seed is None:
                fold_labels = ye[allowed]
            else:
                fold_labels = positional_shuffle(
                    ye[allowed],
                    ge[allowed],
                    seed=shuffle_seed + inner_fold
                )

            est = fit_joint(
                Xv[it],
                yv[it],
                Xe[allowed],
                fold_labels,
                weight
            )

            preds[iv] = est.predict(Xv[iv])

        scores[weight] = position_mae(
            yv, preds, gv
        )

    choice = min(
        weights,
        key=lambda w: (scores[w], w)
    )

    return choice, scores[choice]
