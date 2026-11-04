"""Position-aware binary responder evaluation used by Phase27B/D experiments."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def metrics(y, score, probability=False):
    y = np.asarray(y, int)
    score = np.asarray(score, float)

    if not np.isfinite(score).all():
        raise ValueError("Nonfinite model score")

    if len(np.unique(y)) < 2:
        return None

    n = max(1, int(np.ceil(0.2 * len(y))))

    top = np.argsort(
        -score, kind="stable"
    )[:n]

    hits = int(y[top].sum())
    prevalence = float(y.mean())

    return {
        "n": len(y),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "auroc": float(
            roc_auc_score(y, score)
        ),
        "ap": float(
            average_precision_score(y, score)
        ),
        "top20_hits": hits,
        "top20_selected": n,
        "top20_precision": hits / n,
        "top20_recall": hits / int(y.sum()),
        "top20_lift": (
            hits / n / prevalence
        ),
        "brier": (
            float(brier_score_loss(y, score))
            if probability else np.nan
        ),
    }


def ap_gain_bootstrap(
    df, baseline, candidate, nboot, seed
):
    """Position-cluster bootstrap of AP gain."""

    positions = df.protein_position.unique()

    groups = {
        p: df.loc[
            df.protein_position.eq(p)
        ]
        for p in positions
    }

    y = df.label.to_numpy(int)

    observed = (
        average_precision_score(
            y, df[candidate]
        )
        - average_precision_score(
            y, df[baseline]
        )
    )

    rng = np.random.default_rng(seed)
    draws = np.empty(nboot, float)

    for i in range(nboot):
        sampled = rng.choice(
            positions,
            size=len(positions),
            replace=True,
        )

        q = pd.concat(
            [groups[p] for p in sampled],
            ignore_index=True,
        )

        z = q.label.to_numpy(int)

        if len(np.unique(z)) < 2:
            draws[i] = np.nan
        else:
            draws[i] = (
                average_precision_score(
                    z, q[candidate]
                )
                - average_precision_score(
                    z, q[baseline]
                )
            )

    good = draws[np.isfinite(draws)]

    if len(good) < nboot * 0.95:
        raise ValueError(
            "Too many bootstrap samples "
            "lack a responder class"
        )

    return {
        "candidate": candidate,
        "reference": baseline,
        "ap_gain": float(observed),
        "ci95_low": float(
            np.quantile(good, .025)
        ),
        "ci95_high": float(
            np.quantile(good, .975)
        ),
        "positions": len(positions),
    }
