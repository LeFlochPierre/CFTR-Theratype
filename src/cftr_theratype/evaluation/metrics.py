"""Legacy-compatible position-weighted metrics and paired bootstrap.

Do not count the two CV repeats as independent protein positions.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def position_mae(y, pred, groups):
    d = pd.DataFrame({
        "position": groups,
        "error": np.abs(y - pred),
    })
    return float(d.groupby("position").error.mean().mean())

def paired_ci(d, candidate, baseline, n=5000):
    q = d.copy()

    q["gain"] = (
        np.abs(q.observed - q[baseline])
        - np.abs(q.observed - q[candidate])
    )

    values = (
        q.groupby(["repeat", "protein_position"])
        .gain.mean()
        .groupby("protein_position")
        .mean()
        .to_numpy()
    )

    rng = np.random.default_rng(2607)

    sims = rng.choice(
        values,
        size=(n, len(values)),
        replace=True,
    ).mean(axis=1)

    return {
        "model": candidate,
        "reference": baseline,
        "gain_pp": float(values.mean()),
        "ci95_low": float(np.quantile(sims, 0.025)),
        "ci95_high": float(np.quantile(sims, 0.975)),
        "n_positions": len(values),
    }


def phase26h_paired_gain(df, candidate, reference, target, nboot, seed):
    diff = (
        np.abs(df[target] - df[reference])
        - np.abs(df[target] - df[candidate])
    )

    g = df[["repeat", "protein_position"]].copy()
    g["gain"] = diff

    values = (
        g.groupby(["repeat", "protein_position"])
        .gain.mean()
        .groupby("protein_position")
        .mean()
        .to_numpy()
    )

    rng = np.random.default_rng(seed)

    sims = np.mean(
        rng.choice(
            values,
            size=(nboot, len(values)),
            replace=True
        ),
        axis=1
    )

    return {
        "candidate": candidate,
        "reference": reference,
        "target": target,
        "gain_pp": float(values.mean()),
        "ci95_low": float(np.quantile(sims, .025)),
        "ci95_high": float(np.quantile(sims, .975)),
        "n_independent_positions": len(values)
    }
