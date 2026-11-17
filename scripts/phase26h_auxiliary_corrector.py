#!/usr/bin/env python3
"""Phase 26H-1: zero-shot VTI corrector prediction with ETI supervision.

Mutation-derived features are the only prediction inputs.
ETI training labels at outer/inner held-out positions are excluded.
Protected ETI/VTI validation/test outcomes are never loaded.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path.cwd()
from types import SimpleNamespace
from cftr_theratype.evaluation.metrics import position_mae
from cftr_theratype.models.regressors import phase26d_model
from cftr_theratype.models.auxiliary import (
    WEIGHTS, SPEC,
)

# Compatibility for historical scripts calling h.previous.model(h.SPEC).
# The scientific implementation now lives only in cftr_theratype.
previous = SimpleNamespace(model=phase26d_model, pos_mae=position_mae)

OUT = ROOT / "results/biology/phase26h"

def parsed_position(s):
    from cftr_theratype.datasets.eti_auxiliary import parsed_position as shared
    return shared(s)

def load_data():
    from cftr_theratype.datasets.eti_auxiliary import load_data as canonical_load_data
    return canonical_load_data(ROOT)

def positional_shuffle(y, group, seed):
    from cftr_theratype.models.auxiliary import positional_shuffle as shared
    return shared(y, group, seed)

def fit_joint(Vx, Vy, Ex, Ey, weight):
    from cftr_theratype.models.auxiliary import fit_joint as shared
    return shared(Vx, Vy, Ex, Ey, weight)

def choose_weight(Xv, yv, gv, Xe, ye, ge, weights=WEIGHTS, shuffle_seed=None):
    from cftr_theratype.models.auxiliary import choose_weight as shared
    return shared(Xv, yv, gv, Xe, ye, ge, weights=weights, shuffle_seed=shuffle_seed)

def benchmark(vti, V80, eti, E80, folds, shuffles):
    """Legacy OOF API backed by the verified shared TRAIN-only prediction core."""
    from cftr_theratype.experiments import auxiliary_transfer
    return auxiliary_transfer.evaluate(
        vti, V80, eti, E80, folds, shuffles=shuffles, max_folds=len(folds)
    )


def paired_gain(df, candidate, reference, target, nboot, seed):
    from cftr_theratype.evaluation.metrics import phase26h_paired_gain as shared
    return shared(df, candidate, reference, target, nboot, seed)


def report(out, choices, nboot, shuffles):
    base = pd.read_csv(
        ROOT / "results/biology/phase26d/phase26d_oof.csv",
        dtype={"variant_key": str}
    )

    keys = [
        "repeat", "fold", "variant_key", "protein_position"
    ]

    need = keys + [
        "observed_total",
        "observed_corrector",
        "pred_zero_corrector_seq80",
        "pred_zero_acute_seq80",
        "pred_zero_components_seq80",
        "pred_zero_total_seq80"
    ]

    if not set(need).issubset(base.columns):
        raise ValueError(
            "Phase26D OOF missing required columns"
        )

    base = base[need].copy()

    out = out.merge(
        base,
        on=keys,
        validate="one_to_one",
        suffixes=("", "_baseline"),
        indicator=True
    )

    if (
        not out._merge.eq("both").all()
        or len(out) != len(base)
    ):
        raise ValueError(
            "Phase26D frozen OOF membership mismatch"
        )

    out = out.drop(columns=["_merge"])

    for col in ["observed_total", "observed_corrector"]:
        if not np.allclose(
            out[col],
            out[col + "_baseline"],
            atol=1e-8
        ):
            raise ValueError(
                "Outcome mismatch with Phase26D"
            )

        out = out.drop(
            columns=[col + "_baseline"]
        )

    # Reconstruct total rescue using frozen
    # out-of-fold acute-IVA predictions.
    names = ["joint_real"] + [
        f"joint_sham{k + 1}"
        for k in range(shuffles)
    ]

    for name in names:
        out[name + "_total"] = (
            out[name]
            + out.pred_zero_acute_seq80
        )

    tasks = {
        "Corrector": [
            "pred_zero_corrector_seq80",
            *names
        ],
        "Total": [
            "pred_zero_total_seq80",
            "pred_zero_components_seq80",
            *[name + "_total" for name in names]
        ]
    }

    metrics = []

    for rep, g in out.groupby("repeat"):
        for target, models in tasks.items():
            y = g[
                "observed_" + target.lower()
            ].to_numpy(float)

            pos = g.protein_position.to_numpy(int)

            for name in models:
                pred = g[name].to_numpy(float)
                error = y - pred
                tail = y > 40

                metrics.append({
                    "repeat": rep,
                    "target": target,
                    "model": name,
                    "position_mae_pp": position_mae(
                        y, pred, pos
                    ),
                    "variant_mae_pp": float(
                        np.mean(np.abs(error))
                    ),
                    "rmse_pp": float(
                        np.sqrt(np.mean(error ** 2))
                    ),
                    "gt40_mae_pp": (
                        float(np.mean(np.abs(error[tail])))
                        if tail.any() else np.nan
                    ),
                    "n_gt40": int(tail.sum())
                })

    metrics = pd.DataFrame(metrics)

    comparisons = []

    pairs = [
        (
            "joint_real",
            "pred_zero_corrector_seq80",
            "observed_corrector"
        ),
        (
            "joint_real_total",
            "pred_zero_components_seq80",
            "observed_total"
        ),
        (
            "joint_real_total",
            "pred_zero_total_seq80",
            "observed_total"
        )
    ]

    for candidate, reference, target in pairs:
        comparisons.append(
            paired_gain(
                out, candidate, reference,
                target, nboot, 2608
            )
        )

    for k in range(shuffles):
        comparisons.append(
            paired_gain(
                out,
                "joint_real",
                f"joint_sham{k + 1}",
                "observed_corrector",
                nboot,
                2609 + k
            )
        )

    comp = pd.DataFrame(comparisons)

    OUT.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        OUT / "phase26h_oof.csv",
        index=False
    )

    choices.to_csv(
        OUT / "phase26h_inner_selections.csv",
        index=False
    )

    metrics.to_csv(
        OUT / "phase26h_metrics_by_repeat.csv",
        index=False
    )

    comp.to_csv(
        OUT / "phase26h_bootstrap.csv",
        index=False
    )

    print("\nA. MATCHED POSITION-LEVEL PERFORMANCE")

    summary = (
        metrics.groupby(
            ["target", "model"], as_index=False
        )[
            [
                "position_mae_pp",
                "rmse_pp",
                "gt40_mae_pp"
            ]
        ]
        .mean()
        .sort_values(["target", "position_mae_pp"])
    )

    print(
        summary.round(4).to_string(index=False)
    )

    print("\nB. PAIRED POSITION BOOTSTRAP")
    print("Positive gain = proposed model improved")

    print(
        comp.round(4).to_string(index=False)
    )

    print("\nC. INNER-CV ETI WEIGHT SELECTION")

    print(
        choices.groupby(
            ["model", "eti_weight"]
        ).size().rename("outer_splits").to_string()
    )

    print("\nD. OUTER-FOLD TRAINING DETAILS")

    print(
        choices.round(4).to_string(index=False)
    )

    print(
        "\nPASS: Phase26H-1 completed; "
        "TRAIN-only; outputs:", OUT
    )


def self_test():
    rng = np.random.default_rng(26)

    gp = np.repeat(np.arange(36), 2)
    n = len(gp)

    X = rng.normal(size=(n, 81))
    X[:, -1] = 1

    yc = (
        X[:, 0] * 4
        + X[:, 1] * 2
        + rng.normal(size=n)
    )

    eg = np.repeat(np.arange(44), 2)

    E = rng.normal(size=(len(eg), 81))
    E[:, -1] = 0

    ey = (
        E[:, 0] * 4
        + E[:, 1] * 2
        + rng.normal(size=len(E))
    )

    eysh = positional_shuffle(ey, eg, 3)

    assert len(eysh) == len(ey)
    assert np.isfinite(eysh).all()

    folds = list(
        GroupKFold(3).split(X, yc, gp)
    )

    for it, iv in folds:
        aux = ~np.isin(
            eg,
            np.unique(gp[iv])
        )

        weight, score = choose_weight(
            X[it], yc[it], gp[it],
            E[aux], ey[aux], eg[aux],
            weights=(0.25,)
        )

        mdl = fit_joint(
            X[it], yc[it],
            E[aux], ey[aux],
            weight
        )

        assert np.isfinite(
            mdl.predict(X[iv])
        ).all()

        assert np.isfinite(score)

        sw, ss = choose_weight(
            X[it], yc[it], gp[it],
            E[aux], ey[aux], eg[aux],
            weights=(0.25,),
            shuffle_seed=999
        )

        assert np.isfinite(ss)

    print(
        "PASS synthetic Phase26H: "
        "grouped CV, weighted joint regression, "
        "nested ETI exclusion, shuffled controls"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--self-test",
        action="store_true"
    )

    ap.add_argument(
        "--shuffles",
        type=int,
        default=2
    )

    ap.add_argument(
        "--bootstraps",
        type=int,
        default=10000
    )

    args = ap.parse_args()

    if not 1 <= args.shuffles <= 10:
        raise ValueError("--shuffles must be 1..10")

    if args.self_test:
        return self_test()

    print("=" * 100)
    print(
        "PHASE 26H-1 | ZERO-SHOT VTI CORRECTOR "
        "PREDICTION WITH ETI AUXILIARY SUPERVISION"
    )
    print("=" * 100)

    print(
        "Mutation-only prediction; ETI functional assays "
        "are training labels, not prediction inputs."
    )

    print(
        "Frozen position-disjoint outer folds; "
        "ETI supervision excluded at outer/inner "
        "held-out positions."
    )

    print(
        f"Fixed XGBoost={SPEC}; ETI weights selected "
        f"in inner grouped CV; shuffles={args.shuffles}"
    )

    print(
        "TRAIN ONLY. Protected ETI/VTI outcomes unopened."
    )

    vti, V80, eti, E80, folds = load_data()

    oof, choices = benchmark(
        vti, V80, eti, E80,
        folds, args.shuffles
    )

    report(
        oof, choices,
        args.bootstraps,
        args.shuffles
    )


if __name__ == "__main__":
    main()
