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
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import phase26d_corrector_bottleneck as previous

OUT = ROOT / "results/biology/phase26h"
WEIGHTS = (0.25, 0.5, 1.0)

# Fixed from earlier training-only experiments.
SPEC = (2, 5.0, 180)


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

        scores[weight] = previous.pos_mae(
            yv, preds, gv
        )

    choice = min(
        weights,
        key=lambda w: (scores[w], w)
    )

    return choice, scores[choice]


def benchmark(vti, V80, eti, E80, folds, shuffles):
    keys = vti.variant_key.astype(str).to_numpy()
    vp = vti.protein_position.to_numpy(int)
    vy = vti.corrector_rescue_pp.to_numpy(float)

    ey = eti.corrector_rescue_pp.to_numpy(float)
    ep = eti.protein_position.to_numpy(int)

    # Treatment identity:
    # 1 = VTI, 0 = ETI.
    # Prediction always uses VTI identity.
    vx = np.column_stack([
        V80, np.ones(len(V80))
    ])

    ex = np.column_stack([
        E80, np.zeros(len(E80))
    ])

    rows = []
    choices = []

    for f, (rep, fold, trkeys, tekeys) in enumerate(
        folds, 1
    ):
        it = np.flatnonzero(
            np.isin(keys, list(trkeys))
        )

        iv = np.flatnonzero(
            np.isin(keys, list(tekeys))
        )

        assert len(it) + len(iv) == len(keys)

        heldout = set(vp[iv])

        assert heldout.isdisjoint(set(vp[it]))

        # Exclude ETI labels at the outer-held-out
        # VTI protein positions.
        aux = ~np.isin(
            ep,
            np.array(sorted(heldout), int)
        )

        assert not heldout.intersection(set(ep[aux]))

        if not aux.any():
            raise ValueError("Empty auxiliary training fold")

        # Real ETI supervision.
        wt, score = choose_weight(
            vx[it], vy[it], vp[it],
            ex[aux], ey[aux], ep[aux]
        )

        real = fit_joint(
            vx[it], vy[it],
            ex[aux], ey[aux], wt
        ).predict(vx[iv])

        predictions = {
            "joint_real": real
        }

        choices.append({
            "repeat": rep,
            "fold": fold,
            "model": "joint_real",
            "eti_weight": wt,
            "inner_pos_mae_pp": score,
            "outer_train_vti_positions": len(np.unique(vp[it])),
            "outer_test_positions": len(heldout),
            "outer_eti_variants": int(aux.sum())
        })

        # Negative controls.
        for k in range(shuffles):
            sham_seed = 2608 + 100 * f + k

            surrogate = positional_shuffle(
                ey[aux],
                ep[aux],
                seed=sham_seed
            )

            sw, ss = choose_weight(
                vx[it], vy[it], vp[it],
                ex[aux], ey[aux], ep[aux],
                shuffle_seed=sham_seed
            )

            name = f"joint_sham{k + 1}"

            predictions[name] = fit_joint(
                vx[it], vy[it],
                ex[aux], surrogate, sw
            ).predict(vx[iv])

            choices.append({
                "repeat": rep,
                "fold": fold,
                "model": name,
                "eti_weight": sw,
                "inner_pos_mae_pp": ss,
                "outer_train_vti_positions": len(np.unique(vp[it])),
                "outer_test_positions": len(heldout),
                "outer_eti_variants": int(aux.sum())
            })

        for j, i in enumerate(iv):
            row = {
                "repeat": rep,
                "fold": fold,
                "variant_key": keys[i],
                "protein_position": int(vp[i]),
                "study_cohort": vti.study_cohort.iloc[i],
                "observed_corrector": float(vy[i]),
                "observed_total": float(
                    vti.vti_rescue_pp.iloc[i]
                )
            }

            row.update({
                name: float(pred[j])
                for name, pred in predictions.items()
            })

            rows.append(row)

        print(
            f"completed {f}/{len(folds)}: "
            f"repeat={rep}, fold={fold}, "
            f"test_positions={len(heldout)}, "
            f"ETI_train={aux.sum()}, "
            f"selected_weight={wt}",
            flush=True
        )

    return pd.DataFrame(rows), pd.DataFrame(choices)


def paired_gain(df, candidate, reference, target, nboot, seed):
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
                    "position_mae_pp": previous.pos_mae(
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
