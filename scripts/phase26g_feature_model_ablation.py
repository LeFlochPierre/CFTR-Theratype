#!/usr/bin/env python3
"""Phase 26G: feature-size and model-family benchmark.

TRAIN ONLY. Reuses Phase26D frozen protein-position folds.
Hyperparameters and feature selection use inner grouped CV.
No protected validation/test outcomes are loaded.
"""

import argparse
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

import phase26d_corrector_bottleneck as prior

SETUPS = [
    ("ridge_68", "sequence", "ridge", None),
    ("ridge_80", "sequence_structure", "ridge", None),
    ("elastic_80", "sequence_structure", "elastic", None),
    ("extratrees_80", "sequence_structure", "trees", None),
    ("xgb_68", "sequence", "xgb", None),
    ("xgb_80", "sequence_structure", "xgb", None),
    ("xgb_82", "sequence_structure_thermompnn", "xgb", None),
    ("xgb_selected16_80", "sequence_structure", "xgb", 16),
    ("xgb_selected32_80", "sequence_structure", "xgb", 32),
]

PARAMS = {
    "ridge": [10.0, 100.0],
    "elastic": [0.1, 1.0],
    "trees": [2, 5],
    "xgb": [(2, 5.0, 180), (3, 8.0, 250)],
}


def make_model(kind, params, topk=None):
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


def position_mae(y, pred, groups):
    d = pd.DataFrame({
        "position": groups,
        "error": np.abs(y - pred),
    })
    return float(d.groupby("position").error.mean().mean())


def choose(x, y, groups, kind, topk, fast=False):
    best = None
    grid = PARAMS[kind][:1] if fast else PARAMS[kind]

    for params in grid:
        oof = np.full(len(y), np.nan)

        for tr, te in GroupKFold(3).split(x, y, groups):
            model = make_model(kind, params, topk)

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        choices=["corrector", "total"],
        default="corrector",
    )
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--bootstraps", type=int, default=5000)
    args = ap.parse_args()

    d, features, _, folds, _ = prior.load_train()

    if args.target == "corrector":
        y = d.corrector_rescue_pp.to_numpy(float)
    else:
        y = d.vti_rescue_pp.to_numpy(float)

    positions = d.protein_position.to_numpy(int)
    keys = d.variant_key.astype(str).to_numpy()

    matrices = {}

    for name in sorted({s[1] for s in SETUPS}):
        if name not in features:
            raise ValueError(
                f"Missing feature representation: {name}"
            )

        x = np.array(features[name], dtype=float, copy=True)
        x[np.isinf(x)] = np.nan

        assert x.ndim == 2 and x.shape[0] == len(d)
        matrices[name] = x

        print(
            f"Feature matrix {name}: "
            f"shape={x.shape}, "
            f"missing={np.isnan(x).sum()}"
        )

    print("=" * 100)
    print(
        f"PHASE 26G | FEATURE COUNT x MODEL FAMILY "
        f"| {args.target.upper()}"
    )
    print("=" * 100)
    print(
        "210 variants / 168 independent protein positions; "
        "frozen Phase26D outer folds"
    )
    print(
        "Feature and hyperparameter selection: "
        "inner protein-position-grouped CV"
    )
    print("Mutation-derived features ONLY")
    print("Protected outcomes unopened")
    print("Training-only model comparison: EXPLORATORY")

    if args.fast:
        print("FAST MODE: smoke test, not final tuning")

    predictions = []
    selections = []

    for fi, (rep, fold, trkeys, tekeys) in enumerate(folds, 1):
        tr = np.flatnonzero(np.isin(keys, list(trkeys)))
        te = np.flatnonzero(np.isin(keys, list(tekeys)))

        assert len(tr) + len(te) == len(d)
        assert set(positions[tr]).isdisjoint(positions[te])

        frame = pd.DataFrame({
            "repeat": rep,
            "fold": fold,
            "variant_key": keys[te],
            "protein_position": positions[te],
            "observed": y[te],
            "study_cohort": d.study_cohort.iloc[te].to_numpy(),
        })

        for label, representation, kind, topk in SETUPS:
            x = matrices[representation]

            choice, inner_mae = choose(
                x[tr], y[tr], positions[tr],
                kind, topk, args.fast,
            )

            model = make_model(kind, choice, topk)
            model.fit(x[tr], y[tr])

            frame[label] = model.predict(x[te])

            selections.append({
                "repeat": rep,
                "fold": fold,
                "model": label,
                "representation": representation,
                "kind": kind,
                "top_k": topk,
                "selected_param": str(choice),
                "inner_position_mae": inner_mae,
            })

        predictions.append(frame)

        print(
            f"Completed {fi}/{len(folds)}: "
            f"repeat={rep}, fold={fold}, "
            f"positions={len(np.unique(positions[te]))}",
            flush=True,
        )

    out = pd.concat(predictions, ignore_index=True)

    for rep, g in out.groupby("repeat"):
        assert len(g) == len(d)
        assert g.variant_key.nunique() == len(d)

    names = [s[0] for s in SETUPS]
    metrics = []

    for rep, g in out.groupby("repeat"):
        for name in names:
            errors = (
                g.observed.to_numpy()
                - g[name].to_numpy()
            )

            tail = g.observed.to_numpy() > 40

            metrics.append({
                "repeat": rep,
                "model": name,
                "position_mae_pp": position_mae(
                    g.observed.to_numpy(),
                    g[name].to_numpy(),
                    g.protein_position.to_numpy(),
                ),
                "variant_mae_pp": float(np.mean(np.abs(errors))),
                "rmse_pp": float(np.sqrt(np.mean(errors ** 2))),
                "n_gt40": int(tail.sum()),
                "gt40_mae_pp": (
                    float(np.mean(np.abs(errors[tail])))
                    if tail.any() else np.nan
                ),
            })

    metrics = pd.DataFrame(metrics)

    comparisons = pd.DataFrame([
        paired_ci(out, name, "xgb_80", args.bootstraps)
        for name in names if name != "xgb_80"
    ])

    outdir = ROOT / "results/biology/phase26g"
    outdir.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        outdir / f"phase26g_{args.target}_oof.csv",
        index=False,
    )

    pd.DataFrame(selections).to_csv(
        outdir / f"phase26g_{args.target}_selected.csv",
        index=False,
    )

    metrics.to_csv(
        outdir / f"phase26g_{args.target}_metrics_by_repeat.csv",
        index=False,
    )

    comparisons.to_csv(
        outdir / f"phase26g_{args.target}_bootstrap.csv",
        index=False,
    )

    print("\nA. PERFORMANCE (AVERAGED ACROSS REPEATS)")

    summary = (
        metrics.groupby("model", as_index=False)[
            ["position_mae_pp", "rmse_pp", "gt40_mae_pp"]
        ]
        .mean()
        .sort_values("position_mae_pp")
    )

    print(summary.round(4).to_string(index=False))

    print("\nB. PAIRED BOOTSTRAP VS MATCHED XGBOOST 80")

    print(
        comparisons.sort_values(
            "gain_pp", ascending=False
        ).round(4).to_string(index=False)
    )

    print("\nC. INNER MODEL SELECTIONS")

    print(
        pd.DataFrame(selections)
        .groupby(["model", "selected_param"])
        .size()
        .rename("folds")
        .to_string()
    )

    print(
        f"\nPASS: Phase26G completed; "
        f"saved {args.target} results to {outdir}"
    )


if __name__ == "__main__":
    main()
