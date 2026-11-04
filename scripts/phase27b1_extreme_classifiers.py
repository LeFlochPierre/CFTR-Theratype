#!/usr/bin/env python3
"""Phase 27B-1: fixed-design responder classifiers, TRAIN only.

Position-disjoint repeated folds from Phase 26H.
No validation/test outcomes.
These are development estimates, not new external confirmation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path.cwd()

from cftr_theratype.datasets.eti_auxiliary import load_data as load_eti_train


OUT = ROOT / "results/biology/phase27b"
REF = ROOT / "results/biology/phase26h2/phase26h2_oof.csv"

ID = [
    "repeat", "fold",
    "variant_key", "protein_position"
]

TASKS = [
    ("corrector_low", "corrector", "low", 10),
    ("corrector_high", "corrector", "high", 40),
    ("total_low", "total", "low", 10),
    ("total_high", "total", "high", 40),
]


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


def run(nboot):
    if not REF.is_file():
        raise FileNotFoundError(
            f"Frozen Phase26H2 OOF missing: {REF}"
        )

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    # Existing audited TRAIN-only loader.
    v, V80, _, _, folds = load_eti_train(ROOT)

    V80 = np.asarray(V80, float)

    assert V80.shape == (210, 80)
    assert np.isfinite(
        V80[:, :68]
    ).all()

    vk = v.variant_key.astype(str).to_numpy()
    pp = v.protein_position.to_numpy(int)

    assert len(v) == 210
    assert len(np.unique(pp)) == 168

    yc = v.corrector_rescue_pp.to_numpy(float)
    yt = v.vti_rescue_pp.to_numpy(float)

    assert np.isfinite(yc).all()
    assert np.isfinite(yt).all()

    old = pd.read_csv(
        REF,
        dtype={"variant_key": str}
    )

    required = set(
        ID + [
            "vti_fixed",
            "vti_fixed_total",
            "observed_corrector",
            "observed_total",
        ]
    )

    if not required.issubset(old.columns):
        raise ValueError(
            "Missing frozen OOF columns: "
            f"{sorted(required - set(old.columns))}. "
            f"Available: {list(old.columns)}"
        )

    if len(old) != 420 or old.duplicated(ID).any():
        raise ValueError(
            "Frozen OOF membership invalid"
        )

    if not np.isfinite(
        old[
            ["vti_fixed", "vti_fixed_total"]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Nonfinite frozen baseline"
        )

    # Verify baseline labels against TRAIN identities.
    labelmap = dict(
        zip(vk, zip(yc, yt))
    )

    for row in old.itertuples():
        key = str(row.variant_key)

        if key not in labelmap:
            raise ValueError(
                f"Unknown variant: {key}"
            )

        yy = labelmap[key]

        if not np.allclose(
            [
                row.observed_corrector,
                row.observed_total,
            ],
            yy,
            atol=1e-6,
        ):
            raise ValueError(
                f"Baseline label mismatch: {key}"
            )

    print("=" * 100)
    print(
        "PHASE 27B-1 | EXTREME-RESPONDER "
        "CLASSIFICATION | TRAIN ONLY"
    )
    print("=" * 100)

    print(
        "210 TRAIN variants / 168 positions; "
        "original 2 x 5 position-grouped folds"
    )

    print(
        "Thresholds: low <10 pp, high >40 pp"
    )

    print(
        "Models: frozen regression ranking, "
        "logistic seq68, XGBoost seq80"
    )

    print(
        "No validation/test labels accessed"
    )

    for task, target, side, threshold in TASKS:
        y = make_labels(
            yc if target == "corrector" else yt,
            side,
            threshold,
        )

        print(
            f"  {task}: {sum(y)}/210 positives "
            f"at {len(set(pp[y.astype(bool)]))} positions"
        )

    predictions = []

    for fold_id, (
        repeat, fold, train_keys, test_keys
    ) in enumerate(folds, 1):

        itr = np.flatnonzero(
            np.isin(vk, list(train_keys))
        )

        ite = np.flatnonzero(
            np.isin(vk, list(test_keys))
        )

        if (
            len(itr) + len(ite) != len(v)
            or set(pp[itr]) & set(pp[ite])
        ):
            raise ValueError(
                "Position leakage or incomplete fold"
            )

        matched = old.loc[
            (old.repeat == repeat)
            & (old.fold == fold)
        ]

        if set(matched.variant_key) != set(vk[ite]):
            raise ValueError(
                f"Frozen OOF mismatch: {repeat}:{fold}"
            )

        frozen = matched.set_index(
            "variant_key"
        )

        for task_idx, (
            task, target, side, threshold
        ) in enumerate(TASKS):

            ys = make_labels(
                yc if target == "corrector" else yt,
                side,
                threshold,
            )

            ytr = ys[itr]

            if np.unique(ytr).size != 2:
                raise ValueError(
                    f"One-class training fold: {task}"
                )

            scores = {}

            for method in (
                "logistic_seq68",
                "xgb_seq80",
            ):
                fitted = make_model(
                    method,
                    ytr,
                    27000 + task_idx * 100 + fold_id,
                )

                X = (
                    V80[:, :68]
                    if method == "logistic_seq68"
                    else V80
                )

                fitted.fit(
                    X[itr], ytr
                )

                scores[method] = (
                    fitted.predict_proba(
                        X[ite]
                    )[:, 1]
                )

                if not np.isfinite(
                    scores[method]
                ).all():
                    raise ValueError(
                        f"Nonfinite predictions: {method}"
                    )

            for i, ii in enumerate(ite):
                key = vk[ii]

                baseline_col = (
                    "vti_fixed"
                    if target == "corrector"
                    else "vti_fixed_total"
                )

                continuous = float(
                    frozen.loc[
                        key, baseline_col
                    ]
                )

                predictions.append({
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "variant_key": key,
                    "protein_position": int(pp[ii]),
                    "task": task,
                    "label": int(ys[ii]),
                    "baseline_regression_score": (
                        -continuous
                        if side == "low"
                        else continuous
                    ),
                    "logistic_seq68": float(
                        scores["logistic_seq68"][i]
                    ),
                    "xgb_seq80": float(
                        scores["xgb_seq80"][i]
                    ),
                    "fold_train_prevalence": float(
                        ytr.mean()
                    ),
                })

        print(
            f"completed {fold_id}/{len(folds)}: "
            f"repeat={repeat}, fold={fold}, "
            f"held-out positions={len(set(pp[ite]))}",
            flush=True,
        )

    d = pd.DataFrame(predictions)

    if (
        len(d) != 1680
        or d.duplicated(ID + ["task"]).any()
    ):
        raise ValueError(
            "Incomplete or duplicated OOF predictions"
        )

    result = []

    for task, *_ in TASKS:
        for rep in (1, 2):
            q = d.loc[
                d.task.eq(task)
                & d.repeat.eq(rep)
            ]

            if len(q) != 210:
                raise ValueError(
                    "Incomplete OOF repeat"
                )

            for method in (
                "baseline_regression_score",
                "logistic_seq68",
                "xgb_seq80",
                "fold_train_prevalence",
            ):
                summary = metrics(
                    q.label,
                    q[method],
                    probability=(
                        method
                        != "baseline_regression_score"
                    ),
                )

                if summary is None:
                    raise ValueError(
                        "Single-class evaluation"
                    )

                result.append({
                    "task": task,
                    "repeat": rep,
                    "model": method,
                    **summary,
                })

    # Average two OOF predictions for each variant.
    avg = d.groupby(
        ["task", "variant_key", "protein_position"],
        as_index=False,
    ).agg({
        "label": "first",
        "baseline_regression_score": "mean",
        "logistic_seq68": "mean",
        "xgb_seq80": "mean",
        "fold_train_prevalence": "mean",
    })

    if len(avg) != 840:
        raise ValueError(
            "Incorrect averaged OOF count"
        )

    rows_avg = []
    boots = []

    for task_id, (
        task, *_rest
    ) in enumerate(TASKS):

        q = avg.loc[
            avg.task.eq(task)
        ].copy()

        if len(q) != 210:
            raise ValueError(
                f"Incomplete task {task}"
            )

        for method in (
            "baseline_regression_score",
            "logistic_seq68",
            "xgb_seq80",
            "fold_train_prevalence",
        ):
            summary = metrics(
                q.label,
                q[method],
                probability=(
                    method
                    != "baseline_regression_score"
                ),
            )

            rows_avg.append({
                "task": task,
                "model": method,
                **summary,
            })

        for candidate_id, candidate in enumerate(
            ("logistic_seq68", "xgb_seq80")
        ):
            boots.append({
                "task": task,
                **ap_gain_bootstrap(
                    q,
                    "baseline_regression_score",
                    candidate,
                    nboot,
                    seed=(
                        27100
                        + task_id * 10
                        + candidate_id
                    ),
                ),
            })

    result = pd.DataFrame(result)
    avg_metrics = pd.DataFrame(rows_avg)
    boots = pd.DataFrame(boots)

    d.to_csv(
        OUT / "phase27b1_repeated_oof.csv",
        index=False,
    )

    avg.to_csv(
        OUT / "phase27b1_averaged_oof.csv",
        index=False,
    )

    result.to_csv(
        OUT / "phase27b1_per_repeat_metrics.csv",
        index=False,
    )

    avg_metrics.to_csv(
        OUT / "phase27b1_averaged_metrics.csv",
        index=False,
    )

    boots.to_csv(
        OUT / "phase27b1_ap_bootstrap.csv",
        index=False,
    )

    print("\nA. PER-REPEAT CLASSIFICATION METRICS")

    print(
        result[
            [
                "task", "repeat", "model",
                "positives", "auroc", "ap",
                "top20_hits", "top20_lift", "brier",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nB. TWO-REPEAT AVERAGED OOF CLASSIFICATION")

    print(
        avg_metrics[
            [
                "task", "model", "positives",
                "auroc", "ap", "top20_hits",
                "top20_lift", "brier",
            ]
        ].round(4).to_string(index=False)
    )

    print(
        "\nC. POSITION-CLUSTERED BOOTSTRAP "
        "OF AP GAIN VS REGRESSION RANKING"
    )

    print(
        "Positive gain favours the classifier"
    )

    print(
        boots.round(4).to_string(index=False)
    )

    print("\nD. INTERPRETATION RULES")

    print(
        "Do not optimise thresholds on the "
        "previously opened test set."
    )

    print(
        "If high-responder AP does not improve, "
        "do not launch complex mixtures."
    )

    print(
        "Any apparent gain needs fresh "
        "independent confirmation."
    )

    print("\nPASS: Phase27B-1 completed; TRAIN only.")
    print("Saved:", OUT)


def self_test():
    y = np.array([0, 0, 1, 1, 0])

    values = np.array([
        5., 12., 40., 50., 60.
    ])

    assert np.array_equal(
        make_labels(values, "high", 40),
        [0, 0, 0, 1, 1],
    )

    assert np.array_equal(
        make_labels(values, "low", 10),
        [1, 0, 0, 0, 0],
    )

    a = metrics(
        y,
        np.array([.1, .2, .8, .9, .3]),
        probability=True,
    )

    assert a is not None
    assert np.isclose(a["auroc"], 1.)
    assert np.isclose(a["ap"], 1.)

    z = pd.DataFrame({
        "protein_position": [1, 1, 2, 3, 4, 5],
        "label": [0, 0, 1, 0, 1, 0],
        "baseline": [.1, .3, .5, .2, .6, .4],
        "candidate": [.1, .2, .9, .3, .8, .4],
    })

    b = ap_gain_bootstrap(
        z, "baseline", "candidate", 100, 27
    )

    assert b["positions"] == 5
    assert np.isfinite(b["ap_gain"])

    print(
        "PASS synthetic: responder labels, "
        "ranking, position bootstrap"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--self-test",
        action="store_true",
    )

    ap.add_argument(
        "--bootstraps",
        type=int,
        default=2000,
    )

    args = ap.parse_args()

    if args.self_test:
        self_test()
    else:
        if args.bootstraps < 100:
            ap.error(
                "--bootstraps must be >=100"
            )

        run(args.bootstraps)
