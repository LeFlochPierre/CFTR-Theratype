#!/usr/bin/env python3
"""Phase 27B-3: three-regime CFTR corrector-responder benchmark.

Original VTI TRAIN only; repeated position-disjoint folds.
No protected outcomes; no final-test model modification.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    log_loss,
)
from xgboost import XGBClassifier

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/biology/phase27b"
BASE = OUT / "phase27b1_repeated_oof.csv"

KEY = [
    "repeat", "fold", "variant_key", "protein_position"
]
CLASSES = ["low", "middle", "high"]
METHODS = [
    "prevalence_baseline",
    "coupled_binary",
    "ordinal_logistic",
    "multinomial_logistic",
    "multiclass_xgb",
]


def labels(y):
    y = np.asarray(y, float)
    return np.where(
        y < 10, 0,
        np.where(y > 40, 2, 1)
    ).astype(int)


def logistic(balanced=False):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            solver="lbfgs",
            max_iter=4000,
            class_weight=(
                "balanced" if balanced else None
            ),
        ),
    )


def couple_binary(low, high):
    """Heuristic coupling; not calibrated probabilities."""
    low = np.clip(np.asarray(low, float), 1e-8, 1-1e-8)
    high = np.clip(np.asarray(high, float), 1e-8, 1-1e-8)

    p = np.column_stack([
        low * (1-high),
        (1-low) * (1-high),
        high * (1-low),
    ])

    return p / p.sum(axis=1, keepdims=True)


def ordinal_prob(p_ge10, p_gt40):
    """Ensure P(response >= 10) >= P(response > 40)."""
    a = np.asarray(p_ge10, float).copy()
    b = np.asarray(p_gt40, float).copy()

    violation = a < b
    midpoint = (a[violation] + b[violation]) / 2

    a[violation] = midpoint
    b[violation] = midpoint

    p = np.column_stack([1-a, a-b, b])

    return p, int(violation.sum())


def check_prob(p):
    p = np.asarray(p, float)

    if (
        p.ndim != 2
        or p.shape[1] != 3
        or not np.isfinite(p).all()
        or (p < -1e-10).any()
        or (p > 1+1e-10).any()
        or not np.allclose(
            p.sum(axis=1), 1, atol=1e-8
        )
    ):
        raise ValueError(
            "Invalid three-class probability matrix"
        )


def one_vs_rest(y, p, k):
    positive = (y == k).astype(int)

    if positive.min() == positive.max():
        return np.nan, np.nan, 0

    score = p[:, k]
    n = int(np.ceil(len(y) * 0.2))

    hits = int(
        positive[
            np.argsort(
                -score, kind="stable"
            )[:n]
        ].sum()
    )

    return (
        float(average_precision_score(positive, score)),
        float(roc_auc_score(positive, score)),
        hits,
    )


def evaluate(d):
    y = d.true_class.to_numpy(int)

    p = d[
        ["p_low", "p_middle", "p_high"]
    ].to_numpy(float)

    check_prob(p)

    predicted = p.argmax(axis=1)
    onehot = np.eye(3)[y]

    p_log = np.clip(p, 1e-12, 1.)
    p_log /= p_log.sum(axis=1, keepdims=True)

    result = {
        "n": len(y),
        "low_n": int((y == 0).sum()),
        "middle_n": int((y == 1).sum()),
        "high_n": int((y == 2).sum()),
        "accuracy": float(np.mean(predicted == y)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predicted)
        ),
        "macro_f1": float(
            f1_score(
                y, predicted,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "log_loss": float(
            log_loss(y, p_log, labels=[0, 1, 2])
        ),
        "multiclass_brier": float(
            np.mean(np.sum((p - onehot)**2, axis=1))
        ),
    }

    for k, name in enumerate(CLASSES):
        ap, auc, hits = one_vs_rest(y, p, k)

        result[name + "_ap"] = ap
        result[name + "_auroc"] = auc
        result[name + "_top20_hits"] = hits

        result[name + "_recall"] = float(
            np.mean(predicted[y == k] == k)
        )

    return result


def bootstrap_ap_gain(df, method, cls, nboot, seed):
    a = (
        df.loc[df.model.eq(method)]
        .set_index("variant_key")
        .sort_index()
    )

    b = (
        df.loc[df.model.eq("coupled_binary")]
        .set_index("variant_key")
        .reindex(a.index)
    )

    if (
        a.index.has_duplicates
        or b.isna().any().any()
        or set(a.index) != set(b.index)
    ):
        raise ValueError(
            "Bootstrap model identity mismatch"
        )

    y = (
        a.true_class.to_numpy(int) == cls
    ).astype(int)

    pa = a[f"p_{CLASSES[cls]}"].to_numpy(float)
    pb = b[f"p_{CLASSES[cls]}"].to_numpy(float)

    pos = a.protein_position.to_numpy(int)
    unique = np.unique(pos)

    groups = [
        np.flatnonzero(pos == position)
        for position in unique
    ]

    observed = (
        average_precision_score(y, pa)
        - average_precision_score(y, pb)
    )

    rng = np.random.default_rng(seed)
    draws = []

    for _ in range(nboot):
        chosen = rng.integers(
            len(unique), size=len(unique)
        )

        ix = np.concatenate([
            groups[j] for j in chosen
        ])

        if len(np.unique(y[ix])) < 2:
            continue

        draws.append(
            average_precision_score(y[ix], pa[ix])
            - average_precision_score(y[ix], pb[ix])
        )

    if len(draws) < .95 * nboot:
        raise ValueError(
            "Too many one-class bootstrap samples"
        )

    return {
        "model": method,
        "task": CLASSES[cls],
        "ap_gain": float(observed),
        "ci95_low": float(np.quantile(draws, .025)),
        "ci95_high": float(np.quantile(draws, .975)),
        "positions": len(unique),
    }


def load_binary_oof():
    if not BASE.exists():
        raise FileNotFoundError(
            f"Run Phase27B-1 first: {BASE}"
        )

    old = pd.read_csv(
        BASE, dtype={"variant_key": str}
    )

    if (
        len(old) != 1680
        or old.duplicated(KEY + ["task"]).any()
    ):
        raise ValueError(
            "Phase27B-1 OOF integrity failure"
        )

    pieces = []

    for task, column, output in [
        (
            "corrector_low",
            "xgb_seq80",
            "low_binary"
        ),
        (
            "corrector_high",
            "logistic_seq68",
            "high_binary"
        )
    ]:
        q = old.loc[
            old.task.eq(task),
            KEY + [column]
        ].copy()

        if len(q) != 420:
            raise ValueError(
                f"Incomplete OOF: {task}"
            )

        pieces.append(
            q.rename(columns={column: output})
        )

    combined = pieces[0].merge(
        pieces[1],
        on=KEY,
        validate="one_to_one"
    )

    if len(combined) != 420:
        raise ValueError(
            "Binary OOF merge incomplete"
        )

    return combined


def run(nboot):
    import phase26h_auxiliary_corrector as h

    OUT.mkdir(parents=True, exist_ok=True)

    v, X, _, _, folds = h.load_data()

    X = np.asarray(X, float)
    vk = v.variant_key.astype(str).to_numpy()
    positions = v.protein_position.to_numpy(int)
    yc = v.corrector_rescue_pp.to_numpy(float)

    y = labels(yc)

    if X.shape != (210, 80):
        raise ValueError("Incorrect feature shape")

    if not np.isfinite(X[:, :68]).all():
        raise ValueError(
            "Nonfinite sequence features"
        )

    if len(np.unique(positions)) != 168:
        raise ValueError(
            "Unexpected protein positions"
        )

    if not np.array_equal(
        np.bincount(y, minlength=3),
        [52, 129, 29]
    ):
        raise ValueError(
            "Unexpected responder distribution"
        )

    binary = load_binary_oof()

    rows = []
    diagnostics = []

    print("=" * 100)
    print(
        "PHASE 27B-3 | THREE-CLASS "
        "CORRECTOR RESPONDER PREDICTION"
    )
    print("=" * 100)

    print(
        "TRAIN: 210 variants / 168 positions"
    )
    print(
        "Classes: low<10 (52), "
        "middle 10-40 (129), high>40 (29)"
    )
    print(
        "Frozen outer folds: 2 repetitions x 5 folds"
    )
    print(
        "Methods: prevalence, coupled binary, "
        "ordinal, multinomial, XGBoost"
    )
    print(
        "Protected outcomes unopened; "
        "no regression model changed"
    )

    for i, (
        rep, fold, train_keys, test_keys
    ) in enumerate(folds, 1):

        tr = np.flatnonzero(
            np.isin(vk, list(train_keys))
        )

        te = np.flatnonzero(
            np.isin(vk, list(test_keys))
        )

        if (
            len(tr) + len(te) != 210
            or set(positions[tr]) & set(positions[te])
        ):
            raise ValueError(
                "Outer-fold membership or leakage"
            )

        prior_counts = np.bincount(
            y[tr], minlength=3
        ).astype(float)

        prior = np.tile(
            prior_counts / prior_counts.sum(),
            (len(te), 1)
        )

        previous = binary.loc[
            binary.repeat.eq(rep)
            & binary.fold.eq(fold)
        ]

        if set(previous.variant_key) != set(vk[te]):
            raise ValueError(
                "Frozen binary-fold mismatch"
            )

        previous = (
            previous.set_index("variant_key")
            .loc[vk[te]]
        )

        coupled = couple_binary(
            previous.low_binary.to_numpy(float),
            previous.high_binary.to_numpy(float)
        )

        Xtr68 = X[tr, :68]
        Xte68 = X[te, :68]

        # Ordinal: two ordered cumulative boundaries.
        ge10 = logistic().fit(
            Xtr68,
            (y[tr] >= 1).astype(int)
        )

        gt40 = logistic().fit(
            Xtr68,
            (y[tr] == 2).astype(int)
        )

        ordinal, violations = ordinal_prob(
            ge10.predict_proba(Xte68)[:, 1],
            gt40.predict_proba(Xte68)[:, 1]
        )

        # Explicit three-class classification.
        multi = logistic(balanced=True).fit(
            Xtr68, y[tr]
        )

        multinomial = multi.predict_proba(Xte68)

        classes = (
            multi.named_steps["logisticregression"].classes_
        )

        if not np.array_equal(
            classes, [0, 1, 2]
        ):
            raise ValueError(
                "Multinomial class ordering changed"
            )

        # Shallow multiclass XGBoost.
        counts = np.bincount(
            y[tr], minlength=3
        )

        weights = np.sqrt(
            len(tr) / (3 * counts)
        )

        weights /= np.mean(weights[y[tr]])

        tree = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=160,
            max_depth=2,
            learning_rate=.05,
            min_child_weight=5,
            subsample=.9,
            colsample_bytree=.9,
            reg_lambda=10.,
            tree_method="hist",
            n_jobs=4,
            random_state=27300+i,
            eval_metric="mlogloss",
        )

        tree.fit(
            X[tr],
            y[tr],
            sample_weight=weights[y[tr]]
        )

        xgb = tree.predict_proba(X[te])

        predictions = {
            "prevalence_baseline": prior,
            "coupled_binary": coupled,
            "ordinal_logistic": ordinal,
            "multinomial_logistic": multinomial,
            "multiclass_xgb": xgb,
        }

        for method, p in predictions.items():
            check_prob(p)

            for j, idx in enumerate(te):
                rows.append({
                    "repeat": int(rep),
                    "fold": int(fold),
                    "variant_key": vk[idx],
                    "protein_position": int(
                        positions[idx]
                    ),
                    "true_class": int(y[idx]),
                    "model": method,
                    "p_low": float(p[j, 0]),
                    "p_middle": float(p[j, 1]),
                    "p_high": float(p[j, 2]),
                })

        diagnostics.append({
            "repeat": int(rep),
            "fold": int(fold),
            "ordinal_violations": violations,
            "held_positions": len(set(positions[te]))
        })

        print(
            f"completed {i}/10: repeat={rep}, "
            f"fold={fold}, positions={len(set(positions[te]))}, "
            f"ordinal_crossings={violations}",
            flush=True
        )

    d = pd.DataFrame(rows)

    if (
        len(d) != 2100
        or d.duplicated(KEY + ["model"]).any()
    ):
        raise ValueError(
            "Incomplete three-class OOF predictions"
        )

    for method in METHODS:
        for rep in (1, 2):
            q = d.loc[
                d.model.eq(method)
                & d.repeat.eq(rep)
            ]

            if (
                len(q) != 210
                or q.variant_key.nunique() != 210
            ):
                raise ValueError(
                    "Incomplete model/repeat OOF"
                )

    repeated = []

    for (method, rep), q in d.groupby(
        ["model", "repeat"]
    ):
        repeated.append({
            "model": method,
            "repeat": rep,
            **evaluate(q)
        })

    averaged = d.groupby(
        [
            "model",
            "variant_key",
            "protein_position",
            "true_class"
        ],
        as_index=False
    )[
        ["p_low", "p_middle", "p_high"]
    ].mean()

    if len(averaged) != 1050:
        raise ValueError(
            "Incorrect averaged OOF count"
        )

    summary = pd.DataFrame([
        {
            "model": method,
            **evaluate(q)
        }
        for method, q in averaged.groupby("model")
    ])

    intervals = pd.DataFrame([
        bootstrap_ap_gain(
            averaged,
            method,
            cls,
            nboot,
            seed=27500 + 10*j + cls
        )
        for j, method in enumerate(
            METHODS[2:], 2
        )
        for cls in (0, 2)
    ])

    d.to_csv(
        OUT / "phase27b3_repeated_oof.csv",
        index=False
    )

    averaged.to_csv(
        OUT / "phase27b3_averaged_oof.csv",
        index=False
    )

    pd.DataFrame(repeated).to_csv(
        OUT / "phase27b3_per_repeat_metrics.csv",
        index=False
    )

    summary.to_csv(
        OUT / "phase27b3_averaged_metrics.csv",
        index=False
    )

    intervals.to_csv(
        OUT / "phase27b3_ap_bootstrap.csv",
        index=False
    )

    pd.DataFrame(diagnostics).to_csv(
        OUT / "phase27b3_fold_qc.csv",
        index=False
    )

    print("\nA. THREE-CLASS PERFORMANCE")

    columns = [
        "model",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "low_ap",
        "middle_ap",
        "high_ap",
        "low_auroc",
        "high_auroc",
        "low_top20_hits",
        "high_top20_hits",
        "log_loss",
        "multiclass_brier",
    ]

    print(
        summary[columns]
        .round(4)
        .to_string(index=False)
    )

    print("\nB. PER-REPEAT STABILITY")

    print(
        pd.DataFrame(repeated)[
            [
                "model",
                "repeat",
                "balanced_accuracy",
                "low_ap",
                "high_ap",
                "macro_f1",
            ]
        ].round(4).to_string(index=False)
    )

    print(
        "\nC. POSITION BOOTSTRAP: "
        "AP GAIN VS COUPLED BINARY"
    )

    print(
        intervals.round(4).to_string(index=False)
    )

    print("\nD. THREE-CLASS CONFUSION MATRICES")

    print(
        "Rows = observed low/middle/high; "
        "columns = predicted low/middle/high"
    )

    for method, q in averaged.groupby("model"):
        p = q[
            ["p_low", "p_middle", "p_high"]
        ].to_numpy(float)

        predicted = p.argmax(axis=1)

        print("\n", method)

        print(
            confusion_matrix(
                q.true_class,
                predicted,
                labels=[0, 1, 2]
            )
        )

    print("\nINTERPRETATION")

    print(
        "Predicted probabilities are not "
        "independently calibrated."
    )
    print(
        "Previous standalone classifier AP: "
        "low=0.442, high=0.300."
    )
    print(
        "A coupled three-class model may have "
        "different AP values."
    )
    print(
        "No changes to continuous-response models."
    )

    print("\nPASS: Phase27B-3 completed; TRAIN only.")
    print("Saved:", OUT)


def self_test():
    assert np.array_equal(
        labels([2, 10, 40, 41]),
        [0, 1, 1, 2]
    )

    p = couple_binary(
        [.7, .1], [.1, .8]
    )
    check_prob(p)

    q, violations = ordinal_prob(
        [.2, .8], [.4, .3]
    )

    check_prob(q)

    assert violations == 1
    assert np.isclose(q[0, 1], 0)

    synthetic = pd.DataFrame({
        "true_class": [0, 1, 2, 0, 1, 2],
        "p_low": [.8, .1, .1, .75, .1, .1],
        "p_middle": [.1, .8, .1, .15, .8, .1],
        "p_high": [.1, .1, .8, .1, .1, .8],
    })

    result = evaluate(synthetic)

    assert np.isclose(result["accuracy"], 1)
    assert np.isclose(result["high_ap"], 1)

    print(
        "PASS synthetic: labels, probabilities, "
        "ordinal consistency, metrics"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument(
        "--bootstraps",
        type=int,
        default=1000
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        if args.bootstraps < 100:
            parser.error(
                "--bootstraps must be >=100"
            )

        run(args.bootstraps)
