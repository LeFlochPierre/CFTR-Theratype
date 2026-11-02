#!/usr/bin/env python3
"""Phase 27B-2: responder-guided residual correction.

Nested protein-position cross-fitting, TRAIN only.
Reuses frozen outer-fold predictions from 26H2 and 27B1.
No protected validation/test outcomes.
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

import phase26h_auxiliary_corrector as h
import phase27b1_extreme_classifiers as b

OLD = ROOT / "results/biology/phase26h2/phase26h2_oof.csv"
CLASS = ROOT / "results/biology/phase27b/phase27b1_repeated_oof.csv"
OUT = ROOT / "results/biology/phase27b"

KEY = [
    "repeat", "fold", "variant_key", "protein_position"
]


def meta_model():
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=20.0)
    )


def fit_meta(baseline, low, high, target, seed):
    X = np.column_stack([low, high]).astype(float)
    residual = (
        np.asarray(target, float)
        - np.asarray(baseline, float)
    )

    if (
        not np.isfinite(X).all()
        or not np.isfinite(residual).all()
    ):
        raise ValueError("Nonfinite inner predictions")

    fitted = meta_model().fit(X, residual)

    rng = np.random.default_rng(seed)

    sham = meta_model().fit(
        X[rng.permutation(len(X))],
        residual
    )

    return fitted, sham, float(residual.mean())


def inner_features(V80, y, positions, train_ix, seed):
    """Fit inner OOF models within one outer training fold."""

    X = np.asarray(V80, float)
    tr = np.asarray(train_ix, int)

    groups = positions[tr]
    splitter = GroupKFold(n_splits=4)

    out_base = np.full(len(tr), np.nan)
    out_low = np.full(len(tr), np.nan)
    out_high = np.full(len(tr), np.nan)

    for j, (it, iv) in enumerate(
        splitter.split(X[tr], y[tr], groups),
        1
    ):
        fit_ix = tr[it]
        held_ix = tr[iv]

        if set(positions[fit_ix]) & set(positions[held_ix]):
            raise ValueError("Inner position leakage")

        yl = (y[fit_ix] < 10).astype(int)
        yh = (y[fit_ix] > 40).astype(int)

        if (
            len(np.unique(yl)) != 2
            or len(np.unique(yh)) != 2
        ):
            raise ValueError(
                "Inner fold has a single responder class"
            )

        reg = h.previous.model(h.SPEC)

        reg.fit(
            np.column_stack([
                X[fit_ix],
                np.ones(len(fit_ix))
            ]),
            y[fit_ix]
        )

        out_base[iv] = reg.predict(
            np.column_stack([
                X[held_ix],
                np.ones(len(held_ix))
            ])
        )

        low = b.make_model(
            "xgb_seq80", yl, seed + 100 * j
        )
        low.fit(X[fit_ix], yl)

        out_low[iv] = low.predict_proba(
            X[held_ix]
        )[:, 1]

        high = b.make_model(
            "logistic_seq68", yh,
            seed + 100 * j + 1
        )
        high.fit(X[fit_ix, :68], yh)

        out_high[iv] = high.predict_proba(
            X[held_ix, :68]
        )[:, 1]

    for arr in (out_base, out_low, out_high):
        if not np.isfinite(arr).all():
            raise ValueError(
                "Incomplete inner OOF predictions"
            )

    return out_base, out_low, out_high


def pos_mae(d, target, model):
    q = d[["repeat", "protein_position"]].copy()

    q["loss"] = np.abs(
        d[target].to_numpy(float)
        - d[model].to_numpy(float)
    )

    return float(
        q.groupby(
            ["repeat", "protein_position"]
        ).loss.mean()
        .groupby("protein_position")
        .mean()
        .mean()
    )


def bootstrap(d, target, candidate, reference, nboot, seed):
    q = d[["repeat", "protein_position"]].copy()

    q["gain"] = (
        np.abs(d[target] - d[reference])
        - np.abs(d[target] - d[candidate])
    )

    values = (
        q.groupby(
            ["repeat", "protein_position"]
        )["gain"].mean()
        .groupby("protein_position")
        .mean()
        .to_numpy(float)
    )

    rng = np.random.default_rng(seed)

    indices = rng.integers(
        0, len(values),
        size=(nboot, len(values))
    )

    samples = values[indices].mean(axis=1)

    return {
        "candidate": candidate,
        "reference": reference,
        "target": target,
        "gain_pp": float(values.mean()),
        "ci95_low": float(np.quantile(samples, .025)),
        "ci95_high": float(np.quantile(samples, .975)),
        "positions": len(values)
    }


def load_inputs():
    if not OLD.exists() or not CLASS.exists():
        raise FileNotFoundError(
            "Run Phase26H2 and Phase27B1 first"
        )

    v, V80, _, _, folds = h.load_data()

    keys = v.variant_key.astype(str).to_numpy()
    pp = v.protein_position.to_numpy(int)
    y = v.corrector_rescue_pp.to_numpy(float)

    assert len(v) == 210
    assert len(np.unique(pp)) == 168

    old = pd.read_csv(
        OLD, dtype={"variant_key": str}
    )

    found = pd.read_csv(
        CLASS, dtype={"variant_key": str}
    )

    if (
        len(old) != 420
        or old.duplicated(KEY).any()
        or len(found) != 1680
        or found.duplicated(KEY + ["task"]).any()
    ):
        raise ValueError("Invalid frozen OOF files")

    selected = []

    for task, column, label in [
        ("corrector_low", "xgb_seq80", "low_score"),
        ("corrector_high", "logistic_seq68", "high_score")
    ]:
        q = found.loc[
            found.task.eq(task),
            KEY + ["label", column]
        ].copy()

        q = q.rename(columns={
            column: label,
            "label": task + "_label"
        })

        if len(q) != 420:
            raise ValueError(
                "Incomplete classifier OOF: " + task
            )

        selected.append(q)

    ref = old.merge(
        selected[0],
        on=KEY,
        validate="one_to_one"
    ).merge(
        selected[1],
        on=KEY,
        validate="one_to_one"
    )

    if len(ref) != 420:
        raise ValueError("Outer OOF membership mismatch")

    identity = dict(
        zip(keys, zip(pp, y))
    )

    for row in ref.itertuples(index=False):
        if row.variant_key not in identity:
            raise ValueError("Unknown frozen variant")

        pos, target = identity[row.variant_key]

        if (
            int(pos) != int(row.protein_position)
            or not np.isclose(
                row.observed_corrector,
                target,
                atol=1e-6
            )
        ):
            raise ValueError("Frozen label mismatch")

        if (
            row.corrector_low_label != int(target < 10)
            or row.corrector_high_label != int(target > 40)
        ):
            raise ValueError("Classifier label mismatch")

    numerical = [
        "vti_fixed",
        "vti_fixed_total",
        "low_score",
        "high_score"
    ]

    if not np.isfinite(
        ref[numerical].to_numpy(float)
    ).all():
        raise ValueError("Nonfinite frozen scores")

    return (
        keys, pp, y,
        np.asarray(V80, float),
        folds, ref
    )


def run(nboot):
    keys, pp, y, V80, folds, ref = load_inputs()

    print("=" * 102)
    print(
        "PHASE 27B-2 | NESTED CLASSIFIER-GUIDED "
        "RESIDUAL CORRECTION"
    )
    print("=" * 102)

    print("TRAIN only: 210 variants / 168 positions")
    print("Frozen outer folds: 2 repeats x 5 folds")
    print(
        "Inner models: corrector regression, "
        "low XGB, high logistic"
    )
    print("Meta-model: Ridge alpha=20")
    print(
        "Controls: bias-only, shuffled gates"
    )
    print(
        "Protected validation/test outcomes unopened"
    )

    predictions = []
    diagnostics = []

    for k, (
        repeat, fold, train_keys, test_keys
    ) in enumerate(folds, 1):

        tr = np.flatnonzero(
            np.isin(keys, list(train_keys))
        )

        te = np.flatnonzero(
            np.isin(keys, list(test_keys))
        )

        if (
            len(tr) + len(te) != len(keys)
            or set(pp[tr]) & set(pp[te])
        ):
            raise ValueError("Outer-fold leakage")

        q = ref.loc[
            ref.repeat.eq(repeat)
            & ref.fold.eq(fold)
        ]

        if set(q.variant_key) != set(keys[te]):
            raise ValueError("Frozen fold mismatch")

        q = (
            q.set_index("variant_key")
            .loc[keys[te]]
            .reset_index()
        )

        base, low, high = inner_features(
            V80, y, pp, tr,
            seed=27200 + k
        )

        ridge, sham, bias = fit_meta(
            base, low, high, y[tr],
            seed=27200 + k
        )

        gates = q[
            ["low_score", "high_score"]
        ].to_numpy(float)

        corrected = ridge.predict(gates)
        shuffled = sham.predict(gates)

        original = q.vti_fixed.to_numpy(float)
        total_original = q.vti_fixed_total.to_numpy(float)

        models = {
            "original": original,
            "bias_only": original + bias,
            "classifier_ridge": original + corrected,
            "shuffled_gate": original + shuffled
        }

        for i, ix in enumerate(te):
            row = {
                "repeat": int(repeat),
                "fold": int(fold),
                "variant_key": keys[ix],
                "protein_position": int(pp[ix]),
                "observed_corrector": float(y[ix]),
                "observed_total": float(
                    q.observed_total.iloc[i]
                ),
                "low_score": float(gates[i, 0]),
                "high_score": float(gates[i, 1])
            }

            for name, values in models.items():
                row["corrector_" + name] = float(values[i])

                row["total_" + name] = float(
                    total_original[i]
                    + values[i]
                    - original[i]
                )

            predictions.append(row)

        diagnostics.append({
            "repeat": repeat,
            "fold": fold,
            "train_variants": len(tr),
            "test_variants": len(te),
            "train_positions": len(np.unique(pp[tr])),
            "inner_mae_base_pp": float(
                np.abs(y[tr] - base).mean()
            ),
            "inner_bias_pp": bias,
            "meta_ridge_train_mae_pp": float(
                np.abs(
                    y[tr] - base
                    - ridge.predict(
                        np.column_stack([low, high])
                    )
                ).mean()
            ),
            "meta_coeff_low_standardised": float(
                ridge.named_steps["ridge"].coef_[0]
            ),
            "meta_coeff_high_standardised": float(
                ridge.named_steps["ridge"].coef_[1]
            )
        })

        print(
            f"completed {k}/{len(folds)}: "
            f"repeat={repeat}, fold={fold}, "
            f"held-out positions={len(np.unique(pp[te]))}",
            flush=True
        )

    d = pd.DataFrame(predictions)

    if len(d) != 420 or d.duplicated(KEY).any():
        raise ValueError("Incomplete OOF predictions")

    metrics = []

    for target, actual, prefix in [
        ("Corrector", "observed_corrector", "corrector"),
        ("Total", "observed_total", "total")
    ]:
        yy = d[actual].to_numpy(float)

        for suffix in [
            "original",
            "bias_only",
            "classifier_ridge",
            "shuffled_gate"
        ]:
            name = prefix + "_" + suffix

            for subset, mask in [
                ("ALL", np.ones(len(d), bool)),
                ("low<10", yy < 10),
                ("high>40", yy > 40)
            ]:
                z = d.loc[mask]

                error = (
                    z[name] - z[actual]
                ).to_numpy(float)

                metrics.append({
                    "target": target,
                    "model": name,
                    "subset": subset,
                    "variants": z.variant_key.nunique(),
                    "positions": z.protein_position.nunique(),
                    "position_mae_pp": pos_mae(
                        z, actual, name
                    ),
                    "bias_pp": float(error.mean()),
                    "rmse_pp": float(
                        np.sqrt(np.mean(error ** 2))
                    )
                })

    comparisons = []

    for prefix, target in [
        ("corrector", "observed_corrector"),
        ("total", "observed_total")
    ]:
        for candidate in [
            "classifier_ridge",
            "bias_only",
            "shuffled_gate"
        ]:
            comparisons.append((
                target,
                prefix + "_" + candidate,
                prefix + "_original"
            ))

        comparisons.extend([
            (
                target,
                prefix + "_classifier_ridge",
                prefix + "_bias_only"
            ),
            (
                target,
                prefix + "_classifier_ridge",
                prefix + "_shuffled_gate"
            )
        ])

    gains = pd.DataFrame([
        bootstrap(
            d, target, candidate, reference,
            nboot, seed=27300 + i
        )
        for i, (
            target, candidate, reference
        ) in enumerate(comparisons)
    ])

    average = d.groupby(
        "variant_key", as_index=False
    ).agg({
        "observed_corrector": "first",
        "observed_total": "first",
        "corrector_original": "mean",
        "corrector_classifier_ridge": "mean",
        "total_original": "mean",
        "total_classifier_ridge": "mean",
        "high_score": "mean",
        "low_score": "mean"
    })

    ranking = []

    for title, labels, methods in [
        (
            "Corrector >40",
            average.observed_corrector.to_numpy(float) > 40,
            [
                "corrector_original",
                "corrector_classifier_ridge",
                "high_score"
            ]
        ),
        (
            "Corrector <10",
            average.observed_corrector.to_numpy(float) < 10,
            [
                "corrector_original",
                "corrector_classifier_ridge",
                "low_score"
            ]
        ),
        (
            "Total >40",
            average.observed_total.to_numpy(float) > 40,
            [
                "total_original",
                "total_classifier_ridge"
            ]
        )
    ]:
        for name in methods:
            score = average[name].to_numpy(float)

            if "<10" in title and name != "low_score":
                score = -score

            selected = np.argsort(
                -score, kind="stable"
            )[:42]

            ranking.append({
                "task": title,
                "model": name,
                "auroc": float(
                    roc_auc_score(labels, score)
                ),
                "ap": float(
                    average_precision_score(labels, score)
                ),
                "top20_hits": int(
                    labels[selected].sum()
                ),
                "positives": int(labels.sum())
            })

    OUT.mkdir(parents=True, exist_ok=True)

    d.to_csv(
        OUT / "phase27b2_oof.csv", index=False
    )

    pd.DataFrame(metrics).to_csv(
        OUT / "phase27b2_subgroups.csv", index=False
    )

    gains.to_csv(
        OUT / "phase27b2_bootstrap.csv", index=False
    )

    pd.DataFrame(diagnostics).to_csv(
        OUT / "phase27b2_inner_qc.csv", index=False
    )

    pd.DataFrame(ranking).to_csv(
        OUT / "phase27b2_ranking.csv", index=False
    )

    print("\nA. INNER-CROSSFITTING QC")
    print(
        pd.DataFrame(diagnostics)
        .mean(numeric_only=True)
        .round(4).to_string()
    )

    print("\nB. ZERO-SHOT DEVELOPMENT ERRORS")
    print(
        pd.DataFrame(metrics).round(4).to_string(
            index=False
        )
    )

    print(
        "\nC. POSITION BOOTSTRAP "
        "(positive gain = improvement)"
    )
    print(gains.round(4).to_string(index=False))

    print("\nD. RESPONDER RANKING")
    print(
        pd.DataFrame(ranking).round(4).to_string(
            index=False
        )
    )

    print("\nPASS: Phase27B-2 completed; TRAIN only.")
    print("No final-test retuning.")
    print("Outputs:", OUT)


def self_test():
    rng = np.random.default_rng(271)

    baseline = rng.normal(20, 3, 90)
    low = rng.uniform(.1, .8, 90)
    high = rng.uniform(.01, .4, 90)

    target = (
        baseline - 6 * low + 15 * high
        + rng.normal(0, .1, 90)
    )

    fitted, sham, bias = fit_meta(
        baseline, low, high, target, 27
    )

    X = np.column_stack([low, high])

    assert np.isfinite(fitted.predict(X)).all()
    assert np.isfinite(sham.predict(X)).all()
    assert np.isfinite(bias)

    q = pd.DataFrame({
        "repeat": [1, 1, 2, 2],
        "protein_position": [1, 2, 1, 2],
        "y": [1., 4., 1., 4.],
        "old": [2., 3., 2., 3.],
        "new": [1., 4., 1., 4.]
    })

    assert np.isclose(
        pos_mae(q, "y", "old"), 1.
    )

    assert np.isclose(
        bootstrap(
            q, "y", "new", "old", 100, 20
        )["gain_pp"],
        1.
    )

    print(
        "PASS synthetic: residual stacking, "
        "bias control, MAE, bootstrap"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--self-test", action="store_true"
    )

    ap.add_argument(
        "--bootstraps", type=int, default=10000
    )

    args = ap.parse_args()

    if args.self_test:
        self_test()
    else:
        if args.bootstraps < 100:
            ap.error("--bootstraps must be >=100")

        run(args.bootstraps)
