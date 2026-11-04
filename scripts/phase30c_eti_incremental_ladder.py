#!/usr/bin/env python3
"""Phase 30C: Does prior ETI add value beyond genotype and VTI baseline?

TRAIN only. Original 2x5 position-disjoint folds, restricted to the
66 variants / 56 positions with both ETI and VTI measurements.

No VTI validation/test labels. Phase24G ETI feature columns only.
Every prediction method has exactly the same matched training cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path.cwd()

OUT = ROOT / "results/biology/phase30c"
ETI_CSV = ROOT / "results/biology/phase24g/phase24g_threshold_robustness.csv"
OLD_SPLIT = ROOT / "data/processed/bihler_2024_missense_splits.parquet"
NEW_SPLIT = ROOT / "data/processed/bihler_2026_transfer_splits.parquet"
VTI_PARQUET = ROOT / "data/processed/bihler_2026_vti.parquet"
PHASE30A = ROOT / "results/biology/phase30a/phase30a_oof.csv"

ETI_COLUMNS = [
    "eti_corrector_pp",
    "eti_acute_pp",
    "eti_baseline_fsk",
]

TARGETS = {
    "corrector": "observed_corrector",
    "total_rescue": "observed_total_rescue",
    "treated_absolute": "observed_treated_absolute",
}

COMMON = [
    "train_median",
    "mutation_only",
    "baseline_only",
    "mutation_plus_baseline",
    "eti_only",
    "mutation_plus_eti",
    "baseline_eti_ridge",
    "mutation_plus_baseline_eti",
]

CONFIGS = {
    "corrector": COMMON,
    "total_rescue": COMMON + [
        "mutation_plus_baseline_eti_iva",
        "direct_eti_total",
    ],
    "treated_absolute": COMMON + [
        "mutation_plus_baseline_eti_iva",
        "direct_eti_total_plus_baseline",
    ],
}

KEY = ["repeat", "fold", "variant_key", "protein_position"]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fit_xgb(X_train, y_train, X_test, factory):
    """Locked XGBoost algorithm; no hyperparameter search."""

    X_train = np.asarray(X_train, float)
    X_test = np.asarray(X_test, float)

    # Preserve the historical constant-bias feature convention.
    X_train = np.column_stack([
        X_train,
        np.ones(len(X_train)),
    ])
    X_test = np.column_stack([
        X_test,
        np.ones(len(X_test)),
    ])

    model = factory()
    model.fit(X_train, np.asarray(y_train, float))

    pred = np.asarray(model.predict(X_test), float)

    if pred.shape != (len(X_test),) or not np.isfinite(pred).all():
        raise ValueError("Invalid XGBoost prediction")

    return pred


def fit_ridge(X_train, y_train, X_test):
    """Fixed regularisation; scaling fitted on outer TRAIN only."""

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )

    model.fit(
        np.asarray(X_train, float),
        np.asarray(y_train, float),
    )

    return np.asarray(
        model.predict(np.asarray(X_test, float)),
        float,
    )


def load_matched_data():
    from cftr_theratype.datasets.eti_auxiliary import load_data

    v, X80, _, _, folds = load_data(ROOT)

    if (
        len(v) != 210
        or v.variant_key.nunique() != 210
        or v.protein_position.nunique() != 168
        or len(folds) != 10
    ):
        raise ValueError("Frozen VTI TRAIN cohort changed")

    X80 = np.asarray(X80, float)

    if X80.shape != (210, 80):
        raise ValueError(f"Unexpected mutation features: {X80.shape}")

    v = v.copy()
    v["variant_key"] = v.variant_key.astype(str)

    vti_train = set(v.variant_key)

    old = pd.read_parquet(
        OLD_SPLIT,
        columns=["variant_key", "split"],
    )

    new = pd.read_parquet(
        NEW_SPLIT,
        columns=["variant_key", "transfer_split"],
    )

    old_train = set(
        old.loc[old.split.eq("train"), "variant_key"].astype(str)
    )

    new_train = set(
        new.loc[
            new.transfer_split.eq("train"),
            "variant_key",
        ].astype(str)
    )

    if vti_train != new_train:
        raise ValueError("VTI TRAIN membership mismatch")

    # IMPORTANT:
    # Phase24G includes historical VTI outcomes.
    # Read ONLY prior ETI information, not those VTI labels.
    eti_cols = [
        "variant_key",
        "protein_position",
        "eti_total_pp",
    ] + ETI_COLUMNS

    eti = pd.read_csv(
        ETI_CSV,
        usecols=eti_cols,
        dtype={"variant_key": str},
    )

    if (
        len(eti) != 66
        or eti.variant_key.duplicated().any()
        or eti.protein_position.nunique() != 56
    ):
        raise ValueError(
            "Expected 66 matched ETI variants / 56 positions"
        )

    matched = set(eti.variant_key)

    if not matched.issubset(old_train):
        raise ValueError("Protected ETI variants encountered")

    if not matched.issubset(vti_train):
        raise ValueError("Protected VTI variants encountered")

    vpos = v.set_index("variant_key").protein_position.astype(int)

    check_keys = vpos.index.intersection(matched)

    eti_pos = (
        eti.set_index("variant_key")
        .loc[check_keys, "protein_position"]
        .to_numpy(int)
    )

    vti_pos = vpos.loc[check_keys].to_numpy(int)

    if not np.array_equal(eti_pos, vti_pos):
        raise ValueError("ETI/VTI amino-acid position mismatch")

    # Reuse Phase30A TRAIN outcomes only.
    oof = pd.read_csv(
        PHASE30A,
        dtype={"variant_key": str},
    )

    needed = set(
        KEY
        + list(TARGETS.values())
        + ["observed_baseline"]
    )

    if not needed.issubset(oof.columns):
        raise ValueError(
            f"Phase30A missing columns: {needed-set(oof.columns)}"
        )

    if (
        len(oof) != 420
        or oof.duplicated(["repeat", "variant_key"]).any()
        or set(oof.variant_key) != vti_train
        or set(oof.repeat.unique()) != {1, 2}
    ):
        raise ValueError("Invalid Phase30A TRAIN OOF")

    selected = oof.loc[
        oof.repeat.eq(1),
        ["variant_key", "protein_position"]
        + list(TARGETS.values())
        + ["observed_baseline"],
    ].copy()

    selected_indexed = selected.set_index("variant_key")

    if not np.array_equal(
        selected_indexed.loc[vpos.index, "protein_position"]
        .to_numpy(int),
        vpos.to_numpy(int),
    ):
        raise ValueError("Phase30A protein-position mismatch")

    selected_ordered = selected_indexed.loc[vpos.index]

    if not np.allclose(
        selected_ordered.observed_corrector,
        v.corrector_rescue_pp,
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("Corrector target mismatch")

    if not np.allclose(
        selected_ordered.observed_total_rescue,
        v.vti_rescue_pp,
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("Total VTI target mismatch")

    # Predicate-filter the assay parquet to matched TRAIN IDs
    # BEFORE materialising any assay measurements.
    iva = pd.read_parquet(
        VTI_PARQUET,
        filters=[
            ("variant_key", "in", sorted(matched)),
        ],
        columns=[
            "variant_key",
            "iva_treated_pct_wt",
        ],
    )

    iva["variant_key"] = iva.variant_key.astype(str)

    if (
        len(iva) != 66
        or iva.variant_key.duplicated().any()
        or set(iva.variant_key) != matched
    ):
        raise ValueError("Filtered IVA assay membership mismatch")

    data = v[
        ["variant_key", "protein_position"]
    ].merge(
        selected.drop(columns=["protein_position"]),
        on="variant_key",
        validate="one_to_one",
    )

    data = data.merge(
        eti.drop(columns=["protein_position"]),
        on="variant_key",
        how="inner",
        validate="one_to_one",
    )

    data = data.merge(
        iva,
        on="variant_key",
        validate="one_to_one",
    )

    if (
        len(data) != 66
        or data.protein_position.nunique() != 56
    ):
        raise ValueError("Matched cohort cardinality mismatch")

    index = {
        key: i
        for i, key in enumerate(v.variant_key)
    }

    X = X80[
        [index[k] for k in data.variant_key]
    ]

    return data, X, folds


def evaluate(data, X, folds, factory):
    """Fit all information stages on identical outer TRAIN subsets."""

    if len(data) != 66 or X.shape != (66, 80):
        raise ValueError("Matched cohort expected")

    if np.isinf(X).any():
        raise ValueError("Infinite mutation descriptors")

    if len(folds) != 10:
        raise ValueError("Expected original 2x5 folds")

    required = (
        list(TARGETS.values())
        + ["observed_baseline", "iva_treated_pct_wt"]
        + ETI_COLUMNS
        + ["eti_total_pp"]
    )

    for col in required:
        if not np.isfinite(data[col].to_numpy(float)).all():
            raise ValueError(f"Nonfinite assay data: {col}")

    if not np.allclose(
        data.observed_treated_absolute
        - data.observed_baseline,
        data.observed_total_rescue,
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("VTI assay subtraction identity failed")

    if not np.allclose(
        data.eti_corrector_pp + data.eti_acute_pp,
        data.eti_total_pp,
        atol=0.25,
        rtol=0,
    ):
        raise ValueError("ETI decomposition mismatch")

    keys = data.variant_key.astype(str).to_numpy()
    positions = data.protein_position.to_numpy(int)

    baseline = data.observed_baseline.to_numpy(float)
    iva = data.iva_treated_pct_wt.to_numpy(float)
    eti = data[ETI_COLUMNS].to_numpy(float)

    features = {
        "mutation_only": X,
        "mutation_plus_baseline": np.column_stack([
            X, baseline,
        ]),
        "eti_only": eti,
        "mutation_plus_eti": np.column_stack([
            X, eti,
        ]),
        "mutation_plus_baseline_eti": np.column_stack([
            X, baseline, eti,
        ]),
        "mutation_plus_baseline_eti_iva": np.column_stack([
            X, baseline, eti, iva,
        ]),
    }

    outputs = []
    fold_qc = []

    seen = {
        1: set(),
        2: set(),
    }

    for i, (repeat, fold, train_keys, test_keys) in enumerate(
        folds, 1
    ):
        train = np.flatnonzero(
            np.isin(keys, list(train_keys))
        )
        test = np.flatnonzero(
            np.isin(keys, list(test_keys))
        )

        if (
            not len(train)
            or not len(test)
            or len(train) + len(test) != 66
        ):
            raise ValueError("Incomplete matched outer fold")

        if set(positions[train]) & set(positions[test]):
            raise ValueError("PROTEIN POSITION LEAKAGE")

        if seen[repeat] & set(keys[test]):
            raise ValueError("Repeated held-out mutation")

        seen[repeat].update(keys[test])

        for target, truth_col in TARGETS.items():
            y = data[truth_col].to_numpy(float)

            pred = {
                "train_median": np.full(
                    len(test),
                    np.median(y[train]),
                ),
                "baseline_only": fit_ridge(
                    baseline[train, None],
                    y[train],
                    baseline[test, None],
                ),
                "baseline_eti_ridge": fit_ridge(
                    np.column_stack([
                        baseline[train],
                        eti[train],
                    ]),
                    y[train],
                    np.column_stack([
                        baseline[test],
                        eti[test],
                    ]),
                ),
            }

            for method in CONFIGS[target]:
                if method in features:
                    mat = features[method]

                    pred[method] = fit_xgb(
                        mat[train],
                        y[train],
                        mat[test],
                        factory,
                    )

            if target == "total_rescue":
                pred["direct_eti_total"] = (
                    data.eti_total_pp.to_numpy(float)[test]
                )

            if target == "treated_absolute":
                pred["direct_eti_total_plus_baseline"] = (
                    data.eti_total_pp.to_numpy(float)[test]
                    + baseline[test]
                )

            if set(pred) != set(CONFIGS[target]):
                raise ValueError(
                    f"Unexpected configurations for {target}"
                )

            for j, idx in enumerate(test):
                row = {
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "variant_key": str(keys[idx]),
                    "protein_position": int(positions[idx]),
                    "target": target,
                    "observed": float(y[idx]),
                }

                for method, prediction in pred.items():
                    row[f"pred_{method}"] = float(prediction[j])

                outputs.append(row)

        fold_qc.append({
            "repeat": int(repeat),
            "fold": int(fold),
            "train_records": len(train),
            "test_records": len(test),
            "train_positions": len(set(positions[train])),
            "test_positions": len(set(positions[test])),
        })

        print(
            f"Fold {i:02d}/10: repeat={repeat}, fold={fold}, "
            f"train={len(train)}, heldout={len(test)}",
            flush=True,
        )

    if seen[1] != set(keys) or seen[2] != set(keys):
        raise ValueError("Incomplete out-of-fold coverage")

    oof = pd.DataFrame(outputs)

    if (
        len(oof) != 396
        or oof.duplicated(KEY + ["target"]).any()
    ):
        raise ValueError("Malformed matched OOF")

    return oof, pd.DataFrame(fold_qc)


def metrics_and_gains(oof, bootstraps):
    metrics = []
    gains = []

    comparisons = [
        (
            "mutation_plus_baseline_eti",
            "mutation_plus_baseline",
        ),
        (
            "baseline_eti_ridge",
            "baseline_only",
        ),
        (
            "mutation_plus_eti",
            "mutation_only",
        ),
        (
            "mutation_plus_baseline",
            "mutation_only",
        ),
        (
            "eti_only",
            "mutation_only",
        ),
    ]

    for target, frame in oof.groupby("target"):
        for method in CONFIGS[target]:
            pred_col = f"pred_{method}"

            errors = (
                frame[pred_col] - frame.observed
            ).abs()

            z = frame[
                ["repeat", "protein_position"]
            ].copy()

            z["error"] = errors.to_numpy(float)

            position_mae = float(
                z.groupby(
                    ["repeat", "protein_position"]
                ).error.mean()
                .groupby("protein_position")
                .mean()
                .mean()
            )

            average = frame.groupby("variant_key")[
                ["observed", pred_col]
            ].mean()

            metrics.append({
                "target": target,
                "model": method,
                "position_mae_pp": position_mae,
                "variant_mae_pp": float(errors.mean()),
                "pred_sd_pp": float(
                    average[pred_col].std(ddof=0)
                ),
                "spearman": float(
                    average[pred_col].corr(
                        average.observed,
                        method="spearman",
                    )
                ),
            })

        pairs = comparisons.copy()

        if target != "corrector":
            pairs.append((
                "mutation_plus_baseline_eti_iva",
                "mutation_plus_baseline_eti",
            ))

        for i, (candidate, reference) in enumerate(pairs):
            z = frame[
                ["repeat", "protein_position"]
            ].copy()

            z["gain"] = (
                (
                    frame[f"pred_{reference}"]
                    - frame.observed
                ).abs()
                -
                (
                    frame[f"pred_{candidate}"]
                    - frame.observed
                ).abs()
            )

            by_position = (
                z.groupby(
                    ["repeat", "protein_position"]
                ).gain.mean()
                .groupby("protein_position")
                .mean()
                .to_numpy(float)
            )

            rng = np.random.default_rng(
                3000 + 10 * list(TARGETS).index(target) + i
            )

            indexes = rng.integers(
                0,
                len(by_position),
                size=(bootstraps, len(by_position)),
            )

            samples = by_position[indexes].mean(axis=1)

            gains.append({
                "target": target,
                "candidate": candidate,
                "reference": reference,
                "gain_pp": float(by_position.mean()),
                "ci95_low": float(
                    np.quantile(samples, 0.025)
                ),
                "ci95_high": float(
                    np.quantile(samples, 0.975)
                ),
                "positions": len(by_position),
            })

    return pd.DataFrame(metrics), pd.DataFrame(gains)


def self_test():
    from sklearn.tree import DecisionTreeRegressor

    rng = np.random.default_rng(30)

    positions = np.r_[
        np.arange(56),
        np.arange(10),
    ]

    keys = np.array([
        f"SYN{i:03d}"
        for i in range(66)
    ])

    X = rng.normal(size=(66, 80))

    eti_base = rng.uniform(0, 15, 66)
    eti_corrector = rng.uniform(0, 50, 66)
    eti_acute = rng.uniform(0, 20, 66)

    baseline = eti_base + rng.normal(3, 2, 66)
    corrector = eti_corrector * 0.8 + rng.normal(0, 5, 66)
    total = corrector + eti_acute * 0.8 + rng.normal(0, 3, 66)

    data = pd.DataFrame({
        "variant_key": keys,
        "protein_position": positions,
        "observed_baseline": baseline,
        "observed_corrector": corrector,
        "observed_total_rescue": total,
        "observed_treated_absolute": baseline + total,
        "iva_treated_pct_wt": baseline + eti_acute,
        "eti_corrector_pp": eti_corrector,
        "eti_acute_pp": eti_acute,
        "eti_baseline_fsk": eti_base,
        "eti_total_pp": eti_corrector + eti_acute,
    })

    folds = []

    for repeat in [1, 2]:
        for fold in range(1, 6):
            mask = (
                (positions + (repeat - 1) * 3) % 5 + 1
            ) == fold

            folds.append((
                repeat,
                fold,
                set(keys[~mask]),
                set(keys[mask]),
            ))

    factory = lambda: DecisionTreeRegressor(
        max_depth=3,
        random_state=7,
    )

    oof, qc = evaluate(
        data, X, folds, factory
    )

    m, g = metrics_and_gains(
        oof, 200
    )

    assert len(oof) == 396
    assert len(qc) == 10
    assert len(m) == 28
    assert len(g) == 17

    assert np.isfinite(m.position_mae_pp).all()
    assert set(g.positions) == {56}

    corrupted = data.copy()
    corrupted.loc[
        0, "observed_treated_absolute"
    ] += 1

    try:
        evaluate(corrupted, X, folds, factory)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Corrupted assay identity was not detected"
        )

    print("PASS Phase30C synthetic tests")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--bootstraps",
        type=int,
        default=2000,
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.bootstraps < 100:
        parser.error("Require at least 100 bootstraps")

    self_test()

    from cftr_theratype.models.regressors import phase26d_model
    from cftr_theratype.models.auxiliary import SPEC as FROZEN_XGB_SPEC
    data, X, folds = load_matched_data()

    print("=" * 112)
    print("PHASE 30C | ETI -> VTI INCREMENTAL INFORMATION")
    print("=" * 112)

    print(
        f"Matched TRAIN: {len(data)} variants, "
        f"{data.protein_position.nunique()} positions"
    )

    print("Original 2x5 protein-position-disjoint folds.")
    print("All methods fitted on the same matched subset.")
    print("No validation/test outcomes accessed.")
    print("ETI features come from a separate study source.")

    print("\nA. MATCHED DATA SUMMARY")

    print(
        f"ETI median corrector: "
        f"{data.eti_corrector_pp.median():.3f} pp"
    )

    print(
        f"ETI median total: "
        f"{data.eti_total_pp.median():.3f} pp"
    )

    print(
        f"VTI median baseline: "
        f"{data.observed_baseline.median():.3f} pp"
    )

    print("\nB. MATCHED-COHORT MODEL FITTING")

    oof, qc = evaluate(
        data,
        X,
        folds,
        lambda: phase26d_model(FROZEN_XGB_SPEC),
    )

    metrics, gains = metrics_and_gains(
        oof,
        args.bootstraps,
    )

    print("\nC. GROUPED OOF MAE / RANKING")
    print(
        metrics.round(4).to_string(index=False)
    )

    print("\nD. PAIRED INCREMENTAL GAINS")
    print("Positive gain means the candidate lowers MAE.")

    print(
        gains.round(4).to_string(index=False)
    )

    print("\nE. FOLD / LEAKAGE QC")
    print(qc.to_string(index=False))

    print("\nF. INTERPRETATION WARNINGS")
    print(
        "* These results apply only to the 66 matched variants."
    )
    print(
        "* Do not compare directly with full-210 TRAIN MAEs."
    )
    print(
        "* ETI is a measured assay, not a mutation-only feature."
    )
    print(
        "* Cross-study ETI/VTI assay calibration is not proven."
    )
    print(
        "* Baseline enters subtraction-derived rescue targets."
    )
    print(
        "* Bootstrap intervals are exploratory and not "
        "multiple-comparison adjusted."
    )

    OUT.mkdir(parents=True, exist_ok=True)

    for name, frame in {
        "oof": oof,
        "metrics": metrics,
        "paired_gains": gains,
        "fold_qc": qc,
        "matched_train": data,
    }.items():
        frame.to_csv(
            OUT / f"phase30c_{name}.csv",
            index=False,
        )

    provenance = {
        "eti_source_sha256": sha256(ETI_CSV),
        "phase30a_oof_sha256": sha256(PHASE30A),
        "vti_parquet_sha256": sha256(VTI_PARQUET),
        "n_variants": 66,
        "n_positions": 56,
        "cv": "Inherited 2x5 position-disjoint folds",
        "status": "TRAIN-only exploratory comparison",
    }

    (
        OUT / "phase30c_provenance.json"
    ).write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print("\nPASS Phase30C. Outputs:", OUT)


if __name__ == "__main__":
    main()
