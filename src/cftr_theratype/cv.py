"""Repeated position-grouped cross-validation on the training set."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cftr_theratype.benchmark import (
    PAIRS, SPLITS, PROTOCOL, OUTPUT, SEED, TARGET,
    make_features,
)
from cftr_theratype.reference import bihler_2024_assay_sequence


N_REPEATS = 3
N_FOLDS = 5
N_BOOTSTRAPS = 3000


def position_folds(positions, seed, n_folds=N_FOLDS):
    """Each position occurs in exactly one validation fold per repeat."""
    unique_positions = np.unique(positions)

    if len(unique_positions) < n_folds:
        raise ValueError("Not enough unique positions for cross-validation.")

    shuffled = np.random.default_rng(seed).permutation(
        unique_positions
    )

    for fold, heldout in enumerate(
        np.array_split(shuffled, n_folds), start=1
    ):
        validation = np.isin(positions, heldout)
        training = ~validation

        assert not set(positions[training]) & set(positions[validation])
        yield fold, training, validation


def candidates(seed):
    def forest():
        return RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=4,
            max_features=0.75,
            random_state=seed,
            n_jobs=-1,
        )

    return {
        "median": (
            DummyRegressor(strategy="median"),
            "position",
        ),
        "position_ridge": (
            make_pipeline(
                StandardScaler(),
                Ridge(alpha=10.0),
            ),
            "position",
        ),
        "sequence_ridge": (
            make_pipeline(
                StandardScaler(),
                Ridge(alpha=100.0),
            ),
            "full",
        ),
        "position_forest": (forest(), "position"),
        "sequence_forest": (forest(), "full"),
        "sequence_forest_no_position": (forest(), "no_position"),
    }


def run_cv():
    print("=" * 85)
    print("CFTR-THERATYPE | REPEATED POSITION-GROUPED CROSS-VALIDATION")
    print("=" * 85)

    protocol = json.loads(PROTOCOL.read_text())

    source_hash = hashlib.sha256(PAIRS.read_bytes()).hexdigest()
    if source_hash != protocol["source_sha256"]:
        raise ValueError("Dataset changed since splits were generated.")

    split_df = pd.read_parquet(SPLITS)

    if split_df["variant_key"].duplicated().any():
        raise ValueError("Duplicate variant keys in split table.")

    # Only the ORIGINAL TRAINING partition is used.
    training = split_df.loc[
        split_df["split"] == "train",
        [
            "variant_key", "protein_position",
            "reference_aa", "alternate_aa",
        ],
    ].copy()

    labels = pd.read_parquet(
        PAIRS, columns=["variant_key", TARGET]
    )

    training = training.merge(
        labels,
        on="variant_key",
        how="left",
        validate="one_to_one",
    ).sort_values("variant_key").reset_index(drop=True)

    if training[TARGET].isna().any():
        raise ValueError("Missing training labels.")

    positions = training["protein_position"].to_numpy(dtype=int)
    targets = training[TARGET].to_numpy(dtype=float)

    X, names = make_features(
        training, bihler_2024_assay_sequence()
    )

    position_index = names.index("position_frac")

    matrices = {
        "full": X,
        "position": X[:, [position_index]],
        "no_position": np.delete(X, position_index, axis=1),
    }

    print("\n[1] EXPERIMENT PROTOCOL")
    print(f"Training variants:  {len(training)}")
    print(f"Unique positions:   {len(np.unique(positions))}")
    print(f"Repeats:            {N_REPEATS}")
    print(f"Folds per repeat:   {N_FOLDS}")
    print(f"Candidate models:   {len(candidates(SEED))}")
    print("Original validation: excluded")
    print("Final test:          excluded")

    records = []

    print("\n[2] CROSS-VALIDATION PROGRESS")

    for repeat in range(N_REPEATS):
        folds = list(
            position_folds(
                positions,
                seed=SEED + 100 + repeat,
            )
        )

        coverage = np.zeros(len(training), dtype=int)

        for fold, train_mask, val_mask in folds:
            coverage[val_mask] += 1

            print(
                f"Repeat {repeat + 1}/{N_REPEATS} | "
                f"Fold {fold}/{N_FOLDS} | "
                f"fit={train_mask.sum():3d} | "
                f"heldout={val_mask.sum():3d}",
                flush=True,
            )

            for model_name, (estimator, matrix_name) in candidates(
                SEED + repeat * N_FOLDS + fold
            ).items():
                features = matrices[matrix_name]

                estimator.fit(
                    features[train_mask],
                    targets[train_mask],
                )

                predictions = estimator.predict(
                    features[val_mask]
                )

                indexes = np.flatnonzero(val_mask)

                for idx, prediction in zip(indexes, predictions):
                    records.append({
                        "repeat": repeat + 1,
                        "fold": fold,
                        "model": model_name,
                        "variant_key": training.iloc[idx]["variant_key"],
                        "protein_position": int(positions[idx]),
                        "observed_pp": float(targets[idx]),
                        "predicted_pp": float(prediction),
                        "absolute_error_pp": float(
                            abs(targets[idx] - prediction)
                        ),
                    })

        assert np.all(coverage == 1), (
            "Every training variant must receive one "
            "out-of-fold prediction per repeat."
        )

    predictions = pd.DataFrame(records)

    print("\n[3] POSITION-BALANCED PERFORMANCE")

    # Average each model's out-of-fold error at each protein position.
    # Each position consequently contributes equal weight.
    position_errors = (
        predictions.groupby(
            ["protein_position", "model"]
        )["absolute_error_pp"]
        .mean()
        .unstack("model")
        .sort_index()
    )

    if position_errors.isna().any().any():
        raise ValueError("Incomplete model-position prediction coverage.")

    rng = np.random.default_rng(SEED)
    bootstrap_indices = rng.integers(
        0,
        len(position_errors),
        size=(N_BOOTSTRAPS, len(position_errors)),
    )

    summary = []

    for model in position_errors.columns:
        errors = position_errors[model].to_numpy()

        mean_mae = float(errors.mean())
        bootstrap_mae = errors[bootstrap_indices].mean(axis=1)
        low, high = np.quantile(
            bootstrap_mae, [0.025, 0.975]
        )

        record = {
            "model": model,
            "position_mae_pp": mean_mae,
            "ci_low_pp": float(low),
            "ci_high_pp": float(high),
        }

        for comparator in ("median", "sequence_ridge"):
            if model == comparator:
                continue

            # Positive improvement means the candidate is better.
            improvement = (
                position_errors[comparator].to_numpy() - errors
            )
            bootstrap_gain = improvement[
                bootstrap_indices
            ].mean(axis=1)

            gain_low, gain_high = np.quantile(
                bootstrap_gain, [0.025, 0.975]
            )

            record[f"gain_vs_{comparator}_pp"] = float(
                improvement.mean()
            )
            record[f"gain_vs_{comparator}_ci"] = [
                float(gain_low),
                float(gain_high),
            ]

        summary.append(record)

    summary.sort(key=lambda x: x["position_mae_pp"])

    print(
        f"{'Model':32s} {'Pos-MAE':>9s} "
        f"{'95% CI':>19s} {'Gain vs median':>16s}"
    )

    for result in summary:
        ci = (
            f"[{result['ci_low_pp']:.2f}, "
            f"{result['ci_high_pp']:.2f}]"
        )
        gain = result.get("gain_vs_median_pp", 0.0)

        print(
            f"{result['model']:32s} "
            f"{result['position_mae_pp']:9.3f} "
            f"{ci:>19s} "
            f"{gain:16.3f}"
        )

    print("\n[4] PAIRED MODEL COMPARISONS")

    for result in summary:
        if result["model"] in ("median", "sequence_ridge"):
            continue

        print(f"\n{result['model']}")

        for comparator in ("median", "sequence_ridge"):
            gain = result[f"gain_vs_{comparator}_pp"]
            low, high = result[f"gain_vs_{comparator}_ci"]

            print(
                f"  vs {comparator}: "
                f"improvement={gain:+.3f} pp, "
                f"95% bootstrap CI=[{low:+.3f}, {high:+.3f}]"
            )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(
        OUTPUT / "cv_predictions.csv", index=False
    )

    position_errors.to_csv(
        OUTPUT / "cv_position_errors.csv"
    )

    report = {
        "source_sha256": source_hash,
        "experiment": "repeated_position_grouped_cv",
        "data_partition": "original_training_only",
        "n_repeats": N_REPEATS,
        "n_folds": N_FOLDS,
        "seed": SEED,
        "bootstrap_unit": "protein_position",
        "bootstrap_replicates": N_BOOTSTRAPS,
        "results": summary,
        "original_validation_evaluated": False,
        "test_evaluated": False,
    }

    (OUTPUT / "cv_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("\n[5] REPRODUCIBILITY")
    print(f"Results: {OUTPUT.relative_to(Path.cwd()) if OUTPUT.is_relative_to(Path.cwd()) else OUTPUT}")
    print("Original validation and final test were not evaluated.")
    print("CV COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    run_cv()
