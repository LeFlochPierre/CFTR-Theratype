"""Leakage-aware CFTR ETI-response baselines; validation data only."""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Align import substitution_matrices
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cftr_theratype.reference import bihler_2024_assay_sequence


ROOT = Path(__file__).resolve().parents[2]

PAIRS = ROOT / "data/processed/bihler_2024_theratyping.parquet"
SPLITS = ROOT / "data/processed/bihler_2024_missense_splits.parquet"
PROTOCOL = ROOT / "data/manifests/split_protocol.json"
OUTPUT = ROOT / "results/baselines"

TARGET = "eti_rescue_pp"
SEED = 42
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

HYDROPATHY = {
    "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5,
    "F": 2.8, "G": -0.4, "H": -3.2, "I": 4.5,
    "K": -3.9, "L": 3.8, "M": 1.9, "N": -3.5,
    "P": -1.6, "Q": -3.5, "R": -4.5, "S": -0.8,
    "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
}

CHARGE = {
    aa: 1.0 if aa in "KR" else -1.0 if aa in "DE" else 0.0
    for aa in AMINO_ACIDS
}

BLOSUM62 = substitution_matrices.load("BLOSUM62")


def make_features(frame, sequence):
    """Compute features exclusively from the sequence and mutation."""

    names = [
        "position_frac",
        "blosum62",
        "hydropathy_delta",
        "charge_delta",
        "to_proline",
        "from_proline",
        "to_glycine",
        "from_glycine",
    ]

    names += [f"wt_{aa}" for aa in AMINO_ACIDS]
    names += [f"alt_{aa}" for aa in AMINO_ACIDS]
    names += [f"flank_freq_{aa}" for aa in AMINO_ACIDS]

    rows = []

    for row in frame.itertuples(index=False):
        position = int(row.protein_position)
        wt = str(row.reference_aa)
        alt = str(row.alternate_aa)

        if not (
            1 <= position <= len(sequence)
            and wt in AMINO_ACIDS
            and alt in AMINO_ACIDS
            and wt != alt
            and sequence[position - 1] == wt
        ):
            raise ValueError(
                f"Invalid experimental-reference substitution: "
                f"{wt}{position}{alt}"
            )

        index = position - 1

        # Ten residues on each side, excluding the mutated residue.
        flanks = (
            sequence[max(0, index - 10):index]
            + sequence[index + 1:min(len(sequence), index + 11)]
        )

        values = [
            position / len(sequence),
            float(BLOSUM62[wt, alt]),
            HYDROPATHY[alt] - HYDROPATHY[wt],
            CHARGE[alt] - CHARGE[wt],
            float(alt == "P"),
            float(wt == "P"),
            float(alt == "G"),
            float(wt == "G"),
        ]

        values += [float(wt == aa) for aa in AMINO_ACIDS]
        values += [float(alt == aa) for aa in AMINO_ACIDS]

        values += [
            flanks.count(aa) / len(flanks)
            for aa in AMINO_ACIDS
        ]

        rows.append(values)

    X = np.asarray(rows, dtype=float)

    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains nonfinite values.")

    return X, names


def position_mae_and_ci(y, predictions, positions):
    """Equal-weighted position MAE with a grouped bootstrap interval."""

    errors = pd.DataFrame({
        "position": positions,
        "absolute_error": np.abs(y - predictions),
    })

    position_errors = (
        errors.groupby("position", sort=True)["absolute_error"]
        .mean()
        .to_numpy()
    )

    rng = np.random.default_rng(SEED)

    samples = rng.choice(
        position_errors,
        size=(2000, len(position_errors)),
        replace=True,
    )

    interval = np.quantile(samples.mean(axis=1), [0.025, 0.975])

    return (
        float(position_errors.mean()),
        [float(value) for value in interval],
    )


def evaluate(y, predictions, positions):
    """Continuous-response metrics for the validation partition."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho = float(spearmanr(y, predictions).statistic)

    grouped_mae, interval = position_mae_and_ci(
        y, predictions, positions
    )

    return {
        "mae_pp": float(np.mean(np.abs(y - predictions))),
        "rmse_pp": float(
            np.sqrt(np.mean((y - predictions) ** 2))
        ),
        "spearman": rho if np.isfinite(rho) else None,
        "position_mae_pp": grouped_mae,
        "position_mae_95pct_bootstrap_ci": interval,
    }


def run_baselines():
    print("=" * 85)
    print("CFTR-THERATYPE | SEQUENCE-ONLY BASELINES")
    print("=" * 85)

    # Verify the training data have not changed since splitting.
    protocol = json.loads(PROTOCOL.read_text())

    dataset_hash = hashlib.sha256(PAIRS.read_bytes()).hexdigest()

    if dataset_hash != protocol["source_sha256"]:
        raise ValueError(
            "The source dataset changed after splitting. "
            "Regenerate validation splits before training."
        )

    splits = pd.read_parquet(SPLITS)

    if set(splits["split"]) != {"train", "val", "test"}:
        raise ValueError("Expected train, val and test partitions.")

    if splits["variant_key"].duplicated().any():
        raise ValueError("Duplicate variant identifiers in splits.")

    groups = {
        name: set(
            splits.loc[
                splits["split"] == name,
                "protein_position"
            ]
        )
        for name in ("train", "val", "test")
    }

    for a, b in [
        ("train", "val"),
        ("train", "test"),
        ("val", "test"),
    ]:
        if not groups[a].isdisjoint(groups[b]):
            raise ValueError(f"Protein-position leakage: {a}/{b}")

    # Only training and validation records enter modeling.
    development = splits.loc[
        splits["split"].isin(["train", "val"])
    ].copy()

    labels = pd.read_parquet(
        PAIRS,
        columns=["variant_key", TARGET],
    )

    development = development.merge(
        labels,
        on="variant_key",
        how="left",
        validate="one_to_one",
    )

    if development[TARGET].isna().any():
        raise ValueError("Missing regression targets.")

    development = development.sort_values(
        ["split", "variant_key"]
    ).reset_index(drop=True)

    sequence = bihler_2024_assay_sequence()

    X, feature_names = make_features(development, sequence)

    train = development["split"].eq("train").to_numpy()
    val = development["split"].eq("val").to_numpy()

    y_train = development.loc[
        train, TARGET
    ].to_numpy(dtype=float)

    y_val = development.loc[
        val, TARGET
    ].to_numpy(dtype=float)

    val_positions = development.loc[
        val, "protein_position"
    ].to_numpy(dtype=int)

    print("\n[1] DEVELOPMENT PARTITIONS")
    print(
        f"TRAIN | {train.sum()} variants | "
        f"{len(groups['train'])} positions"
    )
    print(
        f"VAL   | {val.sum()} variants | "
        f"{len(groups['val'])} positions"
    )
    print(
        f"TEST  | {(splits['split'] == 'test').sum()} "
        "variants held out"
    )
    print(f"Sequence-derived features: {len(feature_names)}")

    position_col = feature_names.index("position_frac")

    candidates = [
        (
            "median",
            "-",
            DummyRegressor(strategy="median"),
            X[:, :1],
        ),
    ]

    for alpha in (1.0, 10.0, 100.0):
        candidates.append((
            "position_ridge",
            f"alpha={alpha:g}",
            make_pipeline(
                StandardScaler(),
                Ridge(alpha=alpha),
            ),
            X[:, [position_col]],
        ))

        candidates.append((
            "sequence_ridge",
            f"alpha={alpha:g}",
            make_pipeline(
                StandardScaler(),
                Ridge(alpha=alpha),
            ),
            X,
        ))

    for leaf in (4, 10):
        candidates.append((
            "sequence_forest",
            f"leaf={leaf}",
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=leaf,
                max_features=0.75,
                random_state=SEED,
                n_jobs=-1,
            ),
            X,
        ))

    results = []
    predictions = []

    print("\n[2] FITTING BASELINES")

    for name, setting, estimator, features in candidates:
        estimator.fit(features[train], y_train)

        y_pred = np.asarray(
            estimator.predict(features[val]),
            dtype=float,
        )

        metrics = evaluate(
            y_val, y_pred, val_positions
        )

        results.append({
            "model": name,
            "setting": setting,
            **metrics,
        })

        for key, position, actual, predicted in zip(
            development.loc[val, "variant_key"],
            val_positions,
            y_val,
            y_pred,
        ):
            predictions.append({
                "model": name,
                "setting": setting,
                "variant_key": key,
                "protein_position": int(position),
                "observed_rescue_pp": float(actual),
                "predicted_rescue_pp": float(predicted),
            })

    results.sort(
        key=lambda result: (
            result["position_mae_pp"],
            result["mae_pp"],
        )
    )

    print("\n[3] VALIDATION PERFORMANCE")
    print(
        f"{'Model':22s} {'Setting':12s} "
        f"{'MAE':>8s} {'RMSE':>8s} "
        f"{'Pos-MAE':>10s} {'Spearman':>10s}"
    )

    for result in results:
        rho = (
            f"{result['spearman']:.3f}"
            if result["spearman"] is not None
            else "n/a"
        )

        print(
            f"{result['model']:22s} "
            f"{result['setting']:12s} "
            f"{result['mae_pp']:8.3f} "
            f"{result['rmse_pp']:8.3f} "
            f"{result['position_mae_pp']:10.3f} "
            f"{rho:>10s}"
        )

    best = results[0]

    OUTPUT.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset_sha256": dataset_hash,
        "split_protocol": protocol,
        "reference_sequence_sha256": hashlib.sha256(
            sequence.encode()
        ).hexdigest(),
        "selection_metric": "position_mae_pp",
        "feature_names": feature_names,
        "candidates": results,
        "selected_on_validation": {
            "model": best["model"],
            "setting": best["setting"],
        },
        "test_evaluated": False,
    }

    (OUTPUT / "validation_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    pd.DataFrame(predictions).to_csv(
        OUTPUT / "validation_predictions.csv",
        index=False,
    )

    print("\n[4] SELECTED VALIDATION MODEL")
    print(f"Model:       {best['model']}")
    print(f"Setting:     {best['setting']}")
    print(f"Position MAE: {best['position_mae_pp']:.3f} pp")
    print(
        "Position-bootstrap 95% interval:",
        [
            round(value, 3)
            for value in best["position_mae_95pct_bootstrap_ci"]
        ],
    )

    print("\n[5] REPRODUCIBILITY")
    print(f"Outputs: {OUTPUT.relative_to(ROOT)}")
    print("Test performance: NOT evaluated")
    print("=" * 85)


if __name__ == "__main__":
    run_baselines()
