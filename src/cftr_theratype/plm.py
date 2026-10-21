"""Compare frozen ESM-2 features using training-only grouped CV and validation.

Usage: cftr-data plm
No ESM inference or model fine-tuning occurs in this module.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cftr_theratype.benchmark import PAIRS, SPLITS, PROTOCOL, SEED, TARGET, make_features
from cftr_theratype.cv import position_folds
from cftr_theratype.features.esm2 import OUTPUT as EMBEDDINGS, MODEL_ID, WINDOW
from cftr_theratype.reference import bihler_2024_assay_sequence

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "plm"
ALPHAS = (10.0, 100.0, 1000.0, 10000.0)
REPEATS = 3
FOLDS = 5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_and_load():
    """Validate provenance, strict key alignment and train/validation scope."""
    protocol = json.loads(PROTOCOL.read_text())
    if sha256(PAIRS) != protocol["source_sha256"]:
        raise ValueError("Processed labels changed since split generation.")

    splits = pd.read_parquet(SPLITS)
    if splits["variant_key"].duplicated().any():
        raise ValueError("Duplicate variant keys in split table.")
    if set(splits["split"]) != {"train", "val", "test"}:
        raise ValueError("Expected exactly train/val/test split labels.")
    groups = {
        split: set(splits.loc[splits.split == split, "protein_position"])
        for split in ("train", "val", "test")
    }
    if any(groups[a] & groups[b] for a, b in
           (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Protein-position leakage among splits.")

    dev = (splits.loc[splits.split.isin(["train", "val"])]
           .sort_values(["split", "variant_key"]).reset_index(drop=True))
    labels = pd.read_parquet(PAIRS, columns=["variant_key", TARGET])
    dev = dev.merge(labels, on="variant_key", how="left", validate="one_to_one")
    if dev[TARGET].isna().any():
        raise ValueError("Missing labels for development mutations.")

    sequence = bihler_2024_assay_sequence()
    expected_sequence_sha = hashlib.sha256(sequence.encode()).hexdigest()
    with np.load(EMBEDDINGS, allow_pickle=False) as cache:
        metadata = json.loads(str(cache["metadata"].item()))
        expected = {
            "model_id": MODEL_ID,
            "window": WINDOW,
            "sequence_sha256": expected_sequence_sha,
            "split_sha256": sha256(SPLITS),
            "dataset_sha256": sha256(PAIRS),
            "num_variants": len(dev),
            "mode": "development_only",
        }
        for key, val in expected.items():
            if metadata.get(key) != val:
                raise ValueError(f"Embedding cache mismatch for {key}: {metadata.get(key)}")
        keys = cache["keys"].astype(str)
        if not np.array_equal(keys, dev.variant_key.to_numpy(dtype=str)):
            raise ValueError("ESM cache key order differs from sorted development variants.")
        embeddings = {key: np.asarray(cache[key], dtype=np.float64)
                      for key in ("wt", "mutant", "delta")}

    for name, x in embeddings.items():
        if x.shape != (len(dev), 480) or not np.isfinite(x).all():
            raise ValueError(f"Bad {name} embedding shape or nonfinite values: {x.shape}")
    if not np.allclose(embeddings["mutant"] - embeddings["wt"],
                       embeddings["delta"], rtol=1e-4, atol=1e-5):
        raise ValueError("Saved ESM deltas are not consistent with WT and mutant vectors.")
    sequence_features, names = make_features(dev, sequence)

    matrices = {
        "esm_wt": embeddings["wt"],                 # site-context control
        "esm_mutant": embeddings["mutant"],
        "esm_delta": embeddings["delta"],
        "esm_wt_delta": np.column_stack([embeddings["wt"], embeddings["delta"]]),
        "esm_delta_sequence": np.column_stack([embeddings["delta"], sequence_features]),
    }
    return dev, matrices, metadata


def make_ridge(alpha):
    # Scaling is fitted inside each CV training fold, avoiding leakage.
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr",
                                                 tol=1e-5, max_iter=2000))


def position_mae(y, prediction, positions):
    return float(pd.DataFrame({"position": positions, "err": np.abs(y-prediction)})
                 .groupby("position").err.mean().mean())


def scores(y, prediction, positions):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        correlation = float(spearmanr(y, prediction).statistic)
    return {
        "mae_pp": float(np.mean(np.abs(y - prediction))),
        "rmse_pp": float(np.sqrt(np.mean((y - prediction)**2))),
        "position_mae_pp": position_mae(y, prediction, positions),
        "spearman": correlation if np.isfinite(correlation) else None,
    }


def bootstrap_gain(y, candidate, comparator, positions, n_boot=4000):
    """Paired resampling of held-out positions: positive is an improvement."""
    d = pd.DataFrame({"pos": positions,
                      "gain": np.abs(y-comparator) - np.abs(y-candidate)})
    by_position = d.groupby("pos").gain.mean().to_numpy()
    rng = np.random.default_rng(SEED)
    index = rng.integers(0, len(by_position), (n_boot, len(by_position)))
    ci = np.quantile(by_position[index].mean(axis=1), [0.025, 0.975])
    return float(by_position.mean()), [float(ci[0]), float(ci[1])]


def run_plm():
    print("="*85)
    print("CFTR-THERATYPE | FROZEN ESM-2 REPRESENTATION BENCHMARK")
    print("="*85)
    dev, matrices, metadata = check_and_load()
    train = dev.split.eq("train").to_numpy()
    val = dev.split.eq("val").to_numpy()
    yt = dev.loc[train, TARGET].to_numpy(dtype=float)
    yv = dev.loc[val, TARGET].to_numpy(dtype=float)
    pt = dev.loc[train, "protein_position"].to_numpy(dtype=int)
    pv = dev.loc[val, "protein_position"].to_numpy(dtype=int)

    print("\n[1] DESIGN")
    print(f"Train: {len(yt)} variants / {len(np.unique(pt))} positions")
    print(f"Validation: {len(yv)} variants / {len(np.unique(pv))} positions")
    print("Test: excluded | Embedding model frozen | No neural-network training")
    print(f"CV: {REPEATS} repeats x {FOLDS} position-disjoint folds")
    print(f"Ridge alphas: {ALPHAS}")
    print(f"Representations: {', '.join(matrices)}")

    # Choose alpha for each representation exclusively within train.
    print("\n[2] TRAINING-ONLY GROUPED CROSS-VALIDATION")
    folds = [(tr, ho) for r in range(REPEATS)
             for _, tr, ho in position_folds(pt, seed=SEED+100+r, n_folds=FOLDS)]
    results = []
    selected = {}
    for name, matrix in matrices.items():
        X = matrix[train]
        for alpha in ALPHAS:
            # One prediction per training mutation per repeat.
            out = np.full((REPEATS, len(X)), np.nan, dtype=float)
            for j, (fit_mask, hold_mask) in enumerate(folds):
                model = make_ridge(alpha)
                model.fit(X[fit_mask], yt[fit_mask])
                out[j // FOLDS, hold_mask] = model.predict(X[hold_mask])
            if not np.isfinite(out).all():
                raise ValueError(f"Incomplete CV predictions for {name}, alpha={alpha}")
            by_position = (pd.DataFrame({"position": np.tile(pt, REPEATS),
                                         "error": np.abs(np.tile(yt, REPEATS)-out.ravel())})
                           .groupby("position").error.mean())
            results.append({"representation": name, "alpha": alpha,
                            "position_mae_pp": float(by_position.mean())})
        candidates = [r for r in results if r["representation"] == name]
        best = min(candidates, key=lambda row: (row["position_mae_pp"], row["alpha"]))
        selected[name] = best
        print(f"{name:23s} alpha={best['alpha']:7g} "
              f"CV position MAE={best['position_mae_pp']:.3f} pp", flush=True)

    # Alpha selection is complete; first validation evaluation happens here.
    print("\n[3] ORIGINAL VALIDATION (EXPLORATORY COMPARISON)")
    validation_predictions = pd.DataFrame({"variant_key": dev.loc[val, "variant_key"].to_numpy(),
                                            "protein_position": pv,
                                            "observed_pp": yv})
    summary = []
    for name, matrix in matrices.items():
        best = selected[name]
        model = make_ridge(best["alpha"])
        model.fit(matrix[train], yt)
        predicted = np.asarray(model.predict(matrix[val]), dtype=float)
        validation_predictions[name] = predicted
        result = {"representation": name, "alpha": best["alpha"],
                  "cv_position_mae_pp": best["position_mae_pp"],
                  **scores(yv, predicted, pv)}
        summary.append(result)
        print(f"{name:23s} MAE={result['mae_pp']:6.2f} "
              f"Pos-MAE={result['position_mae_pp']:6.2f} "
              f"Spearman={result['spearman']}")

    # Compare with the *existing*, previously trained forest predictions.
    existing = ROOT / "results/baselines/validation_predictions.csv"
    forest_comparison = None
    if existing.exists():
        historical = pd.read_csv(existing)
        forest = historical.loc[(historical.model == "sequence_forest") &
                                (historical.setting == "leaf=4")]
        if forest.variant_key.duplicated().any():
            raise ValueError("Repeated Forest validation predictions.")
        forest = validation_predictions[["variant_key", "protein_position", "observed_pp"]].merge(
            forest[["variant_key", "protein_position", "observed_rescue_pp", "predicted_rescue_pp"]],
            on=["variant_key", "protein_position"], how="left", validate="one_to_one")
        if forest.predicted_rescue_pp.isna().any() or not np.allclose(
                forest.observed_pp, forest.observed_rescue_pp):
            raise ValueError("Saved Forest predictions do not match validation cohort.")
        candidate = min(summary, key=lambda s: s["cv_position_mae_pp"])["representation"]
        comparison_gain, comparison_ci = bootstrap_gain(
            yv, validation_predictions[candidate].to_numpy(),
            forest.predicted_rescue_pp.to_numpy(), pv)
        forest_comparison = {"selected_plm": candidate,
                             "forest_position_mae_pp": position_mae(
                                 yv, forest.predicted_rescue_pp.to_numpy(), pv),
                             "esm_minus_forest_error_reduction_pp": comparison_gain,
                             "position_bootstrap_95_ci_pp": comparison_ci}
        print("\n[4] PAIRED COMPARISON TO ORIGINAL SEQUENCE FOREST")
        print(f"CV-selected ESM: {candidate}")
        print(f"Forest validation Pos-MAE: {forest_comparison['forest_position_mae_pp']:.3f} pp")
        print(f"ESM error reduction vs Forest: {comparison_gain:+.3f} pp")
        print(f"Paired position-bootstrap 95% CI: [{comparison_ci[0]:+.3f}, {comparison_ci[1]:+.3f}] pp")
    else:
        print("\n[4] Forest comparison skipped: no saved validation predictions.")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT / "train_grouped_cv.csv", index=False)
    validation_predictions.to_csv(OUTPUT / "validation_predictions.csv", index=False)
    report = {"source_data_sha256": sha256(PAIRS), "split_sha256": sha256(SPLITS),
              "embedding_sha256": sha256(EMBEDDINGS), "embedding_metadata": metadata,
              "training_cv": results, "selected_alphas": selected,
              "validation": summary, "comparison": forest_comparison,
              "test_evaluated": False,
              "limitation": "Exploratory validation; previous baseline design used this set."}
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print("\n[5] OUTPUTS")
    print(f"Results: {OUTPUT.relative_to(ROOT)}")
    print("Final test: NOT evaluated")
    print("="*85)


if __name__ == "__main__":
    run_plm()
