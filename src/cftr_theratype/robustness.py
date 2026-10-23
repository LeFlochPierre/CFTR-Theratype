"""Train-only, cohort-balanced CFTR theratyping robustness benchmark.

Experimental question: Can known *training-study* cohort membership be used to
reweight a VTI-only sequence model, without requiring membership at inference?

Reuses frozen 2026 training variants, existing grouped OOF folds and the same
Forest architecture. Cohort is TRAINING METADATA only, never a predictor.
Neither study's validation/test outcomes are loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cftr_theratype.benchmark import make_features
from cftr_theratype.cohort_shift import COHORTS, _load_train_only_labels
from cftr_theratype.reference import bihler_2024_assay_sequence
from cftr_theratype.splits import parse_substitution
from cftr_theratype.transfer import (
    MANIFEST, RESULTS, SEED, SPLIT_OUT, VTI_PAIRS, _estimator, sha256,
)

STRENGTHS = (0.0, 0.5, 1.0)
MODEL_NAMES = {0.0: "vti_forest", 0.5: "balanced_forest_half", 1.0: "balanced_forest_full"}


def cohort_weights(cohorts, strength: float) -> np.ndarray:
    """Interpolate from uniform to inverse-cohort-frequency sample weights.

    Weights sum to sample count, so total effective weight is unchanged.
    Both cohorts must be represented in the *training fold*.
    """
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be between 0 and 1")
    labels = np.asarray(cohorts, dtype=object)
    if labels.ndim != 1 or len(labels) == 0 or set(labels) != set(COHORTS):
        raise ValueError("Both established study cohorts must be present")
    counts = {cohort: int(np.sum(labels == cohort)) for cohort in COHORTS}
    n = len(labels)
    ideal = np.array([n / (2 * counts[label]) for label in labels], dtype=float)
    weights = np.power(ideal, strength)
    weights *= n / weights.sum()
    assert np.isfinite(weights).all() and np.all(weights > 0)
    return weights


def position_metrics(oof: pd.DataFrame, *, n_boot: int = 2500, seed: int = SEED):
    """Compute per-position metrics and paired bootstrap gains.

    Average predictions across repeats per variant before computing error.
    Within each cohort, each protein position contributes equal weight.
    """
    models = tuple(MODEL_NAMES.values())
    required = {"repeat", "model", "variant_key", "protein_position", "study_cohort",
                "observed_pp", "predicted_pp"}
    if not required.issubset(oof.columns):
        raise ValueError(f"Missing OOF columns: {sorted(required - set(oof.columns))}")
    if set(oof.model) != set(models) or set(oof.study_cohort) != set(COHORTS):
        raise ValueError("Incomplete model or cohort coverage")
    if oof.duplicated(["model", "repeat", "variant_key"]).any():
        raise ValueError("Duplicate model/repeat/variant predictions")
    num_repeats = int(oof.repeat.nunique())
    if num_repeats != 2:
        raise ValueError("Expected two repeated training-CV predictions")
    if oof.groupby(["model", "variant_key"]).repeat.nunique().ne(num_repeats).any():
        raise ValueError("Incomplete OOF repeat coverage")
    for key, sub in oof.groupby("variant_key"):
        if sub.protein_position.nunique() != 1 or sub.study_cohort.nunique() != 1:
            raise ValueError(f"Inconsistent position/cohort for {key}")
        if sub.observed_pp.max() - sub.observed_pp.min() > 1e-6:
            raise ValueError(f"Inconsistent observed target for {key}")

    means = oof.groupby(["model", "variant_key", "protein_position", "study_cohort"],
                        as_index=False).agg(observed_pp=("observed_pp", "mean"),
                                             predicted_pp=("predicted_pp", "mean"))
    means["abs_error_pp"] = (means.predicted_pp - means.observed_pp).abs()
    means["signed_error_pp"] = means.predicted_pp - means.observed_pp

    summary = []
    for cohort in ("all", *COHORTS):
        chosen = means if cohort == "all" else means.loc[means.study_cohort.eq(cohort)]
        per_position = chosen.groupby(["protein_position", "model"], sort=True).agg(
            mae_pp=("abs_error_pp", "mean"),
            bias_pp=("signed_error_pp", "mean"),
        ).unstack("model")
        if per_position.isna().any().any():
            raise ValueError("Position coverage is incomplete across models")
        error = per_position["mae_pp"]
        bias = per_position["bias_pp"]
        if len(error) < 3:
            raise ValueError("Insufficient positions for group-bootstrap comparisons")
        rng = np.random.default_rng(seed + (0 if cohort == "all" else COHORTS.index(cohort) + 1))
        draws = rng.integers(0, len(error), size=(n_boot, len(error)))
        baseline = error["vti_forest"].to_numpy(dtype=float)
        for model in models:
            error_vec = error[model].to_numpy(dtype=float)
            signed_vec = bias[model].to_numpy(dtype=float)
            gain = baseline - error_vec
            lo, hi = np.quantile(gain[draws].mean(axis=1), [0.025, 0.975])
            summary.append({
                "cohort": cohort,
                "model": model,
                "n_positions": int(len(error)),
                "position_mae_pp": float(error_vec.mean()),
                "mean_bias_pp": float(signed_vec.mean()),
                "gain_vs_forest_pp": float(gain.mean()),
                "gain_ci_95_pp": [float(lo), float(hi)],
            })
    return pd.DataFrame(summary), means


def run_robustness() -> None:
    print("=" * 84)
    print("CFTR-THERATYPE | COHORT-BALANCED TRAINING — FIXED CV FOLDS")
    print("=" * 84)

    protocol = json.loads(MANIFEST.read_text())
    if (protocol.get("vti_source_sha256") != sha256(VTI_PAIRS)
            or protocol.get("membership_sha256") is None):
        raise ValueError("Frozen VTI provenance has changed or is incomplete")
    original_path = RESULTS / "train_cv_predictions.csv"
    original_summary_path = RESULTS / "summary.json"
    prior = json.loads(original_summary_path.read_text())
    if (prior.get("vti_source_sha256") != sha256(VTI_PAIRS)
            or prior.get("vti_partition_sha256") != sha256(SPLIT_OUT)):
        raise ValueError("Existing OOF predictions do not match frozen data")

    frozen = pd.read_parquet(SPLIT_OUT, columns=[
        "variant_key", "protein_position", "transfer_split", "study_cohort",
        "variant_type", "variant", "hgvs_protein",
    ])
    if frozen.variant_key.isna().any() or frozen.variant_key.duplicated().any():
        raise ValueError("Frozen VTI variant keys are not unique")
    train = frozen.loc[frozen.transfer_split.eq("train")].copy()
    keys = set(train.variant_key.astype(str))
    if len(train) != 210 or train.protein_position.nunique() != 168:
        raise ValueError("Unexpected VTI train size; frozen cohort differs")
    if set(train.study_cohort) != set(COHORTS):
        raise ValueError("Unexpected study strata")

    labels = _load_train_only_labels(keys)[["variant_key", "vti_rescue_pp"]]
    train = train.merge(labels, on="variant_key", validate="one_to_one")
    seq = bihler_2024_assay_sequence()
    parsed = [parse_substitution(row, seq) for _, row in train.iterrows()]
    if any(x[-1] != "valid" or x[0] != pos for x, pos in
           zip(parsed, train.protein_position)):
        raise ValueError("VTI variants do not match experimental reference")
    train["reference_aa"] = [p[1] for p in parsed]
    train["alternate_aa"] = [p[2] for p in parsed]
    train = train.sort_values("variant_key").reset_index(drop=True)
    X, names = make_features(train, seq)
    by_key = train.set_index("variant_key")
    row_idx = {key: i for i, key in enumerate(train.variant_key)}
    positions = train.protein_position.to_numpy(dtype=int)

    raw = pd.read_csv(original_path)
    ref = raw.loc[raw.model.eq("vti_forest")].copy()
    if set(ref.variant_key.astype(str)) != keys:
        raise ValueError("Stored OOF Forest has unexpected variant keys")
    if ref.duplicated(["repeat", "variant_key"]).any():
        raise ValueError("Duplicate OOF Forest predictions")
    if ref.groupby("variant_key").repeat.nunique().ne(2).any():
        raise ValueError("Incomplete repeated OOF predictions")
    if set(ref.repeat.unique()) != {1, 2}:
        raise ValueError("Unexpected stored repeat indices")
    if not np.allclose(ref.observed_pp, ref.variant_key.map(by_key.vti_rescue_pp), rtol=0, atol=1e-5):
        raise ValueError("Stored OOF outcomes disagree with current VTI train outcomes")
    if not np.array_equal(ref.protein_position.to_numpy(int), ref.variant_key.map(by_key.protein_position)):
        raise ValueError("Stored OOF positions disagree with the current split")
    ref["study_cohort"] = ref.variant_key.map(by_key.study_cohort)
    if ref.study_cohort.isna().any():
        raise ValueError("Missing group metadata")

    print("\n[1] PROTOCOL")
    print(f"VTI training: {len(train)} variants, {len(np.unique(positions))} positions")
    print(f"Training cohort sizes: {train.study_cohort.value_counts().to_dict()}")
    print("CV: existing 2 repeats x 5 position-disjoint folds")
    print("Forests: same architecture, only training sample weights change")
    print("Strengths: 0 (existing), 0.5 (partial), 1 (equal cohort totals)")
    print("Retrospective cohort labels: TRAINING WEIGHTS ONLY, never predictors")
    print("Validation/test and ETI outcomes: NOT READ")

    extra = []
    print("\n[2] FITTING COHORT-WEIGHTED FORESTS")
    folds = list(ref.groupby(["repeat", "fold"], sort=True))
    if len(folds) != 10:
        raise ValueError("Expected ten stored CV folds")
    for (repeat, fold), held in folds:
        held_keys = held.variant_key.astype(str).tolist()
        held_positions = set(int(x) for x in held.protein_position)
        held_indices = np.array([row_idx[k] for k in held_keys], dtype=int)
        fit_indices = np.flatnonzero(~np.isin(positions, list(held_positions)))
        if set(positions[fit_indices]).intersection(held_positions):
            raise ValueError("Training position leakage")
        if set(train.iloc[fit_indices].study_cohort) != set(COHORTS):
            raise ValueError("Training fold lacks a study cohort")
        y_fit = train.iloc[fit_indices].vti_rescue_pp.to_numpy(float)
        fold_cohorts = train.iloc[fit_indices].study_cohort.to_numpy()
        # Match the exact random seed used by original transfer.py.
        fit_seed = SEED + (int(repeat) - 1) * 10 + int(fold)
        print(f"Repeat {repeat}/2 | fold {fold}/5 | train={len(fit_indices)} "
              f"held={len(held_indices)}", flush=True)
        for strength in STRENGTHS[1:]:
            forest = _estimator("vti_forest", fit_seed)
            weights = cohort_weights(fold_cohorts, strength)
            forest.fit(X[fit_indices], y_fit, sample_weight=weights)
            preds = forest.predict(X[held_indices])
            for idx, pred in zip(held_indices, preds):
                row = train.iloc[idx]
                extra.append({
                    "repeat": int(repeat), "fold": int(fold),
                    "model": MODEL_NAMES[strength],
                    "variant_key": str(row.variant_key),
                    "protein_position": int(row.protein_position),
                    "study_cohort": str(row.study_cohort),
                    "observed_pp": float(row.vti_rescue_pp),
                    "predicted_pp": float(pred),
                })

    columns = ["repeat", "fold", "model", "variant_key", "protein_position",
               "study_cohort", "observed_pp", "predicted_pp"]
    combined = pd.concat([ref[columns], pd.DataFrame(extra)[columns]], ignore_index=True)
    summary, per_variant = position_metrics(combined)
    print("\n[3] OVERALL AND COHORT-WISE ERRORS")
    print(f"{'Cohort':10s} {'Model':25s} {'Pos-MAE':>9s} {'Bias':>9s} {'Gain vs forest':>15s} {'95% CI gain':>22s}")
    for cohort in ("all", *COHORTS):
        sub = summary.loc[summary.cohort.eq(cohort)]
        for row in sub.itertuples(index=False):
            lo, hi = row.gain_ci_95_pp
            print(f"{cohort:10s} {row.model:25s} {row.position_mae_pp:9.3f} "
                  f"{row.mean_bias_pp:+9.3f} {row.gain_vs_forest_pp:+15.3f} "
                  f"[{lo:+8.3f}, {hi:+8.3f}]")
    print("\n[4] WORST-COHORT METRICS")
    for model in MODEL_NAMES.values():
        part = summary.loc[(summary.model.eq(model)) & (summary.cohort.ne("all"))]
        print(f"{model:25s} worst position MAE={part.position_mae_pp.max():.3f} pp; "
              f"max |cohort bias|={part.mean_bias_pp.abs().max():.3f} pp")

    RESULTS.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RESULTS / "cohort_robustness_oof.csv", index=False)
    summary.to_csv(RESULTS / "cohort_robustness_summary.csv", index=False)
    report = {
        "experiment": "training-cohort-weighted VTI-only forest; fixed 2x5 grouped OOF",
        "vti_source_sha256": sha256(VTI_PAIRS),
        "vti_split_sha256": sha256(SPLIT_OUT),
        "reference_oof_sha256": hashlib.sha256(original_path.read_bytes()).hexdigest(),
        "features": names,
        "cohort_label_used_as_predictor": False,
        "cohort_label_used_for_train_sample_weight": True,
        "test_or_validation_outcomes_accessed": False,
        "predeclared_weight_strengths": list(STRENGTHS),
        "results": summary.to_dict(orient="records"),
        "limitations": [
            "Retrospective, response-selected cohorts, not randomised groups",
            "Exploratory on training data; do not choose models post hoc using sealed tests",
            "Cohort weights cannot guarantee generalisation under unobserved shift",
        ],
    }
    (RESULTS / "cohort_robustness_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n[5] SAVED")
    print("results/transfer/cohort_robustness_oof.csv")
    print("results/transfer/cohort_robustness_summary.csv")
    print("results/transfer/cohort_robustness_summary.json")
    print("No validation or test labels were read.")
    print("=" * 84)


if __name__ == "__main__":
    run_robustness()
