"""Phase 14C: position-safe, TRAIN-only auxiliary Band C prediction ablation.

Usage: python -m cftr_theratype.maturation_prediction

Does not use measured Band C or study cohort as an inference feature.
Does not access either study's validation/test responses or ETI response labels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

from cftr_theratype.maturation import assemble
from cftr_theratype.transfer import ROOT, RESULTS, MANIFEST, SPLIT_OUT, VTI_PAIRS, SEED, sha256, _estimator

MODELS = ("vti_forest", "rescue_plus_band_c", "rescue_plus_shuffled_band_c")
N_REPEATS = 2
N_FOLDS = 5
BOOTSTRAPS = 2500


def paired_multitask_predictions(x_fit: np.ndarray, y_rescue: np.ndarray,
                                 y_band_c: np.ndarray, x_held: np.ndarray, *,
                                 seed: int, shuffled_aux: bool = False) -> np.ndarray:
    """Return [predicted rescue, predicted Band C gain], trained on fold-fit labels only.

    Each response is z-scaled separately within fold training observations.
    Random Forest multi-output impurity gives the two standardized tasks equal weight.
    No held-out phenotypes are accepted by the function.
    """
    x_fit = np.asarray(x_fit, dtype=float)
    x_held = np.asarray(x_held, dtype=float)
    y1 = np.asarray(y_rescue, dtype=float)
    y2 = np.asarray(y_band_c, dtype=float)
    if (x_fit.ndim != 2 or x_held.ndim != 2 or x_fit.shape[1] != x_held.shape[1]
            or y1.shape != (len(x_fit),) or y2.shape != (len(x_fit),) or len(x_fit) < 15):
        raise ValueError("Incompatible or insufficient training features and labels")
    if not all(np.isfinite(a).all() for a in (x_fit, x_held, y1, y2)):
        raise ValueError("Non-finite training labels/features")
    # The negative control destroys sequence-to-maturation correspondence without
    # changing either target's marginal distribution. Never permute held-out labels.
    aux = np.random.default_rng(seed + 711).permutation(y2) if shuffled_aux else y2
    targets = np.column_stack([y1, aux])
    means, scales = targets.mean(axis=0), targets.std(axis=0)
    scales = np.maximum(scales, 1e-8)
    normalized = (targets - means) / scales
    forest = _estimator("vti_forest", seed)
    forest.fit(x_fit, normalized)
    predicted = forest.predict(x_held) * scales + means
    if predicted.shape != (len(x_held), 2) or not np.isfinite(predicted).all():
        raise AssertionError("Non-finite or unexpected multi-target predictions")
    return predicted


def evaluate_position_predictions(raw: pd.DataFrame, *, n_boot: int = BOOTSTRAPS,
                                  seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired, position-balanced errors after averaging two OOF predictions per variant."""
    needed = {"repeat", "model", "variant_key", "protein_position", "study_cohort",
              "observed_pp", "predicted_pp"}
    if not needed.issubset(raw):
        raise ValueError(f"Missing OOF columns: {sorted(needed-set(raw.columns))}")
    if set(raw.model) != set(MODELS) or set(raw.repeat) != {1, 2}:
        raise ValueError("Expected all fixed models and exactly two repeats")
    if raw.duplicated(["repeat", "model", "variant_key"]).any():
        raise ValueError("Duplicate OOF predictions")
    if raw.groupby(["model", "variant_key"]).repeat.nunique().ne(2).any():
        raise ValueError("Incomplete repeated predictions")
    for key, d in raw.groupby("variant_key"):
        if (d.protein_position.nunique() != 1 or d.study_cohort.nunique() != 1
                or d.observed_pp.max() - d.observed_pp.min() > 1e-6):
            raise ValueError(f"Inconsistent annotations or outcome for {key}")
    averaged = (raw.groupby(["model", "variant_key", "protein_position", "study_cohort"],
                            as_index=False)
                .agg(observed_pp=("observed_pp", "mean"),
                     predicted_pp=("predicted_pp", "mean")))
    averaged["abs_error_pp"] = (averaged.predicted_pp - averaged.observed_pp).abs()
    averaged["signed_error_pp"] = averaged.predicted_pp - averaged.observed_pp
    rows = []
    for cohort in ("all", "ETI, VTI", "VTI"):
        sub = averaged if cohort == "all" else averaged[averaged.study_cohort.eq(cohort)]
        by_pos = sub.groupby(["protein_position", "model"])[["abs_error_pp", "signed_error_pp"]].mean()
        errors = by_pos.abs_error_pp.unstack("model")
        signed = by_pos.signed_error_pp.unstack("model")
        if errors.isna().any().any() or set(errors.columns) != set(MODELS):
            raise ValueError("Position comparison is not fully paired")
        if len(errors) < 4:
            raise ValueError("Too few independent positions")
        rng = np.random.default_rng(seed + (0 if cohort == "all" else (1 if cohort == "ETI, VTI" else 2)))
        draws = rng.integers(0, len(errors), (n_boot, len(errors)))
        ref = errors["vti_forest"].to_numpy()
        shuffled = errors["rescue_plus_shuffled_band_c"].to_numpy()
        for model in MODELS:
            current = errors[model].to_numpy()
            gain = ref - current
            gain_vs_null = shuffled - current
            lo, hi = np.quantile(gain[draws].mean(axis=1), [.025, .975])
            null_lo, null_hi = np.quantile(gain_vs_null[draws].mean(axis=1), [.025, .975])
            rows.append({
                "cohort": cohort, "model": model,
                "n_positions": int(len(errors)),
                "position_mae_pp": float(current.mean()),
                "mean_bias_pp": float(signed[model].mean()),
                "gain_vs_forest_pp": float(gain.mean()),
                "gain_vs_forest_ci95": [float(lo), float(hi)],
                "gain_vs_shuffled_aux_pp": float(gain_vs_null.mean()),
                "gain_vs_shuffled_aux_ci95": [float(null_lo), float(null_hi)],
            })
    return pd.DataFrame(rows), averaged


def _validate_reference(ref: pd.DataFrame, train: pd.DataFrame) -> None:
    keys = set(train.variant_key)
    if ref.empty or set(ref.variant_key) != keys:
        raise ValueError("Stored Forest does not cover exactly frozen VTI train keys")
    if ref.duplicated(["repeat", "variant_key"]).any():
        raise ValueError("Duplicate frozen reference predictions")
    if len(ref.groupby(["repeat", "fold"])) != N_REPEATS * N_FOLDS:
        raise ValueError("Expected 10 frozen CV folds")
    if ref.groupby("variant_key").repeat.nunique().ne(N_REPEATS).any():
        raise ValueError("Not every variant has 2 OOF reference predictions")
    meta = train.set_index("variant_key")
    if not np.array_equal(ref.protein_position.to_numpy(int), ref.variant_key.map(meta.protein_position).to_numpy(int)):
        raise ValueError("Reference differs from frozen position membership")
    if not np.allclose(ref.observed_pp.to_numpy(float), ref.variant_key.map(meta.vti_rescue_pp).to_numpy(float), atol=1e-5, rtol=0):
        raise ValueError("Reference response labels differ from current training")
    if (ref.groupby(["repeat", "variant_key"]).fold.nunique().ne(1).any()
            or ref.groupby(["repeat", "protein_position"]).fold.nunique().ne(1).any()):
        raise ValueError("Inconsistent position-disjoint CV fold mapping")


def run() -> None:
    from cftr_theratype.benchmark import make_features
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.splits import parse_substitution
    print("=" * 92)
    print("CFTR-THERATYPE | PHASE 14C: BAND C AS A TRAIN-ONLY AUXILIARY TARGET")
    print("=" * 92)
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("vti_source_sha256") != sha256(VTI_PAIRS):
        raise ValueError("Frozen VTI source fingerprint does not match")
    source_summary = json.loads((RESULTS / "summary.json").read_text())
    if (source_summary.get("vti_source_sha256") != sha256(VTI_PAIRS)
            or source_summary.get("vti_partition_sha256") != sha256(SPLIT_OUT)):
        raise ValueError("Original training OOF provenance mismatch")

    meta = pd.read_parquet(SPLIT_OUT, columns=[
        "variant_key", "protein_position", "variant", "hgvs_protein",
        "variant_type", "study_cohort", "transfer_split",
    ])
    if meta.variant_key.isna().any() or meta.variant_key.duplicated().any():
        raise ValueError("Duplicate/missing keys in frozen membership")
    frozen_train = meta.loc[meta.transfer_split.eq("train")].drop(columns="transfer_split")
    if (len(frozen_train) != 210 or frozen_train.protein_position.nunique() != 168
            or frozen_train.study_cohort.value_counts().to_dict() != {"VTI": 144, "ETI, VTI": 66}):
        raise ValueError("Unexpected VTI training partition")

    # Audited loader predicates on authorised TRAIN keys before bringing assay
    # values (including post-treatment Band C) into pandas memory.
    labels = assemble(ROOT, "VTI")
    label_cols = ["variant_key", "protein_position", "vti_rescue_pp", "band_c_gain_pct_wt"]
    if (set(labels.variant_key) != set(frozen_train.variant_key)
            or labels.variant_key.duplicated().any()):
        raise ValueError("Audited assay values are not exactly VTI TRAIN")
    observed_positions = labels.set_index("variant_key").protein_position
    if not np.array_equal(frozen_train.protein_position.to_numpy(dtype=int),
                          frozen_train.variant_key.map(observed_positions).to_numpy(dtype=int)):
        raise ValueError("Maturation labels do not match frozen protein positions")
    train = frozen_train.merge(labels[label_cols].drop(columns="protein_position"),
                               on="variant_key", validate="one_to_one").sort_values("variant_key").reset_index(drop=True)
    if not np.isfinite(train[["vti_rescue_pp", "band_c_gain_pct_wt"]].to_numpy(float)).all():
        raise ValueError("Missing Band C gain or rescue training labels")

    seq = bihler_2024_assay_sequence()
    parsed = [parse_substitution(row, seq) for _, row in train.iterrows()]
    if any(p[-1] != "valid" or p[0] != int(pos) for p, pos in zip(parsed, train.protein_position)):
        raise ValueError("VTI missense variants do not match assay CFTR reference")
    train["reference_aa"] = [p[1] for p in parsed]
    train["alternate_aa"] = [p[2] for p in parsed]
    X, feature_names = make_features(train, seq)
    positions = train.protein_position.to_numpy(dtype=int)
    row_index = {key: i for i, key in enumerate(train.variant_key)}

    ref = pd.read_csv(RESULTS / "train_cv_predictions.csv")
    ref = ref.loc[ref.model.eq("vti_forest")].copy()
    _validate_reference(ref, train)
    ref["study_cohort"] = ref.variant_key.map(train.set_index("variant_key").study_cohort)
    print("\n[1] PROTOCOL")
    print(f"VTI training: {len(train)} variants / {len(np.unique(positions))} positions")
    print("2 repeats x 5 frozen position-disjoint folds from existing Forest predictions")
    print("Rescue + Band C gain: equally weighted after fold-training standardization")
    print("Negative control: fold-training Band C gain shuffled, same Forest")
    print("Predictor inputs: 68 mutation/sequence features ONLY")
    print("No ETI labels, VTI val/test labels or held-fold Band C measurements used for fitting")

    predictions = []
    print("\n[2] POSITION-GROUPED OUT-OF-FOLD PREDICTIONS")
    for (repeat, fold), held in ref.groupby(["repeat", "fold"], sort=True):
        held_keys = held.variant_key.tolist()
        held_idx = np.array([row_index[k] for k in held_keys], dtype=int)
        held_positions = set(map(int, held.protein_position))
        train_idx = np.flatnonzero(~np.isin(positions, sorted(held_positions)))
        if set(positions[train_idx]) & held_positions:
            raise AssertionError("Training includes held-out protein positions")
        fit_seed = SEED + (int(repeat)-1)*10 + int(fold)
        y_r = train.iloc[train_idx].vti_rescue_pp.to_numpy(float)
        y_c = train.iloc[train_idx].band_c_gain_pct_wt.to_numpy(float)
        predictions_by_model = {}
        for model, shuffle in (("rescue_plus_band_c", False), ("rescue_plus_shuffled_band_c", True)):
            predictions_by_model[model] = paired_multitask_predictions(
                X[train_idx], y_r, y_c, X[held_idx], seed=fit_seed, shuffled_aux=shuffle)
        print(f"  Repeat {repeat}/2 | fold {fold}/5 | fitted={len(train_idx)} held={len(held_idx)}", flush=True)
        train_bc_median = float(np.median(y_c))
        for k, i in enumerate(held_idx):
            row = train.iloc[i]
            for name, predicted in predictions_by_model.items():
                predictions.append({
                    "repeat": int(repeat), "fold": int(fold), "model": name,
                    "variant_key": str(row.variant_key),
                    "protein_position": int(row.protein_position),
                    "study_cohort": str(row.study_cohort),
                    "observed_pp": float(row.vti_rescue_pp),
                    "predicted_pp": float(predicted[k, 0]),
                    "observed_band_c_gain": float(row.band_c_gain_pct_wt),
                    "predicted_band_c_gain": float(predicted[k, 1]) if name == "rescue_plus_band_c" else np.nan,
                    "band_c_fold_train_median": train_bc_median,
                })

    all_oof = pd.concat([
        ref[["repeat", "fold", "model", "variant_key", "protein_position", "study_cohort", "observed_pp", "predicted_pp"]],
        pd.DataFrame(predictions),
    ], ignore_index=True)

    real_band = all_oof.loc[
        all_oof.model.eq("rescue_plus_band_c"),
        "predicted_band_c_gain"
    ]
    if len(real_band) != N_REPEATS * len(train) or not np.isfinite(
        real_band.to_numpy(dtype=float)
    ).all():
        raise ValueError("Missing/non-finite genuine Band C OOF predictions")

    summary, averaged = evaluate_position_predictions(all_oof)
    print("\n[3] RESCUE POSITION MAE AND COHORT BIAS")
    print(f"{'Cohort':10s} {'Model':33s} {'Pos-MAE':>9s} {'Bias':>9s} {'Gain vs Forest':>15s} {'95% bootstrap CI':>24s}")
    for r in summary.itertuples(index=False):
        low, high = r.gain_vs_forest_ci95
        print(f"{r.cohort:10s} {r.model:33s} {r.position_mae_pp:9.3f} {r.mean_bias_pp:+9.3f} "
              f"{r.gain_vs_forest_pp:+15.3f} [{low:+9.3f}, {high:+9.3f}]")

    print("\n[4] BIOLOGICAL AUXILIARY vs SHUFFLED NEGATIVE CONTROL")
    for cohort in ("all", "ETI, VTI", "VTI"):
        r = summary.loc[(summary.cohort.eq(cohort)) & summary.model.eq("rescue_plus_band_c")].iloc[0]
        lo, hi = r.gain_vs_shuffled_aux_ci95
        print(f"  {cohort:10s}: gain={r.gain_vs_shuffled_aux_pp:+.3f} pp CI [{lo:+.3f}, {hi:+.3f}]")

    band = all_oof.loc[all_oof.model.eq("rescue_plus_band_c"), [
        "variant_key", "protein_position", "observed_band_c_gain", "predicted_band_c_gain", "band_c_fold_train_median"
    ]].groupby(["variant_key", "protein_position"], as_index=False).mean(numeric_only=True)
    band["error"] = (band.predicted_band_c_gain - band.observed_band_c_gain).abs()
    band["median_error"] = (band.band_c_fold_train_median - band.observed_band_c_gain).abs()
    per_pos = band.groupby("protein_position")[[
        "error", "median_error", "predicted_band_c_gain", "observed_band_c_gain"
    ]].mean(numeric_only=True)
    band_mae = float(per_pos.error.mean())
    rho = float(spearmanr(per_pos.predicted_band_c_gain, per_pos.observed_band_c_gain).statistic)
    baseline = float(per_pos.median_error.mean())
    print("\n[5] CAN SEQUENCE PREDICT BAND C MATURATION GAIN?")
    print(f"  Equal-position OOF MAE: {band_mae:.3f} pp WT")
    print(f"  Fold-training-median reference MAE: {baseline:.3f} pp WT")
    print(f"  Equal-position OOF Spearman rho: {rho:+.3f}")
    print("  Note: this auxiliary head is jointly supervised by rescue; not a Band C-only predictor")

    outdir = ROOT / "results" / "biology"
    outdir.mkdir(parents=True, exist_ok=True)
    all_oof.to_csv(outdir / "phase14c_band_c_aux_oof.csv", index=False)
    summary.to_csv(outdir / "phase14c_band_c_aux_summary.csv", index=False)
    report = {
        "experiment": "train-only, same frozen grouped CV and original VTI Forest comparator",
        "vti_source_sha256": sha256(VTI_PAIRS),
        "frozen_partition_sha256": sha256(SPLIT_OUT),
        "reference_oof_sha256": hashlib.sha256((RESULTS / "train_cv_predictions.csv").read_bytes()).hexdigest(),
        "n_variants": len(train), "n_positions": int(len(np.unique(positions))),
        "models": list(MODELS), "n_repeats": N_REPEATS, "n_folds": N_FOLDS,
        "auxiliary": "post-treatment Band C %WT minus DMSO Band C %WT",
        "standardization": "per-task fold-training z-score, equal impurity weight",
        "features": feature_names,
        "band_c_auxiliary_oof": {"position_mae_pp": band_mae, "position_spearman": rho,
                                  "constant_median_mae_pp": baseline},
        "results": summary.to_dict(orient="records"),
        "limitations": [
            "Band C is a POST-treatment biological training label, never an inference feature",
            "Correlational association does not guarantee predictive transfer from sequence",
            "Exploratory training-CV ablation after previous model inspection",
            "Multioutput Forest changes joint partitioning relative to single-output baseline",
            "No validation or final test outcome accessed",
        ],
    }
    (outdir / "phase14c_band_c_aux_summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("\nSaved: results/biology/phase14c_band_c_aux_oof.csv")
    print("Saved: results/biology/phase14c_band_c_aux_summary.csv")
    print("Saved: results/biology/phase14c_band_c_aux_summary.json")
    print("VTI val/test and ETI outcomes NOT ACCESSED")
    print("=" * 92)


if __name__ == "__main__":
    run()
