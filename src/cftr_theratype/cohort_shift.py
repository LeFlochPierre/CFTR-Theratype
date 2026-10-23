"""Training-only VTI cohort shift characterisation.

Research question: can observable mutation features or baseline CFTR activity
account for different error patterns between Bihler 2026 study cohorts?

NEVER uses VTI val/test or any ETI outcomes. Never uses study cohort as an
input feature for response prediction. This is descriptive, not causal.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cftr_theratype.benchmark import make_features
from cftr_theratype.reference import bihler_2024_assay_sequence
from cftr_theratype.splits import parse_substitution
from cftr_theratype.transfer import (
    MANIFEST, RESULTS, SPLIT_OUT, VTI_PAIRS, ROOT, sha256,
)

COHORTS = ("ETI, VTI", "VTI")
BASELINE_EDGES = [-np.inf, 5.0, 20.0, 50.0, np.inf]
BASELINE_LABELS = ["<5% WT", "5–<20% WT", "20–<50% WT", "≥50% WT"]


def baseline_bins(values: pd.Series) -> pd.Categorical:
    """Pre-specified, interpretable bins; descriptive, not model inputs."""
    return pd.cut(values, bins=BASELINE_EDGES, labels=BASELINE_LABELS, right=False)


def common_support_difference(frame: pd.DataFrame, value: str,
                              minimum_positions: int = 5) -> dict:
    """Difference between cohorts standardised over shared baseline strata.

    Uses weights proportional to min(number of positions in each cohort, by
    baseline bin); returns descriptive differences only, not causal effects.
    """
    terms = []
    for stratum, sub in frame.groupby("baseline_stratum", observed=True):
        aa = sub.loc[sub.study_cohort.eq(COHORTS[0])]
        bb = sub.loc[sub.study_cohort.eq(COHORTS[1])]
        na = aa.protein_position.nunique()
        nb = bb.protein_position.nunique()
        if na < minimum_positions or nb < minimum_positions:
            continue
        ma = float(aa[value].mean())
        mb = float(bb[value].mean())
        terms.append({
            "stratum": str(stratum), "eti_characterised_positions": int(na),
            "vti_only_positions": int(nb), "mean_difference_pp": ma - mb,
            "weight": int(min(na, nb)),
        })
    if not terms:
        return {"status": "insufficient_common_support", "strata": []}
    d = sum(x["weight"] * x["mean_difference_pp"] for x in terms)
    n = sum(x["weight"] for x in terms)
    return {
        "status": "descriptive_only", "weighted_difference_pp": float(d / n),
        "strata": terms,
        "limitations": "Baseline activity was measured experimentally. Coarse-stratum "
                       "adjustment does not remove response-based cohort selection "
                       "or study effects and is not causal.",
    }


def train_only_cohort_classifier(features: np.ndarray,
                                 cohort: pd.Series,
                                 positions: pd.Series) -> dict:
    """One fixed logistic model, out-of-fold, split by protein position."""
    y = cohort.eq(COHORTS[0]).to_numpy(dtype=int)
    groups = positions.to_numpy(dtype=int)
    if len(np.unique(groups)) < 10 or len(np.unique(y)) != 2:
        raise ValueError("Need both cohorts and at least 10 unique positions")
    probabilities = np.full(len(y), np.nan, dtype=float)
    null_probabilities = np.full(len(y), np.nan, dtype=float)
    fold_rows = []
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20261023)
    for i, (fit, held) in enumerate(folds.split(features, y, groups), 1):
        if not set(groups[fit]).isdisjoint(set(groups[held])):
            raise ValueError("Protein-position leakage in cohort classification")
        if len(np.unique(y[fit])) != 2:
            raise ValueError("A training fold contains only one cohort")
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000),
        )
        estimator.fit(features[fit], y[fit])
        probabilities[held] = estimator.predict_proba(features[held])[:, 1]
        null_probabilities[held] = float(y[fit].mean())
        fold_rows.append({"fold": i, "fit_variants": int(len(fit)),
                          "held_variants": int(len(held)),
                          "held_positions": int(len(np.unique(groups[held])))})
    if not np.isfinite(probabilities).all():
        raise ValueError("Incomplete out-of-fold predictions")
    brier = float(brier_score_loss(y, probabilities))
    null_brier = float(brier_score_loss(y, null_probabilities))
    return {
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "balanced_accuracy_at_half": float(balanced_accuracy_score(
            y, (probabilities >= .5).astype(int))),
        "brier_score": brier,
        "prevalence_only_brier": null_brier,
        "brier_skill_vs_prevalence": float(1 - brier / null_brier),
        "classifier_class_weight": None,
        "positive_class": COHORTS[0], "cv": "5-fold StratifiedGroupKFold by protein position",
        "folds": fold_rows,
        "warning": "Exploratory separability of a response-selected study cohort, not "
                   "a model for predicting clinical responses.",
    }


def _load_train_only_labels(train_keys: set[str]) -> pd.DataFrame:
    """Request only frozen VTI train identifiers from Parquet storage."""
    cols = ["variant_key", "baseline_fsk_pct_wt", "iva_treated_pct_wt",
            "vti_rescue_pp", "vti_increment_over_iva_pp"]
    frame = pd.read_parquet(VTI_PAIRS, columns=cols,
                            filters=[("variant_key", "in", sorted(train_keys))])
    if frame.variant_key.duplicated().any() or set(frame.variant_key) != train_keys:
        raise ValueError("Unexpected VTI training label key coverage")
    if frame[cols[1:]].isna().any().any():
        raise ValueError("Missing VTI train phenotypes")
    return frame


def _load_train_only_oof(keys: set[str]) -> pd.DataFrame:
    path = RESULTS / "train_cv_predictions.csv"
    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary.get("vti_source_sha256") != sha256(VTI_PAIRS)
            or summary.get("vti_partition_sha256") != sha256(SPLIT_OUT)):
        raise ValueError("OOF predictions were generated from another dataset/split")
    raw = pd.read_csv(path)
    required = {"variant_key", "model", "repeat", "protein_position",
                "observed_pp", "predicted_pp"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing OOF columns: {sorted(required-set(raw.columns))}")
    if set(raw.variant_key) != keys:
        raise ValueError("OOF predictions not restricted to original VTI train keys")
    subset = raw.loc[raw.model.eq("vti_forest")].copy()
    if subset.duplicated(["variant_key", "repeat"]).any():
        raise ValueError("Repeated forest OOF predictions")
    if subset.groupby("variant_key").repeat.nunique().ne(2).any():
        raise ValueError("Expected two CV predictions for each train variant")
    # Average *predictions*, then compute signed errors once per variant.
    return (subset.groupby("variant_key", as_index=False)
            .agg(predicted_pp=("predicted_pp", "mean"),
                 observed_pp=("observed_pp", "mean")))


def run_cohort_shift() -> None:
    print("=" * 82)
    print("CFTR-THERATYPE | TRAIN-ONLY COHORT SHIFT CHARACTERISATION")
    print("=" * 82)
    protocol = json.loads(MANIFEST.read_text())
    if protocol["vti_source_sha256"] != sha256(VTI_PAIRS):
        raise ValueError("VTI source differs from frozen transfer protocol")
    if protocol["membership_sha256"] is None:
        raise ValueError("Incomplete frozen transfer manifest")
    frozen = pd.read_parquet(
        SPLIT_OUT, columns=["variant_key", "protein_position", "transfer_split",
                                 "study_cohort", "variant_type", "variant", "hgvs_protein"])
    if frozen.variant_key.duplicated().any():
        raise ValueError("Frozen VTI metadata contains duplicate keys")
    train = frozen.loc[frozen.transfer_split.eq("train")].copy()
    train_keys = set(train.variant_key.astype(str))
    if len(train) != 210 or set(train.study_cohort) != set(COHORTS):
        raise ValueError("Unexpected frozen training cohort; review protocol")

    labels = _load_train_only_labels(train_keys)
    train = train.merge(labels, on="variant_key", validate="one_to_one")
    seq = bihler_2024_assay_sequence()
    parsed = [parse_substitution(row, seq) for _, row in train.iterrows()]
    if any(x[-1] != "valid" for x in parsed):
        raise ValueError("Training variants not validated against assay sequence")
    train["reference_aa"] = [x[1] for x in parsed]
    train["alternate_aa"] = [x[2] for x in parsed]
    X, names = make_features(train, seq)
    train["baseline_stratum"] = baseline_bins(train.baseline_fsk_pct_wt)

    print("\n[1] DESIGN / DATA ISOLATION")
    print(f"VTI training: {len(train)} variants at {train.protein_position.nunique()} positions")
    print("VTI validation/test and ETI outcomes: NOT READ")
    print("Reported study cohort is NOT used as a functional-response feature")

    print("\n[2] COHORT PHENOTYPES (DESCRIPTIVE ONLY)")
    for cohort in COHORTS:
        sub = train.loc[train.study_cohort.eq(cohort)]
        print(f"{cohort}: {len(sub)} variants / {sub.protein_position.nunique()} positions")
        for col in ("baseline_fsk_pct_wt", "iva_treated_pct_wt",
                    "vti_rescue_pp", "vti_increment_over_iva_pp"):
            print(f"  {col:28s} median={sub[col].median():8.3f} "
                  f"IQR=[{sub[col].quantile(.25):7.3f}, {sub[col].quantile(.75):7.3f}]")

    print("\n[3] SEQUENCE-ONLY COHORT SEPARABILITY")
    classifier = train_only_cohort_classifier(
        X, train.study_cohort, train.protein_position)
    print(f"Grouped OOF AUC:  {classifier['roc_auc']:.3f}")
    print(f"Balanced accuracy: {classifier['balanced_accuracy_at_half']:.3f}")
    print(f"Brier score: {classifier['brier_score']:.3f} "
          f"(fold-prevalence-only: {classifier['prevalence_only_brier']:.3f}; "
          f"skill={classifier['brier_skill_vs_prevalence']:+.3f})")
    print("Sequence-based discrimination does not establish the cause of selection")

    print("\n[4] COHORT SHIFT WITHIN BASELINE-FUNCTION STRATA")
    oof = _load_train_only_oof(train_keys)
    merged = train.merge(oof, on="variant_key", validate="one_to_one")
    if not np.allclose(merged.observed_pp, merged.vti_rescue_pp, atol=1e-6):
        raise ValueError("OOF observed targets differ from frozen VTI training labels")
    merged["forest_error_pp"] = merged.predicted_pp - merged.vti_rescue_pp
    summaries = []
    for stratum in BASELINE_LABELS:
        print(f"Baseline {stratum}:")
        for cohort in COHORTS:
            sub = merged.loc[merged.baseline_stratum.eq(stratum) &
                             merged.study_cohort.eq(cohort)]
            count = sub.protein_position.nunique()
            if len(sub):
                resp = float(sub.vti_rescue_pp.mean())
                bias = float(sub.forest_error_pp.mean())
                print(f"  {cohort:8s} n={len(sub):3d}, positions={count:3d}, "
                      f"mean rescue={resp:+7.2f} pp, forest bias={bias:+7.2f} pp")
            else:
                resp, bias = None, None
                print(f"  {cohort:8s} no observations")
            summaries.append({"baseline_stratum": stratum, "study_cohort": cohort,
                              "variants": int(len(sub)), "positions": int(count),
                              "mean_rescue_pp": resp, "forest_bias_pp": bias})

    conditional_response = common_support_difference(merged, "vti_rescue_pp")
    conditional_bias = common_support_difference(merged, "forest_error_pp")
    print("\n[5] COARSE BASELINE STANDARDISATION (EXPLORATORY)")
    for label, result in (("VTI rescue", conditional_response),
                          ("Forest bias", conditional_bias)):
        if result["status"] == "insufficient_common_support":
            print(f"{label}: insufficient shared baseline support")
        else:
            print(f"{label}: ETI-characterised minus VTI-only "
                  f"= {result['weighted_difference_pp']:+.3f} pp "
                  f"across {len(result['strata'])} supported strata")

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(RESULTS / "cohort_shift_baseline_strata.csv", index=False)
    report = {
        "protocol": "training-only descriptive cohort-shift analysis",
        "vti_source_sha256": sha256(VTI_PAIRS),
        "transfer_split_sha256": sha256(SPLIT_OUT),
        "oof_source": "results/transfer/train_cv_predictions.csv",
        "cohort_feature_roc": classifier,
        "conditional_response": conditional_response,
        "conditional_bias": conditional_bias,
        "vti_validation_test_outcomes_accessed": False,
        "eti_outcomes_accessed": False,
        "caveat": "Baseline FSK activity was measured experimentally and is used for "
                  "exploratory characterisation only. Cohort membership is partly "
                  "defined by previous ETI response. Differences are not causal.",
    }
    (RESULTS / "cohort_shift_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\n[6] SAVED")
    print("results/transfer/cohort_shift_baseline_strata.csv")
    print("results/transfer/cohort_shift_summary.json")
    print("Frozen splits and test outcomes were not changed or accessed")
    print("=" * 82)




def fold_diagnostic_controls(train: pd.DataFrame, forest_oof: pd.DataFrame) -> pd.DataFrame:
    """Evaluate prespecified *retrospective* control predictors on the frozen OOF folds.

    `study_cohort` and untreated functional measurements are deliberately
    diagnostic-only features. They are not available for most untested variants,
    and the study-cohort assignment is influenced by earlier ETI response.

    Every fit excludes *all* variants at held-out protein positions.
    """
    from sklearn.linear_model import Ridge

    required_train = {"variant_key", "protein_position", "study_cohort",
                      "baseline_fsk_pct_wt", "iva_treated_pct_wt", "vti_rescue_pp"}
    required_oof = {"variant_key", "protein_position", "repeat", "fold",
                    "model", "observed_pp", "predicted_pp"}
    if not required_train.issubset(train.columns) or not required_oof.issubset(forest_oof.columns):
        raise ValueError("Missing required frozen train/OOF columns")
    if train.variant_key.isna().any() or train.variant_key.duplicated().any():
        raise ValueError("Training keys must be unique")
    train = train.copy()
    train["variant_key"] = train.variant_key.astype(str)
    train = train.set_index("variant_key", drop=False)
    if not set(train.study_cohort).issubset(set(COHORTS)):
        raise ValueError("Unrecognised study cohort")
    if train[["baseline_fsk_pct_wt", "iva_treated_pct_wt", "vti_rescue_pp"]].isna().any().any():
        raise ValueError("Missing train-only assay values")

    oof = forest_oof.loc[forest_oof.model.isin(["median", "vti_forest"])].copy()
    oof["variant_key"] = oof.variant_key.astype(str)
    if set(oof.model) != {"median", "vti_forest"}:
        raise ValueError("Expected stored median and VTI Forest OOF records")
    if set(oof.variant_key) != set(train.index):
        raise ValueError("OOF keys do not exactly match frozen VTI training keys")
    if oof.duplicated(["repeat", "fold", "model", "variant_key"]).any():
        raise ValueError("Duplicated out-of-fold records")
    for model, group in oof.groupby("model"):
        if group.groupby(["repeat", "variant_key"]).size().ne(1).any():
            raise ValueError(f"Incomplete or duplicated repeat coverage for {model}")
        if set(group.repeat.unique()) != set(oof.repeat.unique()):
            raise ValueError("Models have incompatible repeat coverage")
        if not group.groupby("variant_key").repeat.nunique().eq(oof.repeat.nunique()).all():
            raise ValueError("Missing per-repeat predictions")
    target_lookup = train.vti_rescue_pp.astype(float)
    pos_lookup = train.protein_position.astype(int)
    raw_expected = oof.variant_key.map(target_lookup).to_numpy(dtype=float)
    if not np.allclose(raw_expected, oof.observed_pp.to_numpy(dtype=float), atol=1e-5, rtol=0):
        raise ValueError("Stored OOF targets disagree with train-only VTI labels")
    if not np.array_equal(oof.variant_key.map(pos_lookup).to_numpy(dtype=int),
                          oof.protein_position.to_numpy(dtype=int)):
        raise ValueError("Stored OOF positions disagree with frozen membership")

    controls = []
    folds = oof[oof.model.eq("vti_forest")].groupby(["repeat", "fold"], sort=True)
    for (repeat, fold), held in folds:
        held_keys = held.variant_key.astype(str).tolist()
        held_positions = set(train.loc[held_keys, "protein_position"].astype(int))
        fit = train.loc[~train.protein_position.isin(held_positions)]
        query = train.loc[held_keys]
        if len(fit) == 0 or len(query) == 0:
            raise ValueError("Empty grouped fold")
        if not set(fit.protein_position).isdisjoint(held_positions):
            raise ValueError("Position leakage in diagnostic controls")
        grouped_medians = fit.groupby("study_cohort").vti_rescue_pp.median()
        if not set(query.study_cohort).issubset(set(grouped_medians.index)):
            raise ValueError("Missing a cohort in the training fold")
        cohort_oracle = query.study_cohort.map(grouped_medians).to_numpy(dtype=float)

        guesses = {"cohort_median_oracle": cohort_oracle}
        for name, columns in (
            ("measured_fsk_ridge_oracle", ["baseline_fsk_pct_wt"]),
            ("measured_fsk_iva_ridge_oracle", ["baseline_fsk_pct_wt", "iva_treated_pct_wt"]),
        ):
            est = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            est.fit(fit[columns].to_numpy(float), fit.vti_rescue_pp.to_numpy(float))
            guesses[name] = est.predict(query[columns].to_numpy(float))

        for name, preds in guesses.items():
            for key, prediction in zip(held_keys, preds):
                controls.append({"repeat": int(repeat), "fold": int(fold),
                                 "model": name, "variant_key": key,
                                 "protein_position": int(pos_lookup[key]),
                                 "observed_pp": float(target_lookup[key]),
                                 "predicted_pp": float(prediction)})
    out = pd.concat([oof, pd.DataFrame(controls)], ignore_index=True)
    if out.duplicated(["repeat", "model", "variant_key"]).any():
        raise ValueError("Unexpected duplicate diagnostics")
    return out


def _position_metrics(frame: pd.DataFrame, draws: int = 3000, seed: int = 20261023):
    """Aggregate OOF predictions by variant first, then error by position."""
    counts = frame.groupby(["model", "variant_key"]).repeat.nunique()
    expected = frame.repeat.nunique()
    if not counts.eq(expected).all():
        raise ValueError("Models have inconsistent OOF repeat coverage")
    per_variant = frame.groupby(["model", "variant_key", "protein_position"], as_index=False).agg(
        predicted_pp=("predicted_pp", "mean"), observed_pp=("observed_pp", "mean")
    )
    per_variant["absolute_error_pp"] = np.abs(
        per_variant.predicted_pp - per_variant.observed_pp)
    per_variant["signed_error_pp"] = per_variant.predicted_pp - per_variant.observed_pp
    errors = per_variant.groupby(["protein_position", "model"]).absolute_error_pp.mean().unstack()
    if errors.isna().any().any():
        raise ValueError("Incomplete position/error matrix")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(errors), size=(draws, len(errors)))
    baseline = errors["vti_forest"].to_numpy(float)
    metrics = []
    for model in errors.columns:
        vec = errors[model].to_numpy(float)
        gain = baseline - vec
        boot = gain[indices].mean(axis=1)
        lo, hi = np.quantile(boot, [0.025, 0.975])
        metrics.append({"model": str(model), "position_mae_pp": float(vec.mean()),
                        "gain_vs_vti_forest_pp": float(gain.mean()),
                        "gain_ci_95_pp": [float(lo), float(hi)]})
    return sorted(metrics, key=lambda x: x["position_mae_pp"]), per_variant


def run_cohort_shift_controls() -> None:
    """Retrospective *diagnostic ceilings*, never deployable sequence predictors."""
    print("=" * 82)
    print("CFTR-THERATYPE | TRAIN-ONLY COHORT SELECTION CONTROLS")
    print("=" * 82)
    protocol = json.loads(MANIFEST.read_text())
    if protocol.get("vti_source_sha256") != sha256(VTI_PAIRS):
        raise ValueError("VTI source differs from the frozen transfer protocol")
    if protocol.get("membership_sha256") is None:
        raise ValueError("Missing frozen membership checksum")
    frozen = pd.read_parquet(
        SPLIT_OUT,
        columns=["variant_key", "protein_position", "transfer_split", "study_cohort"],
    )
    if frozen.variant_key.isna().any() or frozen.variant_key.duplicated().any():
        raise ValueError("Invalid frozen cohort identifiers")
    train = frozen.loc[frozen.transfer_split.eq("train")].copy()
    train_keys = set(train.variant_key.astype(str))
    labels = _load_train_only_labels(train_keys)
    train = train.merge(labels, on="variant_key", validate="one_to_one")
    if len(train) != 210 or train.protein_position.nunique() != 168:
        raise ValueError("Unexpected training population: check split provenance")

    result_file = RESULTS / "train_cv_predictions.csv"
    result_summary = json.loads((RESULTS / "summary.json").read_text())
    if (result_summary.get("vti_source_sha256") != sha256(VTI_PAIRS)
            or result_summary.get("vti_partition_sha256") != sha256(SPLIT_OUT)):
        raise ValueError("Out-of-fold predictions were generated from different inputs")
    stored = pd.read_csv(result_file)
    all_predictions = fold_diagnostic_controls(train, stored)
    metrics, per_variant = _position_metrics(all_predictions)
    cohort_map = train.set_index("variant_key").study_cohort
    per_variant["study_cohort"] = per_variant.variant_key.map(cohort_map)
    if per_variant.study_cohort.isna().any():
        raise ValueError("Missing cohort annotation")

    print("\n[1] SHARED POSITION-DISJOINT CV FOLDS")
    print(f"Training: {len(train)} variants / {train.protein_position.nunique()} positions")
    print(f"Reuses {stored.repeat.nunique()} repeats of stored grouped OOF partitions")
    print("2024/2026 validation and test outcomes: NOT READ")
    print("\n[2] RETROSPECTIVE CONTROL COMPARISON")
    print(f"{'Control':34s} {'Pos-MAE':>10s} {'Gain vs Forest':>16s} {'95% CI (gain)':>24s}")
    for item in metrics:
        low, high = item["gain_ci_95_pp"]
        print(f"{item['model']:34s} {item['position_mae_pp']:10.3f} "
              f"{item['gain_vs_vti_forest_pp']:+16.3f} [{low:+8.3f}, {high:+8.3f}]")
    print("\n[3] CAUTION")
    print("Cohort-median uses retrospective, response-selected study membership.")
    print("Measured-baseline models require functional assay data unavailable for untested variants.")
    print("Baseline also enters the mathematical definition of the rescue label.")
    print("These are DIAGNOSTIC ORACLES, not prospective or clinically deployable models.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "cohort_shift_controls.json").write_text(json.dumps({
        "experiment": "train-only retrospective cohort/phenotype controls",
        "vti_source_sha256": sha256(VTI_PAIRS),
        "vti_split_sha256": sha256(SPLIT_OUT),
        "oof_predictions_sha256": hashlib.sha256(result_file.read_bytes()).hexdigest(),
        "cohort_annotation_is_post_selection": True,
        "measured_baseline_is_not_sequence_only": True,
        "test_outcomes_accessed": False,
        "results": metrics,
    }, indent=2) + "\n")
    per_variant.to_csv(RESULTS / "cohort_shift_controls_oof.csv", index=False)
    print("\nSaved: results/transfer/cohort_shift_controls.json")
    print("Saved: results/transfer/cohort_shift_controls_oof.csv")
    print("=" * 82)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controls", action="store_true",
                        help="Run retrospective train-only diagnostic controls")
    args = parser.parse_args()
    if args.controls:
        run_cohort_shift_controls()
    else:
        run_cohort_shift()
