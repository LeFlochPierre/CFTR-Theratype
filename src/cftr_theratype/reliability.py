"""Train-only error-shift and risk-coverage diagnostics for CFTR VTI.

Uses existing OOF CSVs, never original assay data or held-out outcomes.
Model disagreement is an exploratory *risk score*, not calibrated uncertainty.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cftr_theratype.transfer import (
    RESULTS, SPLIT_OUT, MANIFEST, VTI_PAIRS, sha256, SEED,
)

N_BOOT = 3000
MODELS_REQUIRED = {"median", "vti_forest", "pooled_forest"}


def cluster_mean_ci(values: np.ndarray, seed: int = SEED) -> tuple[float, float, float]:
    """Bootstrap *protein positions*, which are the independent CV groups."""
    values = np.asarray(values, dtype=float)
    if len(values) < 3 or not np.isfinite(values).all():
        raise ValueError("Need >=3 finite, independent position values")
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    ci = np.quantile(values[draw].mean(axis=1), [0.025, 0.975])
    return float(values.mean()), float(ci[0]), float(ci[1])


def normalise_oof(raw: pd.DataFrame, outcome: str, cohort: pd.DataFrame) -> pd.DataFrame:
    """Allow only complete, repeated OOF predictions of original VTI train rows."""
    ycol, pcol = {
        "total_rescue": ("observed_pp", "predicted_pp"),
        "corrector_increment": ("observed_corrector_pp", "predicted_corrector_pp"),
    }[outcome]
    required = {"repeat", "fold", "model", "variant_key", "protein_position", ycol, pcol}
    if not required.issubset(raw):
        raise ValueError(f"Missing {outcome} columns: {sorted(required - set(raw.columns))}")
    if raw[list(required)].isna().any().any():
        raise ValueError("Incomplete OOF prediction records")

    eligible = cohort.loc[cohort.transfer_split.eq("train"),
                          ["variant_key", "protein_position", "study_cohort"]].copy()
    if eligible.variant_key.isna().any() or eligible.variant_key.duplicated().any():
        raise ValueError("Duplicate/null training identifiers")
    keys = set(eligible.variant_key.astype(str))
    if set(raw.variant_key.astype(str)) != keys:
        raise ValueError("OOF keys are not exactly the frozen VTI training keys")
    if not MODELS_REQUIRED.issubset(set(raw.model.astype(str))):
        raise ValueError("Missing a required OOF model")
    if raw.duplicated(["repeat", "model", "variant_key"]).any():
        raise ValueError("Duplicate OOF model/variant/repeat")
    repeats = sorted(raw["repeat"].unique().tolist())
    if repeats != [1, 2]:
        raise ValueError("Expected exactly two completed CV repeats")
    model_counts = raw.groupby(["model", "variant_key"]).size()
    if not model_counts.eq(2).all():
        raise ValueError("Each model must predict each variant exactly twice")
    expected = set(raw.model.astype(str))
    if len(raw) != len(eligible) * len(expected) * 2:
        raise ValueError("Incomplete or duplicated OOF coverage")
    # Corrector OOF records already contain study_cohort.
    # Verify consistency with frozen membership before merging.
    if "study_cohort" in raw.columns:
        cohort_check = raw[["variant_key", "study_cohort"]].merge(
            eligible[["variant_key", "study_cohort"]],
            on="variant_key",
            validate="many_to_one",
            suffixes=("_oof", "_frozen"),
        )
        matches = cohort_check["study_cohort_oof"].eq(
            cohort_check["study_cohort_frozen"]
        ).fillna(False)

        if not matches.all():
            raise ValueError(
                "OOF study_cohort disagrees with frozen membership"
            )

        raw = raw.drop(columns=["study_cohort"])

    joined = raw.merge(eligible, on="variant_key", validate="many_to_one",
                       suffixes=("_prediction", "_reference"))
    if not joined.protein_position_prediction.eq(joined.protein_position_reference).all():
        raise ValueError("OOF protein-position mismatch")
    joined["protein_position"] = joined["protein_position_reference"].astype(int)
    joined["observed"] = pd.to_numeric(joined[ycol], errors="raise")
    joined["predicted"] = pd.to_numeric(joined[pcol], errors="raise")
    if not np.isfinite(joined[["observed", "predicted"]].to_numpy()).all():
        raise ValueError("Nonfinite observed/predicted values")
    joined["signed_error"] = joined.predicted - joined.observed
    joined["absolute_error"] = joined.signed_error.abs()
    return joined[["repeat", "fold", "model", "variant_key", "protein_position",
                   "study_cohort", "observed", "predicted", "signed_error", "absolute_error"]]


def compute_reliability(oof: pd.DataFrame) -> tuple[list[dict], pd.DataFrame, dict]:
    """Position-balanced cohort errors and exploratory disagreement-risk ranking."""
    groups = oof.groupby(["study_cohort", "protein_position", "model"], as_index=False)
    per_group = groups[["signed_error", "absolute_error"]].mean()
    summary = []
    for (cohort, model), sub in per_group.groupby(["study_cohort", "model"], sort=True):
        mean_abs, abs_lo, abs_hi = cluster_mean_ci(sub.absolute_error.to_numpy())
        bias, bias_lo, bias_hi = cluster_mean_ci(sub.signed_error.to_numpy())
        summary.append({"cohort": str(cohort), "model": str(model),
                        "positions": int(len(sub)), "position_mae_pp": mean_abs,
                        "position_mae_95ci": [abs_lo, abs_hi],
                        "mean_bias_pp": bias, "mean_bias_95ci": [bias_lo, bias_hi]})

    # Paired comparisons, bootstrapped within each cohort's positions.
    gains = []
    for cohort, sub in per_group.groupby("study_cohort"):
        pivot = sub.pivot(index="protein_position", columns="model", values="absolute_error")
        for reference, candidate in [("median", "vti_forest"), ("vti_forest", "pooled_forest")]:
            improvement = (pivot[reference] - pivot[candidate]).dropna()
            mean, lo, hi = cluster_mean_ci(improvement.to_numpy())
            gains.append({"cohort": str(cohort), "reference": reference,
                          "candidate": candidate, "position_count": int(len(improvement)),
                          "gain_pp": mean, "gain_95ci": [lo, hi]})

    # Repeated out-of-fold prediction differences at each position, without
    # ever using its response to compute the disagreement score.
    paired = oof.loc[oof.model.isin(["vti_forest", "pooled_forest"])].pivot(
        index=["repeat", "variant_key", "protein_position", "study_cohort"],
        columns="model", values="predicted")
    if paired.isna().any().any():
        raise ValueError("Missing paired model predictions")
    paired["disagreement"] = (paired["vti_forest"] - paired["pooled_forest"]).abs()
    disagreement = paired.reset_index().groupby("protein_position").disagreement.mean()
    forest = oof.loc[oof.model.eq("vti_forest")].groupby("protein_position").absolute_error.mean()
    risk = pd.DataFrame({"disagreement_pp": disagreement, "forest_mae_pp": forest}).dropna()
    if len(risk) < 10:
        raise ValueError("Insufficient positions for risk ranking")
    rho = float(spearmanr(risk.disagreement_pp, risk.forest_mae_pp).statistic)
    if not np.isfinite(rho):
        rho = None

    risk = risk.sort_values("disagreement_pp", kind="stable").copy()
    # Ties retain numeric protein-position order; outcomes never break risk-score ties.
    curves = []
    for fraction in (0.25, 0.5, 0.75, 1.0):
        k = max(1, int(np.ceil(len(risk) * fraction)))
        selected = risk.iloc[:k]
        curves.append({"retained_fraction": fraction, "positions": k,
                       "position_mae_pp": float(selected.forest_mae_pp.mean())})
    return summary, risk.reset_index(), {
        "gains": gains, "disagreement_error_spearman": rho,
        "risk_coverage": curves,
        "warning": "Exploratory in-training OOF ranking; no conformal coverage guarantee"
    }


def run_reliability() -> None:
    """Inspect errors on TRAINING OOF predictions; do not load test outcomes."""
    print("=" * 82)
    print("CFTR-THERATYPE | TRAIN-ONLY COHORT SHIFT AND RELIABILITY")
    print("=" * 82)
    if not SPLIT_OUT.exists() or not MANIFEST.exists():
        raise FileNotFoundError("Missing frozen VTI transfer protocol")
    protocol = json.loads(MANIFEST.read_text())
    if protocol.get("vti_source_sha256") != sha256(VTI_PAIRS):
        raise ValueError("VTI source changed since splitting")
    cohort = pd.read_parquet(SPLIT_OUT,
                             columns=["variant_key", "protein_position", "study_cohort", "transfer_split"])

    tasks = [
        ("total_rescue", RESULTS / "train_cv_predictions.csv", RESULTS / "summary.json"),
        ("corrector_increment", RESULTS / "corrector_train_cv_predictions.csv",
         RESULTS / "corrector_cv_summary.json"),
    ]
    reports = {}
    risks = []
    for target, file, summary_file in tasks:
        if not file.exists() or not summary_file.exists():
            raise FileNotFoundError(f"Missing training-only CV artifact: {file}")
        previous = json.loads(summary_file.read_text())
        prior_source = (previous.get("vti_source_sha256") if target == "total_rescue"
                        else previous.get("data_fingerprints", {}).get("vti_source_sha256"))
        prior_split = (previous.get("vti_partition_sha256") if target == "total_rescue"
                       else previous.get("vti_partition_sha256"))
        if prior_source != sha256(VTI_PAIRS) or prior_split != sha256(SPLIT_OUT):
            raise ValueError(f"Stale OOF predictions for {target}")
        oof = normalise_oof(pd.read_csv(file), target, cohort)
        rows, risk, extra = compute_reliability(oof)
        risk.insert(0, "target", target)
        risks.append(risk)
        reports[target] = {"by_cohort_model": rows, **extra}

        print(f"\n[{target.upper()}] POSITION-BALANCED COHORT ERRORS")
        print(f"{'Cohort':12s} {'Model':21s} {'MAE':>8s} {'Bias':>9s} {'Bias 95% CI':>22s}")
        for r in rows:
            if r["model"] not in ("median", "vti_forest", "pooled_forest"):
                continue
            a, b = r["mean_bias_95ci"]
            print(f"{r['cohort']:12s} {r['model']:21s} {r['position_mae_pp']:8.3f} "
                  f"{r['mean_bias_pp']:+9.3f} [{a:+8.2f}, {b:+8.2f}]")
        print("\nPaired gains (+ means candidate improves):")
        for g in extra["gains"]:
            lo, hi = g["gain_95ci"]
            print(f"  {g['cohort']:8s}: {g['candidate']} vs {g['reference']} "
                  f"{g['gain_pp']:+.3f} pp 95% CI [{lo:+.3f}, {hi:+.3f}]")
        print(f"\nPooled-versus-VTI prediction disagreement vs error: "
              f"Spearman={extra['disagreement_error_spearman']}")
        print("Selective risk by *lowest* disagreement (exploratory):")
        for item in extra["risk_coverage"]:
            print(f"  Retain {item['retained_fraction']:.0%}: "
                  f"{item['positions']} positions, MAE={item['position_mae_pp']:.3f} pp")

    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.concat(risks, ignore_index=True).to_csv(RESULTS / "reliability_position_scores.csv", index=False)
    payload = {"vti_source_sha256": sha256(VTI_PAIRS),
               "vti_split_sha256": sha256(SPLIT_OUT),
               "data_scope": "train_only_out_of_fold_predictions",
               "results": reports,
               "vti_validation_test_outcomes_loaded": False,
               "eti_validation_test_outcomes_loaded": False}
    (RESULTS / "reliability_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("\nSaved: results/transfer/reliability_summary.json")
    print("Saved: results/transfer/reliability_position_scores.csv")
    print("Neither validation nor final-test outcomes were accessed.")
    print("=" * 82)
