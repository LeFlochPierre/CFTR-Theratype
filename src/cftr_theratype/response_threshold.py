"""Phase 14D: development-only ≥10 pp CFTR VTI responsiveness evaluation.

Usage: python -m cftr_theratype.response_threshold

Reads solely stored training out-of-fold predictions and frozen split *identity*
metadata; does not load either ETI/VTI assay table or validation/test outcomes.
Nothing is retrained. Response status uses raw measured improvement ≥10 pp,
not the source article's posterior/empirical-Bayes response category.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

THRESHOLD_PP = 10.0
BOUNDARY_BAND_PP = 5.0  # exploratory observed-only sensitivity band; NOT clinical uncertainty
MODELS = ("vti_forest", "median", "always_responder")
N_REPEATS = 2
N_FOLDS = 5
N_BOOT = 2000
SEED = 20261023


def validate_and_average(oof: pd.DataFrame, membership: pd.DataFrame,
                         *, expected_variants: int | None = 210,
                         expected_positions: int | None = 168) -> pd.DataFrame:
    """Pair stored OOF predictions with frozen TRAIN-only variant identity.

    All outcomes originate from stored OOF *train* predictions. Membership must
    contain only identifiers/partitions, not any phenotype or treatment outcome.
    """
    required = {"repeat", "fold", "model", "variant_key", "protein_position", "observed_pp", "predicted_pp"}
    if not required.issubset(oof.columns):
        raise ValueError(f"OOF missing {sorted(required-set(oof.columns))}")
    meta_req = {"variant_key", "protein_position", "study_cohort", "transfer_split"}
    if not meta_req.issubset(membership.columns):
        raise ValueError(f"Membership missing {sorted(meta_req-set(membership.columns))}")
    # Strong allowlist to avoid unintentionally handling assay outcomes in metadata.
    forbidden = {"vti_rescue_pp", "eti_rescue_pp", "baseline_fsk_pct_wt", "vti_treated_pct_wt"}
    if forbidden.intersection(membership.columns):
        raise ValueError("Membership includes outcome columns; use identity-only frozen split metadata")
    if membership.variant_key.isna().any() or membership.variant_key.duplicated().any():
        raise ValueError("Frozen membership has null or duplicated variant identifiers")
    train = membership.loc[membership.transfer_split.eq("train"),
                           ["variant_key", "protein_position", "study_cohort"]].copy()
    if expected_variants is not None and len(train) != expected_variants:
        raise ValueError(f"Expected {expected_variants} VTI training variants, got {len(train)}")
    if expected_positions is not None and train.protein_position.nunique() != expected_positions:
        raise ValueError(f"Expected {expected_positions} VTI positions")
    if train.study_cohort.isna().any() or not set(train.study_cohort).issubset({"VTI", "ETI, VTI"}):
        raise ValueError("Unknown study cohort in frozen metadata")

    raw = oof.loc[oof.model.isin(("vti_forest", "median")),
                  ["repeat", "fold", "model", "variant_key", "protein_position", "observed_pp", "predicted_pp"]].copy()
    if set(raw.model) != {"vti_forest", "median"}:
        raise ValueError("Original VTI Forest and fold-trained median baseline both required")
    if raw.empty or set(raw.repeat) != {1, 2}:
        raise ValueError("Expected two repetitions of OOF predictions")
    if raw.duplicated(["repeat", "model", "variant_key"]).any():
        raise ValueError("Duplicated OOF model/variant/repeat")
    if raw.isna().any().any() or not np.isfinite(raw[["observed_pp", "predicted_pp"]].to_numpy(dtype=float)).all():
        raise ValueError("OOF predictions contain non-finite or missing values")
    if set(raw.variant_key) != set(train.variant_key):
        raise ValueError("OOF contains missing/extra variants relative to frozen TRAIN membership")
    # Partition validation only; no val or test response rows touched.
    train_pos = train.set_index("variant_key").protein_position
    if (raw.protein_position.to_numpy(dtype=int) != raw.variant_key.map(train_pos).to_numpy(dtype=int)).any():
        raise ValueError("Protein position disagrees with frozen membership")
    counts = raw.groupby(["variant_key", "model"]).repeat.nunique()
    if counts.ne(N_REPEATS).any() or raw.groupby(["variant_key", "model"]).size().ne(N_REPEATS).any():
        raise ValueError("Incomplete repeated OOF predictions")
    if raw.groupby(["variant_key", "repeat"]).fold.nunique().ne(1).any():
        raise ValueError("Inconsistent fold mapping across models")
    if raw.groupby(["repeat", "protein_position"]).fold.nunique().ne(1).any():
        raise ValueError("Same protein position is assigned to multiple heldout folds")
    for model in ("vti_forest", "median"):
        if len(raw.loc[raw.model.eq(model), ["repeat", "fold"]].drop_duplicates()) != N_REPEATS * N_FOLDS:
            raise ValueError(f"Expected ten nonempty heldout folds for {model}")
    if raw.groupby("variant_key").observed_pp.agg(lambda v: v.max()-v.min()).gt(1e-6).any():
        raise ValueError("Observed label changed across repetitions or model predictions")
    wide = (raw.groupby(["variant_key", "model"], as_index=False)
              .agg(prediction=("predicted_pp", "mean"))
              .pivot(index="variant_key", columns="model", values="prediction")
              .reset_index())
    actual = (raw.loc[raw.model.eq("vti_forest"), ["variant_key", "observed_pp"]]
              .groupby("variant_key", as_index=False).observed_pp.mean())
    merged = (train.merge(actual, on="variant_key", validate="one_to_one")
              .merge(wide, on="variant_key", validate="one_to_one"))
    if merged[["observed_pp", "vti_forest", "median"]].isna().any().any():
        raise ValueError("Incomplete paired observed and prediction data")
    merged["responder"] = merged.observed_pp.ge(THRESHOLD_PP)
    merged["near_threshold_observed_5pp"] = (merged.observed_pp - THRESHOLD_PP).abs().le(BOUNDARY_BAND_PP)
    merged["forest_prediction_margin_pp"] = merged.vti_forest - THRESHOLD_PP
    merged["forest_predicted_responder"] = merged.vti_forest.ge(THRESHOLD_PP)
    merged["forest_false_responder"] = merged.forest_predicted_responder & ~merged.responder
    return merged.sort_values("variant_key").reset_index(drop=True)


def position_weights(frame: pd.DataFrame) -> np.ndarray:
    """Equal total weight per protein position *within the evaluation cohort*."""
    if frame.empty:
        raise ValueError("Cannot weight an empty cohort")
    counts = frame.groupby("protein_position").variant_key.transform("size")
    return 1.0 / counts.to_numpy(dtype=float)


def _safe_float(v: float) -> float | None:
    return float(v) if np.isfinite(v) else None


def classification_metrics(frame: pd.DataFrame, scores: np.ndarray | None, *,
                           threshold_pp: float = THRESHOLD_PP) -> dict:
    y = frame.responder.to_numpy(dtype=bool)
    if scores is None:  # explicit nonmodel, always-responsive comparator
        predicted = np.ones(len(y), dtype=bool)
        scores = np.full(len(y), 1.0)
    else:
        scores = np.asarray(scores, dtype=float)
        if scores.shape != (len(y),) or not np.isfinite(scores).all():
            raise ValueError("Invalid continuous prediction score")
        predicted = scores >= threshold_pp
    w = position_weights(frame)
    tp = float(w[y & predicted].sum())
    tn = float(w[~y & ~predicted].sum())
    fp = float(w[~y & predicted].sum())
    fn = float(w[y & ~predicted].sum())
    positive = tp + fn
    negative = tn + fp
    sensitivity = tp / positive if positive else float('nan')
    specificity = tn / negative if negative else float('nan')
    prec_nonresponse = tn / (tn + fn) if tn + fn else float('nan')
    both_classes = y.any() and (~y).any()
    auc = roc_auc_score(y, scores, sample_weight=w) if both_classes else float('nan')
    ap_nonresponse = average_precision_score(~y, -scores, sample_weight=w) if both_classes else float('nan')
    return {
        "balanced_accuracy": _safe_float((sensitivity + specificity) / 2.0),
        "sensitivity_responder": _safe_float(sensitivity),
        "specificity_nonresponder": _safe_float(specificity),
        "precision_nonresponder": _safe_float(prec_nonresponse),
        "auroc_continuous_rescue_score": _safe_float(auc),
        "auprc_nonresponder": _safe_float(ap_nonresponse),
        "nonresponder_prevalence_position_weighted": float(negative / w.sum()),
        "weighted_tp": tp, "weighted_tn": tn, "weighted_fp": fp, "weighted_fn": fn,
        "variant_tp": int(np.sum(y & predicted)), "variant_tn": int(np.sum(~y & ~predicted)),
        "variant_fp": int(np.sum(~y & predicted)), "variant_fn": int(np.sum(y & ~predicted)),
    }


def paired_ba_gain_bootstrap(frame: pd.DataFrame, candidate: np.ndarray, *,
                             n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Paired position-cluster CI for balanced-accuracy gain vs always-responder.

    Use position-level confusion *proportions* so each sampled position has
    equal total weight. Vectorised across bootstrap draws for fast, exact
    position-cluster resampling without treating variant rows as independent.
    """
    score = np.asarray(candidate, dtype=float)
    if score.shape != (len(frame),) or not np.isfinite(score).all():
        raise ValueError("Candidate predictions must be finite and paired with variants")
    truth = frame.responder.to_numpy(dtype=bool)
    pred = score >= THRESHOLD_PP
    positions = frame.protein_position.to_numpy()
    per_pos = []
    for p in sorted(np.unique(positions)):
        sel = positions == p
        ys, ps = truth[sel], pred[sel]
        per_pos.append([
            np.mean(ys & ps),      # true responders predicted responsive
            np.mean(ys),           # measured responders
            np.mean(~ys & ~ps),    # true nonresponders predicted nonresponsive
            np.mean(~ys),          # measured nonresponders
        ])
    aggregates = np.asarray(per_pos, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(aggregates), size=(n_boot, len(aggregates)))
    totals = aggregates[draws].sum(axis=1)
    valid = (totals[:, 1] > 0) & (totals[:, 3] > 0)
    values = 0.5 * (totals[valid, 0] / totals[valid, 1]
                    + totals[valid, 2] / totals[valid, 3]) - 0.5
    if len(values) < max(50, n_boot // 2):
        return {"gain": None, "ci95": None, "usable_replicates": int(len(values))}
    ci = np.quantile(values, [.025, .975])
    actual = classification_metrics(frame, score)["balanced_accuracy"]
    return {"gain": float(actual - 0.5), "ci95": [float(ci[0]), float(ci[1])],
            "usable_replicates": int(len(values))}


def run() -> None:
    from cftr_theratype.transfer import ROOT, RESULTS, SPLIT_OUT, MANIFEST, VTI_PAIRS, sha256
    print("=" * 89)
    print("CFTR-THERATYPE | PHASE 14D — TRAIN-ONLY IN-VITRO 10 PP RESPONSIVENESS")
    print("=" * 89)
    print("Task: raw measured VTI rescue >=10 percentage points WT (not empirical Bayes category)")
    print("Data: saved 2 x 5 position-held-out TRAIN OOF predictions; nothing retrained")
    print("No ETI or VTI validation/test outcome table is opened")
    if not all(p.exists() for p in (MANIFEST, VTI_PAIRS, SPLIT_OUT, RESULTS / "summary.json", RESULTS / "train_cv_predictions.csv")):
        raise FileNotFoundError("Missing original frozen VTI transfer files")
    frozen = json.loads(MANIFEST.read_text())
    summary = json.loads((RESULTS / "summary.json").read_text())
    vti_hash, split_hash = sha256(VTI_PAIRS), sha256(SPLIT_OUT)
    if (frozen.get("vti_source_sha256") != vti_hash
            or summary.get("vti_source_sha256") != vti_hash
            or summary.get("vti_partition_sha256") != split_hash):
        raise ValueError("OOF or transfer membership provenance mismatch")

    membership = pd.read_parquet(SPLIT_OUT, columns=[
        "variant_key", "protein_position", "study_cohort", "transfer_split"])
    reference_file = RESULTS / "train_cv_predictions.csv"
    predictions = pd.read_csv(reference_file)
    dataset = validate_and_average(predictions, membership)
    print(f"VTI frozen TRAIN: {len(dataset)} variants / {dataset.protein_position.nunique()} positions")
    print("Repeats averaged per variant; every protein position weighted equally within each cohort")

    rows = []
    print("\n[1] RESPONSIVENESS AT THE PRESPECIFIED >=10 PP THRESHOLD")
    for j, cohort in enumerate(("all", "ETI, VTI", "VTI")):
        subset = dataset if cohort == "all" else dataset.loc[dataset.study_cohort.eq(cohort)].copy()
        n_pos = subset.protein_position.nunique()
        n_resp, n_nr = int(subset.responder.sum()), int((~subset.responder).sum())
        print(f"\n  {cohort} | {len(subset)} variants / {n_pos} positions | ")
        print(f"    Responders={n_resp} ({n_resp/len(subset):.1%}); nonresponders={n_nr} ({n_nr/len(subset):.1%}); "
              f"measured near-boundary 5–15 pp={int(subset.near_threshold_observed_5pp.sum())}")
        print(f"    {'Model':21s} {'BalAcc':>7s} {'Sens':>7s} {'Spec':>7s} {'AUROC':>7s} {'NR-AP':>7s} {'FP':>4s} {'FN':>4s}")
        for model in MODELS:
            score = None if model == "always_responder" else subset[model].to_numpy(dtype=float)
            m = classification_metrics(subset, score)
            fmt = lambda v: f"{v:7.3f}" if v is not None else "    n/a"
            print(f"    {model:21s} {fmt(m['balanced_accuracy'])} {fmt(m['sensitivity_responder'])} "
                  f"{fmt(m['specificity_nonresponder'])} {fmt(m['auroc_continuous_rescue_score'])} "
                  f"{fmt(m['auprc_nonresponder'])} {m['variant_fp']:4d} {m['variant_fn']:4d}")
            rows.append({"cohort": cohort, "model": model, "n_variants": int(len(subset)),
                         "n_positions": int(n_pos), "n_responder_variants": n_resp,
                         "n_nonresponder_variants": n_nr,
                         "near_boundary_5pp_variants": int(subset.near_threshold_observed_5pp.sum()),
                         **m})
    print("\n[2] FOREST BALANCED-ACCURACY GAIN VS ALWAYS-RESPONDER (position-bootstrap)")
    gains = {}
    for j, cohort in enumerate(("all", "ETI, VTI", "VTI")):
        sub = dataset if cohort == "all" else dataset.loc[dataset.study_cohort.eq(cohort)].copy()
        result = paired_ba_gain_bootstrap(sub, sub.vti_forest.to_numpy(float), seed=SEED + j)
        gains[cohort] = result
        if result["ci95"] is not None:
            lo, hi = result["ci95"]
            print(f"  {cohort:10s}: delta BA={result['gain']:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
        else:
            print(f"  {cohort:10s}: bootstrap unavailable; only one response class in too many draws")

    print("\n[3] INTERPRETATION / LIMITATIONS")
    print("  BA=0.50 for always-responder when both classes present; it has zero specificity.")
    print("  AUROC and nonresponder average precision use *continuous predicted rescue as rank score*.")
    print("  Continuous rescue predictions are NOT calibrated responder probabilities.")
    print("  Measured outcome 5–15 pp is a post-hoc sensitivity flag, NOT prospectively knowable.")
    print("  Raw >=10 pp differs from the source study's empirical-Bayes response classifications.")
    print("  Exploratory train-only analysis; previous training CV/model comparisons were already inspected.")
    print("  No validation/test outcomes were accessed; no models or thresholds refitted.")

    output = ROOT / "results/biology"
    output.mkdir(parents=True, exist_ok=True)
    variants_file = output / "phase14d_response10_train_oof.csv"
    summary_file = output / "phase14d_response10_summary.json"
    dataset.to_csv(variants_file, index=False)
    report = {
        "experiment": "Phase14D fixed >=10 pp raw VTI response classification from stored grouped OOF predictions",
        "threshold_pp_wt": THRESHOLD_PP, "boundary_flag_pp": BOUNDARY_BAND_PP,
        "vti_source_sha256": vti_hash, "vti_partition_sha256": split_hash,
        "oof_sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest(),
        "n_training_variants": len(dataset), "n_training_positions": dataset.protein_position.nunique(),
        "response_definition": "vti_rescue_pp >=10 pp WT; this is not the publication's EB response label",
        "comparators": list(MODELS), "metrics": rows, "paired_gain_vs_always_positive": gains,
        "notes": ["Only saved training OOF response labels and predictions loaded",
                  "Held-out position groups preserved and two repeats averaged by variant",
                  "No calibration performed; regression margins are not probabilities",
                  "Selection-enriched ETI-characterised cohort reported separately",
                  "All metrics exploratory; no fresh confirmatory test evaluation"],
    }
    summary_file.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("\nSaved:", variants_file.relative_to(ROOT))
    print("Saved:", summary_file.relative_to(ROOT))
    print("=" * 89)

if __name__ == "__main__":
    run()
