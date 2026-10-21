"""Train-only grouped-CV benchmark for the corrector-associated response.

Models the response after corrector pretreatment relative to acute ivacaftor
on DMSO, using the same FSK+IVA readout within each study:
  ETI_corrector = FSK+IVA (ELX/TEZ) - FSK+IVA (DMSO)
  VTI_corrector = FSK+IVA (VNZ/TEZ) - FSK+IVA (DMSO)

Crucially, features contain SEQUENCE properties only. Neither measured
baseline activity, IVA responsiveness, study cohort nor response group is an
input feature. No ETI/VTI validation or test labels are loaded.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from cftr_theratype.transfer import (
    ROOT, RESULTS, SEED, VTI_PAIRS, ETI_PAIRS, ETI_SPLITS,
    SPLIT_OUT, MANIFEST, MEMBERSHIP, MODELS, sha256,
    _fit_and_predict, _load_allowed_labels, _feature_frame,
)

TARGET_VTI = "vti_increment_over_iva_pp"
TARGET_ETI = "eti_increment_over_iva_pp"
N_REPEATS = 2
N_BOOT = 2500


def per_position_errors(predictions: pd.DataFrame) -> pd.DataFrame:
    """One mean absolute error per position, treatment cohort, and model.

    Repeated CV appearances of a variant contribute equally to its position's
    mean. Each distinct position is counted once within each study cohort.
    """
    required = {"model", "protein_position", "study_cohort", "absolute_error_pp"}
    if not required.issubset(predictions):
        raise ValueError(f"Missing columns: {sorted(required - set(predictions.columns))}")
    if predictions[list(required)].isna().any().any():
        raise ValueError("Missing prediction metadata or errors")
    return (predictions.groupby(
        ["study_cohort", "protein_position", "model"],
        as_index=False,
    )["absolute_error_pp"].mean())


def cluster_gains(frame: pd.DataFrame, candidate: str, reference: str,
                  seed: int = SEED, n_boot: int = N_BOOT) -> tuple[float, float, float]:
    """Position-paired gain (reference MAE minus candidate MAE) and 95% CI."""
    wide = frame.pivot(index="protein_position", columns="model", values="absolute_error_pp")
    if candidate not in wide or reference not in wide:
        raise ValueError("Required paired models not present")
    errors = wide[[reference, candidate]].dropna()
    if len(errors) < 3:
        raise ValueError("Not enough paired positions")
    gains = (errors[reference] - errors[candidate]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(gains), size=(n_boot, len(gains)))
    ci = np.quantile(gains[idx].mean(axis=1), [0.025, 0.975])
    return float(gains.mean()), float(ci[0]), float(ci[1])



def print_total_rescue_strata(training_metadata: pd.DataFrame) -> None:
    """Read prior TRAIN-ONLY CV predictions, never any held-out outcomes.

    Provides a consistent cohort-level comparator for corrector-increment CV.
    This is descriptive; cohort membership is an evaluation stratum, not a
    prediction input, and comparisons across different targets are not MAE
    improvements on the same scale.
    """
    source = RESULTS / "train_cv_predictions.csv"
    summary_path = RESULTS / "summary.json"
    if not source.exists() or not summary_path.exists():
        print("\n[0] Previous total-rescue CV not found; skip contextual summary")
        return
    summary = json.loads(summary_path.read_text())
    if (summary.get("vti_source_sha256") != sha256(VTI_PAIRS)
            or summary.get("vti_partition_sha256") != sha256(SPLIT_OUT)):
        raise ValueError("Prior total-rescue CV comes from different source or frozen split")
    cv = pd.read_csv(source)
    metadata = training_metadata[["variant_key", "study_cohort", "protein_position"]]
    if not set(cv.variant_key).issubset(set(metadata.variant_key)):
        raise ValueError("Previous total-rescue CV includes nontraining keys")
    if (cv.groupby(["model", "variant_key"]).size() != N_REPEATS).any():
        raise ValueError("Prior CV coverage differs from two repeats per variant")
    cv = cv.merge(metadata, on="variant_key", how="inner", validate="many_to_one",
                  suffixes=("_prediction", "_membership"))
    if not cv.protein_position_prediction.eq(cv.protein_position_membership).all():
        raise ValueError("Position mismatch in prior total-response CV")
    cv["protein_position"] = cv["protein_position_membership"]
    cv["signed_error_pp"] = cv.predicted_pp - cv.observed_pp
    print("\n[0] PREVIOUS TRAIN-ONLY TOTAL-RESCUE CV BY STUDY COHORT")
    print("  Cohort is evaluation metadata only, not a model feature")
    for subgroup, rows in cv.groupby("study_cohort"):
        n_variants = rows.variant_key.nunique()
        n_positions = rows.protein_position.nunique()
        print(f"  {subgroup}: {n_variants} variants / {n_positions} positions")
        for model in ("median", "vti_forest", "pooled_forest", "eti_zero_shot_forest"):
            selected = rows.loc[rows.model.eq(model)]
            per_pos = selected.groupby("protein_position")[[
                "absolute_error_pp", "signed_error_pp"]].mean()
            print(f"    {model:25s} pos-MAE={per_pos.absolute_error_pp.mean():7.3f} "
                  f"mean bias={per_pos.signed_error_pp.mean():+7.3f} pp")

def run_mechanism() -> None:
    """Compare direct VTI and ETI-transfer baselines for CORRECTOR rescue."""
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.cv import position_folds
    from cftr_theratype.splits import parse_substitution

    print("=" * 83)
    print("CFTR-THERATYPE | CORRECTOR-ASSOCIATED ETI -> VTI TRANSFER")
    print("=" * 83)

    protocol = json.loads(MANIFEST.read_text())
    hashes = {
        "vti_source_sha256": sha256(VTI_PAIRS),
        "membership_sha256": sha256(MEMBERSHIP),
        "eti_split_sha256": sha256(ETI_SPLITS),
    }
    for key, digest in hashes.items():
        if protocol.get(key) != digest:
            raise ValueError(f"Frozen transfer protocol fingerprint mismatch: {key}")

    cohort = pd.read_parquet(SPLIT_OUT)
    if cohort.variant_key.isna().any() or cohort.variant_key.duplicated().any():
        raise ValueError("VTI membership contains duplicated or null keys")
    if cohort.groupby("protein_position", dropna=True).transfer_split.nunique().max() > 1:
        raise ValueError("Same position appears in different VTI partitions")

    vti_train = cohort.loc[cohort.transfer_split.eq("train"), [
        "variant_key", "protein_position", "variant_type", "hgvs_protein",
        "variant", "study_cohort",
    ]].copy()
    heldout_vti_pos = set(cohort.loc[cohort.transfer_split.isin(["val", "test"]),
                                      "protein_position"].dropna().astype(int))
    if set(vti_train.protein_position.astype(int)) & heldout_vti_pos:
        raise ValueError("VTI training position overlaps val/test")

    eti_splits = pd.read_parquet(ETI_SPLITS)
    eti_train = eti_splits.loc[eti_splits.split.eq("train"), [
        "variant_key", "protein_position", "reference_aa", "alternate_aa",
    ]].copy()
    protected_eti_positions = set(eti_splits.loc[
        eti_splits.split.isin(["val", "test"]), "protein_position"].astype(int))
    if set(vti_train.protein_position.astype(int)) & protected_eti_positions:
        raise ValueError("ETI protected positions leaked into VTI training")
    if set(eti_train.protein_position.astype(int)) & protected_eti_positions:
        raise ValueError("ETI group split compromised")

    # Explicitly materialise only allowed training outcomes via Parquet filters.
    vti_y = _load_allowed_labels(VTI_PAIRS, set(vti_train.variant_key), TARGET_VTI)
    eti_y = _load_allowed_labels(ETI_PAIRS, set(eti_train.variant_key), TARGET_ETI)
    vti_train = vti_train.merge(vti_y, on="variant_key", validate="one_to_one")
    eti_train = eti_train.merge(eti_y, on="variant_key", validate="one_to_one")
    if len(vti_train) != len(vti_y) or len(eti_train) != len(eti_y):
        raise ValueError("Training label coverage mismatch")
    if vti_train[TARGET_VTI].isna().any() or eti_train[TARGET_ETI].isna().any():
        raise ValueError("Missing corrector-increment training outcomes")

    seq = bihler_2024_assay_sequence()
    metadata = [parse_substitution(row, seq) for _, row in vti_train.iterrows()]
    if any(status != "valid" or pos != int(row.protein_position)
           for (pos, _, _, status), row in zip(metadata, vti_train.itertuples(index=False))):
        raise ValueError("VTI mutant does not match 2024 experimental reference")
    vti_train["reference_aa"] = [item[1] for item in metadata]
    vti_train["alternate_aa"] = [item[2] for item in metadata]

    vti_train = vti_train.sort_values("variant_key").reset_index(drop=True)
    eti_train = eti_train.sort_values("variant_key").reset_index(drop=True)
    xv = _feature_frame(vti_train, seq)
    xe = _feature_frame(eti_train, seq)
    yv = vti_train[TARGET_VTI].to_numpy(float)
    ye = eti_train[TARGET_ETI].to_numpy(float)
    pv = vti_train.protein_position.to_numpy(int)
    pe = eti_train.protein_position.to_numpy(int)
    study = vti_train.study_cohort.astype(str).to_numpy()

    print_total_rescue_strata(vti_train)

    print("\n[1] LEAKAGE-SAFE INPUTS")
    print(f"VTI training:       {len(yv)} variants / {len(np.unique(pv))} positions")
    print(f"ETI training:       {len(ye)} variants / {len(np.unique(pe))} positions")
    print(f"2026 study strata:  {pd.Series(study).value_counts().to_dict()}")
    print("Targets:             corrected FSK+IVA minus DMSO FSK+IVA (pp WT)")
    print("Features:            protein sequence and substitution only")
    print("2024/2026 validation and final test outcomes: NOT LOADED")

    predictions = []
    print("\n[2] GROUPED TRAINING CV (2 REPEATS X 5 FOLDS)")
    for repeat in range(N_REPEATS):
        coverage = np.zeros(len(yv), dtype=int)
        for fold, tr_mask, val_mask in position_folds(
            pv, seed=SEED + 100 + repeat,
        ):
            coverage[val_mask] += 1
            held_positions = set(pv[val_mask])
            excluded_pos = held_positions | heldout_vti_pos
            e_mask = ~np.isin(pe, list(excluded_pos))
            if set(pe[e_mask]) & excluded_pos:
                raise ValueError("ETI fold leakage")
            print(f"  Repeat {repeat+1}/{N_REPEATS} | fold {fold}/5 | "
                  f"VTI train={tr_mask.sum()} held={val_mask.sum()} | "
                  f"ETI allowed={e_mask.sum()}", flush=True)
            for model in MODELS:
                pred = _fit_and_predict(
                    model,
                    xv[tr_mask], yv[tr_mask],
                    xe[e_mask], ye[e_mask],
                    xv[val_mask], seed=SEED + repeat * 10 + fold,
                )
                for index, value in zip(np.flatnonzero(val_mask), pred):
                    predictions.append({
                        "repeat": repeat + 1,
                        "fold": fold,
                        "model": model,
                        "variant_key": str(vti_train.iloc[index].variant_key),
                        "protein_position": int(pv[index]),
                        "study_cohort": study[index],
                        "observed_corrector_pp": float(yv[index]),
                        "predicted_corrector_pp": float(value),
                        "absolute_error_pp": float(abs(yv[index] - value)),
                    })
        if not np.all(coverage == 1):
            raise ValueError("Each variant must have one CV prediction per repeat")

    cv = pd.DataFrame(predictions)
    by_pos = per_position_errors(cv)
    all_positions = cv.groupby(["protein_position", "model"], as_index=False)[
        "absolute_error_pp"].mean()
    overall = all_positions.groupby("model").absolute_error_pp.mean().sort_values()

    print("\n[3] OVERALL CV: CORRECTOR INCREMENT POSITION MAE")
    for name, value in overall.items():
        print(f"  {name:25s} {value:8.3f} pp")

    print("\n[4] PRE-SPECIFIED STRATIFIED CV (REPORTING ONLY)")
    subgroup_report = {}
    for subgroup in sorted(set(study)):
        sub = by_pos.loc[by_pos.study_cohort.eq(subgroup)].copy()
        piv = sub.pivot(index="protein_position", columns="model", values="absolute_error_pp")
        subgroup_report[subgroup] = {
            "variants": int(sum(study == subgroup)),
            "positions": int(piv.shape[0]),
            "position_mae_pp": {str(k): float(v) for k, v in piv.mean().items()},
        }
        print(f"  {subgroup} | {subgroup_report[subgroup]['variants']} variants / "
              f"{len(piv)} positions")
        for model in MODELS:
            print(f"    {model:25s} {piv[model].mean():8.3f} pp")

    print("\n[5] PAIRED POSITION-CLUSTERED GAINS")
    comparisons = ["vti_forest", "pooled_forest", "eti_zero_shot_forest"]
    gain_report = {}
    for candidate in comparisons:
        gain, lo, hi = cluster_gains(all_positions, candidate, "median")
        gain_report[candidate] = [gain, lo, hi]
        print(f"  {candidate:25s} vs median | improvement={gain:+.3f} pp "
              f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    gain, lo, hi = cluster_gains(all_positions, "pooled_forest", "vti_forest")
    gain_report["pooled_vs_vti_forest"] = [gain, lo, hi]
    print(f"  pooled_forest vs vti_forest   | improvement={gain:+.3f} pp "
          f"95% CI [{lo:+.3f}, {hi:+.3f}]")

    RESULTS.mkdir(parents=True, exist_ok=True)
    cv.to_csv(RESULTS / "corrector_train_cv_predictions.csv", index=False)
    report = {
        "target_vti": TARGET_VTI,
        "target_eti": TARGET_ETI,
        "data_fingerprints": hashes,
        "vti_partition_sha256": sha256(SPLIT_OUT),
        "cv_protocol": "2 repeats x 5 position-disjoint folds, training only",
        "overall_position_mae_pp": {str(k): float(v) for k, v in overall.items()},
        "study_strata": subgroup_report,
        "paired_position_gains": gain_report,
        "vti_val_or_test_outcomes_used": False,
        "eti_val_or_test_outcomes_used": False,
        "warning": "Exploratory corrector-response comparison, not clinical or causal superiority",
    }
    (RESULTS / "corrector_cv_summary.json").write_text(
        json.dumps(report, indent=2) + "\n")
    print("\nSaved: results/transfer/corrector_train_cv_predictions.csv")
    print("Saved: results/transfer/corrector_cv_summary.json")
    print("No validation or test outcomes accessed. CORRECTOR CV COMPLETE")
    print("=" * 83)


if __name__ == "__main__":
    run_mechanism()
