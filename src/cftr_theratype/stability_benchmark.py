"""Phase 16C: ThermoMPNN incremental value for CFTR VTI, training-only OOF.

Run: python -m cftr_theratype.stability_benchmark
Never loads VTI/ETI validation or test outcomes. Original 2 x 5 position-held-out
folds, matched original forest. Stabilization is *not* ligand binding or rescue.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from cftr_theratype import structure_benchmark as base

N_SHUFFLES = 5
SEED = 20261023
N_BOOT = 2500
MODEL_STABILITY = "structural_plus_thermompnn"
MODEL_STRUCTURE = "structural_forest"
MODEL_SEQUENCE = "vti_forest"
SHUFFLE_PREFIX = "shuffled_thermompnn_"


def verify_stability_artifacts(root: Path, expected_keys: set[str]) -> pd.DataFrame:
    """Check adapter provenance, input hashes, identities, coverage, finite scores."""
    location = root / "results/structure/phase16_thermompnn"
    csv = location / "vti_train_stability_features.csv"
    report_path = location / "thermompnn_provenance.json"
    candidate_path = root / "results/structure/phase16_foldx/candidates_train_only.csv"
    pdb_path = root / "results/structure/phase16_foldx/6O2P_cftr_canonical_numbering.pdb"
    raw_path = location / "ThermoMPNN_inference_6O2P_cftr_canonical_numbering.csv"
    for path in (csv, report_path, candidate_path, pdb_path, raw_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required stability input missing: {path}")
    report = json.loads(report_path.read_text())
    if (report.get("protocol") != "phase16b_thermompnn_single_mutation_reference_6O2P_v1"
            or report.get("n_outcome_rows_accessed") != 0
            or report.get("n_train_variants") != 210
            or report.get("n_scored") != 195
            or report.get("n_missing_structure") != 15):
        raise ValueError("Unexpected ThermoMPNN provenance/coverage; recheck Phase 16B")
    for path, key in ((candidate_path, "candidate_manifest_sha256"),
                      (pdb_path, "pdb_sha256"), (raw_path, "prediction_file_sha256")):
        if base.sha256(path) != report.get(key):
            raise ValueError(f"ThermoMPNN provenance hash mismatch: {path}")
    # The adapter already verified all 23,580 raw scores against PDB identities.
    # Recheck the train-only manifest here so no identity is silently reassigned.
    df = pd.read_csv(csv, dtype={"variant_key": str})
    candidates = pd.read_csv(candidate_path, dtype={"variant_key": str})
    required = {"variant_key", "protein_position", "reference_aa", "alternate_aa",
                "status", "thermompnn_ddg_pred", "thermompnn_status"}
    if not required.issubset(df) or not required.issubset(set(df.columns) | set(candidates.columns)):
        raise ValueError("ThermoMPNN feature columns absent")
    if len(df) != 210 or df.variant_key.duplicated().any() or set(df.variant_key) != expected_keys:
        raise ValueError("Stability features must exactly match 210 frozen VTI train variants")
    fields = ["variant_key", "protein_position", "reference_aa", "alternate_aa", "status"]
    if len(candidates) != 210 or candidates.variant_key.duplicated().any():
        raise ValueError("Mutation candidate manifest changed")
    verify = df[fields].merge(candidates[fields], on="variant_key", validate="one_to_one",
                              suffixes=("_features", "_candidates"), indicator=True)
    if len(verify) != 210 or not verify._merge.eq("both").all():
        raise ValueError("ThermoMPNN candidates do not match")
    for col in fields[1:]:
        if not verify[f"{col}_features"].eq(verify[f"{col}_candidates"]).all():
            raise ValueError(f"Mutation identity mismatch: {col}")
    score = pd.to_numeric(df.thermompnn_ddg_pred, errors="raise").to_numpy(float)
    predicted = df.thermompnn_status.eq("predicted").to_numpy()
    if (predicted.sum() != 195 or (~predicted).sum() != 15
            or not df.loc[predicted, "status"].eq("eligible").all()
            or not df.loc[~predicted, "status"].eq("unresolved_in_6O2P").all()
            or not np.isfinite(score[predicted]).all()
            or not np.isnan(score[~predicted]).all()
            or np.isinf(score).any()):
        raise ValueError("Unexpected stability score values or structural missingness")
    return df


def stability_matrix(train: pd.DataFrame, scores: pd.DataFrame) -> np.ndarray:
    """Mutation-level stability score plus fixed availability indicator."""
    if not train.variant_key.is_unique or not scores.variant_key.is_unique:
        raise ValueError("Duplicate variant identities")
    merged = train[["variant_key", "protein_position", "reference_aa", "alternate_aa"]].merge(
        scores[["variant_key", "protein_position", "reference_aa", "alternate_aa",
                "thermompnn_ddg_pred", "thermompnn_status"]],
        on="variant_key", how="left", validate="one_to_one", suffixes=("_train", "_score"), indicator=True)
    if not merged._merge.eq("both").all():
        raise ValueError("Incomplete stability join")
    for col in ("protein_position", "reference_aa", "alternate_aa"):
        if not merged[f"{col}_train"].eq(merged[f"{col}_score"]).all():
            raise ValueError(f"Score mapped to incorrect mutation: {col}")
    observed = merged.thermompnn_status.eq("predicted").to_numpy(dtype=int)
    value = merged.thermompnn_ddg_pred.to_numpy(dtype=float)
    if not np.array_equal(observed == 1, np.isfinite(value)):
        raise ValueError("Score missingness does not agree with structural status")
    return np.column_stack([value, observed.astype(float)])


def shuffled_stability(matrix: np.ndarray, seed: int) -> np.ndarray:
    """Label-free permutation among *scored* mutations, keeping missing flags fixed."""
    x = np.asarray(matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError("Expected stability score and availability flag")
    if not np.array_equal(np.isfinite(x[:, 0]), x[:, 1] == 1):
        raise ValueError("Stability availability mismatch")
    out = x.copy()
    ix = np.flatnonzero(np.isfinite(x[:, 0]))
    if len(ix) < 2:
        raise ValueError("Too few scores for a negative control")
    out[ix, 0] = x[np.random.default_rng(seed).permutation(ix), 0]
    return out


def validate_phase15c_oof(old: pd.DataFrame, train: pd.DataFrame) -> None:
    required = {"model", "repeat", "fold", "variant_key", "protein_position",
                "study_cohort", "observed_pp", "predicted_pp"}
    if not required.issubset(old):
        raise ValueError("Phase 15C OOF missing columns")
    if set(old.model) != set(base.MODELS):
        raise ValueError("Phase 15C models changed")
    if old.duplicated(["model", "repeat", "variant_key"]).any():
        raise ValueError("Duplicated Phase 15C OOF rows")
    by_train = train.set_index("variant_key")
    for model in base.MODELS:
        subset = old.loc[old.model.eq(model)]
        if len(subset) != 2 * len(train) or set(subset.variant_key) != set(train.variant_key):
            raise ValueError("Phase 15C OOF coverage changed")
        if subset.groupby("variant_key").repeat.nunique().ne(2).any():
            raise ValueError("Incomplete Phase 15C fold predictions")
    if not old.variant_key.map(by_train.protein_position).eq(old.protein_position).all():
        raise ValueError("Stored OOF mutation positions changed")
    if not old.variant_key.map(by_train.study_cohort).eq(old.study_cohort).all():
        raise ValueError("Stored OOF cohort changed")
    if not np.allclose(old.variant_key.map(by_train.vti_rescue_pp).to_numpy(float),
                       old.observed_pp.to_numpy(float), rtol=0, atol=1e-6):
        raise ValueError("Stored OOF truth no longer matches frozen TRAIN")
    for (rep, key), group in old.groupby(["repeat", "variant_key"]):
        if group.fold.nunique() != 1:
            raise ValueError("Fold identity differs across models")
    forest = old.loc[old.model.eq(MODEL_SEQUENCE)]
    base.validate_stored_forest(forest, train)


def averaged_rows(oof: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    if oof.duplicated(["model", "repeat", "variant_key"]).any():
        raise ValueError("Duplicate predictions")
    if set(oof.model) != set(models):
        raise ValueError("Missing expected models")
    expected_keys = set(oof.loc[oof.model.eq(models[0]), "variant_key"])
    for name in models:
        part = oof.loc[oof.model.eq(name)]
        if len(part) != 2 * len(expected_keys) or set(part.variant_key) != expected_keys:
            raise ValueError("Missing OOF predictions")
        if part.groupby("variant_key").repeat.nunique().ne(2).any():
            raise ValueError("Incomplete OOF repeat coverage")
    for _, group in oof.groupby("variant_key"):
        if (group.protein_position.nunique() != 1 or group.study_cohort.nunique() != 1
                or group.observed_pp.max() - group.observed_pp.min() > 1e-6):
            raise ValueError("Inconsistent identity metadata / observed response")
    avg = oof.groupby(["model", "variant_key", "protein_position", "study_cohort"], as_index=False).agg(
        observed_pp=("observed_pp", "mean"), predicted_pp=("predicted_pp", "mean"))
    if not np.isfinite(avg[["observed_pp", "predicted_pp"]].to_numpy(float)).all():
        raise ValueError("Non-finite OOF predictions")
    avg["abs_error_pp"] = (avg.predicted_pp - avg.observed_pp).abs()
    avg["signed_error_pp"] = avg.predicted_pp - avg.observed_pp
    return avg


def evaluate(oof: pd.DataFrame, *, n_boot: int = N_BOOT) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = [MODEL_SEQUENCE, MODEL_STRUCTURE, MODEL_STABILITY] + [f"{SHUFFLE_PREFIX}{j}" for j in range(N_SHUFFLES)]
    avg = averaged_rows(oof, models)
    summary, classifiers = [], []
    for cohort_index, cohort in enumerate(("all", "ETI, VTI", "VTI")):
        sub = avg if cohort == "all" else avg.loc[avg.study_cohort.eq(cohort)]
        grouped = sub.groupby(["protein_position", "model"]).abs_error_pp.mean().unstack("model")
        biases = sub.groupby(["protein_position", "model"]).signed_error_pp.mean().unstack("model")
        if grouped.isna().any().any():
            raise ValueError("Missing model prediction at one of the cohort's positions")
        err_base = grouped[MODEL_STRUCTURE].to_numpy(float)
        err_null = grouped[[f"{SHUFFLE_PREFIX}{i}" for i in range(N_SHUFFLES)]].mean(axis=1).to_numpy(float)
        rng = np.random.default_rng(SEED + cohort_index)
        draws = rng.integers(0, len(grouped), size=(n_boot, len(grouped)))
        for model in models:
            error = grouped[model].to_numpy(float)
            delta_struct = err_base - error
            delta_null = err_null - error
            ci_struct = np.quantile(delta_struct[draws].mean(axis=1), [.025, .975])
            ci_null = np.quantile(delta_null[draws].mean(axis=1), [.025, .975])
            summary.append({
                "cohort": cohort, "model": model, "n_positions": int(len(grouped)),
                "position_mae_pp": float(error.mean()),
                "bias_pp": float(biases[model].mean()),
                "gain_vs_structural_pp": float(delta_struct.mean()),
                "gain_vs_structural_ci95": [float(z) for z in ci_struct],
                "gain_vs_mean_shuffled_pp": float(delta_null.mean()),
                "gain_vs_mean_shuffled_ci95": [float(z) for z in ci_null],
            })
        for model in (MODEL_SEQUENCE, MODEL_STRUCTURE, MODEL_STABILITY, *[f"{SHUFFLE_PREFIX}{j}" for j in range(N_SHUFFLES)]):
            frame = sub.loc[sub.model.eq(model)].copy()
            resp = frame.observed_pp.to_numpy(float) >= 10.0
            scores = frame.predicted_pp.to_numpy(float)
            predicted_resp = scores >= 10.0
            weights = (1 / frame.groupby("protein_position").variant_key.transform("size")).to_numpy(float)
            if not resp.any() or resp.all():
                raise ValueError("Both assay-response classes required for threshold metrics")
            cls = {
                "cohort": cohort, "model": model,
                "nonresponder_auroc": float(roc_auc_score(~resp, -scores, sample_weight=weights)),
                "nonresponder_ap": float(average_precision_score(~resp, -scores, sample_weight=weights)),
                "nonresponders_detected": int(np.count_nonzero((~resp) & (~predicted_resp))),
                "nonresponders_total": int(np.count_nonzero(~resp)),
                "false_nonresponder_alerts": int(np.count_nonzero(resp & (~predicted_resp))),
            }
            # Fixed 20% variant-level research triage, not optimized from labels.
            indices = np.argsort(scores, kind="stable")[:int(np.ceil(0.2 * len(frame)))]
            cls["triage_selected"] = int(len(indices))
            cls["triage_nr_caught"] = int(np.count_nonzero(~resp[indices]))
            classifiers.append(cls)
    return pd.DataFrame(summary), pd.DataFrame(classifiers), avg


def run(root: Path | None = None) -> None:
    from cftr_theratype.benchmark import make_features
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.splits import parse_substitution
    from cftr_theratype.transfer import (ROOT, VTI_PAIRS, SPLIT_OUT, MANIFEST, RESULTS,
                                         SEED as TRANSFER_SEED, _load_allowed_labels, _estimator)

    root = root or ROOT
    oof_path = root / "results/structure/phase15c_train_oof.csv"
    phase15c_path = root / "results/structure/phase15c_summary.json"
    struct_path = root / "results/structure/phase15b_position_features.csv"
    struct_manifest = root / "results/structure/phase15b_feature_summary.json"
    transfer_oof = root / "results/transfer/train_cv_predictions.csv"
    transfer_summary = root / "results/transfer/summary.json"
    for path in (VTI_PAIRS, SPLIT_OUT, MANIFEST, oof_path, phase15c_path, struct_path,
                 struct_manifest, transfer_oof, transfer_summary):
        if not path.is_file():
            raise FileNotFoundError(path)
    p15 = json.loads(phase15c_path.read_text())
    orig_summary = json.loads(transfer_summary.read_text())
    protocol = json.loads(MANIFEST.read_text())
    if (p15.get("phase") != "15C_structure_benchmark"
            or p15.get("baseline_reproduced_per_fold") is not True
            or p15.get("validation_or_test_outcomes_loaded") is not False
            or p15.get("vti_source_sha256") != base.sha256(VTI_PAIRS)
            or p15.get("vti_partition_sha256") != base.sha256(SPLIT_OUT)
            or p15.get("structure_csv_sha256") != base.sha256(struct_path)
            or p15.get("structure_manifest_sha256") != base.sha256(struct_manifest)
            or p15.get("baseline_oof_sha256") != base.sha256(transfer_oof)
            or p15.get("baseline_summary_sha256") != base.sha256(transfer_summary)
            or orig_summary.get("vti_source_sha256") != base.sha256(VTI_PAIRS)
            or protocol.get("vti_source_sha256") != base.sha256(VTI_PAIRS)):
        raise ValueError("Phase 15C/transfer provenance mismatch")
    structure = base.read_feature_table(struct_path, struct_manifest, root)
    metadata = pd.read_parquet(SPLIT_OUT, columns=["variant_key", "protein_position", "transfer_split",
                                                   "study_cohort", "variant", "variant_type", "hgvs_protein"])
    train = metadata.loc[metadata.transfer_split.eq("train")].copy()
    if len(train) != 210 or train.variant_key.duplicated().any() or train.protein_position.nunique() != 168:
        raise ValueError("VTI frozen training split no longer matches")
    labels = _load_allowed_labels(VTI_PAIRS, set(train.variant_key), "vti_rescue_pp")
    train = train.merge(labels[["variant_key", "vti_rescue_pp"]], on="variant_key", validate="one_to_one")
    if train.vti_rescue_pp.isna().any():
        raise ValueError("Missing allowed VTI train labels")
    seq = bihler_2024_assay_sequence()
    parsed = [parse_substitution(row, seq) for _, row in train.iterrows()]
    if any(p[-1] != "valid" or p[0] != pos for p, pos in zip(parsed, train.protein_position)):
        raise ValueError("Unverified amino-acid substitutions")
    train["reference_aa"] = [p[1] for p in parsed]
    train["alternate_aa"] = [p[2] for p in parsed]
    train = train.sort_values("variant_key").reset_index(drop=True)
    keys = set(train.variant_key)
    scores = verify_stability_artifacts(root, keys)
    Xstab = stability_matrix(train, scores)
    Xseq, cols = make_features(train, seq)
    Xseq = np.asarray(Xseq, float)
    if Xseq.shape != (210, 68) or not np.isfinite(Xseq).all():
        raise ValueError("Sequence baseline has unexpected dimensions")
    positions = train.protein_position.to_numpy(int)
    Xstruct = base.structural_matrix(positions, structure)
    old = pd.read_csv(oof_path, dtype={"variant_key": str})
    validate_phase15c_oof(old, train)
    old_reference = old.loc[old.model.isin([MODEL_STRUCTURE, MODEL_SEQUENCE])].copy()
    reference = old.loc[old.model.eq(MODEL_STRUCTURE)]
    # fixed, label-independent shuffles; preserve original missing positions
    control_features = [shuffled_stability(Xstab, seed=SEED + 100*i) for i in range(N_SHUFFLES)]
    by_key = {str(k): i for i, k in enumerate(train.variant_key)}
    y = train.vti_rescue_pp.to_numpy(float)
    outputs = []
    print("="*92)
    print("CFTR-THERATYPE | PHASE 16C: MUTATION STABILITY INCREMENTAL VALUE — TRAIN ONLY")
    print("="*92)
    print(f"Frozen VTI TRAIN: {len(train)} variants / {len(np.unique(positions))} positions")
    print(f"ThermoMPNN: {int(np.isfinite(Xstab[:,0]).sum())} scored; "
          f"{int(np.isnan(Xstab[:,0]).sum())} unresolved; scores label-free")
    print("Input: sequence 68 + reference structure 12 + stability score + availability flag")
    print(f"Controls: {N_SHUFFLES} label-free score permutations; missingness retained")
    print("Reference: 6O2P V470, engineered E1371Q; no ligand affinity predicted")
    print("Validation/test labels: NEVER LOADED; no decision-threshold tuning")
    print("[1] MATCHED POSITION-GROUPED 2 x 5 FOLDS")
    for (rep, fold), held in reference.groupby(["repeat", "fold"], sort=True):
        held_idx = np.array([by_key[str(k)] for k in held.variant_key], dtype=int)
        held_pos = np.unique(positions[held_idx])
        fit_idx = np.flatnonzero(~np.isin(positions, held_pos))
        if len(fit_idx) + len(held_idx) != len(train) or np.isin(positions[fit_idx], held_pos).any():
            raise ValueError("Position leakage / incomplete held-out fold")
        fold_seed = TRANSFER_SEED + (int(rep)-1)*10 + int(fold)
        tr, te = base.impute_fold(Xstruct[fit_idx], Xstruct[held_idx])
        Xtr, Xte = np.column_stack([Xseq[fit_idx], tr]), np.column_stack([Xseq[held_idx], te])
        est = _estimator("vti_forest", fold_seed)
        est.fit(Xtr, y[fit_idx])
        expected = est.predict(Xte)
        if not np.allclose(expected, held.predicted_pp.to_numpy(float), rtol=0, atol=1e-6):
            raise ValueError(f"Frozen structure Forest not reproduced in repeat {rep}, fold {fold}")
        runs = [(MODEL_STABILITY, Xstab)] + [
            (f"{SHUFFLE_PREFIX}{j}", control_features[j]) for j in range(N_SHUFFLES)]
        for name, stab in runs:
            stab_tr, stab_te = base.impute_fold(stab[fit_idx], stab[held_idx])
            forest = _estimator("vti_forest", fold_seed)
            forest.fit(np.column_stack([Xtr, stab_tr]), y[fit_idx])
            preds = forest.predict(np.column_stack([Xte, stab_te]))
            if not np.isfinite(preds).all():
                raise ValueError(f"Non-finite prediction: {name}")
            for i, pred in zip(held_idx, preds):
                outputs.append({
                    "repeat": int(rep), "fold": int(fold), "model": name,
                    "variant_key": str(train.iloc[i].variant_key),
                    "protein_position": int(positions[i]),
                    "study_cohort": str(train.iloc[i].study_cohort),
                    "observed_pp": float(y[i]), "predicted_pp": float(pred),
                })
        print(f"  repeat={rep}/2 fold={fold}/5 | fit={len(fit_idx):3d} held={len(held_idx):3d} "
              "| structural Forest reproduced", flush=True)
    combined = pd.concat([old_reference, pd.DataFrame(outputs)], ignore_index=True)
    summary, cls, avg = evaluate(combined)
    print("\n[2] RESCUE POSITION MAE (LOWER IS BETTER)")
    print(f"{'Cohort':10s} {'Model':31s} {'MAE':>7s} {'Bias':>8s} {'Gain vs struct':>15s} {'95% CI':>22s}")
    wanted = (MODEL_SEQUENCE, MODEL_STRUCTURE, MODEL_STABILITY)
    for row in summary.loc[summary.model.isin(wanted)].itertuples(index=False):
        lo, hi = row.gain_vs_structural_ci95
        print(f"{row.cohort:10s} {row.model:31s} {row.position_mae_pp:7.3f} "
              f"{row.bias_pp:+8.3f} {row.gain_vs_structural_pp:+15.3f} [{lo:+9.3f},{hi:+9.3f}]")
    print("\n[3] STABILITY SIGNAL vs SHUFFLED-STABILITY CONTROLS")
    for cohort in ("all", "ETI, VTI", "VTI"):
        rows = summary.loc[(summary.cohort.eq(cohort)) & (summary.model.str.startswith(SHUFFLE_PREFIX))]
        stab_row = summary.loc[summary.cohort.eq(cohort) & summary.model.eq(MODEL_STABILITY)].iloc[0]
        lo, hi = stab_row.gain_vs_mean_shuffled_ci95
        print(f"  {cohort:10s} | shuffled MAE mean={rows.position_mae_pp.mean():.3f} "
              f"range=[{rows.position_mae_pp.min():.3f},{rows.position_mae_pp.max():.3f}] | "
              f"real vs mean shuffle gain={stab_row.gain_vs_mean_shuffled_pp:+.3f} "
              f"95% position CI [{lo:+.3f},{hi:+.3f}]")
    print("\n[4] RAW >=10 PP NONRESPONSE RANKING AND FIXED 20% TRIAGE")
    print(f"{'Cohort':10s} {'Model':31s} {'AUROC':>7s} {'NR-AP':>7s} {'NR caught':>12s} {'Triage NR':>12s}")
    for row in cls.loc[cls.model.isin(wanted)].itertuples(index=False):
        print(f"{row.cohort:10s} {row.model:31s} {row.nonresponder_auroc:7.3f} "
              f"{row.nonresponder_ap:7.3f} {row.nonresponders_detected:3d}/{row.nonresponders_total:<3d} "
              f"{row.triage_nr_caught:3d}/{row.triage_selected:<3d}")
    print("\n[5] WRITING PROVENANCE + TRAINING-ONLY RESULTS")
    outdir = root / "results/structure"
    outdir.mkdir(parents=True, exist_ok=True)
    out_oof = outdir / "phase16c_stability_train_oof.csv"
    out_summary = outdir / "phase16c_stability_summary.csv"
    out_classifier = outdir / "phase16c_stability_nonresponse.csv"
    out_json = outdir / "phase16c_stability_summary.json"
    combined.to_csv(out_oof, index=False)
    summary.to_csv(out_summary, index=False)
    cls.to_csv(out_classifier, index=False)
    source_report = json.loads((root / "results/structure/phase16_thermompnn/thermompnn_provenance.json").read_text())
    report = {
        "phase": "16C_thermompnn_stability_comparison",
        "status": "exploratory_training_only",
        "n_vti_train": int(len(train)), "n_positions": int(train.protein_position.nunique()),
        "n_scored": int(np.isfinite(Xstab[:, 0]).sum()),
        "n_shuffles": N_SHUFFLES, "seed": SEED,
        "n_cv_repeats": 2, "n_cv_folds": 5,
        "phase15c_summary_sha256": base.sha256(phase15c_path),
        "phase15c_oof_sha256": base.sha256(oof_path),
        "stability_features_sha256": base.sha256(root / "results/structure/phase16_thermompnn/vti_train_stability_features.csv"),
        "stability_provenance_sha256": base.sha256(root / "results/structure/phase16_thermompnn/thermompnn_provenance.json"),
        "thermompnn_checkpoint_sha256": source_report["thermompnn_pretrained_checkpoint_sha256"],
        "thermompnn_git_commit": source_report["thermompnn_source_git_commit"],
        "true_feature": "ThermoMPNN predicted stability score; negative stabilising in upstream convention; no drug binding",
        "reference_construct": "6O2P V470 and engineered E1371Q, unlike experimental assay M470",
        "stability_score_missing_imputation": "train-fold median with availability indicator; no variants dropped",
        "score_shuffles": "Five label-free across-scored-variant permutations, missingness held fixed",
        "model": "250-tree random forest with original 0.75 max_features and leaf-size 4; unchanged",
        "structure_forest_reproduced": True,
        "position_metrics": summary.to_dict(orient="records"),
        "nonresponse_metrics": cls.to_dict(orient="records"),
        "validation_test_outcomes_accessed": False,
        "cautions": ["No new dataset split, tuning, or final-test evaluation",
                     "Stability from single 6O2P reference conformation, not experimentally observed mutants",
                     "No ligand binding or clinical drug response predictions",
                     "Exploratory fold comparisons after substantial prior model exploration",
                     "Raw 10pp assay classification differs from empirical-Bayes classifications"],
    }
    out_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for path in (out_oof, out_summary, out_classifier, out_json):
        print(f"  {path.relative_to(root)}")
    print("PASS: All original folds reproduced; no validation/test outcomes accessed")
    print("="*92)


if __name__ == "__main__":
    run()
