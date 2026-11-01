"""Phase 15C: position-safe VTI structure-augmented Forest, TRAIN-ONLY.

Run: python -m cftr_theratype.structure_benchmark

The 2024/2026 validation and final-test outcomes are NEVER read. Structural
features come from static experimental *reference* CFTR conformations, not
experimentally solved mutant proteins or mutation-specific affinity scores.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# Prespecified; no result-based filtering, no hyperparameter search.
STRUCTURE_COLUMNS = (
    "reference_contacts_8A",
    "reference_long_range_contacts_8A",
    "reference_protein_sasa_A2",
    "reference_protein_relative_sasa",
    "6O2P_VX7_ca_distance_A",
    "8EIQ_VX7_ca_distance_A",
    "8EIQ_CV6_ca_distance_A",
    "8EIQ_WJX_ca_distance_A",
    "has_exact_reference",
    "6O2P_resolved_exact",
    "8EIQ_resolved_exact",
    "assay_reference_M470_vs_PDB_V470",
)
DISTANCE_AND_GEOMETRY = STRUCTURE_COLUMNS[:8]
MODELS = ("vti_forest", "structural_forest", "shuffled_structure_forest")
COHORTS = ("ETI, VTI", "VTI")
N_REPEATS = 2
N_FOLDS = 5
N_BOOT = 3000
SHUFFLE_SEED = 20261023
THRESHOLD_PP = 10.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_feature_table(path: Path, summary_path: Path, root: Path) -> pd.DataFrame:
    """Validate structural provenance and numeric-only position features."""
    report = json.loads(summary_path.read_text())
    if (report.get("phase") != "15B_structure_features" or
            report.get("outcomes_loaded") is not False or
            report.get("test_outcomes_loaded") is not False):
        raise ValueError("Phase 15B summary does not certify label-free features")
    if not set(STRUCTURE_COLUMNS).issubset(report.get("structural_predictor_columns", [])):
        raise ValueError("Phase 15B manifest lacks prespecified predictors")
    refs = {
        "results/structure/phase15_coverage_summary.json": report.get("phase15a_summary_sha256"),
        "results/structure/phase15_coverage_positions.csv": report.get("phase15a_position_map_sha256"),
    }
    refs.update({f"data/external/structures/{pdb}.cif": h for pdb, h
                 in report.get("structures_sha256", {}).items()})
    if set(report.get("structures_sha256", {})) != {"6O2P", "8EIQ"}:
        raise ValueError("Missing experimental structure provenance")
    for rel, h in refs.items():
        if not h or not (root / rel).is_file() or sha256(root / rel) != h:
            raise ValueError(f"Structural provenance mismatch: {rel}")

    df = pd.read_csv(path)
    needed = {"protein_position", *STRUCTURE_COLUMNS}
    if not needed.issubset(df):
        raise ValueError(f"Structural columns missing: {sorted(needed-set(df))}")
    if df.protein_position.isna().any() or df.protein_position.duplicated().any():
        raise ValueError("Structural reference positions must be unique/non-null")
    if not df.protein_position.between(1, 1480).all():
        raise ValueError("Invalid canonical residue position")
    df = df[["protein_position", *STRUCTURE_COLUMNS]].copy()
    for col in STRUCTURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Nonnumeric structural feature: {col}")
        if np.isinf(df[col].to_numpy(dtype=float)).any():
            raise ValueError(f"Infinite structural descriptor: {col}")
    for col in STRUCTURE_COLUMNS[8:]:
        if df[col].isna().any() or not df[col].isin([0, 1]).all():
            raise ValueError(f"Malformed structural availability flag: {col}")
    expected = (df["6O2P_resolved_exact"] | df["8EIQ_resolved_exact"]).astype(int)
    if not df.has_exact_reference.eq(expected).all():
        raise ValueError("Structural resolution flags disagree")
    if df.loc[df.has_exact_reference.eq(0), list(DISTANCE_AND_GEOMETRY)].notna().any().any():
        raise ValueError("Unresolved positions contain reference geometry")
    return df


def structural_matrix(positions: np.ndarray, table: pd.DataFrame) -> np.ndarray:
    """Do not drop unresolved positions; retain NaNs for fold-only imputation."""
    positions = np.asarray(positions, dtype=int)
    if table.protein_position.duplicated().any():
        raise ValueError("Duplicated position in structural feature table")
    by_pos = table.set_index("protein_position")
    missing = sorted(set(positions) - set(by_pos.index))
    if missing:
        raise ValueError(f"Variant positions omitted from structural table: {missing[:8]}")
    result = by_pos.loc[positions, list(STRUCTURE_COLUMNS)].to_numpy(dtype=float)
    if np.isinf(result).any():
        raise ValueError("Non-finite (infinite) feature matrix")
    return result


def impute_fold(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median for each feature learned *only* from the training fold.

    An all-missing training column uses zero as a fixed fallback and remains
    present: train and test retain exactly the same feature dimensions.
    Availability flags (included separately) distinguish missing geometry.
    """
    a, b = np.asarray(train, float), np.asarray(test, float)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("Inconsistent 2D structural features")
    if a.shape[0] < 2 or np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("Invalid structural fold")
    replacements = np.array([
        float(np.median(a[np.isfinite(a[:, j]), j])) if np.isfinite(a[:, j]).any() else 0.0
        for j in range(a.shape[1])
    ])
    af = np.where(np.isnan(a), replacements[None, :], a)
    bf = np.where(np.isnan(b), replacements[None, :], b)
    if not np.isfinite(af).all() or not np.isfinite(bf).all():
        raise ValueError("Fold-local imputation yielded non-finite values")
    return af, bf


def shuffled_by_position(matrix: np.ndarray, positions: np.ndarray, seed: int = SHUFFLE_SEED) -> np.ndarray:
    """Fixed negative control: permute complete structural vectors among TRAIN positions.

    Different substitutions at one protein position receive the same bogus
    structural vector. No response measurements enter the permutation.
    """
    p = np.asarray(positions, dtype=int)
    x = np.asarray(matrix, dtype=float)
    if len(p) != len(x) or len(np.unique(p)) < 2:
        raise ValueError("Invalid position-level permutation")
    unique, inv = np.unique(p, return_inverse=True)
    representatives = np.stack([x[np.flatnonzero(inv == j)[0]] for j in range(len(unique))])
    for j in range(len(unique)):
        observed = x[inv == j]
        if not np.allclose(observed, representatives[j], equal_nan=True):
            raise ValueError("Structure differs between substitutions at same position")
    perm = np.random.default_rng(seed).permutation(len(unique))
    return representatives[perm][inv]


def validate_stored_forest(reference: pd.DataFrame, train: pd.DataFrame) -> None:
    """Fail closed on OOF coverage/fold leakage and unexpected train identity."""
    need = {"repeat", "fold", "model", "variant_key", "protein_position", "observed_pp", "predicted_pp"}
    if not need.issubset(reference):
        raise ValueError(f"Stored OOF missing columns: {sorted(need-set(reference))}")
    if not set(reference.model) == {"vti_forest"}:
        raise ValueError("OOF reference must contain only vti_forest")
    if reference.duplicated(["repeat", "variant_key"]).any():
        raise ValueError("Duplicate reference OOF predictions")
    if set(reference.repeat) != set(range(1, N_REPEATS+1)) or set(reference.fold) != set(range(1, N_FOLDS+1)):
        raise ValueError("Stored OOF repeat/fold indices changed")
    if len(reference) != N_REPEATS * len(train) or set(reference.variant_key) != set(train.variant_key):
        raise ValueError("Stored OOF does not cover precisely VTI frozen TRAIN")
    if reference.groupby("variant_key").repeat.nunique().ne(N_REPEATS).any():
        raise ValueError("Missing OOF repeat for a variant")
    if reference.groupby(["repeat", "variant_key"]).size().ne(1).any():
        raise ValueError("Nonunique OOF repeat/variant prediction")
    if reference.groupby(["repeat", "protein_position"]).fold.nunique().ne(1).any():
        raise ValueError("Same protein position appears in multiple held-out folds")
    if reference.groupby(["repeat", "fold"]).size().shape[0] != N_REPEATS * N_FOLDS:
        raise ValueError("Expected ten nonempty OOF folds")
    by_key = train.set_index("variant_key")
    if not np.array_equal(reference.protein_position.to_numpy(int),
                          reference.variant_key.map(by_key.protein_position).to_numpy(int)):
        raise ValueError("OOF positions disagree with frozen split")
    truth = reference.variant_key.map(by_key.vti_rescue_pp).to_numpy(float)
    if not np.allclose(reference.observed_pp.to_numpy(float), truth, atol=1e-6, rtol=0):
        raise ValueError("OOF truth disagrees with TRAIN-only current outcome values")
    if not np.isfinite(reference.predicted_pp.to_numpy(float)).all():
        raise ValueError("Non-finite original OOF predictions")


def position_scores(oof: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = SHUFFLE_SEED):
    """Two-repeat mean prediction for each variant, then equal-position scoring."""
    needed = {"repeat", "model", "variant_key", "protein_position", "study_cohort", "observed_pp", "predicted_pp"}
    if not needed.issubset(oof):
        raise ValueError(f"Missing predictions: {sorted(needed - set(oof))}")
    if set(oof.model) != set(MODELS) or set(oof.study_cohort) != set(COHORTS):
        raise ValueError("Missing model or VTI cohort")
    if oof.duplicated(["repeat", "variant_key", "model"]).any():
        raise ValueError("Duplicated model/repeat/variant")
    if oof.groupby(["variant_key", "model"]).repeat.nunique().ne(N_REPEATS).any():
        raise ValueError("Incomplete repeat coverage")
    if oof.groupby(["variant_key", "model"]).size().ne(N_REPEATS).any():
        raise ValueError("Incomplete model predictions")
    for key, group in oof.groupby("variant_key"):
        if group.protein_position.nunique() != 1 or group.study_cohort.nunique() != 1 or group.observed_pp.max()-group.observed_pp.min() > 1e-6:
            raise ValueError(f"Inconsistent truth/metadata for {key}")
    means = (oof.groupby(["model", "variant_key", "protein_position", "study_cohort"], as_index=False)
             .agg(observed_pp=("observed_pp", "mean"), predicted_pp=("predicted_pp", "mean")))
    means["abs_error_pp"] = (means.predicted_pp - means.observed_pp).abs()
    means["signed_error_pp"] = means.predicted_pp - means.observed_pp
    results = []
    for j, cohort in enumerate(("all", *COHORTS)):
        subset = means if cohort == "all" else means.loc[means.study_cohort.eq(cohort)]
        group = (subset.groupby(["protein_position", "model"])
                 .agg(error=("abs_error_pp", "mean"), bias=("signed_error_pp", "mean"))
                 .unstack("model"))
        if group.isna().any().any():
            raise ValueError(f"Unpaired positional comparisons in {cohort}")
        loss, bias = group["error"], group["bias"]
        base = loss["vti_forest"].to_numpy(float)
        shared = loss["shuffled_structure_forest"].to_numpy(float)
        draws = np.random.default_rng(seed + j).integers(0, len(base), size=(n_boot, len(base)))
        for model in MODELS:
            error = loss[model].to_numpy(float)
            delta = base-error
            shuffled_delta = shared-error
            ci = np.quantile(delta[draws].mean(axis=1), [.025, .975])
            ci_shuffle = np.quantile(shuffled_delta[draws].mean(axis=1), [.025, .975])
            results.append({
                "cohort": cohort, "model": model, "n_positions": int(len(loss)),
                "position_mae_pp": float(error.mean()),
                "mean_bias_pp": float(bias[model].mean()),
                "gain_vs_forest_pp": float(delta.mean()),
                "gain_vs_forest_ci95_pp": [float(v) for v in ci],
                "gain_vs_shuffled_pp": float(shuffled_delta.mean()),
                "gain_vs_shuffled_ci95_pp": [float(v) for v in ci_shuffle],
            })
    return pd.DataFrame(results), means


def threshold_metrics(means: pd.DataFrame) -> pd.DataFrame:
    """Raw >=10 pp assay threshold; never a calibrated responder probability."""
    rows = []
    for cohort in ("all", *COHORTS):
        sub = means if cohort == "all" else means.loc[means.study_cohort.eq(cohort)]
        for name in MODELS:
            frame = sub.loc[sub.model.eq(name)]
            truth = frame.observed_pp.to_numpy(float) >= THRESHOLD_PP
            scores = frame.predicted_pp.to_numpy(float)
            pred = scores >= THRESHOLD_PP
            per_pos_n = frame.groupby("protein_position").variant_key.transform("size").to_numpy(float)
            weights = 1 / per_pos_n
            if not (truth.any() and (~truth).any()):
                raise ValueError("Both threshold classes needed to assess discrimination")
            sens = float(weights[truth & pred].sum() / weights[truth].sum())
            spec = float(weights[~truth & ~pred].sum() / weights[~truth].sum())
            rows.append({
                "cohort": cohort, "model": name,
                "balanced_accuracy": (sens + spec)/2,
                "sensitivity_responder": sens,
                "specificity_nonresponder": spec,
                "auroc": float(roc_auc_score(truth, scores, sample_weight=weights)),
                "nonresponder_average_precision": float(average_precision_score(~truth, -scores, sample_weight=weights)),
                "false_responder_variants": int(np.sum((~truth) & pred)),
                "nonresponders_detected": int(np.sum((~truth) & (~pred))),
                "n_nonresponders": int(np.sum(~truth)),
            })
    return pd.DataFrame(rows)


def run() -> None:
    from cftr_theratype.benchmark import make_features
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.splits import parse_substitution
    from cftr_theratype.transfer import (ROOT, VTI_PAIRS, SPLIT_OUT, MANIFEST, RESULTS,
                                         SEED, _load_allowed_labels, _estimator)

    root = ROOT
    structure_path = root / "results/structure/phase15b_position_features.csv"
    structure_manifest = root / "results/structure/phase15b_feature_summary.json"
    ref_oof_path = RESULTS / "train_cv_predictions.csv"
    ref_summary_path = RESULTS / "summary.json"
    for path in (VTI_PAIRS, SPLIT_OUT, MANIFEST, ref_oof_path, ref_summary_path,
                 structure_path, structure_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    transfer_manifest = json.loads(MANIFEST.read_text())
    transfer_summary = json.loads(ref_summary_path.read_text())
    if (transfer_manifest.get("vti_source_sha256") != sha256(VTI_PAIRS)
            or transfer_summary.get("vti_source_sha256") != sha256(VTI_PAIRS)
            or transfer_summary.get("vti_partition_sha256") != sha256(SPLIT_OUT)):
        raise ValueError("Frozen transfer provenance mismatch; refuse retraining")
    structure = read_feature_table(structure_path, structure_manifest, root)

    # Identity-only membership, and strictly TRAIN-filtered assay outcomes.
    membership = pd.read_parquet(SPLIT_OUT, columns=[
        "variant_key", "protein_position", "transfer_split", "study_cohort",
        "variant", "variant_type", "hgvs_protein",
    ])
    if membership.variant_key.isna().any() or membership.variant_key.duplicated().any():
        raise ValueError("Nonunique/missing frozen VTI variant keys")
    train = membership.loc[membership.transfer_split.eq("train")].copy()
    if len(train) != 210 or train.protein_position.nunique() != 168:
        raise ValueError("Unexpected VTI training cohort size")
    if set(train.study_cohort) != set(COHORTS):
        raise ValueError("Unexpected VTI training cohort composition")
    keys = set(train.variant_key.astype(str))
    train = train.merge(_load_allowed_labels(VTI_PAIRS, keys, "vti_rescue_pp")[["variant_key", "vti_rescue_pp"]],
                        on="variant_key", validate="one_to_one")
    if len(train) != 210 or train.vti_rescue_pp.isna().any():
        raise ValueError("Missing train-only labels")
    seq = bihler_2024_assay_sequence()
    parsed = [parse_substitution(row, seq) for _, row in train.iterrows()]
    if any(p[-1] != "valid" or p[0] != pos for p, pos in zip(parsed, train.protein_position)):
        raise ValueError("Missense variants do not match experimental M470 reference")
    train["reference_aa"] = [p[1] for p in parsed]
    train["alternate_aa"] = [p[2] for p in parsed]
    train = train.sort_values("variant_key").reset_index(drop=True)
    Xseq, sequence_features = make_features(train, seq)
    Xseq = np.asarray(Xseq, dtype=float)
    if Xseq.shape != (210, 68) or not np.isfinite(Xseq).all():
        raise ValueError(f"Unexpected frozen sequence feature shape/values {Xseq.shape}")
    positions = train.protein_position.to_numpy(int)
    Xstruct = structural_matrix(positions, structure)
    Xshuffled = shuffled_by_position(Xstruct, positions)
    y = train.vti_rescue_pp.to_numpy(float)
    row_index = {str(k): i for i, k in enumerate(train.variant_key)}
    by_key = train.set_index("variant_key")
    raw = pd.read_csv(ref_oof_path)
    reference = raw.loc[raw.model.eq("vti_forest")].copy()
    validate_stored_forest(reference, train)
    reference["study_cohort"] = reference.variant_key.map(by_key.study_cohort)
    if reference.study_cohort.isna().any():
        raise ValueError("OOF reference cohort assignment failed")

    print("=" * 94)
    print("CFTR-THERATYPE | PHASE 15C STRUCTURE-AUGMENTED THERATYPING — TRAIN ONLY")
    print("=" * 94)
    print("[1] FROZEN COMPARISON")
    print("  Input: 68 original sequence features + 12 predeclared reference structural descriptors")
    print("  Model: same 250-tree, leaf-4 Forest, feature-fraction 0.75, identical seeds")
    print("  Baseline: original, saved VTI-only Forest OOF predictions")
    print("  Control: position-wise shuffled structural vectors (fixed seed)")
    print("  Fold median imputation: FIT ONLY on each training fold")
    print("  Both final tests and validation assays: NOT READ")
    covered = (structure.set_index("protein_position").loc[np.unique(positions), "has_exact_reference"].astype(int))
    print(f"  VTI training: {len(train)} variants / {len(np.unique(positions))} positions; "
          f"resolved reference environments {int(covered.sum())}/{len(covered)} ({covered.mean():.1%})")
    print("  Source structures: 6O2P (ivacaftor), 8EIQ (ETI); NOT vanzacaftor ligand complexes")

    outputs = []
    print("\n[2] STORED FOLDS + TRAIN-ONLY FITS")
    for (rep, fold), held in reference.groupby(["repeat", "fold"], sort=True):
        held_indices = np.array([row_index[k] for k in held.variant_key], dtype=int)
        held_positions = set(positions[held_indices])
        fit_indices = np.flatnonzero(~np.isin(positions, list(held_positions)))
        if set(positions[fit_indices]) & held_positions or not len(fit_indices):
            raise ValueError("Protein position leakage into training fold")
        if len(held_indices) + len(fit_indices) != len(train):
            raise ValueError("Held-out position fold does not cover whole development set")
        fold_seed = SEED + (int(rep)-1)*10 + int(fold)
        # Refit the *original* model to validate alignment with cached OOF.
        baseline = _estimator("vti_forest", fold_seed)
        baseline.fit(Xseq[fit_indices], y[fit_indices])
        baseline_pred = baseline.predict(Xseq[held_indices])
        if not np.allclose(baseline_pred, held.predicted_pp.to_numpy(float), rtol=0, atol=1e-6):
            raise ValueError(f"Frozen Forest not reproducible in repeat={rep}, fold={fold}; stop")
        for model, struct in (("structural_forest", Xstruct),
                              ("shuffled_structure_forest", Xshuffled)):
            tr, te = impute_fold(struct[fit_indices], struct[held_indices])
            augmented = _estimator("vti_forest", fold_seed)
            augmented.fit(np.column_stack([Xseq[fit_indices], tr]), y[fit_indices])
            preds = augmented.predict(np.column_stack([Xseq[held_indices], te]))
            for idx, pred in zip(held_indices, preds):
                outputs.append({
                    "repeat": int(rep), "fold": int(fold), "model": model,
                    "variant_key": str(train.iloc[idx].variant_key),
                    "protein_position": int(positions[idx]),
                    "study_cohort": str(train.iloc[idx].study_cohort),
                    "observed_pp": float(y[idx]), "predicted_pp": float(pred),
                })
        print(f"  repeat={rep}/2 fold={fold}/5; fit={len(fit_indices):3d} held={len(held_indices):3d} "
              f"| baseline reproduced", flush=True)

    combined = pd.concat([
        reference[["repeat", "fold", "model", "variant_key", "protein_position",
                   "study_cohort", "observed_pp", "predicted_pp"]],
        pd.DataFrame(outputs),
    ], ignore_index=True)
    summary, averaged = position_scores(combined)
    cls = threshold_metrics(averaged)
    print("\n[3] VTI RESCUE (POSITION-AVERAGED MAE, lower better)")
    print(f"  {'Cohort':10s} {'Model':28s} {'MAE':>7s} {'Bias':>8s} {'Gain':>8s} {'95% CI gain':>23s}")
    for row in summary.itertuples(index=False):
        lo, hi = row.gain_vs_forest_ci95_pp
        print(f"  {row.cohort:10s} {row.model:28s} {row.position_mae_pp:7.3f} "
              f"{row.mean_bias_pp:+8.3f} {row.gain_vs_forest_pp:+8.3f} [{lo:+9.3f},{hi:+9.3f}]")
    print("\n[4] STRUCTURAL INFORMATION vs SHUFFLED-STRUCTURE CONTROL")
    for row in summary.loc[summary.model.eq("structural_forest")].itertuples(index=False):
        lo, hi = row.gain_vs_shuffled_ci95_pp
        print(f"  {row.cohort:10s}: gain={row.gain_vs_shuffled_pp:+.3f} pp 95% CI [{lo:+.3f}, {hi:+.3f}]")
    print("\n[5] RAW >=10 PP RESPONSE THRESHOLD (TRAIN OOF, NOT EB RESPONSE LABEL)")
    print(f"  {'Cohort':10s} {'Model':28s} {'BA':>6s} {'Spec':>6s} {'AUROC':>7s} {'NR-AP':>7s} {'NR caught':>10s}")
    for row in cls.itertuples(index=False):
        print(f"  {row.cohort:10s} {row.model:28s} {row.balanced_accuracy:6.3f} "
              f"{row.specificity_nonresponder:6.3f} {row.auroc:7.3f} "
              f"{row.nonresponder_average_precision:7.3f} {row.nonresponders_detected:3d}/{row.n_nonresponders:<3d}")

    outdir = root / "results/structure"
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "phase15c_train_oof.csv"
    summary_path = outdir / "phase15c_summary.json"
    table_path = outdir / "phase15c_summary.csv"
    combined.to_csv(out_path, index=False)
    summary.to_csv(table_path, index=False)
    report = {
        "phase": "15C_structure_benchmark", "status": "exploratory_training_CV",
        "vti_source_sha256": sha256(VTI_PAIRS), "vti_partition_sha256": sha256(SPLIT_OUT),
        "baseline_oof_sha256": sha256(ref_oof_path), "baseline_summary_sha256": sha256(ref_summary_path),
        "structure_csv_sha256": sha256(structure_path),
        "structure_manifest_sha256": sha256(structure_manifest),
        "sequence_columns": list(sequence_features),
        "structural_columns": list(STRUCTURE_COLUMNS),
        "shuffled_control_seed": SHUFFLE_SEED, "cv_repeats": N_REPEATS, "cv_folds": N_FOLDS,
        "n_train_variants": len(train), "n_train_positions": len(np.unique(positions)),
        "n_positions_resolved": int(covered.sum()),
        "baseline_reproduced_per_fold": True,
        "rescue_position_metrics": summary.to_dict(orient="records"),
        "raw_10pp_threshold_metrics": cls.to_dict(orient="records"),
        "validation_or_test_outcomes_loaded": False,
        "cautions": [
            "No mutation-specific structural conformations or binding affinity predictions",
            "Reference geometry may act as a residue-position proxy rather than direct drug mechanism",
            "SASA calculated without membrane and without ligands",
            "The experimental ligands belong to 6O2P and 8EIQ; not vanzacaftor",
            "VTI test structural coverage is lower than VTI training coverage; no split excluded",
            "Missing structural features median-imputed using training-fold-only values",
            "Raw >=10 pp response is distinct from published empirical-Bayes categories",
            "Exploratory training CV; prior study/hyperparameter exploration limits confirmatory inference",
        ],
    }
    summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("\n[6] SAVED")
    for file in (out_path, table_path, summary_path):
        print(f"  {file.relative_to(root)}")
    print("PASS: Frozen folds and Forest reproduced; no validation/test outcomes accessed")
    print("=" * 94)


if __name__ == "__main__":
    run()
