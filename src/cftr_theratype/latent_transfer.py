"""Train-only, position-held-out biologically anchored ETI->VTI latent transfer.

Pre-registered exploratory candidates (no hyperparameter search):
  1. VTI-only low-rank regression (rank 4)
  2. Shared ETI/VTI response-head low-rank regression (rank 4)
  3. Same shared model plus FSK and FSK+IVA auxiliary heads, per study

Uses the *existing* 2x5 VTI position-fold protocol and archived VTI Forest OOF
predictions. Within each fold, ETI outcomes at held VTI positions and at all
frozen VTI val/test positions are never read. ETI 2024 val/test and all VTI
val/test outcomes are never loaded. No study-cohort label is an input feature.

This is multi-task representation learning, NOT adversarial or distributional
cohort alignment. Treatment/study are confounded, so separate heads cannot
identify a causal treatment effect. Repeated group CV is exploratory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cftr_theratype.transfer import (
    ROOT, RESULTS, MANIFEST, MEMBERSHIP, SPLIT_OUT, VTI_PAIRS,
    ETI_PAIRS, ETI_SPLITS, SEED, sha256, _feature_frame,
)

DIMENSION = 4
EPOCHS = 260
LEARNING_RATE = 0.02
WEIGHT_DECAY = 0.015
REPEATS = 2
N_FOLDS = 5
MODELS = ("vti_latent", "vti_anchored", "shared_latent", "anchored_latent")
VTI_COLS = ("vti_rescue_pp", "baseline_fsk_pct_wt", "iva_treated_pct_wt")
ETI_COLS = ("eti_rescue_pp", "baseline_fsk_pct_wt", "iva_treated_pct_wt")
HEAD_NAMES = ("eti_rescue", "vti_rescue", "eti_fsk", "vti_fsk", "eti_iva", "vti_iva")
# Sum of weights = 1 for each model; baseline/IVA tasks are auxiliary only.
LOSS_WEIGHTS = {
    "vti_latent":       {"vti_rescue": 1.0},
    "vti_anchored":     {"vti_rescue": 0.7, "vti_fsk": 0.15, "vti_iva": 0.15},
    "shared_latent":    {"eti_rescue": 0.5, "vti_rescue": 0.5},
    "anchored_latent":  {"eti_rescue": 0.35, "vti_rescue": 0.35,
                         "eti_fsk": 0.075, "vti_fsk": 0.075,
                         "eti_iva": 0.075, "vti_iva": 0.075},
}


def embargo_eti_positions(eti_positions, vti_held_positions, vti_protected_positions):
    """Explicitly reject ETI supervision at any held/protected VTI position."""
    all_blocked = set(map(int, vti_held_positions)) | set(map(int, vti_protected_positions))
    mask = ~np.isin(np.asarray(eti_positions, dtype=int), sorted(all_blocked))
    if set(np.asarray(eti_positions, dtype=int)[mask]) & all_blocked:
        raise AssertionError("ETI position embargo failed")
    return mask


def restricted_train_outcomes(path: Path, keys: set[str], columns: tuple[str, ...]) -> pd.DataFrame:
    """Predicate-pushdown limits responses to explicit, previously verified TRAIN keys."""
    if not keys:
        raise ValueError("Empty allowed training-label key set")
    requested = ["variant_key", *columns]
    frame = pd.read_parquet(path, engine="pyarrow", columns=requested,
                            filters=[("variant_key", "in", sorted(keys))])
    if frame.variant_key.isna().any() or frame.variant_key.duplicated().any():
        raise ValueError("Training outcomes contain invalid/duplicated variant IDs")
    if set(frame.variant_key) != keys:
        raise ValueError("Training outcomes incomplete or returned unexpected keys")
    if frame[list(columns)].isna().any().any():
        raise ValueError("Some training assay measurements are missing")
    if not np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all():
        raise ValueError("Non-finite training assay outcomes")
    return frame


def _normalization(x_fit: np.ndarray, x_query: np.ndarray):
    # Fit statistics only on fold TRAIN features; not held-out variant features.
    avg = x_fit.mean(axis=0)
    std = x_fit.std(axis=0)
    std[std < 1.e-7] = 1.
    return (x_fit - avg) / std, (x_query - avg) / std


def fit_latent(model_name: str, X_v, y_v, X_e, y_e, X_held, *, seed=0,
               rank=DIMENSION, epochs=EPOCHS):
    """Fit a linear low-rank multi-task representation and return VTI prediction.

    y_v/y_e: numpy (N,3) [rescue, DMSO FSK, DMSO FSK+IVA]. The auxiliary
    labels are training supervision only; held-out labels are never arguments.
    Shared encoder z = standardized_features @ W, followed by task-specific
    linear heads. The bottleneck enforces rank <= 4 (not a deep neural net).
    Every head is standardized to its *own* fold training target distribution.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch required: conda/pip install torch in CFTR_ENV") from exc
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    if rank < 1 or epochs < 1:
        raise ValueError("Invalid fixed training specification")
    X_v = np.asarray(X_v, dtype=np.float64)
    X_e = np.asarray(X_e, dtype=np.float64)
    X_held = np.asarray(X_held, dtype=np.float64)
    y_v = np.asarray(y_v, dtype=np.float64)
    y_e = np.asarray(y_e, dtype=np.float64)
    if X_v.ndim != 2 or X_e.ndim != 2 or X_held.ndim != 2 or X_v.shape[1] != X_e.shape[1] or X_held.shape[1] != X_v.shape[1]:
        raise ValueError("Incompatible feature dimensions")
    if y_v.shape != (len(X_v), 3) or y_e.shape != (len(X_e), 3):
        raise ValueError("Expected exactly three assay targets per study")
    if len(X_v) < 15 or (model_name in {"shared_latent", "anchored_latent"} and len(X_e) < 15):
        raise ValueError("Insufficient training data")
    if not all(np.isfinite(x).all() for x in (X_v, X_e, X_held, y_v, y_e)):
        raise ValueError("Inputs contain NaN/Inf")

    use_eti = model_name in {"shared_latent", "anchored_latent"}
    x_raw = np.vstack([X_v, X_e]) if use_eti else X_v
    x_norm, x_hold = _normalization(x_raw, X_held)
    # 6 target columns: study-specific heads; unobserved entries masked out.
    targets = np.zeros((len(x_raw), 6), dtype=np.float64)
    mask = np.zeros((len(x_raw), 6), dtype=np.float64)
    targets[:len(X_v), [1, 3, 5]] = y_v
    mask[:len(X_v), [1, 3, 5]] = 1.0
    if use_eti:
        targets[len(X_v):, [0, 2, 4]] = y_e
        mask[len(X_v):, [0, 2, 4]] = 1.0
    means = np.zeros(6)
    scales = np.ones(6)
    normalized = np.zeros_like(targets)
    weights = np.zeros(6)
    for j, name in enumerate(HEAD_NAMES):
        indices = mask[:, j].astype(bool)
        if not indices.any():
            continue
        means[j] = targets[indices, j].mean()
        scales[j] = max(targets[indices, j].std(), 1.0)
        normalized[indices, j] = (targets[indices, j] - means[j]) / scales[j]
        weights[j] = LOSS_WEIGHTS[model_name].get(name, 0.0)

    # Small deterministic CPU model; fixed seed and single-thread execution.
    torch.set_num_threads(1)
    torch.manual_seed(int(seed))
    dtype = torch.float32
    features = torch.as_tensor(x_norm, dtype=dtype)
    held = torch.as_tensor(x_hold, dtype=dtype)
    y = torch.as_tensor(normalized, dtype=dtype)
    m = torch.as_tensor(mask, dtype=dtype)
    w = torch.as_tensor(weights, dtype=dtype)
    observed_count = m.sum(dim=0).clamp_min(1.)
    network = torch.nn.Sequential(
        torch.nn.Linear(X_v.shape[1], rank, bias=False),
        torch.nn.Linear(rank, 6, bias=True),
    )
    with torch.no_grad():
        network[1].weight.mul_(0.1)
        network[1].bias.zero_()
    optimizer = torch.optim.Adam(network.parameters(), lr=LEARNING_RATE)
    for _ in range(epochs):
        optimizer.zero_grad()
        outputs = network(features)
        per_head = ((outputs - y).square() * m).sum(dim=0) / observed_count
        regularizer = sum(p.square().sum() for p in network.parameters() if p.ndim >= 2)
        # Regressor weights regularized; biases are unpenalized.
        loss = (per_head * w).sum() + WEIGHT_DECAY * regularizer
        if not torch.isfinite(loss):
            raise ValueError("Latent optimization diverged")
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred_norm = network(held).numpy().astype(float)
    # Always use the VTI rescue prediction. Auxiliary predictions are only
    # meaningful for anchored_latent (other heads received zero loss weight).
    out = pred_norm[:, [1, 3, 5]] * scales[[1, 3, 5]] + means[[1, 3, 5]]
    if not np.isfinite(out).all():
        raise ValueError("Nonfinite latent predictions")
    if model_name not in {"vti_anchored", "anchored_latent"}:
        out[:, 1:] = np.nan
    return out


def group_metrics(oof: pd.DataFrame, *, seed=SEED, bootstrap=3000):
    """Average 2 repeats per variant first, then position-balanced errors."""
    required = {"model", "repeat", "variant_key", "protein_position", "study_cohort", "observed_pp", "predicted_pp"}
    if not required.issubset(oof):
        raise ValueError(f"Missing columns: {sorted(required-set(oof))}")
    if oof.duplicated(["model", "repeat", "variant_key"]).any():
        raise ValueError("Duplicate OOF rows")
    if set(oof.repeat.unique()) != {1, 2}:
        raise ValueError("Expected 2 CV repeats")
    if oof.groupby(["model", "variant_key"]).repeat.nunique().ne(2).any():
        raise ValueError("Incomplete CV repeat coverage")
    by_key = oof.groupby("variant_key", sort=False)
    if (by_key.protein_position.nunique().ne(1).any() or
            by_key.study_cohort.nunique().ne(1).any() or
            (by_key.observed_pp.max() - by_key.observed_pp.min()).abs().gt(1e-5).any()):
        raise ValueError("Inconsistent cohort, mutation position or target across models")
    if not np.isfinite(oof[["observed_pp", "predicted_pp"]].to_numpy(float)).all():
        raise ValueError("Nonfinite predictions/targets")
    by_variant = oof.groupby(["model", "variant_key", "protein_position", "study_cohort"], as_index=False).agg(
        observed_pp=("observed_pp", "mean"), predicted_pp=("predicted_pp", "mean"))
    by_variant["abs_error_pp"] = (by_variant.predicted_pp - by_variant.observed_pp).abs()
    by_variant["signed_error_pp"] = by_variant.predicted_pp - by_variant.observed_pp
    models = ("vti_forest", *MODELS)
    if set(by_variant.model) != set(models):
        raise ValueError("Reference/latent models missing")
    groups = ["all", "ETI, VTI", "VTI"]
    summary = []
    for grp in groups:
        sub = by_variant if grp == "all" else by_variant.loc[by_variant.study_cohort == grp]
        p = sub.groupby(["protein_position", "model"], as_index=False)[["abs_error_pp", "signed_error_pp"]].mean()
        errors = p.pivot(index="protein_position", columns="model", values="abs_error_pp")
        signed = p.pivot(index="protein_position", columns="model", values="signed_error_pp")
        if errors.isna().any().any() or signed.isna().any().any() or set(errors.columns) != set(models):
            raise ValueError("Incomplete paired positions by group")
        if len(errors) < 5:
            raise ValueError("Insufficient positions")
        reference = errors["vti_forest"].to_numpy(float)
        rng = np.random.default_rng(seed + groups.index(grp))
        draws = rng.integers(0, len(errors), size=(bootstrap, len(errors)))
        for name in models:
            candidate = errors[name].to_numpy(float)
            gains = reference - candidate
            lo, hi = np.quantile(gains[draws].mean(axis=1), [0.025, 0.975])
            summary.append({"cohort": grp, "model": name, "n_positions": len(errors),
                            "position_mae_pp": float(candidate.mean()),
                            "signed_bias_pp": float(signed[name].mean()),
                            "gain_vs_forest_pp": float(gains.mean()),
                            "gain_ci_low_pp": float(lo), "gain_ci_high_pp": float(hi)})
    return pd.DataFrame(summary), by_variant


def _load_and_check_training():
    """Only frozen *membership* is read until allowed training keys established."""
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.splits import parse_substitution
    protocol = json.loads(MANIFEST.read_text())
    required_hashes = {"vti_source_sha256": sha256(VTI_PAIRS),
                       "membership_sha256": sha256(MEMBERSHIP),
                       "eti_split_sha256": sha256(ETI_SPLITS)}
    if any(protocol.get(k) != v for k, v in required_hashes.items()):
        raise ValueError("Frozen source fingerprints do not match transfer manifest")
    membership = pd.read_parquet(SPLIT_OUT, columns=[
        "variant_key", "protein_position", "transfer_split", "study_cohort",
        "variant", "variant_type", "hgvs_protein",
    ])
    if membership.variant_key.isna().any() or membership.variant_key.duplicated().any():
        raise ValueError("VTI membership variant_key invalid")
    if membership.groupby("protein_position", dropna=True).transfer_split.nunique().max() > 1:
        raise ValueError("VTI protein position crosses split partitions")
    vti = membership.loc[membership.transfer_split.eq("train")].copy()
    protected = set(membership.loc[membership.transfer_split.isin(["val", "test"]), "protein_position"].dropna().astype(int))
    if set(vti.protein_position.astype(int)) & protected:
        raise ValueError("VTI train overlaps protected VTI positions")
    if len(vti) != 210 or vti.protein_position.nunique() != 168:
        raise ValueError("Frozen VTI train does not match checkpoint")
    labels = restricted_train_outcomes(VTI_PAIRS, set(vti.variant_key), VTI_COLS)
    vti = vti.merge(labels, on="variant_key", validate="one_to_one")
    sequence = bihler_2024_assay_sequence()
    parsed = [parse_substitution(r, sequence) for _, r in vti.iterrows()]
    if any(p[3] != "valid" or int(p[0]) != int(pos)
           for p, pos in zip(parsed, vti.protein_position)):
        raise ValueError("VTI missense/reference mismatch")
    vti["reference_aa"] = [p[1] for p in parsed]
    vti["alternate_aa"] = [p[2] for p in parsed]
    vti = vti.sort_values("variant_key").reset_index(drop=True)

    eti_membership = pd.read_parquet(ETI_SPLITS, columns=[
        "variant_key", "protein_position", "split", "reference_aa", "alternate_aa",
    ])
    if eti_membership.variant_key.isna().any() or eti_membership.variant_key.duplicated().any():
        raise ValueError("ETI variant_key invalid")
    if eti_membership.groupby("protein_position", dropna=True).split.nunique().max() > 1:
        raise ValueError("ETI positions are not disjoint between splits")
    eti = eti_membership.loc[eti_membership.split.eq("train")].copy()
    # Strong embargo applies to ETI development if protected in VTI protocol.
    eti = eti.loc[~eti.protein_position.isin(protected)].copy()
    eti = eti.sort_values("variant_key").reset_index(drop=True)
    if set(eti.protein_position.astype(int)) & protected:
        raise AssertionError("ETI protected-position exclusion failed")
    Xv = _feature_frame(vti, sequence)
    Xe = _feature_frame(eti, sequence)
    yv = vti[list(VTI_COLS)].to_numpy(float)
    return vti, eti, Xv, Xe, yv, protected


def _load_reference_oof(vti, prior_path: Path):
    summary_path = RESULTS / "summary.json"
    summary = json.loads(summary_path.read_text())
    if (summary.get("vti_source_sha256") != sha256(VTI_PAIRS) or
            summary.get("vti_partition_sha256") != sha256(SPLIT_OUT) or
            summary.get("eti_split_sha256") != sha256(ETI_SPLITS)):
        raise ValueError("Existing Forest OOF does not match frozen data")
    original = pd.read_csv(prior_path)
    original = original.loc[original.model.eq("vti_forest")].copy()
    keys = set(vti.variant_key)
    if set(original.variant_key) != keys or original.groupby("variant_key").repeat.nunique().ne(REPEATS).any():
        raise ValueError("Original Forest OOF lacks full frozen training coverage")
    if original.duplicated(["repeat", "variant_key"]).any() or set(original.repeat.unique()) != {1, 2}:
        raise ValueError("Duplicate/invalid original Forest folds")
    by_key = vti.set_index("variant_key")
    if not np.allclose(original.observed_pp.to_numpy(float), original.variant_key.map(by_key.vti_rescue_pp), atol=1e-5, rtol=0):
        raise ValueError("Original Forest OOF observed targets do not match study")
    if not np.array_equal(original.protein_position.to_numpy(int), original.variant_key.map(by_key.protein_position).to_numpy(int)):
        raise ValueError("Original Forest OOF protein positions changed")
    original["study_cohort"] = original.variant_key.map(by_key.study_cohort)
    if original.study_cohort.isna().any():
        raise ValueError("Original Forest OOF lacks cohort metadata")
    return original[["repeat", "fold", "model", "variant_key", "protein_position", "study_cohort", "observed_pp", "predicted_pp"]]


def run_experiment(*, smoke: bool = False):
    from cftr_theratype.cv import position_folds
    print("=" * 88)
    print("CFTR-THERATYPE | BIOLOGICALLY ANCHORED MULTI-TASK LATENT TRANSFER")
    print("=" * 88)
    print("[1] FROZEN EXPERIMENT")
    print(f"Rank={DIMENSION}; epochs={EPOCHS}; loss weights={json.dumps(LOSS_WEIGHTS)}")
    print("Sequence-only at inference; treatment-specific heads; per-study functional auxiliaries")
    print("Study membership/response group NOT model features; neither final test ever opened")
    vti, eti, Xv, Xe, yv, protected = _load_and_check_training()
    reference = _load_reference_oof(vti, RESULTS / "train_cv_predictions.csv")
    pv = vti.protein_position.to_numpy(int)
    pe = eti.protein_position.to_numpy(int)
    print(f"VTI train={len(vti)}/{len(set(pv))} positions; ETI candidate train={len(eti)}/{len(set(pe))} positions")
    rows = []
    for repeat in range(REPEATS):
        for fold, tr, held in position_folds(pv, seed=SEED + 100 + repeat):
            withheld = set(pv[held])
            allowed = embargo_eti_positions(pe, withheld, protected)
            allowed_meta = eti.loc[allowed, ["variant_key", "protein_position"]]
            # STRICT: load labels from parquet ONLY for this fold's allowed ETI keys.
            outcomes = restricted_train_outcomes(ETI_PAIRS, set(allowed_meta.variant_key), ETI_COLS)
            allowed_meta = allowed_meta.merge(outcomes, on="variant_key", validate="one_to_one")
            ye = allowed_meta[list(ETI_COLS)].to_numpy(float)
            xe = Xe[allowed]
            if len(xe) != len(ye):
                raise ValueError("ETI label/feature row misalignment")
            # `allowed_meta` order follows left metadata order, due to stable merge.
            if list(allowed_meta.variant_key) != list(eti.loc[allowed, "variant_key"]):
                raise ValueError("ETI label/feature alignment mismatch")
            if set(allowed_meta.protein_position.astype(int)) & (withheld | protected):
                raise ValueError("ETI fold supervision leaks into held-out VTI positions")
            fold_reference = reference.loc[(reference.repeat.eq(repeat + 1)) & (reference.fold.eq(fold))]
            if set(fold_reference.variant_key) != set(vti.loc[held, "variant_key"]):
                raise ValueError("Current grouped folds differ from archived Forest folds")
            print(f"  Repeat {repeat+1}/{REPEATS}, fold {fold}/{N_FOLDS} | "
                  f"VTI train={tr.sum():3d}, held={held.sum():3d}; ETI allowed={allowed.sum():3d}", flush=True)
            for name in MODELS:
                pred = fit_latent(name, Xv[tr], yv[tr], xe, ye, Xv[held],
                                  seed=SEED + repeat * 100 + fold * 10,
                                  epochs=12 if smoke else EPOCHS)
                for idx, p in zip(np.flatnonzero(held), pred):
                    rows.append({"repeat": repeat + 1, "fold": int(fold), "model": name,
                                 "variant_key": str(vti.iloc[idx].variant_key),
                                 "protein_position": int(pv[idx]),
                                 "study_cohort": str(vti.iloc[idx].study_cohort),
                                 "observed_pp": float(yv[idx, 0]),
                                 "predicted_pp": float(p[0]),
                                 "observed_fsk_pp": float(yv[idx, 1]),
                                 "predicted_fsk_pp": float(p[1]),
                                 "observed_iva_pp": float(yv[idx, 2]),
                                 "predicted_iva_pp": float(p[2])})
            if smoke:
                print("Smoke check passed: fold supervision and latent fit completed; no research results saved")
                return

    candidates = pd.DataFrame(rows)
    if candidates.groupby(["model", "variant_key"]).repeat.nunique().ne(REPEATS).any():
        raise ValueError("Incomplete prediction coverage")
    combined = pd.concat([reference, candidates], ignore_index=True)
    summary, per_variant = group_metrics(combined)
    print("\n[2] POSITION-BALANCED CV, TRAIN ONLY (+ gain means better than Forest)")
    print(f"{'Cohort':11s} {'Model':20s} {'MAE':>8s} {'Bias':>9s} {'Gain':>9s} {'95% CI for gain':>23s}")
    for r in summary.itertuples(index=False):
        print(f"{r.cohort:11s} {r.model:20s} {r.position_mae_pp:8.3f} "
              f"{r.signed_bias_pp:+9.3f} {r.gain_vs_forest_pp:+9.3f} "
              f"[{r.gain_ci_low_pp:+8.3f}, {r.gain_ci_high_pp:+8.3f}]")
    print("\n[3] CRITICAL PAIRED ABLATIONS (positive means improvement)")
    for reference_name, candidate_name, label in (
        ("shared_latent", "anchored_latent", "biology added to joint training"),
        ("vti_anchored", "anchored_latent", "joint ETI/VTI vs VTI-only biological tasks"),
        ("vti_latent", "shared_latent", "ETI rescue sharing without auxiliaries"),
    ):
        print(f"  {label}:")
        for cohort in ("all", "ETI, VTI", "VTI"):
            chosen = per_variant if cohort == "all" else per_variant.loc[
                per_variant.study_cohort.eq(cohort)]
            per_pos = chosen.groupby(["protein_position", "model"]).abs_error_pp.mean().unstack()
            delta = (per_pos[reference_name] - per_pos[candidate_name]).to_numpy(float)
            rng = np.random.default_rng(SEED + 90 + (0 if cohort == "all" else (1 if cohort == "ETI, VTI" else 2)))
            sampled = rng.integers(0, len(delta), size=(3000, len(delta)))
            low, high = np.quantile(delta[sampled].mean(axis=1), [.025, .975])
            print(f"    {cohort:9s} | gain={delta.mean():+.3f} pp "
                  f"95% CI [{low:+.3f}, {high:+.3f}]")
    print("\n[4] ANCHORED LATENT AUXILIARY PREDICTIONS (diagnostic; VTI held fold)")
    anchored = candidates.loc[candidates.model.eq("anchored_latent")].copy()
    aux = anchored.groupby(["variant_key", "protein_position"], as_index=False).agg(
        observed_fsk_pp=("observed_fsk_pp", "mean"),
        predicted_fsk_pp=("predicted_fsk_pp", "mean"),
        observed_iva_pp=("observed_iva_pp", "mean"),
        predicted_iva_pp=("predicted_iva_pp", "mean"))
    for stem in ("fsk", "iva"):
        aux[f"abs_{stem}"] = (aux[f"predicted_{stem}_pp"] - aux[f"observed_{stem}_pp"]).abs()
        print(f"  VTI {stem.upper()} position MAE: {aux.groupby('protein_position')[f'abs_{stem}'].mean().mean():.3f} pp")
    print("\n[5] CORRECTNESS: no VTI val/test, ETI val/test or held-fold ETI outcomes loaded")
    print("Auxiliary outputs are predictions from sequence, not held-out assay inputs")
    print("Important: this is fixed training-only exploratory CV, NOT external validation")
    RESULTS.mkdir(parents=True, exist_ok=True)
    combined.to_csv(RESULTS / "latent_transfer_oof.csv", index=False)
    summary.to_csv(RESULTS / "latent_transfer_summary.csv", index=False)
    report = {
        "hypothesis": "shared functional phenotypes improve cross-treatment generalisation",
        "method": "linear low-rank shared sequence encoder with study/treatment heads",
        "rank": DIMENSION, "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "loss_weights": LOSS_WEIGHTS,
        "fold_protocol": "2x5 grouped CV, identical positions as frozen Forest OOF",
        "source_sha256": sha256(VTI_PAIRS), "partition_sha256": sha256(SPLIT_OUT),
        "eti_splits_sha256": sha256(ETI_SPLITS),
        "models": list(MODELS), "results": summary.to_dict(orient="records"),
        "vti_validation_test_used": False, "eti_validation_test_used": False,
        "limitations": ["Study and treatment confounded", "small response-selected overlapping cohort",
                        "auxiliary labels partially algebraically related to rescue",
                        "exploratory repeated CV; no model selection on test"]}
    (RESULTS / "latent_transfer_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved: results/transfer/latent_transfer_oof.csv")
    print("Saved: results/transfer/latent_transfer_summary.csv")
    print("Saved: results/transfer/latent_transfer_summary.json")
    print("=" * 88)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="One fold, 12 epochs; no outputs")
    args = parser.parse_args()
    run_experiment(smoke=args.smoke)


if __name__ == "__main__":
    main()
