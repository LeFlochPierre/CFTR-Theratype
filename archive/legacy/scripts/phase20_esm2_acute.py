#!/usr/bin/env python3
"""CFTR-Theratype Phase 20: frozen ESM-2, position-disjoint acute IVA CV.

A. Extract site and local-window ESM-2 embeddings for exactly the 210 frozen
   VTI TRAIN variants, using a 511-AA window of assay-reference CFTR (M470).
B. Compare the exact frozen Phase 19C sequence+structure XGBoost baseline
   against a small, INNER-CV-selected ESM-PCA augmentation in the identical
   2x5 OUTER position-disjoint folds.

Never accesses protected VTI/ETI validation/test assay outcomes. Neural model
is frozen; PCA, imputation and regression are fitted strictly inside CV folds.

Commands:
  python scripts/phase20_esm2_acute.py extract --device cuda --batch-size 4
  python scripts/phase20_esm2_acute.py benchmark --smoke
  python scripts/phase20_esm2_acute.py benchmark

WARNING: The best-of-multiple-ESM-representations selection is exploratory;
outer-fold predictions are valid held-out estimates for an inner-selected
pipeline, but the final best model must be validated independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from cftr_theratype import model_zoo as mz
from cftr_theratype.transfer import ROOT
from cftr_theratype.reference import bihler_2024_assay_sequence

OUT = ROOT / "results/boltz/phase20"
EMBED = OUT / "esm2_vti_train_210.npz"
MODEL_ID = "facebook/esm2_t12_35M_UR50D"
WINDOW = 511
SEEDS = (11, 22, 33)  # unused by frozen ESM-2, retained for provenance context
TARGET = "acute_iva_vti_pp"
REPS = ("site_delta", "window_delta", "site_wt_delta")
N_PCS = (8, 16, 32)
BASELINE = "sequence_structure__xgboost"
SEED_OFFSET = 20261027


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def position_mae(y, pred, position):
    y = np.asarray(y, float)
    pred = np.asarray(pred, float)
    if not (len(y) == len(pred) == len(position)):
        raise ValueError("Invalid MAE vectors")
    d = pd.DataFrame({"pos": position, "err": np.abs(y - pred)})
    return float(d.groupby("pos").err.mean().mean())


def frozen_train():
    train, mats, _, outer = mz.prepare(ROOT)
    if len(train) != 210 or train.protein_position.nunique() != 168 or len(outer) != 10:
        raise ValueError("Frozen VTI TRAIN size/positions/folds unexpectedly changed")
    if train.variant_key.duplicated().any():
        raise ValueError("Duplicate variant keys in frozen TRAIN")
    seq = bihler_2024_assay_sequence().upper()
    if len(seq) < 1400 or len(seq) > 1600:
        raise ValueError("Unexpected assay reference sequence length")
    required = {"variant_key", "protein_position", "reference_aa", "alternate_aa"}
    if not required.issubset(train.columns):
        raise ValueError(f"Required frozen metadata columns absent: {required - set(train.columns)}")
    for r in train.itertuples(index=False):
        p = int(r.protein_position)
        if not (1 <= p <= len(seq)) or seq[p - 1] != str(r.reference_aa).upper() or len(str(r.alternate_aa)) != 1:
            raise ValueError(f"Mutation/reference mismatch at {r.variant_key} position {p}")
    return train, mats, outer, seq


def meta_fingerprint(train, seq):
    records = [
        [str(r.variant_key), int(r.protein_position), str(r.reference_aa), str(r.alternate_aa)]
        for r in train.itertuples(index=False)
    ]
    encoded = json.dumps(records, ensure_ascii=True, separators=(",", ":"))
    return {
        "variant_metadata_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
        "assay_reference_sha256": hashlib.sha256(seq.encode()).hexdigest(),
    }


def window_for_position(seq: str, pos: int):
    """Returns fixed 511 AA substring and mutation offset, 0-based."""
    center = pos - 1
    start = max(0, min(center - WINDOW // 2, len(seq) - WINDOW))
    frag = seq[start:start + WINDOW]
    relative = center - start
    if len(frag) != WINDOW or frag[relative] != seq[center]:
        raise ValueError(f"Window/reference mismatch at {pos}")
    return frag, relative


def extract(args):
    import torch
    from transformers import AutoModel, AutoTokenizer

    train, _, _, seq = frozen_train()
    OUT.mkdir(parents=True, exist_ok=True)
    if EMBED.exists() and not args.overwrite:
        raise FileExistsError(f"Cache exists, refusing overwrite: {EMBED}; use --overwrite intentionally")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu (slower) or repair torch")

    print("=" * 92, flush=True)
    print("PHASE 20A | FROZEN ESM-2 CFTR MUTATION FEATURE EXTRACTION", flush=True)
    print("=" * 92, flush=True)
    print(f"Model: {MODEL_ID} | Window: {WINDOW} AA | Variants: {len(train)}", flush=True)
    print(f"Device: {args.device} | Torch: {torch.__version__}", flush=True)
    print("TRAIN ONLY: no held-out mutation outcomes read", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID).to(args.device).eval()
    if int(model.config.hidden_size) != 480:
        raise ValueError(f"Expected 480-dimensional ESM-2, got {model.config.hidden_size}")

    # Deduplicate reference windows by mutation position; variants sharing the
    # same protein position share WT embedding but retain distinct mutant ones.
    jobs = []
    for position in sorted(set(train.protein_position.astype(int))):
        frag, offset = window_for_position(seq, position)
        jobs.append(("wt", position, frag, offset))
    for r in train.itertuples(index=False):
        p = int(r.protein_position)
        frag, offset = window_for_position(seq, p)
        alt = str(r.alternate_aa).upper()
        if alt in {"*", "X", "-"}:
            raise ValueError(f"Not a missense substitution: {r.variant_key}")
        mutant = frag[:offset] + alt + frag[offset + 1:]
        if mutant[offset] != alt or len(mutant) != len(frag):
            raise ValueError("Bad mutant sequence creation")
        jobs.append(("mut", str(r.variant_key), mutant, offset))

    wt = {}
    mut = {}
    batch_size = max(1, int(args.batch_size))
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        encoded = tokenizer(
            [r[2] for r in batch], return_tensors="pt", padding=True,
            add_special_tokens=True, truncation=False,
        )
        seq_len = int(encoded["input_ids"].shape[1])
        if seq_len != WINDOW + 2:
            raise ValueError(f"ESM tokenization unexpectedly maps {WINDOW} AA to {seq_len} tokens")
        encoded = {k: v.to(args.device) for k, v in encoded.items()}
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state.detach().float().cpu().numpy()
        for j, (kind, key, _sequence, offset) in enumerate(batch):
            site = np.asarray(output[j, offset + 1], dtype=np.float32).copy()
            mean = np.asarray(output[j, 1:WINDOW + 1].mean(axis=0), dtype=np.float32).copy()
            if site.shape != (480,) or mean.shape != (480,):
                raise ValueError("Bad embedding shape")
            if not (np.isfinite(site).all() and np.isfinite(mean).all()):
                raise ValueError("ESM produced nonfinite embeddings")
            (wt if kind == "wt" else mut)[key] = (site, mean)
        if start == 0 or (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(jobs):
            print(f"ESM progress: {min(start+batch_size, len(jobs))}/{len(jobs)} sequences", flush=True)

    wt_site, mut_site, wt_mean, mut_mean = [], [], [], []
    for r in train.itertuples(index=False):
        a, b = wt[int(r.protein_position)]
        c, d = mut[str(r.variant_key)]
        wt_site.append(a); wt_mean.append(b); mut_site.append(c); mut_mean.append(d)
    wt_site, wt_mean, mut_site, mut_mean = (
        np.stack(x).astype(np.float32)
        for x in (wt_site, wt_mean, mut_site, mut_mean)
    )
    arrays = {
        "site_delta": mut_site - wt_site,
        "window_delta": mut_mean - wt_mean,
        "site_wt_delta": np.column_stack((wt_site, mut_site - wt_site)),
    }
    for name, a in arrays.items():
        if a.shape[0] != 210 or not np.isfinite(a).all():
            raise ValueError(f"Invalid {name}: {a.shape}")
        print(f"{name:16s} shape={a.shape} median_row_norm={np.median(np.linalg.norm(a,axis=1)):.4f}", flush=True)

    metadata = {
        "phase": "20A_ESM2_frozen_VTI_TRAIN_only",
        "model_id": MODEL_ID,
        "model_revision_hash": getattr(model.config, "_commit_hash", None),
        "window_aa": WINDOW,
        "sequence_context": "Bihler 2024 assay CFTR reference (M470), centered 511 AA window",
        "aggregation": "single mutation-site token and mean local residue embeddings",
        "n_variants": 210,
        "n_positions": 168,
        "no_heldout_outcomes_read": True,
        "python": platform.python_version(),
        "torch": torch.__version__,
        **meta_fingerprint(train, seq),
    }
    # Atomic replacement to avoid leaving a partial cache on interrupted runs.
    temp = OUT / "esm2_vti_train_210.partial.npz"
    np.savez_compressed(
        temp,
        keys=train.variant_key.to_numpy(dtype=str),
        positions=train.protein_position.to_numpy(int),
        metadata=json.dumps(metadata, sort_keys=True),
        **arrays,
    )
    temp.replace(EMBED)
    print(f"PASS: {len(jobs)} frozen ESM sequences; 210 matched training variants", flush=True)
    print(f"Saved: {EMBED.relative_to(ROOT)}", flush=True)
    print(f"SHA256: {sha256(EMBED)}", flush=True)


def load_embeddings(train, seq):
    if not EMBED.exists():
        raise FileNotFoundError(f"ESM embeddings missing; run 'extract' first: {EMBED}")
    with np.load(EMBED, allow_pickle=False) as obj:
        keys = obj["keys"].astype(str)
        positions = obj["positions"].astype(int)
        meta = json.loads(str(obj["metadata"].item()))
        arrays = {name: np.asarray(obj[name], np.float64) for name in REPS}
    if not np.array_equal(keys, train.variant_key.astype(str).to_numpy()):
        raise ValueError("ESM key order differs from frozen training metadata")
    if not np.array_equal(positions, train.protein_position.to_numpy(int)):
        raise ValueError("ESM position order differs from frozen training metadata")
    expected = meta_fingerprint(train, seq)
    for name, value in expected.items():
        if meta.get(name) != value:
            raise ValueError(f"ESM provenance mismatch: {name}")
    if meta.get("model_id") != MODEL_ID or meta.get("window_aa") != WINDOW or meta.get("n_variants") != 210:
        raise ValueError("ESM model/window/size mismatch")
    for name, array in arrays.items():
        expected_dims = 960 if name == "site_wt_delta" else 480
        if array.shape != (210, expected_dims) or not np.isfinite(array).all():
            raise ValueError(f"Invalid cached ESM features: {name} {array.shape}")
    return arrays, meta


def fold_augment(x_static_train, x_static_eval, emb_train, emb_eval, k, seed):
    """Fit ESM preprocessing strictly on *this* training fold."""
    e_scaler = StandardScaler()
    a = e_scaler.fit_transform(emb_train)
    b = e_scaler.transform(emb_eval)
    # Randomized PCA for speed; no outcome-dependent feature selection.
    reducer = PCA(n_components=int(k), svd_solver="randomized", random_state=seed)
    az = reducer.fit_transform(a)
    bz = reducer.transform(b)
    if not np.isfinite(az).all() or not np.isfinite(bz).all():
        raise ValueError("PCA emitted non-finite features")
    return np.column_stack((x_static_train, az)), np.column_stack((x_static_eval, bz))


def select_esm(x_static, y, positions, arrays, params, seed, *, smoke=False):
    """Choose representation/PCA size using INNER held-out positions only.

    Compute 32-PC train-fitted projection once per representation/inner split,
    then reuse its leading 8/16/32 PCs. No outer-heldout examples influence PCA.
    """
    inner = list(GroupKFold(n_splits=3).split(x_static, y, groups=positions))
    dimensions = N_PCS[:1] if smoke else N_PCS
    best, best_score, details = None, float("inf"), []
    n_static = int(x_static.shape[1])
    for representation in REPS:
        emb = arrays[representation]
        predictions = {k: np.full(len(y), np.nan) for k in dimensions}
        for ii, (fit, held) in enumerate(inner, 1):
            if set(positions[fit]) & set(positions[held]):
                raise ValueError("Inner position leakage")
            # PCA trained on this inner-fit partition, never on held-out rows.
            a_all, b_all = fold_augment(
                x_static[fit], x_static[held], emb[fit], emb[held],
                max(N_PCS), seed + ii)
            for k in dimensions:
                a = a_all[:, :n_static + k]
                b = b_all[:, :n_static + k]
                reg = mz.estimator("xgboost", params, seed + ii)
                reg.fit(a, y[fit])
                predictions[k][held] = np.asarray(reg.predict(b), float)
        for k, pred in predictions.items():
            if not np.isfinite(pred).all():
                raise ValueError("Incomplete inner CV predictions")
            score = position_mae(y, pred, positions)
            details.append({"representation": representation,
                            "n_components": int(k),
                            "inner_position_mae_pp": score})
            if score < best_score - 1e-10:
                best, best_score = (representation, k), score
    return best, float(best_score), details


def paired_bootstrap(avg, comparison, control, *, n_boot=20_000):
    test = avg.loc[avg.model.isin([comparison, control])].copy()
    unique = test.groupby(["model", "variant_key"]).size()
    if not unique.eq(1).all():
        raise ValueError("Duplicate OOF keys")
    test["error"] = (test.observed_pp - test.predicted_pp).abs()
    wide = test.groupby(["protein_position", "model"]).error.mean().unstack()
    diff = (wide[control] - wide[comparison]).to_numpy(float)
    if len(diff) != 168 or not np.isfinite(diff).all():
        raise ValueError("Incomplete position-level bootstrap")
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(diff), (n_boot, len(diff)))
    bs = diff[idx].mean(axis=1)
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return {"gain_pp": float(diff.mean()), "ci95_low": float(lo), "ci95_high": float(hi),
            "positions_improved": int((diff > 0).sum()),
            "fraction_positive": float((bs > 0).mean())}


def benchmark(args):
    train, mats, outer, seq = frozen_train()
    arrays, meta = load_embeddings(train, seq)
    labels_path = ROOT / "results/boltz/phase19b/vti_train_acute_iva_targets.csv"
    labels = pd.read_csv(labels_path, dtype={"variant_key": str})
    if len(labels) != 210 or labels.variant_key.duplicated().any() or set(labels.variant_key) != set(train.variant_key):
        raise ValueError("Phase 19B TRAIN-only outcomes differ from frozen training set")
    aligned = train[["variant_key"]].merge(labels[["variant_key", TARGET]],
                                            on="variant_key", sort=False, validate="one_to_one")
    if aligned.variant_key.tolist() != train.variant_key.tolist():
        raise ValueError("Target order changed")
    y = aligned[TARGET].to_numpy(float)
    if not np.isfinite(y).all():
        raise ValueError("Missing/nonfinite acute IVA outcomes")
    pos = train.protein_position.to_numpy(int)
    X = np.asarray(mats["sequence_structure"], float)
    if X.shape != (210, 80) or np.isinf(X).any():
        raise ValueError(f"Expected sequence+structure 210x80, got {X.shape}")

    choice_path = ROOT / "results/boltz/phase19c/acute_iva_vti_pp_inner_selection.csv"
    selections = pd.read_csv(choice_path)
    selections = selections.loc[selections.model.eq(BASELINE)]
    if len(selections) != 10 or selections.duplicated(["repeat", "fold"]).any():
        raise ValueError("Expected 10 Phase19C hyperparameter selections")
    params_by_fold = {(int(r.repeat), int(r.fold)): json.loads(r.hyperparams)
                      for r in selections.itertuples(index=False)}

    print("=" * 106, flush=True)
    print("PHASE 20B | ESM-2 MUTATION FEATURES — FROZEN ACUTE IVA NESTED BENCHMARK", flush=True)
    print("=" * 106, flush=True)
    print("210 VTI TRAIN variants / 168 positions | 2 x 5 frozen outer folds", flush=True)
    print("ESM model frozen. PCA/normalization fit inside each inner and outer training fold.", flush=True)
    print("ESM representation and PCA size: selected ONLY on inner 3-fold grouped CV.", flush=True)
    print("XGBoost parameters: existing per-outer-fold Phase19C inner choices, no new tuning.", flush=True)
    print("Validation/test outcomes: NOT LOADED", flush=True)
    print("Input embeddings SHA256:", sha256(EMBED), flush=True)
    print("Mode:", "SMOKE (first outer fold only)" if args.smoke else "FULL", flush=True)

    oof, selections_out = [], []
    for rep, fold, fit, held in (outer[:1] if args.smoke else outer):
        rep, fold = int(rep), int(fold)
        if set(pos[fit]) & set(pos[held]):
            raise ValueError("Outer group leakage")
        params = params_by_fold[(rep, fold)]
        seed = mz.SEED + rep * 100 + fold
        reg = mz.estimator("xgboost", params, seed)
        reg.fit(X[fit], y[fit])
        pb = np.asarray(reg.predict(X[held]), float)

        selected, inner_mae, inner_details = select_esm(
            X[fit], y[fit], pos[fit], {k: v[fit] for k, v in arrays.items()}, params, seed,
            smoke=args.smoke,
        )
        representation, ncomp = selected
        a_all, b_all = fold_augment(
            X[fit], X[held], arrays[representation][fit], arrays[representation][held],
            max(N_PCS), seed)
        a = a_all[:, :X.shape[1] + ncomp]
        b = b_all[:, :X.shape[1] + ncomp]
        reg = mz.estimator("xgboost", params, seed)
        reg.fit(a, y[fit])
        pe = np.asarray(reg.predict(b), float)
        if pb.shape != pe.shape or pb.shape != (len(held),) or not np.isfinite(pb).all() or not np.isfinite(pe).all():
            raise ValueError("Invalid OOF predictions")

        baseline_mae = position_mae(y[held], pb, pos[held])
        esm_mae = position_mae(y[held], pe, pos[held])
        print(f"[OUTER] {rep}/{fold} fit={len(fit)} held={len(held)} "
              f"control={baseline_mae:.3f} ESM={esm_mae:.3f} "
              f"inner={inner_mae:.3f} selected={representation}@{ncomp}pc", flush=True)
        selections_out.append({"repeat": rep, "fold": fold, "representation": representation,
                               "n_pca_components": int(ncomp), "inner_position_mae_pp": inner_mae,
                               "params": json.dumps(params, sort_keys=True),
                               "candidates": json.dumps(inner_details, sort_keys=True)})
        for name, predictions in (("control_xgb", pb), ("esm_pca_xgb", pe)):
            for idx, pred in zip(held, predictions):
                oof.append({"repeat": rep, "fold": fold, "model": name,
                            "variant_key": str(train.iloc[idx].variant_key),
                            "protein_position": int(pos[idx]),
                            "observed_pp": float(y[idx]), "predicted_pp": float(pred)})

    if args.smoke:
        print("PASS: Smoke fold, baseline plus ESM model, no outputs written", flush=True)
        return

    frame = pd.DataFrame(oof)
    if frame.duplicated(["repeat", "model", "variant_key"]).any() or len(frame) != 840:
        raise ValueError("OOF predictions incomplete or duplicated")
    # Parity with original Phase19C baseline verifies identical preprocessing.
    hist_path = ROOT / "results/boltz/phase19c/acute_iva_vti_pp_oof.csv"
    hist = pd.read_csv(hist_path, dtype={"variant_key": str})
    hist = hist.loc[hist.model.eq(BASELINE)].copy()
    j = frame.loc[frame.model.eq("control_xgb")].merge(
        hist[["repeat", "fold", "variant_key", "observed_pp", "predicted_pp"]],
        on=["repeat", "fold", "variant_key"], how="outer", validate="one_to_one",
        suffixes=("_new", "_old"), indicator=True,
    )
    if len(j) != 420 or not j._merge.eq("both").all():
        raise ValueError("Original Phase19C frozen OOF reference doesn't cover the same keys")
    pred_delta = float(np.max(np.abs(j.predicted_pp_new - j.predicted_pp_old)))
    y_delta = float(np.max(np.abs(j.observed_pp_new - j.observed_pp_old)))
    if pred_delta > 1e-3 or y_delta > 1e-8:
        raise ValueError(f"Phase19C baseline parity failed: max prediction diff {pred_delta:.6g}, label diff {y_delta:.6g}")
    print(f"PASS: Exact Phase19C OOF parity (max prediction difference={pred_delta:.8f} pp)", flush=True)

    avg = frame.groupby(["model", "variant_key", "protein_position"], as_index=False).agg(
        observed_pp=("observed_pp", "first"),
        predicted_pp=("predicted_pp", "mean"), n_repeats=("repeat", "nunique"))
    if len(avg) != 420 or not avg.n_repeats.eq(2).all():
        raise ValueError("Incomplete repeat-averaged OOF")
    summary = []
    for model, g in avg.groupby("model"):
        yy = g.observed_pp.to_numpy(float)
        pp = g.predicted_pp.to_numpy(float)
        posg = g.protein_position.to_numpy(int)
        high20 = yy > 20
        high40 = yy > 40
        low2 = yy <= 2
        summary.append({
            "model": model,
            "position_mae_pp": position_mae(yy, pp, posg),
            "variant_mae_pp": float(np.mean(np.abs(yy - pp))),
            "rmse_pp": float(np.sqrt(np.mean((yy - pp) ** 2))),
            "high20_mae_pp": float(np.mean(np.abs(yy[high20] - pp[high20]))),
            "high40_mae_pp": float(np.mean(np.abs(yy[high40] - pp[high40]))),
            "low2_mae_pp": float(np.mean(np.abs(yy[low2] - pp[low2]))),
            "high20_recall": float(np.mean(pp[high20] > 20)),
            "max_predicted_pp": float(pp.max()),
        })
    summary = pd.DataFrame(summary).sort_values("position_mae_pp")
    boot = paired_bootstrap(avg, "esm_pca_xgb", "control_xgb")

    # Phase19I reliability-aware arm is the strongest previously observed
    # acute-IVA benchmark. Compare it post hoc on exactly the same identities;
    # it never participates in ESM inner selection or training.
    prior_path = ROOT / "results/boltz/phase19i/phase19i_averaged.csv"
    prior_comparison = None
    if prior_path.is_file():
        prior = pd.read_csv(prior_path, dtype={"variant_key": str})
        prior = prior.loc[prior.model.eq("reliability_weighted")].copy()
        if len(prior) != 210 or prior.variant_key.duplicated().any():
            raise ValueError("Invalid Phase19I reliability reference")
        ref = avg.loc[avg.model.eq("esm_pca_xgb"),
                      ["variant_key", "protein_position", "observed_pp"]].merge(
            prior[["variant_key", "protein_position", "observed_pp", "predicted_pp"]],
            on="variant_key", suffixes=("_esm", "_prior"), validate="one_to_one")
        if (len(ref) != 210
                or not np.array_equal(ref.protein_position_esm.to_numpy(int), ref.protein_position_prior.to_numpy(int))
                or not np.allclose(ref.observed_pp_esm, ref.observed_pp_prior, rtol=0, atol=1e-8)):
            raise ValueError("Phase19I reliable benchmark key/label mismatch")
        prior_renamed = prior.assign(model="phase19i_reliability_weighted")
        comparator_frame = pd.concat([avg, prior_renamed], ignore_index=True)
        prior_comparison = paired_bootstrap(
            comparator_frame, "esm_pca_xgb", "phase19i_reliability_weighted")
        print("\nCOMPARISON VS PRIOR BEST (Phase19I reliability-weighted)", flush=True)
        print("  Prior MAE: " + f"{position_mae(prior.observed_pp, prior.predicted_pp, prior.protein_position):.4f} pp", flush=True)
        print("  ESM gain vs prior (positive = better): "
              + f"{prior_comparison['gain_pp']:+.4f} pp", flush=True)
        print("  Exploratory 95% position bootstrap: "
              + f"[{prior_comparison['ci95_low']:+.4f}, {prior_comparison['ci95_high']:+.4f}]", flush=True)

    print("\n" + "=" * 106, flush=True)
    print("PHASE 20B | REPEATED OOF RESULTS (168 positions)", flush=True)
    print("=" * 106, flush=True)
    print(summary.round(4).to_string(index=False), flush=True)
    print("\nPAIRED POSITION BOOTSTRAP (ESM vs control)", flush=True)
    print(json.dumps(boot, indent=2), flush=True)
    print("WARNING: Exploratory CI doesn't adjust for comparing other methods or studying this dataset.", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    paths = {
        "oof": OUT / "phase20_oof.csv", "averaged": OUT / "phase20_averaged.csv",
        "summary": OUT / "phase20_summary.csv", "selection": OUT / "phase20_inner_selection.csv",
        "bootstrap": OUT / "phase20_paired_bootstrap.json", "report": OUT / "phase20_provenance.json",
    }
    frame.to_csv(paths["oof"], index=False)
    avg.to_csv(paths["averaged"], index=False)
    summary.to_csv(paths["summary"], index=False)
    pd.DataFrame(selections_out).to_csv(paths["selection"], index=False)
    paths["bootstrap"].write_text(json.dumps({"vs_phase19c_control": boot,
                                                 "vs_phase19i_reliability": prior_comparison}, indent=2) + "\n")
    provenance = {
        "phase": "20B_ESM2_nested_grouped_acute_IVA",
        "n_train_variants": 210, "n_positions": 168,
        "outer": "Frozen Phase15C 2x5 position-held-out folds",
        "inner": "3-fold GroupKFold on outer training positions",
        "outcome": TARGET,
        "baseline_parity_max_abs_pp": pred_delta,
        "esm_embeddings_sha256": sha256(EMBED),
        "phase19b_labels_sha256": sha256(labels_path),
        "phase19c_selection_sha256": sha256(choice_path),
        "phase19c_oof_sha256": sha256(hist_path),
        "pca_options": list(N_PCS), "representations": list(REPS),
        "vs_phase19i_reliability": prior_comparison,
        "esms": meta,
        "all_outputs_training_only": True,
        "protected_validation_and_test_outcomes_accessed": False,
        "warning": "Post-hoc exploratory development; final comparison requires independent validation",
    }
    paths["report"].write_text(json.dumps(provenance, indent=2) + "\n")
    print("\nOUTPUTS", flush=True)
    for k, p in paths.items():
        print(f"  {k}: {p.relative_to(ROOT)}", flush=True)
    print("PASS: Phase20B complete. No protected validation/test outcomes read.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("extract", help="Extract frozen ESM-2 embeddings, VTI TRAIN only")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("benchmark", help="Nested grouped-CV ablation versus exact Phase19C")
    p.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.action == "extract":
        extract(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
