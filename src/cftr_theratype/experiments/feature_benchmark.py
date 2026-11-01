"""Reproduction of Phase26G benchmark using Stage1/2 canonical modules.

Only the frozen VTI TRAIN cohort is loaded. By default, outputs go to the
maintenance/refactor area and NEVER overwrite published/historical results.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd

from cftr_theratype.evaluation.metrics import paired_ci, position_mae
from cftr_theratype.models.regressors import phase26g_choose, phase26g_model

# EXACT Phase26G specs, in original order; do not optimize during refactor.
SETUPS = [
    ("ridge_68", "sequence", "ridge", None),
    ("ridge_80", "sequence_structure", "ridge", None),
    ("elastic_80", "sequence_structure", "elastic", None),
    ("extratrees_80", "sequence_structure", "trees", None),
    ("xgb_68", "sequence", "xgb", None),
    ("xgb_80", "sequence_structure", "xgb", None),
    ("xgb_82", "sequence_structure_thermompnn", "xgb", None),
    ("xgb_selected16_80", "sequence_structure", "xgb", 16),
    ("xgb_selected32_80", "sequence_structure", "xgb", 32),
]
KEYS = ("repeat", "fold", "variant_key")


def evaluate(d, features, folds, *, target="corrector", fast=False, max_folds=None,
             bootstraps=5000):
    """Pure evaluation with injected already-verified data (for testing).

    max_folds is for smoke tests only; full fold coverage is mandatory for
    comparing with historical predictions.
    """
    if target not in {"corrector", "total"}:
        raise ValueError("target must be corrector or total")
    if max_folds is not None and max_folds < 1:
        raise ValueError("max_folds must be >= 1")
    if bootstraps < 1:
        raise ValueError("bootstraps must be positive")
    expected = "corrector_rescue_pp" if target == "corrector" else "vti_rescue_pp"
    y = d[expected].to_numpy(float)
    positions = d.protein_position.to_numpy(int)
    keys = d.variant_key.astype(str).to_numpy()
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate VTI variant keys")
    matrices = {}
    for name in sorted({s[1] for s in SETUPS}):
        if name not in features:
            raise ValueError(f"Missing feature representation: {name}")
        x = np.array(features[name], dtype=float, copy=True)
        x[np.isinf(x)] = np.nan
        if x.ndim != 2 or x.shape[0] != len(d):
            raise ValueError(f"Invalid feature matrix {name}: {x.shape}")
        matrices[name] = x
        print(f"Feature matrix {name}: shape={x.shape}, missing={np.isnan(x).sum()}")
    predictions = []
    selections = []
    chosen_folds = folds if max_folds is None else folds[:max_folds]
    if not chosen_folds:
        raise ValueError("No frozen folds supplied")
    for fi, (rep, fold, trkeys, tekeys) in enumerate(chosen_folds, 1):
        tr = np.flatnonzero(np.isin(keys, list(trkeys)))
        te = np.flatnonzero(np.isin(keys, list(tekeys)))
        if (len(tr) + len(te) != len(d) or len(tr) == 0 or len(te) == 0
                or set(positions[tr]) & set(positions[te])):
            raise ValueError(f"Invalid protein-position fold {rep}/{fold}")
        frame = pd.DataFrame({
            "repeat": rep, "fold": fold, "variant_key": keys[te],
            "protein_position": positions[te], "observed": y[te],
            "study_cohort": d.study_cohort.iloc[te].to_numpy(),
        })
        for label, representation, kind, topk in SETUPS:
            x = matrices[representation]
            choice, inner_mae = phase26g_choose(
                x[tr], y[tr], positions[tr], kind, topk, fast,
            )
            model = phase26g_model(kind, choice, topk)
            model.fit(x[tr], y[tr])
            frame[label] = model.predict(x[te])
            selections.append({
                "repeat": rep, "fold": fold, "model": label,
                "representation": representation, "kind": kind,
                "top_k": topk, "selected_param": str(choice),
                "inner_position_mae": inner_mae,
            })
        predictions.append(frame)
        print(f"Completed {fi}/{len(chosen_folds)}: repeat={rep}, fold={fold}, "
              f"held-out positions={len(set(positions[te]))}", flush=True)
    out = pd.concat(predictions, ignore_index=True)
    if out.duplicated(list(KEYS)).any():
        raise ValueError("Duplicated OOF row keys")
    if max_folds is None:
        if len(chosen_folds) != 10 or set(out.repeat) != {1, 2}:
            raise ValueError("Expected frozen 2x5 evaluation")
        for rep, g in out.groupby("repeat"):
            if len(g) != len(d) or g.variant_key.nunique() != len(d):
                raise ValueError(f"Incomplete repeat {rep}")
    names = [s[0] for s in SETUPS]
    metrics = []
    for rep, g in out.groupby("repeat"):
        for name in names:
            errors = g.observed.to_numpy() - g[name].to_numpy()
            tail = g.observed.to_numpy() > 40
            metrics.append({
                "repeat": rep, "model": name,
                "position_mae_pp": position_mae(
                    g.observed.to_numpy(), g[name].to_numpy(),
                    g.protein_position.to_numpy(),
                ),
                "variant_mae_pp": float(np.mean(np.abs(errors))),
                "rmse_pp": float(np.sqrt(np.mean(errors ** 2))),
                "n_gt40": int(tail.sum()),
                "gt40_mae_pp": float(np.mean(np.abs(errors[tail]))) if tail.any() else np.nan,
            })
    metrics = pd.DataFrame(metrics)
    comparisons = pd.DataFrame([
        paired_ci(out, name, "xgb_80", bootstraps)
        for name in names if name != "xgb_80"
    ])
    return out, pd.DataFrame(selections), metrics, comparisons


def save_results(tables, target, outdir):
    dest = Path(outdir)
    dest.mkdir(parents=True, exist_ok=True)
    names = ("oof", "selected", "metrics_by_repeat", "bootstrap")
    paths = [dest / f"phase26g_{target}_{name}.csv" for name in names]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing result: {path}")
    for path, table in zip(paths, tables):
        table.to_csv(path, index=False)
    print(f"Saved independent refactor benchmark outputs to {dest}")


def run(root, *, target="corrector", fast=False, max_folds=None,
        bootstraps=5000, outdir=None):
    root = Path(root).resolve()
    # Import lazily so synthetic unit tests do not require the full data stack.
    from cftr_theratype.datasets.frozen_vti import load_train
    # This will call only canonical TRAIN-only loader.
    d, features, _, folds, reference = load_train(root)
    print(f"STAGE3 | canonical Phase26G reproduction | target={target} "
          f"frozen reference={reference} | fast={fast} | max_folds={max_folds}")
    result = evaluate(d, features, folds, target=target, fast=fast,
                      max_folds=max_folds, bootstraps=bootstraps)
    if outdir is None:
        suffix = "smoke" if fast or max_folds is not None else "full"
        outdir = root / "results/maintenance/refactor_audit/stage3" / suffix
    save_results(result, target, outdir)
    return result


def compare_oof(new_path, reference_path, *, atol=1e-6, require_full=True):
    """Compare held-out observations and predictions by exact frozen keys.

    Does not read other outcomes, mutate data, or retrain anything.
    """
    new_path, reference_path = Path(new_path), Path(reference_path)
    new = pd.read_csv(new_path, dtype={"variant_key": str})
    old = pd.read_csv(reference_path, dtype={"variant_key": str})
    required = set(KEYS) | {"protein_position", "observed"}
    if required - set(new) or required - set(old):
        raise ValueError(f"OOF schema mismatch: new={sorted(set(new))}, old={sorted(set(old))}")
    for label, frame in (("new", new), ("reference", old)):
        if frame.duplicated(list(KEYS)).any():
            raise ValueError(f"Duplicate {label} keys")
        if require_full and (len(frame) != 420 or set(frame.repeat) != {1, 2}
                             or frame.groupby("repeat").variant_key.nunique().to_dict() != {1: 210, 2: 210}):
            raise ValueError(f"Incomplete {label} frozen OOF")
    cols = [c for c in new if c in old and c not in KEYS]
    if "xgb_80" not in cols:
        raise ValueError("No reference xgb_80 prediction")
    merged = new.merge(old, on=list(KEYS), suffixes=("_new", "_old"),
                       how="outer", indicator=True, validate="one_to_one")
    if len(merged) != len(new) or len(merged) != len(old) or not merged._merge.eq("both").all():
        raise ValueError("Held-out row membership mismatch")
    numeric = []
    for col in cols:
        a, b = merged[col + "_new"], merged[col + "_old"]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            va, vb = a.to_numpy(float), b.to_numpy(float)
            if not np.allclose(va, vb, rtol=0, atol=atol, equal_nan=True):
                distance = np.abs(va - vb)
                distance[np.isnan(distance)] = np.inf
                raise ValueError(f"PARITY FAIL {col}: max_abs_difference={np.max(distance):.8g}")
            numeric.append((col, float(np.nanmax(np.abs(va - vb))) if np.isfinite(va - vb).any() else 0.0))
        elif not a.fillna("<NA>").equals(b.fillna("<NA>")):
            raise ValueError(f"PARITY FAIL categorical {col}")
    print("PASS: held-out OOF rows, observations and predictions agree")
    print(f"Rows: {len(merged)} | numeric columns: {len(numeric)} | tolerance: {atol}")
    for col, diff in numeric:
        print(f"  {col}: max_abs_diff={diff:.8g}")
    return {"rows": len(merged), "numeric_columns": len(numeric),
            "max_abs_diff": max((v for _, v in numeric), default=0.0)}
