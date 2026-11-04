#!/usr/bin/env python3
"""CFTR-Theratype 31I.2: leakage-safe assay audit and cross-fitted transfer pilot.

Two explicit stages (from repository root):

  python scripts/phase31i_auxiliary_transfer.py audit \
      --bihler-eti data/processed/REPLACE.csv --bihler-target EXACT_ETI_COLUMN \
      --vti-meta data/processed/bihler_2026_transfer_splits.parquet

  python scripts/phase31i_auxiliary_transfer.py pilot \
      --bihler-eti data/processed/REPLACE.csv --bihler-target EXACT_ETI_COLUMN \
      --vti-train results/biology/REPLACE_FROZEN_TRAIN.csv \
      --fold-col EXACT_FROZEN_FOLD_COLUMN \
      --vti-target corrector=EXACT_COLUMN --vti-target total=EXACT_COLUMN \
      --features results/biology/REPLACE_CANONICAL_FEATURE_BANK.parquet \
      --feature-cols results/biology/REPLACE_FEATURE_COLUMNS.txt

No VTI response labels are read in 'audit' mode. 'pilot' reads only the
explicitly provided VTI TRAIN file; never auto-loads VTI val/test responses.

Required project exports:
  * Bihler 2024 ETI-labelled source (ETI target passed explicitly).
  * Frozen VTI TRAIN only, one row per VTI variant with aa_code, frozen fold and
    selected target columns (optionally variant_key and position).
  * Genotype-only FEATURE BANK (including eligible auxiliary substitutions),
    using the SAME feature engineering as VTI; no observed assay features.
  * A list of frozen genotype feature column names, one per line, OR --feature-prefix.

The frozen downstream estimator here is *a prespecified Ridge pilot*, NOT a
claim of reproducing a previous Phase 26D/26G model. Same estimator,
preprocessing and hyperparameters for M0-M3, with no outer-fold tuning.

Dependencies: numpy, pandas, scipy, scikit-learn, pyarrow for parquet.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, theilslopes
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

S5_NAME = "phase31a_castellani_s5_transport.csv"
S6_NAME = "phase31a_castellani_s6_processing.csv"
ROOT = Path.cwd()
OUT_DEFAULT = ROOT / "results/biology/phase31i"
from cftr_theratype.datasets.assay_identifiers import (
    THREE, ID_CANDIDATES, POSITION_CANDIDATES,
    norm_aa as _canonical_norm_aa, with_ids as _canonical_with_ids,
)
from cftr_theratype.datasets.assay_tables import (
    sha256 as _canonical_sha256,
    resolve_file as _canonical_resolve_file,
    numeric as _canonical_numeric,
    table_aux as _canonical_table_aux,
    agreement as _canonical_agreement,
    permute_by_position as _canonical_permute_by_position,
)
from cftr_theratype.models.assay_teachers import make_pilot_teacher as _canonical_make_pilot_teacher

SAFE_META = set(ID_CANDIDATES + POSITION_CANDIDATES + [
    "variant_key", "mutation_type", "is_control", "study", "source_row", "split",
    "transfer_split", "eti_split", "partition", "fold", "outer_fold", "cv_fold",
])
BAD_FEATURE_WORDS = re.compile(r"(rescue|response|outcome|target|label|matur|assay|fsk|iva|(?:^|_)eti(?:_|$)|(?:^|_)vti(?:_|$))", re.I)


def sha256(path: Path) -> str:
    """Legacy API; canonical file digest."""
    return _canonical_sha256(path)


def json_safe(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    raise TypeError(f"Cannot serialize {type(x).__name__}")


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, default=json_safe, allow_nan=False) + "\n")


def resolve_file(requested: str | None, fallback: str | None = None) -> Path:
    """Legacy API; preserve the currently selected repository root."""
    return _canonical_resolve_file(requested, fallback, root=ROOT)


def columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        import pyarrow.parquet as pq
        return pq.read_schema(path).names
    if path.suffix.lower() == ".csv":
        return list(pd.read_csv(path, nrows=0).columns)
    raise ValueError(f"Only CSV/Parquet supported: {path}")


def load(path: Path, select: list[str] | None = None) -> pd.DataFrame:
    names = columns(path)
    if select is not None:
        missing = set(select) - set(names)
        if missing:
            raise ValueError(f"{path}: required columns missing {sorted(missing)}; available: {names}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, usecols=select, low_memory=False)
    return pd.read_parquet(path, columns=select)


def load_metadata_only(path: Path, extra: list[str] | None = None) -> pd.DataFrame:
    names = columns(path)
    use = [s for s in names if s in SAFE_META or (extra and s in extra)]
    if not use:
        raise ValueError(f"Metadata source has no recognized identifiers: {names}")
    return load(path, use)


def norm_aa(value) -> str | None:
    """Legacy API: canonical Phase31 substitution normalisation."""
    return _canonical_norm_aa(value)


def with_ids(df: pd.DataFrame, name: str, strict: bool = False) -> pd.DataFrame:
    """Legacy API: canonical Phase31 substitution and position validation."""
    return _canonical_with_ids(df, name, strict=strict)


def numeric(series: pd.Series, label: str) -> pd.Series:
    """Legacy API; canonical strict numeric parsing."""
    return _canonical_numeric(series, label)


def table_aux(df: pd.DataFrame, name: str, target: str, study: str | None = None) -> tuple[pd.DataFrame, dict]:
    """Legacy API; canonical auxiliary assay table normalization."""
    return _canonical_table_aux(df, name, target, study)


def agreement(a: pd.DataFrame, b: pd.DataFrame, an: str, bn: str) -> tuple[dict, pd.DataFrame]:
    """Legacy API; canonical substitution-level assay agreement."""
    return _canonical_agreement(a, b, an, bn)


def audit(args, p: dict[str, Path], features: pd.DataFrame | None, vti_ids: pd.DataFrame | None):
    print("=" * 85)
    print("PHASE 31I.2 A | EXACT-ASSAY COVERAGE / AGREEMENT AUDIT")
    print("Bihler vs Castellani ETI-labelled coverage; no VTI outcomes in audit mode")
    print("=" * 85)
    s5 = load(p["s5"])
    s6 = load(p["s6"])
    bihler = load(p["bihler"])
    if bool(args.bihler_filter_col) != bool(args.bihler_filter_value):
        raise ValueError("Specify --bihler-filter-col AND --bihler-filter-value together")
    if args.bihler_filter_col:
        if args.bihler_filter_col not in bihler:
            raise ValueError(f"Bihler filter column {args.bihler_filter_col!r} not found: {list(bihler)}")
        before = len(bihler)
        bihler = bihler.loc[
            bihler[args.bihler_filter_col].astype(str).str.strip().str.lower() ==
            args.bihler_filter_value.strip().lower()
        ].copy()
        print(f"Bihler filter: {before} -> {len(bihler)} rows ({args.bihler_filter_col}={args.bihler_filter_value})")
        if bihler.empty:
            raise ValueError("Bihler treatment filter selected zero rows")
    else:
        condition_cols = [c for c in bihler if c.lower() in
                          {"treatment", "drug", "compound", "arm", "condition", "treatment_group"}]
        for col in condition_cols:
            levels = sorted(bihler[col].dropna().astype(str).str.strip().unique())
            if len(levels) > 1:
                raise ValueError(
                    f"Bihler source has multiple {col} arms {levels[:12]}; specify ETI-labelled source "
                    "or --bihler-filter-col/--bihler-filter-value. No silent pooling of treatments."
                )
    print("Input columns:")
    for n, d in [("S5", s5), ("S6", s6), ("Bihler", bihler)]:
        print(f"  {n}: {list(d)}")
    aux, summaries = {}, []
    for tag, df, target, study in [
        ("bihler", bihler, args.bihler_target, None),
        ("p289", s5, "eti_rescue_pp", "P289"),
        ("u032", s5, "eti_rescue_pp", "U032"),
        ("p289_s6_baseline", s6, args.s6_baseline, "P289"),
        ("p289_s6_treated", s6, args.s6_treated, "P289"),
        ("u032_s6_baseline", s6, args.s6_baseline, "U032"),
        ("u032_s6_treated", s6, args.s6_treated, "U032"),
    ]:
        aux[tag], entry = table_aux(df, tag, target, study)
        summaries.append(entry)
    bihpos = set(aux["bihler"].pos31i2.astype(int))
    vtipos = set(vti_ids.pos31i2.dropna().astype(int)) if vti_ids is not None else set()
    vtiaa = set(vti_ids.aa31i.dropna()) if vti_ids is not None else set()
    feature_aa = set(features.aa31i.dropna()) if features is not None else set()
    for row in summaries:
        d = aux[row["assay"]]
        pos = set(d.pos31i2.astype(int))
        row.update({
            "positions_with_Bihler_ETI_labels": len(pos & bihpos),
            "additional_positions_vs_Bihler_ETI_labelled": len(pos - bihpos),
            "positions_in_VTI_TRAIN": len(pos & vtipos) if vti_ids is not None else np.nan,
            "exact_substitutions_in_VTI_TRAIN": len(set(d.aa31i) & vtiaa) if vti_ids is not None else np.nan,
            "feature_covered_substitutions": len(set(d.aa31i) & feature_aa) if features is not None else np.nan,
        })
    report = pd.DataFrame(summaries)
    report.to_csv(args.out / "phase31i2_audit.csv", index=False)
    contrast = [("bihler", "p289"), ("bihler", "u032"), ("p289", "u032")]
    comparisons, matched = [], []
    for an, bn in contrast:
        result, matches = agreement(aux[an], aux[bn], an, bn)
        comparisons.append(result)
        matched.append(matches)
    pd.DataFrame(comparisons).to_csv(args.out / "phase31i2_agreement.csv", index=False)
    pd.concat(matched, ignore_index=True).to_csv(args.out / "phase31i2_exact_overlap.csv", index=False)
    print("\nCoverage (each assay uses independent scales):")
    print(report[["assay", "source_rows", "unique_aa_substitutions", "unique_positions",
                  "additional_positions_vs_Bihler_ETI_labelled", "exact_substitutions_in_VTI_TRAIN",
                  "feature_covered_substitutions"]].to_string(index=False))
    print("\nAgreement: matched *exact amino-acid substitutions*, per-assay median replicates:")
    print(pd.DataFrame(comparisons).round(3).to_string(index=False))
    print("NOTE: median differences and slopes are descriptive ONLY; assay scales may differ.")
    if features is None:
        print("BLOCKER for pilot: genotype-only auxiliary feature bank not specified (--features).")
    status = {
        "phase": "31I.2", "stage": "audit", "n_vti_outcomes_loaded": 0,
        "replicate_handling": "median per canonical amino-acid substitution within EACH assay",
        "agreement_warning": "Do not pool rescue values; P289 and U032 may differ in scales.",
        "feature_coverage_tested": features is not None,
        "source_schemas": {"s5": list(s5), "s6": list(s6), "bihler": list(bihler)},
        "source_files": {name: {"path": str(path), "sha256": sha256(path)} for name, path in p.items()},
        "bihler_filter": {"column": args.bihler_filter_col, "value": args.bihler_filter_value},
        "target_columns": {"bihler": args.bihler_target, "s5": "eti_rescue_pp",
                           "s6_baseline": args.s6_baseline, "s6_treated": args.s6_treated},
    }
    return aux, status


def load_feature_bank(path: Path, names: list[str]) -> pd.DataFrame:
    df = load(path, [c for c in columns(path) if c in set(ID_CANDIDATES + POSITION_CANDIDATES + names)])
    missing = set(names) - set(df)
    if missing:
        raise ValueError(f"Feature bank missing frozen columns: {sorted(missing)}")
    df = with_ids(df, "feature bank")
    df = df.loc[df.aa31i.notna() & df.pos31i2.notna(), ["aa31i", "pos31i2"] + names].copy()
    for col in names:
        df[col] = numeric(df[col], f"feature.{col}")
        if df[col].notna().sum() == 0:
            raise ValueError(f"Feature '{col}' has no finite values")
    if df.empty:
        raise ValueError("Feature bank contains no parseable substitutions")
    # Duplicated aa codes are acceptable only if all frozen features identical.
    distinct = df.groupby("aa31i", dropna=False)[names].nunique(dropna=False)
    if (distinct > 1).any(axis=1).any():
        inconsistent = distinct.index[(distinct > 1).any(axis=1)].tolist()[:12]
        raise ValueError(f"Conflicting feature rows for same protein substitution: {inconsistent}")
    return df.drop_duplicates("aa31i").reset_index(drop=True)


def frozen_columns(args) -> list[str] | None:
    if args.feature_cols:
        path = resolve_file(args.feature_cols)
        if path.suffix.lower() == ".json":
            contents = json.loads(path.read_text())
            if not isinstance(contents, list):
                raise ValueError("JSON feature list must be an array of column-name strings")
            names = contents
        else:
            names = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    elif args.feature_prefix and args.features:
        names = sorted(n for n in columns(resolve_file(args.features)) if n.startswith(args.feature_prefix))
    else:
        return None
    if not names or len(names) != len(set(names)) or not all(isinstance(n, str) for n in names):
        raise ValueError("Frozen feature list empty/duplicate/invalid")
    suspicious = [n for n in names if BAD_FEATURE_WORDS.search(n)]
    if suspicious:
        raise ValueError(f"Assay-derived/target-like feature names forbidden: {suspicious}. Use genotype-only features.")
    return names


def load_vti_ids(path: Path, *, train_only: bool = True) -> pd.DataFrame:
    d = with_ids(load_metadata_only(path), "VTI metadata")
    if "transfer_split" in d:
        if train_only:
            d = d[d.transfer_split.astype(str).str.lower() == "train"].copy()
    elif "split" in d and train_only:
        d = d[d.split.astype(str).str.lower() == "train"].copy()
    if not len(d):
        raise ValueError("VTI metadata has no TRAIN rows")
    return d


def load_vti_train(args, p: Path, feature_bank: pd.DataFrame, feature_names: list[str]):
    requested_targets = {}
    for expression in args.vti_target:
        if "=" not in expression:
            raise ValueError("--vti-target expects name=EXACT_COLUMN, e.g. corrector=resc_corrector")
        name, col = expression.split("=", 1)
        if not name or not col or name in requested_targets:
            raise ValueError(f"Invalid or duplicate VTI target declaration: {expression}")
        requested_targets[name] = col
    if not requested_targets:
        raise ValueError("Pilot requires at least one --vti-target name=EXACT_COLUMN")
    schema = columns(p)
    names = [x for x in schema if x in set(SAFE_META) | set(requested_targets.values()) | {args.fold_col}]
    d = with_ids(load(p, names), "VTI TRAIN", strict=True)
    if "transfer_split" in d:
        if not d.transfer_split.astype(str).str.lower().eq("train").all():
            raise ValueError("VTI TRAIN file includes non-TRAIN partitions. Export only TRAIN.")
    if args.fold_col not in d:
        raise ValueError(f"Missing frozen fold column '{args.fold_col}' in {p}; columns: {schema}")
    if d[args.fold_col].isna().any():
        raise ValueError("VTI frozen folds have missing assignments")
    d["fold31i"] = d[args.fold_col].astype(str)
    if len(d) != args.expect_rows or d.pos31i2.nunique() != args.expect_positions:
        raise ValueError(f"Unexpected VTI TRAIN size: {len(d)} rows / {d.pos31i2.nunique()} positions; "
                         f"expected {args.expect_rows}/{args.expect_positions}. Check Phase26D export.")
    nfold = d.fold31i.nunique()
    if nfold != 5:
        raise ValueError(f"Frozen folds must have 5 unique values, got {nfold}")
    if d.groupby("pos31i2").fold31i.nunique().max() != 1:
        raise ValueError("POSITION LEAKAGE: same protein position assigned to multiple frozen outer folds")
    for name, col in requested_targets.items():
        d[f"y31i_{name}"] = numeric(d[col], f"VTI_TRAIN.{col}")
        if not d[f"y31i_{name}"].notna().all():
            raise ValueError(f"Missing VTI target values for {name}; use a frozen complete-case TRAIN export")
    # No use of assay target columns or VTI data in the genotype feature bank.
    d["vti_row31i"] = np.arange(len(d))
    bank = feature_bank.drop(columns="pos31i2")
    d = d.merge(bank, on="aa31i", how="left", validate="many_to_one", indicator="feature_merge")
    absent = d.feature_merge.ne("both")
    if absent.any():
        raise ValueError(f"Feature bank missing {int(absent.sum())}/{len(d)} VTI TRAIN rows: "
                         f"{d.loc[absent, 'aa31i'].unique().tolist()[:20]}. Do not impute absent complete genotypes.")
    if d[feature_names].isna().all(axis=1).any():
        raise ValueError("At least one VTI mutation has all frozen feature columns missing")
    return d.sort_values("vti_row31i").reset_index(drop=True), requested_targets


def model(alpha: float):
    """Legacy API; canonical original Ridge pilot constructor."""
    return _canonical_make_pilot_teacher(alpha)


def teacher_tables(aux: dict[str, pd.DataFrame], bank: pd.DataFrame, feature_names: list[str]):
    fitted = {}
    for key, d in aux.items():
        full = d.merge(bank.drop(columns="pos31i2"), on="aa31i", how="inner", validate="one_to_one")
        full = full[full[feature_names].notna().any(axis=1)].copy()
        if len(full):
            full.pos31i2 = full.pos31i2.astype(int)
        fitted[key] = full
    return fitted


def permute_by_position(d: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Legacy API; canonical position-group negative control."""
    return _canonical_permute_by_position(d, rng)


def teacher_prediction(d: pd.DataFrame, training: pd.DataFrame, outer_validation: pd.DataFrame,
                       names: list[str], args, seed: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Inner group OOF for ALL outer TRAIN rows; refit teacher for outer VALID."""
    groups = training.pos31i2.astype(int).to_numpy()
    held = set(outer_validation.pos31i2.astype(int))
    if not set(d.pos31i2.astype(int)).isdisjoint(held):
        raise AssertionError("Auxiliary teacher has outer validation position(s)!")
    if len(d) < args.min_teacher_rows or d.pos31i2.nunique() < 5:
        raise ValueError(f"Teacher shortfall: {len(d)} rows / {d.pos31i2.nunique()} positions")
    inner = GroupKFold(n_splits=5)
    oof = np.full(len(training), np.nan, dtype=float)
    for itr, iva in inner.split(training, groups=groups):
        inner_pos = set(groups[iva])
        sub = d[~d.pos31i2.isin(inner_pos)]
        if len(sub) < args.min_teacher_rows or sub.pos31i2.nunique() < 5:
            raise ValueError(f"Inner teacher shortfall: {len(sub)} rows / {sub.pos31i2.nunique()} positions")
        if not set(sub.pos31i2).isdisjoint(inner_pos | held):
            raise AssertionError("Inner/outer position contamination")
        reg = model(args.teacher_alpha)
        reg.fit(sub[names].to_numpy(dtype=float), sub.value31i.to_numpy(dtype=float))
        oof[iva] = reg.predict(training.iloc[iva][names].to_numpy(dtype=float))
    if not np.isfinite(oof).all():
        raise AssertionError("Non-finite cross-fitted teacher predictions")
    final = model(args.teacher_alpha)
    final.fit(d[names].to_numpy(dtype=float), d.value31i.to_numpy(dtype=float))
    val = final.predict(outer_validation[names].to_numpy(dtype=float))
    return oof, val, {"teacher_rows": len(d), "teacher_positions": int(d.pos31i2.nunique()),
                      "outer_excluded_positions": len(held), "inner_folds": 5}


def metric(y: np.ndarray, pred: np.ndarray) -> dict:
    n = len(y)
    rho = float(spearmanr(y, pred).statistic) if n >= 3 and np.unique(y).size > 1 and np.unique(pred).size > 1 else np.nan
    return {"n": n, "mae": float(mean_absolute_error(y, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y, pred))), "spearman": rho}


def compute_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (target, mod), grp in oof.groupby(["target", "model"], sort=True):
        for scope, data in [("variant", grp),
                            ("position", grp.groupby("pos31i2", as_index=False)[["observed", "prediction"]].mean())]:
            rows.append({"target": target, "model": mod, "scope": scope, "fold": "POOLED",
                         **metric(data.observed.to_numpy(), data.prediction.to_numpy())})
        for f, part in grp.groupby("fold", sort=True):
            for scope, data in [("variant", part),
                                ("position", part.groupby("pos31i2", as_index=False)[["observed", "prediction"]].mean())]:
                rows.append({"target": target, "model": mod, "scope": scope, "fold": f,
                             **metric(data.observed.to_numpy(), data.prediction.to_numpy())})
    return pd.DataFrame(rows)


def bootstraps(oof: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    rows = []
    for target, d in oof.groupby("target", sort=True):
        base = d[d.model == "M0"].copy()
        base = base.sort_values("vti_row31i")
        for choice in ["M1", "M2", "M3", "M3_perm"]:
            alt = d[d.model == choice].sort_values("vti_row31i")
            if alt.empty:
                continue
            if not np.array_equal(base.vti_row31i.to_numpy(), alt.vti_row31i.to_numpy()):
                raise AssertionError("Baseline-vs-model row mismatch")
            if not np.array_equal(base.pos31i2.to_numpy(), alt.pos31i2.to_numpy()):
                raise AssertionError("Baseline-vs-model position mismatch")
            for scope in ["variant", "position"]:
                if scope == "variant":
                    a = base[["pos31i2", "observed", "prediction"]].copy()
                    b = alt.prediction.to_numpy()
                    a["pred_alt"] = b
                else:
                    tmp = base[["pos31i2", "observed", "prediction"]].copy()
                    tmp["pred_alt"] = alt.prediction.to_numpy()
                    a = tmp.groupby("pos31i2", as_index=False)[["observed", "prediction", "pred_alt"]].mean()
                errors = a.assign(base_err=np.abs(a.observed - a.prediction),
                                  alt_err=np.abs(a.observed - a.pred_alt))
                groups = [g[["base_err", "alt_err"]].to_numpy() for _, g in errors.groupby("pos31i2", sort=True)]
                n_group = len(groups)
                # Deterministic resampling of *protein positions*, including all rows in each sampled group.
                rng = np.random.default_rng(seed)
                # Vectorized cluster bootstrap: resample entire protein positions.
                bsum = np.array([x[:, 0].sum() for x in groups])
                asum = np.array([x[:, 1].sum() for x in groups])
                nrows = np.array([len(x) for x in groups])
                idx = rng.integers(0, n_group, size=(reps, n_group))
                diffs = ((bsum[idx] - asum[idx]).sum(axis=1) /
                         nrows[idx].sum(axis=1))
                rows.append({"target": target, "model": choice, "scope": scope,
                             "mae_improvement_vs_M0": float(np.mean(errors.base_err - errors.alt_err)),
                             "ci_low": float(np.quantile(diffs, .025)),
                             "ci_high": float(np.quantile(diffs, .975)),
                             "bootstrap_reps": reps, "bootstrap_seed": seed,
                             "n_positions": n_group})
    return pd.DataFrame(rows)


def pilot(args, aux, audit_info, feature_names, bank, p):
    print("\n" + "=" * 85)
    print("PHASE 31I.2 B | FROZEN VTI TRAIN POSITION-HELD-OUT PILOT")
    print("Ridge exploratory baseline; ONLY supplied frozen TRAIN responses are loaded")
    print("=" * 85)
    vt, targets = load_vti_train(args, p["vti_train"], bank, feature_names)
    teachers = teacher_tables(aux, bank, feature_names)
    print(f"VTI TRAIN: {len(vt)} rows, {vt.pos31i2.nunique()} positions, {vt.fold31i.nunique()} frozen folds")
    print("Feature bank coverage, labelled auxiliary observations:")
    for k, t in teachers.items():
        print(f"  {k:>17}: {len(t):4d} substitutions / {t.pos31i2.nunique():3d} positions")
    configurations = {
        "M0": [],
        "M1": ["bihler"],
        "M2": ["bihler", "p289", "u032"],
        "M3": ["bihler", "p289", "u032", "p289_s6_baseline", "p289_s6_treated",
               "u032_s6_baseline", "u032_s6_treated"],
        "M3_perm": ["bihler", "p289", "u032", "p289_s6_baseline", "p289_s6_treated",
                    "u032_s6_baseline", "u032_s6_treated"],
    }
    # Hold out every VTI validation position from ALL teacher fits, including auxiliary-only labels.
    preds, audit_rows, statuses = [], [], {}
    for fidx, fold in enumerate(sorted(vt.fold31i.unique())):
        tr = vt[vt.fold31i != fold].copy()
        va = vt[vt.fold31i == fold].copy()
        held = set(va.pos31i2.astype(int))
        assert held.isdisjoint(set(tr.pos31i2.astype(int)))
        print(f"\nOUTER {fold}: VTI TRAIN={len(tr)} / VALID={len(va)}; held positions={len(held)}")
        # Same teacher features serve all downstream targets. Cross-fitting is never target-based.
        feat_tr, feat_va, perm_tr, perm_va = {}, {}, {}, {}
        for j, (tag, d) in enumerate(teachers.items()):
            kept = d[~d.pos31i2.isin(held)].copy()
            audit_rows.append({"fold": fold, "teacher": tag, "available_rows": len(d),
                               "available_positions": d.pos31i2.nunique(), "post_exclusion_rows": len(kept),
                               "post_exclusion_positions": kept.pos31i2.nunique(),
                               "excluded_heldout_rows": len(d) - len(kept),
                               "excluded_heldout_positions": len(set(d.pos31i2) & held)})
            if not kept.empty:
                assert set(kept.pos31i2).isdisjoint(held)
            try:
                fit_tr, fit_va, _ = teacher_prediction(kept, tr, va, feature_names, args,
                                                       seed=args.seed + fidx * 1000 + j)
                feat_tr[tag], feat_va[tag] = fit_tr, fit_va
            except ValueError as err:
                print(f"  SKIP teacher {tag}: {err}")
                continue
            # Permute labels by POSITION in each assay separately, within outer-training only.
            seed = args.seed + 10000 * (fidx + 1) + 101 * j
            shuffled = permute_by_position(kept, np.random.default_rng(seed))
            pp_tr, pp_va, _ = teacher_prediction(shuffled, tr, va, feature_names, args, seed)
            perm_tr[tag], perm_va[tag] = pp_tr, pp_va
        for target in targets:
            yname = f"y31i_{target}"
            for config, required in configurations.items():
                negative = config.endswith("_perm")
                source_tr = perm_tr if negative else feat_tr
                source_va = perm_va if negative else feat_va
                absent = sorted(set(required) - set(source_tr))
                if absent:
                    statuses[f"{fold}/{target}/{config}"] = f"SKIPPED: missing teachers {absent}"
                    continue
                xtrain = tr[feature_names].to_numpy(dtype=float)
                xvalid = va[feature_names].to_numpy(dtype=float)
                if required:
                    xtrain = np.column_stack([xtrain] + [source_tr[t] for t in required])
                    xvalid = np.column_stack([xvalid] + [source_va[t] for t in required])
                reg = model(args.downstream_alpha)
                reg.fit(xtrain, tr[yname].to_numpy(dtype=float))
                pred = reg.predict(xvalid)
                for (row, yp) in zip(va.itertuples(index=False), pred):
                    # Access VTI outcome values ONLY from explicitly supplied TRAIN rows.
                    rec = {"vti_row31i": int(row.vti_row31i), "aa31i": row.aa31i,
                           "pos31i2": int(row.pos31i2), "fold": fold, "target": target,
                           "model": config, "observed": float(getattr(row, yname)),
                           "prediction": float(yp)}
                    preds.append(rec)
                statuses[f"{fold}/{target}/{config}"] = "PASS"
    oof = pd.DataFrame(preds)
    if oof.empty:
        raise RuntimeError("No complete outer OOF predictions; inspect missing teachers and feature coverage")
    # Only evaluate configurations present in EVERY fold, with identical VTI row coverage.
    full = set(vt.vti_row31i.astype(int))
    for (target, cfg), grp in oof.groupby(["target", "model"]):
        if set(grp.vti_row31i) != full or grp.vti_row31i.duplicated().any():
            statuses[f"{target}/{cfg}/pooled"] = "INCOMPLETE (removed from pooled metrics)"
            oof = oof[~((oof.target == target) & (oof.model == cfg))]
    if not ((oof.model == "M0").any()):
        raise RuntimeError("M0 OOF incomplete")
    metrics = compute_metrics(oof)
    bootstrap = bootstraps(oof, args.bootstrap, args.seed)
    oof.sort_values(["target", "model", "vti_row31i"]).to_csv(args.out / "phase31i2_oof_predictions.csv", index=False)
    metrics.to_csv(args.out / "phase31i2_metrics.csv", index=False)
    bootstrap.to_csv(args.out / "phase31i2_bootstrap.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(args.out / "phase31i2_teacher_folds.csv", index=False)
    audit_info.update({
        "stage": "pilot", "vti_train_source": {"path": str(p["vti_train"]), "sha256": sha256(p["vti_train"])},
        "feature_bank_source": {"path": str(p["features"]), "sha256": sha256(p["features"])},
        "vti_train_rows": len(vt), "vti_train_positions": vt.pos31i2.nunique(),
        "targets": targets, "fold_col": args.fold_col,
        "fold_ids": sorted(vt.fold31i.unique()),
        "feature_names": feature_names,
        "prespecified_models": {"teacher": f"Ridge(alpha={args.teacher_alpha})",
                                 "downstream": f"Ridge(alpha={args.downstream_alpha})"},
        "transform": "median imputation + StandardScaler, fitted on each teacher/downstream training fold ONLY",
        "teacher_crossfit": "inner 5-fold GroupKFold on VTI TRAIN protein positions; outer val always excluded",
        "negative_control": "within each assay, independent outer-fold permutation of position-level medians preserving residuals",
        "statuses": statuses, "bootstrap_reps": args.bootstrap, "seed": args.seed,
        "external_vti_val_or_test_labels_loaded": False,
    })
    save_json(args.out / "phase31i2_provenance.json", audit_info)
    print("\nPOOLED OOF metrics (MAE; each target separate):")
    print(metrics[(metrics.fold == "POOLED")].round(4).to_string(index=False))
    print("\nPaired MAE improvement relative to M0 (positive = better):")
    print(bootstrap.round(4).to_string(index=False) if len(bootstrap) else "No additional complete models")
    print("\nPASS: only frozen VTI TRAIN evaluated; CI crossing zero means exploratory evidence, NOT established gain.")
    return oof, metrics, bootstrap


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["audit", "pilot"])
    ap.add_argument("--s5", help="Castellani S5 table (autodiscover exact filename if omitted)")
    ap.add_argument("--s6", help="Castellani S6 table (autodiscover exact filename if omitted)")
    ap.add_argument("--bihler-eti", required=True, help="Bihler 2024 table with actual ETI-labelled values")
    ap.add_argument("--bihler-target", required=True, help="EXACT numeric Bihler ETI column (no guessing)")
    ap.add_argument("--bihler-filter-col", help="For mixed-condition Bihler source: explicit treatment/arm column")
    ap.add_argument("--bihler-filter-value", help="ETI arm value to retain; must accompany --bihler-filter-col")
    ap.add_argument("--s6-baseline", default="baseline_mature_fraction")
    ap.add_argument("--s6-treated", default="eti_mature_fraction")
    ap.add_argument("--vti-meta", default="data/processed/bihler_2026_transfer_splits.parquet",
                    help="Only identifiers/splits read; never response labels in audit")
    ap.add_argument("--vti-train", help="VTI TRAIN ONLY, with frozen folds and named VTI outcome(s)")
    ap.add_argument("--vti-target", action="append", default=[], help="name=EXACT_COLUMN; repeat for targets")
    ap.add_argument("--fold-col", default="fold", help="Existing five position-held-out VTI fold labels")
    ap.add_argument("--features", help="Genotype-only feature bank for VTI & ALL auxiliary substitutions")
    ap.add_argument("--feature-cols", help="Frozen genotype feature names, one per line or JSON list")
    ap.add_argument("--feature-prefix", help="Alternative explicit genotype-only prefix (e.g. seq_)")
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--teacher-alpha", type=float, default=10.0)
    ap.add_argument("--downstream-alpha", type=float, default=10.0)
    ap.add_argument("--min-teacher-rows", type=int, default=10)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--expect-rows", type=int, default=210)
    ap.add_argument("--expect-positions", type=int, default=168)
    args = ap.parse_args(argv)
    if args.teacher_alpha <= 0 or args.downstream_alpha <= 0 or args.bootstrap < 100:
        ap.error("alphas must be positive and bootstrap reps >=100")
    args.out.mkdir(parents=True, exist_ok=True)
    p = {"s5": resolve_file(args.s5, S5_NAME), "s6": resolve_file(args.s6, S6_NAME),
         "bihler": resolve_file(args.bihler_eti)}
    names = frozen_columns(args)
    if args.features and names is None:
        ap.error("--features requires --feature-cols or --feature-prefix; cannot guess genotype columns")
    if names and not args.features:
        ap.error("--feature-cols / --feature-prefix requires --features")
    bank = None
    if args.features:
        p["features"] = resolve_file(args.features)
        bank = load_feature_bank(p["features"], names)
    vti_meta = None
    if args.vti_meta:
        vti_meta_path = Path(args.vti_meta)
        if vti_meta_path.exists():
            vti_meta = load_vti_ids(vti_meta_path)
        elif args.stage == "audit":
            print(f"WARNING: VTI metadata missing: {vti_meta_path}; VTI coverage unavailable")
    # Audit stage never opens VTI outcome tables. Pilot opens it AFTER auxiliary audit.
    aux, info = audit(args, p, bank, vti_meta)
    if args.stage == "audit":
        save_json(args.out / "phase31i2_provenance.json", info)
        print("\nPASS: audit-only; NO VTI response labels loaded or model fitting performed")
        return
    if bank is None or not names:
        ap.error("Pilot requires --features and frozen --feature-cols or --feature-prefix")
    if not args.vti_train:
        ap.error("Pilot requires explicit frozen --vti-train TRAIN-only file")
    p["vti_train"] = resolve_file(args.vti_train)
    pilot(args, aux, info, names, bank, p)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, AssertionError, RuntimeError) as exc:
        print(f"\nPHASE 31I.2 STOP (no silent fallback): {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
