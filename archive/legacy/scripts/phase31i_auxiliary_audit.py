#!/usr/bin/env python3
"""CFTR-Theratype Phase 31I.1: position/provenance audit for ETI auxiliary transfer.

From project root: python scripts/phase31i_auxiliary_audit.py
Only auxiliary ETI/maturation labels and mutation identifiers are read.
Never loads VTI response labels or fits a predictive model.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / "results/biology/phase31i"
REQUIRED = {
    "castellani_s5": "phase31a_castellani_s5_transport.csv",
    "castellani_s6": "phase31a_castellani_s6_processing.csv",
    "vti_overlap": "phase31a_vti_train_overlap.csv",
    "bihler_overlap": "phase31a_bihler_eti_overlap.csv",
}
OPTIONAL = {
    "bihler_universe": ROOT / "data/processed/bihler_2024_theratyping.parquet",
    "vti_partitions": ROOT / "data/processed/bihler_2026_transfer_splits.parquet",
}
ID_COLS = [
    "variant_key", "aa_code", "hgvs_protein", "protein_position", "position",
    "variant", "study", "study_cohort", "split", "transfer_split",
    "eti_split", "partition", "set", "group",
]
SPLIT_COLS = ["transfer_split", "split", "partition", "set"]


def find_unique(filename: str) -> Path:
    hits = sorted(p for p in ROOT.rglob(filename) if OUT not in p.parents)
    if not hits:
        raise FileNotFoundError(f"Cannot locate {filename} below {ROOT}")
    if len(hits) > 1:
        hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in hits}
        if len(hashes) != 1:
            raise RuntimeError(f"Ambiguous non-identical copies of {filename}: {hits}")
        print(f"WARNING: {len(hits)} identical copies of {filename}; using {hits[0]}")
    return hits[0]


def read_ids(path: Path) -> pd.DataFrame:
    """Read only identifiers/partitions; do not load any VTI outcome columns."""
    if path.suffix == ".csv":
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        keep = [c for c in cols if c in ID_COLS]
        if not keep:
            raise ValueError(f"No recognized ID columns in {path.name}: {cols}")
        return pd.read_csv(path, usecols=keep, low_memory=False)
    import pyarrow.parquet as pq
    cols = pq.read_schema(path).names
    keep = [c for c in cols if c in ID_COLS]
    if not keep:
        raise ValueError(f"No recognized ID columns in {path.name}: {cols}")
    return pd.read_parquet(path, columns=keep)


def protein_pos(frame: pd.DataFrame) -> pd.Series:
    for col in ("protein_position", "position"):
        if col in frame:
            vals = pd.to_numeric(frame[col], errors="coerce")
            if vals.notna().any():
                return vals.astype("Int64")
    for col in ("aa_code", "hgvs_protein", "variant"):
        if col in frame:
            raw = frame[col].astype("string").str.strip()
            # Only parse protein notation: L973F, p.L973F, p.Leu973Phe, p.Phe508del.
            pattern = r"^(?:p\.)?(?:[A-Z][a-z]{0,2})(\d+)(?:[A-Z][a-z]{0,2}|del|dup|Ter|\*)$"
            vals = pd.to_numeric(raw.str.extract(pattern, expand=False), errors="coerce")
            if vals.notna().any():
                return vals.astype("Int64")
    raise ValueError(f"Cannot identify protein positions; available columns: {frame.columns.tolist()}")


def with_position(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    d = frame.copy()
    d["pos31i"] = protein_pos(d)
    miss = int(d.pos31i.isna().sum())
    if miss:
        print(f"WARNING {label}: {miss}/{len(d)} rows lack parseable protein position")
    return d


def posset(frame: pd.DataFrame) -> set[int]:
    return set(map(int, frame.pos31i.dropna().unique()))


def run() -> None:
    print("=" * 82)
    print("PHASE 31I.1 | CASTELLANI AUXILIARY COVERAGE + POSITION AUDIT")
    print("No VTI outcomes loaded | No model fitted | No TEST or VAL response labels")
    print("=" * 82)
    OUT.mkdir(parents=True, exist_ok=True)
    paths = {key: find_unique(name) for key, name in REQUIRED.items()}
    for k, p in paths.items():
        print(f"{k:>18}: {p.relative_to(ROOT)}")

    s5 = with_position(pd.read_csv(paths["castellani_s5"], low_memory=False), "S5")
    s6 = with_position(pd.read_csv(paths["castellani_s6"], low_memory=False), "S6")
    vti = with_position(read_ids(paths["vti_overlap"]), "VTI-overlap metadata")
    bihler_overlap = with_position(read_ids(paths["bihler_overlap"]), "Bihler-overlap metadata")

    if "study" not in s5:
        raise ValueError("S5 is missing 'study'; cannot keep P289/U032 separate")
    s5["study"] = s5.study.astype(str).str.upper().str.strip()
    study_set = set(s5.study.unique())
    if not {"P289", "U032"}.issubset(study_set):
        raise ValueError(f"Expected P289 and U032 in S5; got {sorted(study_set)}")
    s5 = s5[s5.study.isin(["P289", "U032"])].copy()
    if "eti_rescue_pp" not in s5.columns:
        raise ValueError("S5 has no 'eti_rescue_pp' column; inspect columns before modelling: " + str(s5.columns.tolist()))
    s5["eti_rescue_pp"] = pd.to_numeric(s5.eti_rescue_pp, errors="coerce")
    if not s5.eti_rescue_pp.notna().any():
        raise ValueError("S5 ETI rescue has no numeric observations")
    if "study" in s6:
        s6["study"] = s6.study.astype(str).str.upper().str.strip()

    # The overlap file may contain ONLY matched variants: do not call its position set
    # the entire 210-variant VTI training universe.
    vti_overlap_positions = posset(vti)
    bihler_overlap_positions = posset(bihler_overlap)
    universe = None
    if OPTIONAL["bihler_universe"].exists():
        universe = with_position(read_ids(OPTIONAL["bihler_universe"]), "Bihler processed variant universe")
        bihler_positions = posset(universe)
        reference = "Bihler 2024 processed variant universe (not necessarily ETI-labelled)"
    else:
        bihler_positions = bihler_overlap_positions
        reference = "Bihler overlap file ONLY (not full auxiliary coverage)"
        print("WARNING: Full Bihler processed universe not found; novel-position counts are upper bounds.")

    vti_train_positions = None
    if OPTIONAL["vti_partitions"].exists():
        parts = read_ids(OPTIONAL["vti_partitions"])
        spl = next((c for c in SPLIT_COLS if c in parts), None)
        if spl is not None:
            parts = with_position(parts, "VTI split metadata")
            train = parts[parts[spl].astype(str).str.lower() == "train"]
            if len(train):
                vti_train_positions = posset(train)
                print(f"VTI TRAIN metadata: {len(train)} rows, {len(vti_train_positions)} positions ({spl})")
            else:
                print(f"WARNING: '{spl}' does not contain a 'train' partition; skipping full VTI TRAIN coverage")
        else:
            print("WARNING: VTI partition file has no recognized split field")
    if vti_train_positions is None:
        print("WARNING: Cannot infer complete VTI TRAIN universe; overlap counts only below.")

    print("\nS5 numeric auxiliary labels by assay:")
    print(s5.groupby("study")["eti_rescue_pp"].agg(["count", "median", "min", "max"]).to_string())
    print("\nReference:", reference)
    print("S6 columns:", s6.columns.tolist())

    all_s5_pos = posset(s5)
    rows = []
    for study in ("P289", "U032", "ALL_UNION"):
        d = s5 if study == "ALL_UNION" else s5[s5.study == study]
        p = posset(d)
        rows.append({
            "assay": study,
            "n_rows": len(d),
            "n_ETI_labelled_rows": int(d.eti_rescue_pp.notna().sum()),
            "n_variant_keys": int(d.variant_key.nunique()) if "variant_key" in d else np.nan,
            "n_aa_substitutions": int(d.aa_code.nunique()) if "aa_code" in d else np.nan,
            "n_protein_positions": len(p),
            "n_positions_in_Bihler_reference": len(p & bihler_positions),
            "n_positions_NOT_in_Bihler_reference": len(p - bihler_positions),
            "n_positions_in_VTI_overlap_table": len(p & vti_overlap_positions),
            "n_positions_in_full_VTI_TRAIN": len(p & vti_train_positions) if vti_train_positions is not None else np.nan,
        })
    coverage = pd.DataFrame(rows)
    print("\nCoverage by assay (position-level, not independent variant counts):")
    print(coverage.to_string(index=False))

    s6pos = posset(s6)
    per_position = []
    for p in sorted(all_s5_pos):
        per_position.append({
            "protein_position": p,
            "in_P289": p in posset(s5[s5.study == "P289"]),
            "in_U032": p in posset(s5[s5.study == "U032"]),
            "in_S6_maturation_table": p in s6pos,
            "in_Bihler_reference": p in bihler_positions,
            "in_VTI_overlap_table": p in vti_overlap_positions,
            "in_full_VTI_TRAIN": p in vti_train_positions if vti_train_positions is not None else pd.NA,
        })
    manifest = pd.DataFrame(per_position)
    coverage.to_csv(OUT / "phase31i_coverage_by_study.csv", index=False)
    manifest.to_csv(OUT / "phase31i_position_manifest.csv", index=False)
    result = {
        "phase": "31I.1",
        "no_vti_outcomes_loaded": True,
        "no_models_fitted": True,
        "reference_definition": reference,
        "source_paths": {k: str(v.relative_to(ROOT)) for k, v in paths.items()},
        "s5_columns": list(s5.columns),
        "s6_columns": list(s6.columns),
        "vti_overlap_metadata_columns": list(vti.columns),
        "bihler_overlap_metadata_columns": list(bihler_overlap.columns),
        "coverage_by_assay": coverage.astype(object).where(coverage.notna(), None).to_dict(orient="records"),
        "warnings": [
            "Position overlaps do not imply identical assay labels.",
            "S6 position overlap does not prove variant-specific matched maturation.",
            "For model fitting use Phase26D frozen position-held-out folds and fit all auxiliary teachers strictly within each outer TRAIN position set.",
            "VTI external validation and test remain untouched.",
        ],
    }
    # Work around pandas/numpy scalar JSON conversion.
    def obj_json(x):
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return None if np.isnan(x) else float(x)
        raise TypeError(type(x).__name__)
    (OUT / "phase31i_audit.json").write_text(json.dumps(result, indent=2, default=obj_json, allow_nan=False))
    print("\nSaved:")
    for name in ("phase31i_coverage_by_study.csv", "phase31i_position_manifest.csv", "phase31i_audit.json"):
        print(" ", OUT / name)
    print("PASS: metadata + auxiliary-label audit only; no VTI outcomes or models accessed")


if __name__ == "__main__":
    run()
