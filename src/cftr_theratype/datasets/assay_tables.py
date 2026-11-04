"""Canonical Phase31 assay table normalization, provenance and negative controls.

Pure table transformations and a root-explicit assay file resolver. Biological
labels are handled only when passed explicitly as source DataFrames; no VTI
response files are loaded here.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, theilslopes

from cftr_theratype.datasets.assay_identifiers import with_ids

S5_NAME = "phase31a_castellani_s5_transport.csv"
S6_NAME = "phase31a_castellani_s6_processing.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_file(requested: str | None, fallback: str | None = None, *, root: Path | None = None) -> Path:
    root = Path.cwd() if root is None else Path(root)
    if requested:
        p = Path(requested).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Missing input: {p}")
        return p
    if not fallback:
        raise ValueError("Required input path not specified")
    hits = sorted(p for p in root.rglob(fallback) if "phase31i" not in p.parts)
    if not hits:
        raise FileNotFoundError(f"Could not find {fallback} under {root}")
    if len(hits) > 1 and len({sha256(p) for p in hits}) != 1:
        raise ValueError(f"Ambiguous non-identical copies of {fallback}: {hits}")
    return hits[0]


def numeric(series: pd.Series, label: str) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    bad = series.notna() & vals.isna() & series.astype("string").str.strip().ne("")
    if int(bad.sum()) > 0:
        examples = series.loc[bad].astype(str).unique()[:4]
        print(f"WARNING: {label}: {int(bad.sum())} nonnumeric values converted to missing ({list(examples)})")
    return vals.replace([np.inf, -np.inf], np.nan).astype(float)


def table_aux(df: pd.DataFrame, name: str, target: str, study: str | None = None) -> tuple[pd.DataFrame, dict]:
    if target not in df:
        raise ValueError(f"{name}: target '{target}' absent. Columns: {list(df)}")
    d = with_ids(df, name)
    if study is not None:
        if "study" not in d:
            raise ValueError(f"{name}: requires study column to separate assays")
        d = d.loc[d.study.astype(str).str.strip().str.upper() == study].copy()
    d["value31i"] = numeric(d[target], f"{name}.{target}")
    types = d.get("mutation_type", pd.Series("(missing)", index=d.index)).astype(str)
    bad = d.aa31i.isna() | d.pos31i2.isna()
    details = {k: int(v) for k, v in types.loc[bad].value_counts().items()}
    usable = d.loc[~d.control31i & ~bad & d.value31i.notna(), ["aa31i", "pos31i2", "value31i"]].copy()
    # Legitimate repeats and different nucleotide HGVS for same protein change:
    # unit of auxiliary observation is one amino-acid substitution per assay.
    dup_n = len(usable) - usable.aa31i.nunique()
    grp = usable.groupby(["aa31i", "pos31i2"], as_index=False, observed=True).value31i.median()
    grp["assay"] = name
    report = {
        "assay": name, "source_rows": int(len(d)), "valid_numeric_targets": int(d.value31i.notna().sum()),
        "excluded_controls": int(d.control31i.sum()), "unresolved_positions": int(d.pos31i2.isna().sum()),
        "unresolved_substitutions": int(d.aa31i.isna().sum()),
        "unresolved_by_mutation_type": json.dumps(details, sort_keys=True),
        "eligible_source_rows": int(len(usable)), "deduplicated_aa_repeats": int(dup_n),
        "unique_aa_substitutions": int(len(grp)), "unique_positions": int(grp.pos31i2.nunique()),
    }
    return grp, report


def agreement(a: pd.DataFrame, b: pd.DataFrame, an: str, bn: str) -> tuple[dict, pd.DataFrame]:
    x = a[["aa31i", "value31i"]].rename(columns={"value31i": "x"})
    y = b[["aa31i", "value31i"]].rename(columns={"value31i": "y"})
    merged = x.merge(y, on="aa31i", how="inner", validate="one_to_one")
    n = len(merged)
    ans = {"first": an, "second": bn, "n_exact_substitutions": n,
           "spearman": np.nan, "pearson": np.nan, "theil_sen_slope_y_on_x": np.nan,
           "median_difference_y_minus_x": np.nan}
    if n:
        ans["median_difference_y_minus_x"] = float(np.median(merged.y - merged.x))
    if n >= 3 and merged.x.nunique() > 1 and merged.y.nunique() > 1:
        ans["spearman"] = float(spearmanr(merged.x, merged.y).statistic)
        ans["pearson"] = float(pearsonr(merged.x, merged.y).statistic)
        ans["theil_sen_slope_y_on_x"] = float(theilslopes(merged.y.to_numpy(), merged.x.to_numpy()).slope)
    merged["first"] = an
    merged["second"] = bn
    return ans, merged


def permute_by_position(d: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Negative control: shuffle position medians, retaining within-position residuals."""
    d = d.copy()
    med = d.groupby("pos31i2", sort=True).value31i.median()
    positions = med.index.to_numpy()
    shuffled = rng.permutation(med.to_numpy())
    mapping = pd.Series(shuffled, index=positions)
    d["value31i"] = d.value31i - d.pos31i2.map(med) + d.pos31i2.map(mapping)
    return d
