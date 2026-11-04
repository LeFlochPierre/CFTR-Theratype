"""Canonical Phase31 substitution identifier and position checks.

No phenotype/outcome columns are accessed here.
"""
from __future__ import annotations

import re
import pandas as pd

THREE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C", "Gln": "Q",
    "Glu": "E", "Gly": "G", "His": "H", "Ile": "I", "Leu": "L", "Lys": "K",
    "Met": "M", "Phe": "F", "Pro": "P", "Ser": "S", "Thr": "T", "Trp": "W",
    "Tyr": "Y", "Val": "V", "Ter": "*", "Sec": "U",
}
ID_CANDIDATES = ["aa_code", "hgvs_protein", "protein_change", "protein_variant", "variant"]
POSITION_CANDIDATES = ["protein_position", "position", "pos31i"]
def norm_aa(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip().replace(" ", "")
    s = re.sub(r"^p\.", "", s, flags=re.I)
    m = re.fullmatch(r"([A-Za-z]{1,3})(\d{1,4})([A-Za-z]{1,3}|\*)", s)
    if not m:
        return None
    a, n, b = m.groups()
    def one(t):
        if len(t) == 1 and t.upper() in "ACDEFGHIKLMNPQRSTVWY*U":
            return t.upper()
        return THREE.get(t.capitalize())
    a, b = one(a), one(b)
    if a is None or b is None or a == b:
        return None
    pos = int(n)
    if not 1 <= pos <= 1480:
        return None
    return f"{a}{pos}{b}"


def with_ids(df: pd.DataFrame, name: str, strict: bool = False) -> pd.DataFrame:
    d = df.copy()
    opts = [c for c in ID_CANDIDATES if c in d]
    if not opts:
        raise ValueError(f"{name}: no supported protein-substitution ID; columns: {list(d)}")
    normalized = pd.Series([None] * len(d), index=d.index, dtype="object")
    for c in opts:
        normalized = normalized.combine_first(d[c].map(norm_aa))
    d["aa31i"] = normalized
    derived = pd.to_numeric(d["aa31i"].str.extract(r"^[A-Z*](\d+)[A-Z*]$", expand=False), errors="coerce")
    d["pos31i2"] = derived.astype("Int64")
    # Position can be present for non-missense variants, but they are excluded
    # from substitution-level teacher fits unless representation is specified.
    for c in POSITION_CANDIDATES:
        if c in d:
            given = pd.to_numeric(d[c], errors="coerce").astype("Int64")
            conflict = given.notna() & d.pos31i2.notna() & (given != d.pos31i2)
            if bool(conflict.any()):
                raise ValueError(f"{name}: {int(conflict.sum())} conflicting positions in '{c}' vs protein ID")
            d["pos31i2"] = d["pos31i2"].fillna(given)
            break
    if "is_control" in d:
        flag = d.is_control.astype("string").str.strip().str.lower()
        if (~flag.isin(["true", "false", "1", "0", "yes", "no", "nan", "<na>"])).any():
            raise ValueError(f"{name}: ambiguous values of is_control: {flag.unique().tolist()[:15]}")
        control = flag.isin(["true", "1", "yes"])
    else:
        control = pd.Series(False, index=d.index)
    if "mutation_type" in d:
        mt = d.mutation_type.astype("string").str.lower().str.strip()
        control |= mt.str.contains(r"control|wild.?type|^wt$", regex=True, na=False)
    d["control31i"] = control.fillna(False).astype(bool)
    if strict and (d["aa31i"].isna() | d["pos31i2"].isna() | d["control31i"]).any():
        bad = d[d.aa31i.isna() | d.pos31i2.isna() | d.control31i]
        raise ValueError(f"{name}: {len(bad)} VTI rows lack missense substitution, position, or are controls")
    return d
