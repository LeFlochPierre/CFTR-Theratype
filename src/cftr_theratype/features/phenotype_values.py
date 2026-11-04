"""Conservative phenotype spreadsheet numeric parsing from Phase26K-2.

Do not infer values from ranges, formulas, multi-value cells or ambiguous
formats. Preserve the original classification labels for audit reports.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

SINGLE = re.compile(
    rf"^\s*({NUMBER})\s*%?\s*$"
)

PLUSMINUS = re.compile(
    rf"^\s*({NUMBER})\s*(?:±|\+/-)\s*({NUMBER})\s*%?\s*$"
)


def parse_value(value):
    if pd.isna(value):
        return np.nan, "missing"

    if isinstance(
        value,
        (int, float, np.integer, np.floating)
    ):
        return (
            (float(value), "numeric")
            if np.isfinite(value)
            else (np.nan, "missing")
        )

    s = (
        str(value)
        .strip()
        .replace("\u2212", "-")
        .replace("\u2013", "-")
        .replace("\xa0", " ")
    )

    missing = {
        "", "n/a", "na", "nd", "nan",
        "none", "-", "not determined"
    }

    if s.casefold() in missing:
        return np.nan, "missing"

    m = SINGLE.fullmatch(s)

    if m:
        return float(m.group(1)), "numeric_string"

    m = PLUSMINUS.fullmatch(s)

    if m:
        return float(m.group(1)), "mean_plusminus"

    return np.nan, "UNRECOGNISED"
