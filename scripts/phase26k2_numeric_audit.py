#!/usr/bin/env python3
"""Phase 26K-2: Tedman phenotype-format and corrector association audit.

TRAIN ONLY. No protected ETI/VTI outcomes.
Parse numeric values conservatively; unrecognised formats remain missing.
"""

from pathlib import Path
import sys
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

from phase26k_trafficking_preflight import (
    load_allowed,
    mutation_code,
    position_from_code,
)

BOOK = ROOT / "data/external/tedman2025/cftr_expression.xlsx"
OUT = ROOT / "results/biology/phase26k"

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


def self_test():
    examples = [
        (5.2, 5.2),
        ("2.5", 2.5),
        ("12.1 ± 1.4", 12.1),
        ("1.2e2", 120.0),
        ("5.2 +/- 0.3", 5.2),
    ]

    for value, expected in examples:
        actual, _ = parse_value(value)
        assert np.isclose(actual, expected)

    for value in [
        "1.5, 2.6",
        "5.2 (3.1)",
        "=AVERAGE(A1:A3)",
        "ND",
    ]:
        actual, _ = parse_value(value)
        assert np.isnan(actual)

    print("PASS: Conservative numeric parser self-test")


def main():
    self_test()

    if not BOOK.is_file():
        raise FileNotFoundError(BOOK)

    # Reuse the protected-partition-safe loader from Phase26K.
    v, e, protected = load_allowed()

    raw = pd.read_excel(
        BOOK,
        sheet_name="Sheet4"
    )

    cols = [
        c for c in raw.columns
        if "Surface Immuno-staining Intensity" in str(c)
    ]

    if len(cols) != 8:
        raise ValueError(
            f"Expected 8 phenotype columns, found {len(cols)}"
        )

    raw["mutation"] = raw["Variant"].map(mutation_code)

    raw["position"] = raw.mutation.map(
        position_from_code
    )

    ext = raw.loc[
        raw.mutation.notna()
        & ~raw.position.isin(protected)
    ].copy()

    print("=" * 100)
    print(
        "PHASE 26K-2 | TEDMAN NUMERIC FORMAT "
        "AND CORRECTOR ASSOCIATION AUDIT"
    )
    print("=" * 100)

    print(f"Workbook rows: {len(raw)}")
    print(f"Eligible parsed rows: {len(ext)}")
    print(f"Unique mutations: {ext.mutation.nunique()}")

    print(
        "Exact VTI TRAIN matches:",
        len(set(ext.mutation) & set(v.mutation))
    )

    print(
        "Exact ETI TRAIN matches:",
        len(set(ext.mutation) & set(e.mutation))
    )

    print(
        "NOTE: Surface immunostaining is not "
        "functional rescue in %WT."
    )

    reports = []
    associations = []

    parsed_table = ext[
        ["mutation", "position"]
    ].copy()

    for c in cols:
        series = ext[c]

        parsed = series.map(parse_value)

        numeric = pd.Series(
            [x[0] for x in parsed],
            index=ext.index,
            dtype=float
        )

        formats = pd.Series(
            [x[1] for x in parsed],
            index=ext.index
        )

        parsed_table[str(c)] = numeric

        n_valid = int(numeric.notna().sum())
        n_bad = int(
            (formats == "UNRECOGNISED").sum()
        )

        print(f"\nCOLUMN: {c}")

        print(
            f"Non-empty: {series.notna().sum()} | "
            f"Parsed: {n_valid} | "
            f"Unrecognised: {n_bad}"
        )

        print(
            "Format counts:",
            formats.value_counts().to_dict()
        )

        print(
            "Raw examples:",
            [
                repr(x)[:100]
                for x in series.dropna().head(5).tolist()
            ]
        )

        if n_bad:
            print(
                "UNRECOGNISED examples:",
                [
                    repr(x)[:100]
                    for x in series.loc[
                        formats.eq("UNRECOGNISED")
                    ].head(5).tolist()
                ]
            )

        reports.append({
            "column": str(c),
            "nonempty": int(series.notna().sum()),
            "numeric": n_valid,
            "unrecognised": n_bad,
            "format_counts": str(
                formats.value_counts().to_dict()
            ),
        })

        if n_valid < 5:
            continue

        q = ext[["mutation"]].copy()
        q["phenotype"] = numeric.to_numpy()

        q = (
            q.dropna(subset=["phenotype"])
            .groupby("mutation", as_index=False)
            .phenotype.mean()
        )

        for label, frame, ycol in [
            ("VTI corrector", v, "vti_corrector"),
            ("ETI corrector", e, "eti_corrector"),
        ]:
            paired = q.merge(
                frame[["mutation", ycol]],
                on="mutation",
                validate="one_to_one"
            )

            paired = paired.loc[
                np.isfinite(paired[ycol])
            ].copy()

            n = len(paired)

            if (
                n >= 5
                and paired.phenotype.nunique() > 1
                and paired[ycol].nunique() > 1
            ):
                rho = float(
                    spearmanr(
                        paired.phenotype,
                        paired[ycol]
                    ).statistic
                )
            else:
                rho = np.nan

            associations.append({
                "column": str(c),
                "target": label,
                "n_exact": n,
                "spearman_rho": rho,
            })

    OUT.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(reports).to_csv(
        OUT / "phase26k2_parse_qc.csv",
        index=False
    )

    parsed_table.to_csv(
        OUT / "phase26k2_parsed_surface_train.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("EXPLORATORY EXACT-MUTATION ASSOCIATIONS")
    print("=" * 100)

    if associations:
        results = pd.DataFrame(associations)

        print(
            results.round(3).to_string(index=False)
        )

        results.to_csv(
            OUT / "phase26k2_correlations.csv",
            index=False
        )
    else:
        print(
            "No usable associations. Inspect raw examples; "
            "do not infer absence of biological signal."
        )

    print("\nPASS: Phase26K-2 completed")
    print("Protected ETI/VTI outcomes unopened")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
