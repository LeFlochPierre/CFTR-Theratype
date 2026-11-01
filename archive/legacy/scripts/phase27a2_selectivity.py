#!/usr/bin/env python3
"""Phase 27A-2: VX-121 vs VX-445 selectivity.

Exploratory TRAIN-only analysis.
No model fitting or protected outcomes.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata, pearsonr

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

SRC = ROOT / "data/external/jhangiani2026_vx121"
OUT = ROOT / "results/biology/phase27a"

PRIMARY = {
    "vehicle121": "VC-DMSO AVERAGE SCALED INTENSITY",
    "treated121": "VX-121 AVERAGE SCALED INTENSITY",
    "es121": "VX-121 ES",
    "vehicle445": "VC Average Scaled Intensity",
    "treated445": "VX-445 Average Scaled Intensity",
    "es445": "VX445 EFFECT SIZE",
}

ALTERNATE = {
    "vehicle121": "VC -AVG SCALED INTENSITY (For VX-121)",
    "treated121": "VX-121 AVG SCALED INTENSITY",
    "vehicle445": "VC - Average Scaled Int",
    "treated445": "VX-445 Average Scaled Int",
}


def log2_ratio(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    result = np.full(len(a), np.nan)

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (a > 0)
        & (b > 0)
    )

    result[valid] = np.log2(a[valid] / b[valid])

    return result


def read_sheet(
    book, sheet, columns,
    mutation_code, position_from_code, protected
):
    raw = pd.read_excel(book, sheet_name=sheet)

    required = ["VARIANT", *columns.values()]

    missing = [
        c for c in required
        if c not in raw.columns
    ]

    if missing:
        raise ValueError(
            f"{sheet}: missing columns {missing}"
        )

    raw["mutation"] = raw.VARIANT.map(mutation_code)
    raw["position"] = raw.mutation.map(
        position_from_code
    )

    unparsed = raw.loc[
        raw.VARIANT.notna() & raw.mutation.isna(),
        "VARIANT"
    ].astype(str).head(15).tolist()

    eligible = raw.loc[
        raw.mutation.notna()
        & ~raw.position.isin(protected)
    ].copy()

    for short, original in columns.items():
        eligible[short] = pd.to_numeric(
            eligible[original],
            errors="coerce"
        )

    eligible["vx121_log2fc"] = log2_ratio(
        eligible.treated121,
        eligible.vehicle121
    )

    eligible["vx445_log2fc"] = log2_ratio(
        eligible.treated445,
        eligible.vehicle445
    )

    eligible["selectivity_log2"] = (
        eligible.vx121_log2fc
        - eligible.vx445_log2fc
    )

    eligible["normalised_response_ratio"] = (
        np.exp2(eligible.selectivity_log2)
    )

    measures = [
        "vehicle121", "treated121",
        "vehicle445", "treated445"
    ]

    clean = []
    conflicts = []
    duplicate_qc = []

    for mutation, group in eligible.groupby(
        "mutation", sort=True
    ):
        if len(group) > 1:
            consistent = (
                group[measures].notna().all().all()
                and np.allclose(
                    group[measures].to_numpy(float),
                    group[measures].iloc[0].to_numpy(float),
                    atol=1e-8,
                    rtol=1e-8
                )
            )

            record = {
                "sheet": sheet,
                "mutation": mutation,
                "rows": len(group),
                "consistent": consistent
            }

            for col in measures:
                record[f"range_{col}"] = float(
                    group[col].max() - group[col].min()
                )

            duplicate_qc.append(record)

            if not consistent:
                conflicts.append(group)
                continue

        clean.append(group.iloc[[0]])

    clean = (
        pd.concat(clean, ignore_index=True)
        if clean else eligible.iloc[:0].copy()
    )

    conflicts = (
        pd.concat(conflicts, ignore_index=True)
        if conflicts else eligible.iloc[:0].copy()
    )

    selected = [
        "mutation", "position",
        *columns.keys(),
        "vx121_log2fc",
        "vx445_log2fc",
        "selectivity_log2",
        "normalised_response_ratio"
    ]

    clean = clean[selected].copy()

    if clean.mutation.duplicated().any():
        raise ValueError(
            "Nonunique substitutions after QC"
        )

    if set(clean.position.astype(int)) & set(protected):
        raise ValueError(
            "Protected positions present"
        )

    return (
        raw,
        eligible,
        clean,
        conflicts,
        pd.DataFrame(duplicate_qc),
        unparsed
    )


def correlation(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    mask = np.isfinite(a) & np.isfinite(b)

    if mask.sum() < 5:
        return np.nan

    if (
        np.unique(a[mask]).size < 2
        or np.unique(b[mask]).size < 2
    ):
        return np.nan

    return float(
        spearmanr(a[mask], b[mask]).statistic
    )


def partial_spearman(d, x, y, control):
    """Exploratory rank association after adjustment."""

    z = d[
        [x, y, control]
    ].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(z) < 20:
        return np.nan

    ranks = np.column_stack([
        rankdata(z[col])
        for col in (x, y, control)
    ])

    X = np.column_stack([
        np.ones(len(z)),
        ranks[:, 2]
    ])

    rx = (
        ranks[:, 0]
        - X @ np.linalg.lstsq(
            X, ranks[:, 0], rcond=None
        )[0]
    )

    ry = (
        ranks[:, 1]
        - X @ np.linalg.lstsq(
            X, ranks[:, 1], rcond=None
        )[0]
    )

    if (
        np.std(rx) < 1e-12
        or np.std(ry) < 1e-12
    ):
        return np.nan

    return float(
        pearsonr(rx, ry).statistic
    )


def self_test():
    x = log2_ratio(
        [4, 2, 0],
        [1, 2, 1]
    )

    assert np.allclose(x[:2], [2, 0])
    assert np.isnan(x[2])

    assert np.isclose(
        correlation(
            [1, 2, 3, 4, 5],
            [2, 4, 6, 8, 10]
        ),
        1
    )

    print(
        "PASS: fold-change calculations, "
        "zero handling and correlations"
    )


def main():
    self_test()

    from phase26k_trafficking_preflight import (
        load_allowed,
        mutation_code,
        position_from_code
    )

    v, e, protected = load_allowed()

    books = list(
        SRC.glob("*StatisticalModels*.xlsx")
    )

    if len(books) != 1:
        raise FileNotFoundError(
            "Expected one VX-121 StatisticalModels "
            f"workbook; found {books}"
        )

    book = books[0]

    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print(
        "PHASE 27A-2 | VX-121 VS VX-445 "
        "DRUG-SELECTIVITY ANALYSIS"
    )
    print("=" * 104)

    print("Workbook:", book.name)
    print(
        "VTI/ETI training outcomes only; "
        "protected positions excluded"
    )
    print(
        "No model fitting or protected outcome access"
    )

    (
        raw, eligible, clean,
        conflicts, duplicates, unparsed
    ) = read_sheet(
        book,
        "Effect Size Comparison",
        PRIMARY,
        mutation_code,
        position_from_code,
        protected
    )

    print("\nA. SOURCE AND DUPLICATE QC")

    print(
        "Raw sheet rows:",
        len(raw)
    )
    print(
        "Parsed unique substitutions:",
        raw.mutation.nunique()
    )
    print(
        "Eligible rows:",
        len(eligible)
    )
    print(
        "Eligible unique substitutions:",
        eligible.mutation.nunique()
    )
    print(
        "Retained unambiguous substitutions:",
        len(clean)
    )
    print(
        "Quarantined conflicting rows:",
        len(conflicts)
    )

    print(
        "Unparsed variant examples:",
        unparsed
    )

    if len(duplicates):
        print("\nRepeated protein substitutions:")
        print(
            duplicates.round(4).to_string(
                index=False
            )
        )

    duplicates.to_csv(
        OUT / "phase27a2_duplicate_qc.csv",
        index=False
    )

    conflicts.to_csv(
        OUT / "phase27a2_conflicting_duplicates.csv",
        index=False
    )

    (
        alt_raw, alt_eligible, alt,
        alt_conflicts, alt_duplicates, _
    ) = read_sheet(
        book,
        "Condensed Comparison Sheet",
        ALTERNATE,
        mutation_code,
        position_from_code,
        protected
    )

    print("\nB. CROSS-SHEET RECONCILIATION")

    first = set(clean.mutation)
    second = set(alt.mutation)

    print(
        "Effect Size Comparison only:",
        sorted(first - second)
    )

    print(
        "Condensed Comparison only:",
        sorted(second - first)
    )

    paired = clean.merge(
        alt,
        on="mutation",
        suffixes=("_primary", "_condensed"),
        validate="one_to_one"
    )

    for feature in (
        "vx121_log2fc",
        "vx445_log2fc",
        "selectivity_log2"
    ):
        difference = np.abs(
            paired[f"{feature}_primary"]
            - paired[f"{feature}_condensed"]
        )

        print(
            f"{feature}: "
            f"n={difference.notna().sum()}, "
            f"median discrepancy={difference.median():.5f}, "
            f"max discrepancy={difference.max():.5f}"
        )

    if len(alt_conflicts):
        print(
            "Condensed-sheet conflicting substitutions:",
            alt_conflicts.mutation.nunique()
        )

    print("\nC. VEHICLE-NORMALISED RESPONSE DISTRIBUTIONS")

    features = [
        "vx121_log2fc",
        "vx445_log2fc",
        "selectivity_log2",
        "normalised_response_ratio"
    ]

    for feature in features:
        x = clean[feature].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()

        print(
            f"{feature}: n={len(x)}, "
            f"median={x.median():.4f}, "
            f"p05={x.quantile(.05):.4f}, "
            f"p95={x.quantile(.95):.4f}"
        )

    print(
        "VX-121 vs VX-445 fold-response Spearman:",
        round(
            correlation(
                clean.vx121_log2fc,
                clean.vx445_log2fc
            ), 4
        )
    )

    print(
        "Fraction with greater relative "
        "VX-121 response:",
        round(
            float(
                (
                    clean.selectivity_log2 > 0
                ).mean()
            ), 4
        )
    )

    print(
        "\nD. MUTATIONS WITH EXTREME DRUG SELECTIVITY"
    )

    clean["in_eti_train"] = clean.mutation.isin(
        set(e.mutation)
    )

    clean["in_vti_train"] = clean.mutation.isin(
        set(v.mutation)
    )

    display = [
        "mutation",
        "position",
        "vx121_log2fc",
        "vx445_log2fc",
        "selectivity_log2",
        "in_eti_train",
        "in_vti_train"
    ]

    for title, ascending in (
        ("Relative VX-445 preference", True),
        ("Relative VX-121 preference", False)
    ):
        print("\n" + title)

        print(
            clean.sort_values(
                "selectivity_log2",
                ascending=ascending
            )[display]
            .head(10)
            .round(3)
            .to_string(index=False)
        )

    print(
        "\nE. TRAIN-ONLY FUNCTIONAL RESPONSE ASSOCIATIONS"
    )

    rows = []

    targets = [
        ("ETI corrector", e, "eti_corrector"),
        ("VTI corrector", v, "vti_corrector")
    ]

    phenotypes = [
        "vx121_log2fc",
        "vx445_log2fc",
        "selectivity_log2",
        "es121",
        "es445"
    ]

    for label, dataset, target in targets:
        joined = clean.merge(
            dataset[["mutation", target]],
            on="mutation",
            validate="one_to_one"
        )

        print(
            f"\n{label}: "
            f"{len(joined)} exact matches / "
            f"{joined.position.nunique()} positions"
        )

        for phenotype in phenotypes:
            subset = joined[
                ["position", phenotype, target]
            ].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()

            by_position = (
                subset.groupby(
                    "position", as_index=False
                )[[phenotype, target]]
                .mean()
            )

            rows.append({
                "target": label,
                "phenotype": phenotype,
                "mutations": len(subset),
                "positions": len(by_position),
                "rho_variant": correlation(
                    subset[phenotype],
                    subset[target]
                ),
                "rho_position": correlation(
                    by_position[phenotype],
                    by_position[target]
                )
            })

        if label == "ETI corrector":
            partial = partial_spearman(
                joined,
                "vx121_log2fc",
                target,
                "vx445_log2fc"
            )

            print(
                "Partial rank association: VX-121 "
                "with ETI corrector, adjusting for VX-445:",
                round(partial, 4)
            )

    associations = pd.DataFrame(rows)

    print(
        associations.round(4).to_string(
            index=False
        )
    )

    clean.to_csv(
        OUT / "phase27a2_selectivity_train.csv",
        index=False
    )

    associations.to_csv(
        OUT / "phase27a2_associations.csv",
        index=False
    )

    paired.to_csv(
        OUT / "phase27a2_cross_sheet_qc.csv",
        index=False
    )

    print("\nF. INTERPRETATION")

    print(
        "Surface-expression selectivity is not "
        "the same as functional VTI rescue."
    )

    print(
        "All associations are exploratory. "
        "VX-121 and VX-445 are highly related assays."
    )

    print(
        "Do not select model features from "
        "correlations with protected outcomes."
    )

    print("\nPASS: Phase27A-2 completed.")
    print("No model fitting or protected outcome access.")
    print("Saved:", OUT)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        main()
