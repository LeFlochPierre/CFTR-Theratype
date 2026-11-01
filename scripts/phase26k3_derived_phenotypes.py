#!/usr/bin/env python3
"""Phase 26K-3: Tedman treatment-effect surface phenotypes vs corrector rescue.

Exploratory, training-only phenotype audit.
No model fitting or protected outcomes.
"""

from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

BOOK = ROOT / "data/external/tedman2025/cftr_expression.xlsx"
OUT = ROOT / "results/biology/phase26k"

NAMES = {
    "p_apo": "Parental Surface Immuno-staining Intensity (Apo)",
    "p_661": "Parental Surface Immuno-staining Intensity (VX-661)",
    "p_445": "Parental Surface Immuno-staining Intensity (VX-445)",
    "p_dual": "Parental Surface Immuno-staining Intensity (VX-445 + VX-661)",
    "k_apo": "CANX KO Surface Immuno-staining Intensity (Apo)",
    "k_661": "CANX KO Surface Immuno-staining Intensity (VX-661)",
    "k_445": "CANX KO Surface Immuno-staining Intensity (VX-445)",
    "k_dual": "CANX KO Surface Immuno-staining Intensity (VX-445 + VX-661)",
}

FEATURES = [
    ("p_apo", "Untreated parental expression"),
    ("p_dual", "Dual-treated parental expression"),
    ("p_gain_dual", "Parental dual minus untreated"),
    ("p_log2fc_dual", "Parental log2(dual/untreated)"),
    ("p_gain_445", "Parental VX-445 minus untreated"),
    ("p_gain_661", "Parental VX-661 minus untreated"),
    ("p_dual_above_best", "Dual minus best individual corrector"),
    ("k_gain_dual", "CANX KO dual minus untreated"),
    ("k_log2fc_dual", "CANX KO log2(dual/untreated)"),
    ("canx_delta_log2fc", "Parental minus CANX KO fold response"),
]


def log_ratio(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    out = np.full(len(a), np.nan, float)

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (a > 0)
        & (b > 0)
    )

    out[valid] = np.log2(a[valid] / b[valid])

    return out


def make_derived(t):
    z = t.copy()

    z["p_gain_dual"] = z.p_dual - z.p_apo

    z["p_log2fc_dual"] = log_ratio(
        z.p_dual, z.p_apo
    )

    z["p_gain_445"] = z.p_445 - z.p_apo
    z["p_gain_661"] = z.p_661 - z.p_apo

    z["p_dual_above_best"] = (
        z.p_dual - np.maximum(z.p_445, z.p_661)
    )

    z["k_gain_dual"] = z.k_dual - z.k_apo

    z["k_log2fc_dual"] = log_ratio(
        z.k_dual, z.k_apo
    )

    z["canx_delta_log2fc"] = (
        z.p_log2fc_dual - z.k_log2fc_dual
    )

    return z


def self_test():
    x = pd.DataFrame({
        "p_apo": [100., 200.],
        "p_661": [200., 240.],
        "p_445": [300., 260.],
        "p_dual": [400., 350.],
        "k_apo": [50., 40.],
        "k_661": [90., 80.],
        "k_445": [120., 100.],
        "k_dual": [150., 120.],
    })

    z = make_derived(x)

    assert np.allclose(
        z.p_gain_dual, [300, 150]
    )

    assert np.allclose(
        z.p_log2fc_dual,
        [2., np.log2(1.75)]
    )

    assert np.allclose(
        z.p_dual_above_best,
        [100, 90]
    )

    assert np.allclose(
        z.k_gain_dual,
        [100, 80]
    )

    assert np.allclose(
        z.canx_delta_log2fc.iloc[0],
        2 - np.log2(3)
    )

    assert np.isnan(
        log_ratio([1.], [0.])[0]
    )

    print(
        "PASS synthetic: absolute gains, fold changes, "
        "CANX contrast, zero-denominator handling"
    )


def association(ext, labels, label, value_col):
    merged = ext.merge(
        labels[["mutation", value_col]],
        on="mutation",
        validate="one_to_one"
    )

    summary = []

    for feature, description in FEATURES:
        good = (
            merged[
                ["position", feature, value_col]
            ]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        n = len(good)

        if (
            n >= 5
            and good[feature].nunique() > 1
            and good[value_col].nunique() > 1
        ):
            rho = float(
                spearmanr(
                    good[feature],
                    good[value_col]
                ).statistic
            )
        else:
            rho = np.nan

        # Descriptive position-level aggregation.
        byp = good.groupby(
            "position", as_index=False
        )[[feature, value_col]].mean()

        if (
            len(byp) >= 5
            and byp[feature].nunique() > 1
            and byp[value_col].nunique() > 1
        ):
            posrho = float(
                spearmanr(
                    byp[feature],
                    byp[value_col]
                ).statistic
            )
        else:
            posrho = np.nan

        summary.append({
            "target": label,
            "feature": feature,
            "description": description,
            "matched_variants": n,
            "matched_positions": len(byp),
            "spearman_variant": rho,
            "spearman_position": posrho,
        })

    return pd.DataFrame(summary)


def main():
    self_test()

    from phase26k_trafficking_preflight import (
        load_allowed,
        mutation_code,
        position_from_code
    )

    from phase26k2_numeric_audit import parse_value

    if not BOOK.is_file():
        raise FileNotFoundError(BOOK)

    # TRAIN outcomes only, with protected VTI positions.
    v, e, protected = load_allowed()

    d = pd.read_excel(
        BOOK, sheet_name="Sheet4"
    )

    absent = [
        name for name in NAMES.values()
        if name not in d.columns
    ]

    if absent:
        raise ValueError(
            f"Missing surface-expression columns: {absent}"
        )

    d["mutation"] = d["Variant"].map(
        mutation_code
    )

    d["position"] = d.mutation.map(
        position_from_code
    )

    # Exclude protected positions before comparisons.
    ext = d.loc[
        d.mutation.notna()
        & ~d.position.isin(protected),
        ["mutation", "position", *NAMES.values()]
    ].copy()

    if ext.mutation.duplicated().any():
        raise ValueError(
            "Duplicate eligible substitutions"
        )

    if len(ext) != 180:
        raise ValueError(
            f"Expected 180 eligible substitutions; "
            f"found {len(ext)}"
        )

    for short, original in NAMES.items():
        parsed = ext[original].map(
            parse_value
        )

        formats = pd.Series(
            [x[1] for x in parsed],
            index=ext.index
        )

        if formats.eq("UNRECOGNISED").any():
            raise ValueError(
                f"Unrecognised format: {original}"
            )

        ext[short] = [
            x[0] for x in parsed
        ]

    ext = ext[
        ["mutation", "position", *NAMES]
    ].copy()

    ext = make_derived(ext)

    if ext[[*NAMES]].isna().any().any():
        raise ValueError(
            "Missing raw assay means"
        )

    assert set(
        ext.position.astype(int)
    ).isdisjoint(protected)

    print("=" * 105)
    print(
        "PHASE 26K-3 | CORRECTOR-INDUCED "
        "SURFACE EXPRESSION | TRAIN ONLY"
    )
    print("=" * 105)

    print(
        f"Tedman eligible mutations: {len(ext)} | "
        f"positions: {ext.position.nunique()}"
    )

    print(
        "Surface immunostaining is not channel "
        "function or %WT functional rescue."
    )

    print(
        "Treatment: VX-445/VX-661, "
        "not VNZ/TEZ-specific measurements."
    )

    print(
        "Absolute differences retain fluorescence "
        "units; fold responses are dimensionless."
    )

    print(
        "Reported ± uncertainties are not used "
        "as weights."
    )

    print(
        "Protected positions excluded; "
        "protected outcomes unopened."
    )

    print("\nA. PHENOTYPE QC AND DISTRIBUTIONS")

    summaries = []

    for feature, description in FEATURES:
        x = pd.to_numeric(
            ext[feature],
            errors="coerce"
        ).to_numpy(float)

        good = x[np.isfinite(x)]

        summaries.append({
            "feature": feature,
            "n": len(good),
            "missing": len(x) - len(good),
            "median": (
                float(np.median(good))
                if len(good) else np.nan
            ),
            "p05": (
                float(np.quantile(good, .05))
                if len(good) else np.nan
            ),
            "p95": (
                float(np.quantile(good, .95))
                if len(good) else np.nan
            ),
        })

    distribution = pd.DataFrame(summaries)

    print(
        distribution.round(3).to_string(index=False)
    )

    print("\nB. EXACT-MUTATION OVERLAPS")

    print(
        "VTI TRAIN matches:",
        len(set(ext.mutation) & set(v.mutation))
    )

    print(
        "ETI TRAIN matches:",
        len(set(ext.mutation) & set(e.mutation))
    )

    results = pd.concat([
        association(
            ext, v,
            "VTI corrector",
            "vti_corrector"
        ),
        association(
            ext, e,
            "ETI corrector",
            "eti_corrector"
        )
    ], ignore_index=True)

    print("\nC. DERIVED PHENOTYPE ASSOCIATIONS")
    print(
        "Correlations are exploratory. "
        "VTI has only 14 exact matches."
    )

    for target in (
        "ETI corrector",
        "VTI corrector"
    ):
        q = results.loc[
            results.target.eq(target)
        ]

        print(f"\n{target}")

        print(
            q[
                [
                    "feature",
                    "matched_variants",
                    "matched_positions",
                    "spearman_variant",
                    "spearman_position"
                ]
            ].round(3).to_string(index=False)
        )

    print("\nD. PRIMARY CORRECTOR-CHANGE PHENOTYPES")

    primary = results.loc[
        results.feature.isin([
            "p_gain_dual",
            "p_log2fc_dual",
            "k_gain_dual",
            "k_log2fc_dual"
        ])
    ]

    print(
        primary[
            [
                "target",
                "feature",
                "matched_variants",
                "spearman_variant",
                "spearman_position"
            ]
        ].round(3).to_string(index=False)
    )

    print(
        "\nDo not select an auxiliary phenotype "
        "solely by its observed correlation."
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    ext.to_csv(
        OUT / "phase26k3_derived_surface_train.csv",
        index=False
    )

    results.to_csv(
        OUT / "phase26k3_all_associations.csv",
        index=False
    )

    distribution.to_csv(
        OUT / "phase26k3_distribution_qc.csv",
        index=False
    )

    print("\nPASS: Phase26K-3 completed.")
    print(
        "No model fitting or protected outcomes."
    )
    print("Saved:", OUT)


if __name__ == "__main__":
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
