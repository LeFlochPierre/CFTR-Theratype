#!/usr/bin/env python3
"""Phase27A-3: VX121 differential-signal robustness, TRAIN-only."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import HuberRegressor

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/biology/phase27a"
DMS = OUT / "phase27a2_selectivity_train.csv"
TED = ROOT / "results/biology/phase26k/phase26k3_derived_surface_train.csv"
SRC = ROOT / "data/external/jhangiani2026_vx121"


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)

    if (
        ok.sum() < 5
        or len(np.unique(a[ok])) < 2
        or len(np.unique(b[ok])) < 2
    ):
        return np.nan

    return float(spearmanr(a[ok], b[ok]).statistic)


def partial_rho(data, x, y, z):
    q = data[[x, y, z]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(q) < 10:
        return np.nan

    xyz = np.column_stack([
        rankdata(q[c]) for c in [x, y, z]
    ])

    control = np.column_stack([
        np.ones(len(q)), xyz[:, 2]
    ])

    u = xyz[:, 0] - control @ np.linalg.lstsq(
        control, xyz[:, 0], rcond=None
    )[0]

    v = xyz[:, 1] - control @ np.linalg.lstsq(
        control, xyz[:, 1], rcond=None
    )[0]

    if np.std(u) < 1e-12 or np.std(v) < 1e-12:
        return np.nan

    return float(np.corrcoef(u, v)[0, 1])


def position_bootstrap(d, x, y, n=2000, seed=27):
    q = d[["position", x, y]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    positions = np.sort(q.position.unique())

    chunks = {
        p: q.loc[
            q.position == p, [x, y]
        ].to_numpy(float)
        for p in positions
    }

    rng = np.random.default_rng(seed)
    values = []

    for _ in range(n):
        sample = rng.choice(
            positions,
            size=len(positions),
            replace=True
        )

        rows = np.vstack([
            chunks[p] for p in sample
        ])

        values.append(
            rho(rows[:, 0], rows[:, 1])
        )

    return (
        float(np.nanquantile(values, .025)),
        float(np.nanquantile(values, .975))
    )


def self_test():
    a = np.arange(10., dtype=float)

    assert np.isclose(rho(a, a + 1), 1)
    assert np.isnan(rho(np.ones(10), a))

    d = pd.DataFrame({
        "position": np.repeat(np.arange(10), 2),
        "x": np.tile(a, 2),
        "y": np.tile(a, 2)
    })

    low, high = position_bootstrap(
        d, "x", "y", 100
    )

    assert np.isclose(low, 1)
    assert np.isclose(high, 1)

    print(
        "PASS synthetic: correlations and "
        "clustered bootstrap"
    )


def main(nboot):
    self_test()

    from phase26k_trafficking_preflight import (
        load_allowed,
        mutation_code
    )

    v, e, protected = load_allowed()

    if not DMS.exists() or not TED.exists():
        raise FileNotFoundError(
            "Run Phase27A-2 and Phase26K-3 first"
        )

    d = pd.read_csv(
        DMS, dtype={"mutation": str}
    )

    assert len(d) == 175
    assert d.mutation.is_unique

    assert set(
        d.position.astype(int)
    ).isdisjoint(protected)

    required = [
        "vx121_log2fc", "vx445_log2fc",
        "selectivity_log2",
        "vehicle121", "treated121",
        "vehicle445", "treated445"
    ]

    if not np.isfinite(
        d[required].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Nonfinite DMS-derived phenotypes"
        )

    assert np.allclose(
        d.selectivity_log2,
        d.vx121_log2fc - d.vx445_log2fc,
        atol=1e-9
    )

    print("=" * 104)
    print(
        "PHASE 27A-3 | VX-121 DIFFERENTIAL SIGNAL "
        "AND SOURCE QC | TRAIN ONLY"
    )
    print("=" * 104)

    print(
        "175 mutations; training outcomes only; "
        "protected positions excluded"
    )

    print(
        "\nA. DRUG-RESPONSE ASSOCIATIONS "
        "AND MEASUREMENT CHECKS"
    )

    comparisons = [
        ("treated121", "treated445"),
        ("vx121_log2fc", "vx445_log2fc"),
        ("es121", "vx121_log2fc"),
        ("es445", "vx445_log2fc")
    ]

    for left, right in comparisons:
        if left not in d or right not in d:
            continue

        print(
            f"{left} vs {right}: "
            f"rho={rho(d[left], d[right]):.4f}"
        )

    print(
        "Median VX121-VX445 log2FC:",
        round(float(d.selectivity_log2.median()), 4)
    )

    print(
        "Positive differences:",
        int((d.selectivity_log2 > 0).sum()),
        "/", len(d)
    )

    print(
        "NOTE: historical assays have different scales; "
        "this does not determine which drug is superior."
    )

    print(
        "\nB. VX-121 RESPONSE RESIDUAL "
        "AFTER ACCOUNTING FOR VX-445"
    )

    x = d[["vx445_log2fc"]].to_numpy(float)
    y = d.vx121_log2fc.to_numpy(float)

    model = HuberRegressor(
        epsilon=1.35,
        alpha=.01
    ).fit(x, y)

    d["vx121_residual_445"] = (
        y - model.predict(x)
    )

    median = float(
        d.vx121_residual_445.median()
    )

    d["vx121_residual_445_centered"] = (
        d.vx121_residual_445 - median
    )

    print(
        "Robust intercept:",
        round(float(model.intercept_), 4)
    )

    print(
        "Robust slope:",
        round(float(model.coef_[0]), 4)
    )

    print(
        "Residual standard deviation:",
        round(float(d.vx121_residual_445.std()), 4)
    )

    for title, ascending in [
        ("Most negative residual", True),
        ("Most positive residual", False)
    ]:
        print("\n" + title)

        columns = [
            "mutation", "position",
            "vx121_log2fc",
            "vx445_log2fc",
            "selectivity_log2",
            "vx121_residual_445_centered"
        ]

        print(
            d.sort_values(
                "vx121_residual_445",
                ascending=ascending
            )[columns]
            .head(10)
            .round(3)
            .to_string(index=False)
        )

    print(
        "\nC. TRAIN-ONLY FUNCTIONAL ASSOCIATIONS"
    )

    results = []

    for label, frame, target in [
        ("ETI corrector", e, "eti_corrector"),
        ("VTI corrector", v, "vti_corrector")
    ]:
        q = d.merge(
            frame[["mutation", target]],
            on="mutation",
            validate="one_to_one"
        )

        print(
            f"\n{label}: {len(q)} mutations / "
            f"{q.position.nunique()} positions"
        )

        features = [
            "vx121_log2fc",
            "vx445_log2fc",
            "selectivity_log2",
            "vx121_residual_445_centered"
        ]

        for feature in features:
            r = rho(q[feature], q[target])

            low, high = position_bootstrap(
                q, feature, target,
                n=nboot
            )

            print(
                f"{feature:32s} "
                f"rho={r: .4f} "
                f"CI95=[{low: .4f}, {high: .4f}]"
            )

            results.append({
                "target": label,
                "feature": feature,
                "n": len(q),
                "positions": q.position.nunique(),
                "rho": r,
                "ci95_low": low,
                "ci95_high": high
            })

        print(
            "Partial Spearman(VX121, corrector | VX445):",
            round(
                partial_rho(
                    q, "vx121_log2fc",
                    target, "vx445_log2fc"
                ), 4
            )
        )

    print(
        "\nD. OVERLAP WITH TEDMAN TRAFFICKING PHENOTYPE"
    )

    ted = pd.read_csv(
        TED,
        usecols=[
            "mutation",
            "p_log2fc_dual"
        ],
        dtype={"mutation": str}
    )

    joined = d.merge(
        ted, on="mutation",
        validate="one_to_one"
    )

    print("Tedman matched mutations:", len(joined))

    for feature in [
        "vx121_log2fc",
        "vx445_log2fc",
        "selectivity_log2",
        "vx121_residual_445_centered"
    ]:
        print(
            f"{feature:32s} "
            f"rho={rho(joined[feature], joined.p_log2fc_dual):.4f}"
        )

    print(
        "\nE. DUPLICATE DNA IDENTITIES "
        "AND CROSS-SHEET DISCREPANCY"
    )

    books = list(
        SRC.glob("*StatisticalModels*.xlsx")
    )

    if len(books) != 1:
        raise ValueError(
            f"Expected one workbook; found {books}"
        )

    for sheet in [
        "Effect Size Comparison",
        "Condensed Comparison Sheet"
    ]:
        raw = pd.read_excel(
            books[0],
            sheet_name=sheet
        )

        raw["mutation"] = raw.VARIANT.map(
            mutation_code
        )

        raw["position"] = (
            raw.mutation.str.extract(
                r"[A-Z]([0-9]+)[A-Z]",
                expand=False
            ).astype(float)
        )

        raw = raw.loc[
            ~raw.position.isin(protected)
        ]

        dup = raw.loc[
            raw.mutation.notna()
            & raw.mutation.duplicated(keep=False)
        ]

        columns = [
            c for c in [
                "mutation",
                "VARIANT",
                "Nucleotide Seq",
                "VX-121 ES",
                "VX445 EFFECT SIZE",
                "VX-121 EFFECT SIZE"
            ]
            if c in dup.columns
        ]

        print(
            f"\n{sheet}: "
            f"{len(dup)} repeated-substitution rows"
        )

        if len(dup):
            print(
                dup[columns]
                .head(20)
                .to_string(index=False)
            )

        discordant = raw.loc[
            raw.mutation.isin([
                "R352W", "R342W"
            ])
        ]

        if len(discordant):
            print(
                "R352W/R342W records:"
            )

            show = [
                c for c in [
                    "VARIANT",
                    "Nucleotide Seq",
                    "mutation"
                ]
                if c in discordant.columns
            ]

            print(
                discordant[show].to_string(
                    index=False
                )
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    d.to_csv(
        OUT / "phase27a3_residual_selectivity_train.csv",
        index=False
    )

    pd.DataFrame(results).to_csv(
        OUT / "phase27a3_cluster_associations.csv",
        index=False
    )

    print(
        "\nPASS: Phase27A-3 completed."
    )

    print(
        "No response-model fitting "
        "or protected outcome access."
    )

    print("Saved:", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    parser.add_argument(
        "--bootstraps",
        type=int,
        default=2000
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        main(args.bootstraps)
