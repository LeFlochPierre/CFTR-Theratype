#!/usr/bin/env python3
"""Phase 30G: Frozen ETI-to-VTI regional confounding audit."""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

SOURCE = Path(
    "results/biology/phase30e/"
    "phase30e_variant_predictions.csv"
)
OUT = Path("results/biology/phase30g")

BUDGETS = (7, 14, 20)
THRESHOLDS = (5, 10, 15, 20)


def evaluate(df, permutations=20000):
    required = {
        "target", "variant_key", "protein_position",
        "observed_delta_pp", "xgb_delta_pred_pp"
    }
    if not required.issubset(df.columns):
        raise ValueError(
            f"Missing columns: {required - set(df.columns)}"
        )

    x = df[df.target == "total_rescue"].copy()

    if (
        len(x) != 66
        or x.variant_key.nunique() != 66
        or x.protein_position.nunique() != 56
    ):
        raise ValueError("Expected 66 variants / 56 positions")

    x["region"] = np.where(
        x.protein_position.between(500, 650),
        "500-650",
        "outside"
    )

    # One highest-scoring variant per protein position.
    ranked = (
        x.sort_values(
            ["xgb_delta_pred_pp", "variant_key"],
            ascending=[False, True],
            kind="stable"
        )
        .drop_duplicates("protein_position")
        .reset_index(drop=True)
    )

    region = ranked.region.eq("500-650").to_numpy()

    # Randomise response labels WITHIN regions.
    # This preserves each region's event frequency.
    rng = np.random.default_rng(300)
    results = []

    for threshold in THRESHOLDS:
        event = (
            ranked.observed_delta_pp.to_numpy(float)
            >= threshold
        ).astype(np.int8)

        null = np.empty(
            (permutations, len(ranked)),
            dtype=np.int8
        )

        for is_region in (False, True):
            idx = np.flatnonzero(region == is_region)

            block = np.broadcast_to(
                event[idx],
                (permutations, len(idx))
            ).copy()

            null[:, idx] = rng.permuted(
                block, axis=1
            )

        cumulative = null.cumsum(axis=1)

        for k in BUDGETS:
            hits = int(event[:k].sum())
            simulated = cumulative[:, k - 1]

            results.append({
                "threshold_pp": threshold,
                "budget": k,
                "xgb_hits": hits,
                "xgb_precision": hits / k,
                "region_positions_selected":
                    int(region[:k].sum()),
                "region_expected_hits":
                    float(simulated.mean()),
                "conditional_p_exploratory":
                    float(
                        (1 + (simulated >= hits).sum())
                        / (permutations + 1)
                    )
            })

    summary = []

    for name, group in ranked.groupby("region"):
        row = {
            "region": name,
            "positions": len(group),
            "median_vti_minus_eti":
                group.observed_delta_pp.median()
        }

        for threshold in THRESHOLDS:
            row[f"events_ge{threshold}"] = int(
                (group.observed_delta_pp >= threshold).sum()
            )

        summary.append(row)

    return (
        pd.DataFrame(summary),
        pd.DataFrame(results),
        ranked
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--permutations", type=int, default=20000
    )
    args = parser.parse_args()

    if args.permutations < 1000:
        parser.error("Require at least 1000 permutations")

    df = pd.read_csv(
        SOURCE, dtype={"variant_key": str}
    )

    summary, results, ranked = evaluate(
        df, args.permutations
    )

    print("=" * 95)
    print("PHASE 30G | REGIONAL CONFOUNDING AUDIT")
    print("=" * 95)
    print("TRAIN only; no model fitting or protected labels.")
    print("Residues 500-650: exploratory post-hoc region.")

    print("\nA. EVENT PREVALENCE BY REGION")
    print(
        summary.round(3).to_string(index=False)
    )

    print("\nB. XGBOOST VS REGION-CONDITIONED RANDOM")
    print(
        results.round(4).to_string(index=False)
    )

    print("\nC. TOP 14 RANKED POSITIONS")
    print(
        ranked.head(14)[[
            "variant_key",
            "protein_position",
            "region",
            "observed_delta_pp",
            "xgb_delta_pred_pp"
        ]].round(3).to_string(index=False)
    )

    OUT.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUT / "phase30g_region_summary.csv",
        index=False
    )
    results.to_csv(
        OUT / "phase30g_conditional_test.csv",
        index=False
    )
    ranked.to_csv(
        OUT / "phase30g_ranked_positions.csv",
        index=False
    )

    print("\nLIMITATIONS")
    print("- The region was chosen after examining results.")
    print("- Permutation probabilities are exploratory.")
    print("- Position representatives are XGBoost-selected.")
    print("- OOF scores combine different fold models.")
    print("- Cross-study assay comparability is unresolved.")
    print("- No independent validation.")

    print("\nPASS Phase 30G; saved:", OUT)


if __name__ == "__main__":
    main()
