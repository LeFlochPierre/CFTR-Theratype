#!/usr/bin/env python3
"""Phase 30F: Position-aware prioritisation audit.

TRAIN only. Reuses frozen Phase 30E predictions.
No new model fitting or protected outcomes.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path(
    "results/biology/phase30e/"
    "phase30e_variant_predictions.csv"
)
OUT = Path("results/biology/phase30f")

REQUIRED = {
    "target",
    "variant_key",
    "protein_position",
    "eti_total_pp",
    "observed_delta_pp",
    "xgb_delta_pred_pp",
    "ridge_delta_pred_pp",
}

RULES = {
    "xgb_predicted_gain":
        lambda x: x.xgb_delta_pred_pp.to_numpy(float),
    "ridge_predicted_gain":
        lambda x: x.ridge_delta_pred_pp.to_numpy(float),
    "low_eti_total":
        lambda x: -x.eti_total_pp.to_numpy(float),
    "high_eti_total":
        lambda x: x.eti_total_pp.to_numpy(float),
    "xgb_predicted_absolute_difference":
        lambda x: np.abs(
            x.xgb_delta_pred_pp.to_numpy(float)
        ),
}


def evaluate(x, simulations=20000, seed=30):
    missing = REQUIRED - set(x.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    x = x.loc[x.target == "total_rescue"].copy()
    x["variant_key"] = x.variant_key.astype(str)

    if (
        x.empty
        or x.variant_key.duplicated().any()
        or x[["variant_key", "protein_position"]]
            .isna().any().any()
    ):
        raise ValueError("Invalid variant identifiers")

    numeric = [
        "eti_total_pp",
        "observed_delta_pp",
        "xgb_delta_pred_pp",
        "ridge_delta_pred_pp",
    ]

    if not np.isfinite(
        x[numeric].to_numpy(float)
    ).all():
        raise ValueError("Nonfinite measurements")

    if (
        len(x) != 66
        or x.protein_position.nunique() != 56
    ):
        raise ValueError(
            "Expected 66 variants / 56 positions"
        )

    x["event"] = x.observed_delta_pp >= 10

    positions = sorted(
        x.protein_position.unique()
    )

    probabilities = np.array([
        x.loc[
            x.protein_position == p, "event"
        ].mean()
        for p in positions
    ])

    n_positions = len(positions)
    budgets = [7, 14, 20]

    rng = np.random.default_rng(seed)
    random_hits = {}

    for k in budgets:
        sampled = np.argsort(
            rng.random((simulations, n_positions)),
            axis=1,
        )[:, :k]

        probability = probabilities[sampled]

        random_hits[k] = (
            rng.random((simulations, k))
            < probability
        ).sum(axis=1)

    results = []
    selected = []

    for method, scoring in RULES.items():

        ranked = x.assign(
            score=scoring(x)
        ).sort_values(
            ["score", "variant_key"],
            ascending=[False, True],
            kind="stable",
        )

        # Keep highest-ranked variant at each position.
        unique = ranked.drop_duplicates(
            "protein_position",
            keep="first",
        ).reset_index(drop=True)

        assert len(unique) == 56

        for k in budgets:
            top = unique.head(k)
            hits = int(top.event.sum())

            random = random_hits[k]

            results.append({
                "rule": method,
                "tested_positions": k,
                "hits": hits,
                "precision": hits / k,
                "random_expected_hits":
                    float(random.mean()),
                "exploratory_random_p":
                    float(
                        (1 + np.sum(random >= hits))
                        / (simulations + 1)
                    ),
                "total_positive_variants":
                    int(x.event.sum()),
                "positive_positions":
                    int(
                        x.groupby(
                            "protein_position"
                        ).event.any().sum()
                    ),
                "selected_positions_517_to_601":
                    int(
                        top.protein_position
                        .between(517, 601).sum()
                    ),
            })

            if k == 7:
                selected.append(
                    top.assign(rule=method)[[
                        "rule",
                        "variant_key",
                        "protein_position",
                        "eti_total_pp",
                        "observed_delta_pp",
                        "xgb_delta_pred_pp",
                        "ridge_delta_pred_pp",
                        "event",
                    ]]
                )

    return (
        pd.DataFrame(results),
        pd.concat(selected, ignore_index=True),
    )


def self_test():
    rng = np.random.default_rng(1)

    x = pd.DataFrame({
        "target": "total_rescue",
        "variant_key": [
            f"v{i}" for i in range(66)
        ],
        "protein_position":
            list(range(56)) + list(range(10)),
        "eti_total_pp":
            rng.normal(20, 10, 66),
        "observed_delta_pp":
            rng.normal(5, 10, 66),
        "xgb_delta_pred_pp":
            rng.normal(5, 10, 66),
        "ridge_delta_pred_pp":
            rng.normal(5, 10, 66),
    })

    result, top = evaluate(x, 1000)

    assert len(result) == 15
    assert len(top) == 35
    assert (
        top.groupby("rule")
        .protein_position.nunique() == 7
    ).all()

    bad = x.copy()
    bad.loc[0, "protein_position"] = None

    try:
        evaluate(bad, 1000)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Missing position not detected"
        )

    print("PASS Phase30F synthetic QC")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=20000,
    )

    args = parser.parse_args()

    self_test()

    if args.self_test:
        return

    if args.simulations < 1000:
        parser.error(
            "Require >=1000 simulations"
        )

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    data = pd.read_csv(
        INPUT,
        dtype={"variant_key": str},
    )

    results, selected = evaluate(
        data, args.simulations
    )

    print("=" * 100)
    print(
        "PHASE 30F | POSITION-DEDUPLICATED "
        "EXPERIMENT PRIORITISATION"
    )
    print("=" * 100)

    print(
        "TRAIN only. No new model fitting. "
        "One tested variant per protein position."
    )

    print("\nA. RANKING RULE COMPARISONS")
    print(
        results.round(4).to_string(index=False)
    )

    print("\nB. TOP 7 DISTINCT POSITIONS PER RULE")
    print(
        selected.round(3).to_string(index=False)
    )

    OUT.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUT / "phase30f_position_budget.csv",
        index=False,
    )

    selected.to_csv(
        OUT / "phase30f_top7_by_rule.csv",
        index=False,
    )

    print("\nLIMITATIONS")
    print(
        "- Random comparison selects positions "
        "uniformly and one variant per position."
    )
    print(
        "- P-values exploratory, not adjusted "
        "for earlier development choices."
    )
    print(
        "- ETI/VTI cross-study scale "
        "harmonisation remains unresolved."
    )
    print(
        "- No independent validation performed."
    )

    print("\nPASS Phase30F; outputs:", OUT)


if __name__ == "__main__":
    main()
