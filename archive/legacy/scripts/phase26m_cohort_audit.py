#!/usr/bin/env python3
"""Phase 26M: TRAIN-only cohort and protein-substitution consistency audit.

Uses Phase26I saved OOF summaries.
No model fitting or protected outcome access.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path.cwd()
SOURCE = ROOT / "results/biology/phase26i/phase26i_variant_errors.csv"
OUT = ROOT / "results/biology/phase26m"

REQUIRED = {
    "variant_key", "position", "hgvs", "cohort",
    "true_corrector", "pred_corrector",
    "true_total", "pred_total",
    "true_acute", "pred_acute",
    "ae_corrector", "ae_total",
    "gain_corrector", "gain_total",
    "baseline_fsk", "corrected_sem", "treated_sem"
}


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Run Phase26I first: {SOURCE}"
        )

    d = pd.read_csv(
        SOURCE, dtype={"variant_key": str}
    )

    missing = REQUIRED - set(d.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if d.variant_key.duplicated().any():
        raise ValueError("Duplicate variant_key")

    if len(d) != 210 or d.position.nunique() != 168:
        raise ValueError(
            "Expected 210 VTI variants / 168 positions"
        )

    for name in ("corrector", "acute", "total"):
        d["residual_" + name] = (
            d["pred_" + name]
            - d["true_" + name]
        )

    if not np.allclose(
        d.true_corrector + d.true_acute,
        d.true_total,
        atol=1e-5
    ):
        raise ValueError(
            "Measured components do not reconstruct total"
        )

    if not np.allclose(
        d.pred_corrector + d.pred_acute,
        d.pred_total,
        atol=1e-4
    ):
        raise ValueError(
            "Predicted components do not reconstruct total"
        )

    print("=" * 100)
    print(
        "PHASE 26M | VTI COHORT SHIFT AND "
        "PROTEIN-SUBSTITUTION CONSISTENCY"
    )
    print("=" * 100)

    print(
        "210 TRAIN variants / 168 protein positions"
    )
    print(
        "Predictions averaged over two frozen OOF repeats"
    )
    print(
        "No model fitting; protected validation/test "
        "outcomes unopened"
    )

    print("\nA. COHORT RESPONSE DISTRIBUTIONS")

    rows = []

    groups = [
        ("ALL", d)
    ] + [
        (str(name), group)
        for name, group in d.groupby("cohort")
    ]

    for name, group in groups:
        row = {
            "cohort": name,
            "variants": len(group),
            "positions": group.position.nunique()
        }

        for target in ("corrector", "total"):
            y = group[
                "true_" + target
            ].to_numpy(float)

            row.update({
                target + "_mean": np.mean(y),
                target + "_median": np.median(y),
                target + "_p10": np.quantile(y, .1),
                target + "_p90": np.quantile(y, .9),
                target + "_low_lt10": int((y < 10).sum()),
                target + "_high_gt40": int((y > 40).sum())
            })

        row["baseline_median"] = (
            group.baseline_fsk.median()
        )

        row["treated_sem_median"] = (
            group.treated_sem.median()
        )

        rows.append(row)

    cohorts = pd.DataFrame(rows)

    print(
        cohorts.round(3).to_string(index=False)
    )

    print("\nB. MODEL ERRORS BY COHORT")

    errors = []

    for name, group in groups:
        for target in (
            "corrector", "acute", "total"
        ):
            residual = group[
                "residual_" + target
            ].to_numpy(float)

            errors.append({
                "cohort": name,
                "target": target,
                "variants": len(group),
                "positions": group.position.nunique(),
                "mean_signed_error": np.mean(residual),
                "mae_of_mean_predictions": np.mean(
                    np.abs(residual)
                ),
                "underpredicted_fraction": np.mean(
                    residual < 0
                )
            })

    errors = pd.DataFrame(errors)

    print(
        errors.round(3).to_string(index=False)
    )

    print(
        "NOTE: Errors of predictions averaged over repeats "
        "differ from the original repeated-OOF MAEs."
    )

    print("\nC. LOW AND HIGH RESPONDERS WITHIN EACH COHORT")

    strata = []

    for cohort, group in d.groupby("cohort"):
        for target in ("corrector", "total"):
            y = group["true_" + target]

            subsets = [
                ("low_lt10", y < 10),
                (
                    "intermediate_10_40",
                    (y >= 10) & (y <= 40)
                ),
                ("high_gt40", y > 40)
            ]

            for label, mask in subsets:
                part = group.loc[mask]

                if part.empty:
                    continue

                residual = part[
                    "residual_" + target
                ].to_numpy(float)

                strata.append({
                    "cohort": cohort,
                    "target": target,
                    "subset": label,
                    "variants": len(part),
                    "positions": part.position.nunique(),
                    "measured_mean": part[
                        "true_" + target
                    ].mean(),
                    "predicted_mean": part[
                        "pred_" + target
                    ].mean(),
                    "signed_error": residual.mean(),
                    "mae": np.abs(residual).mean()
                })

    strata = pd.DataFrame(strata)

    print(
        strata.round(3).to_string(index=False)
    )

    print(
        "\nD. DIFFERENT DNA VARIANTS WITH "
        "THE SAME PROTEIN SUBSTITUTION"
    )

    collisions = []

    for hgvs, group in d.groupby(
        "hgvs", dropna=False
    ):
        if len(group) <= 1:
            continue

        collisions.append({
            "hgvs": hgvs,
            "position": int(group.position.iloc[0]),
            "dna_variants": len(group),
            "variant_keys": "; ".join(
                group.variant_key.astype(str)
            ),
            "corrector_min": group.true_corrector.min(),
            "corrector_max": group.true_corrector.max(),
            "corrector_range": (
                group.true_corrector.max()
                - group.true_corrector.min()
            ),
            "total_min": group.true_total.min(),
            "total_max": group.true_total.max(),
            "total_range": (
                group.true_total.max()
                - group.true_total.min()
            )
        })

    collisions = pd.DataFrame(collisions)

    if len(collisions):
        print(
            collisions.sort_values(
                "total_range", ascending=False
            ).round(3).to_string(index=False)
        )
    else:
        print("No repeated protein substitutions")

    print(
        "\nE. PROTEIN-ONLY LABEL-COLLISION LOWER BOUND"
    )

    for target in (
        "true_corrector", "true_total"
    ):
        total_min_error = 0.0

        for _, group in d.groupby(
            "hgvs", dropna=False
        ):
            if len(group) <= 1:
                continue

            median = group[target].median()

            total_min_error += (
                group[target] - median
            ).abs().sum()

        print(
            f"{target}: minimum forced MAE from "
            f"exact protein-label collisions = "
            f"{total_min_error / len(d):.4f} pp"
        )

    print(
        "This is a duplicate-only lower bound, not "
        "an estimate of the model's total irreducible error."
    )

    print("\nF. WORST TOTAL-RESPONSE FAILURES")

    show = [
        "variant_key", "hgvs", "cohort",
        "true_corrector", "pred_corrector",
        "true_acute", "pred_acute",
        "true_total", "pred_total",
        "ae_total", "gain_total"
    ]

    print(
        d.nlargest(15, "ae_total")[show]
        .round(3).to_string(index=False)
    )

    OUT.mkdir(parents=True, exist_ok=True)

    outputs = {
        "cohorts": cohorts,
        "errors": errors,
        "response_strata": strata,
        "same_protein": collisions,
        "all_variants": d
    }

    for name, frame in outputs.items():
        frame.to_csv(
            OUT / f"phase26m_{name}.csv",
            index=False
        )

    print("\nPASS: Phase26M completed")
    print("Saved:", OUT)
    print("Protected outcomes unopened")


if __name__ == "__main__":
    main()
