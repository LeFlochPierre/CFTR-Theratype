#!/usr/bin/env python3
"""Phase 30E: TRAIN-only publication feasibility audit.

No new model fits, hyperparameter tuning, or VAL/TEST outcomes.
Uses Phase30C matched TRAIN data and Phase30D frozen OOF.
Thresholds are exploratory, NOT clinical cutoffs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/biology/phase30e"
A = ROOT / "results/biology/phase30a/phase30a_oof.csv"
D = ROOT / "results/biology/phase30d/phase30d_oof.csv"
OLD = ROOT / "data/processed/bihler_2024_missense_splits.parquet"
NEW = ROOT / "data/processed/bihler_2026_transfer_splits.parquet"

ID = [
    "repeat", "fold", "variant_key",
    "protein_position", "target"
]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def bootstrap_gain(data, nboot, seed):
    pos = (
        data.groupby("protein_position")
        .gain.mean()
        .to_numpy(float)
    )

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0, len(pos), size=(nboot, len(pos))
    )
    samples = pos[indices].mean(axis=1)

    return {
        "gain_pp": float(pos.mean()),
        "ci95_low": float(np.quantile(samples, .025)),
        "ci95_high": float(np.quantile(samples, .975)),
        "positions": len(pos),
    }


def audit(full, matched, frozen, boot=2000):

    # A. STRICT COHORT AND OOF QC

    if (
        len(full) != 210
        or full.variant_key.nunique() != 210
        or full.protein_position.nunique() != 168
    ):
        raise ValueError("Full TRAIN cohort changed")

    if (
        len(matched) != 66
        or matched.variant_key.nunique() != 66
        or matched.protein_position.nunique() != 56
    ):
        raise ValueError("Matched TRAIN cohort changed")

    if not set(matched.variant_key).issubset(set(full.variant_key)):
        raise ValueError("Matched variants outside TRAIN")

    source = full.set_index("variant_key")

    for column in [
        "protein_position",
        "observed_baseline",
        "observed_corrector",
        "observed_total_rescue",
    ]:
        a = matched[column].to_numpy(float)
        b = source.loc[
            matched.variant_key, column
        ].to_numpy(float)

        if not np.allclose(a, b, atol=1e-6, rtol=0):
            raise ValueError(f"Matched mismatch: {column}")

    if not np.allclose(
        matched.eti_total_pp,
        matched.eti_corrector_pp + matched.eti_acute_pp,
        atol=.25,
        rtol=0,
    ):
        raise ValueError("ETI decomposition mismatch")

    if not np.allclose(
        matched.observed_treated_absolute,
        matched.observed_baseline
        + matched.observed_total_rescue,
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("VTI target arithmetic mismatch")

    if (
        len(frozen) != 264
        or frozen.duplicated(ID).any()
        or set(frozen.repeat) != {1, 2}
        or set(frozen.target) != {"corrector", "total_rescue"}
    ):
        raise ValueError("Invalid Phase30D OOF")

    for target, truth in [
        ("corrector", "observed_corrector"),
        ("total_rescue", "observed_total_rescue"),
    ]:
        z = frozen[frozen.target == target]

        for repeat in [1, 2]:
            r = z[z.repeat == repeat].set_index("variant_key")

            if set(r.index) != set(matched.variant_key):
                raise ValueError("Missing OOF variants")

            if not np.allclose(
                r.loc[matched.variant_key, "observed"],
                matched[truth],
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError("Frozen OOF label mismatch")

            if not np.array_equal(
                r.loc[
                    matched.variant_key, "protein_position"
                ].to_numpy(int),
                matched.protein_position.to_numpy(int),
            ):
                raise ValueError("OOF position mismatch")

            if (
                r.groupby("protein_position")
                .fold.nunique().max() > 1
            ):
                raise ValueError("Position leakage")

    # B. MATCHED VS UNMATCHED REPRESENTATIVENESS

    full = full.copy()

    full["has_eti"] = full.variant_key.isin(
        matched.variant_key
    )

    full["corrector_regime"] = np.select(
        [
            full.observed_corrector < 10,
            full.observed_corrector > 40,
        ],
        ["low", "high"],
        default="middle",
    )

    full["baseline_quartile"] = pd.qcut(
        full.observed_baseline,
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    )

    coverage = []

    for dimension in [
        "ALL",
        "corrector_regime",
        "baseline_quartile",
    ]:
        groups = (
            [("ALL", full)]
            if dimension == "ALL"
            else list(full.groupby(dimension, observed=True))
        )

        for group, part in groups:
            selected = part[part.has_eti]

            coverage.append({
                "dimension": dimension,
                "group": str(group),
                "vti_train_variants": len(part),
                "matched_variants": len(selected),
                "match_fraction": len(selected) / len(part),
                "vti_train_positions":
                    part.protein_position.nunique(),
                "matched_positions":
                    selected.protein_position.nunique(),
                "full_corrector_median":
                    float(part.observed_corrector.median()),
                "matched_corrector_median":
                    float(selected.observed_corrector.median())
                    if len(selected) else np.nan,
            })

    coverage = pd.DataFrame(coverage)

    # C. ETI/VTI DIFFERENCES

    x = matched[[
        "variant_key",
        "protein_position",
        "observed_baseline",
        "observed_corrector",
        "observed_total_rescue",
        "eti_corrector_pp",
        "eti_total_pp",
        "eti_acute_pp",
        "eti_baseline_fsk",
    ]].copy()

    x["delta_corrector_pp"] = (
        x.observed_corrector - x.eti_corrector_pp
    )

    x["delta_total_pp"] = (
        x.observed_total_rescue - x.eti_total_pp
    )

    differences = []
    crossings = []

    targets = [
        (
            "corrector",
            "observed_corrector",
            "eti_corrector_pp",
            "delta_corrector_pp",
        ),
        (
            "total_rescue",
            "observed_total_rescue",
            "eti_total_pp",
            "delta_total_pp",
        ),
    ]

    for target, vti, eti, delta in targets:
        values = x[delta].to_numpy(float)

        differences.append({
            "target": target,
            "variants": len(x),
            "positions": x.protein_position.nunique(),
            "eti_median": float(x[eti].median()),
            "vti_median": float(x[vti].median()),
            "delta_mean": float(values.mean()),
            "delta_median": float(np.median(values)),
            "delta_q10": float(np.quantile(values, .10)),
            "delta_q90": float(np.quantile(values, .90)),
            "abs_delta_median": float(np.median(abs(values))),
            "abs_delta_ge10": int((abs(values) >= 10).sum()),
            "abs_delta_ge20": int((abs(values) >= 20).sum()),
            "vti_advantage_ge10": int((values >= 10).sum()),
            "vti_advantage_ge20": int((values >= 20).sum()),
            "eti_advantage_ge10": int((values <= -10).sum()),
            "eti_advantage_ge20": int((values <= -20).sum()),
            "spearman": float(
                x[vti].corr(x[eti], method="spearman")
            ),
        })

        for threshold in [10., 20.]:
            crossings.append({
                "target": target,
                "threshold_pp": threshold,
                "eti_below_vti_above": int((
                    (x[eti] < threshold)
                    & (x[vti] >= threshold)
                ).sum()),
                "eti_above_vti_below": int((
                    (x[eti] >= threshold)
                    & (x[vti] < threshold)
                ).sum()),
                "eti_below": int(
                    (x[eti] < threshold).sum()
                ),
                "vti_above": int(
                    (x[vti] >= threshold).sum()
                ),
            })

    # D. FROZEN MODEL PERFORMANCE ON ETI/VTI DELTA

    predictions = []
    metrics = []
    gains = []
    budgets = []

    for target, vti, eti, delta in targets:

        f = frozen[frozen.target == target]

        ridge_col = (
            "pred_eti_corrector_ridge"
            if target == "corrector"
            else "pred_eti_total_ridge"
        )

        cols = [
            "pred_direct_identity",
            "pred_eti_3_xgb_prior",
            ridge_col,
        ]

        if not set(cols).issubset(f.columns):
            raise ValueError("Missing Phase30D predictions")

        p = (
            f.groupby("variant_key")[cols]
            .mean()
            .reset_index()
            .merge(
                x[[
                    "variant_key",
                    "protein_position",
                    vti, eti, delta
                ]],
                on="variant_key",
                validate="one_to_one",
            )
        )

        if not np.allclose(
            p.pred_direct_identity,
            p[eti],
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError("Direct ETI identity mismatch")

        p["target"] = target
        p["observed_delta_pp"] = p[delta]

        p["xgb_delta_pred_pp"] = (
            p.pred_eti_3_xgb_prior - p[eti]
        )

        p["ridge_delta_pred_pp"] = (
            p[ridge_col] - p[eti]
        )

        predictions.append(p)

        for name, pred in [
            ("direct_eti", np.zeros(len(p))),
            ("eti_3_xgb", p.xgb_delta_pred_pp.to_numpy(float)),
            ("eti_1_ridge", p.ridge_delta_pred_pp.to_numpy(float)),
        ]:
            z = p.copy()

            z["error"] = abs(
                z.observed_delta_pp.to_numpy(float) - pred
            )

            metrics.append({
                "target": target,
                "model": name,
                "position_mae": float(
                    z.groupby("protein_position")
                    .error.mean().mean()
                ),
                "variant_mae": float(z.error.mean()),
                "pred_delta_sd": float(np.std(pred)),
            })

        z = p.copy()

        z["gain"] = (
            abs(z.observed_delta_pp)
            - abs(
                z.observed_delta_pp
                - z.xgb_delta_pred_pp
            )
        )

        gains.append({
            "target": target,
            "candidate": "eti_3_xgb",
            "reference": "direct_eti",
            **bootstrap_gain(
                z, boot,
                123 if target == "corrector" else 124
            ),
        })

        # E. FIXED-BUDGET TEST PRIORITISATION
        # Rankings use only previous ETI and frozen OOF
        # predictions, never the actual VTI response.

        ks = [
            int(np.ceil(len(p) * fraction))
            for fraction in [.10, .20, .30]
        ]

        labels = {
            "vti_gain_ge10": (
                p.observed_delta_pp.to_numpy(float) >= 10
            ),
            "vti_gain_ge20": (
                p.observed_delta_pp.to_numpy(float) >= 20
            ),
            "abs_difference_ge10": (
                abs(p.observed_delta_pp.to_numpy(float)) >= 10
            ),
            "abs_difference_ge20": (
                abs(p.observed_delta_pp.to_numpy(float)) >= 20
            ),
        }

        priorities = {
            "xgb_predicted_delta":
                p.xgb_delta_pred_pp.to_numpy(float),
            "xgb_predicted_abs_delta":
                abs(p.xgb_delta_pred_pp.to_numpy(float)),
            "low_eti_response":
                -p[eti].to_numpy(float),
        }

        for event, truth in labels.items():
            positives = int(truth.sum())

            for method, score in priorities.items():
                order = np.argsort(
                    -score, kind="stable"
                )

                for k in ks:
                    found = int(
                        truth[order[:k]].sum()
                    )

                    budgets.append({
                        "target": target,
                        "event": event,
                        "priority_rule": method,
                        "tested_variants": k,
                        "total_events": positives,
                        "found_events": found,
                        "precision": found / k,
                        "recall":
                            found / positives
                            if positives else np.nan,
                        "random_expected_hits":
                            positives * k / len(p),
                    })

    return {
        "coverage": coverage,
        "differences": pd.DataFrame(differences),
        "threshold_crossing": pd.DataFrame(crossings),
        "variant_predictions": pd.concat(
            predictions, ignore_index=True
        ),
        "delta_metrics": pd.DataFrame(metrics),
        "paired_gains": pd.DataFrame(gains),
        "testing_budget": pd.DataFrame(budgets),
    }


def self_test():
    rng = np.random.default_rng(57)

    positions = np.r_[
        np.arange(168),
        rng.choice(np.arange(168), 42),
    ]

    positions[56:66] = np.arange(10)
    positions[168:178] = np.arange(56, 66)

    keys = [f"SYN{i:03d}" for i in range(210)]

    baseline = rng.uniform(0, 45, 210)
    corrector = rng.uniform(1, 55, 210)
    total = corrector + rng.uniform(0, 14, 210)

    full = pd.DataFrame({
        "variant_key": keys,
        "protein_position": positions,
        "observed_baseline": baseline,
        "observed_corrector": corrector,
        "observed_total_rescue": total,
    })

    m = full.iloc[:66].copy()

    m["observed_treated_absolute"] = (
        m.observed_baseline + m.observed_total_rescue
    )

    m["eti_corrector_pp"] = (
        m.observed_corrector
        + rng.normal(0, 8, 66)
    )

    m["eti_acute_pp"] = rng.uniform(0, 10, 66)
    m["eti_total_pp"] = (
        m.eti_corrector_pp + m.eti_acute_pp
    )

    m["eti_baseline_fsk"] = rng.uniform(0, 25, 66)

    rows = []

    for repeat in [1, 2]:
        for target, col in [
            ("corrector", "observed_corrector"),
            ("total_rescue", "observed_total_rescue"),
        ]:
            for _, r in m.iterrows():
                direct = (
                    r.eti_corrector_pp
                    if target == "corrector"
                    else r.eti_total_pp
                )

                rows.append({
                    "repeat": repeat,
                    "fold": int(r.protein_position % 5 + 1),
                    "variant_key": r.variant_key,
                    "protein_position":
                        int(r.protein_position),
                    "target": target,
                    "observed": r[col],
                    "pred_direct_identity": direct,
                    "pred_eti_3_xgb_prior": direct + 1,
                    "pred_eti_corrector_ridge": direct + 2,
                    "pred_eti_total_ridge": direct + 2,
                })

    frozen = pd.DataFrame(rows)

    output = audit(full, m, frozen, 200)

    assert output["coverage"].iloc[0].matched_variants == 66
    assert len(output["variant_predictions"]) == 132
    assert len(output["testing_budget"]) == 72

    corrupted = frozen.copy()
    corrupted.loc[0, "observed"] += 3

    try:
        audit(full, m, corrupted, 100)
    except ValueError:
        pass
    else:
        raise AssertionError("OOF parity error not caught")

    print("PASS: Phase30E synthetic tests")


def split_metadata():
    # Membership only: no protected assay outcomes.
    old = pd.read_parquet(
        OLD, columns=["variant_key", "split"]
    )

    new = pd.read_parquet(
        NEW, columns=["variant_key", "transfer_split"]
    )

    for table, label in [
        (old, "split"),
        (new, "transfer_split"),
    ]:
        if (
            table.groupby("variant_key")
            [label].nunique().max() > 1
        ):
            raise ValueError("Contradictory split assignments")

    old = old.drop_duplicates("variant_key")
    new = new.drop_duplicates("variant_key")

    merged = old.merge(
        new, on="variant_key", validate="one_to_one"
    )

    return (
        merged.groupby(["split", "transfer_split"])
        .size()
        .reset_index(name="overlapping_variant_keys")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test", action="store_true"
    )
    parser.add_argument(
        "--bootstraps", type=int, default=2000
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.bootstraps < 100:
        parser.error("Require at least 100 bootstraps")

    self_test()

    import phase30c_eti_incremental_ladder as c

    matched, _, _, _ = c.load_matched_data()

    columns = [
        "repeat",
        "variant_key",
        "protein_position",
        "observed_baseline",
        "observed_corrector",
        "observed_total_rescue",
    ]

    all_oof = pd.read_csv(
        A,
        usecols=columns,
        dtype={"variant_key": str},
    )

    if (
        len(all_oof) != 420
        or all_oof.duplicated(
            ["repeat", "variant_key"]
        ).any()
        or set(all_oof.repeat) != {1, 2}
    ):
        raise ValueError("Phase30A OOF invalid")

    for col in [
        "observed_baseline",
        "observed_corrector",
        "observed_total_rescue",
    ]:
        spread = (
            all_oof.groupby("variant_key")[col].max()
            - all_oof.groupby("variant_key")[col].min()
        )

        if spread.abs().max() > 1e-7:
            raise ValueError("Target changed across repeats")

    full = all_oof[
        all_oof.repeat == 1
    ].drop(columns="repeat").copy()

    frozen = pd.read_csv(
        D, dtype={"variant_key": str}
    )

    print("=" * 105)
    print("PHASE 30E | ETI → VTI PUBLICATION FEASIBILITY")
    print("=" * 105)
    print("TRAIN only. No new model fitting.")
    print("210 VTI TRAIN records; 66 matched ETI/VTI records.")
    print("ETI/VTI assay scale harmonisation not verified.")

    results = audit(
        full, matched, frozen, args.bootstraps
    )

    print("\nA. MATCHED COHORT REPRESENTATIVENESS")
    print(
        results["coverage"].round(3).to_string(index=False)
    )

    print("\nB. ETI-VTI RESPONSE DIFFERENCES")
    print("Delta = VTI - ETI (nominal percentage points)")
    print(
        results["differences"].round(3).to_string(index=False)
    )

    print("\nC. EXPLORATORY THRESHOLD CROSSINGS")
    print(
        results["threshold_crossing"].to_string(index=False)
    )

    print("\nD. DIRECT ETI VS FROZEN XGBOOST")
    print(
        results["delta_metrics"].round(3).to_string(index=False)
    )
    print(
        results["paired_gains"].round(3).to_string(index=False)
    )

    print("\nE. EXPERIMENT PRIORITISATION")
    budget = results["testing_budget"]

    selected = budget[
        (budget.target == "total_rescue")
        & budget.event.isin([
            "vti_gain_ge10",
            "abs_difference_ge10",
        ])
    ]

    print(selected.round(3).to_string(index=False))

    print("\nF. LARGEST ETI/VTI DISAGREEMENTS")
    p = results["variant_predictions"]
    p = p[p.target == "total_rescue"].copy()

    p["abs_delta"] = abs(p.observed_delta_pp)

    print(
        p.sort_values(
            "abs_delta", ascending=False
        )[[
            "variant_key",
            "protein_position",
            "eti_total_pp",
            "observed_total_rescue",
            "observed_delta_pp",
            "xgb_delta_pred_pp",
        ]].head(15).round(3).to_string(index=False)
    )

    print("\nG. SPLIT OVERLAP METADATA ONLY")
    metadata = split_metadata()
    print(metadata.to_string(index=False))

    print("\nH. INTERPRETATION LIMITATIONS")
    print("* Cross-study ETI/VTI measurement scales are not yet harmonised.")
    print("* Differences are not established clinical treatment benefits.")
    print("* 10/20 pp are exploratory thresholds, not clinical cutoffs.")
    print("* ETI measurement SEM and covariance not verified.")
    print("* Matched cohort is small and underrepresents high responders.")
    print("* Development OOF is not independent external validation.")
    print("* Two OOF repeats are averaged before prioritisation scoring.")

    OUT.mkdir(parents=True, exist_ok=True)

    for name, frame in {
        **results,
        "split_overlap": metadata,
    }.items():
        frame.to_csv(
            OUT / f"phase30e_{name}.csv",
            index=False,
        )

    provenance = {
        "status": "TRAIN-only, no model fitting",
        "n_full": 210,
        "n_matched": 66,
        "phase30a_sha256": digest(A),
        "phase30d_sha256": digest(D),
        "eti_source_sha256": digest(c.ETI_CSV),
        "thresholds_pp": [10, 20],
        "bootstraps": args.bootstraps,
        "clinical_validation": False,
    }

    (OUT / "phase30e_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print("\nPASS Phase30E; saved:", OUT)


if __name__ == "__main__":
    main()
