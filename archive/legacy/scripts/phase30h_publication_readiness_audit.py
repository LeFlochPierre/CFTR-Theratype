#!/usr/bin/env python3
"""Phase 30H: ETI/VTI publication-readiness audit.

TRAIN outcomes only; protected splits use identifiers only.
No model fitting or tuning.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/biology/phase30h"

A = ROOT / "results/biology/phase30a/phase30a_oof.csv"
E = ROOT / "results/biology/phase30e/phase30e_variant_predictions.csv"

OLD = ROOT / "data/processed/bihler_2024_missense_splits.parquet"
NEW = ROOT / "data/processed/bihler_2026_transfer_splits.parquet"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load():
    import phase30c_eti_incremental_ladder as c

    matched, _, _, _ = c.load_matched_data()

    full = pd.read_csv(A, dtype={"variant_key": str})
    predictions = pd.read_csv(E, dtype={"variant_key": str})

    # ONLY read identifiers and split assignments.
    old = pd.read_parquet(
        OLD, columns=["variant_key", "split"]
    )
    new = pd.read_parquet(
        NEW, columns=["variant_key", "transfer_split"]
    )

    old["variant_key"] = old.variant_key.astype(str)
    new["variant_key"] = new.variant_key.astype(str)

    return matched, full, predictions, old, new


def audit(matched, full, predictions, old, new, strict=True):

    required = [
        "repeat", "fold", "variant_key",
        "protein_position", "observed_baseline",
        "observed_corrector", "observed_total_rescue",
    ]

    if not set(required).issubset(full.columns):
        raise ValueError("Phase30A schema mismatch")

    if (
        full.duplicated(["repeat", "variant_key"]).any()
        or set(full.repeat) != {1, 2}
    ):
        raise ValueError("Invalid repeated OOF")

    for col in [
        "protein_position",
        "observed_baseline",
        "observed_corrector",
        "observed_total_rescue",
    ]:
        spread = (
            full.groupby("variant_key")[col].max()
            - full.groupby("variant_key")[col].min()
        )
        if spread.abs().max() > 1e-6:
            raise ValueError(f"Changed target: {col}")

    if (
        full.groupby(
            ["repeat", "protein_position"]
        ).fold.nunique().max() > 1
    ):
        raise ValueError("Position leakage in Phase30A")

    v = full.loc[
        full.repeat == 1, required[2:]
    ].copy()

    matched = matched.copy()
    matched["variant_key"] = matched.variant_key.astype(str)

    if (
        v.variant_key.duplicated().any()
        or matched.variant_key.duplicated().any()
    ):
        raise ValueError("Duplicate variant IDs")

    if not set(matched.variant_key).issubset(
        set(v.variant_key)
    ):
        raise ValueError("Protected variants in TRAIN")

    if strict and (
        len(full) != 420
        or len(v) != 210
        or v.protein_position.nunique() != 168
        or len(matched) != 66
        or matched.protein_position.nunique() != 56
    ):
        raise ValueError("Frozen cohort changed")

    lookup = v.set_index("variant_key")

    for col in [
        "protein_position",
        "observed_baseline",
        "observed_corrector",
        "observed_total_rescue",
    ]:
        expected = lookup.loc[
            matched.variant_key, col
        ].to_numpy(float)

        if not np.allclose(
            matched[col].to_numpy(float),
            expected,
            atol=1e-6,
            rtol=0,
        ):
            raise ValueError(f"Matched target mismatch: {col}")

    for col in [
        "eti_baseline_fsk",
        "eti_corrector_pp",
        "eti_total_pp",
        "observed_baseline",
        "observed_total_rescue",
    ]:
        if not np.isfinite(
            matched[col].to_numpy(float)
        ).all():
            raise ValueError(f"Invalid data: {col}")

    # A. METADATA-ONLY PAIRING

    if (
        old.groupby("variant_key").split.nunique().max() > 1
        or new.groupby(
            "variant_key"
        ).transfer_split.nunique().max() > 1
    ):
        raise ValueError("Contradictory splits")

    old = old.drop_duplicates("variant_key")
    new = new.drop_duplicates("variant_key")

    overlap = old.merge(
        new,
        on="variant_key",
        validate="one_to_one",
    )

    old_train = set(
        old.loc[old.split == "train", "variant_key"]
    )
    new_train = set(
        new.loc[
            new.transfer_split == "train", "variant_key"
        ]
    )

    if set(v.variant_key) != new_train:
        raise ValueError("VTI TRAIN membership mismatch")

    if not set(matched.variant_key).issubset(
        old_train & new_train
    ):
        raise ValueError("Matched cohort outside TRAIN")

    split_table = (
        overlap.groupby(["split", "transfer_split"])
        .size()
        .reset_index(name="paired_variant_ids")
    )

    print("\nA. ETI/VTI PAIRING — SPLIT METADATA ONLY")
    print(split_table.to_string(index=False))

    # B. REPRESENTATIVENESS

    v["has_eti"] = v.variant_key.isin(
        matched.variant_key
    )

    v["regime"] = np.select(
        [
            v.observed_corrector < 10,
            v.observed_corrector > 40,
        ],
        ["low", "high"],
        default="middle",
    )

    v["region"] = np.where(
        v.protein_position.between(500, 650),
        "500-650",
        "other",
    )

    coverage = []

    for axis in ["ALL", "regime", "region"]:
        groups = (
            [("ALL", v)]
            if axis == "ALL"
            else list(v.groupby(axis))
        )

        for group, subset in groups:
            selected = subset[subset.has_eti]

            coverage.append({
                "dimension": axis,
                "group": group,
                "all_variants": len(subset),
                "matched_variants": len(selected),
                "match_fraction":
                    len(selected) / len(subset),
                "all_positions":
                    subset.protein_position.nunique(),
                "matched_positions":
                    selected.protein_position.nunique(),
            })

    coverage = pd.DataFrame(coverage)

    print("\nB. MATCHED-COHORT REPRESENTATIVENESS")
    print(coverage.round(3).to_string(index=False))

    # C. UNTREATED BASELINE COMPARABILITY

    baseline = matched[[
        "variant_key",
        "protein_position",
        "eti_baseline_fsk",
        "observed_baseline",
    ]].copy()

    baseline["vti_minus_eti_baseline"] = (
        baseline.observed_baseline
        - baseline.eti_baseline_fsk
    )

    difference = baseline.vti_minus_eti_baseline

    baseline_summary = pd.DataFrame([{
        "n_variants": len(baseline),
        "n_positions": baseline.protein_position.nunique(),
        "eti_baseline_median":
            baseline.eti_baseline_fsk.median(),
        "vti_baseline_median":
            baseline.observed_baseline.median(),
        "difference_median": difference.median(),
        "median_absolute_difference":
            difference.abs().median(),
        "difference_q10": difference.quantile(0.10),
        "difference_q90": difference.quantile(0.90),
        "spearman": baseline.eti_baseline_fsk.corr(
            baseline.observed_baseline,
            method="spearman",
        ),
        "pearson": baseline.eti_baseline_fsk.corr(
            baseline.observed_baseline,
            method="pearson",
        ),
    }])

    print("\nC. CROSS-STUDY UNTREATED BASELINE COMPARISON")
    print(
        baseline_summary.round(3).to_string(index=False)
    )

    print(
        "These are different experimental campaigns, "
        "not paired technical replicates."
    )

    # D. REGION AND PREDICTION CONSISTENCY

    p = predictions[
        predictions.target == "total_rescue"
    ].copy()

    required_predictions = [
        "variant_key",
        "protein_position",
        "observed_delta_pp",
        "xgb_delta_pred_pp",
        "eti_total_pp",
        "observed_total_rescue",
    ]

    if not set(required_predictions).issubset(p.columns):
        raise ValueError("Missing Phase30E fields")

    if (
        p.variant_key.duplicated().any()
        or set(p.variant_key) != set(matched.variant_key)
    ):
        raise ValueError("Phase30E cohort mismatch")

    expected = matched.set_index("variant_key").loc[
        p.variant_key
    ]

    if not np.array_equal(
        p.protein_position.to_numpy(int),
        expected.protein_position.to_numpy(int),
    ):
        raise ValueError("Protein position mismatch")

    if not np.allclose(
        p.observed_total_rescue,
        expected.observed_total_rescue,
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("VTI outcome mismatch")

    if not np.allclose(
        p.eti_total_pp,
        expected.eti_total_pp,
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError("ETI measurement mismatch")

    if not np.allclose(
        p.observed_delta_pp,
        p.observed_total_rescue - p.eti_total_pp,
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("Response-difference mismatch")

    p["event_ge10"] = p.observed_delta_pp >= 10

    p["region"] = np.where(
        p.protein_position.between(500, 650),
        "500-650",
        "other",
    )

    representative = (
        p.sort_values(
            ["xgb_delta_pred_pp", "variant_key"],
            ascending=[False, True],
            kind="stable",
        )
        .drop_duplicates("protein_position")
    )

    if strict and len(representative) != 56:
        raise ValueError("Position count mismatch")

    region_results = []

    for subset_name, subset in [
        ("all_variants", p),
        ("xgb_selected_representative", representative),
    ]:
        for region, group in subset.groupby("region"):
            region_results.append({
                "sample": subset_name,
                "region": region,
                "n_variants": len(group),
                "n_positions":
                    group.protein_position.nunique(),
                "events_ge10": int(group.event_ge10.sum()),
                "event_fraction":
                    group.event_ge10.mean(),
            })

    region_results = pd.DataFrame(region_results)

    print("\nD. REGIONAL SENSITIVITY — POST-HOC")
    print(
        region_results.round(3).to_string(index=False)
    )

    # E. POSITION LABEL SENSITIVITY

    position = (
        p.groupby("protein_position")
        .agg(
            variants=("variant_key", "size"),
            any_event=("event_ge10", "any"),
            all_events=("event_ge10", "all"),
        )
        .reset_index()
    )

    mixed = position[
        (position.variants > 1)
        & (position.any_event != position.all_events)
    ]

    counts = pd.DataFrame([{
        "matched_variants": len(p),
        "unique_positions": len(position),
        "positive_variants": int(p.event_ge10.sum()),
        "positive_positions_any_variant":
            int(position.any_event.sum()),
        "positive_selected_representatives":
            int(representative.event_ge10.sum()),
        "multi_variant_positions":
            int((position.variants > 1).sum()),
        "mixed_outcome_positions": len(mixed),
    }])

    print("\nE. POSITION-LEVEL EVENT COUNT AUDIT")
    print(counts.to_string(index=False))

    # F. READINESS GATES

    split_lookup = {
        (row["split"], row["transfer_split"]):
            int(row["paired_variant_ids"])
        for _, row in split_table.iterrows()
    }

    validation_matches = split_lookup.get(
        ("train", "val"), 0
    )

    test_matches = split_lookup.get(
        ("train", "test"), 0
    )

    gates = pd.DataFrame([
        {
            "check": "ETI/VTI protocols identical",
            "status": "NO",
            "reason": "Published method differences",
        },
        {
            "check": "Untreated assay comparability",
            "status": "DESCRIPTIVE",
            "reason": "Quantified, not harmonised",
        },
        {
            "check": "Matched TRAIN cohort",
            "status": "SMALL",
            "reason": "66 variants, 56 positions",
        },
        {
            "check": "Paired VTI validation",
            "status": "PREVIOUSLY EXPLORED",
            "reason": f"{validation_matches} identifiers",
        },
        {
            "check": "Paired independent VTI test",
            "status": (
                "NONE IDENTIFIED"
                if test_matches == 0
                else "REQUIRES REVIEW"
            ),
            "reason": f"{test_matches} ETI-TRAIN/VTI-TEST matches",
        },
        {
            "check": "Regional effect replicated",
            "status": "NOT YET",
            "reason": "Region selected post hoc",
        },
        {
            "check": "Prospective prioritisation",
            "status": "NOT YET",
            "reason": "Development OOF only",
        },
    ])

    print("\nF. PUBLICATION READINESS")
    print(gates.to_string(index=False))

    return {
        "split_overlap": split_table,
        "coverage": coverage,
        "baseline_summary": baseline_summary,
        "baseline_pairs": baseline,
        "regional_sensitivity": region_results,
        "position_counts": counts,
        "mixed_positions": mixed,
        "readiness_gates": gates,
    }


def self_test():
    rng = np.random.default_rng(30)

    keys = np.array([
        f"SYN{i:03d}" for i in range(210)
    ])

    positions = np.r_[
        np.arange(168),
        np.arange(42),
    ]

    train = pd.DataFrame({
        "variant_key": keys,
        "protein_position": positions,
        "observed_baseline":
            rng.uniform(0, 70, 210),
        "observed_corrector":
            rng.uniform(0, 60, 210),
        "observed_total_rescue":
            rng.uniform(0, 90, 210),
    })

    full = pd.concat([
        train.assign(
            repeat=r,
            fold=(positions + (r - 1) * 3) % 5 + 1,
        )
        for r in (1, 2)
    ], ignore_index=True)

    chosen = np.r_[
        np.arange(56),
        np.arange(168, 178),
    ]

    matched = train.iloc[chosen].copy()

    matched["eti_baseline_fsk"] = (
        matched.observed_baseline
        + rng.normal(0, 3, 66)
    )
    matched["eti_corrector_pp"] = (
        matched.observed_corrector
        + rng.normal(0, 5, 66)
    )
    matched["eti_total_pp"] = (
        matched.observed_total_rescue
        + rng.normal(0, 9, 66)
    )

    p = matched.copy()
    p["target"] = "total_rescue"
    p["observed_delta_pp"] = (
        p.observed_total_rescue - p.eti_total_pp
    )
    p["xgb_delta_pred_pp"] = (
        p.observed_delta_pp + rng.normal(0, 5, 66)
    )

    old = pd.DataFrame({
        "variant_key": keys,
        "split": "train",
    })
    new = pd.DataFrame({
        "variant_key": keys,
        "transfer_split": "train",
    })

    output = audit(
        matched, full, p, old, new, strict=False
    )

    assert (
        output["position_counts"]
        .iloc[0].unique_positions == 56
    )

    corrupt = p.copy()
    corrupt.loc[
        corrupt.index[0], "observed_delta_pp"
    ] += 3

    try:
        audit(
            matched, full, corrupt, old, new,
            strict=False,
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Corrupted response difference not detected"
        )

    print("\nPASS Phase30H synthetic tests")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test", action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    matched, full, predictions, old, new = load()

    print("=" * 108)
    print(
        "PHASE 30H | ETI/VTI PUBLICATION READINESS"
    )
    print("=" * 108)
    print(
        "TRAIN outcomes only. Protected splits: "
        "metadata only. No model fitting."
    )

    results = audit(
        matched, full, predictions, old, new
    )

    OUT.mkdir(parents=True, exist_ok=True)

    for name, frame in results.items():
        frame.to_csv(
            OUT / f"phase30h_{name}.csv",
            index=False,
        )

    source_files = {}

    for path in [A, E, OLD, NEW]:
        source_files[str(path.relative_to(ROOT))] = {
            "sha256": sha256(path),
            "bytes": path.stat().st_size,
        }

    provenance = {
        "status": "TRAIN only; read-only",
        "references": {
            "ETI_2024":
                "https://doi.org/10.1016/j.jcf.2024.02.006",
            "VTI_2026":
                "https://doi.org/10.1016/j.jcf.2026.08.010",
        },
        "published_protocol_differences": [
            "Two versus four WT control wells",
            "Three laboratories versus one",
            "Different transepithelial resistance QC",
        ],
        "caveats": [
            "Assay normalisation not yet harmonised",
            "Region selected post hoc",
            "Previously explored validation is not untouched",
        ],
        "source_files": source_files,
    }

    (
        OUT / "phase30h_provenance.json"
    ).write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print("\nPASS Phase30H. Outputs:", OUT)


if __name__ == "__main__":
    main()
