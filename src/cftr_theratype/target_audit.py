"""Training-only target definitions and assay-quality audit for CFTR-Theratype.

This is descriptive and read-only; never train on, nor read outcome rows from,
held-out validation/test partitions. Run: python -m cftr_theratype.target_audit
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SETTINGS = {
    "ETI": {
        "paired": "bihler_2024_theratyping.parquet",
        "conditions": "bihler_2024_conditions.parquet",
        "membership": "bihler_2024_missense_splits.parquet",
        "split_col": "split",
        "split_value": "train",
        "expected_n": 422,
        "rescue": "eti_rescue_pp",
        "increment": "eti_increment_over_iva_pp",
        "treated": "eti_treated_pct_wt",
        "baseline_sem": "baseline_fsk_sem",
        "iva_sem": "iva_treated_sem",
        "treated_sem": "iva_treated_sem_eti_condition",
        "treated_source_row": "source_row_elx_tez",
    },
    "VTI": {
        "paired": "bihler_2026_vti.parquet",
        "conditions": "bihler_2026_conditions.parquet",
        "membership": "bihler_2026_transfer_splits.parquet",
        "split_col": "transfer_split",
        "split_value": "TRAIN",
        "expected_n": 210,
        "rescue": "vti_rescue_pp",
        "increment": "vti_increment_over_iva_pp",
        "treated": "vti_treated_pct_wt",
        "baseline_sem": "fsk_sem_dmso",
        "iva_sem": "fsk_iva_sem_dmso",
        "treated_sem": "fsk_iva_sem_vnz_tez",
        "treated_source_row": "source_row_vnz_tez",
    },
}

COND_FIELDS = [
    "variant_key", "source_excel_row", "band C, %WT", "band C, %Total",
    "C-B ratio, %WT", "Total protein, %WT", "band C (%WT) SEM",
]


def selected_rows(path: Path, allowed_keys: list[str], columns: list[str]) -> pd.DataFrame:
    """Arrow predicate filtering: only authorised keys returned to analysis code."""
    if not allowed_keys:
        raise ValueError("Empty authorised training key list")
    import pyarrow.dataset as ds
    d = ds.dataset(str(path), format="parquet")
    missing = set(columns) - set(d.schema.names)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    table = d.to_table(
        columns=columns,
        filter=ds.field("variant_key").isin(allowed_keys),
    )
    frame = table.to_pandas()
    if not set(frame["variant_key"]).issubset(set(allowed_keys)):
        raise AssertionError("Outcome rows beyond authorised training keys")
    return frame


def approximate_diff_sem(a: pd.Series, b: pd.Series) -> pd.Series:
    """sqrt(SEM_A^2 + SEM_B^2) if independent; NOT an actual paired-label SEM."""
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    if (a.dropna() < 0).any() or (b.dropna() < 0).any():
        raise ValueError("Negative SEM in source data")
    return np.hypot(a, b)


def attach_condition_rows(paired: pd.DataFrame, conditions: pd.DataFrame, source_col: str,
                          prefix: str) -> pd.DataFrame:
    """Match the EXACT source workbook row, not a duplicate/control measurement."""
    if conditions.duplicated(["variant_key", "source_excel_row"]).any():
        raise ValueError("Nonunique condition source-row identifiers")
    rhs = conditions.rename(columns={
        "source_excel_row": source_col,
        **{c: prefix + c for c in COND_FIELDS if c not in {"variant_key", "source_excel_row"}}
    })
    merged = paired.merge(rhs, how="left", on=["variant_key", source_col],
                          validate="one_to_one", indicator=True)
    if not (merged["_merge"] == "both").all():
        raise ValueError(f"Condition rows missing for {prefix}")
    return merged.drop(columns="_merge")


def summarise_numeric(values: pd.Series) -> dict:
    x = pd.to_numeric(values, errors="coerce")
    finite = x[np.isfinite(x)]
    if finite.empty:
        return {"n": 0, "median": None, "q25": None, "q75": None, "negative_fraction": None}
    return {
        "n": int(len(finite)), "median": float(finite.median()),
        "q25": float(finite.quantile(.25)), "q75": float(finite.quantile(.75)),
        "negative_fraction": float((finite < 0).mean()),
    }


def audit_one(root: Path, label: str) -> tuple[pd.DataFrame, dict]:
    cfg = SETTINGS[label]
    base = root / "data" / "processed"
    membership = pd.read_parquet(base / cfg["membership"],
                                 columns=["variant_key", cfg["split_col"]])
    train = membership.loc[membership[cfg["split_col"]].astype(str).str.lower()
                           == cfg["split_value"].lower(), "variant_key"]
    keys = train.dropna().astype(str).tolist()
    if len(keys) != cfg["expected_n"] or len(set(keys)) != len(keys):
        raise ValueError(f"{label}: expected {cfg['expected_n']} unique training variants; got {len(keys)}")

    fields = ["variant_key", "source_row_dmso", cfg["treated_source_row"],
              "baseline_fsk_pct_wt", "iva_treated_pct_wt", cfg["treated"],
              cfg["rescue"], cfg["increment"], cfg["baseline_sem"],
              cfg["iva_sem"], cfg["treated_sem"]]
    if label == "VTI":
        fields += ["study_cohort", "wells_dmso", "wells_vnz_tez"]
    paired = selected_rows(base / cfg["paired"], keys, fields)
    if len(paired) != cfg["expected_n"] or paired["variant_key"].duplicated().any():
        raise ValueError(f"{label}: paired table missing/duplicating authorised training variants")
    if set(paired["variant_key"]) != set(keys):
        raise ValueError(f"{label}: paired table key mismatch")

    identities = {
        "rescue": pd.to_numeric(paired[cfg["treated"]], errors="coerce")
                  - pd.to_numeric(paired["baseline_fsk_pct_wt"], errors="coerce")
                  - pd.to_numeric(paired[cfg["rescue"]], errors="coerce"),
        "increment": pd.to_numeric(paired[cfg["treated"]], errors="coerce")
                     - pd.to_numeric(paired["iva_treated_pct_wt"], errors="coerce")
                     - pd.to_numeric(paired[cfg["increment"]], errors="coerce"),
    }
    for name, diff in identities.items():
        if diff.isna().any() or not np.allclose(diff.to_numpy(), 0.0, atol=1e-6):
            raise ValueError(f"{label}: {name} is not the expected arithmetic contrast")

    paired["approx_rescue_sem_independent"] = approximate_diff_sem(
        paired[cfg["baseline_sem"]], paired[cfg["treated_sem"]])
    paired["approx_increment_sem_independent"] = approximate_diff_sem(
        paired[cfg["iva_sem"]], paired[cfg["treated_sem"]])

    conditions = selected_rows(base / cfg["conditions"], keys, COND_FIELDS)
    paired = attach_condition_rows(paired, conditions, "source_row_dmso", "dmso_")
    paired = attach_condition_rows(paired, conditions, cfg["treated_source_row"], "treated_")

    ret = {
        "study": label, "train_variants": len(paired),
        "targets": {name: summarise_numeric(paired[col]) for name, col in {
            "absolute_treated": cfg["treated"],
            "total_rescue": cfg["rescue"],
            "corrector_increment_over_iva": cfg["increment"],
            "baseline_fsk": "baseline_fsk_pct_wt", "iva_treated": "iva_treated_pct_wt",
        }.items()},
        "approx_sem": {name: summarise_numeric(paired[col]) for name, col in {
            "total_rescue": "approx_rescue_sem_independent",
            "corrector_increment_over_iva": "approx_increment_sem_independent",
        }.items()},
        "condition_coverage": {},
    }
    for stage in ("dmso_", "treated_"):
        for name in ("band C, %WT", "band C, %Total", "C-B ratio, %WT", "Total protein, %WT", "band C (%WT) SEM"):
            ser = pd.to_numeric(paired[stage + name], errors="coerce")
            ret["condition_coverage"][stage + name] = {
                "n_finite": int(np.isfinite(ser).sum()), "n_total": len(ser),
            }
    if label == "VTI":
        ret["cohorts"] = {}
        for cohort, grp in paired.groupby("study_cohort", dropna=False):
            ret["cohorts"][str(cohort)] = {
                "n": len(grp),
                "rescue": summarise_numeric(grp[cfg["rescue"]]),
                "approx_rescue_sem_independent": summarise_numeric(grp["approx_rescue_sem_independent"]),
            }
    return paired, ret


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output_dir = root / "results" / "biology"
    reports = {}
    detailed = []
    print("=" * 82)
    print("CFTR-THERATYPE | PHASE 14 TARGET & ASSAY-QUALITY AUDIT — TRAIN ONLY")
    print("=" * 82)
    for label in ("ETI", "VTI"):
        data, report = audit_one(root, label)
        reports[label] = report
        data.insert(0, "study", label)
        detailed.append(data)
        print(f"\n[{label}] {len(data)} training variants. Label identities: PASS")
        for t, stat in report["targets"].items():
            print(f"  {t:31s} median={stat['median']:8.3f}"
                  f"  IQR=[{stat['q25']:7.3f}, {stat['q75']:7.3f}]"
                  f"  negative={stat['negative_fraction']:.1%}")
        for t, stat in report["approx_sem"].items():
            print(f"  Approx SEM ({t}): median={stat['median']}; available={stat['n']}/{len(data)}")
        for stage in ("dmso_", "treated_"):
            p = report["condition_coverage"][stage + "band C, %WT"]
            print(f"  Band C %WT {stage[:-1]}: {p['n_finite']}/{p['n_total']} finite")
        for cohort, summary in report.get("cohorts", {}).items():
            print(f"  Cohort {cohort}: n={summary['n']}, "
                  f"rescue median={summary['rescue']['median']:.3f}, "
                  f"SEM proxy median={summary['approx_rescue_sem_independent']['median']}")
    output_dir.mkdir(parents=True, exist_ok=True)
    all_data = pd.concat(detailed, ignore_index=True, sort=False)
    csv_file = output_dir / "phase14_train_only_target_audit.csv"
    json_file = output_dir / "phase14_target_audit_summary.json"
    all_data.to_csv(csv_file, index=False)
    payload = {
        "scope": "Frozen ETI and VTI TRAIN partitions only",
        "sem_caveat": "sqrt(SEM_A^2+SEM_B^2) assumes independence, ignores covariance and WT normalisation errors; not a validated label SEM or noise floor",
        "results": reports,
    }
    json_file.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print("\nWARNING: The reported difference SEM assumes independent measurements.")
    print("  Cross-condition covariance/normalisation uncertainty is unknown; do NOT treat")
    print("  these values as calibrated label uncertainty or a statistical noise floor.")
    print(f"Saved: {csv_file.relative_to(root)}")
    print(f"Saved: {json_file.relative_to(root)}")
    print("No validation/test outcomes were selected. Frozen datasets unchanged.")
    print("=" * 82)


if __name__ == "__main__":
    main()
