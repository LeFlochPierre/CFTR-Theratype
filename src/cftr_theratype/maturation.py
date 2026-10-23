"""Train-only analysis of CFTR maturation vs functional response.

Run: python -m cftr_theratype.maturation
No fitting, tuning, or reading validation/test outcomes; band C is used for
mechanistic diagnostics, NOT as an inference-time feature.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cftr_theratype.target_audit import SETTINGS, audit_one

BOOTSTRAPS = 500
SEED = 20261023
PAIRS = [
    ("band_c_gain_pct_wt", "corrector_increment", True, "Primary: maturity gain vs corrector increment"),
    ("band_c_gain_pct_wt", "total_rescue", True, "Maturity gain vs total rescue"),
    ("band_c_treated_pct_wt", "absolute_treated", True, "Mature protein vs treated function"),
    ("band_c_dmso_pct_wt", "baseline_fsk", False, "Baseline mature protein vs baseline function"),
    ("band_c_gain_pct_total", "corrector_increment", True, "Sensitivity: C fraction gain vs corrector increment"),
]


def _finite(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def position_table(df: pd.DataFrame, fields: list[str]) -> pd.DataFrame:
    """Group mean by protein position to avoid treating variants at one locus as independent."""
    if df["protein_position"].isna().any():
        raise ValueError("Missing protein position in authorised training rows")
    if df["variant_key"].duplicated().any():
        raise ValueError("Duplicate training variant keys")
    cols = {f: _finite(df[f]) for f in fields}
    work = pd.DataFrame(cols)
    work.insert(0, "protein_position", df["protein_position"].to_numpy())
    return work.groupby("protein_position", as_index=False).mean(numeric_only=True)


def rank_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray | None = None) -> float:
    """Rank association; with controls, partial Spearman via rank-residual correlation.

    This is descriptive association, not causal adjustment.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    if x.size != y.size or x.size < 8:
        return float("nan")
    rx = pd.Series(x).rank().to_numpy(dtype=float)
    ry = pd.Series(y).rank().to_numpy(dtype=float)
    if controls is not None:
        controls = np.asarray(controls, dtype=float)
        if controls.ndim != 2 or controls.shape[0] != len(rx):
            raise ValueError("Controls must be a [n_positions, n_covariates] array")
        z = np.column_stack([np.ones(len(rx)), *[
            pd.Series(controls[:, i]).rank().to_numpy(dtype=float)
            for i in range(controls.shape[1])
        ]])
        rx = rx - z @ np.linalg.lstsq(z, rx, rcond=None)[0]
        ry = ry - z @ np.linalg.lstsq(z, ry, rcond=None)[0]
    if np.std(rx) < 1e-10 or np.std(ry) < 1e-10:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def position_bootstrap_association(position_df: pd.DataFrame,
                                   biomarker: str, target: str,
                                   adjust: bool, *, bootstraps: int = BOOTSTRAPS,
                                   seed: int = SEED) -> dict:
    covariates = ["baseline_fsk", "iva_treated"] if adjust else []
    fields = [biomarker, target] + covariates
    data = position_df[fields].replace([np.inf, -np.inf], np.nan).dropna()
    n = len(data)
    if n < 15:
        return {"positions": n, "rho": None, "ci_low": None, "ci_high": None,
                "bootstrap_valid": 0}
    a = data[biomarker].to_numpy(float)
    y = data[target].to_numpy(float)
    cov = data[covariates].to_numpy(float) if covariates else None
    observed = rank_corr(a, y, cov)
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(bootstraps):
        idx = rng.integers(0, n, n)
        v = rank_corr(a[idx], y[idx], cov[idx] if cov is not None else None)
        if np.isfinite(v):
            boot.append(v)
    return {
        "positions": n,
        "rho": float(observed) if np.isfinite(observed) else None,
        "ci_low": float(np.quantile(boot, 0.025)) if len(boot) >= 20 else None,
        "ci_high": float(np.quantile(boot, 0.975)) if len(boot) >= 20 else None,
        "bootstrap_valid": len(boot),
    }


def assemble(root: Path, label: str) -> pd.DataFrame:
    """Select only frozen TRAIN outcomes through the pre-existing audited loader."""
    paired, _ = audit_one(root, label)
    cfg = SETTINGS[label]
    base = root / "data" / "processed"
    membership = pd.read_parquet(base / cfg["membership"],
                                 columns=["variant_key", "protein_position", cfg["split_col"]])
    membership = membership.loc[membership[cfg["split_col"]].astype(str).str.upper() == "TRAIN",
                                ["variant_key", "protein_position"]]
    if membership.duplicated("variant_key").any():
        raise ValueError("Duplicate frozen training membership")
    x = paired.merge(membership, on="variant_key", how="left", validate="one_to_one")
    if len(x) != cfg["expected_n"] or x["protein_position"].isna().any():
        raise ValueError("Training membership and assay outcomes do not match")
    x["study"] = label
    x["baseline_fsk"] = _finite(x["baseline_fsk_pct_wt"])
    x["iva_treated"] = _finite(x["iva_treated_pct_wt"])
    x["absolute_treated"] = _finite(x[cfg["treated"]])
    x["total_rescue"] = _finite(x[cfg["rescue"]])
    x["corrector_increment"] = _finite(x[cfg["increment"]])
    x["band_c_dmso_pct_wt"] = _finite(x["dmso_band C, %WT"])
    x["band_c_treated_pct_wt"] = _finite(x["treated_band C, %WT"])
    x["band_c_gain_pct_wt"] = x["band_c_treated_pct_wt"] - x["band_c_dmso_pct_wt"]
    x["band_c_gain_pct_total"] = (_finite(x["treated_band C, %Total"])
                                   - _finite(x["dmso_band C, %Total"]))
    x["approx_rescue_sem_independent"] = _finite(x["approx_rescue_sem_independent"])
    if label == "ETI":
        x["study_cohort"] = "ETI training"
    return x


def summarise_stratum(sub: pd.DataFrame, name: str) -> tuple[dict, list[dict]]:
    fields = sorted(set(a for a, _, _, _ in PAIRS) | set(b for _, b, _, _ in PAIRS)
                    | {"baseline_fsk", "iva_treated"})
    pos = position_table(sub, fields)
    row = {"stratum": name, "n_variants": len(sub), "n_positions": len(pos)}
    for f in ("band_c_dmso_pct_wt", "band_c_treated_pct_wt",
              "band_c_gain_pct_wt", "band_c_gain_pct_total"):
        clean = _finite(sub[f]).dropna()
        row[f + "_finite"] = len(clean)
        row[f + "_median"] = float(clean.median()) if len(clean) else None
        row[f + "_q25"] = float(clean.quantile(.25)) if len(clean) else None
        row[f + "_q75"] = float(clean.quantile(.75)) if len(clean) else None
    comparisons = []
    for i, (marker, target, adjust, description) in enumerate(PAIRS):
        stat = position_bootstrap_association(pos, marker, target, adjust,
                                               seed=SEED + i)
        comparisons.append({"stratum": name, "biomarker": marker,
                            "target": target, "rank_adjusted": adjust,
                            "description": description, **stat})
    return row, comparisons


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    eti, vti = assemble(root, "ETI"), assemble(root, "VTI")
    groups = [
        ("ETI training", eti),
        ("VTI training overall", vti),
        ("VTI prior ETI", vti[vti["study_cohort"] == "ETI, VTI"].copy()),
        ("VTI only", vti[vti["study_cohort"] == "VTI"].copy()),
    ]
    if len(groups[2][1]) != 66 or len(groups[3][1]) != 144:
        raise ValueError("Unexpected frozen VTI training cohort composition")
    print("=" * 87)
    print("CFTR-THERATYPE | PHASE 14B MATURATION–FUNCTION ASSOCIATIONS — TRAIN ONLY")
    print("=" * 87)
    rows, comparisons = [], []
    for name, df in groups:
        summary, results = summarise_stratum(df, name)
        rows.append(summary)
        comparisons.extend(results)
        print(f"\n[{name}] {summary['n_variants']} variants / {summary['n_positions']} positions")
        for field in ("band_c_dmso_pct_wt", "band_c_treated_pct_wt", "band_c_gain_pct_wt"):
            print(f"  {field:30s} median={summary[field + '_median']:+8.3f} "
                  f"IQR=[{summary[field+'_q25']:+8.3f},{summary[field+'_q75']:+8.3f}]")
        print("  Equal-position rank correlations (95% position-bootstrap CI):")
        for result in results:
            r, lo, hi = (result[k] for k in ("rho", "ci_low", "ci_high"))
            val = f"{r:+.3f} [{lo:+.3f},{hi:+.3f}]" if r is not None and lo is not None else "N/A"
            qualifier = "partial*" if result["rank_adjusted"] else "raw"
            print(f"    {result['description'][:53]:53s} {qualifier:8s} {val}")
    target_dir = root / "results" / "biology"
    target_dir.mkdir(parents=True, exist_ok=True)
    position_file = target_dir / "phase14b_maturation_training_rows.csv"
    comparison_file = target_dir / "phase14b_maturation_associations.csv"
    summary_file = target_dir / "phase14b_maturation_summary.json"
    keep = ["variant_key", "protein_position", "study", "study_cohort",
            "baseline_fsk", "iva_treated", "absolute_treated", "total_rescue",
            "corrector_increment", "band_c_dmso_pct_wt", "band_c_treated_pct_wt",
            "band_c_gain_pct_wt", "band_c_gain_pct_total", "approx_rescue_sem_independent"]
    pd.concat([eti[keep], vti[keep]], ignore_index=True).to_csv(position_file, index=False)
    pd.DataFrame(comparisons).to_csv(comparison_file, index=False)
    summary_file.write_text(json.dumps({
        "design": "Frozen ETI and VTI TRAIN only; no outcome rows from validation or test",
        "association": "Equal-position rank correlations; partial* uses residualised ranks conditional on baseline FSK and DMSO FSK+IVA",
        "inference_limitation": "Band C is measured after treatment; never use it as an input to a sequence-only predictor",
        "multiplicity": "Exploratory associations, no confirmatory p-values or model selection",
        "bootstrap_replicates": BOOTSTRAPS, "seed": SEED,
        "strata": rows, "associations": comparisons,
    }, indent=2, allow_nan=False) + "\n")
    print("\n*Partial rank association after adjusting for measured baseline FSK and FSK+IVA.")
    print("  Adjustment is associative, not causal. SEMs not propagated to this correlation.")
    print("  Band C is post-treatment information: it is not an inference-time feature.")
    print("  Multiple exploratory comparisons; no outcome rows from validation/test accessed.")
    for path in (position_file, comparison_file, summary_file):
        print(f"Saved: {path.relative_to(root)}")
    print("=" * 87)


if __name__ == "__main__":
    main()
