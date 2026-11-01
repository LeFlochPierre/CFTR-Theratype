#!/usr/bin/env python3
"""Phase 21C: Tedman 2025 external CFTR expression signals vs frozen VTI TRAIN biology.

Exploratory biology audit ONLY, not a CV benchmark. Reads frozen TRAIN-only
VTI labels, never VTI/ETI validation/test outcomes. Builds a safe auxiliary
label table, but future nested CV MUST still remove each outer/inner held-out
protein position from supervised external training.

Use: python scripts/phase21c_external_biology_signal_audit.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cftr_theratype import model_zoo as mz
from cftr_theratype.transfer import ROOT

CANDIDATES = ROOT / "results/biology/phase21b/tedman2025_initial_aux_candidates.csv"
MATURATION = ROOT / "results/biology/phase21a/vti_train_maturation_assays.csv"
ACUTE = ROOT / "results/boltz/phase19b/vti_train_acute_iva_targets.csv"
SPLITS = ROOT / "data/processed/bihler_2026_transfer_splits.parquet"
OUT = ROOT / "results/biology/phase21c"
SOURCE_SHA256 = "116681f64f3527360d92d84ead2599b041a6e7605d4bc588c9bc4298ed8b19f8"
BASIS = ("parent", "canx_ko")
CONDITIONS = ("apo", "vx661", "vx445", "vx445_vx661")
MEANS = [f"{b}_{c}_mean" for b in BASIS for c in CONDITIONS]
PMS = [s.replace("_mean", "_pm") for s in MEANS]


def assert_safe_external(aux: pd.DataFrame, mem: pd.DataFrame):
    forbidden = set(mem.loc[mem.transfer_split.ne("train"), "protein_position"].dropna().astype(int))
    if "protected_2024_validation_or_test_position" in mem:
        raw = mem.protected_2024_validation_or_test_position
        flag = raw.fillna(False).astype(str).str.lower().isin(("true", "1", "yes", "t"))
        forbidden |= set(mem.loc[flag, "protein_position"].dropna().astype(int))
    assert not set(aux.protein_position.astype(int)).intersection(forbidden), "Protected position in auxiliary pool"
    for col in ("vti_val_position", "vti_test_position", "vti_excluded_position", "protected_2024_position"):
        if col in aux:
            bad = aux[col].astype(str).str.lower().isin(("true", "1", "yes", "t"))
            assert not bad.any(), f"Protected flag detected: {col}"
    assert aux["variant_short"].notna().all(), "Auxiliary row not protein-parseable"
    assert aux.variant_short.is_unique, "Duplicate external protein substitution"
    assert aux.protein_position.notna().all()


def rho(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 6 or len(np.unique(a[good])) < 2 or len(np.unique(b[good])) < 2:
        return np.nan, int(good.sum())
    stat = spearmanr(a[good], b[good])
    return float(stat.statistic), int(good.sum())


def main():
    for path in (CANDIDATES, MATURATION, ACUTE, SPLITS):
        if not path.is_file():
            raise FileNotFoundError(f"Missing required file: {path}")
    OUT.mkdir(parents=True, exist_ok=True)
    aux = pd.read_csv(CANDIDATES)
    mem = pd.read_parquet(SPLITS)
    assert {"transfer_split", "protein_position"}.issubset(mem)
    assert len(mem) == 415, f"Unexpected frozen partition size: {len(mem)}"
    assert set(MEANS + PMS + ["variant_short", "protein_position"]).issubset(aux), "Missing external columns"
    assert_safe_external(aux, mem)

    train, _features, _frozen_oof, _outer = mz.prepare(ROOT)
    assert len(train) == 210 and train.protein_position.nunique() == 168
    assert train.variant_key.is_unique
    train = train.copy()
    train["variant_short"] = (
        train.reference_aa.astype(str).str.upper() +
        train.protein_position.astype(int).astype(str) +
        train.alternate_aa.astype(str).str.upper()
    )
    assert (train.variant_short.str.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY][1-9][0-9]*[ACDEFGHIKLMNPQRSTVWY]", na=False)).all()

    train_positions = set(train.protein_position.astype(int))
    aux_train = aux.loc[aux.protein_position.astype(int).isin(train_positions)].copy()
    aux_novel = aux.loc[~aux.protein_position.astype(int).isin(train_positions)].copy()

    print("=" * 102)
    print("PHASE 21C | EXTERNAL BIOLOGY SIGNAL / INTERNAL CFTR MATURATION AUDIT")
    print("=" * 102)
    print("TRAIN-ONLY descriptive analysis; no protected VTI/ETI outcomes loaded")
    print(f"Auxiliary pool: {len(aux)} variants at {aux.protein_position.nunique()} positions")
    print(f"At VTI TRAIN positions: {len(aux_train)} variants / {aux_train.protein_position.nunique()} positions")
    print(f"At novel positions: {len(aux_novel)} variants / {aux_novel.protein_position.nunique()} positions")
    print(f"Expected outer-fold filtering: remove supervised external records at held-out protein positions")
    print("Note: ± fields are reported uncertainty but not assumed to be SEM or SD.")

    for c in MEANS + PMS:
        aux[c] = pd.to_numeric(aux[c], errors="coerce")
    if aux[MEANS].isna().any().any():
        raise ValueError("Missing candidate external expression values")
    if (aux[MEANS] < 0).any().any():
        raise ValueError("Negative intensity: log1p transformation invalid")
    for c in MEANS:
        aux[f"log1p_{c.removesuffix('_mean')}"] = np.log1p(aux[c])
    for basis in BASIS:
        base = f"log1p_{basis}_apo"
        for cond in CONDITIONS[1:]:
            aux[f"log_ratio_{basis}_{cond}_vs_apo"] = aux[f"log1p_{basis}_{cond}"] - aux[base]
    for cond in CONDITIONS:
        aux[f"log_ratio_canxko_parent_{cond}"] = aux[f"log1p_canx_ko_{cond}"] - aux[f"log1p_parent_{cond}"]
    aux_out = OUT / "tedman2025_candidate_biology_targets.csv"
    aux.to_csv(aux_out, index=False)
    print("\nEXTERNAL INTENSITY / UNCERTAINTY QC")
    for c in MEANS:
        intensity = aux[c].to_numpy(float)
        plusminus = aux[c.replace("_mean", "_pm")].to_numpy(float)
        rel = plusminus / np.maximum(intensity, 1.)
        print(f"{c:29s} median={np.median(intensity):9.2f}"
              f" IQR=({np.quantile(intensity,.25):.2f},{np.quantile(intensity,.75):.2f})"
              f" ±/intensity median={np.nanmedian(rel):.3f}"
              f" >0.5={np.count_nonzero(rel>0.5):3d}")
    print("\nEXTERNAL CFTR SUB-DOMAIN COVERAGE")
    domain_counts = aux.groupby("CFTR sub-domain", dropna=False).agg(
        substitutions=("variant_short", "nunique"),
        positions=("protein_position", "nunique"),
    ).sort_values("substitutions", ascending=False)
    print(domain_counts.to_string())
    print("\nCORRECTOR / CALNEXIN BIOLOGICAL CONTRASTS (log ratios of intensity+1)")
    for c in [f"log_ratio_{b}_vx445_vx661_vs_apo" for b in BASIS] + [
        f"log_ratio_canxko_parent_{condition}" for condition in ("apo", "vx445_vx661")]:
        x = aux[c]
        print(f"{c:45s} median={x.median():+.3f} p10={x.quantile(.1):+.3f} p90={x.quantile(.9):+.3f}")

    ac = pd.read_csv(ACUTE, dtype={"variant_key": str})
    assert {"variant_key", "acute_iva_vti_pp"}.issubset(ac)
    assert len(ac) == 210 and ac.variant_key.is_unique and set(ac.variant_key) == set(train.variant_key)
    ac = ac[["variant_key", "acute_iva_vti_pp"]]
    mat = pd.read_csv(MATURATION, dtype={"variant_key": str})
    assert {"variant_key", "Treatment", "band C, %WT", "Total protein, %WT"}.issubset(mat)
    assert len(mat) == 420
    assert set(mat.variant_key) == set(train.variant_key)
    assert not mat.duplicated(["variant_key", "Treatment"]).any()
    mat = mat.loc[mat.Treatment.eq("VX121+VX661"),
                  ["variant_key", "band C, %WT", "band C, %Total", "C-B ratio, %WT", "Total protein, %WT"]]
    assert len(mat) == 210
    vti = train[["variant_key", "variant_short", "protein_position"]].merge(
        ac, on="variant_key", validate="one_to_one").merge(mat, on="variant_key", validate="one_to_one")
    for c in ("acute_iva_vti_pp", "band C, %WT", "band C, %Total", "C-B ratio, %WT", "Total protein, %WT"):
        vti[c] = pd.to_numeric(vti[c], errors="coerce")
        assert np.isfinite(vti[c]).all(), f"Nonfinite TRAIN target: {c}"
    # Multiple nucleotide substitutions can encode the SAME protein mutation.
    # Aggregate them to one record for exact substitution-level comparisons.
    grouped = vti.groupby(["variant_short", "protein_position"], as_index=False).agg(
        n_nucleotide_variants=("variant_key", "size"),
        acute_iva_vti_pp=("acute_iva_vti_pp", "mean"),
        band_c_wt=("band C, %WT", "mean"),
        band_c_total=("band C, %Total", "mean"),
        cb_ratio=("C-B ratio, %WT", "mean"),
        total_protein=("Total protein, %WT", "mean"),
    )
    external_exact = aux.merge(
        grouped, on=["variant_short", "protein_position"], how="inner", validate="one_to_one"
    )
    external_exact["log_parent_dual"] = np.log1p(external_exact.parent_vx445_vx661_mean)
    external_exact["log_parent_apo"] = np.log1p(external_exact.parent_apo_mean)
    external_exact["log_parent_corrector_gain"] = (external_exact.log_parent_dual - external_exact.log_parent_apo)
    print("\nEXACT PROTEIN-SUBSTITUTION CROSS-STUDY MATCHES")
    print("Exact substitution matches:", len(external_exact))
    print("Nucleotide-level VTI training records represented:", external_exact.n_nucleotide_variants.sum())
    print("These are the only records used for direct cross-assay correlations;")
    print("a different substitution at the SAME position is NOT considered a match.")
    if not external_exact.empty:
        keys = ["variant_short", "protein_position", "n_nucleotide_variants", "band_c_wt",
                "acute_iva_vti_pp", "parent_vx445_vx661_mean", "log_parent_corrector_gain"]
        print(external_exact[keys].sort_values("protein_position").round(3).to_string(index=False))
    exact_file = OUT / "tedman_bihler_exact_train_matches.csv"
    external_exact.to_csv(exact_file, index=False)

    assays = {"baseline parent PME": "log_parent_apo",
              "parent dual-corrector PME": "log_parent_dual",
              "parent corrector log-gain": "log_parent_corrector_gain"}
    responses = {"treated band-C %WT": "band_c_wt", "treated mature fraction": "band_c_total",
                 "acute IVA increment": "acute_iva_vti_pp"}
    print("\nHIGH IVA RESPONDER COVERAGE IN THE EXACT CROSS-STUDY MATCHES")
    for threshold in (20., 40.):
        all_high = grouped.loc[grouped.acute_iva_vti_pp > threshold]
        matched_high = external_exact.loc[external_exact.acute_iva_vti_pp > threshold]
        print(f"> {threshold:g} pp: VTI TRAIN {len(all_high)} distinct protein substitutions; "
              f"Tedman exact matches {len(matched_high)}")
        if len(matched_high):
            print("  ", ", ".join(matched_high.variant_short.astype(str).tolist()))
    print("\nPRESPECIFIED EXPLORATORY CROSS-STUDY SPEARMAN CORRELATIONS")
    print("Limited sample size: descriptive only; do not interpret as validation or tune choices on these results.")
    corr_records = []
    for aname, acol in assays.items():
        for rname, rcol in responses.items():
            corr, n = rho(external_exact[acol], external_exact[rcol])
            corr_records.append({"external_measure": aname, "vti_measure": rname,
                                 "n_substitutions": n, "spearman_rho": corr})
            print(f"{aname:32s} vs {rname:26s} n={n:2d} rho={corr:+.3f}")
    corr_file = OUT / "phase21c_exact_match_correlations.csv"
    pd.DataFrame(corr_records).to_csv(corr_file, index=False)

    print("\nINTERNAL VTI BIOLOGY / ACUTE POTENTIATION CORRELATIONS")
    for label, col in (("band-C treated", "band C, %WT"),
                       ("mature fraction treated", "band C, %Total"),
                       ("total protein treated", "Total protein, %WT")):
        raw_corr, n = rho(vti[col], vti.acute_iva_vti_pp)
        posavg = vti.groupby("protein_position", as_index=False)[[col, "acute_iva_vti_pp"]].mean()
        pos_corr, np_ = rho(posavg[col], posavg.acute_iva_vti_pp)
        print(f"{label:27s} per-variant n={n} rho={raw_corr:+.3f}"
              f" | position-mean n={np_} rho={pos_corr:+.3f}")

    report = {
        "phase": "21C", "vti_train_variants": len(vti),
        "external_auxiliary_variants": len(aux), "external_auxiliary_positions": int(aux.protein_position.nunique()),
        "external_novel_variants": len(aux_novel), "external_novel_positions": int(aux_novel.protein_position.nunique()),
        "external_exact_vti_train_substitutions": len(external_exact),
        "cross_study_correlations_exploratory": True,
        "protected_vti_and_eti_labels_loaded": False,
        "future_nested_cv_constraint": "Exclude outer and inner held-out protein positions before external supervised training; no measured external phenotype at prediction time",
        "source_workbook_sha256": SOURCE_SHA256,
        "outputs": [str(p.relative_to(ROOT)) for p in (aux_out, exact_file, corr_file)],
    }
    report_path = OUT / "phase21c_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print("\nOUTPUTS")
    for p in (aux_out, exact_file, corr_file, report_path):
        print("  ", p.relative_to(ROOT))
    print("PASS: Matched exact protein substitutions; derived external biological targets; VTI TRAIN-only labels")
    print("STOP: No target modelling/outer CV run. Feature preprocessing and auxiliary supervision must be fold-local.")


if __name__ == "__main__":
    main()
