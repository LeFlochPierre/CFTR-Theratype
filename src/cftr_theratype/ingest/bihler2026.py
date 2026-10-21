"""Ingest Bihler et al. (2026) VTI assay without modifying 2024 labels."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from cftr_theratype.build import ROOT

SOURCE = ROOT / "data/raw/bihler_2026/mmc1.xlsx"
OUTPUT = ROOT / "data/processed"
MANIFEST = ROOT / "data/manifests/data_sources.json"


def treatment_name(raw):
    name = re.sub(r"[^A-Z0-9]", "", str(raw).upper())
    if name in {"DMSO", "VEHICLE"}:
        return "DMSO"
    if (any(s in name for s in ("VNZ", "VANZ", "VX121"))
            and any(s in name for s in ("TEZ", "VX661"))):
        return "VNZ/TEZ"
    return "UNKNOWN"


def build_2026():
    print("=" * 80)
    print("CFTR-THERATYPE | BUILD BIHLER 2026 VTI DATASET")
    print("=" * 80)

    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    df = pd.read_excel(SOURCE, sheet_name="By name", header=3)
    df.columns = df.columns.astype(str).str.strip()
    needed = ["Variant", "HGVS cDNA", "HGVS protein", "Variant type",
              "Treatment", "FSK, %WT", "FSK + IVA, %WT", "Study", "Response group"]
    missing = set(needed) - set(df.columns)
    if missing:
        raise ValueError(f"2026 source is missing columns: {sorted(missing)}")

    df["source_excel_row"] = df.index + 5
    df = df[df["Variant"].notna() & df["Treatment"].notna()].copy()
    df["treatment_norm"] = df["Treatment"].map(treatment_name)

    print("\n[1] RAW TREATMENTS AND COHORTS")
    print(df["Treatment"].value_counts(dropna=False).to_string())
    print("\nStudy strata:")
    print(df["Study"].fillna("<missing>").value_counts().to_string())
    print("\nResponse groups:")
    print(df["Response group"].fillna("<missing>").value_counts().to_string())

    unknown = df.loc[df.treatment_norm == "UNKNOWN", "Treatment"].unique().tolist()
    if unknown:
        raise ValueError(f"Unrecognised treatments; check their experimental identity: {unknown}")
    if set(df.treatment_norm) != {"DMSO", "VNZ/TEZ"}:
        raise ValueError("Expected both DMSO and VNZ/TEZ experimental conditions")

    cdna = df["HGVS cDNA"].astype("string").str.strip()
    df["variant_key"] = cdna.where(cdna.notna() & cdna.ne(""), pd.NA)
    names = df["Variant"].astype("string").str.strip()
    df["is_control"] = names.str.contains(r"ctrl|control|^WT$|wild.?type", case=False,
                                           regex=True, na=False)
    df["is_alternative"] = names.str.contains(r"\*", regex=True, na=False)
    df["study_cohort"] = df["Study"].astype("string").str.strip()
    for col in ("FSK, %WT", "FSK + IVA, %WT"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Preserve every original condition and its explicit provenance.
    OUTPUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT / "bihler_2026_conditions.parquet", index=False)

    eligible = df.loc[
        ~df.is_control & ~df.is_alternative & df.variant_key.notna()
    ].copy()
    n_noneligible = len(df) - len(eligible)
    counts = eligible.groupby(["variant_key", "treatment_norm"]).size().unstack(fill_value=0)
    for condition in ("DMSO", "VNZ/TEZ"):
        if condition not in counts.columns:
            raise ValueError(f"No {condition} treatment observations found")
    eligible_keys = counts.index[(counts["DMSO"] == 1) & (counts["VNZ/TEZ"] == 1)]
    unresolved_keys = counts.index.difference(eligible_keys)
    unresolved = eligible[eligible.variant_key.isin(unresolved_keys)]
    unresolved.to_csv(OUTPUT / "bihler_2026_unresolved.csv", index=False)

    vehicle = eligible.loc[
        eligible.variant_key.isin(eligible_keys) & eligible.treatment_norm.eq("DMSO")
    ].set_index("variant_key").reindex(eligible_keys)
    treated = eligible.loc[
        eligible.variant_key.isin(eligible_keys) & eligible.treatment_norm.eq("VNZ/TEZ")
    ].set_index("variant_key").reindex(eligible_keys)

    for column in ("HGVS protein", "Variant type", "Study", "Response group"):
        a = vehicle[column].fillna("<missing>").astype(str)
        b = treated[column].fillna("<missing>").astype(str)
        if not a.equals(b):
            mismatch = a.index[a.ne(b)].tolist()[:10]
            raise ValueError(f"Disagreement between treatment rows in {column}: {mismatch}")

    paired = pd.DataFrame(index=eligible_keys)
    paired.index.name = "variant_key"
    paired["variant"] = vehicle["Variant"]
    paired["hgvs_cdna"] = vehicle["HGVS cDNA"]
    paired["hgvs_protein"] = vehicle["HGVS protein"]
    paired["variant_type"] = vehicle["Variant type"]
    paired["study_cohort"] = vehicle["Study"]
    paired["response_group_reported"] = vehicle["Response group"]
    paired["source_row_dmso"] = vehicle["source_excel_row"]
    paired["source_row_vnz_tez"] = treated["source_excel_row"]

    paired["baseline_fsk_pct_wt"] = vehicle["FSK, %WT"]
    paired["iva_treated_pct_wt"] = vehicle["FSK + IVA, %WT"]
    paired["vnz_tez_fsk_pct_wt"] = treated["FSK, %WT"]
    paired["vti_treated_pct_wt"] = treated["FSK + IVA, %WT"]
    paired["iva_rescue_pp"] = paired.iva_treated_pct_wt - paired.baseline_fsk_pct_wt
    paired["corrector_rescue_pp"] = paired.vnz_tez_fsk_pct_wt - paired.baseline_fsk_pct_wt
    paired["vti_rescue_pp"] = paired.vti_treated_pct_wt - paired.baseline_fsk_pct_wt
    paired["vti_increment_over_iva_pp"] = paired.vti_treated_pct_wt - paired.iva_treated_pct_wt

    if "# Wells" in df:
        paired["wells_dmso"] = pd.to_numeric(vehicle["# Wells"], errors="coerce")
        paired["wells_vnz_tez"] = pd.to_numeric(treated["# Wells"], errors="coerce")

    for raw, name in (
        ("FSK (%WT) SEM", "fsk_sem"),
        ("FSK+IVA (%WT) SEM", "fsk_iva_sem"),
    ):
        if raw in df:
            paired[name + "_dmso"] = pd.to_numeric(vehicle[raw], errors="coerce")
            paired[name + "_vnz_tez"] = pd.to_numeric(treated[raw], errors="coerce")

    mandatory = ["baseline_fsk_pct_wt", "iva_treated_pct_wt",
                 "vnz_tez_fsk_pct_wt", "vti_treated_pct_wt"]
    complete = paired[mandatory].notna().all(axis=1)
    incomplete_n = int((~complete).sum())
    if incomplete_n:
        paired.loc[~complete].reset_index().to_csv(
            OUTPUT / "bihler_2026_incomplete_pairs.csv", index=False
        )
    model_df = paired.loc[complete].reset_index()
    model_df.to_parquet(OUTPUT / "bihler_2026_vti.parquet", index=False)

    print("\n[2] PAIRING QC")
    print(f"Source rows:               {len(df)}")
    print(f"Controls/alternate/missing: {n_noneligible}")
    print(f"Candidate identifiers:     {len(counts)}")
    print(f"Unambiguous pairs:        {len(eligible_keys)}")
    print(f"Unresolved identifiers:   {len(unresolved_keys)}")
    print(f"Incomplete pairs:         {incomplete_n}")
    print(f"Usable VTI pairs:         {len(model_df)}")
    print("\nUsable pairs by original study stratum:")
    print(model_df.study_cohort.fillna("<missing>").value_counts().to_string())
    print("\nVTI rescue (pp of WT):")
    print(model_df.vti_rescue_pp.describe(percentiles=[.05, .5, .95]).to_string())

    old_file = OUTPUT / "bihler_2024_theratyping.parquet"
    if old_file.is_file():
        previous = pd.read_parquet(old_file, columns=["variant_key", "hgvs_protein"])
        shared = previous.merge(model_df[["variant_key", "hgvs_protein", "study_cohort"]], on="variant_key", how="inner", validate="one_to_one", suffixes=("_2024", "_2026"))
        consistent = shared.hgvs_protein_2024.fillna("<missing>").eq(
            shared.hgvs_protein_2026.fillna("<missing>")
        )
        print("\n[3] CROSS-STUDY MATCHING (NOT INDEPENDENT TESTING)")
        print(f"Exact cDNA overlaps:       {len(shared)}")
        print(f"Consistent protein HGVS:   {int(consistent.sum())}/{len(shared)}")
        print(f"Non-overlapping VTI pairs: {len(model_df) - len(shared)}")
        if (~consistent).any():
            print("Protein annotation differences:")
            print(shared.loc[~consistent, ["variant_key", "hgvs_protein_2024", "hgvs_protein_2026"]].head(10).to_string(index=False))
        shared.to_csv(OUTPUT / "bihler_2024_2026_exact_overlaps.csv", index=False)

    source_sha = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest["bihler_2026_build"] = {
        "source_sha256": source_sha,
        "treatment": "VNZ+TEZ (24h) then acute IVA",
        "primary_label": "vti_rescue_pp",
        "label_formula": "(VNZ+TEZ pretreated FSK+IVA %WT) - (DMSO pretreated FSK %WT)",
        "n_complete_pairs": len(model_df),
        "n_unresolved_identifiers": len(unresolved_keys),
        "note": "In-vitro FRT assay; not direct VTD clinical outcome. Study strata and response group retained as metadata only."
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print("\nOutputs: data/processed/bihler_2026_vti.parquet")
    print("2024 ETI model labels and held-out split were not modified.")
    print("BUILD COMPLETE")
