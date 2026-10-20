"""Build reproducible, variant-level CFTR theratyping labels."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/bihler_2024/mmc3.xlsx"
OUTPUT = ROOT / "data/processed"
MANIFEST = ROOT / "data/manifests/data_sources.json"

FUNCTION = "FSK, %WT"
FUNCTION_IVA = "FSK + IVA, %WT"


def normalize_treatment(value):
    text = re.sub(r"[\s/+_-]", "", str(value).upper())

    if text == "DMSO":
        return "DMSO"

    if ("ELX" in text or "ELEXACAFTOR" in text) and (
        "TEZ" in text or "TEZACAFTOR" in text
    ):
        return "ELX/TEZ"

    return str(value).strip()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build():
    print("=" * 85)
    print("CFTR-THERATYPE | BUILD BIHLER 2024 THERATYPING DATASET")
    print("=" * 85)

    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    # Row 4 contains column headers; the first observation is Excel row 5.
    df = pd.read_excel(SOURCE, sheet_name="Table S1", header=3)
    df.columns = [str(c).strip() for c in df.columns]

    required = [
        "Variant", "HGVS cDNA", "HGVS protein", "Variant type",
        "Treatment", FUNCTION, FUNCTION_IVA,
    ]
    missing = [column for column in required if column not in df]

    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["source_excel_row"] = df.index + 5

    # Exclude non-data footer or separator rows, but never alter measurements.
    df = df.loc[
        df["Variant"].notna() & df["Treatment"].notna()
    ].copy()

    df["treatment_norm"] = df["Treatment"].map(normalize_treatment)

    print("\n[1] ORIGINAL CONDITIONS")
    print(f"Source records: {len(df)}")
    print(df["Treatment"].value_counts(dropna=False).to_string())

    unexpected = sorted(
        set(df["treatment_norm"]) - {"DMSO", "ELX/TEZ"}
    )
    if unexpected:
        raise ValueError(
            "Unrecognised treatment labels. Review these before pairing: "
            f"{unexpected}"
        )

    if set(df["treatment_norm"]) != {"DMSO", "ELX/TEZ"}:
        raise ValueError(
            "Both DMSO and ELX/TEZ must be available to construct ETI labels."
        )

    # DNA identifiers prevent accidental merging of different nucleotide
    # substitutions encoding the same amino-acid change.
    cdna = df["HGVS cDNA"].astype("string").str.strip()
    valid_cdna = cdna.notna() & cdna.ne("")

    df["variant_key"] = cdna.where(
        valid_cdna,
        "legacy:" + df["Variant"].astype("string").str.strip(),
    )

    for column in (FUNCTION, FUNCTION_IVA):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    print("\n[2] VARIANT / TREATMENT PAIRING")

    counts = (
        df.groupby(["variant_key", "treatment_norm"])
        .size()
        .unstack(fill_value=0)
    )

    # Avoid choosing arbitrary rows or averaging duplicate records.
    valid_keys = counts.index[
        (counts["DMSO"] == 1) & (counts["ELX/TEZ"] == 1)
    ]

    excluded = counts.drop(index=valid_keys)

    print(f"Distinct variant identifiers: {len(counts)}")
    print(f"Unambiguous treatment pairs:  {len(valid_keys)}")
    print(f"Identifiers needing review:   {len(excluded)}")

    if len(excluded):
        print("\nExamples requiring review:")
        print(excluded.head(12).to_string())

    if len(valid_keys) == 0:
        raise ValueError("No unambiguous treatment pairs found.")

    dms = (
        df.loc[
            (df["variant_key"].isin(valid_keys))
            & (df["treatment_norm"] == "DMSO")
        ]
        .set_index("variant_key")
        .reindex(valid_keys)
    )

    eti = (
        df.loc[
            (df["variant_key"].isin(valid_keys))
            & (df["treatment_norm"] == "ELX/TEZ")
        ]
        .set_index("variant_key")
        .reindex(valid_keys)
    )

    paired = pd.DataFrame(index=valid_keys)
    paired.index.name = "variant_key"

    paired["variant"] = dms["Variant"]
    paired["hgvs_cdna"] = dms["HGVS cDNA"]
    paired["hgvs_protein"] = dms["HGVS protein"]
    paired["variant_type"] = dms["Variant type"]

    paired["source_row_dmso"] = dms["source_excel_row"]
    paired["source_row_elx_tez"] = eti["source_excel_row"]

    # Functional measurements are expressed in % of WT CFTR function.
    paired["baseline_fsk_pct_wt"] = dms[FUNCTION]
    paired["iva_treated_pct_wt"] = dms[FUNCTION_IVA]
    paired["elx_tez_fsk_pct_wt"] = eti[FUNCTION]
    paired["eti_treated_pct_wt"] = eti[FUNCTION_IVA]

    # Differences are percentage points of WT function, not fold changes.
    paired["iva_rescue_pp"] = (
        paired["iva_treated_pct_wt"]
        - paired["baseline_fsk_pct_wt"]
    )

    paired["corrector_rescue_pp"] = (
        paired["elx_tez_fsk_pct_wt"]
        - paired["baseline_fsk_pct_wt"]
    )

    paired["eti_rescue_pp"] = (
        paired["eti_treated_pct_wt"]
        - paired["baseline_fsk_pct_wt"]
    )

    paired["eti_increment_over_iva_pp"] = (
        paired["eti_treated_pct_wt"]
        - paired["iva_treated_pct_wt"]
    )

    # Keep the original reported measurement uncertainties separate.
    sem_columns = {
        "FSK (%WT) SEM": "baseline_fsk_sem",
        "FSK+IVA (%WT) SEM": "iva_treated_sem",
    }

    for original, destination in sem_columns.items():
        if original in df.columns:
            paired[destination] = dms[original]
            paired[destination + "_eti_condition"] = eti[original]

    paired = paired.reset_index()

    target_columns = [
        "baseline_fsk_pct_wt",
        "iva_treated_pct_wt",
        "elx_tez_fsk_pct_wt",
        "eti_treated_pct_wt",
    ]

    incomplete = paired[target_columns].isna().any(axis=1)

    print("\n[3] FUNCTIONAL LABEL COVERAGE")
    print(f"Paired variants:             {len(paired)}")
    print(f"Missing functional values:  {incomplete.sum()}")

    # Preserve all conditions in a lossless, separate dataset.
    # Incomplete pairs remain here for future investigation.
    model_data = paired.loc[~incomplete].copy()

    print(f"Usable paired observations: {len(model_data)}")

    if model_data.empty:
        raise ValueError("No complete functional-response labels.")

    print("\n[4] PRIMARY LABEL DISTRIBUTIONS")

    for column in [
        "baseline_fsk_pct_wt",
        "iva_rescue_pp",
        "eti_treated_pct_wt",
        "eti_rescue_pp",
        "eti_increment_over_iva_pp",
    ]:
        x = model_data[column]
        print(
            f"{column:30s} "
            f"median={x.median():9.3f} "
            f"p05={x.quantile(0.05):9.3f} "
            f"p95={x.quantile(0.95):9.3f}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)

    conditions_path = OUTPUT / "bihler_2024_conditions.parquet"
    pairs_path = OUTPUT / "bihler_2024_theratyping.parquet"
    review_path = OUTPUT / "bihler_2024_unresolved_pairs.csv"

    df.to_parquet(conditions_path, index=False)
    model_data.to_parquet(pairs_path, index=False)

    excluded.reset_index().to_csv(review_path, index=False)

    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest["bihler_2024_build"] = {
        "source_sha256": sha256(SOURCE),
        "n_source_records": len(df),
        "n_unambiguous_pairs": len(valid_keys),
        "n_complete_pairs": len(model_data),
        "n_unresolved_identifiers": len(excluded),
        "label_unit": "percentage points of wild-type CFTR function",
        "eti_rescue_definition": (
            "ELX/TEZ-pretreated FSK+IVA %WT minus "
            "DMSO-pretreated FSK %WT"
        ),
        "warning": (
            "In vitro FRT-cell response, not clinical treatment benefit. "
            "Unresolved duplicate conditions excluded without aggregation."
        ),
    }

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n[5] OUTPUTS")
    print(f"Conditions: {conditions_path.relative_to(ROOT)}")
    print(f"Paired data: {pairs_path.relative_to(ROOT)}")
    print(f"Review:     {review_path.relative_to(ROOT)}")
    print(f"Manifest:   {MANIFEST.relative_to(ROOT)}")
    print("\nBUILD COMPLETE — NO MODEL TRAINING PERFORMED")
    print("=" * 85)


if __name__ == "__main__":
    build()
