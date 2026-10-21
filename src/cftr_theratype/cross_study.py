"""Cross-study membership and leakage checks; never reads ETI or VTI labels.

The original 2024 position-disjoint test and validation partitions remain sealed.
This module is a prerequisite for a defensible 2024->2026 transfer experiment,
not a cross-treatment evaluation itself.
"""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from cftr_theratype.build import ROOT

PROCESSED = ROOT / "data/processed"
MANIFEST = ROOT / "data/manifests/cross_study_protocol.json"
SPLIT_FILE = PROCESSED / "bihler_2024_missense_splits.parquet"
PAIRS_2024 = PROCESSED / "bihler_2024_theratyping.parquet"
VTI_2026 = PROCESSED / "bihler_2026_vti.parquet"
OUTPUT = PROCESSED / "bihler_2026_cross_study_membership.parquet"

MISSENSE_HGVS = re.compile(r"^p\.\(?[A-Z][a-z]{2}(\d+)[A-Z][a-z]{2}\)?$")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def map_cross_study_membership(pairs2024, split2024, vti2026):
    """Return identifier-only VTI cohort annotations without accessing outcome labels."""

    for name, frame in (
        ("2024 pairs", pairs2024),
        ("2024 splits", split2024),
        ("2026 VTI", vti2026),
    ):
        if frame["variant_key"].isna().any() or frame["variant_key"].duplicated().any():
            raise ValueError(f"{name}: missing or repeated variant_key")

    if set(split2024["split"]) != {"train", "val", "test"}:
        raise ValueError("Unexpected 2024 split labels")

    old_keys = set(pairs2024["variant_key"])
    if not set(split2024["variant_key"]).issubset(old_keys):
        raise ValueError("Split file refers to variants absent from 2024 source")

    position_splits = split2024.groupby("protein_position")["split"].nunique()
    if (position_splits > 1).any():
        raise ValueError("2024 positions are shared across frozen partitions")

    split_by_position = (
        split2024.drop_duplicates("protein_position")
        .set_index("protein_position")["split"].to_dict()
    )
    split_by_variant = split2024.set_index("variant_key")["split"].to_dict()
    protein_by_variant = pairs2024.set_index("variant_key")["hgvs_protein"].to_dict()

    # Whitelist only identity and study-membership metadata, not outcome columns.
    columns = ["variant_key", "hgvs_protein", "variant_type", "study_cohort"]
    out = vti2026[columns].copy()
    out["overlaps_2024_exact_cdna"] = out["variant_key"].isin(old_keys)
    out["frozen_2024_variant_split"] = (
        out["variant_key"].map(split_by_variant).fillna("not_in_2024_split")
    )
    out["2024_hgvs_protein"] = out["variant_key"].map(protein_by_variant)

    comparable = out["overlaps_2024_exact_cdna"]
    agreement = (
        out.loc[comparable, "hgvs_protein"].fillna("<missing>").astype(str)
        == out.loc[comparable, "2024_hgvs_protein"].fillna("<missing>").astype(str)
    )
    if not agreement.all():
        failed = out.loc[comparable].loc[~agreement, "variant_key"].tolist()
        raise ValueError(f"Protein annotation disagrees for overlap: {failed[:10]}")

    def position_and_status(row):
        if str(row.variant_type).strip().lower() != "missense":
            return pd.NA, "non_missense"
        match = MISSENSE_HGVS.fullmatch(str(row.hgvs_protein).strip())
        if not match:
            return pd.NA, "unparseable_missense"
        position = int(match.group(1))
        if not 1 <= position <= 1480:
            return pd.NA, "invalid_position"
        return position, "missense_position_parsed"

    parsed = [position_and_status(row) for row in out.itertuples(index=False)]
    out["protein_position"] = pd.array([p for p, _ in parsed], dtype="Int64")
    out["position_parse_status"] = [s for _, s in parsed]
    out["frozen_2024_position_split"] = (
        out["protein_position"].map(split_by_position).fillna("unseen_2024_position")
    )

    out["protected_2024_validation_or_test_position"] = (
        out["frozen_2024_position_split"].isin(["val", "test"])
    )

    # Informational only: an additional VTI-specific grouped split is still
    # necessary before using these variants for VTI model training.
    out["candidate_for_transfer_development"] = (
        out["position_parse_status"].eq("missense_position_parsed")
        & ~out["protected_2024_validation_or_test_position"]
    )

    return out


def run_cross_study():
    print("=" * 82)
    print("CFTR-THERATYPE | CROSS-STUDY MEMBERSHIP / LEAKAGE PROTOCOL")
    print("=" * 82)

    required = (PAIRS_2024, SPLIT_FILE, VTI_2026)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    # Deliberately SELECT COLUMNS to make loading outcome values impossible.
    old = pd.read_parquet(
        PAIRS_2024, columns=["variant_key", "hgvs_protein"]
    )
    split = pd.read_parquet(
        SPLIT_FILE, columns=["variant_key", "protein_position", "split"]
    )
    vti = pd.read_parquet(
        VTI_2026,
        columns=["variant_key", "hgvs_protein", "variant_type", "study_cohort"],
    )
    out = map_cross_study_membership(old, split, vti)

    print("\n[1] EXACT CDNA OVERLAP")
    print(f"2024 paired identifiers:     {len(old)}")
    print(f"2024 missense split records: {len(split)}")
    print(f"2026 paired VTI records:    {len(vti)}")
    print(f"Exact 2024/2026 overlap:    {out.overlaps_2024_exact_cdna.sum()}")
    print(f"2026-only cDNA identifiers: {(~out.overlaps_2024_exact_cdna).sum()}")

    print("\n[2] COHORT VS EXACT OVERLAP")
    print(pd.crosstab(
        out["study_cohort"].fillna("<missing>"),
        out["overlaps_2024_exact_cdna"],
        rownames=["2026 study cohort"],
        colnames=["Has 2024 cDNA"],
    ).to_string())

    print("\n[3] 2026 VARIANT IDENTIFIERS")
    print(out["position_parse_status"].value_counts(dropna=False).to_string())

    print("\n[4] FROZEN 2024 SPLIT MEMBERSHIP (IDENTIFIERS ONLY)")
    print(out["frozen_2024_variant_split"].value_counts().to_string())
    print("\n2026 missense positions relative to 2024 frozen position groups:")
    print(out["frozen_2024_position_split"].value_counts().to_string())
    print(f"Protected val/test positions: {out.protected_2024_validation_or_test_position.sum()} variants")
    print(f"Candidate transfer-development variants: {out.candidate_for_transfer_development.sum()}")
    print("NOTE: candidates are NOT an approved train split; VTI requires its own group split.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    out.drop(columns="2024_hgvs_protein").to_parquet(OUTPUT, index=False)

    summary = {
        "purpose": "Cross-study identifier mapping and frozen 2024 test protection",
        "2024_pairs_sha256": sha256(PAIRS_2024),
        "2024_split_sha256": sha256(SPLIT_FILE),
        "2026_vti_sha256": sha256(VTI_2026),
        "n_vti_pairs": int(len(out)),
        "n_exact_overlaps": int(out.overlaps_2024_exact_cdna.sum()),
        "n_protected_by_2024_position": int(out.protected_2024_validation_or_test_position.sum()),
        "n_transfer_development_candidates_before_vti_split": int(
            out.candidate_for_transfer_development.sum()
        ),
        "2024_eti_outcomes_loaded": False,
        "2026_vti_outcomes_loaded": False,
        "note": "Position parsing is not 2026 WT sequence validation. Do not assume VTI and ETI response levels are batch-equivalent.",
    }
    MANIFEST.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n[5] OUTPUTS")
    print(f"Identifier-only mapping: {OUTPUT.relative_to(ROOT)}")
    print(f"Protocol: {MANIFEST.relative_to(ROOT)}")
    print("2024 ETI and 2026 VTI outcomes: NEVER READ")
    print("2024 validation/test partitions: UNCHANGED")
    print("=" * 82)


if __name__ == "__main__":
    run_cross_study()
