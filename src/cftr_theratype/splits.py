"""Reproducible, position-disjoint CFTR variant splits."""

import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from cftr_theratype.reference import (
    bihler_2024_assay_sequence,
    write_bihler_2024_assay_fasta,
)
from Bio.SeqUtils import seq1
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/processed/bihler_2024_theratyping.parquet"
OUTPUT = ROOT / "data/processed/bihler_2024_missense_splits.parquet"
PROTOCOL = ROOT / "data/manifests/split_protocol.json"

SEED = 42

HGVS_PATTERN = re.compile(
    r"^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$"
)
SHORT_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def parse_substitution(row, sequence):
    """Return (position, wild_type, alternate, status)."""

    if str(row["variant_type"]).strip().lower() != "missense":
        return None, None, None, "non_missense"

    protein = str(row["hgvs_protein"]).strip()
    parsed = HGVS_PATTERN.fullmatch(protein)

    if parsed:
        wt3, position, alt3 = parsed.groups()
        wt = seq1(wt3)
        alt = seq1(alt3)
    else:
        parsed = SHORT_PATTERN.fullmatch(str(row["variant"]).strip())

        if not parsed:
            return None, None, None, "unparseable_missense"

        wt, position, alt = parsed.groups()

    position = int(position)

    if (
        wt not in AMINO_ACIDS
        or alt not in AMINO_ACIDS
        or wt == alt
        or not 1 <= position <= len(sequence)
        or sequence[position - 1] != wt
    ):
        return position, wt, alt, "reference_mismatch"

    return position, wt, alt, "valid"


def build_splits():
    print("=" * 85)
    print("CFTR-THERATYPE | POSITION-HELD-OUT VALIDATION SPLITS")
    print("=" * 85)

    df = pd.read_parquet(SOURCE)
    sequence = bihler_2024_assay_sequence()
    assay_fasta = write_bihler_2024_assay_fasta()
    print(f"Experimental reference: {assay_fasta.relative_to(ROOT)}")
    print("Protein background: M470 (Bihler 2024)")

    parsed = [
        parse_substitution(row, sequence)
        for _, row in df.iterrows()
    ]

    metadata = pd.DataFrame(
        parsed,
        columns=[
            "protein_position",
            "reference_aa",
            "alternate_aa",
            "parse_status",
        ],
        index=df.index,
    )

    print("\n[1] VARIANT ELIGIBILITY")
    print(f"Available paired observations: {len(df)}")
    print(metadata["parse_status"].value_counts().to_string())

    if (metadata["parse_status"] == "reference_mismatch").any():
        bad = df.loc[
            metadata["parse_status"] == "reference_mismatch",
            ["variant_key", "variant", "hgvs_protein"],
        ]
        raise ValueError(
            "Missense substitutions disagree with UniProt P13569:\n"
            + bad.to_string(index=False)
        )

    eligible = metadata["parse_status"] == "valid"
    model_df = pd.concat(
        [df.loc[eligible], metadata.loc[eligible]],
        axis=1,
    ).reset_index(drop=True)

    if len(model_df) < 50:
        raise ValueError("Insufficient valid missense variants.")

    if model_df["variant_key"].duplicated().any():
        raise ValueError("Duplicate variant keys in modeling data.")

    model_df["protein_position"] = (
        model_df["protein_position"].astype(int)
    )

    print("\n[2] POSITION-DISJOINT PARTITIONS")

    outer = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )

    dev_idx, test_idx = next(
        outer.split(
            model_df,
            groups=model_df["protein_position"],
        )
    )

    dev_df = model_df.iloc[dev_idx]

    inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.125,
        random_state=SEED + 1,
    )

    train_rel, val_rel = next(
        inner.split(
            dev_df,
            groups=dev_df["protein_position"],
        )
    )

    train_idx = dev_df.iloc[train_rel].index
    val_idx = dev_df.iloc[val_rel].index

    model_df["split"] = "unassigned"
    model_df.loc[train_idx, "split"] = "train"
    model_df.loc[val_idx, "split"] = "val"
    model_df.loc[test_idx, "split"] = "test"

    groups = {
        name: set(
            model_df.loc[
                model_df["split"] == name,
                "protein_position",
            ]
        )
        for name in ("train", "val", "test")
    }

    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
    assert (model_df["split"] != "unassigned").all()

    for name in ("train", "val", "test"):
        subset = model_df[model_df["split"] == name]
        print(
            f"{name.upper():6s} | "
            f"{len(subset):4d} variants | "
            f"{len(groups[name]):4d} distinct positions"
        )

    print("\n[3] DEVELOPMENT LABEL DISTRIBUTIONS")
    print("The test labels are intentionally not summarised.")

    for name in ("train", "val"):
        values = model_df.loc[
            model_df["split"] == name,
            "eti_rescue_pp",
        ]
        print(
            f"{name.upper():6s} | "
            f"median={values.median():8.3f} | "
            f"p05={values.quantile(0.05):8.3f} | "
            f"p95={values.quantile(0.95):8.3f}"
        )

    split_cols = [
        "variant_key",
        "variant",
        "hgvs_protein",
        "protein_position",
        "reference_aa",
        "alternate_aa",
        "split",
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    model_df[split_cols].to_parquet(OUTPUT, index=False)

    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()

    protocol = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": digest,
        "seed": SEED,
        "strategy": "GroupShuffleSplit by canonical CFTR position",
        "target": "eti_rescue_pp",
        "protein_reference": "Bihler 2024 experimental WT, M470",
        "reference_relative_to_uniprot": "P13569:p.Val470Met",
        "assay_reference_fasta": str(assay_fasta.relative_to(ROOT)),
        "outer_test_position_fraction": 0.20,
        "inner_validation_fraction_of_remaining_positions": 0.125,
        "eligible_variant_type": "missense",
        "n_eligible_variants": len(model_df),
        "n_distinct_positions": int(
            model_df["protein_position"].nunique()
        ),
        "counts": {
            name: {
                "variants": int((model_df["split"] == name).sum()),
                "positions": len(groups[name]),
            }
            for name in ("train", "val", "test")
        },
    }

    PROTOCOL.parent.mkdir(parents=True, exist_ok=True)
    PROTOCOL.write_text(json.dumps(protocol, indent=2) + "\n")

    print("\n[4] SAVED OUTPUTS")
    print(f"Splits:   {OUTPUT.relative_to(ROOT)}")
    print(f"Protocol: {PROTOCOL.relative_to(ROOT)}")
    print("\nPASS: No protein-position leakage across partitions.")
    print("SPLITS COMPLETE")
    print("=" * 85)


if __name__ == "__main__":
    build_splits()
