"""Dataset inspection and provenance tracking for CFTR-Theratype."""

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from Bio import SeqIO


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
MANIFEST = ROOT / "data" / "manifests" / "data_sources.json"
FASTA = RAW / "uniprot" / "P13569.fasta"

EXTENSIONS = {".xlsx", ".xls", ".xlsm"}
VARIANT_PATTERN = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def checksum(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_mcdonald(workbook, sequence):
    """Validate known McDonald columns without modifying source measurements."""

    df = pd.read_excel(workbook, sheet_name="Supp Table 3")
    counts = df["variant"].value_counts()
    duplicated = counts[counts > 1]

    matches, mismatches, invalid = 0, [], []

    for variant in df["variant"].astype(str):
        parsed = VARIANT_PATTERN.fullmatch(variant.strip())

        if not parsed:
            invalid.append(variant)
            continue

        wt, position, _ = parsed.groups()
        position = int(position)

        if 1 <= position <= len(sequence) and sequence[position - 1] == wt:
            matches += 1
        else:
            mismatches.append(variant)

    print("\n  MCDONALD QC")
    print(f"    Measurements:          {len(df)}")
    print(f"    Unique variants:       {df['variant'].nunique()}")
    print(f"    Reference AA matches:  {matches}/{len(df)}")
    print(f"    Reference mismatches:  {len(mismatches)}")
    print(f"    Unparseable variants:  {len(invalid)}")

    print("\n    Treatments:")
    for name, count in df["Treatment"].value_counts(dropna=False).items():
        print(f"      {name}: {count}")

    print("\n    Repeated variants:")
    if duplicated.empty:
        print("      None")
    else:
        for variant, count in duplicated.items():
            subset = df.loc[df["variant"] == variant]
            unique_measures = subset[
                ["FSK, %WT", "FSK + IVA, %WT"]
            ].drop_duplicates().shape[0]
            print(
                f"      {variant}: {count} rows, "
                f"{unique_measures} distinct functional measurement pairs"
            )

    if mismatches or invalid:
        print("    WARNING: Variant identifiers require review.")
        print("    Examples:", (mismatches + invalid)[:10])

    print("\n    SEM availability:")
    for col in df.columns:
        if "SEM" in col:
            print(f"      {col}: {df[col].notna().sum()}/{len(df)}")

    return {
        "rows": len(df),
        "unique_variants": int(df["variant"].nunique()),
        "reference_matches": matches,
        "reference_mismatches": mismatches,
        "unparseable_variants": invalid,
        "repeated_variants": {
            str(k): int(v) for k, v in duplicated.items()
        },
    }


def preview_sheet(frame):
    """Print a compact raw preview, preserving original header positions."""

    for row_idx in range(min(5, len(frame))):
        cells = []

        for value in frame.iloc[row_idx, :12]:
            if pd.notna(value) and str(value).strip():
                value = str(value).replace("\n", " ").strip()
                cells.append(value[:50])

        if cells:
            print(f"      Raw row {row_idx + 1}: {' | '.join(cells)}")


def inspect():
    """Inspect all public-source Excel workbooks with one reproducible command."""

    if not FASTA.exists():
        raise FileNotFoundError(f"Missing reference sequence: {FASTA}")

    sequence = str(SeqIO.read(FASTA, "fasta").seq)

    workbooks = sorted(
        path for path in RAW.rglob("*")
        if path.is_file() and path.suffix.lower() in EXTENSIONS
    )

    print("=" * 85)
    print("CFTR-THERATYPE | UNIFIED DATA INSPECTION")
    print("=" * 85)
    print(f"Reference: UniProt P13569 | {len(sequence)} residues")
    print(f"Excel workbooks discovered: {len(workbooks)}")

    if not workbooks:
        raise FileNotFoundError(f"No Excel workbooks found under {RAW}")

    manifest = {
        "reference": "UniProt P13569",
        "reference_length": len(sequence),
        "reference_sha256": checksum(FASTA),
        "workbooks": [],
    }

    for path in workbooks:
        relative_path = str(path.relative_to(ROOT))

        print("\n" + "=" * 85)
        print(f"SOURCE: {relative_path}")
        print(f"Size: {path.stat().st_size:,} bytes")
        print(f"SHA256: {checksum(path)}")

        record = {
            "path": relative_path,
            "sha256": checksum(path),
            "sheets": [],
        }

        try:
            with pd.ExcelFile(path) as excel:
                print(f"Sheets: {excel.sheet_names}")

                for sheet_name in excel.sheet_names:
                    # header=None is intentional: many scientific
                    # supplements use multi-row or nonstandard headers.
                    frame = pd.read_excel(
                        excel, sheet_name=sheet_name, header=None
                    )

                    print(f"\n  SHEET: {sheet_name}")
                    print(
                        f"    Raw dimensions: "
                        f"{frame.shape[0]} rows x {frame.shape[1]} columns"
                    )
                    preview_sheet(frame)

                    record["sheets"].append({
                        "name": sheet_name,
                        "raw_rows": len(frame),
                        "raw_columns": len(frame.columns),
                    })

                if path.name == "table_s3.xlsx" and "mcdonald_2024" in path.parts:
                    record["qc"] = inspect_mcdonald(excel, sequence)

        except Exception as exc:
            print(f"ERROR: Could not inspect workbook: {exc}")
            record["error"] = str(exc)

        manifest["workbooks"].append(record)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    print("\n" + "=" * 85)
    print("DATA INSPECTION COMPLETE")
    print(f"Manifest: {MANIFEST.relative_to(ROOT)}")
    print("No measurements have been merged, deduplicated, or transformed.")
    print("=" * 85)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inspect", "build"])
    args = parser.parse_args()

    if args.command == "inspect":
        inspect()
    elif args.command == "build":
        from cftr_theratype.build import build
        build()


if __name__ == "__main__":
    main()
