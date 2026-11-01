#!/usr/bin/env python3
"""Phase 27A-1: VX-121 DMS source and phenotype audit.

TRAIN-only identity comparisons.
No protected VTI outcomes, no model fitting.
No comparison of raw fluorescence across instruments.
"""

from pathlib import Path
from io import BytesIO
import argparse
import hashlib
import json
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

from phase26k_trafficking_preflight import (
    load_allowed,
    mutation_code,
    position_from_code,
)

from phase26k2_numeric_audit import parse_value

SRC = ROOT / "data/external/jhangiani2026_vx121"
OUT = ROOT / "results/biology/phase27a"

EXT = {".xlsx", ".xls", ".csv", ".tsv"}


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(chunk)

    return h.hexdigest()


def source_payloads(source):
    """Discover spreadsheets, including inside ZIP archives."""

    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue

        if path.name.startswith("~$"):
            continue

        if path.suffix.lower() in EXT:
            yield (
                str(path.relative_to(source)),
                path.read_bytes(),
                sha256(path)
            )

        elif path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(path) as z:
                    for entry in z.infolist():
                        if entry.is_dir():
                            continue

                        if (
                            Path(entry.filename).suffix.lower()
                            not in EXT
                        ):
                            continue

                        if entry.file_size > 100 * 1024 * 1024:
                            continue

                        yield (
                            str(path.relative_to(source))
                            + "::"
                            + entry.filename,
                            z.read(entry),
                            sha256(path)
                        )

            except (OSError, zipfile.BadZipFile) as ex:
                print(
                    "WARN unreadable archive:",
                    path, ex
                )


def read_tables(filename, payload):
    """Try initial header positions to find experimental tables."""

    suffix = Path(filename).suffix.lower()

    try:
        if suffix in (".xlsx", ".xls"):
            xls = pd.ExcelFile(BytesIO(payload))

            for sheet in xls.sheet_names:
                for header in range(4):
                    try:
                        d = pd.read_excel(
                            xls,
                            sheet_name=sheet,
                            header=header
                        )

                        yield str(sheet), header, d

                    except (ValueError, TypeError):
                        continue

        elif suffix in (".csv", ".tsv"):
            sep = (
                "\t" if suffix == ".tsv"
                else None
            )

            for header in range(4):
                try:
                    d = pd.read_csv(
                        BytesIO(payload),
                        header=header,
                        sep=sep,
                        engine="python",
                        encoding="utf-8-sig",
                        on_bad_lines="skip"
                    )

                    yield "text", header, d

                except (
                    ValueError,
                    UnicodeDecodeError,
                    pd.errors.ParserError
                ):
                    continue

    except Exception as ex:
        print(
            f"WARN cannot parse {filename}: {ex}"
        )


def extract_candidate(d, known):
    """Identify plausible protein-substitution column."""

    if len(d) < 5:
        return None

    candidates = []

    for col in d.columns:
        if str(col).startswith("Unnamed:"):
            continue

        if pd.api.types.is_numeric_dtype(d[col]):
            continue

        if isinstance(d[col], pd.DataFrame):
            continue

        parsed = d[col].map(mutation_code)
        valid = parsed.dropna()

        if valid.nunique() < 5:
            continue

        score = (
            len(set(valid) & known),
            valid.nunique(),
            int(valid.notna().sum())
        )

        candidates.append(
            (score, str(col), parsed)
        )

    if not candidates:
        return None

    candidates.sort(
        key=lambda q: q[0],
        reverse=True
    )

    return (
        candidates[0][1],
        candidates[0][2],
        candidates[0][0],
        [
            (str(col), values.nunique())
            for _, col, values in candidates[:3]
        ]
    )


def assay_hint(col):
    """Classify labels without assuming their units."""

    s = (
        str(col)
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )

    kinds = []

    mapping = [
        ("VX121", ("vx121", "vanzacaftor")),
        ("VX445", ("vx445", "elexacaftor")),
        ("VX661", ("vx661", "tezacaftor")),
        (
            "VEHICLE",
            (
                "vehicle", "dmso", "untreated",
                "apo", "control"
            )
        ),
        (
            "FOLD_CHANGE",
            (
                "foldchange", "log2fc",
                "logfc", "log2fold"
            )
        ),
        ("SCALED", ("scaled",)),
        (
            "REPLICATE",
            ("replicate", "rep1", "rep2", "rep3")
        ),
    ]

    for label, variants in mapping:
        if any(x in s for x in variants):
            kinds.append(label)

    return (
        ",".join(kinds)
        if kinds
        else "UNCLASSIFIED"
    )


def self_test():
    d = pd.DataFrame({
        "Variant": [
            "p.Arg3Trp", "R31C", "G551D",
            "V1240G", "Y1032C", "I37R"
        ],
        "VX-121 replicate 1": [
            "12 ± 2", "15 ± 1", "NA",
            "22 ± 3", "6 ± 1", "9 ± 2"
        ],
        "Vehicle": [1, 2, 3, 4, 5, 6]
    })

    found = extract_candidate(
        d, {"R3W", "R31C"}
    )

    assert found is not None
    assert found[0] == "Variant"
    assert found[1].notna().sum() == 6

    assert (
        assay_hint("VX-121 replicate 1")
        == "VX121,REPLICATE"
    )

    assert parse_value("12 ± 2")[0] == 12

    assert position_from_code("Y1032C") == 1032

    print(
        "PASS synthetic: mutation parser, "
        "sheet identification, assay labels, "
        "numeric formats"
    )


def main():
    self_test()

    OUT.mkdir(parents=True, exist_ok=True)
    SRC.mkdir(parents=True, exist_ok=True)

    # Existing protected-partition-safe loader.
    # VTI/ETI TRAIN observations only.
    v, e, protected = load_allowed()

    vset = set(v.mutation)
    eset = set(e.mutation)

    tpath = (
        ROOT
        / "results/biology/phase26k"
        / "phase26k3_derived_surface_train.csv"
    )

    if not tpath.is_file():
        raise FileNotFoundError(
            f"Previous Tedman audit missing: {tpath}"
        )

    t = pd.read_csv(
        tpath,
        usecols=["mutation"],
        dtype=str
    )

    tset = set(t.mutation.dropna())

    known = tset | vset | eset

    print("=" * 103)
    print(
        "PHASE 27A-1 | VX-121 DMS SOURCE "
        "INVENTORY AND MUTATION OVERLAPS"
    )
    print("=" * 103)

    print(
        "Study: Jhangiani et al. 2026, "
        "PMC13252174, Supplementary Table S1"
    )

    print(
        "Library overlaps earlier Tedman work; "
        "not 232 independent new variants."
    )

    print(
        f"VTI TRAIN: {len(v)} | "
        f"ETI eligible TRAIN: {len(e)} | "
        f"Tedman eligible: {len(tset)}"
    )

    print(
        f"Protected VTI validation/test positions: "
        f"{len(protected)}"
    )

    print(
        "Protected assay outcomes unopened."
    )

    print("Input directory:", SRC)

    payloads = list(source_payloads(SRC))

    if not payloads:
        print("\nNO LOCAL EXPERIMENTAL FILES FOUND.")

        print(
            "Download Supplementary Table S1 "
            "from:"
        )

        print(
            "https://pmc.ncbi.nlm.nih.gov/"
            "articles/PMC13252174/"
        )

        print(
            "Additional data: "
            "https://data.mendeley.com/"
            "datasets/3gxywvsxns/1"
        )

        print(
            "Place Excel/CSV/TSV/ZIP files in:",
            SRC
        )

        print(
            "Then rerun. No experimental "
            "results have been inferred."
        )

        raise SystemExit(2)

    print("\nA. INPUT FILES")

    for name, blob, digest in payloads:
        print(
            f"  {name}: "
            f"{len(blob):,} bytes | "
            f"SHA256={digest[:16]}..."
        )

    inventory = []
    profiles = []
    values = []
    identities = []

    for name, blob, digest in payloads:
        # Select the most plausible mutation header
        # separately for each sheet.
        sheets = {}

        for sheet, header, d in read_tables(
            name, blob
        ):
            if d.empty:
                continue

            match = extract_candidate(
                d, known
            )

            if match is None:
                continue

            (
                keycol, mutations,
                score, alternatives
            ) = match

            rank = (
                score[0],
                score[1],
                score[2],
                -header
            )

            if (
                sheet not in sheets
                or rank > sheets[sheet][0]
            ):
                sheets[sheet] = (
                    rank, header, d,
                    keycol, mutations,
                    alternatives
                )

        if not sheets:
            print(
                f"WARN: {name} contains no "
                "recognised mutation-indexed sheet."
            )

        for sheet, info in sheets.items():
            (
                _, header, d, keycol,
                mutations, alternatives
            ) = info

            q = d.copy()

            q["__mutation"] = mutations

            q = q.loc[
                q.__mutation.notna()
            ].copy()

            q["__position"] = (
                q.__mutation.map(
                    position_from_code
                )
            )

            n_before = q.__mutation.nunique()

            n_protected = int(
                q.__position.isin(
                    protected
                ).sum()
            )

            # External measurements at held-out
            # VTI positions are not exported.
            q = q.loc[
                ~q.__position.isin(protected)
            ].copy()

            mutation_set = set(q.__mutation)

            print("\nB. SHEET")
            print(
                f"File: {name} | "
                f"Sheet: {sheet} | "
                f"Header row: {header}"
            )

            print(
                "Mutation column:",
                keycol
            )

            print(
                "Alternative candidates:",
                alternatives
            )

            print(
                f"Parsed mutations: {n_before} | "
                f"Eligible: {len(mutation_set)} | "
                f"Protected rows excluded: "
                f"{n_protected}"
            )

            print(
                f"Exact overlap: "
                f"Tedman={len(mutation_set & tset)} | "
                f"ETI TRAIN={len(mutation_set & eset)} | "
                f"VTI TRAIN={len(mutation_set & vset)}"
            )

            print(
                "New vs Tedman:",
                len(mutation_set - tset)
            )

            print(
                "Repeated mutation rows:",
                len(q) - len(mutation_set)
            )

            print(
                "Example mutations:",
                sorted(mutation_set)[:8]
            )

            inventory.append({
                "source": name,
                "sheet": str(sheet),
                "header": header,
                "rows": len(d),
                "mutation_column": keycol,
                "parsed_unique": n_before,
                "eligible_unique": len(mutation_set),
                "protected_rows_removed": n_protected,
                "overlap_tedman": len(
                    mutation_set & tset
                ),
                "overlap_eti_train": len(
                    mutation_set & eset
                ),
                "overlap_vti_train": len(
                    mutation_set & vset
                ),
                "novel_vs_tedman": len(
                    mutation_set - tset
                ),
                "sha256_parent": digest
            })

            for mutation in sorted(mutation_set):
                identities.append({
                    "source": name,
                    "sheet": str(sheet),
                    "mutation": mutation,
                    "protein_position":
                        position_from_code(mutation),
                    "in_tedman": mutation in tset,
                    "in_eti_train": mutation in eset,
                    "in_vti_train": mutation in vset
                })

            column_stats = []

            for col in d.columns:
                if col == keycol:
                    continue

                if str(col).lower() in (
                    "position",
                    "protein_position",
                    "index",
                    "id"
                ):
                    continue

                if isinstance(q[col], pd.DataFrame):
                    continue

                parsed = q[col].map(parse_value)

                numeric = np.array(
                    [item[0] for item in parsed],
                    dtype=float
                )

                finite = np.isfinite(numeric)

                if finite.sum() < 5:
                    continue

                if np.unique(
                    numeric[finite]
                ).size < 3:
                    continue

                hint = assay_hint(str(col))

                formats = (
                    pd.Series([
                        item[1] for item in parsed
                    ])
                    .value_counts()
                    .to_dict()
                )

                column_stats.append((
                    str(col),
                    int(finite.sum()),
                    hint,
                    formats
                ))

                profiles.append({
                    "source": name,
                    "sheet": str(sheet),
                    "column": str(col),
                    "assay_hint": hint,
                    "n_values": int(finite.sum()),
                    "n_unique_mutations": int(
                        q.loc[
                            finite, "__mutation"
                        ].nunique()
                    ),
                    "min": float(
                        numeric[finite].min()
                    ),
                    "median": float(
                        np.median(
                            numeric[finite]
                        )
                    ),
                    "max": float(
                        numeric[finite].max()
                    ),
                    "formats": json.dumps(formats)
                })

                for mutation, pos, value in zip(
                    q.__mutation,
                    q.__position,
                    numeric
                ):
                    if np.isfinite(value):
                        values.append({
                            "source": name,
                            "sheet": str(sheet),
                            "mutation": mutation,
                            "protein_position": int(pos),
                            "column": str(col),
                            "value": float(value)
                        })

            print(
                "Numeric candidate columns:",
                len(column_stats)
            )

            for (
                col, n, hint, formats
            ) in column_stats[:24]:
                print(
                    f"  {col[:76]} | "
                    f"n={n} | "
                    f"hint={hint} | "
                    f"formats={formats}"
                )

            if len(column_stats) > 24:
                print(
                    f"... {len(column_stats)-24} "
                    "additional columns saved."
                )

    if not inventory:
        raise RuntimeError(
            "No usable mutation-indexed sheets. "
            "Please check Supplementary Table S1."
        )

    inv = pd.DataFrame(inventory)

    coverage = (
        pd.DataFrame(identities)
        .drop_duplicates()
    )

    numeric_inventory = pd.DataFrame(profiles)
    numeric_values = pd.DataFrame(values)

    outputs = {
        "sheet_inventory": inv,
        "mutation_coverage": coverage,
        "numeric_column_inventory": numeric_inventory,
        "numeric_values_long": numeric_values
    }

    for name, data in outputs.items():
        data.to_csv(
            OUT / f"phase27a_{name}.csv",
            index=False
        )

    allmut = set(coverage.mutation)

    print("\nC. UNIQUE MUTATIONS ACROSS ALL SOURCES")

    print(
        "Eligible mutation union:",
        len(allmut)
    )

    print(
        "Independent protein positions:",
        len({
            position_from_code(x)
            for x in allmut
        })
    )

    print(
        "Overlap Tedman:",
        len(allmut & tset)
    )

    print(
        "Novel relative to Tedman:",
        len(allmut - tset)
    )

    print(
        "Overlap ETI TRAIN:",
        len(allmut & eset)
    )

    print(
        "Overlap VTI TRAIN:",
        len(allmut & vset)
    )

    print("\nD. ASSAY INTERPRETATION RULES")

    print(
        "Do not compare VX-121 and VX-445 "
        "raw fluorescence directly."
    )

    print(
        "Check vehicle normalisation and "
        "replicate definitions first."
    )

    print(
        "Surface expression is not equal "
        "to functional VTI rescue."
    )

    print(
        "Protected positions excluded "
        "from exported measurements."
    )

    print("\nPASS: Phase27A-1 source audit complete.")
    print("No model fitting or protected outcome access.")
    print("Saved:", OUT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true"
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        main()
