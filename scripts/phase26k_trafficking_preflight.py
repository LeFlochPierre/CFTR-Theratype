#!/usr/bin/env python3
"""Phase 26K: external CFTR trafficking phenotype audit.

TRAIN ONLY. No protected ETI/VTI outcomes loaded.
Exploratory phenotype associations, not a model benchmark.
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.cwd()
BOOK = ROOT / "data/external/tedman2025/cftr_expression.xlsx"
OUT = ROOT / "results/biology/phase26k"

from cftr_theratype.features.mutation_codes import (
    AA, ALLOWED, PATTERN,
    aa_symbol as _canonical_aa_symbol,
    mutation_code as _canonical_mutation_code,
    position_from_code as _canonical_position_from_code,
)


def aa_symbol(token):
    """Legacy API: canonical amino acid abbreviation parser."""
    return _canonical_aa_symbol(token)


def mutation_code(value):
    """Legacy API: canonical free-text CFTR substitution parser."""
    return _canonical_mutation_code(value)


def position_from_code(code):
    """Legacy API: canonical substitution position parser."""
    return _canonical_position_from_code(code)


def load_allowed():
    """Legacy API; original TRAIN filtering is shared with Phase26K-2."""
    from cftr_theratype.datasets.phenotype_training import load_allowed as shared
    return shared(ROOT)


def main():
    if not BOOK.exists():
        raise FileNotFoundError(
            f"Missing Tedman workbook: {BOOK}"
        )

    v, e, protected = load_allowed()

    print("=" * 100)
    print(
        "PHASE 26K | EXTERNAL CFTR TRAFFICKING "
        "AND CORRECTOR RESPONSE | TRAIN ONLY"
    )
    print("=" * 100)

    print(
        "VTI TRAIN: 210 variants | "
        "ETI eligible TRAIN: 383 variants"
    )

    print(
        "Matching uses exact protein substitution, "
        "not merely protein position."
    )

    print(
        "Protected validation/test outcomes unopened."
    )

    xls = pd.ExcelFile(BOOK)

    print("\nWorkbook sheets:", xls.sheet_names)

    matches = []
    summary = []

    for sheet in xls.sheet_names:
        d = pd.read_excel(
            xls, sheet_name=sheet
        )

        print(
            f"\nSHEET {sheet!r}: "
            f"rows={len(d)}, columns={len(d.columns)}"
        )

        print(
            "Columns:",
            [str(c) for c in d.columns]
        )

        candidates = []

        # Identify protein substitution columns
        # without assuming a particular Excel header.
        for col in d.columns:
            if d[col].dtype.kind in "biufc":
                continue

            keys = d[col].map(mutation_code)
            parsed = int(keys.notna().sum())

            if parsed >= 5:
                hits = len(
                    set(keys.dropna())
                    & (
                        set(v.mutation)
                        | set(e.mutation)
                    )
                )

                candidates.append(
                    (
                        hits,
                        parsed,
                        str(col),
                        col,
                        keys
                    )
                )

        if not candidates:
            print(
                "WARNING: No automatically recognised "
                "mutation identity column."
            )

            print(
                d.head(3).to_string(
                    index=False,
                    max_cols=12
                )
            )
            continue

        candidates.sort(
            key=lambda x: (x[0], x[1]),
            reverse=True
        )

        print(
            "\nMutation identifier candidates:",
            [
                (q[2], q[1], q[0])
                for q in candidates[:3]
            ]
        )

        hits, parsed, label, idcol, keys = (
            candidates[0]
        )

        print(
            f"Selected mutation column: {label!r}"
        )

        print(
            f"Parsed rows: {parsed} | "
            f"TRAIN matches: {hits}"
        )

        ext = d.copy()
        ext["mutation"] = keys

        ext = ext.loc[
            ext.mutation.notna()
        ].copy()

        ext["position"] = ext.mutation.map(
            position_from_code
        )

        before = len(ext)

        ext = ext.loc[
            ~ext.position.isin(protected)
        ].copy()

        print(
            "External rows removed at protected "
            "VTI positions:",
            before - len(ext)
        )

        print(
            f"Eligible external mutations: "
            f"{ext.mutation.nunique()} | "
            f"positions: {ext.position.nunique()}"
        )

        if ext.empty:
            continue

        print(
            "Exact overlap with VTI TRAIN:",
            len(
                set(ext.mutation)
                & set(v.mutation)
            )
        )

        print(
            "Exact overlap with ETI TRAIN:",
            len(
                set(ext.mutation)
                & set(e.mutation)
            )
        )

        print(
            "Novel substitutions relative to VTI TRAIN:",
            len(
                set(ext.mutation)
                - set(v.mutation)
            )
        )

        print(
            "Example parsed mutations:",
            ext.mutation.head(7).tolist()
        )

        numeric_candidates = []

        for col in d.columns:
            if col == idcol:
                continue

            if str(col).lower() in (
                "index", "id", "position",
                "protein_position"
            ):
                continue

            numeric = pd.to_numeric(
                ext[col],
                errors="coerce"
            )

            if (
                numeric.notna().sum()
                >= max(10, int(.2 * len(ext)))
                and numeric.nunique(dropna=True) >= 5
            ):
                numeric_candidates.append(
                    (col, numeric)
                )

        print(
            "Numeric phenotype candidate columns:",
            len(numeric_candidates)
        )

        for col, values in numeric_candidates:
            q = ext[
                ["mutation", "position"]
            ].copy()

            q["surface_measurement"] = (
                values.to_numpy(float)
            )

            q = (
                q.dropna(
                    subset=["surface_measurement"]
                )
                .groupby(
                    "mutation", as_index=False
                )
                .surface_measurement.mean()
            )

            for target, labels, target_col in [
                (
                    "VTI corrector",
                    v,
                    "vti_corrector"
                ),
                (
                    "ETI corrector",
                    e,
                    "eti_corrector"
                )
            ]:
                pair = q.merge(
                    labels[
                        ["mutation", target_col]
                    ],
                    on="mutation",
                    validate="one_to_one"
                )

                pair = pair.loc[
                    np.isfinite(
                        pair.surface_measurement
                    )
                    & np.isfinite(
                        pair[target_col]
                    )
                ]

                rho = (
                    float(
                        spearmanr(
                            pair.surface_measurement,
                            pair[target_col]
                        ).statistic
                    )
                    if len(pair) >= 5
                    else np.nan
                )

                summary.append({
                    "sheet": sheet,
                    "column": str(col),
                    "target": target,
                    "n_exact_mutations": len(pair),
                    "spearman_exploratory": rho,
                    "numeric_external_substitutions": len(q)
                })

                if len(pair):
                    tmp = pair.assign(
                        sheet=sheet,
                        phenotype_column=str(col),
                        target=target
                    )

                    matches.append(
                        tmp.rename(
                            columns={
                                target_col:
                                    "corrector_response"
                            }
                        )
                    )

        seen = pd.DataFrame([
            x for x in summary
            if x["sheet"] == sheet
        ])

        if len(seen):
            print(
                "\nExploratory associations "
                "(exact mutation matches only):"
            )

            print(
                seen[
                    [
                        "column",
                        "target",
                        "n_exact_mutations",
                        "spearman_exploratory"
                    ]
                ].round(3).to_string(
                    index=False,
                    max_rows=24
                )
            )

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    if summary:
        pd.DataFrame(summary).to_csv(
            OUT / "phase26k_phenotype_relevance.csv",
            index=False
        )

    if matches:
        pd.concat(
            matches,
            ignore_index=True
        ).to_csv(
            OUT / "phase26k_training_matches.csv",
            index=False
        )

    print(
        "\nPASS: External phenotype feasibility audit complete."
    )

    print(
        "No protected outcomes or model fitting."
    )

    print(
        "Small-sample correlations and multiple "
        "phenotype comparisons are exploratory only."
    )

    print("Saved:", OUT)


if __name__ == "__main__":
    main()
