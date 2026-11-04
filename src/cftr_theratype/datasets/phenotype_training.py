"""Historical Phase26K protected-partition-safe phenotype data loader.

The source-label datasets are filtered to TRAIN IDs *before* materialisation.
Auxiliary ETI records at VTI validation/test positions are excluded. This
module does not open VAL or TEST assay outcomes.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from cftr_theratype.features.mutation_codes import mutation_code, position_from_code


def load_allowed(root=None):
    """Load pre-filtered VTI/ETI TRAIN assays and exclude protected positions."""
    root = Path.cwd() if root is None else Path(root)
    vtisplit = pd.read_parquet(
        root / "data/processed/bihler_2026_transfer_splits.parquet",
        columns=[
            "variant_key",
            "transfer_split",
            "protein_position",
            "hgvs_protein"
        ]
    )

    oldsplit = pd.read_parquet(
        root / "data/processed/bihler_2024_missense_splits.parquet",
        columns=["variant_key", "split"]
    )

    vkeys = set(
        vtisplit.loc[
            vtisplit.transfer_split.eq("train"),
            "variant_key"
        ].astype(str)
    )

    ekeys = set(
        oldsplit.loc[
            oldsplit.split.eq("train"),
            "variant_key"
        ].astype(str)
    )

    protected = set(
        pd.to_numeric(
            vtisplit.loc[
                vtisplit.transfer_split.isin(
                    ["val", "test"]
                ),
                "protein_position"
            ],
            errors="coerce"
        ).dropna().astype(int)
    )

    assert len(vkeys) == 210
    assert len(ekeys) == 422
    assert len(protected) == 68

    # Load TRAIN assay labels only.
    v = pd.read_parquet(
        root / "data/processed/bihler_2026_vti.parquet",
        columns=[
            "variant_key",
            "hgvs_protein",
            "corrector_rescue_pp",
            "vti_rescue_pp"
        ],
        filters=[
            ("variant_key", "in", sorted(vkeys))
        ]
    )

    e = pd.read_parquet(
        root / "data/processed/bihler_2024_theratyping.parquet",
        columns=[
            "variant_key",
            "hgvs_protein",
            "corrector_rescue_pp"
        ],
        filters=[
            ("variant_key", "in", sorted(ekeys))
        ]
    )

    assert len(v) == 210
    assert set(v.variant_key.astype(str)) == vkeys
    assert len(e) == 422
    assert set(e.variant_key.astype(str)) == ekeys

    for frame in (v, e):
        frame["mutation"] = frame.hgvs_protein.map(
            mutation_code
        )
        frame["position"] = frame.mutation.map(
            position_from_code
        )

        if frame.mutation.isna().any():
            raise ValueError(
                "Unparseable training mutation; "
                "inspect protein notation before proceeding."
            )

    # Remove auxiliary ETI observations at protected
    # VTI validation/test protein positions.
    e = e.loc[
        ~e.position.isin(protected)
    ].copy()

    assert len(e) == 383

    v = v.groupby(
        "mutation", as_index=False
    ).agg(
        vti_corrector=(
            "corrector_rescue_pp", "mean"
        ),
        vti_total=("vti_rescue_pp", "mean")
    )

    e = e.groupby(
        "mutation", as_index=False
    ).agg(
        eti_corrector=(
            "corrector_rescue_pp", "mean"
        )
    )

    return v, e, protected
