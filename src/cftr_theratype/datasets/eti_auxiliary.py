"""Canonical train-only ETI auxiliary data with global protected-position exclusion.

Legacy Phase26H parity is required before migrating callers.
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd



def parsed_position(s):
    s = str(s)
    m = re.search(
        r'(?:p\.)?\(?[A-Za-z]{3}(\d{1,4})(?:[A-Za-z]{3}|Ter|\*|=)',
        s
    )
    if not m:
        m = re.search(
            r'(?:p\.)?\(?[A-Za-z](\d{1,4})[A-Za-z*]',
            s
        )
    return int(m.group(1)) if m else None

def load_data(root=None):
    from cftr_theratype.datasets.frozen_vti import load_train
    ROOT = Path.cwd() if root is None else Path(root)
    from cftr_theratype.benchmark import make_features
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.splits import parse_substitution
    from cftr_theratype import structure_benchmark as structure

    vti, mats, _, folds, ref = load_train(ROOT)

    assert len(vti) == 210
    assert vti.protein_position.nunique() == 168

    vti_keys = set(vti.variant_key.astype(str))

    # Membership metadata only.
    etisplit = pd.read_parquet(
        ROOT / "data/processed/bihler_2024_missense_splits.parquet",
        columns=["variant_key", "split"]
    )

    vtisplit = pd.read_parquet(
        ROOT / "data/processed/bihler_2026_transfer_splits.parquet",
        columns=[
            "variant_key", "protein_position", "transfer_split"
        ]
    )

    etitrain_keys = set(
        etisplit.loc[
            etisplit.split.eq("train"), "variant_key"
        ].astype(str)
    )

    protected = set(
        pd.to_numeric(
            vtisplit.loc[
                vtisplit.transfer_split.isin(["val", "test"]),
                "protein_position"
            ],
            errors="coerce"
        ).dropna().astype(int)
    )

    assert len(etitrain_keys) == 422
    assert len(protected) == 68

    # Only ETI TRAIN labels are loaded.
    eti = pd.read_parquet(
        ROOT / "data/processed/bihler_2024_theratyping.parquet",
        columns=[
            "variant_key",
            "variant",
            "hgvs_cdna",
            "hgvs_protein",
            "variant_type",
            "corrector_rescue_pp"
        ],
        filters=[
            ("variant_key", "in", sorted(etitrain_keys))
        ]
    )

    eti["variant_key"] = eti.variant_key.astype(str)

    assert len(eti) == 422
    assert eti.variant_key.is_unique
    assert set(eti.variant_key) == etitrain_keys

    eti["protein_position"] = eti.hgvs_protein.map(
        parsed_position
    )

    if eti.protein_position.isna().any():
        raise ValueError(
            "Could not parse ETI TRAIN protein positions: "
            + str(
                eti.loc[
                    eti.protein_position.isna(),
                    ["variant_key", "hgvs_protein"]
                ].head().to_dict("records")
            )
        )

    eti["protein_position"] = (
        eti.protein_position.astype(int)
    )

    # Globally protected VTI validation/test positions.
    eti = eti.loc[
        ~eti.protein_position.isin(protected)
    ].copy()

    eti["corrector_rescue_pp"] = pd.to_numeric(
        eti.corrector_rescue_pp,
        errors="raise"
    )

    assert np.isfinite(
        eti.corrector_rescue_pp.to_numpy(float)
    ).all()

    assert len(eti) == 383
    assert eti.protein_position.nunique() == 276

    eti = eti.sort_values(
        "variant_key"
    ).reset_index(drop=True)

    eti["transfer_split"] = "train"
    eti["study_cohort"] = "ETI"

    # Generate the exact same mutation descriptors.
    seq = bihler_2024_assay_sequence()

    parsed = [
        parse_substitution(row, seq)
        for _, row in eti.iterrows()
    ]

    bad = [
        (i, p, eti.iloc[i].variant_key)
        for i, p in enumerate(parsed)
        if p[-1] != "valid"
        or p[0] != int(eti.iloc[i].protein_position)
    ]

    if bad:
        raise ValueError(
            f"ETI substitution/reference mismatch: {bad[:5]}"
        )

    eti["reference_aa"] = [p[1] for p in parsed]
    eti["alternate_aa"] = [p[2] for p in parsed]

    E68, names = make_features(eti, seq)
    E68 = np.asarray(E68, dtype=float)

    if E68.shape != (383, 68):
        raise ValueError(
            f"Unexpected ETI sequence features: {E68.shape}"
        )

    if not np.isfinite(E68).all():
        raise ValueError(
            "Nonfinite ETI sequence features"
        )

    # Reuse frozen Phase 15 structural descriptors.
    struct = structure.read_feature_table(
        ROOT / "results/structure/phase15b_position_features.csv",
        ROOT / "results/structure/phase15b_feature_summary.json",
        ROOT
    )

    E12 = np.asarray(
        structure.structural_matrix(
            eti.protein_position.to_numpy(int),
            struct
        ),
        float
    )

    if E12.shape != (383, 12):
        raise ValueError(
            f"Unexpected ETI structural features: {E12.shape}"
        )

    E80 = np.column_stack([E68, E12])
    E80[np.isinf(E80)] = np.nan

    V80 = np.asarray(
        mats["sequence_structure"],
        float
    ).copy()

    V80[np.isinf(V80)] = np.nan

    assert V80.shape == (210, 80)

    # Cross-study feature parity for exact matched variants.
    shared = sorted(
        vti_keys.intersection(set(eti.variant_key))
    )

    vi = {
        k: i
        for i, k in enumerate(vti.variant_key.astype(str))
    }

    ei = {
        k: i
        for i, k in enumerate(eti.variant_key.astype(str))
    }

    check_v = V80[[vi[k] for k in shared]]
    check_e = E80[[ei[k] for k in shared]]

    mismatch = np.logical_xor(
        np.isnan(check_v),
        np.isnan(check_e)
    )

    finite = (
        np.isfinite(check_v)
        & np.isfinite(check_e)
    )

    maxdiff = float(
        np.max(np.abs(check_v[finite] - check_e[finite]))
    ) if finite.any() else float("nan")

    print(
        f"Feature parity: {len(shared)} exact matches; "
        f"max difference={maxdiff:.6g}; "
        f"NaN mismatches={mismatch.sum()}"
    )

    if (
        len(shared) != 66
        or mismatch.any()
        or maxdiff > 1e-7
    ):
        raise ValueError(
            "ETI/VTI mutation features disagree. "
            "Refusing to train."
        )

    novel = (
        ~eti.protein_position.isin(
            vti.protein_position
        )
    ).sum()

    print(
        f"ETI auxiliary data: {len(eti)} variants / "
        f"{eti.protein_position.nunique()} positions"
    )

    print(
        f"Novel VTI positions: {novel} ETI variants"
    )

    print(
        f"ETI structural missing values: "
        f"{np.isnan(E12).sum()}"
    )

    print(
        f"Frozen outer splits: {len(folds)} | "
        f"reference={ref}"
    )

    return vti, V80, eti, E80, folds
