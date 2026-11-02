#!/usr/bin/env python3
"""Phase 26F: permitted TRAIN-only ETI corrector auxiliary-data audit."""

from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path.cwd()
OUT = ROOT / "results/biology/phase26f"


def protein_pos(s):
    s = str(s).strip()
    m = re.search(
        r"(?:p\.)?\(?[A-Za-z]{3}(\d{1,4})(?:[A-Za-z]{3}|Ter|\*|=)",
        s
    )
    if m is None:
        m = re.search(
            r"(?:p\.)?\(?[A-Za-z](\d{1,4})[A-Za-z*]", s
        )
    if m is None:
        return np.nan
    p = int(m.group(1))
    return p if 1 <= p <= 1480 else np.nan


def main():
    print("=" * 100)
    print("PHASE 26F | ETI CORRECTOR AUXILIARY SUPERVISION | TRAIN ONLY")
    print("=" * 100)

    old_split = pd.read_parquet(
        ROOT / "data/processed/bihler_2024_missense_splits.parquet",
        columns=["variant_key", "split"]
    )

    vti_split = pd.read_parquet(
        ROOT / "data/processed/bihler_2026_transfer_splits.parquet",
        columns=["variant_key", "protein_position", "transfer_split"]
    )

    assert old_split.variant_key.is_unique
    assert vti_split.variant_key.is_unique

    eti_keys = set(
        old_split.loc[
            old_split.split.eq("train"), "variant_key"
        ].astype(str)
    )

    vti_keys = set(
        vti_split.loc[
            vti_split.transfer_split.eq("train"), "variant_key"
        ].astype(str)
    )

    assert len(vti_keys) == 210

    # Predicate filtering prevents protected outcomes being loaded.
    eti = pd.read_parquet(
        ROOT / "data/processed/bihler_2024_theratyping.parquet",
        columns=[
            "variant_key",
            "hgvs_protein",
            "corrector_rescue_pp",
            "eti_rescue_pp"
        ],
        filters=[("variant_key", "in", sorted(eti_keys))]
    )

    vti = pd.read_parquet(
        ROOT / "data/processed/bihler_2026_vti.parquet",
        columns=[
            "variant_key",
            "corrector_rescue_pp",
            "vti_rescue_pp"
        ],
        filters=[("variant_key", "in", sorted(vti_keys))]
    )

    assert set(eti.variant_key).issubset(eti_keys)
    assert set(vti.variant_key) == vti_keys
    assert eti.variant_key.is_unique
    assert vti.variant_key.is_unique

    eti["protein_position"] = eti.hgvs_protein.map(protein_pos)
    eti["eti_corrector_pp"] = pd.to_numeric(
        eti.corrector_rescue_pp, errors="coerce"
    )
    vti["vti_corrector_pp"] = pd.to_numeric(
        vti.corrector_rescue_pp, errors="coerce"
    )

    eti = eti[
        ["variant_key", "protein_position",
         "eti_corrector_pp", "eti_rescue_pp"]
    ]

    vt = vti_split.loc[
        vti_split.transfer_split.eq("train"),
        ["variant_key", "protein_position"]
    ].merge(
        vti[["variant_key", "vti_corrector_pp", "vti_rescue_pp"]],
        on="variant_key",
        validate="one_to_one"
    )

    print("\nA. ALLOWED TRAINING DATA")
    print(
        f"ETI TRAIN: {len(eti)} variants; "
        f"parsed positions: {eti.protein_position.notna().sum()}"
    )
    print(
        f"VTI TRAIN: {len(vt)} variants; "
        f"{vt.protein_position.nunique()} positions"
    )

    if eti.protein_position.isna().any():
        print(
            "Unparsed ETI variant examples:",
            eti.loc[
                eti.protein_position.isna(), "variant_key"
            ].head(8).tolist()
        )

    protected_positions = set(
        pd.to_numeric(
            vti_split.loc[
                vti_split.transfer_split.isin(["val", "test"]),
                "protein_position"
            ],
            errors="coerce"
        ).dropna().astype(int)
    )

    eti_aux = eti.loc[
        eti.protein_position.notna()
        & np.isfinite(eti.eti_corrector_pp)
    ].copy()

    eti_aux["protein_position"] = (
        eti_aux.protein_position.astype(int)
    )

    removed = eti_aux.protein_position.isin(protected_positions)

    print(
        "Protected VTI validation/test positions:",
        len(protected_positions)
    )
    print(
        "ETI training rows excluded at protected positions:",
        int(removed.sum())
    )

    eti_aux = eti_aux.loc[~removed].copy()

    print(
        f"Eligible ETI auxiliary pool: {len(eti_aux)} variants "
        f"/ {eti_aux.protein_position.nunique()} positions"
    )

    # Descriptive comparison using training outcomes only.
    paired = vt.merge(
        eti_aux[["variant_key", "eti_corrector_pp"]],
        on="variant_key",
        how="inner",
        validate="one_to_one"
    )

    assert set(paired.variant_key).issubset(vti_keys & eti_keys)

    paired = paired.loc[
        np.isfinite(paired.vti_corrector_pp)
    ].copy()

    print("\nB. ETI-VTI MATCHED CORRECTOR RESPONSE")

    print(
        f"Matched variants: {len(paired)} | "
        f"positions: {paired.protein_position.nunique()}"
    )

    if len(paired) >= 5:
        rho, pvalue = spearmanr(
            paired.eti_corrector_pp,
            paired.vti_corrector_pp
        )

        difference = (
            paired.vti_corrector_pp - paired.eti_corrector_pp
        )

        position_error = pd.DataFrame({
            "position": paired.protein_position,
            "error": np.abs(difference)
        }).groupby("position").error.mean().mean()

        print(f"Corrector Spearman correlation: {rho:.4f}")
        print(f"Descriptive p-value: {pvalue:.5g}")

        print(
            "Direct ETI-to-VTI corrector variant MAE:",
            round(np.abs(difference).mean(), 4), "pp"
        )

        print(
            "Direct ETI-to-VTI corrector position MAE:",
            round(position_error, 4), "pp"
        )

        print(
            "Median VTI-ETI corrector difference:",
            round(np.median(difference), 4), "pp"
        )

        print(
            "Corrector response >40 pp:",
            "ETI =", int((paired.eti_corrector_pp > 40).sum()),
            "| VTI =", int((paired.vti_corrector_pp > 40).sum())
        )

    print("\nC. NOVEL AUXILIARY CORRECTOR DATA")

    vti_positions = set(vt.protein_position.astype(int))

    novel = eti_aux.loc[
        ~eti_aux.protein_position.isin(vti_positions)
    ]

    print(
        "Auxiliary ETI variants at novel positions:",
        len(novel)
    )
    print(
        "Independent novel protein positions:",
        novel.protein_position.nunique()
    )

    print("\nD. FOLD-SPECIFIC LEAKAGE AUDIT")

    oof_path = (
        ROOT / "results/biology/phase26d/phase26d_oof.csv"
    )

    fold_report = []

    if oof_path.exists():
        oof = pd.read_csv(
            oof_path,
            usecols=["repeat", "fold", "protein_position"]
        ).drop_duplicates()

        for (rep, fold), test in oof.groupby(
            ["repeat", "fold"]
        ):
            heldout = set(
                test.protein_position.astype(int)
            )

            eligible = eti_aux.loc[
                ~eti_aux.protein_position.isin(heldout)
            ]

            fold_report.append({
                "repeat": rep,
                "fold": fold,
                "heldout_positions": len(heldout),
                "eligible_eti_variants": len(eligible),
                "eligible_eti_positions":
                    eligible.protein_position.nunique(),
                "removed_for_outer_holdout":
                    len(eti_aux) - len(eligible)
            })

        folds = pd.DataFrame(fold_report)
        print(folds.to_string(index=False))
    else:
        folds = pd.DataFrame()
        print("Phase26D OOF missing")

    print("\nE. EXISTING AUXILIARY FEATURE FILES")

    feature_files = [
        p for p in (ROOT / "data").rglob("*")
        if p.is_file()
        and any(
            key in p.name.lower()
            for key in (
                "feature",
                "embedding",
                "structure",
                "esm"
            )
        )
    ]

    for p in sorted(feature_files)[:35]:
        print(" ", p.relative_to(ROOT))

    print(
        f"Found {len(feature_files)} candidate files "
        "(first 35 displayed)"
    )

    OUT.mkdir(parents=True, exist_ok=True)

    paired.to_csv(
        OUT / "phase26f_paired_train_corrector.csv",
        index=False
    )

    if len(folds):
        folds.to_csv(
            OUT / "phase26f_auxiliary_fold_sizes.csv",
            index=False
        )

    pd.DataFrame({
        "path": [
            str(p.relative_to(ROOT))
            for p in feature_files
        ]
    }).to_csv(
        OUT / "phase26f_feature_candidates.csv",
        index=False
    )

    print("\nPASS: Phase26F training-only audit completed")
    print("No protected VTI or ETI outcomes accessed")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
