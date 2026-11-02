#!/usr/bin/env python3
"""Phase 24E: assay uncertainty audit, training records only."""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

BASE = Path("data/processed")
OUT = Path("results/biology/phase24e")
OUT.mkdir(parents=True, exist_ok=True)

p = pd.read_csv(
    "results/biology/phase24d/phase24d_baseline_corrected_function_audit.csv",
    dtype={"variant_key": str}
)
assert len(p) == 66
assert p.variant_key.is_unique
assert p.protein_position.nunique() == 56
keys = set(p.variant_key)

s24 = pd.read_parquet(
    BASE / "bihler_2024_missense_splits.parquet",
    columns=["variant_key", "split"]
)
s26 = pd.read_parquet(
    BASE / "bihler_2026_transfer_splits.parquet",
    columns=["variant_key", "transfer_split"]
)

assert keys.issubset(set(s24.loc[s24.split.eq("train"), "variant_key"]))
assert keys.issubset(set(s26.loc[s26.transfer_split.eq("train"), "variant_key"]))

def read_allowed(path, cols):
    return ds.dataset(str(path), format="parquet").to_table(
        columns=cols,
        filter=ds.field("variant_key").isin(sorted(keys))
    ).to_pandas()

def study(year, paired_file, prefix):
    raw = read_allowed(
        BASE / f"bihler_{year}_conditions.parquet",
        [
            "variant_key", "source_excel_row",
            "FSK, %WT", "FSK + IVA, %WT",
            "FSK (%WT) SEM", "FSK+IVA (%WT) SEM"
        ]
    )

    treated_col = (
        "source_row_elx_tez" if year == 2024
        else "source_row_vnz_tez"
    )

    pairs = read_allowed(
        BASE / paired_file,
        ["variant_key", "source_row_dmso", treated_col]
    )

    assert len(pairs) == 66 and pairs.variant_key.is_unique
    assert raw.source_excel_row.is_unique

    result = pd.DataFrame({"variant_key": sorted(keys)})

    for arm, source in [
        ("dmso", "source_row_dmso"),
        ("corr", treated_col)
    ]:
        matched = pairs[["variant_key", source]].merge(
            raw,
            left_on=["variant_key", source],
            right_on=["variant_key", "source_excel_row"],
            how="left",
            validate="one_to_one"
        )

        assert len(matched) == 66
        assert matched.source_excel_row.notna().all()

        part = matched[
            [
                "variant_key",
                "FSK, %WT",
                "FSK + IVA, %WT",
                "FSK (%WT) SEM",
                "FSK+IVA (%WT) SEM"
            ]
        ].rename(columns={
            "FSK, %WT": f"{prefix}_{arm}_fsk",
            "FSK + IVA, %WT": f"{prefix}_{arm}_fskiva",
            "FSK (%WT) SEM": f"{prefix}_{arm}_fsk_sem",
            "FSK+IVA (%WT) SEM": f"{prefix}_{arm}_fskiva_sem"
        })

        result = result.merge(
            part, on="variant_key", validate="one_to_one"
        )

    return result

p = p.merge(
    study(2024, "bihler_2024_theratyping.parquet", "eti"),
    on="variant_key",
    validate="one_to_one"
)
p = p.merge(
    study(2026, "bihler_2026_vti.parquet", "vti"),
    on="variant_key",
    validate="one_to_one"
)

# Verify all raw measurements reconcile with Phase 24D.
np.testing.assert_allclose(
    p.vti_corr_fskiva - p.vti_dmso_fsk,
    p.vti_total_pp, atol=1e-3, rtol=0
)
np.testing.assert_allclose(
    p.eti_corr_fskiva - p.eti_dmso_fsk,
    p.eti_total_pp, atol=1e-3, rtol=0
)
np.testing.assert_allclose(
    p.vti_corr_fsk - p.eti_corr_fsk,
    p.corrected_activity_shift, atol=1e-3, rtol=0
)
np.testing.assert_allclose(
    p.vti_dmso_fsk - p.eti_dmso_fsk,
    p.baseline_shift, atol=1e-3, rtol=0
)

# Approximate SEM combinations.
# Assumes independent errors; covariance and study effects unknown.
terms = {
    "baseline": [
        "vti_dmso_fsk_sem", "eti_dmso_fsk_sem"
    ],
    "corrected": [
        "vti_corr_fsk_sem", "eti_corr_fsk_sem"
    ],
    "corrector_gain": [
        "vti_corr_fsk_sem", "eti_corr_fsk_sem",
        "vti_dmso_fsk_sem", "eti_dmso_fsk_sem"
    ],
    "acute_iva": [
        "vti_corr_fskiva_sem", "eti_corr_fskiva_sem",
        "vti_corr_fsk_sem", "eti_corr_fsk_sem"
    ],
    "total_rescue": [
        "vti_corr_fskiva_sem", "eti_corr_fskiva_sem",
        "vti_dmso_fsk_sem", "eti_dmso_fsk_sem"
    ]
}

for name, columns in terms.items():
    for col in columns:
        p[col] = pd.to_numeric(p[col], errors="coerce")

    p[f"{name}_sem_proxy"] = np.sqrt(
        sum(p[col] ** 2 for col in columns)
    )

p["total_disagreement_over_sem_proxy"] = (
    p.delta_total.abs()
    / p.total_rescue_sem_proxy.replace(0, np.nan)
)
p["corrected_shift_over_sem_proxy"] = (
    p.corrected_activity_shift.abs()
    / p.corrected_sem_proxy.replace(0, np.nan)
)

p["big_disagreement"] = p.delta_total.abs() > 20

print("=" * 100)
print("PHASE 24E | ETI-VTI DISAGREEMENT VS ASSAY UNCERTAINTY")
print("=" * 100)
print("66 exact variants / 56 protein positions")
print("2024 TRAIN and VTI TRAIN only")
print("Source-row assay values verified")
print("No protected outcomes opened")
print("WARNING: SEM proxies assume independence and omit study effects.")

print("\nA. ASSAY UNCERTAINTY COVERAGE")
for name in terms:
    x = p[f"{name}_sem_proxy"]
    print(
        f"{name:20s} n={x.notna().sum():2d}/66 "
        f"median={x.median():8.3f} "
        f"p90={x.quantile(.9):8.3f}"
    )

print("\nB. TOTAL DISAGREEMENT RELATIVE TO SEM")
r = p.total_disagreement_over_sem_proxy
print("Available ratios:", r.notna().sum())
print("Median |difference| / SEM proxy:", round(r.median(), 3))

for cutoff in [1, 2, 3]:
    print(
        f"Ratio above {cutoff}: "
        f"{int((r > cutoff).sum())}/{r.notna().sum()}"
    )

large = p.loc[p.big_disagreement]

print("\nC. LARGE TOTAL DISAGREEMENTS (>20 pp)")
print("Number:", len(large))

for cutoff in [1, 2, 3]:
    rr = large.total_disagreement_over_sem_proxy
    print(
        f"Ratio above {cutoff}: "
        f"{int((rr > cutoff).sum())}/{rr.notna().sum()}"
    )

print("\nD. LARGEST DISAGREEMENTS")
show = [
    "variant_key",
    "protein_position",
    "delta_total",
    "baseline_shift",
    "corrected_activity_shift",
    "delta_acute",
    "total_rescue_sem_proxy",
    "total_disagreement_over_sem_proxy",
    "corrected_sem_proxy",
    "corrected_shift_over_sem_proxy"
]

print(
    p.assign(abs_total=p.delta_total.abs())
    .sort_values("abs_total", ascending=False)[show]
    .head(15)
    .round(3)
    .to_string(index=False)
)

output = OUT / "phase24e_sem_vs_regimen_disagreement.csv"
p.to_csv(output, index=False)

print("\nSaved:", output)
print("PASS: Phase 24E completed; training-only")
