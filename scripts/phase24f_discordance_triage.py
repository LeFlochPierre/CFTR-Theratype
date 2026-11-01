#!/usr/bin/env python3
"""Phase 24F: train-only cross-regimen discrepancy reliability audit."""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("results/biology/phase24f")
OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(
    "results/biology/phase24e/phase24e_sem_vs_regimen_disagreement.csv",
    dtype={"variant_key": str},
)

assert len(d) == 66
assert d.variant_key.is_unique
assert d.protein_position.nunique() == 56

# Verify TRAIN-only provenance.
s24 = pd.read_parquet(
    "data/processed/bihler_2024_missense_splits.parquet",
    columns=["variant_key", "split"],
)
s26 = pd.read_parquet(
    "data/processed/bihler_2026_transfer_splits.parquet",
    columns=["variant_key", "transfer_split"],
)

keys = set(d.variant_key)
assert keys.issubset(set(s24.loc[s24.split.eq("train"), "variant_key"]))
assert keys.issubset(set(s26.loc[s26.transfer_split.eq("train"), "variant_key"]))

# Check additive decomposition.
np.testing.assert_allclose(
    d.delta_total,
    d.delta_corrector + d.delta_acute,
    atol=1e-3, rtol=0,
)

z = d.total_disagreement_over_sem_proxy
assert np.isfinite(z).all()
assert (d.total_rescue_sem_proxy > 0).all()

# All definitions are exploratory.
d["large_disagreement"] = d.delta_total.abs() > 20
d["above_2sem"] = z > 2
d["above_3sem"] = z > 3

d["large_supported_2sem"] = (
    d.large_disagreement & d.above_2sem
)
d["large_supported_3sem"] = (
    d.large_disagreement & d.above_3sem
)

d["corrector_dominant"] = (
    d.delta_corrector.abs() > d.delta_acute.abs()
)
d["corrected_activity_dominant"] = (
    d.corrected_activity_shift.abs() > d.baseline_shift.abs()
)

# Research-only threshold: total rescue <10 pp.
d["eti_low"] = d.eti_total_pp < 10
d["vti_low"] = d.vti_total_pp < 10

d["eti_low_vti_high"] = d.eti_low & ~d.vti_low
d["eti_high_vti_low"] = ~d.eti_low & d.vti_low

d["ambiguous_identity"] = d.variant_key.str.contains(
    r"\s+or\s+|/|;", regex=True, case=False, na=False
)

print("=" * 100)
print("PHASE 24F | CROSS-REGIMEN DISCREPANCY RELIABILITY")
print("=" * 100)
print("66 variants / 56 positions")
print("TRAIN-only, no protected outcomes read")
print("SEM ratios descriptive; independence assumed")

print("\nA. EVIDENCE TIERS")

for label, mask in [
    ("All discrepancies >20 pp", d.large_disagreement),
    ("All discrepancies >2x SEM proxy", d.above_2sem),
    ("All discrepancies >3x SEM proxy", d.above_3sem),
    ("Large AND >2x SEM", d.large_supported_2sem),
    ("Large AND >3x SEM", d.large_supported_3sem),
]:
    print(
        f"{label:37s} variants={mask.sum():3d} "
        f"positions={d.loc[mask, 'protein_position'].nunique():3d}"
    )

print("\nB. MECHANISMS AMONG LARGE SUPPORTED DISCREPANCIES")

s = d.loc[d.large_supported_2sem].copy()
print("Number:", len(s))
print("Corrector component dominates:",
      int(s.corrector_dominant.sum()), "/", len(s))
print("Corrected activity exceeds baseline shift:",
      int(s.corrected_activity_dominant.sum()), "/", len(s))
print("VTI higher:", int((s.delta_total > 0).sum()))
print("ETI higher:", int((s.delta_total < 0).sum()))

print("\nC. IN-VITRO 10 pp THRESHOLD DISAGREEMENT")

print(pd.crosstab(
    d.eti_low, d.vti_low,
    rownames=["ETI <10"],
    colnames=["VTI <10"],
).to_string())

for label, mask in [
    ("ETI <10 but VTI >=10", d.eti_low_vti_high),
    ("ETI >=10 but VTI <10", d.eti_high_vti_low),
]:
    x = d.loc[mask]
    print(
        f"{label:28s} variants={len(x):2d} "
        f"positions={x.protein_position.nunique():2d} "
        f">2xSEM={int(x.above_2sem.sum()):2d}"
    )

print("\nD. BEST-SUPPORTED LARGE DISAGREEMENTS")
show = [
    "variant_key", "protein_position",
    "eti_total_pp", "vti_total_pp",
    "delta_total", "delta_corrector", "delta_acute",
    "total_rescue_sem_proxy",
    "total_disagreement_over_sem_proxy",
]
print(
    s.sort_values(
        "total_disagreement_over_sem_proxy",
        ascending=False,
    )[show].round(3).to_string(index=False)
)

print("\nE. AMBIGUOUS IDENTITY SENSITIVITY")
print("Ambiguous variant keys:", int(d.ambiguous_identity.sum()))
clean = d.loc[~d.ambiguous_identity]
print("Unambiguous variants:", len(clean))
print("Large supported >2xSEM:",
      int(clean.large_supported_2sem.sum()))
print("ETI-low / VTI-high:",
      int(clean.eti_low_vti_high.sum()))

d.to_csv(
    OUT / "phase24f_discrepancy_evidence.csv",
    index=False
)

print("\nSaved:", OUT / "phase24f_discrepancy_evidence.csv")
print("PASS: Phase24F TRAIN-only reliability audit")
