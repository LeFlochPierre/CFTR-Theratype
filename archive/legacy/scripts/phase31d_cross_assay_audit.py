from pathlib import Path
import numpy as np
import pandas as pd

SOURCE = Path("results/biology/phase31c/phase31c_training_overlap.csv")
PAIRED = Path("results/biology/phase31a/phase31a_bihler_eti_overlap.csv")
OUT = Path("results/biology/phase31d")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE)
paired = pd.read_csv(PAIRED)

required = {
    "study", "aa_code", "protein_position",
    "bihler_eti", "bihler_baseline",
    "eti_rescue_pp", "castellani_rescue_263",
    "baseline_pct_normal", "n_dna_variants",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

assert not df.duplicated(["study", "aa_code"]).any()

paired_codes = set(
    paired["aa_code"].dropna().astype(str).str.upper()
)
df["in_original_paired_cohort"] = (
    df.aa_code.astype(str).str.upper().isin(paired_codes)
)

df["difference_native"] = df.eti_rescue_pp - df.bihler_eti
df["difference_263"] = (
    df.castellani_rescue_263 - df.bihler_eti
)

df["bihler_responder10"] = df.bihler_eti >= 10
df["castellani_responder10"] = df.eti_rescue_pp >= 10

df["response_group"] = pd.cut(
    df.bihler_eti,
    bins=[-np.inf, 10, 40, np.inf],
    labels=["low_<10", "mid_10_40", "high_>40"],
    right=False,
)

def evaluate(x):
    if x.empty:
        return None

    d = x.difference_native
    d263 = x.difference_263

    agreement = (
        x.bihler_responder10
        == x.castellani_responder10
    )

    return {
        "n": len(x),
        "n_positions": x.protein_position.nunique(),
        "bihler_median": x.bihler_eti.median(),
        "castellani_median": x.eti_rescue_pp.median(),
        "median_signed_difference": d.median(),
        "median_abs_difference": d.abs().median(),
        "median_abs_difference_263": d263.abs().median(),
        "spearman": (
            x.bihler_eti.corr(
                x.eti_rescue_pp, method="spearman"
            ) if len(x) >= 3 else np.nan
        ),
        "agreement10": agreement.mean(),
        "bihler_pos10": int(x.bihler_responder10.sum()),
        "castellani_pos10": int(x.castellani_responder10.sum()),
        "bihler_only_pos10": int((
            x.bihler_responder10
            & ~x.castellani_responder10
        ).sum()),
        "castellani_only_pos10": int((
            ~x.bihler_responder10
            & x.castellani_responder10
        ).sum()),
    }

print("=" * 90)
print("PHASE 31D | RESPONSE-DEPENDENT CROSS-ASSAY DISAGREEMENT")
print("TRAIN-ONLY | NO MODELLING | NO HARMONISATION CLAIM")
print("=" * 90)

rows = []

for study, group in df.groupby("study"):
    print(f"\n{'=' * 65}")
    print("STUDY:", study)
    print(f"{'=' * 65}")

    subsets = {
        "all_train": group,
        "original_paired_subset": group[
            group.in_original_paired_cohort
        ],
        "outside_paired_subset": group[
            ~group.in_original_paired_cohort
        ],
    }

    for label, subset in subsets.items():
        result = evaluate(subset)
        if result is None:
            continue
        rows.append({
            "study": study,
            "group": label,
            **result
        })

    print("\nA. RESPONSE MAGNITUDE STRATA")
    stratified = []
    for label, x in group.groupby(
        "response_group", observed=True
    ):
        result = evaluate(x)
        if result is not None:
            stratified.append({
                "group": str(label),
                **result
            })

    print(pd.DataFrame(stratified).round(3).to_string(index=False))

    print("\nB. RESPONDER CLASSIFICATION, NATIVE SCALES")
    print(pd.crosstab(
        group.bihler_responder10,
        group.castellani_responder10,
        rownames=["Bihler >=10"],
        colnames=["Castellani >=10"],
    ).to_string())

    print("\nC. BASELINE FUNCTION COMPARISON")
    b = group[[
        "bihler_baseline", "baseline_pct_normal"
    ]].dropna()

    print("N:", len(b))
    print("Spearman:", round(
        b.bihler_baseline.corr(
            b.baseline_pct_normal,
            method="spearman"
        ), 3
    ))
    print("Median absolute difference:", round(
        (b.bihler_baseline - b.baseline_pct_normal)
        .abs().median(), 3
    ))

    print("\nD. TOP 5 RESCUE DISAGREEMENTS")
    top = group.assign(
        absdiff=group.difference_native.abs()
    ).sort_values("absdiff", ascending=False).head(5)

    print(top[[
        "aa_code", "bihler_eti", "eti_rescue_pp",
        "castellani_rescue_263",
        "difference_native"
    ]].round(2).to_string(index=False))

summary = pd.DataFrame(rows)

print("\n" + "=" * 90)
print("E. PAIRED SUBSET SELECTION AUDIT")
print(summary.round(3).to_string(index=False))

df.to_csv(
    OUT / "phase31d_variant_disagreements.csv",
    index=False
)
summary.to_csv(
    OUT / "phase31d_cohort_comparison.csv",
    index=False
)

print("\nPASS: Frozen Bihler 2024 TRAIN only")
print("Responder threshold is exploratory cross-assay agreement.")
print("No VTI or protected Bihler outcomes accessed.")
print("Results:", OUT)
