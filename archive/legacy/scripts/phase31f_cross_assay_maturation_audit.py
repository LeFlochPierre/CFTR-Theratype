"""Phase 31F: Calibration robustness and protein maturation.

TRAIN-only exploratory study.
No new model fitting; no validation/test outcomes.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PRED = Path(
    "results/biology/phase31e/"
    "phase31e_group_cv_predictions.csv"
)
S6 = Path("phase31a_castellani_s6_processing.csv")
OUT = Path("results/biology/phase31f")
OUT.mkdir(parents=True, exist_ok=True)

for path in [PRED, S6]:
    if not path.is_file():
        raise FileNotFoundError(path)

pred = pd.read_csv(PRED)

models = [
    "identity",
    "isotonic_response",
    "median_shift",
]

pred = pred[pred.model.isin(models)].copy()

keys = [
    "study", "target_scale", "aa_code",
    "protein_position", "fold",
    "bihler_eti_pp", "target_pp",
]

assert not pred.duplicated(
    ["study", "target_scale", "model", "aa_code"]
).any()

p = pred.pivot(
    index=keys,
    columns="model",
    values="prediction_pp"
).reset_index()

assert set(models).issubset(p.columns)

for model in models:
    p[f"ae_{model}"] = (
        p.target_pp - p[model]
    ).abs()

p["gain_iso_identity"] = (
    p.ae_identity - p.ae_isotonic_response
)

p["gain_iso_shift"] = (
    p.ae_median_shift - p.ae_isotonic_response
)

p["residual_iso"] = (
    p.target_pp - p.isotonic_response
)

rng = np.random.default_rng(31)


def bootstrap_improvement(data, baseline, candidate):
    """Bootstrap positions, keeping substitutions together."""

    grouped = data.groupby("protein_position").agg(
        n=("aa_code", "size"),
        baseline_sum=(baseline, "sum"),
        candidate_sum=(candidate, "sum"),
    )

    arr = grouped[
        ["n", "baseline_sum", "candidate_sum"]
    ].to_numpy(dtype=float)

    n_positions = len(arr)
    draws = rng.integers(
        0, n_positions, size=(3000, n_positions)
    )

    sampled = arr[draws]

    gain = (
        sampled[:, :, 1].sum(axis=1)
        - sampled[:, :, 2].sum(axis=1)
    ) / sampled[:, :, 0].sum(axis=1)

    return np.quantile(gain, [0.025, 0.975])


print("=" * 88)
print("PHASE 31F | CALIBRATION ROBUSTNESS + PROTEIN MATURATION")
print("TRAIN only | no new model fitting")
print("=" * 88)

print("\nA. POSITION-BOOTSTRAPPED MAE IMPROVEMENTS")

for (study, scale), group in p.groupby(
    ["study", "target_scale"]
):
    identity_gain = group.gain_iso_identity.mean()
    shift_gain = group.gain_iso_shift.mean()

    ci_identity = bootstrap_improvement(
        group, "ae_identity", "ae_isotonic_response"
    )

    ci_shift = bootstrap_improvement(
        group, "ae_median_shift", "ae_isotonic_response"
    )

    print(f"\n{study} / {scale}")
    print(
        f"N={len(group)} | "
        f"positions={group.protein_position.nunique()}"
    )
    print(
        f"Isotonic improvement vs identity: "
        f"{identity_gain:+.3f} pp "
        f"CI [{ci_identity[0]:+.3f}, {ci_identity[1]:+.3f}]"
    )
    print(
        f"Isotonic improvement vs median shift: "
        f"{shift_gain:+.3f} pp "
        f"CI [{ci_shift[0]:+.3f}, {ci_shift[1]:+.3f}]"
    )

s6 = pd.read_csv(S6)

features = [
    "baseline_mature_fraction",
    "eti_mature_fraction",
    "baseline_mature_pct_normal",
    "eti_mature_pct_normal",
]

required = {"study", "aa_code", *features}
assert required.issubset(s6.columns)

s6 = s6.loc[
    s6.aa_code.notna(),
    ["study", "aa_code", *features]
].copy()

s6["aa_code"] = (
    s6.aa_code.astype(str).str.upper().str.strip()
)

assert not s6.duplicated(
    ["study", "aa_code"]
).any()

native = p[
    p.target_scale == "published_native"
].copy()

native["aa_code"] = (
    native.aa_code.astype(str).str.upper().str.strip()
)

joined = native.merge(
    s6,
    on=["study", "aa_code"],
    how="left",
    validate="one_to_one",
)

joined["mature_fraction_gain"] = (
    joined.eti_mature_fraction
    - joined.baseline_mature_fraction
)

joined["mature_pct_gain"] = (
    joined.eti_mature_pct_normal
    - joined.baseline_mature_pct_normal
)

joined["response_group"] = pd.cut(
    joined.bihler_eti_pp,
    [-np.inf, 10, 40, np.inf],
    labels=[
        "Bihler <10",
        "Bihler 10-<40",
        "Bihler >=40",
    ],
    right=False,
)

print("\nB. CALIBRATION BENEFIT BY RESPONSE MAGNITUDE")

for (study, stratum), group in joined.groupby(
    ["study", "response_group"],
    observed=True
):
    print(
        f"{study} | {stratum} | n={len(group)}"
        f" | direct MAE={group.ae_identity.mean():.3f}"
        f" | isotonic MAE={group.ae_isotonic_response.mean():.3f}"
        f" | improvement={group.gain_iso_identity.mean():+.3f}"
    )

print("\nC. PROTEIN MATURATION ASSOCIATIONS")

for study, group in joined.groupby("study"):

    available = group.dropna(
        subset=[
            "baseline_mature_pct_normal",
            "eti_mature_pct_normal",
            "residual_iso",
        ]
    )

    print(
        f"\n{study}: S6 matched "
        f"{len(available)}/{len(group)} variants"
    )

    for feature in [
        "baseline_mature_pct_normal",
        "mature_pct_gain",
        "mature_fraction_gain",
    ]:
        for outcome in [
            "residual_iso",
            "ae_isotonic_response",
        ]:
            x = available[
                [feature, outcome]
            ].dropna()

            rho = (
                x[feature].corr(
                    x[outcome],
                    method="spearman",
                )
                if len(x) >= 4 else np.nan
            )

            print(
                f"Spearman({feature}, {outcome}) "
                f"= {rho:+.3f} | n={len(x)}"
            )

print("\nD. LARGEST REMAINING CALIBRATION ERRORS")

for study, group in joined.groupby("study"):

    top = group.assign(
        abs_residual=group.residual_iso.abs()
    ).nlargest(8, "abs_residual")

    print(f"\n{study}")

    print(
        top[[
            "aa_code",
            "bihler_eti_pp",
            "target_pp",
            "isotonic_response",
            "residual_iso",
            "mature_pct_gain",
        ]].round(2).to_string(index=False)
    )

p.to_csv(
    OUT / "phase31f_oof_calibration_gains.csv",
    index=False,
)

joined.to_csv(
    OUT / "phase31f_native_s6_residual_audit.csv",
    index=False,
)

print("\nPASS: Phase 31F complete")
print("No new predictive models fitted.")
print("Bootstrap is conditional on existing CV predictions.")
print("Model selection and training uncertainty are not included.")
print("Post-treatment maturation is explanatory evidence,")
print("not a pretreatment prediction feature.")
print("Results:", OUT)
