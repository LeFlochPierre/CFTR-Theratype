"""Phase 31G: Protein maturation and cross-assay transfer.

TRAIN-only exploratory analysis.
All target-assay maturation measurements are post-treatment
experimental information, not pretreatment prediction features.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SOURCE = Path(
    "results/biology/phase31f/"
    "phase31f_native_s6_residual_audit.csv"
)

OUT = Path("results/biology/phase31g")
OUT.mkdir(parents=True, exist_ok=True)

REQUIRED = [
    "study",
    "aa_code",
    "protein_position",
    "bihler_eti_pp",
    "target_pp",
    "baseline_mature_pct_normal",
    "mature_pct_gain",
    "mature_fraction_gain",
]

SPECS = {
    "isotonic_only": [],
    "iso_plus_baseline_maturation": [
        "baseline_mature_pct_normal"
    ],
    "iso_plus_maturation_gain": [
        "mature_pct_gain"
    ],
    "iso_plus_fraction_gain": [
        "mature_fraction_gain"
    ],
    "iso_plus_gain_and_baseline": [
        "mature_pct_gain",
        "baseline_mature_pct_normal",
    ],
}


def signed_log1p(a):
    a = np.asarray(a, dtype=float)
    return np.sign(a) * np.log1p(np.abs(a))


def boot_ci(
    group,
    candidate,
    reference="ae_isotonic_only",
    n_boot=3000,
):
    """Position-cluster bootstrap of MAE improvement."""

    values = group.groupby("protein_position").agg(
        n=("aa_code", "size"),
        ref_sum=(reference, "sum"),
        candidate_sum=(candidate, "sum"),
    ).to_numpy(dtype=float)

    rng = np.random.default_rng(3107)

    draws = rng.integers(
        0,
        len(values),
        size=(n_boot, len(values)),
    )

    samples = values[draws].sum(axis=1)

    gains = (
        samples[:, 1] - samples[:, 2]
    ) / samples[:, 0]

    return np.quantile(gains, [0.025, 0.975])


if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

df = pd.read_csv(SOURCE)

missing = set(REQUIRED) - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

assert not df.duplicated(["study", "aa_code"]).any()
assert set(df.study.unique()) <= {"P289", "U032"}

all_predictions = []

print("=" * 90)
print("PHASE 31G | MATURATION-BASED CROSS-ASSAY EXPLANATION")
print("TRAIN only | 5-fold position-held-out CV")
print("Target-assay maturation is post-treatment information")
print("=" * 90)


for study, original in df.groupby("study"):

    data = original.dropna(
        subset=REQUIRED
    ).copy().reset_index(drop=True)

    x = data.bihler_eti_pp.to_numpy(dtype=float)
    y = data.target_pp.to_numpy(dtype=float)

    groups = data.protein_position.to_numpy(dtype=int)

    print(
        f"\nSTUDY {study}"
        f" | usable={len(data)}/{len(original)}"
        f" | positions={len(np.unique(groups))}"
    )

    assert len(np.unique(groups)) >= 5
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()

    predictions = {
        name: np.full(len(data), np.nan)
        for name in SPECS
    }

    fold_ids = np.zeros(len(data), dtype=int)

    cv = GroupKFold(n_splits=5)

    for fold, (train, test) in enumerate(
        cv.split(x, y, groups=groups),
        start=1,
    ):

        assert not (
            set(groups[train]) & set(groups[test])
        )

        fold_ids[test] = fold

        # Base calibration: Bihler ETI -> Castellani ETI
        iso = IsotonicRegression(
            increasing=True,
            out_of_bounds="clip",
        )

        iso.fit(x[train], y[train])

        base_train = iso.predict(x[train])
        base_test = iso.predict(x[test])

        predictions["isotonic_only"][test] = base_test

        # Residuals are computed on the outer training fold.
        residual_train = y[train] - base_train

        for name, features in SPECS.items():

            if not features:
                continue

            # Prespecified signed log transform to reduce
            # influence of extreme protein-maturation values.
            X_train = signed_log1p(
                data.loc[train, features].to_numpy(float)
            )

            X_test = signed_log1p(
                data.loc[test, features].to_numpy(float)
            )

            model = make_pipeline(
                StandardScaler(),
                Ridge(alpha=20.0),
            )

            model.fit(X_train, residual_train)

            predictions[name][test] = (
                base_test + model.predict(X_test)
            )

    block = data[[
        "study",
        "aa_code",
        "protein_position",
        "bihler_eti_pp",
        "target_pp",
        "mature_pct_gain",
        "baseline_mature_pct_normal",
        "mature_fraction_gain",
    ]].copy()

    block["fold"] = fold_ids

    for name, pred in predictions.items():

        assert np.isfinite(pred).all()

        block[f"pred_{name}"] = pred

        block[f"ae_{name}"] = np.abs(
            y - pred
        )

    all_predictions.append(block)

    print("\nA. HELD-OUT PREDICTION PERFORMANCE")

    for name in SPECS:

        ae = block[f"ae_{name}"]

        mae = ae.mean()

        rmse = np.sqrt(
            np.mean((y - predictions[name]) ** 2)
        )

        if name == "isotonic_only":
            print(
                f"{name:34s}"
                f" MAE={mae:.3f}"
                f" RMSE={rmse:.3f}"
                " | REFERENCE"
            )
            continue

        gain = (
            block.ae_isotonic_only.mean() - mae
        )

        ci = boot_ci(
            block,
            f"ae_{name}",
        )

        print(
            f"{name:34s}"
            f" MAE={mae:.3f}"
            f" RMSE={rmse:.3f}"
            f" | gain={gain:+.3f}"
            f" | CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]"
        )

    # Explore whether association persists within
    # Bihler response-magnitude strata.
    block["iso_residual"] = (
        y - predictions["isotonic_only"]
    )

    block["stratum"] = pd.cut(
        x,
        [-np.inf, 10, 40, np.inf],
        labels=["<10", "10-<40", ">=40"],
        right=False,
    )

    print("\nB. WITHIN-STRATUM ASSOCIATIONS")

    for label, group in block.groupby(
        "stratum",
        observed=True,
    ):

        rho = (
            group.iso_residual.corr(
                group.mature_pct_gain,
                method="spearman",
            )
            if len(group) >= 5
            else np.nan
        )

        print(
            f"{label}"
            f" | n={len(group)}"
            f" | residual/maturation rho={rho:+.3f}"
            f" | iso MAE={group.ae_isotonic_only.mean():.3f}"
            f" | plus-gain MAE="
            f"{group.ae_iso_plus_maturation_gain.mean():.3f}"
        )


joined = pd.concat(
    all_predictions,
    ignore_index=True,
)

joined.to_csv(
    OUT / "phase31g_oof_predictions.csv",
    index=False,
)

summary = []

for study, group in joined.groupby("study"):

    for name in SPECS:

        gain = (
            group.ae_isotonic_only.mean()
            - group[f"ae_{name}"].mean()
        )

        ci = (
            [np.nan, np.nan]
            if name == "isotonic_only"
            else boot_ci(group, f"ae_{name}")
        )

        summary.append({
            "study": study,
            "model": name,
            "n": len(group),
            "positions": group.protein_position.nunique(),
            "mae": group[f"ae_{name}"].mean(),
            "gain_vs_iso": gain,
            "ci_low": ci[0],
            "ci_high": ci[1],
        })

pd.DataFrame(summary).to_csv(
    OUT / "phase31g_summary.csv",
    index=False,
)

print("\nPASS: Phase 31G completed.")
print("Bihler TRAIN only; no protected outcomes accessed.")
print("Results:", OUT)
print(
    "Post-treatment maturation is explanatory,"
    " not a prospective prediction feature."
)
print(
    "Bootstrap is conditional on existing folds and"
    " chosen models, not independent validation."
)
