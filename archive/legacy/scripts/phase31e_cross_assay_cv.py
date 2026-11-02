"""Phase 31E: Position-held-out cross-assay calibration.

Compare Bihler 2024 ETI measurements against Castellani
P289 and U032 responses.

TRAIN-only exploratory analysis.
No protected validation/test outcomes accessed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    cohen_kappa_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SOURCE = Path(
    "results/biology/phase31c/phase31c_training_overlap.csv"
)
OUT = Path("results/biology/phase31e")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    "train_median",
    "identity",
    "median_shift",
    "ridge_response",
    "ridge_response_baseline",
    "isotonic_response",
]

assert SOURCE.exists(), f"Missing input: {SOURCE}"

raw = pd.read_csv(SOURCE)

required = {
    "study",
    "aa_code",
    "protein_position",
    "bihler_eti",
    "bihler_baseline",
    "eti_rescue_pp",
    "castellani_rescue_263",
}

missing = required - set(raw.columns)
assert not missing, f"Missing columns: {missing}"

assert not raw.duplicated(["study", "aa_code"]).any()
assert set(raw.study.unique()) <= {"P289", "U032"}

all_predictions = []
all_scores = []


for study, sub in raw.groupby("study", sort=True):

    data = sub.dropna(
        subset=[
            "protein_position",
            "bihler_eti",
            "bihler_baseline",
            "eti_rescue_pp",
            "castellani_rescue_263",
        ]
    ).copy().reset_index(drop=True)

    groups = data.protein_position.astype(int).to_numpy()
    x = data.bihler_eti.to_numpy(dtype=float)
    baseline = data.bihler_baseline.to_numpy(dtype=float)

    assert len(np.unique(groups)) >= 5
    assert np.isfinite(x).all()
    assert np.isfinite(baseline).all()

    for scale, target in [
        ("published_native", "eti_rescue_pp"),
        (
            "raw_current_263_sensitivity",
            "castellani_rescue_263",
        ),
    ]:

        y = data[target].to_numpy(dtype=float)
        assert np.isfinite(y).all()

        preds = {
            model: np.full(len(data), np.nan)
            for model in MODELS
        }

        fold_ids = np.full(len(data), -1, dtype=int)

        splitter = GroupKFold(n_splits=5)

        for fold, (train, test) in enumerate(
            splitter.split(x, y, groups=groups),
            start=1,
        ):

            assert not (
                set(groups[train]) & set(groups[test])
            )

            fold_ids[test] = fold

            # 1. Train-fold median baseline
            preds["train_median"][test] = np.median(
                y[train]
            )

            # 2. Direct transfer: Bihler -> Castellani
            preds["identity"][test] = x[test]

            # 3. Median offset correction
            shift = np.median(
                y[train] - x[train]
            )

            preds["median_shift"][test] = (
                x[test] + shift
            )

            # 4. Ridge calibration, ETI response only
            lin = make_pipeline(
                StandardScaler(),
                Ridge(alpha=1.0),
            )

            lin.fit(
                x[train].reshape(-1, 1),
                y[train],
            )

            preds["ridge_response"][test] = (
                lin.predict(
                    x[test].reshape(-1, 1)
                )
            )

            # 5. Ridge with measured baseline function
            fit2 = np.column_stack([
                x[train],
                baseline[train],
            ])

            test2 = np.column_stack([
                x[test],
                baseline[test],
            ])

            lin2 = make_pipeline(
                StandardScaler(),
                Ridge(alpha=1.0),
            )

            lin2.fit(fit2, y[train])

            preds["ridge_response_baseline"][test] = (
                lin2.predict(test2)
            )

            # 6. Monotonic nonlinear calibration
            iso = IsotonicRegression(
                increasing=True,
                out_of_bounds="clip",
            )

            iso.fit(x[train], y[train])

            preds["isotonic_response"][test] = (
                iso.predict(x[test])
            )

        assert (fold_ids > 0).all()

        for model, p in preds.items():

            assert np.isfinite(p).all()

            mae = float(
                np.mean(np.abs(y - p))
            )

            rmse = float(
                np.sqrt(np.mean((y - p) ** 2))
            )

            rho = pd.Series(y).corr(
                pd.Series(p),
                method="spearman",
            )

            result = {
                "study": study,
                "target_scale": scale,
                "model": model,
                "n": len(y),
                "positions": len(np.unique(groups)),
                "mae_pp": mae,
                "rmse_pp": rmse,
                "spearman": rho,
            }

            if scale == "published_native":

                true_resp = y >= 10
                pred_resp = p >= 10

                result.update({
                    "accuracy10": float(
                        np.mean(true_resp == pred_resp)
                    ),
                    "balanced_accuracy10": float(
                        balanced_accuracy_score(
                            true_resp,
                            pred_resp,
                        )
                    ),
                    "kappa10": float(
                        cohen_kappa_score(
                            true_resp,
                            pred_resp,
                        )
                    ),
                })

            all_scores.append(result)

            for i in range(len(y)):

                all_predictions.append({
                    "study": study,
                    "target_scale": scale,
                    "model": model,
                    "aa_code": data.at[i, "aa_code"],
                    "protein_position": int(groups[i]),
                    "fold": int(fold_ids[i]),
                    "bihler_eti_pp": float(x[i]),
                    "bihler_baseline_pp": float(
                        baseline[i]
                    ),
                    "target_pp": float(y[i]),
                    "prediction_pp": float(p[i]),
                })


summary = pd.DataFrame(all_scores).sort_values(
    ["study", "target_scale", "mae_pp"]
)

predictions = pd.DataFrame(all_predictions)

summary.to_csv(
    OUT / "phase31e_group_cv_scores.csv",
    index=False,
)

predictions.to_csv(
    OUT / "phase31e_group_cv_predictions.csv",
    index=False,
)


print("=" * 88)
print("PHASE 31E | GROUPED CROSS-ASSAY CALIBRATION")
print("5-fold by protein position")
print("Bihler ETI TRAIN only")
print("=" * 88)

for (study, scale), score in summary.groupby(
    ["study", "target_scale"],
    sort=True,
):

    print(
        f"\n{study} / {scale}"
        f" | n={int(score.n.iloc[0])}"
        f" | positions={int(score.positions.iloc[0])}"
    )

    cols = [
        "model",
        "mae_pp",
        "rmse_pp",
        "spearman",
    ]

    if scale == "published_native":
        cols += [
            "balanced_accuracy10",
            "kappa10",
        ]

    print(
        score[cols]
        .round(3)
        .to_string(index=False)
    )


print("\nPASS: Phase 31E complete.")
print("Position-held-out cross-assay calibration.")
print("TRAIN-only exploratory evaluation.")
print("No protected Bihler outcomes accessed.")
print(
    "Common 263 denominator is sensitivity analysis, "
    "not validated assay harmonisation."
)
print("Results saved to:", OUT)
