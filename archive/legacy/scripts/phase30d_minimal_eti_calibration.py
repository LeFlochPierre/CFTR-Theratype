#!/usr/bin/env python3
"""Phase 30D: Minimal ETI-to-VTI calibration.

TRAIN only: 66 matched variants / 56 protein positions.
Uses original 2x5 grouped folds.
Reuses the Phase30C ETI-only XGBoost OOF predictions.
No validation/test labels or new hyperparameter searches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/biology/phase30d"

OLD_OOF = (
    ROOT
    / "results/biology/phase30c/phase30c_oof.csv"
)

TARGETS = {
    "corrector": "observed_corrector",
    "total_rescue": "observed_total_rescue",
}

FEATURES = {
    "eti_corrector": ["eti_corrector_pp"],
    "eti_total": ["eti_total_pp"],
    "eti_acute": ["eti_acute_pp"],
    "eti_baseline": ["eti_baseline_fsk"],
    "eti_3": [
        "eti_corrector_pp",
        "eti_acute_pp",
        "eti_baseline_fsk",
    ],
    "eti_total_vtibase": [
        "eti_total_pp",
        "observed_baseline",
    ],
    "eti_3_vtibase": [
        "eti_corrector_pp",
        "eti_acute_pp",
        "eti_baseline_fsk",
        "observed_baseline",
    ],
}

MODELS = [
    "train_median",
    "direct_identity",
    "direct_median_shift",
    "eti_corrector_ridge",
    "eti_total_ridge",
    "eti_acute_ridge",
    "eti_baseline_ridge",
    "eti_3_ridge",
    "eti_total_vtibase_ridge",
    "eti_3_vtibase_ridge",
    "eti_3_xgb_prior",
]

PRIMARY = {
    "corrector": [
        ("direct_median_shift", "direct_identity"),
        ("eti_corrector_ridge", "direct_identity"),
        ("eti_corrector_ridge", "train_median"),
        ("eti_3_ridge", "eti_corrector_ridge"),
        ("eti_3_xgb_prior", "eti_corrector_ridge"),
        ("eti_3_xgb_prior", "eti_3_ridge"),
        ("eti_3_vtibase_ridge", "eti_3_ridge"),
    ],
    "total_rescue": [
        ("direct_median_shift", "direct_identity"),
        ("eti_total_ridge", "direct_identity"),
        ("eti_total_ridge", "train_median"),
        ("eti_3_ridge", "eti_total_ridge"),
        ("eti_3_xgb_prior", "eti_total_ridge"),
        ("eti_3_xgb_prior", "eti_3_ridge"),
        ("eti_3_vtibase_ridge", "eti_3_ridge"),
    ],
}

ID = [
    "repeat",
    "fold",
    "variant_key",
    "protein_position",
    "target",
]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ridge_predict(xtrain, ytrain, xtest):
    """Fixed regularisation; TRAIN-only scaling."""

    model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )

    model.fit(
        np.asarray(xtrain, float),
        np.asarray(ytrain, float),
    )

    return np.asarray(
        model.predict(np.asarray(xtest, float)),
        float,
    )


def position_summary(df, column):
    """Average duplicate variants and CV repeats per position."""

    return (
        df.groupby(["repeat", "protein_position"])[column]
        .mean()
        .groupby("protein_position")
        .mean()
    )


def load_prior_reference(path, matched):
    old = pd.read_csv(
        path,
        dtype={"variant_key": str},
    )

    required = set(
        ID + [
            "observed",
            "pred_eti_only",
            "pred_baseline_only",
        ]
    )

    if not required.issubset(old.columns):
        raise ValueError(
            f"Phase30C missing: {sorted(required - set(old.columns))}"
        )

    if len(old) != 396 or old.duplicated(ID).any():
        raise ValueError("Invalid Phase30C OOF rows")

    if set(old.variant_key) != set(matched.variant_key):
        raise ValueError("Matched cohort changed")

    if set(old.repeat) != {1, 2}:
        raise ValueError("Incorrect repeat IDs")

    if old.protein_position.nunique() != 56:
        raise ValueError("Incorrect number of protein positions")

    lookup = matched.set_index("variant_key")

    for target, source in TARGETS.items():
        subset = old.loc[old.target == target]

        for _, row in subset.iterrows():
            expected = float(
                lookup.loc[row.variant_key, source]
            )

            if not np.isclose(
                row.observed,
                expected,
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError(
                    f"Phase30C outcome mismatch: {target}"
                )

    return old


def evaluate(matched, folds, previous):
    m = matched

    if (
        len(m) != 66
        or m.variant_key.nunique() != 66
        or m.protein_position.nunique() != 56
    ):
        raise ValueError("Expected 66 variants / 56 positions")

    if len(folds) != 10:
        raise ValueError("Expected original 2x5 folds")

    needed = (
        list(TARGETS.values())
        + [
            "observed_baseline",
            "observed_treated_absolute",
        ]
        + sorted(set(sum(FEATURES.values(), [])))
    )

    for col in needed:
        if (
            col not in m
            or not np.isfinite(m[col].to_numpy(float)).all()
        ):
            raise ValueError(f"Missing or nonfinite: {col}")

    if not np.allclose(
        m.observed_total_rescue + m.observed_baseline,
        m.observed_treated_absolute,
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError("VTI target arithmetic mismatch")

    if not np.allclose(
        m.eti_total_pp,
        m.eti_corrector_pp + m.eti_acute_pp,
        atol=0.25,
        rtol=0,
    ):
        raise ValueError("ETI component arithmetic mismatch")

    if (
        len(previous) != 396
        or previous.duplicated(ID).any()
        or set(previous.variant_key) != set(m.variant_key)
    ):
        raise ValueError("Invalid prior benchmark")

    reference = previous.set_index(ID)

    keys = m.variant_key.astype(str).to_numpy()
    positions = m.protein_position.to_numpy(int)

    matrices = {
        name: m[columns].to_numpy(float)
        for name, columns in FEATURES.items()
    }

    targets = {
        name: m[column].to_numpy(float)
        for name, column in TARGETS.items()
    }

    results = []
    qc = []

    seen = {1: set(), 2: set()}

    for i, (repeat, fold, train_keys, test_keys) in enumerate(
        folds, 1
    ):
        train = np.flatnonzero(
            np.isin(keys, list(train_keys))
        )

        test = np.flatnonzero(
            np.isin(keys, list(test_keys))
        )

        if (
            len(train) == 0
            or len(test) == 0
            or len(train) + len(test) != 66
        ):
            raise ValueError("Incomplete matched fold")

        if set(positions[train]) & set(positions[test]):
            raise ValueError("PROTEIN POSITION LEAKAGE")

        if seen[repeat] & set(keys[test]):
            raise ValueError("Repeated held-out variant")

        seen[repeat].update(keys[test])

        qc.append({
            "repeat": int(repeat),
            "fold": int(fold),
            "train_n": len(train),
            "heldout_n": len(test),
            "train_positions": len(set(positions[train])),
            "heldout_positions": len(set(positions[test])),
        })

        for target, y in targets.items():
            direct_feature = (
                "eti_corrector"
                if target == "corrector"
                else "eti_total"
            )

            direct = matrices[direct_feature][:, 0]

            predictions = {
                "train_median": np.full(
                    len(test),
                    np.median(y[train]),
                ),
                "direct_identity": direct[test],
                "direct_median_shift": (
                    direct[test]
                    + np.median(y[train] - direct[train])
                ),
            }

            for name, matrix in matrices.items():
                predictions[name + "_ridge"] = ridge_predict(
                    matrix[train],
                    y[train],
                    matrix[test],
                )

            for j, idx in enumerate(test):
                old_key = (
                    int(repeat),
                    int(fold),
                    str(keys[idx]),
                    int(positions[idx]),
                    target,
                )

                if old_key not in reference.index:
                    raise ValueError(
                        f"Missing Phase30C prediction: {old_key}"
                    )

                old = reference.loc[old_key]

                if not np.isclose(
                    old.observed,
                    y[idx],
                    atol=1e-6,
                    rtol=0,
                ):
                    raise ValueError("Frozen outcome mismatch")

                row_predictions = {
                    name: float(values[j])
                    for name, values in predictions.items()
                }

                row_predictions["eti_3_xgb_prior"] = float(
                    old.pred_eti_only
                )

                if set(row_predictions) != set(MODELS):
                    raise ValueError(
                        "Unexpected model configurations"
                    )

                if not np.isfinite(
                    list(row_predictions.values())
                ).all():
                    raise ValueError("Nonfinite predictions")

                results.append({
                    "repeat": int(repeat),
                    "fold": int(fold),
                    "variant_key": str(keys[idx]),
                    "protein_position": int(positions[idx]),
                    "target": target,
                    "observed": float(y[idx]),
                    "observed_baseline": float(
                        m.observed_baseline.iloc[idx]
                    ),
                    **{
                        "pred_" + name: value
                        for name, value in row_predictions.items()
                    },
                })

        print(
            f"Fold {i:02d}/10 "
            f"repeat={repeat} fold={fold} "
            f"train={len(train)} heldout={len(test)}",
            flush=True,
        )

    if (
        seen[1] != set(keys)
        or seen[2] != set(keys)
    ):
        raise ValueError("Incomplete OOF coverage")

    oof = pd.DataFrame(results)

    if len(oof) != 264 or oof.duplicated(ID).any():
        raise ValueError("Expected 264 OOF rows")

    return oof, pd.DataFrame(qc)


def assess(oof, bootstraps):
    metrics = []
    gains = []
    calibration = []
    regimes = []

    for target in TARGETS:
        frame = oof.loc[oof.target == target].copy()
        truth = frame.observed.to_numpy(float)

        for model in MODELS:
            prediction = frame[
                "pred_" + model
            ].to_numpy(float)

            errors = np.abs(truth - prediction)

            frame["error"] = errors

            position_mae = float(
                position_summary(frame, "error").mean()
            )

            averaged = frame.groupby("variant_key")[
                ["observed", "pred_" + model]
            ].mean()

            metrics.append({
                "target": target,
                "model": model,
                "position_mae_pp": position_mae,
                "variant_mae_pp": float(errors.mean()),
                "pred_sd_pp": float(
                    averaged["pred_" + model].std(ddof=0)
                ),
                "spearman": float(
                    averaged["pred_" + model].corr(
                        averaged.observed,
                        method="spearman",
                    )
                ),
            })

        for i, (candidate, reference) in enumerate(
            PRIMARY[target]
        ):
            temp = frame[
                ["repeat", "protein_position"]
            ].copy()

            temp["gain"] = (
                np.abs(
                    truth
                    - frame["pred_" + reference].to_numpy(float)
                )
                - np.abs(
                    truth
                    - frame["pred_" + candidate].to_numpy(float)
                )
            )

            values = position_summary(
                temp,
                "gain",
            ).to_numpy(float)

            rng = np.random.default_rng(
                4400 + 50 * list(TARGETS).index(target) + i
            )

            indexes = rng.integers(
                0,
                len(values),
                size=(bootstraps, len(values)),
            )

            samples = values[indexes].mean(axis=1)

            gains.append({
                "target": target,
                "candidate": candidate,
                "reference": reference,
                "gain_pp": float(values.mean()),
                "ci95_low": float(
                    np.quantile(samples, 0.025)
                ),
                "ci95_high": float(
                    np.quantile(samples, 0.975)
                ),
                "positions": len(values),
            })

        selected_models = [
            "direct_identity",
            "direct_median_shift",
            (
                "eti_corrector_ridge"
                if target == "corrector"
                else "eti_total_ridge"
            ),
            "eti_3_ridge",
            "eti_3_xgb_prior",
        ]

        for model in selected_models:
            prediction = frame.groupby("variant_key")[
                "pred_" + model
            ].mean()

            observed = frame.groupby("variant_key")[
                "observed"
            ].mean()

            slope = (
                float(np.polyfit(
                    prediction.to_numpy(float),
                    observed.to_numpy(float),
                    1,
                )[0])
                if prediction.std() > 1e-8
                else float("nan")
            )

            calibration.append({
                "target": target,
                "model": model,
                "mean_prediction_pp": float(
                    prediction.mean()
                ),
                "mean_truth_pp": float(
                    observed.mean()
                ),
                "signed_bias_pp": float(
                    (prediction - observed).mean()
                ),
                "calibration_slope_descriptive": slope,
            })

        corrector = oof.loc[
            oof.target == "corrector",
            ["variant_key", "observed"],
        ].drop_duplicates("variant_key")

        corrector = corrector.copy()

        corrector["regime"] = np.select(
            [
                corrector.observed < 10,
                corrector.observed > 40,
            ],
            ["low", "high"],
            default="middle",
        )

        frame = frame.merge(
            corrector[["variant_key", "regime"]],
            on="variant_key",
            validate="many_to_one",
        )

        for regime, subset in frame.groupby("regime"):
            for model in selected_models:
                temp = subset[
                    ["repeat", "protein_position"]
                ].copy()

                temp["error"] = np.abs(
                    subset.observed.to_numpy(float)
                    - subset["pred_" + model].to_numpy(float)
                )

                regimes.append({
                    "target": target,
                    "regime": regime,
                    "model": model,
                    "variants": subset.variant_key.nunique(),
                    "positions": subset.protein_position.nunique(),
                    "position_mae_pp": float(
                        position_summary(temp, "error").mean()
                    ),
                })

    return (
        pd.DataFrame(metrics),
        pd.DataFrame(gains),
        pd.DataFrame(calibration),
        pd.DataFrame(regimes),
    )


def self_test():
    rng = np.random.default_rng(30)

    positions = np.r_[
        np.arange(56),
        np.arange(10),
    ]

    keys = np.array([
        f"SYN{i:03d}"
        for i in range(66)
    ])

    ec = rng.uniform(2, 60, 66)
    ea = rng.uniform(0, 25, 66)
    eb = rng.uniform(0, 30, 66)

    baseline = eb + rng.normal(2, 1, 66)
    corrector = 0.75 * ec + rng.normal(0, 4, 66)
    total = 0.7 * (ec + ea) + rng.normal(0, 5, 66)

    m = pd.DataFrame({
        "variant_key": keys,
        "protein_position": positions,
        "observed_baseline": baseline,
        "observed_corrector": corrector,
        "observed_total_rescue": total,
        "observed_treated_absolute": baseline + total,
        "eti_corrector_pp": ec,
        "eti_acute_pp": ea,
        "eti_baseline_fsk": eb,
        "eti_total_pp": ec + ea,
    })

    folds = []

    for repeat in (1, 2):
        for fold in range(1, 6):
            mask = (
                (positions + (repeat - 1) * 3) % 5 + 1
            ) == fold

            folds.append((
                repeat,
                fold,
                set(keys[~mask]),
                set(keys[mask]),
            ))

    synthetic_reference = []

    for repeat, fold, _, heldout in folds:
        selected = np.flatnonzero(
            np.isin(keys, list(heldout))
        )

        for idx in selected:
            targets = {
                "corrector": "observed_corrector",
                "total_rescue": "observed_total_rescue",
                "treated_absolute": "observed_treated_absolute",
            }

            for target, col in targets.items():
                synthetic_reference.append({
                    "repeat": repeat,
                    "fold": fold,
                    "variant_key": keys[idx],
                    "protein_position": int(positions[idx]),
                    "target": target,
                    "observed": float(m[col].iloc[idx]),
                    "pred_eti_only": float(m[col].iloc[idx]) + 2,
                    "pred_baseline_only": float(m[col].iloc[idx]) + 3,
                })

    old = pd.DataFrame(synthetic_reference)

    oof, qc = evaluate(m, folds, old)

    metrics, gains, calibration, regimes = assess(
        oof,
        150,
    )

    assert len(oof) == 264
    assert len(qc) == 10
    assert len(metrics) == 22
    assert len(gains) == 14
    assert set(gains.positions) == {56}
    assert np.isfinite(metrics.position_mae_pp).all()

    total_oof = oof.loc[
        oof.target == "total_rescue"
    ].merge(
        m[["variant_key", "eti_total_pp"]],
        on="variant_key",
    )

    assert np.allclose(
        total_oof.pred_direct_identity,
        total_oof.eti_total_pp,
    )

    corrupted = m.copy()
    corrupted.loc[
        0, "observed_treated_absolute"
    ] += 1

    try:
        evaluate(corrupted, folds, old)
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Target arithmetic corruption not detected"
        )

    print(
        "PASS Phase30D synthetic tests: "
        "66 variants, 56 positions, grouped CV, "
        "identity and arithmetic QC"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--bootstraps",
        type=int,
        default=2000,
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.bootstraps < 100:
        parser.error("Require at least 100 bootstraps")

    self_test()

    if not OLD_OOF.is_file():
        raise FileNotFoundError(
            f"Run Phase30C first: {OLD_OOF}"
        )

    import phase30c_eti_incremental_ladder as prior

    m, _, folds, _ = prior.load_matched_data()

    old = load_prior_reference(
        OLD_OOF,
        m,
    )

    print("=" * 104)
    print("PHASE 30D | MINIMAL ETI-TO-VTI CALIBRATION")
    print("=" * 104)

    print(
        "66 matched TRAIN variants, 56 positions, "
        "original 2x5 position-disjoint folds."
    )
    print("No new XGBoost fits and no VAL/TEST labels.")
    print("Ridge alpha=10; scaling fitted on outer TRAIN only.")
    print(
        "ETI total = ETI corrector + ETI acute: "
        "these are not independent components."
    )

    print("\nA. LOCKED CALIBRATION EVALUATION")

    oof, qc = evaluate(
        m,
        folds,
        old,
    )

    metrics, gains, calibration, regimes = assess(
        oof,
        args.bootstraps,
    )

    print("\nB. POSITION-WEIGHTED OOF MAE AND SPEARMAN")
    print(metrics.round(4).to_string(index=False))

    print("\nC. PAIRED POSITION-CLUSTER GAINS")
    print("Positive gain means LOWER MAE.")
    print(gains.round(4).to_string(index=False))

    print("\nD. OOF CALIBRATION DIAGNOSTICS")
    print(calibration.round(4).to_string(index=False))

    print("\nE. MAE BY TRUE CORRECTOR REGIME")
    print("Diagnostic only; true regime never used for prediction.")
    print(regimes.round(3).to_string(index=False))

    print("\nF. POSITION-DISJOINT FOLD QC")
    print(qc.to_string(index=False))

    print("\nG. LIMITATIONS")
    print("Exploratory TRAIN analysis, not independent validation.")
    print("ETI/VTI cross-study assay calibration not established.")
    print("Bootstrap intervals are not comparison-adjusted.")
    print("No protected ETI/VTI outcomes accessed.")

    OUT.mkdir(parents=True, exist_ok=True)

    outputs = {
        "oof": oof,
        "metrics": metrics,
        "paired_gains": gains,
        "calibration": calibration,
        "regime_mae": regimes,
        "fold_qc": qc,
    }

    for name, frame in outputs.items():
        frame.to_csv(
            OUT / f"phase30d_{name}.csv",
            index=False,
        )

    provenance = {
        "prior_oof_sha256": sha256(OLD_OOF),
        "eti_source_sha256": sha256(prior.ETI_CSV),
        "phase30a_sha256": sha256(prior.PHASE30A),
        "n_variants": 66,
        "n_positions": 56,
        "cv": "Original repeated position-disjoint 2x5",
        "methods": (
            "Direct identity, median shift, Ridge alpha=10, "
            "frozen prior XGBoost ETI-only OOF"
        ),
        "status": "TRAIN-only exploratory analysis",
    }

    (
        OUT / "phase30d_provenance.json"
    ).write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print("\nPASS Phase30D; saved:", OUT)


if __name__ == "__main__":
    main()
