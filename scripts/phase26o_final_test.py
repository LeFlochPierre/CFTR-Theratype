#!/usr/bin/env python3
"""Phase 26O: precommitted, one-time held-out VTI test evaluation.

--prepare: train on TRAIN+VAL, predict TEST using metadata only.
--score: open TEST assay outcomes once and score frozen predictions.
No post-test model selection or tuning.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/biology/phase26o"
SPLITS = ROOT / "data/processed/bihler_2026_transfer_splits.parquet"
ASSAYS = ROOT / "data/processed/bihler_2026_vti.parquet"
VAL_SCORES = ROOT / "results/biology/phase26n/phase26n_validation_scores.csv"
VAL_MANIFEST = ROOT / "results/biology/phase26n/phase26n_lock_manifest.json"

PRED = OUT / "phase26o_TEST_predictions_LOCKED.csv"
LOCK = OUT / "phase26o_lock_manifest.json"
SCORES = OUT / "phase26o_test_scores.csv"
GAINS = OUT / "phase26o_test_gains.csv"
SUBGROUPS = OUT / "phase26o_test_subgroups.csv"
PER_VARIANT = OUT / "phase26o_test_scored_variants.csv"

SCRIPT = Path(__file__).resolve()
SPEC = (2, 5.0, 180)

PRED_COLS = [
    "corrector_components",
    "acute_components",
    "total_components",
    "total_direct"
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partitions():
    columns = [
        "variant_key", "protein_position", "transfer_split",
        "study_cohort", "variant", "variant_type", "hgvs_protein"
    ]

    d = pd.read_parquet(SPLITS, columns=columns)
    d["variant_key"] = d.variant_key.astype(str)

    if d.variant_key.duplicated().any():
        raise ValueError("Duplicate frozen variant identities")

    parts = {
        name: d.loc[d.transfer_split.eq(name)]
        .sort_values("variant_key")
        .reset_index(drop=True)
        for name in ("train", "val", "test")
    }

    expected = {
        "train": (210, 168),
        "val": (52, 42),
        "test": (33, 26)
    }

    for name, x in parts.items():
        got = (len(x), x.protein_position.nunique())
        if got != expected[name]:
            raise ValueError(
                f"Partition {name} changed: {got}"
            )

    names = list(parts)

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(parts[a].protein_position) & set(
                parts[b].protein_position
            ):
                raise ValueError(
                    f"Position leakage: {a} vs {b}"
                )

    return parts


def feature_matrix(meta):
    from cftr_theratype.benchmark import make_features
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.splits import parse_substitution
    from cftr_theratype import structure_benchmark as structure

    meta = meta.reset_index(drop=True).copy()

    seq = bihler_2024_assay_sequence()
    parsed = [
        parse_substitution(row, seq)
        for _, row in meta.iterrows()
    ]

    bad = [
        str(meta.loc[i, "variant_key"])
        for i, p in enumerate(parsed)
        if p[-1] != "valid"
        or int(p[0]) != int(meta.loc[i, "protein_position"])
    ]

    if bad:
        raise ValueError(
            f"Mutation/reference mismatch: {bad[:5]}"
        )

    meta["reference_aa"] = [p[1] for p in parsed]
    meta["alternate_aa"] = [p[2] for p in parsed]

    x68, _ = make_features(meta, seq)
    x68 = np.asarray(x68, float)

    if (
        x68.shape != (len(meta), 68)
        or not np.isfinite(x68).all()
    ):
        raise ValueError("Invalid sequence features")

    table = structure.read_feature_table(
        ROOT / "results/structure/phase15b_position_features.csv",
        ROOT / "results/structure/phase15b_feature_summary.json",
        ROOT
    )

    x12 = np.asarray(
        structure.structural_matrix(
            meta.protein_position.to_numpy(int), table
        ),
        float
    )

    if x12.shape != (len(meta), 12):
        raise ValueError("Invalid structural features")

    x80 = np.column_stack([x68, x12])
    x80[np.isinf(x80)] = np.nan

    return x80


def allowed_labels(keys, partition):
    keys = set(map(str, keys))

    # Filter strictly to the explicitly permitted split.
    d = pd.read_parquet(
        ASSAYS,
        columns=[
            "variant_key",
            "corrector_rescue_pp",
            "vti_rescue_pp"
        ],
        filters=[
            ("variant_key", "in", sorted(keys))
        ]
    )

    d["variant_key"] = d.variant_key.astype(str)

    if (
        len(d) != len(keys)
        or d.variant_key.duplicated().any()
        or set(d.variant_key) != keys
    ):
        raise ValueError(
            f"{partition}: label membership mismatch"
        )

    for col in ("corrector_rescue_pp", "vti_rescue_pp"):
        d[col] = pd.to_numeric(d[col], errors="raise")

        if not np.isfinite(d[col].to_numpy(float)).all():
            raise ValueError(
                f"Nonfinite {partition} labels: {col}"
            )

    return d


def validate_selection():
    if not VAL_SCORES.exists() or not VAL_MANIFEST.exists():
        raise FileNotFoundError(
            "Phase26N validation records missing"
        )

    s = pd.read_csv(VAL_SCORES)

    def value(target, model):
        q = s.loc[
            s.target.eq(target) & s.model.eq(model),
            "position_mae_pp"
        ]

        if len(q) != 1:
            raise ValueError(
                f"Missing validation model: {model}"
            )

        return float(q.iloc[0])

    cv = value("Corrector", "corrector_vti_only")
    ce = value("Corrector", "corrector_eti_assisted")
    tv = value("Total", "total_vti_components")
    te = value("Total", "total_eti_components")
    td = value("Total", "total_vti_direct")

    if not (cv < ce and tv < te and tv < td):
        raise ValueError(
            "Unexpected validation ranking"
        )

    if not np.isclose(tv, 15.6107, atol=0.003):
        raise ValueError(
            "Validation results do not match Phase26N"
        )

    return {
        "corrector_vti_only_mae": cv,
        "corrector_eti_assisted_mae": ce,
        "total_vti_components_mae": tv,
        "total_eti_components_mae": te,
        "total_vti_direct_mae": td
    }


def prepare():
    if any(
        p.exists()
        for p in (
            PRED, LOCK, SCORES, GAINS,
            SUBGROUPS, PER_VARIANT
        )
    ):
        raise FileExistsError(
            "Final test already prepared or scored"
        )

    from joblib import dump
    import phase26d_corrector_bottleneck as previous

    decision = validate_selection()
    parts = partitions()

    train, mats, _, _, _ = previous.load_train()

    if set(train.variant_key.astype(str)) != set(
        parts["train"].variant_key
    ):
        raise ValueError("Training membership mismatch")

    Xtrain = np.asarray(
        mats["sequence_structure"], float
    ).copy()

    Xtrain[np.isinf(Xtrain)] = np.nan

    if Xtrain.shape != (210, 80):
        raise ValueError("Training feature shape changed")

    # TRAIN+VALIDATION labels are allowed now.
    # TEST labels are NOT accessed in prepare().
    vlabels = allowed_labels(
        parts["val"].variant_key,
        "validation"
    )

    val = parts["val"].merge(
        vlabels,
        on="variant_key",
        validate="one_to_one"
    )

    Xval = feature_matrix(val)
    Xtest = feature_matrix(parts["test"])

    # Check feature parity before combining datasets.
    # Preserve the frozen loader's row alignment.
    Xreconstructed = feature_matrix(train)

    if not np.allclose(
        Xreconstructed,
        Xtrain,
        rtol=0,
        atol=1e-8,
        equal_nan=True
    ):
        raise ValueError(
            "Training feature order/parity mismatch"
        )

    X = np.vstack([Xtrain, Xval])

    yc = np.r_[
        train.corrector_rescue_pp.to_numpy(float),
        val.corrector_rescue_pp.to_numpy(float)
    ]

    yt = np.r_[
        train.vti_rescue_pp.to_numpy(float),
        val.vti_rescue_pp.to_numpy(float)
    ]

    ya = yt - yc

    if X.shape != (262, 80):
        raise ValueError("Expected 262 x 80 final features")

    if not all(
        np.isfinite(y).all()
        for y in (yc, ya, yt)
    ):
        raise ValueError("Nonfinite final training outcomes")

    # Frozen model selection: no hyperparameter search.
    corrector = previous.model(SPEC)
    corrector.fit(
        np.column_stack([X, np.ones(len(X))]),
        yc
    )

    acute = previous.model(SPEC)
    acute.fit(X, ya)

    direct = previous.model(SPEC)
    direct.fit(X, yt)

    pred = parts["test"][
        [
            "variant_key", "protein_position",
            "study_cohort", "hgvs_protein"
        ]
    ].copy()

    pred["corrector_components"] = corrector.predict(
        np.column_stack([
            Xtest,
            np.ones(len(Xtest))
        ])
    )

    pred["acute_components"] = acute.predict(Xtest)

    pred["total_components"] = (
        pred.corrector_components
        + pred.acute_components
    )

    pred["total_direct"] = direct.predict(Xtest)

    if not np.isfinite(
        pred[PRED_COLS].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("Nonfinite TEST predictions")

    OUT.mkdir(parents=True, exist_ok=True)

    models = {
        "corrector": corrector,
        "acute": acute,
        "direct_total": direct
    }

    model_hashes = {}

    for name, model in models.items():
        path = OUT / f"phase26o_final_{name}.joblib"

        if path.exists():
            raise FileExistsError(path)

        dump(model, path)
        model_hashes[name] = sha(path)

    pred.to_csv(PRED, index=False)

    manifest = {
        "phase": "26O_FINAL_TEST_PRECOMMITTED",
        "model": "VTI-only components; train+validation refit",
        "model_spec": list(SPEC),
        "training_variants": 262,
        "training_positions": 210,
        "test_variants": 33,
        "test_positions": 26,
        "primary_endpoint": "total VTI rescue",
        "primary_metric": "position_weighted_MAE_pp",
        "primary_prediction": "total_components",
        "secondary_predictions": [
            "corrector_components",
            "total_direct"
        ],
        "validation_decision": decision,
        "test_outcomes_accessed": False,
        "script_sha256": sha(SCRIPT),
        "split_sha256": sha(SPLITS),
        "predictions_sha256": sha(PRED),
        "model_sha256": model_hashes
    }

    LOCK.write_text(
        json.dumps(
            manifest, indent=2, sort_keys=True
        ) + "\n"
    )

    print("=" * 95)
    print("PASS: PHASE26O FINAL TEST PREDICTIONS LOCKED")
    print("=" * 95)
    print("TRAIN + VAL: 262 variants / 210 positions")
    print("TEST: 33 variants / 26 positions")
    print("Selected model: VTI-only corrector + acute")
    print("Validation outcomes: used for final refit")
    print("TEST outcomes: NOT OPENED")
    print("Prediction SHA256:", sha(PRED))
    print("Predictions:", PRED)
    print("Manifest:", LOCK)


def position_errors(d, target, model):
    q = d[["protein_position"]].copy()

    q["ae"] = np.abs(
        d[target].to_numpy(float)
        - d[model].to_numpy(float)
    )

    return q.groupby("protein_position").ae.mean()


def metrics(d, target, model):
    residual = (
        d[model].to_numpy(float)
        - d[target].to_numpy(float)
    )

    return {
        "position_mae_pp": float(
            position_errors(d, target, model).mean()
        ),
        "variant_mae_pp": float(
            np.abs(residual).mean()
        ),
        "rmse_pp": float(
            np.sqrt(np.mean(residual ** 2))
        ),
        "bias_pp": float(residual.mean())
    }


def paired_bootstrap(
    d, target, candidate, reference, nboot
):
    a = position_errors(d, target, candidate)
    b = position_errors(d, target, reference)

    gains = (b - a).to_numpy(float)

    rng = np.random.default_rng(26110)

    samples = rng.choice(
        gains,
        size=(nboot, len(gains)),
        replace=True
    ).mean(axis=1)

    return {
        "candidate": candidate,
        "reference": reference,
        "gain_pp": float(gains.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "positions": len(gains)
    }


def score(nboot):
    if not PRED.exists() or not LOCK.exists():
        raise FileNotFoundError("Run --prepare first")

    if any(
        p.exists()
        for p in (SCORES, GAINS, SUBGROUPS, PER_VARIANT)
    ):
        raise FileExistsError(
            "Test already scored; refusing overwrite"
        )

    manifest = json.loads(LOCK.read_text())

    if (
        sha(SCRIPT) != manifest["script_sha256"]
        or sha(SPLITS) != manifest["split_sha256"]
        or sha(PRED) != manifest["predictions_sha256"]
    ):
        raise ValueError(
            "Frozen prediction integrity check failed"
        )

    for name, digest in manifest["model_sha256"].items():
        path = OUT / f"phase26o_final_{name}.joblib"

        if sha(path) != digest:
            raise ValueError(
                f"Final model changed: {name}"
            )

    parts = partitions()

    pred = pd.read_csv(
        PRED, dtype={"variant_key": str}
    )

    if (
        len(pred) != 33
        or pred.variant_key.duplicated().any()
        or set(pred.variant_key)
        != set(parts["test"].variant_key)
    ):
        raise ValueError("TEST membership mismatch")

    # The ONLY TEST outcome read in this script.
    # It occurs exclusively in --score mode.
    labels = allowed_labels(
        parts["test"].variant_key,
        "TEST"
    )

    d = pred.merge(
        labels,
        on="variant_key",
        validate="one_to_one"
    )

    d["observed_corrector"] = d.corrector_rescue_pp
    d["observed_total"] = d.vti_rescue_pp

    d["observed_acute"] = (
        d.observed_total - d.observed_corrector
    )

    tasks = [
        (
            "Corrector",
            "observed_corrector",
            "corrector_components"
        ),
        (
            "Acute",
            "observed_acute",
            "acute_components"
        ),
        (
            "Total",
            "observed_total",
            "total_components"
        ),
        (
            "Total",
            "observed_total",
            "total_direct"
        )
    ]

    results = []
    subgroups = []

    for target_name, target, model in tasks:
        results.append({
            "target": target_name,
            "model": model,
            "variants": len(d),
            "positions": d.protein_position.nunique(),
            **metrics(d, target, model)
        })

        masks = {
            "ALL": np.ones(len(d), bool),
            "observed_low_lt10": (
                d[target].to_numpy(float) < 10
            ),
            "observed_high_gt40": (
                d[target].to_numpy(float) > 40
            )
        }

        for cohort in d.study_cohort.unique():
            masks[f"cohort_{cohort}"] = (
                d.study_cohort.eq(cohort).to_numpy()
            )

        for name, mask in masks.items():
            if not mask.any():
                continue

            q = d.loc[mask]

            subgroups.append({
                "target": target_name,
                "model": model,
                "subgroup": name,
                "variants": len(q),
                "positions": q.protein_position.nunique(),
                **metrics(q, target, model)
            })

    results = pd.DataFrame(results)

    gains = pd.DataFrame([
        paired_bootstrap(
            d,
            "observed_total",
            "total_components",
            "total_direct",
            nboot
        )
    ])

    subgroups = pd.DataFrame(subgroups)

    results.to_csv(SCORES, index=False)
    gains.to_csv(GAINS, index=False)
    subgroups.to_csv(SUBGROUPS, index=False)
    d.to_csv(PER_VARIANT, index=False)

    (
        OUT / "phase26o_TEST_SCORED_LOCK.json"
    ).write_text(
        json.dumps({
            "test_scored_once": True,
            "test_variants": 33,
            "test_positions": 26,
            "predictions_sha256":
                manifest["predictions_sha256"]
        }, indent=2, sort_keys=True) + "\n"
    )

    print("=" * 100)
    print(
        "PHASE 26O | FINAL INDEPENDENT VTI TEST"
    )
    print("=" * 100)
    print("33 variants / 26 held-out positions")

    print("\nA. FROZEN PREDICTION METRICS")
    print(results.round(4).to_string(index=False))

    print("\nB. COMPONENTS VS DIRECT TOTAL")
    print("Positive bootstrap gain = components better")
    print(gains.round(4).to_string(index=False))

    print("\nC. EXPLORATORY SUBGROUPS")
    print(subgroups.round(3).to_string(index=False))

    print("\nPASS: TEST SCORED ONCE")
    print("No further tuning on test outcomes.")
    print("Saved:", OUT)


def self_test():
    q = pd.DataFrame({
        "protein_position": [1, 1, 2, 3],
        "y": [1., 3., 5., 7.],
        "candidate": [2., 2., 5., 7.],
        "reference": [1., 3., 6., 6.]
    })

    assert np.isclose(
        metrics(q, "y", "candidate")["position_mae_pp"],
        1 / 3
    )

    assert np.isclose(
        paired_bootstrap(
            q, "y", "candidate", "reference", 500
        )["gain_pp"],
        1 / 3
    )

    print(
        "PASS: position MAE and paired-bootstrap self-test"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    mode = ap.add_mutually_exclusive_group(required=True)

    mode.add_argument("--self-test", action="store_true")
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--score", action="store_true")

    ap.add_argument(
        "--bootstraps", type=int, default=10000
    )

    args = ap.parse_args()

    if args.self_test:
        self_test()
    elif args.prepare:
        prepare()
    else:
        score(args.bootstraps)
