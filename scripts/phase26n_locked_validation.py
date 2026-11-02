#!/usr/bin/env python3
"""Phase 26N: one-time frozen VTI validation.

--prepare: train only; predict validation from metadata.
--score: read validation outcomes and score frozen predictions.
VTI test and protected ETI outcomes are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/biology/phase26n"
PRED = OUT / "phase26n_validation_predictions_LOCKED.csv"
MANIFEST = OUT / "phase26n_lock_manifest.json"
SCORES = OUT / "phase26n_validation_scores.csv"
GAINS = OUT / "phase26n_validation_gains.csv"
SUBGROUPS = OUT / "phase26n_validation_subgroups.csv"

SOURCE = Path(__file__).resolve()
SPLITS = ROOT / "data/processed/bihler_2026_transfer_splits.parquet"
VTI = ROOT / "data/processed/bihler_2026_vti.parquet"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_metadata():
    cols = [
        "variant_key", "protein_position", "transfer_split",
        "study_cohort", "variant", "variant_type", "hgvs_protein"
    ]

    meta = pd.read_parquet(SPLITS, columns=cols)
    meta["variant_key"] = meta.variant_key.astype(str)

    if meta.variant_key.duplicated().any():
        raise ValueError("Duplicate frozen variant_key")

    train = meta.loc[meta.transfer_split.eq("train")].copy()
    val = (
        meta.loc[meta.transfer_split.eq("val")]
        .sort_values("variant_key")
        .reset_index(drop=True)
    )
    test = meta.loc[meta.transfer_split.eq("test")].copy()

    counts = (
        len(train), train.protein_position.nunique(),
        len(val), val.protein_position.nunique(), len(test)
    )

    if counts != (210, 168, 52, 42, 33):
        raise ValueError(f"Frozen partitions changed: {counts}")

    parts = [train, val, test]
    for i in range(3):
        for j in range(i + 1, 3):
            if set(parts[i].protein_position) & set(parts[j].protein_position):
                raise ValueError("Protein-position overlap between partitions")

    return train, val, test


def validation_features(val):
    from cftr_theratype.benchmark import make_features
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.splits import parse_substitution
    from cftr_theratype import structure_benchmark as structure

    seq = bihler_2024_assay_sequence()
    parsed = [parse_substitution(row, seq) for _, row in val.iterrows()]

    bad = [
        (i, str(val.loc[i, "variant_key"]))
        for i, p in enumerate(parsed)
        if p[-1] != "valid"
        or int(p[0]) != int(val.loc[i, "protein_position"])
    ]

    if bad:
        raise ValueError(f"Validation reference mismatch: {bad[:5]}")

    q = val.copy()
    q["reference_aa"] = [p[1] for p in parsed]
    q["alternate_aa"] = [p[2] for p in parsed]

    x68, _ = make_features(q, seq)
    x68 = np.asarray(x68, float)

    if x68.shape != (52, 68) or not np.isfinite(x68).all():
        raise ValueError("Invalid validation sequence features")

    struct = structure.read_feature_table(
        ROOT / "results/structure/phase15b_position_features.csv",
        ROOT / "results/structure/phase15b_feature_summary.json",
        ROOT
    )

    x12 = np.asarray(
        structure.structural_matrix(
            q.protein_position.to_numpy(int), struct
        ),
        float
    )

    if x12.shape != (52, 12):
        raise ValueError("Invalid validation structural features")

    x80 = np.column_stack([x68, x12])
    x80[np.isinf(x80)] = np.nan

    return x80


def prepare():
    if any(p.exists() for p in (PRED, MANIFEST, SCORES)):
        raise FileExistsError(
            "Validation already prepared/scored. Refusing to overwrite lock."
        )

    train_meta, val, test = validate_metadata()

    import phase26h_auxiliary_corrector as h

    v, V80, e, E80, folds = h.load_data()

    if set(v.variant_key.astype(str)) != set(train_meta.variant_key):
        raise ValueError("VTI train membership mismatch")

    protected = (
        set(val.protein_position)
        | set(test.protein_position)
    )

    if set(e.protein_position) & protected:
        raise ValueError("ETI labels overlap protected VTI positions")

    yc = v.corrector_rescue_pp.to_numpy(float)
    yt = v.vti_rescue_pp.to_numpy(float)
    ya = yt - yc

    ye = e.corrector_rescue_pp.to_numpy(float)

    vp = v.protein_position.to_numpy(int)
    ep = e.protein_position.to_numpy(int)

    VX = np.column_stack([V80, np.ones(len(V80))])
    EX = np.column_stack([E80, np.zeros(len(E80))])

    # Choose ETI weight using TRAIN ONLY.
    weight, inner_mae = h.choose_weight(
        VX, yc, vp, EX, ye, ep
    )

    print(
        f"TRAIN-only ETI weight={weight}, "
        f"inner position MAE={inner_mae:.4f} pp"
    )

    # No validation outcomes are accessed here.
    Xval80 = validation_features(val)
    Xval = np.column_stack([Xval80, np.ones(len(Xval80))])

    # Same fixed model specification as Phase26H.
    vti_corrector = h.previous.model(h.SPEC)
    vti_corrector.fit(VX, yc)

    eti_corrector = h.fit_joint(
        VX, yc, EX, ye, weight
    )

    acute = h.previous.model(h.SPEC)
    acute.fit(V80, ya)

    direct_total = h.previous.model(h.SPEC)
    direct_total.fit(V80, yt)

    pred = val[
        ["variant_key", "protein_position", "study_cohort", "hgvs_protein"]
    ].copy()

    pred["corrector_vti_only"] = vti_corrector.predict(Xval)
    pred["corrector_eti_assisted"] = eti_corrector.predict(Xval)
    pred["acute_vti_only"] = acute.predict(Xval80)

    pred["total_vti_components"] = (
        pred.corrector_vti_only + pred.acute_vti_only
    )

    pred["total_eti_components"] = (
        pred.corrector_eti_assisted + pred.acute_vti_only
    )

    pred["total_vti_direct"] = direct_total.predict(Xval80)

    prediction_columns = [
        "corrector_vti_only",
        "corrector_eti_assisted",
        "acute_vti_only",
        "total_vti_components",
        "total_eti_components",
        "total_vti_direct",
    ]

    prediction_values = pred[prediction_columns].to_numpy(
        dtype=np.float64
    )

    if not np.isfinite(prediction_values).all():
        raise ValueError("Nonfinite validation predictions")

    OUT.mkdir(parents=True, exist_ok=True)
    pred.to_csv(PRED, index=False)

    manifest = {
        "phase": "26N_validation_precommitted",
        "n_vti_train": 210,
        "n_eti_train": len(e),
        "n_validation": 52,
        "n_validation_positions": 42,
        "n_test_untouched": 33,
        "model_spec": str(h.SPEC),
        "selected_eti_weight": float(weight),
        "inner_train_mae_pp": float(inner_mae),
        "primary_metric": "position_weighted_MAE_pp",
        "primary_corrector": [
            "corrector_eti_assisted", "corrector_vti_only"
        ],
        "primary_total": [
            "total_eti_components", "total_vti_components"
        ],
        "additional_reference": "total_vti_direct",
        "validation_outcomes_accessed": False,
        "script_sha256": sha256(SOURCE),
        "predictions_sha256": sha256(PRED)
    }

    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    print("\n" + "=" * 90)
    print("PASS: PHASE26N PREDICTIONS FROZEN")
    print("=" * 90)
    print("VTI TRAIN:", len(v))
    print("ETI TRAIN:", len(e))
    print("Validation identities:", len(pred))
    print("Validation protein positions:", val.protein_position.nunique())
    print("Untouched test variants:", len(test))
    print("Validation assay outcomes: NOT OPENED")
    print("Predictions:", PRED)
    print("Lock manifest:", MANIFEST)


def position_errors(df, target, model):
    t = df[["protein_position"]].copy()
    t["ae"] = np.abs(
        df[target].to_numpy(float)
        - df[model].to_numpy(float)
    )
    return t.groupby("protein_position").ae.mean()


def metrics(df, target, model):
    y = df[target].to_numpy(float)
    p = df[model].to_numpy(float)
    error = p - y

    return {
        "position_mae_pp": float(
            position_errors(df, target, model).mean()
        ),
        "variant_mae_pp": float(np.mean(np.abs(error))),
        "rmse_pp": float(np.sqrt(np.mean(error ** 2))),
        "bias_pp": float(np.mean(error))
    }


def bootstrap(df, target, candidate, reference, nboot, seed):
    a = position_errors(df, target, candidate)
    b = position_errors(df, target, reference)

    gains = (b - a).to_numpy(float)
    rng = np.random.default_rng(seed)

    samples = rng.choice(
        gains,
        size=(nboot, len(gains)),
        replace=True
    ).mean(axis=1)

    return {
        "candidate": candidate,
        "reference": reference,
        "target": target,
        "gain_pp": float(gains.mean()),
        "ci95_low": float(np.quantile(samples, .025)),
        "ci95_high": float(np.quantile(samples, .975)),
        "positions": len(gains)
    }


def score(nboot):
    if not PRED.is_file() or not MANIFEST.is_file():
        raise FileNotFoundError("Run --prepare first")

    if any(p.exists() for p in (SCORES, GAINS, SUBGROUPS)):
        raise FileExistsError("Validation already scored")

    manifest = json.loads(MANIFEST.read_text())

    if sha256(PRED) != manifest["predictions_sha256"]:
        raise ValueError("Frozen predictions have changed")

    if sha256(SOURCE) != manifest["script_sha256"]:
        raise ValueError("Script changed after predictions were frozen")

    _, val, _ = validate_metadata()

    pred = pd.read_csv(PRED, dtype={"variant_key": str})
    keys = set(val.variant_key.astype(str))

    if (
        len(pred) != 52
        or pred.variant_key.duplicated().any()
        or set(pred.variant_key) != keys
    ):
        raise ValueError("Frozen prediction membership mismatch")

    # The ONLY protected-outcome read: VTI VALIDATION.
    assay = pd.read_parquet(
        VTI,
        columns=[
            "variant_key",
            "corrector_rescue_pp",
            "vti_rescue_pp"
        ],
        filters=[
            ("variant_key", "in", sorted(keys))
        ]
    )

    assay["variant_key"] = assay.variant_key.astype(str)

    if (
        len(assay) != 52
        or assay.variant_key.duplicated().any()
        or set(assay.variant_key) != keys
    ):
        raise ValueError("Filtered validation labels incomplete")

    d = pred.merge(
        assay, on="variant_key", validate="one_to_one"
    )

    d["observed_corrector"] = pd.to_numeric(
        d.corrector_rescue_pp, errors="raise"
    )
    d["observed_total"] = pd.to_numeric(
        d.vti_rescue_pp, errors="raise"
    )

    if not np.isfinite(
        d[["observed_corrector", "observed_total"]]
        .to_numpy(float)
    ).all():
        raise ValueError("Nonfinite validation outcomes")

    tasks = [
        (
            "Corrector",
            "observed_corrector",
            ["corrector_vti_only", "corrector_eti_assisted"]
        ),
        (
            "Total",
            "observed_total",
            [
                "total_vti_components",
                "total_eti_components",
                "total_vti_direct"
            ]
        )
    ]

    results = []
    subgroups = []

    for task, target, models in tasks:
        y = d[target].to_numpy(float)

        masks = {
            "ALL": np.ones(len(d), dtype=bool),
            "observed_low_lt10": y < 10,
            "observed_high_gt40": y > 40
        }

        for cohort in d.study_cohort.unique():
            masks[f"cohort_{cohort}"] = (
                d.study_cohort.eq(cohort).to_numpy()
            )

        for model in models:
            results.append({
                "target": task,
                "model": model,
                "variants": len(d),
                "positions": d.protein_position.nunique(),
                **metrics(d, target, model)
            })

            for name, mask in masks.items():
                if not np.any(mask):
                    continue

                part = d.loc[mask]

                subgroups.append({
                    "target": task,
                    "model": model,
                    "subgroup": name,
                    "variants": len(part),
                    "positions": part.protein_position.nunique(),
                    **metrics(part, target, model)
                })

    results = pd.DataFrame(results)

    gains = pd.DataFrame([
        bootstrap(
            d, "observed_corrector",
            "corrector_eti_assisted", "corrector_vti_only",
            nboot, 261
        ),
        bootstrap(
            d, "observed_total",
            "total_eti_components", "total_vti_components",
            nboot, 262
        ),
        bootstrap(
            d, "observed_total",
            "total_eti_components", "total_vti_direct",
            nboot, 263
        )
    ])

    subgroups = pd.DataFrame(subgroups)

    results.to_csv(SCORES, index=False)
    gains.to_csv(GAINS, index=False)
    subgroups.to_csv(SUBGROUPS, index=False)

    (OUT / "phase26n_validation_scored_LOCK.json").write_text(
        json.dumps({
            "validation_scored": True,
            "test_outcomes_accessed": False,
            "predictions_sha256": manifest["predictions_sha256"]
        }, indent=2) + "\n"
    )

    print("\n" + "=" * 95)
    print("PHASE 26N | FIRST INDEPENDENT VTI VALIDATION")
    print("=" * 95)
    print("Variants: 52 | independent protein positions: 42")
    print("\nA. FROZEN MODEL PERFORMANCE")
    print(results.round(4).to_string(index=False))

    print("\nB. PAIRED POSITION BOOTSTRAP")
    print("Positive gain = candidate improves")
    print(gains.round(4).to_string(index=False))

    print("\nC. EXPLORATORY SUBGROUPS")
    print(subgroups.round(3).to_string(index=False))

    print("\nPASS: Validation scored once; test untouched")
    print("Saved:", SCORES)


def self_test():
    d = pd.DataFrame({
        "protein_position": [1, 1, 2, 3],
        "y": [1., 3., 5., 7.],
        "a": [2., 2., 6., 5.],
        "b": [1., 3., 5., 7.]
    })

    assert np.isclose(
        metrics(d, "y", "a")["position_mae_pp"],
        4 / 3
    )

    gain = bootstrap(d, "y", "b", "a", 200, 26)

    assert gain["gain_pp"] > 0
    assert gain["positions"] == 3

    print(
        "PASS synthetic: position-weighted MAE "
        "and clustered bootstrap"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    modes = parser.add_mutually_exclusive_group(required=True)

    modes.add_argument("--prepare", action="store_true")
    modes.add_argument("--score", action="store_true")
    modes.add_argument("--self-test", action="store_true")

    parser.add_argument("--bootstraps", type=int, default=10000)

    args = parser.parse_args()

    if args.self_test:
        self_test()
    elif args.prepare:
        prepare()
    else:
        score(args.bootstraps)
