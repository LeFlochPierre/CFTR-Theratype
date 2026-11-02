#!/usr/bin/env python3
"""Phase 24B | Train-only ETI-assay-conditioned total VTI rescue calibration.

Use case: an ETI functional measurement is ALREADY available for the exact variant.
This is NOT zero-shot sequence-only prediction.

Input: Phase23A paired ETI 2024 TRAIN / VTI 2026 TRAIN data;
       frozen VTI TRAIN labels from model_zoo.prepare();
       stored Phase17A TRAIN outer-OOF predictions.

Calibration: fixed-parameter models fitted with protein-position-held-out
5-fold CV. Direct ETI measurement is the primary comparator.

Uncertainty: empirical group-level direct-ETI residual quantiles computed
on fit positions only. These are exploratory, not clinical intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 20261028

TARGET = "vti_rescue_pp"
ETI = "eti_total_fsk_iva_rescue_pp"

ETI_FEATURES = [
    ETI,
    "eti_acute_iva_pp",
    "dmso_acute_iva_pp",
    "eti_corrector_fsk_gain_pp",
]

METHODS = (
    "direct_eti",
    "median_vti_prior",
    "median_offset",
    "ridge_eti",
    "huber_eti",
    "ridge_eti_components",
)


def check(test, explanation):
    if not bool(test):
        raise AssertionError(explanation)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pos_mae(y, p, pos):
    e = pd.DataFrame({
        "position": pos,
        "err": np.abs(np.asarray(y, float) - np.asarray(p, float)),
    })
    return float(e.groupby("position").err.mean().mean())


def position_screening(y, p, pos, limit=10.0, fraction=0.2):
    """Nonresponse: ALL variants at a position have total rescue <10 pp."""

    t = pd.DataFrame({
        "position": pos,
        "y": y,
        "p": p,
    }).groupby("position").agg(
        observed_max=("y", "max"),
        score_max=("p", "max"),
    )

    truth = t.observed_max.to_numpy(float) < limit
    score = -t.score_max.to_numpy(float)

    n = len(truth)
    k = int(np.ceil(fraction * n))

    order = np.argsort(-score, kind="stable")[:k]

    hits = int(truth[order].sum())
    total = int(truth.sum())

    return {
        "nonresponse_n_positions": total,
        "nonresponse_prevalence": float(truth.mean()),
        "nonresponse_ap": (
            float(average_precision_score(truth, score))
            if 0 < total < n else np.nan
        ),
        "nonresponse_auroc": (
            float(roc_auc_score(truth, score))
            if 0 < total < n else np.nan
        ),
        "top20_n_screened": k,
        "top20_hits": hits,
        "top20_precision": hits / k,
        "top20_recall": hits / total if total else np.nan,
    }


def summary(y, p, pos):
    y = np.asarray(y, float)
    p = np.asarray(p, float)

    a = np.abs(y - p)

    return {
        "position_mae_pp": pos_mae(y, p, pos),
        "variant_mae_pp": float(a.mean()),
        "rmse_pp": float(np.sqrt(np.mean((y - p) ** 2))),
        "bias_pred_minus_obs_pp": float(np.mean(p - y)),
        "median_abs_error_pp": float(np.median(a)),
        "n_errors_gt20pp": int((a > 20).sum()),
        "n_errors_gt40pp": int((a > 40).sum()),
        **position_screening(y, p, pos),
    }


def quantile_group_interval(train_err, train_pos, alpha=0.1):
    """Finite-sample quantile using maximum residual at each position."""

    z = pd.DataFrame({
        "pos": train_pos,
        "err": np.abs(train_err),
    }).groupby("pos").err.max().to_numpy(float)

    n = len(z)
    k = int(np.ceil((n + 1) * (1 - alpha)))

    if k > n:
        return float("inf")

    return float(np.partition(z, k - 1)[k - 1])


def grouped_calibration(d, n_splits=5, alpha=0.1):

    y = d[TARGET].to_numpy(float)
    pos = d.protein_position.to_numpy(int)
    eti = d[ETI].to_numpy(float)
    xm = d[ETI_FEATURES].to_numpy(float)

    check(
        np.isfinite(y).all() and np.isfinite(xm).all(),
        "Nonfinite paired ETI/VTI values",
    )

    preds = {
        k: np.full(len(d), np.nan)
        for k in METHODS
    }

    fold_id = np.zeros(len(d), int)
    band_q = np.full(len(d), np.nan)

    splitter = GroupKFold(n_splits=n_splits)

    for fold, (fi, hi) in enumerate(
        splitter.split(xm, y, groups=pos), 1
    ):

        check(
            set(pos[fi]).isdisjoint(set(pos[hi])),
            "Position leakage in calibration fold",
        )

        fold_id[hi] = fold

        # Direct measurement
        preds["direct_eti"][hi] = eti[hi]

        # Training-only median
        preds["median_vti_prior"][hi] = np.median(y[fi])

        # Median ETI-to-VTI offset
        preds["median_offset"][hi] = (
            eti[hi] + np.median(y[fi] - eti[fi])
        )

        specs = {
            "ridge_eti": (
                make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=10),
                ),
                eti[fi, None],
                eti[hi, None],
            ),
            "huber_eti": (
                make_pipeline(
                    StandardScaler(),
                    HuberRegressor(
                        alpha=1.0,
                        epsilon=1.35,
                        max_iter=2000,
                    ),
                ),
                eti[fi, None],
                eti[hi, None],
            ),
            "ridge_eti_components": (
                make_pipeline(
                    StandardScaler(),
                    Ridge(alpha=10),
                ),
                xm[fi],
                xm[hi],
            ),
        }

        for name, (mod, xfit, xheld) in specs.items():
            mod.fit(xfit, y[fi])
            preds[name][hi] = mod.predict(xheld)

        # Uncertainty estimated ONLY from fitting positions
        band_q[hi] = quantile_group_interval(
            y[fi] - eti[fi],
            pos[fi],
            alpha=alpha,
        )

    check(
        np.all(fold_id > 0),
        "Incomplete grouped CV fold assignment",
    )

    check(
        all(np.isfinite(pr).all() for pr in preds.values()),
        "Invalid calibration prediction",
    )

    return preds, fold_id, band_q


def paired_pos_bootstrap(
    y, pos, method_pred, reference_pred,
    n_boot, seed=SEED
):

    v = (
        np.abs(
            np.asarray(y, float)
            - np.asarray(reference_pred, float)
        )
        -
        np.abs(
            np.asarray(y, float)
            - np.asarray(method_pred, float)
        )
    )

    by_pos = pd.DataFrame({
        "pos": pos,
        "gain": v,
    }).groupby("pos").gain.mean().to_numpy(float)

    rng = np.random.default_rng(seed)

    boot = np.array([
        np.mean(
            rng.choice(
                by_pos,
                len(by_pos),
                replace=True,
            )
        )
        for _ in range(n_boot)
    ])

    return {
        "gain_pp": float(by_pos.mean()),
        "ci95_lo": float(np.quantile(boot, 0.025)),
        "ci95_hi": float(np.quantile(boot, 0.975)),
        "bootstrap_positive_fraction": float((boot > 0).mean()),
        "positions_improved": int((by_pos > 0).sum()),
    }


def self_test():

    rng = np.random.default_rng(42)

    npos = 56
    pos = np.repeat(np.arange(npos), 2)
    n = len(pos)

    d = pd.DataFrame({
        "variant_key": [f"v{i}" for i in range(n)],
        "protein_position": pos,
    })

    d[ETI] = rng.normal(40, 20, n)
    d["eti_acute_iva_pp"] = rng.normal(12, 8, n)
    d["dmso_acute_iva_pp"] = rng.normal(6, 4, n)

    d["eti_corrector_fsk_gain_pp"] = (
        d[ETI] - d["eti_acute_iva_pp"]
    )

    d[TARGET] = (
        3 + 0.75 * d[ETI]
        + rng.normal(0, 7, n)
    )

    preds, fold, q = grouped_calibration(d)

    check(
        len(preds) == 6
        and all(np.isfinite(v).all() for v in preds.values()),
        "Prediction failure",
    )

    check(
        len(np.unique(fold)) == 5
        and np.isfinite(q).all(),
        "Group folds / uncertainty failure",
    )

    for name, p in preds.items():
        check(
            np.isfinite(
                summary(d[TARGET], p, pos)["position_mae_pp"]
            ),
            "Metric failed " + name,
        )

    bs = paired_pos_bootstrap(
        d[TARGET],
        pos,
        preds["ridge_eti"],
        preds["direct_eti"],
        200,
    )

    check(
        np.isfinite(bs["gain_pp"]),
        "Bootstrap failed",
    )

    print(
        "PASS: Grouped calibration, preprocessing, "
        "uncertainty and bootstrap self-test"
    )


def main():

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--self-test",
        action="store_true",
    )

    parser.add_argument(
        "--bootstraps",
        type=int,
        default=5000,
    )

    args = parser.parse_args()

    if args.self_test:
        return self_test()

    check(
        args.bootstraps >= 100,
        "At least 100 bootstrap samples required",
    )

    from cftr_theratype import model_zoo as mz
    from cftr_theratype.transfer import ROOT

    root = Path(ROOT)

    train, _, _, outer = mz.prepare(root)

    check(
        len(train) == 210
        and train.protein_position.nunique() == 168
        and len(outer) == 10,
        "Frozen VTI training membership unexpected",
    )

    source = (
        root
        / "results/biology/phase23a/"
        / "eti_vti_exact_train_matches.csv"
    )

    check(
        source.is_file(),
        f"Paired TRAIN source absent: {source}",
    )

    df = pd.read_csv(
        source,
        dtype={"variant_key": str},
    )

    req = {
        "variant_key",
        "protein_position",
        *ETI_FEATURES,
    }

    check(
        req.issubset(df),
        f"Missing required columns: {req - set(df.columns)}",
    )

    check(
        len(df) == 66
        and df.variant_key.is_unique
        and df.protein_position.nunique() == 56,
        "Expected 66 matched TRAIN variants / 56 positions",
    )

    check(
        set(df.variant_key).issubset(set(train.variant_key)),
        "Nontraining VTI variant in matched input",
    )

    df = df.merge(
        train[
            [
                "variant_key",
                "protein_position",
                TARGET,
                "study_cohort",
            ]
        ],
        on="variant_key",
        suffixes=("_eti", ""),
        how="left",
        validate="one_to_one",
    )

    check(
        (
            df.protein_position_eti.to_numpy(int)
            == df.protein_position.to_numpy(int)
        ).all(),
        "Position mismatch",
    )

    check(
        df.study_cohort.notna().all(),
        "Missing VTI TRAIN membership",
    )

    check(
        df[TARGET].notna().all(),
        "Missing VTI TRAIN labels",
    )

    df = df.sort_values(
        ["protein_position", "variant_key"]
    ).reset_index(drop=True)

    print("=" * 105)
    print(
        "PHASE 24B | ETI-ASSAY-CONDITIONED TOTAL VTI RESCUE"
    )
    print("=" * 105)

    print(
        "DEPLOYMENT: Exact mutation already experimentally "
        "assayed under ETI"
    )

    print(
        "66 paired variants / 56 positions; "
        "2024 TRAIN + VTI TRAIN only"
    )

    print(
        "Protected validation/test/excluded outcomes unopened"
    )

    y = df[TARGET].to_numpy(float)
    pos = df.protein_position.to_numpy(int)
    eti = df[ETI].to_numpy(float)

    print("\nA. RAW CROSS-REGIMEN AGREEMENT")

    print(
        f"ETI/VTI Spearman rho="
        f"{spearmanr(eti, y).statistic:+.4f}"
    )

    print(
        f"Direct ETI - VTI median signed difference="
        f"{np.median(eti - y):+.3f} pp"
    )

    print(
        f"Median absolute difference="
        f"{np.median(np.abs(eti - y)):.3f} pp"
    )

    print(
        "Disagreements >10, >20, >40 pp:",
        int((np.abs(eti - y) > 10).sum()),
        int((np.abs(eti - y) > 20).sum()),
        int((np.abs(eti - y) > 40).sum()),
    )

    print(
        "VTI below 10 pp:",
        int((y < 10).sum()),
        "| ETI below 10 pp:",
        int((eti < 10).sum()),
    )

    print(
        pd.crosstab(
            eti < 10,
            y < 10,
            rownames=["ETI total <10"],
            colnames=["VTI total <10"],
        ).to_string()
    )

    preds, folds, qs = grouped_calibration(df)

    historical = (
        root
        / "results/model_zoo/"
        / "phase17a_train_predictions_averaged.csv"
    )

    check(
        historical.is_file(),
        "Frozen Phase17A averaged OOF missing",
    )

    h = pd.read_csv(
        historical,
        dtype={"variant_key": str},
    )

    h = h.loc[
        h.model.eq("sequence_structure__xgboost"),
        [
            "variant_key",
            "protein_position",
            "observed_pp",
            "predicted_pp",
        ],
    ]

    check(
        h.variant_key.is_unique and len(h) == 210,
        "Bad historical zero-shot comparator schema",
    )

    ht = h.merge(
        train[["variant_key", TARGET]],
        on="variant_key",
    )

    check(
        np.allclose(
            ht.observed_pp,
            ht[TARGET],
            atol=1e-5,
            rtol=0,
        ),
        "Historical total VTI target mismatch",
    )

    matched = df[
        ["variant_key", "protein_position", TARGET]
    ].merge(
        h,
        on="variant_key",
        validate="one_to_one",
        suffixes=("", "_old"),
    )

    check(
        len(matched) == len(df)
        and np.array_equal(
            matched.protein_position,
            matched.protein_position_old,
        ),
        "Historical comparator alignment mismatch",
    )

    preds["historical_zero_shot_80feature"] = (
        matched.predicted_pp.to_numpy(float)
    )

    rows = [
        {"model": name, **summary(y, v, pos)}
        for name, v in preds.items()
    ]

    metrics = pd.DataFrame(rows).sort_values(
        "position_mae_pp"
    )

    print("\n" + "=" * 105)
    print("B. MATCHED 66-VARIANT / 56-POSITION PERFORMANCE")
    print("=" * 105)
    print(metrics.round(4).to_string(index=False))

    contrasts = [
        ("direct_eti", "median_vti_prior"),
        ("median_offset", "direct_eti"),
        ("ridge_eti", "direct_eti"),
        ("huber_eti", "direct_eti"),
        ("ridge_eti_components", "direct_eti"),
        (
            "direct_eti",
            "historical_zero_shot_80feature",
        ),
    ]

    boot = []

    for a, b in contrasts:
        boot.append({
            "model": a,
            "comparator": b,
            **paired_pos_bootstrap(
                y,
                pos,
                preds[a],
                preds[b],
                args.bootstraps,
                SEED,
            ),
        })

    bootstrap = pd.DataFrame(boot)

    print("\n" + "=" * 105)
    print("C. PAIRED POSITION BOOTSTRAP")
    print("=" * 105)
    print(bootstrap.round(4).to_string(index=False))

    covered = np.abs(y - eti) <= qs

    pos_coverage = pd.DataFrame({
        "pos": pos,
        "covered": covered,
    }).groupby("pos").covered.all()

    widths = 2 * qs

    print("\n" + "=" * 105)
    print("D. DIRECT ETI UNCERTAINTY BANDS")
    print("=" * 105)

    print(
        f"Intended position coverage: 90% | "
        f"Observed: {pos_coverage.mean():.3f}"
    )

    print(
        f"Observed per-variant coverage: "
        f"{covered.mean():.3f}"
    )

    print(
        f"Median half-width: {np.median(qs):.3f} pp"
    )

    print(
        f"Median full width: {np.median(widths):.3f} pp"
    )

    print(
        f"Maximum half-width: {np.max(qs):.3f} pp"
    )

    print(
        "NOTE: Exploratory group-based bands, "
        "not clinical prediction intervals"
    )

    errors = df[
        [
            "variant_key",
            "protein_position",
            ETI,
            TARGET,
        ]
    ].copy()

    errors["direct_abs_error_pp"] = np.abs(y - eti)
    errors["direct_signed_error_pp"] = eti - y

    print("\n" + "=" * 105)
    print("E. LARGEST CROSS-REGIMEN DISAGREEMENTS")
    print("=" * 105)

    print(
        errors.sort_values(
            "direct_abs_error_pp",
            ascending=False,
        ).head(12).round(3).to_string(index=False)
    )

    out = root / "results/biology/phase24b"
    out.mkdir(parents=True, exist_ok=True)

    predfile = out / "phase24b_paired_predictions.csv"

    o = df[
        [
            "variant_key",
            "protein_position",
            TARGET,
            *ETI_FEATURES,
            "study_cohort",
        ]
    ].copy()

    o["calibration_fold"] = folds
    o["direct_eti_90pct_band_halfwidth_pp"] = qs
    o["direct_eti_90pct_band_covered"] = covered

    for name, v in preds.items():
        o["pred_" + name] = v

    o.to_csv(predfile, index=False)

    metrics.to_csv(
        out / "phase24b_summary.csv",
        index=False,
    )

    bootstrap.to_csv(
        out / "phase24b_bootstrap.csv",
        index=False,
    )

    errors.to_csv(
        out / "phase24b_cross_regimen_disagreement.csv",
        index=False,
    )

    report = {
        "phase": "24B",
        "n_paired_train_variants": len(df),
        "n_paired_positions": int(
            df.protein_position.nunique()
        ),
        "n_bootstraps": int(args.bootstraps),
        "source_sha256": sha256(source),
        "deployment_requires_measured_eti_assay_for_exact_variant": True,
        "folds": 5,
        "uncertainty_group_max_quantile_alpha": 0.1,
        "group_coverage_observed": float(
            pos_coverage.mean()
        ),
        "training_only": True,
        "protected_outcomes_opened": False,
        "caveats": [
            "Only overlapping 2024 TRAIN and VTI TRAIN mutations",
            "ETI assay must be known at prediction time",
            "Position-held-out calibration; no independent clinical validation",
            "Multiple exploratory methods considered",
            "Uncertainty assumes exchangeable positions",
            "Prediction bands are not clinical calibration",
            "10 pp threshold is an in-vitro research marker",
        ],
    }

    (
        out / "phase24b_report.json"
    ).write_text(
        json.dumps(report, indent=2) + "\n"
    )

    print("\nOUTPUTS")

    for pth in [
        predfile,
        out / "phase24b_summary.csv",
        out / "phase24b_bootstrap.csv",
        out / "phase24b_cross_regimen_disagreement.csv",
        out / "phase24b_report.json",
    ]:
        print(" ", pth.relative_to(root))

    print(
        "\nPASS: Position-held-out assay-conditioned "
        "calibration completed; protected outcomes unopened."
    )


if __name__ == "__main__":
    main()
