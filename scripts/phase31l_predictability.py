#!/usr/bin/env python3
"""Phase 31L: TRAIN-only SEM, feature signal, and learning curves."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path.cwd() / "scripts"))

from cftr_theratype.datasets.frozen_vti import load_train as load_vti_train
from cftr_theratype.models.regressors import phase26g_model
from cftr_theratype.features.regions import domains_from_file, domain_of
from cftr_theratype.evaluation.historical_oof import load_m0

OUT = Path("results/biology/phase31l")

TARGET = {
    "corrector": "corrector_rescue_pp",
    "total": "vti_rescue_pp",
}

SEM = [
    "fsk_sem_dmso",
    "fsk_sem_vnz_tez",
    "fsk_iva_sem_vnz_tez",
    "wells_dmso",
    "wells_vnz_tez",
]


def score(y, pred, pos):
    return pd.DataFrame({
        "p": pos,
        "ae": abs(y - pred)
    }).groupby("p").ae.mean().mean()


def source():
    d, matrices, _, folds, ref = load_vti_train(Path.cwd())
    oof, _ = load_m0(d, root=Path.cwd())

    X = np.asarray(matrices["sequence_structure"], float)

    if X.shape != (210, 80):
        raise ValueError("Unexpected feature matrix shape")

    if d.protein_position.nunique() != 168 or len(folds) != 10:
        raise ValueError("Frozen partition mismatch")

    seq = np.asarray(matrices["sequence"], float)

    if not np.array_equal(X[:, :68], seq, equal_nan=True):
        raise ValueError("Sequence feature ordering mismatch")

    return d, X, folds, oof


def inspect(d, X, oof):
    keys = d.variant_key.astype(str).tolist()

    assay = pd.read_parquet(
        "data/processed/bihler_2026_vti.parquet",
        columns=["variant_key", *SEM],
        filters=[("variant_key", "in", keys)],
    )

    if assay.variant_key.duplicated().any() or set(assay.variant_key) != set(keys):
        raise ValueError("SEM table does not match VTI TRAIN")

    for c in SEM:
        assay[c] = pd.to_numeric(assay[c], errors="coerce")

    if (assay[SEM[:3]] < 0).any().any():
        raise ValueError("Negative SEM found")

    assay["corrector_sem_proxy"] = np.hypot(
        assay.fsk_sem_dmso,
        assay.fsk_sem_vnz_tez,
    )

    assay["total_sem_proxy"] = np.hypot(
        assay.fsk_sem_dmso,
        assay.fsk_iva_sem_vnz_tez,
    )

    summary = []

    for target in TARGET:
        z = oof[oof.target.eq(target)].copy()
        z["ae"] = (z.prediction - z.observed).abs()

        z = z.groupby(
            ["variant_key", "protein_position"], as_index=False
        ).ae.mean()

        z = z.merge(assay, on="variant_key", validate="one_to_one")

        field = f"{target}_sem_proxy"

        q = z.groupby("protein_position")[
            [field, "ae"]
        ].mean().dropna()

        rho = (
            spearmanr(q[field], q.ae).statistic
            if len(q) > 8 and q[field].nunique() > 1
            else np.nan
        )

        summary.append({
            "target": target,
            "n_with_sem": int(z[field].notna().sum()),
            "n_positions_with_sem": len(q),
            "median_sem_proxy_pp": z[field].median(),
            "median_abs_error_pp": z.ae.median(),
            "rho_sem_vs_error_by_position": rho,
        })

        z.to_csv(
            OUT / f"phase31l_{target}_sem_errors.csv",
            index=False,
        )

    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / "phase31l_sem_summary.csv", index=False)

    print("\n31L-A | SEM PROXIES (NOT A NOISE CEILING)")
    print(summary.round(3).to_string(index=False))

    # ---------------------------------------------------------
    # 31L-B: Mutation representation signal
    # ---------------------------------------------------------

    pos = np.unique(d.protein_position.to_numpy(int))

    Xpos = (
        pd.DataFrame(X)
        .assign(p=d.protein_position.to_numpy(int))
        .groupby("p")
        .mean()
        .reindex(pos)
    )

    features = []
    domains = []

    regions = domains_from_file(None)
    annotations = np.array([
        domain_of(p, regions) for p in pos
    ])

    for target, col in TARGET.items():
        ypos = (
            d.groupby("protein_position")[col]
            .mean()
            .reindex(pos)
            .to_numpy(float)
        )

        for j in range(80):
            a = Xpos[j].to_numpy(float)
            valid = np.isfinite(a) & np.isfinite(ypos)

            rho = (
                spearmanr(a[valid], ypos[valid]).statistic
                if valid.sum() >= 15
                and np.unique(a[valid]).size >= 3
                else np.nan
            )

            features.append({
                "target": target,
                "feature_index": j,
                "type": "sequence" if j < 68 else "structure",
                "n_positions": int(valid.sum()),
                "spearman": rho,
            })

        for region in np.unique(annotations):
            a = ypos[annotations == region]

            domains.append({
                "target": target,
                "domain_approx": region,
                "n_positions": len(a),
                "response_mean_pp": np.mean(a),
                "response_sd_pp": np.std(a),
            })

    features = pd.DataFrame(features)
    features["abs_rho"] = features.spearman.abs()

    features.to_csv(
        OUT / "phase31l_feature_signal.csv", index=False
    )

    domains = pd.DataFrame(domains)
    domains.to_csv(
        OUT / "phase31l_domain_variation.csv", index=False
    )

    print("\n31L-B | TOP UNIVARIATE FEATURE CORRELATIONS")
    print(
        features.sort_values("abs_rho", ascending=False)
        .groupby("target").head(5)
        .round(3).to_string(index=False)
    )

    print("\n31L-B | APPROXIMATE DOMAIN VARIATION")
    print(domains.round(3).to_string(index=False))


def curves(d, X, folds, oof, quick=False, draws=3):
    keys = d.variant_key.astype(str).to_numpy()
    pos = d.protein_position.to_numpy(int)

    frozen = {
        t: oof[oof.target.eq(t)]
        .set_index(["repeat", "fold", "variant_key"])
        .prediction
        for t in TARGET
    }

    rows = []
    chosen_folds = folds[:2] if quick else folds

    for fi, (repeat, fold, trkeys, tekeys) in enumerate(chosen_folds, 1):
        it = np.flatnonzero(np.isin(keys, list(trkeys)))
        iv = np.flatnonzero(np.isin(keys, list(tekeys)))

        assert len(it) + len(iv) == len(d)
        assert set(pos[it]).isdisjoint(pos[iv])

        positions = np.unique(pos[it])

        sizes = (
            [20, 80] if quick else [20, 40, 80, 120]
        ) + [len(positions)]

        for draw in range(1 if quick else draws):
            rng = np.random.default_rng(
                31120 + int(repeat) * 100000
                + int(fold) * 1000 + draw
            )

            order = rng.permutation(positions)

            for n in sizes:
                full = n == len(positions)

                if full and draw > 0:
                    continue

                tr = it[np.isin(pos[it], order[:n])]
                assert len(set(pos[tr])) == n

                for target, col in TARGET.items():
                    y = d[col].to_numpy(float)

                    models = ["median", "ridge80", "xgb80_fixed"]

                    if full:
                        models.append("M0_frozen")

                    for model in models:
                        if model == "median":
                            med = (
                                pd.DataFrame({
                                    "p": pos[tr], "y": y[tr]
                                })
                                .groupby("p").y.mean().median()
                            )
                            pred = np.full(len(iv), med)

                        elif model == "ridge80":
                            fit = make_pipeline(
                                SimpleImputer(
                                    strategy="median",
                                    keep_empty_features=True,
                                ),
                                StandardScaler(),
                                Ridge(alpha=100.0),
                            )
                            fit.fit(X[tr], y[tr])
                            pred = fit.predict(X[iv])

                        elif model == "xgb80_fixed":
                            fit = phase26g_model(
                                "xgb", (2, 5.0, 180)
                            )
                            fit.fit(X[tr], y[tr])
                            pred = fit.predict(X[iv])

                        else:
                            pred = np.array([
                                frozen[target].loc[
                                    (int(repeat), int(fold), k)
                                ]
                                for k in keys[iv]
                            ])

                        for j, prediction in zip(iv, pred):
                            rows.append((
                                target, model,
                                "full" if full else str(n),
                                draw, int(repeat), int(fold),
                                keys[j], int(pos[j]),
                                float(y[j]), float(prediction),
                            ))

        print(
            f"31L-C fold {fi}/{len(chosen_folds)}: "
            f"repeat={repeat} fold={fold} "
            f"train_positions={len(positions)}",
            flush=True,
        )

    columns = [
        "target", "model", "training_positions",
        "draw", "repeat", "fold", "variant_key",
        "protein_position", "observed", "prediction",
    ]

    allpred = pd.DataFrame(rows, columns=columns)

    filename = (
        "phase31l_curves_quick.csv"
        if quick else "phase31l_curves_oof.csv"
    )
    allpred.to_csv(OUT / filename, index=False)

    if quick:
        print("PASS: Two-fold smoke test only")
        return

    metrics = []

    for (target, model, size, draw, repeat), z in allpred.groupby(
        ["target", "model", "training_positions", "draw", "repeat"]
    ):
        if len(z) != len(d) or z.variant_key.duplicated().any():
            raise ValueError("Incomplete out-of-fold predictions")

        metrics.append({
            "target": target,
            "model": model,
            "training_positions": size,
            "draw": draw,
            "repeat": repeat,
            "position_mae_pp": score(
                z.observed.to_numpy(float),
                z.prediction.to_numpy(float),
                z.protein_position.to_numpy(int),
            ),
        })

    metrics = pd.DataFrame(metrics)

    metrics.to_csv(
        OUT / "phase31l_curve_repeat_draw_metrics.csv",
        index=False,
    )

    summary = (
        metrics.groupby(
            ["target", "model", "training_positions"],
            as_index=False,
        )
        .agg(
            mean_position_mae_pp=("position_mae_pp", "mean"),
            sd_repeat_draw=("position_mae_pp", "std"),
            n_repeat_draw=("position_mae_pp", "size"),
        )
    )

    for target, expected in [
        ("corrector", 12.9800),
        ("total", 17.5109),
    ]:
        baseline = metrics[
            (metrics.target == target)
            & (metrics.model == "M0_frozen")
        ]

        if len(baseline) != 2 or not np.isclose(
            baseline.position_mae_pp.mean(),
            expected, atol=0.002,
        ):
            raise ValueError(
                f"Frozen M0 baseline mismatch: {target}"
            )

    summary.to_csv(
        OUT / "phase31l_curve_summary.csv",
        index=False,
    )

    print("\n31L-C | POSITION-LEVEL LEARNING CURVES")
    print(summary.round(3).to_string(index=False))
    print("\nPASS: TRAIN-only curves. VAL/TEST untouched.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["inspect", "curves"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--draws", type=int, default=3)
    args = ap.parse_args()

    if not 1 <= args.draws <= 10:
        ap.error("draws must be 1..10")

    OUT.mkdir(parents=True, exist_ok=True)
    d, X, folds, oof = source()

    print(
        f"Phase31L: {len(d)} VTI TRAIN variants, "
        f"{d.protein_position.nunique()} positions, "
        f"{len(folds)} frozen outer folds"
    )

    if args.mode == "inspect":
        inspect(d, X, oof)
    else:
        curves(d, X, folds, oof, args.quick, args.draws)


if __name__ == "__main__":
    main()
