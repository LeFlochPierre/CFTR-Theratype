#!/usr/bin/env python3
"""Phase 26L: cross-fitted Tedman trafficking teacher.

TRAIN ONLY. Mutation-derived inference.
Tedman surface measurements are training targets, never inference inputs.
All Tedman labels at outer-held-out protein positions are excluded.
Teacher predictions for training positions are cross-fitted.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

import phase26h_auxiliary_corrector as h

OUT = ROOT / "results/biology/phase26l"
TEDMAN = ROOT / "results/biology/phase26k/phase26k3_derived_surface_train.csv"
OLD = ROOT / "results/biology/phase26h2/phase26h2_oof.csv"
WEIGHTS = ROOT / "results/biology/phase26h/phase26h_inner_selections.csv"

KEYS = ["repeat", "fold", "variant_key", "protein_position"]


def mutation_key(df):
    return (
        df["reference_aa"].astype(str).str.upper()
        + df["protein_position"].astype(int).astype(str)
        + df["alternate_aa"].astype(str).str.upper()
    )


def teacher():
    # Fixed in advance. No tuning against VTI test-fold results.
    return make_pipeline(
        StandardScaler(),
        Ridge(alpha=100.0)
    )


def position_shuffle(y, positions, seed):
    """Shuffle position means, preserving within-position variation."""
    y = np.asarray(y, float)
    p = np.asarray(positions, int)

    _, inverse = np.unique(p, return_inverse=True)

    means = (
        np.bincount(inverse, weights=y)
        / np.bincount(inverse)
    )

    rng = np.random.default_rng(seed)
    shuffled_means = rng.permutation(means)

    return y - means[inverse] + shuffled_means[inverse]


def teacher_features(
    Tx, ty, tp,
    Xtrain_v, ptrain_v,
    Xtrain_e, ptrain_e,
    Xtest,
    seed=None
):
    """Position-cross-fitted teacher predictions for training rows."""

    Tx = np.asarray(Tx, float)
    ty = np.asarray(ty, float)
    tp = np.asarray(tp, int)

    assert np.isfinite(Tx).all()
    assert np.isfinite(ty).all()

    if len(np.unique(tp)) < 10:
        raise ValueError("Too few independent teacher positions")

    source_y = (
        ty if seed is None
        else position_shuffle(ty, tp, seed)
    )

    full = teacher().fit(Tx, source_y)

    pv = full.predict(Xtrain_v)
    pe = full.predict(Xtrain_e)
    pt = full.predict(Xtest)

    source_oof = np.full(len(Tx), np.nan)
    source_baseline = np.full(len(Tx), np.nan)

    nfold = min(5, len(np.unique(tp)))

    for tr, te in GroupKFold(nfold).split(
        Tx, source_y, groups=tp
    ):
        assert not set(tp[tr]).intersection(tp[te])

        mdl = teacher().fit(
            Tx[tr], source_y[tr]
        )

        source_oof[te] = mdl.predict(Tx[te])
        source_baseline[te] = np.mean(source_y[tr])

        held_positions = np.unique(tp[te])

        mv = np.isin(ptrain_v, held_positions)
        me = np.isin(ptrain_e, held_positions)

        if mv.any():
            pv[mv] = mdl.predict(Xtrain_v[mv])

        if me.any():
            pe[me] = mdl.predict(Xtrain_e[me])

    for z in (pv, pe, pt, source_oof, source_baseline):
        if not np.isfinite(z).all():
            raise ValueError("Nonfinite teacher predictions")

    return pv, pe, pt, source_oof, source_baseline


def pos_mae(df, target, pred):
    q = df[["repeat", "protein_position"]].copy()

    q["loss"] = np.abs(
        df[target].to_numpy(float)
        - df[pred].to_numpy(float)
    )

    return float(
        q.groupby(["repeat", "protein_position"])
        .loss.mean()
        .groupby("protein_position")
        .mean()
        .mean()
    )


def bootstrap(df, candidate, baseline, target, nboot, seed):
    q = df[["repeat", "protein_position"]].copy()

    q["gain"] = (
        np.abs(df[target] - df[baseline])
        - np.abs(df[target] - df[candidate])
    )

    z = (
        q.groupby(["repeat", "protein_position"])
        .gain.mean()
        .groupby("protein_position")
        .mean()
        .to_numpy()
    )

    rng = np.random.default_rng(seed)

    sims = rng.choice(
        z,
        size=(nboot, len(z)),
        replace=True
    ).mean(axis=1)

    return {
        "candidate": candidate,
        "reference": baseline,
        "target": target,
        "gain_pp": float(z.mean()),
        "ci95_low": float(np.quantile(sims, .025)),
        "ci95_high": float(np.quantile(sims, .975)),
        "positions": len(z)
    }


def load():
    if not TEDMAN.exists():
        raise FileNotFoundError(
            f"Run Phase 26K-3 first: {TEDMAN}"
        )

    if not OLD.exists() or not WEIGHTS.exists():
        raise FileNotFoundError(
            "Need Phase26H-1/H-2 saved results"
        )

    v, V80, e, E80, folds = h.load_data()

    v = v.copy()
    e = e.copy()

    v["mutation"] = mutation_key(v)
    e["mutation"] = mutation_key(e)



    # Repeated protein substitutions are permitted, but feature
    # representations must agree across their variant keys.
    for name, frame, matrix in (
        ("VTI", v, V80),
        ("ETI", e, E80)
    ):
        if not frame.variant_key.astype(str).is_unique:
            raise ValueError(
                f"{name}: duplicate variant_key identifiers"
            )

        if frame.mutation.isna().any():
            raise ValueError(
                f"{name}: missing protein substitutions"
            )

        duplicates = frame.loc[
            frame.mutation.duplicated(keep=False),
            ["mutation", "variant_key", "protein_position"]
        ]

        print(
            f"{name}: {len(frame)} variants; "
            f"{frame.mutation.nunique()} unique substitutions; "
            f"{len(duplicates)} rows with repeated substitutions"
        )

        if len(duplicates):
            print(
                duplicates.sort_values(
                    ["mutation", "variant_key"]
                ).head(12).to_string(index=False)
            )

        for mutation, indices in frame.groupby(
            "mutation", sort=False
        ).indices.items():

            indices = np.asarray(indices, dtype=int)

            if len(indices) > 1 and not np.allclose(
                matrix[indices],
                matrix[indices[0]],
                rtol=0,
                atol=1e-8,
                equal_nan=True
            ):
                raise ValueError(
                    f"{name}: inconsistent features for "
                    f"protein substitution {mutation}"
                )

    td = pd.read_csv(
        TEDMAN,
        dtype={"mutation": str}
    )

    assert len(td) == 180
    assert td.mutation.is_unique

    td = td[
        ["mutation", "position", "p_log2fc_dual"]
    ].copy()

    # First experiment uses exact ETI TRAIN matches.
    # The 62 remaining eligible Tedman mutations
    # require additional feature-generation QC.

    # One feature representative per protein substitution.
    # Original ETI rows remain unchanged for model training.
    eti_feature_map = (
        e[["mutation", "variant_key", "protein_position"]]
        .sort_values("variant_key")
        .drop_duplicates("mutation", keep="first")
    )

    if not eti_feature_map.mutation.is_unique:
        raise ValueError(
            "ETI feature mapping contains duplicates"
        )

    td = td.merge(
        eti_feature_map,
        on="mutation",
        how="inner",
        validate="one_to_one"
    )

    if len(td) != 118:
        raise ValueError(
            f"Expected 118 Tedman/ETI matches, found {len(td)}"
        )

    if not (
        td.position.astype(int)
        == td.protein_position.astype(int)
    ).all():
        raise ValueError(
            "Tedman/ETI protein position mismatch"
        )

    eidx = {
        key: i
        for i, key in enumerate(
            e.variant_key.astype(str)
        )
    }

    source_ix = np.array([
        eidx[k]
        for k in td.variant_key.astype(str)
    ], int)

    Tp = td.protein_position.to_numpy(int)
    Ty = td.p_log2fc_dual.to_numpy(float)

    # Sequence features only for small teacher dataset.
    Tx = E80[source_ix, :68].copy()

    assert Tx.shape == (118, 68)
    assert len(np.unique(Tp)) == 101
    assert np.isfinite(Tx).all()
    assert np.isfinite(Ty).all()

    wd = pd.read_csv(WEIGHTS)

    wd = wd.loc[
        wd.model.eq("joint_real")
    ]

    assert len(wd) == 10
    assert not wd.duplicated(
        ["repeat", "fold"]
    ).any()

    weights = {
        (int(row.repeat), int(row.fold)):
            float(row.eti_weight)
        for row in wd.itertuples()
    }

    old = pd.read_csv(
        OLD,
        dtype={"variant_key": str}
    )

    assert len(old) == 420
    assert not old.duplicated(KEYS).any()

    return (
        v, V80, e, E80, folds,
        Tx, Ty, Tp, weights, old
    )


def run(nshuffles, nboot):
    (
        v, V80, e, E80, folds,
        Tx, Ty, Tp, weights, old
    ) = load()

    vk = v.variant_key.astype(str).to_numpy()
    vp = v.protein_position.to_numpy(int)
    ep = e.protein_position.to_numpy(int)

    yc = v.corrector_rescue_pp.to_numpy(float)
    ye = e.corrector_rescue_pp.to_numpy(float)

    VX = np.column_stack([
        V80, np.ones(len(v))
    ])

    EX = np.column_stack([
        E80, np.zeros(len(e))
    ])

    print("=" * 102)
    print(
        "PHASE 26L | CROSSFITTED TEDMAN "
        "TRAFFICKING TEACHER | TRAIN ONLY"
    )
    print("=" * 102)

    print(
        f"Tedman/ETI training matches: {len(Ty)} variants "
        f"/ {len(np.unique(Tp))} protein positions"
    )

    print(
        "Teacher phenotype: log2 fold change, "
        "VX-445/VX-661 vs untreated parental cells"
    )

    print(
        "Teacher: Ridge, 68 sequence features, "
        "position-cross-fitted"
    )

    print(
        "Downstream: frozen ETI-assisted XGBoost "
        "+ predicted trafficking response"
    )

    print(
        f"Shuffled-phenotype controls: {nshuffles}"
    )

    print(
        "Mutation-only inference; "
        "protected outcomes unopened"
    )

    print(
        "Exploratory experiment: phenotype selected "
        "after examining training correlations"
    )

    rows = []
    qc = []

    for f, (
        rep, fold, trkeys, tekeys
    ) in enumerate(folds, 1):

        itr = np.flatnonzero(
            np.isin(vk, list(trkeys))
        )

        ite = np.flatnonzero(
            np.isin(vk, list(tekeys))
        )

        assert len(itr) + len(ite) == len(v)
        assert set(vp[itr]).isdisjoint(vp[ite])

        # Exclude all ETI and Tedman labels at
        # outer VTI held-out protein positions.
        aux = ~np.isin(ep, vp[ite])
        tmask = ~np.isin(Tp, vp[ite])

        assert not set(ep[aux]).intersection(vp[ite])
        assert not set(Tp[tmask]).intersection(vp[ite])

        if tmask.sum() < 50:
            raise ValueError(
                "Too few eligible Tedman labels in fold"
            )

        ew = weights[(int(rep), int(fold))]

        # Reproduce the frozen ETI-assisted model.
        ref = h.fit_joint(
            VX[itr], yc[itr],
            EX[aux], ye[aux],
            ew
        ).predict(VX[ite])

        predictions = {
            "joint_refit": ref
        }

        for k in range(nshuffles + 1):
            name = (
                "tedman_real"
                if k == 0
                else f"tedman_sham{k}"
            )

            seed = (
                None
                if k == 0
                else 26700 + 100 * f + k
            )

            pv, pe, pt, source_oof, source_mean = (
                teacher_features(
                    Tx[tmask],
                    Ty[tmask],
                    Tp[tmask],
                    V80[itr, :68],
                    vp[itr],
                    E80[aux, :68],
                    ep[aux],
                    V80[ite, :68],
                    seed=seed
                )
            )

            x_v = np.column_stack([VX[itr], pv])
            x_e = np.column_stack([EX[aux], pe])
            x_t = np.column_stack([VX[ite], pt])

            model = h.fit_joint(
                x_v, yc[itr],
                x_e, ye[aux],
                ew
            )

            predictions[name] = model.predict(x_t)

            if k == 0:
                rho = spearmanr(
                    Ty[tmask], source_oof
                ).statistic
            else:
                rho = np.nan

            qc.append({
                "repeat": rep,
                "fold": fold,
                "model": name,
                "teacher_variants": int(tmask.sum()),
                "teacher_positions": int(
                    np.unique(Tp[tmask]).size
                ),
                "eti_aux_variants": int(aux.sum()),
                "eti_weight": ew,
                "teacher_oof_spearman": (
                    float(rho)
                    if np.isfinite(rho)
                    else np.nan
                ),
                "teacher_oof_mae": float(
                    np.abs(
                        Ty[tmask] - source_oof
                    ).mean()
                ),
                "mean_teacher_mae": float(
                    np.abs(
                        Ty[tmask] - source_mean
                    ).mean()
                ),
                "teacher_pred_std_test": float(
                    np.std(pt)
                )
            })

        for j, i in enumerate(ite):
            row = {
                "repeat": int(rep),
                "fold": int(fold),
                "variant_key": vk[i],
                "protein_position": int(vp[i])
            }

            for name, pred in predictions.items():
                row[name] = float(pred[j])

            rows.append(row)

        print(
            f"completed {f}/{len(folds)}: "
            f"repeat={rep}, fold={fold}, "
            f"Tedman_train={tmask.sum()}, "
            f"ETI_train={aux.sum()}, ETI_weight={ew}",
            flush=True
        )

    new = pd.DataFrame(rows)

    if len(new) != len(old):
        raise ValueError(
            "Incomplete OOF predictions"
        )

    d = old.merge(
        new,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    if (
        len(d) != len(old)
        or not d._merge.eq("both").all()
    ):
        raise ValueError(
            "Frozen OOF membership mismatch"
        )

    d = d.drop(columns="_merge")

    delta = float(
        np.max(
            np.abs(
                d.joint_refit - d.joint_real
            )
        )
    )

    print(
        "\nOriginal ETI-assisted model "
        f"reproducibility: max difference={delta:.8g}"
    )

    if delta > 1e-5:
        raise ValueError(
            "Frozen Phase26H model did not reproduce"
        )

    names = ["tedman_real"] + [
        f"tedman_sham{i}"
        for i in range(1, nshuffles + 1)
    ]

    for name in names:
        d[name + "_total"] = (
            d[name]
            + d.pred_zero_acute_seq80
        )

    metrics = []

    tasks = [
        (
            "Corrector",
            "observed_corrector",
            ["joint_real", *names]
        ),
        (
            "Total",
            "observed_total",
            [
                "joint_real_total",
                *[n + "_total" for n in names]
            ]
        )
    ]

    for target, actual, methods in tasks:
        y = d[actual].to_numpy(float)

        for model in methods:
            for subset, mask in [
                ("ALL", np.ones(len(d), bool)),
                ("low<10", y < 10),
                ("high>40", y > 40)
            ]:
                q = d.loc[mask]

                metrics.append({
                    "target": target,
                    "model": model,
                    "subset": subset,
                    "variants": q.variant_key.nunique(),
                    "positions": q.protein_position.nunique(),
                    "position_mae_pp": pos_mae(
                        q, actual, model
                    ),
                    "bias_pp": float(
                        (q[model] - q[actual]).mean()
                    ),
                    "rmse_pp": float(
                        np.sqrt(
                            np.mean(
                                (q[model] - q[actual]) ** 2
                            )
                        )
                    )
                })

    metrics = pd.DataFrame(metrics)

    comparisons = [
        (
            "tedman_real",
            "joint_real",
            "observed_corrector"
        ),
        (
            "tedman_real_total",
            "joint_real_total",
            "observed_total"
        )
    ]

    comparisons += [
        (
            "tedman_real",
            f"tedman_sham{i}",
            "observed_corrector"
        )
        for i in range(1, nshuffles + 1)
    ]

    gains = pd.DataFrame([
        bootstrap(
            d, candidate, baseline,
            target, nboot, 26700 + i
        )
        for i, (candidate, baseline, target)
        in enumerate(comparisons)
    ])

    qc = pd.DataFrame(qc)

    print(
        "\nA. TEACHER GENERALISATION "
        "(held-out protein positions)"
    )

    print(
        qc.groupby("model", as_index=False)[
            [
                "teacher_variants",
                "teacher_oof_spearman",
                "teacher_oof_mae",
                "mean_teacher_mae",
                "teacher_pred_std_test"
            ]
        ].mean().round(4).to_string(index=False)
    )

    print("\nB. ZERO-SHOT VTI RESPONSE ERRORS")

    print(
        metrics.round(4).to_string(index=False)
    )

    print(
        "\nC. PAIRED POSITION BOOTSTRAP "
        "(positive gain = improvement)"
    )

    print(
        gains.round(4).to_string(index=False)
    )

    print("\nD. CORRECTOR HIGH-RESPONDER RANKING")

    q = d.groupby(
        "variant_key", as_index=False
    ).agg({
        "observed_corrector": "first",
        "joint_real": "mean",
        "tedman_real": "mean"
    })

    positive = (
        q.observed_corrector.to_numpy(float) > 40
    ).astype(int)

    for name in ("joint_real", "tedman_real"):
        prediction = q[name].to_numpy(float)
        n = int(np.ceil(0.2 * len(q)))

        hits = int(
            positive[
                np.argsort(-prediction)[:n]
            ].sum()
        )

        print(
            f"{name}: "
            f"AUROC={roc_auc_score(positive, prediction):.4f}, "
            f"AP={average_precision_score(positive, prediction):.4f}, "
            f"top20_hits={hits}/{positive.sum()} "
            f"(selected {n})"
        )

    OUT.mkdir(parents=True, exist_ok=True)

    d.to_csv(
        OUT / "phase26l_oof.csv",
        index=False
    )

    metrics.to_csv(
        OUT / "phase26l_subgroups.csv",
        index=False
    )

    gains.to_csv(
        OUT / "phase26l_bootstrap.csv",
        index=False
    )

    qc.to_csv(
        OUT / "phase26l_teacher_qc.csv",
        index=False
    )

    print(
        "\nPASS: Phase26L completed; TRAIN only."
    )

    print("Saved:", OUT)


def self_test():
    rng = np.random.default_rng(26)

    positions = np.repeat(
        np.arange(25), 2
    )

    x = rng.normal(
        size=(len(positions), 68)
    )

    y = (
        2 + .5 * x[:, 0]
        + rng.normal(
            scale=.1, size=len(positions)
        )
    )

    shuffled = position_shuffle(
        y, positions, 100
    )

    assert len(shuffled) == len(y)
    assert np.isfinite(shuffled).all()

    pv, pe, pt, po, pbase = teacher_features(
        x, y, positions,
        x[:15], positions[:15],
        x[15:32], positions[15:32],
        x[32:],
        seed=None
    )

    for value in (pv, pe, pt, po, pbase):
        assert np.isfinite(value).all()

    assert len(pv) == 15
    assert len(pe) == 17
    assert len(pt) == len(x) - 32

    assert np.mean(np.abs(po - y)) > 0

    test = pd.DataFrame({
        "repeat": [1, 1, 2, 2],
        "protein_position": [1, 2, 1, 2],
        "target": [3., 5., 3., 5.],
        "old": [4., 4., 4., 4.],
        "new": [3., 5., 3., 5.]
    })

    assert bootstrap(
        test, "new", "old",
        "target", 100, 26
    )["gain_pp"] > 0

    print(
        "PASS synthetic Phase26L: "
        "position cross-fitting, permutation, bootstrap"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--self-test",
        action="store_true"
    )

    ap.add_argument(
        "--shuffles",
        type=int,
        default=2
    )

    ap.add_argument(
        "--bootstraps",
        type=int,
        default=10000
    )

    args = ap.parse_args()

    if args.self_test:
        self_test()
    else:
        if not 1 <= args.shuffles <= 5:
            ap.error("--shuffles must be 1..5")

        if args.bootstraps < 100:
            ap.error("--bootstraps must be >=100")

        run(args.shuffles, args.bootstraps)
