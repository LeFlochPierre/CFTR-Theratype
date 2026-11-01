#!/usr/bin/env python3
"""Phase 26J: frozen-fold, train-only response-tail reweighting.

Test whether emphasizing extreme training responses reduces
prediction shrinkage for VTI corrector and total rescue.

Uses fold-local VTI training labels only for weighting.
Reuses ETI weights previously selected by inner grouped CV.
Protected ETI/VTI validation/test outcomes are never loaded.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

import phase26h_auxiliary_corrector as h

OUT = ROOT / "results/biology/phase26j"
OLD = ROOT / "results/biology/phase26h2/phase26h2_oof.csv"
SELECTION = ROOT / "results/biology/phase26h/phase26h_inner_selections.csv"
KEYS = ["repeat", "fold", "variant_key", "protein_position"]


def weights(y, kind, multiplier):
    y = np.asarray(y, float)
    if kind == "both":
        extreme = (y < 10) | (y > 40)
    elif kind == "high":
        extreme = y > 40
    else:
        raise ValueError(kind)
    return np.where(extreme, float(multiplier), 1.)


def fit_corrector(Xv, yv, Xe, ye, eti_weight,
                  kind=None, multiplier=1):
    X = np.vstack([Xv, Xe])
    y = np.r_[yv, ye]

    wv = (
        np.ones(len(yv))
        if kind is None
        else weights(yv, kind, multiplier)
    )

    sample_weight = np.r_[
        wv, np.full(len(ye), eti_weight)
    ]

    model = h.previous.model(h.SPEC)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def fit_total(X, y, kind=None, multiplier=1):
    mdl = h.previous.model(h.SPEC)

    w = (
        np.ones(len(y))
        if kind is None
        else weights(y, kind, multiplier)
    )

    mdl.fit(X, y, sample_weight=w)
    return mdl


def pos_mae(df, target, pred):
    temp = df[["repeat", "protein_position"]].copy()
    temp["ae"] = np.abs(
        df[target].to_numpy(float)
        - df[pred].to_numpy(float)
    )

    return float(
        temp.groupby(["repeat", "protein_position"])
        .ae.mean()
        .groupby("protein_position")
        .mean()
        .mean()
    )


def paired_ci(df, cand, ref, target, boot, seed):
    z = df[["repeat", "protein_position"]].copy()

    z["gain"] = (
        np.abs(df[target] - df[ref])
        - np.abs(df[target] - df[cand])
    )

    vals = (
        z.groupby(["repeat", "protein_position"])
        .gain.mean()
        .groupby("protein_position")
        .mean()
        .to_numpy(float)
    )

    rng = np.random.default_rng(seed)

    sims = rng.choice(
        vals,
        size=(boot, len(vals)),
        replace=True
    ).mean(axis=1)

    return dict(
        candidate=cand,
        reference=ref,
        target=target,
        gain_pp=float(vals.mean()),
        ci95_low=float(np.quantile(sims, .025)),
        ci95_high=float(np.quantile(sims, .975)),
        positions=len(vals)
    )


def run(boot):
    if not OLD.exists() or not SELECTION.exists():
        raise FileNotFoundError(
            "Need completed Phase26H-1 and Phase26H-2 OOF files"
        )

    v, V, e, E, folds = h.load_data()

    vk = v.variant_key.astype(str).to_numpy()
    vp = v.protein_position.to_numpy(int)
    ep = e.protein_position.to_numpy(int)

    yc = v.corrector_rescue_pp.to_numpy(float)
    yt = v.vti_rescue_pp.to_numpy(float)
    ye = e.corrector_rescue_pp.to_numpy(float)

    VX = np.column_stack([V, np.ones(len(V))])
    EX = np.column_stack([E, np.zeros(len(E))])

    old = pd.read_csv(OLD, dtype={"variant_key": str})

    assert len(old) == 420
    assert not old.duplicated(KEYS).any()

    previous = pd.read_csv(SELECTION)
    previous = previous[
        previous.model.eq("joint_real")
    ]

    assert len(previous) == 10
    assert not previous.duplicated(
        ["repeat", "fold"]
    ).any()

    lookup = {
        (int(r.repeat), int(r.fold)): float(r.eti_weight)
        for r in previous.itertuples()
    }

    print("=" * 100)
    print("PHASE 26J | TAIL-AWARE ETI-ASSISTED CFTR RESPONSE")
    print("=" * 100)

    print(
        "210 VTI variants / 168 positions; "
        "same frozen repeated grouped folds"
    )
    print(
        "Strategies: both tails x2; high responders x3"
    )
    print(
        "Thresholds <10 and >40 pp: exploratory "
        "training-derived definitions"
    )
    print(
        "ETI weights fixed from previous inner CV."
    )
    print(
        "Mutation-only inference; protected outcomes unopened."
    )

    rows = []

    for n, (repeat, fold, trkeys, tekeys) in enumerate(
        folds, 1
    ):
        itr = np.flatnonzero(
            np.isin(vk, list(trkeys))
        )

        ite = np.flatnonzero(
            np.isin(vk, list(tekeys))
        )

        assert len(itr) + len(ite) == len(v)
        assert not set(vp[itr]).intersection(vp[ite])

        allow = ~np.isin(ep, vp[ite])

        assert not set(ep[allow]).intersection(vp[ite])

        wt = lookup[(int(repeat), int(fold))]

        # Reproduce original ETI-assisted model.
        base = fit_corrector(
            VX[itr], yc[itr],
            EX[allow], ye[allow],
            wt
        )

        part = pd.DataFrame({
            "repeat": repeat,
            "fold": fold,
            "variant_key": vk[ite],
            "protein_position": vp[ite]
        })

        part["joint_refit"] = base.predict(VX[ite])

        # Corrector: double both tails.
        part["corrector_both2"] = fit_corrector(
            VX[itr], yc[itr],
            EX[allow], ye[allow],
            wt, "both", 2
        ).predict(VX[ite])

        # Corrector: triple strong responders.
        part["corrector_high3"] = fit_corrector(
            VX[itr], yc[itr],
            EX[allow], ye[allow],
            wt, "high", 3
        ).predict(VX[ite])

        # Direct zero-shot total rescue.
        part["direct_total_fixed"] = fit_total(
            V[itr], yt[itr]
        ).predict(V[ite])

        part["direct_total_both2"] = fit_total(
            V[itr], yt[itr],
            "both", 2
        ).predict(V[ite])

        part["direct_total_high3"] = fit_total(
            V[itr], yt[itr],
            "high", 3
        ).predict(V[ite])

        rows.append(part)

        print(
            f"completed {n}/{len(folds)}: "
            f"repeat={repeat}, fold={fold}, "
            f"ETI weight={wt}, "
            f"held-out positions={len(np.unique(vp[ite]))}",
            flush=True
        )

    joined = pd.concat(rows, ignore_index=True)

    if len(joined) != len(old):
        raise ValueError("Incomplete OOF membership")

    d = old.merge(
        joined,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    if not d._merge.eq("both").all() or len(d) != len(old):
        raise ValueError("OOF alignment mismatch")

    d.drop(columns="_merge", inplace=True)

    if not np.allclose(
        d["joint_refit"],
        d["joint_real"],
        rtol=0,
        atol=1e-5
    ):
        delta = float(
            np.abs(
                d["joint_refit"] - d["joint_real"]
            ).max()
        )
        raise ValueError(
            "Old joint model did not reproduce: "
            f"max prediction difference={delta:.6g}"
        )

    print(
        "\nPASS: Original ETI-assisted OOF predictions reproduced"
    )

    # Reconstruct total rescue using frozen acute prediction.
    for name in ["corrector_both2", "corrector_high3"]:
        d[name + "_total"] = (
            d[name] + d.pred_zero_acute_seq80
        )

    tasks = {
        "Corrector": (
            "observed_corrector",
            [
                "joint_real",
                "corrector_both2",
                "corrector_high3",
                "vti_fixed"
            ]
        ),
        "Total": (
            "observed_total",
            [
                "joint_real_total",
                "corrector_both2_total",
                "corrector_high3_total",
                "pred_zero_total_seq80",
                "direct_total_fixed",
                "direct_total_both2",
                "direct_total_high3"
            ]
        )
    }

    print("\nA. RESPONSE DISTRIBUTION ERRORS")
    print(
        "True-response strata are descriptive only; "
        "not available at deployment."
    )

    stats = []

    for task, (target, methods) in tasks.items():
        y = d[target].to_numpy(float)

        for model in methods:
            subgroups = [
                ("all", np.ones(len(d), bool)),
                ("low<10", y < 10),
                ("middle10to40", (y >= 10) & (y <= 40)),
                ("high>40", y > 40),
                ("very_high>60", y > 60)
            ]

            for subset, mask in subgroups:
                temp = d.loc[mask]

                if temp.empty:
                    continue

                stats.append({
                    "target": task,
                    "model": model,
                    "subset": subset,
                    "variants": temp.variant_key.nunique(),
                    "positions": temp.protein_position.nunique(),
                    "position_mae_pp": pos_mae(
                        temp, target, model
                    ),
                    "bias_pp": float(
                        (temp[model] - temp[target]).mean()
                    ),
                    "rmse_pp": float(
                        np.sqrt(
                            np.mean(
                                (temp[model] - temp[target]) ** 2
                            )
                        )
                    )
                })

    stats = pd.DataFrame(stats)

    print(
        stats.loc[
            stats.subset.isin(["all", "low<10", "high>40"])
        ].round(3).to_string(index=False)
    )

    print("\nB. PAIRED POSITION BOOTSTRAP")
    print("Positive gain = lower MAE")

    tests = [
        (
            "corrector_both2",
            "joint_real",
            "observed_corrector"
        ),
        (
            "corrector_high3",
            "joint_real",
            "observed_corrector"
        ),
        (
            "corrector_both2_total",
            "joint_real_total",
            "observed_total"
        ),
        (
            "corrector_high3_total",
            "joint_real_total",
            "observed_total"
        ),
        (
            "direct_total_both2",
            "direct_total_fixed",
            "observed_total"
        ),
        (
            "direct_total_high3",
            "direct_total_fixed",
            "observed_total"
        ),
        (
            "direct_total_both2",
            "pred_zero_total_seq80",
            "observed_total"
        ),
        (
            "direct_total_high3",
            "pred_zero_total_seq80",
            "observed_total"
        )
    ]

    boots = pd.DataFrame([
        paired_ci(d, a, b, y, boot, 26100 + i)
        for i, (a, b, y) in enumerate(tests)
    ])

    print(
        boots.round(4).to_string(index=False)
    )

    print("\nC. HIGH-RESPONDER RANKING")
    print("Predictions averaged over two OOF repeats")

    aggregate = {
        "observed_corrector": "first",
        "observed_total": "first"
    }

    aggregate.update({
        m: "mean"
        for task in tasks.values()
        for m in task[1]
    })

    q = d.groupby(
        "variant_key", as_index=False
    ).agg(aggregate)

    ranking = []

    for task, (target, models) in tasks.items():
        truth = (
            q[target].to_numpy(float) > 40
        ).astype(int)

        for m in models:
            pred = q[m].to_numpy(float)

            k = int(np.ceil(0.2 * len(q)))

            hits = int(
                truth[
                    np.argsort(-pred, kind="stable")[:k]
                ].sum()
            )

            ranking.append({
                "target": task,
                "model": m,
                "n_positives": int(truth.sum()),
                "AUROC": roc_auc_score(truth, pred),
                "AP": average_precision_score(truth, pred),
                "top20_hits": hits,
                "top20_selected": k
            })

    ranking = pd.DataFrame(ranking)

    print(
        ranking.round(4).to_string(index=False)
    )

    OUT.mkdir(parents=True, exist_ok=True)

    d.to_csv(
        OUT / "phase26j_oof.csv",
        index=False
    )

    stats.to_csv(
        OUT / "phase26j_subgroups.csv",
        index=False
    )

    boots.to_csv(
        OUT / "phase26j_bootstrap.csv",
        index=False
    )

    ranking.to_csv(
        OUT / "phase26j_ranking.csv",
        index=False
    )

    print(
        "\nPASS: Phase26J completed; TRAIN only."
    )
    print("Outputs:", OUT)


def self_test():
    rng = np.random.default_rng(26)

    X = rng.normal(size=(40, 81))
    E = rng.normal(size=(65, 81))

    E[:, -1] = 0
    X[:, -1] = 1

    y = rng.normal(24, 15, 40)
    ye = rng.normal(24, 15, 65)

    assert weights(
        np.array([1, 21, 55]), "both", 2
    ).tolist() == [2., 1., 2.]

    assert weights(
        np.array([1, 21, 55]), "high", 3
    ).tolist() == [1., 1., 3.]

    model = fit_corrector(
        X, y, E, ye, .25, "both", 2
    )

    assert np.isfinite(
        model.predict(X[:5])
    ).all()

    z = pd.DataFrame({
        "repeat": [1, 1, 2, 2],
        "protein_position": [1, 2, 1, 2],
        "t": [3., 4., 3., 4.],
        "a": [2., 3., 2., 3.],
        "b": [4., 4., 4., 4.]
    })

    assert np.isfinite(
        paired_ci(z, "a", "b", "t", 100, 2)["gain_pp"]
    )

    print(
        "PASS synthetic Phase26J: weighted fitting, "
        "position bootstrap, response strata"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--self-test", action="store_true"
    )
    parser.add_argument(
        "--bootstraps", type=int, default=10000
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        run(args.bootstraps)
