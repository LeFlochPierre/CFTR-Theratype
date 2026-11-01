#!/usr/bin/env python3
"""Phase 26H-1: zero-shot VTI corrector prediction with ETI supervision.

Mutation-derived features are the only prediction inputs.
ETI training labels at outer/inner held-out positions are excluded.
Protected ETI/VTI validation/test outcomes are never loaded.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import phase26d_corrector_bottleneck as previous

OUT = ROOT / "results/biology/phase26h"
WEIGHTS = (0.25, 0.5, 1.0)

# Fixed from earlier training-only experiments.
SPEC = (2, 5.0, 180)


def parsed_position(s):
    s = str(s)
    m = re.search(
        r'(?:p\.)?\(?[A-Za-z]{3}(\d{1,4})(?:[A-Za-z]{3}|Ter|\*|=)',
        s
    )
    if not m:
        m = re.search(
            r'(?:p\.)?\(?[A-Za-z](\d{1,4})[A-Za-z*]',
            s
        )
    return int(m.group(1)) if m else None


def load_data():
    from cftr_theratype.benchmark import make_features
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.splits import parse_substitution
    from cftr_theratype import structure_benchmark as structure

    vti, mats, _, folds, ref = previous.load_train()

    assert len(vti) == 210
    assert vti.protein_position.nunique() == 168

    vti_keys = set(vti.variant_key.astype(str))

    # Membership metadata only.
    etisplit = pd.read_parquet(
        ROOT / "data/processed/bihler_2024_missense_splits.parquet",
        columns=["variant_key", "split"]
    )

    vtisplit = pd.read_parquet(
        ROOT / "data/processed/bihler_2026_transfer_splits.parquet",
        columns=[
            "variant_key", "protein_position", "transfer_split"
        ]
    )

    etitrain_keys = set(
        etisplit.loc[
            etisplit.split.eq("train"), "variant_key"
        ].astype(str)
    )

    protected = set(
        pd.to_numeric(
            vtisplit.loc[
                vtisplit.transfer_split.isin(["val", "test"]),
                "protein_position"
            ],
            errors="coerce"
        ).dropna().astype(int)
    )

    assert len(etitrain_keys) == 422
    assert len(protected) == 68

    # Only ETI TRAIN labels are loaded.
    eti = pd.read_parquet(
        ROOT / "data/processed/bihler_2024_theratyping.parquet",
        columns=[
            "variant_key",
            "variant",
            "hgvs_cdna",
            "hgvs_protein",
            "variant_type",
            "corrector_rescue_pp"
        ],
        filters=[
            ("variant_key", "in", sorted(etitrain_keys))
        ]
    )

    eti["variant_key"] = eti.variant_key.astype(str)

    assert len(eti) == 422
    assert eti.variant_key.is_unique
    assert set(eti.variant_key) == etitrain_keys

    eti["protein_position"] = eti.hgvs_protein.map(
        parsed_position
    )

    if eti.protein_position.isna().any():
        raise ValueError(
            "Could not parse ETI TRAIN protein positions: "
            + str(
                eti.loc[
                    eti.protein_position.isna(),
                    ["variant_key", "hgvs_protein"]
                ].head().to_dict("records")
            )
        )

    eti["protein_position"] = (
        eti.protein_position.astype(int)
    )

    # Globally protected VTI validation/test positions.
    eti = eti.loc[
        ~eti.protein_position.isin(protected)
    ].copy()

    eti["corrector_rescue_pp"] = pd.to_numeric(
        eti.corrector_rescue_pp,
        errors="raise"
    )

    assert np.isfinite(
        eti.corrector_rescue_pp.to_numpy(float)
    ).all()

    assert len(eti) == 383
    assert eti.protein_position.nunique() == 276

    eti = eti.sort_values(
        "variant_key"
    ).reset_index(drop=True)

    eti["transfer_split"] = "train"
    eti["study_cohort"] = "ETI"

    # Generate the exact same mutation descriptors.
    seq = bihler_2024_assay_sequence()

    parsed = [
        parse_substitution(row, seq)
        for _, row in eti.iterrows()
    ]

    bad = [
        (i, p, eti.iloc[i].variant_key)
        for i, p in enumerate(parsed)
        if p[-1] != "valid"
        or p[0] != int(eti.iloc[i].protein_position)
    ]

    if bad:
        raise ValueError(
            f"ETI substitution/reference mismatch: {bad[:5]}"
        )

    eti["reference_aa"] = [p[1] for p in parsed]
    eti["alternate_aa"] = [p[2] for p in parsed]

    E68, names = make_features(eti, seq)
    E68 = np.asarray(E68, dtype=float)

    if E68.shape != (383, 68):
        raise ValueError(
            f"Unexpected ETI sequence features: {E68.shape}"
        )

    if not np.isfinite(E68).all():
        raise ValueError(
            "Nonfinite ETI sequence features"
        )

    # Reuse frozen Phase 15 structural descriptors.
    struct = structure.read_feature_table(
        ROOT / "results/structure/phase15b_position_features.csv",
        ROOT / "results/structure/phase15b_feature_summary.json",
        ROOT
    )

    E12 = np.asarray(
        structure.structural_matrix(
            eti.protein_position.to_numpy(int),
            struct
        ),
        float
    )

    if E12.shape != (383, 12):
        raise ValueError(
            f"Unexpected ETI structural features: {E12.shape}"
        )

    E80 = np.column_stack([E68, E12])
    E80[np.isinf(E80)] = np.nan

    V80 = np.asarray(
        mats["sequence_structure"],
        float
    ).copy()

    V80[np.isinf(V80)] = np.nan

    assert V80.shape == (210, 80)

    # Cross-study feature parity for exact matched variants.
    shared = sorted(
        vti_keys.intersection(set(eti.variant_key))
    )

    vi = {
        k: i
        for i, k in enumerate(vti.variant_key.astype(str))
    }

    ei = {
        k: i
        for i, k in enumerate(eti.variant_key.astype(str))
    }

    check_v = V80[[vi[k] for k in shared]]
    check_e = E80[[ei[k] for k in shared]]

    mismatch = np.logical_xor(
        np.isnan(check_v),
        np.isnan(check_e)
    )

    finite = (
        np.isfinite(check_v)
        & np.isfinite(check_e)
    )

    maxdiff = float(
        np.max(np.abs(check_v[finite] - check_e[finite]))
    ) if finite.any() else float("nan")

    print(
        f"Feature parity: {len(shared)} exact matches; "
        f"max difference={maxdiff:.6g}; "
        f"NaN mismatches={mismatch.sum()}"
    )

    if (
        len(shared) != 66
        or mismatch.any()
        or maxdiff > 1e-7
    ):
        raise ValueError(
            "ETI/VTI mutation features disagree. "
            "Refusing to train."
        )

    novel = (
        ~eti.protein_position.isin(
            vti.protein_position
        )
    ).sum()

    print(
        f"ETI auxiliary data: {len(eti)} variants / "
        f"{eti.protein_position.nunique()} positions"
    )

    print(
        f"Novel VTI positions: {novel} ETI variants"
    )

    print(
        f"ETI structural missing values: "
        f"{np.isnan(E12).sum()}"
    )

    print(
        f"Frozen outer splits: {len(folds)} | "
        f"reference={ref}"
    )

    return vti, V80, eti, E80, folds


def positional_shuffle(y, group, seed):
    """Shuffle position means; retain within-position offsets."""

    rng = np.random.default_rng(seed)
    positions = np.unique(group)

    means = pd.DataFrame({
        "g": group,
        "y": y
    }).groupby("g").y.mean()

    shuffled = rng.permutation(
        means.reindex(positions).to_numpy()
    )

    mapping = dict(zip(positions, shuffled))

    return np.array([
        mapping[g] + (v - means.loc[g])
        for v, g in zip(y, group)
    ], float)


def fit_joint(Vx, Vy, Ex, Ey, weight):
    if len(Vx) == 0 or len(Ex) == 0:
        raise ValueError("Empty training subset")

    x = np.vstack([Vx, Ex])
    y = np.r_[Vy, Ey]

    w = np.r_[
        np.ones(len(Vx)),
        np.full(len(Ex), weight)
    ]

    estimator = previous.model(SPEC)
    estimator.fit(
        x, y,
        sample_weight=w
    )

    return estimator


def choose_weight(
    Xv, yv, gv, Xe, ye, ge,
    weights=WEIGHTS,
    shuffle_seed=None
):
    scores = {}

    for weight in weights:
        preds = np.full(len(yv), np.nan)

        cv = GroupKFold(3)

        for inner_fold, (it, iv) in enumerate(
            cv.split(Xv, yv, gv), 1
        ):
            # Remove ETI observations at inner-held-out
            # VTI positions before fitting or shuffling.
            allowed = ~np.isin(
                ge,
                np.unique(gv[iv])
            )

            if not np.any(allowed):
                raise ValueError(
                    "No eligible inner-fold ETI supervision"
                )

            if shuffle_seed is None:
                fold_labels = ye[allowed]
            else:
                fold_labels = positional_shuffle(
                    ye[allowed],
                    ge[allowed],
                    seed=shuffle_seed + inner_fold
                )

            est = fit_joint(
                Xv[it],
                yv[it],
                Xe[allowed],
                fold_labels,
                weight
            )

            preds[iv] = est.predict(Xv[iv])

        scores[weight] = previous.pos_mae(
            yv, preds, gv
        )

    choice = min(
        weights,
        key=lambda w: (scores[w], w)
    )

    return choice, scores[choice]


def benchmark(vti, V80, eti, E80, folds, shuffles):
    keys = vti.variant_key.astype(str).to_numpy()
    vp = vti.protein_position.to_numpy(int)
    vy = vti.corrector_rescue_pp.to_numpy(float)

    ey = eti.corrector_rescue_pp.to_numpy(float)
    ep = eti.protein_position.to_numpy(int)

    # Treatment identity:
    # 1 = VTI, 0 = ETI.
    # Prediction always uses VTI identity.
    vx = np.column_stack([
        V80, np.ones(len(V80))
    ])

    ex = np.column_stack([
        E80, np.zeros(len(E80))
    ])

    rows = []
    choices = []

    for f, (rep, fold, trkeys, tekeys) in enumerate(
        folds, 1
    ):
        it = np.flatnonzero(
            np.isin(keys, list(trkeys))
        )

        iv = np.flatnonzero(
            np.isin(keys, list(tekeys))
        )

        assert len(it) + len(iv) == len(keys)

        heldout = set(vp[iv])

        assert heldout.isdisjoint(set(vp[it]))

        # Exclude ETI labels at the outer-held-out
        # VTI protein positions.
        aux = ~np.isin(
            ep,
            np.array(sorted(heldout), int)
        )

        assert not heldout.intersection(set(ep[aux]))

        if not aux.any():
            raise ValueError("Empty auxiliary training fold")

        # Real ETI supervision.
        wt, score = choose_weight(
            vx[it], vy[it], vp[it],
            ex[aux], ey[aux], ep[aux]
        )

        real = fit_joint(
            vx[it], vy[it],
            ex[aux], ey[aux], wt
        ).predict(vx[iv])

        predictions = {
            "joint_real": real
        }

        choices.append({
            "repeat": rep,
            "fold": fold,
            "model": "joint_real",
            "eti_weight": wt,
            "inner_pos_mae_pp": score,
            "outer_train_vti_positions": len(np.unique(vp[it])),
            "outer_test_positions": len(heldout),
            "outer_eti_variants": int(aux.sum())
        })

        # Negative controls.
        for k in range(shuffles):
            sham_seed = 2608 + 100 * f + k

            surrogate = positional_shuffle(
                ey[aux],
                ep[aux],
                seed=sham_seed
            )

            sw, ss = choose_weight(
                vx[it], vy[it], vp[it],
                ex[aux], ey[aux], ep[aux],
                shuffle_seed=sham_seed
            )

            name = f"joint_sham{k + 1}"

            predictions[name] = fit_joint(
                vx[it], vy[it],
                ex[aux], surrogate, sw
            ).predict(vx[iv])

            choices.append({
                "repeat": rep,
                "fold": fold,
                "model": name,
                "eti_weight": sw,
                "inner_pos_mae_pp": ss,
                "outer_train_vti_positions": len(np.unique(vp[it])),
                "outer_test_positions": len(heldout),
                "outer_eti_variants": int(aux.sum())
            })

        for j, i in enumerate(iv):
            row = {
                "repeat": rep,
                "fold": fold,
                "variant_key": keys[i],
                "protein_position": int(vp[i]),
                "study_cohort": vti.study_cohort.iloc[i],
                "observed_corrector": float(vy[i]),
                "observed_total": float(
                    vti.vti_rescue_pp.iloc[i]
                )
            }

            row.update({
                name: float(pred[j])
                for name, pred in predictions.items()
            })

            rows.append(row)

        print(
            f"completed {f}/{len(folds)}: "
            f"repeat={rep}, fold={fold}, "
            f"test_positions={len(heldout)}, "
            f"ETI_train={aux.sum()}, "
            f"selected_weight={wt}",
            flush=True
        )

    return pd.DataFrame(rows), pd.DataFrame(choices)


def paired_gain(df, candidate, reference, target, nboot, seed):
    diff = (
        np.abs(df[target] - df[reference])
        - np.abs(df[target] - df[candidate])
    )

    g = df[["repeat", "protein_position"]].copy()
    g["gain"] = diff

    values = (
        g.groupby(["repeat", "protein_position"])
        .gain.mean()
        .groupby("protein_position")
        .mean()
        .to_numpy()
    )

    rng = np.random.default_rng(seed)

    sims = np.mean(
        rng.choice(
            values,
            size=(nboot, len(values)),
            replace=True
        ),
        axis=1
    )

    return {
        "candidate": candidate,
        "reference": reference,
        "target": target,
        "gain_pp": float(values.mean()),
        "ci95_low": float(np.quantile(sims, .025)),
        "ci95_high": float(np.quantile(sims, .975)),
        "n_independent_positions": len(values)
    }


def report(out, choices, nboot, shuffles):
    base = pd.read_csv(
        ROOT / "results/biology/phase26d/phase26d_oof.csv",
        dtype={"variant_key": str}
    )

    keys = [
        "repeat", "fold", "variant_key", "protein_position"
    ]

    need = keys + [
        "observed_total",
        "observed_corrector",
        "pred_zero_corrector_seq80",
        "pred_zero_acute_seq80",
        "pred_zero_components_seq80",
        "pred_zero_total_seq80"
    ]

    if not set(need).issubset(base.columns):
        raise ValueError(
            "Phase26D OOF missing required columns"
        )

    base = base[need].copy()

    out = out.merge(
        base,
        on=keys,
        validate="one_to_one",
        suffixes=("", "_baseline"),
        indicator=True
    )

    if (
        not out._merge.eq("both").all()
        or len(out) != len(base)
    ):
        raise ValueError(
            "Phase26D frozen OOF membership mismatch"
        )

    out = out.drop(columns=["_merge"])

    for col in ["observed_total", "observed_corrector"]:
        if not np.allclose(
            out[col],
            out[col + "_baseline"],
            atol=1e-8
        ):
            raise ValueError(
                "Outcome mismatch with Phase26D"
            )

        out = out.drop(
            columns=[col + "_baseline"]
        )

    # Reconstruct total rescue using frozen
    # out-of-fold acute-IVA predictions.
    names = ["joint_real"] + [
        f"joint_sham{k + 1}"
        for k in range(shuffles)
    ]

    for name in names:
        out[name + "_total"] = (
            out[name]
            + out.pred_zero_acute_seq80
        )

    tasks = {
        "Corrector": [
            "pred_zero_corrector_seq80",
            *names
        ],
        "Total": [
            "pred_zero_total_seq80",
            "pred_zero_components_seq80",
            *[name + "_total" for name in names]
        ]
    }

    metrics = []

    for rep, g in out.groupby("repeat"):
        for target, models in tasks.items():
            y = g[
                "observed_" + target.lower()
            ].to_numpy(float)

            pos = g.protein_position.to_numpy(int)

            for name in models:
                pred = g[name].to_numpy(float)
                error = y - pred
                tail = y > 40

                metrics.append({
                    "repeat": rep,
                    "target": target,
                    "model": name,
                    "position_mae_pp": previous.pos_mae(
                        y, pred, pos
                    ),
                    "variant_mae_pp": float(
                        np.mean(np.abs(error))
                    ),
                    "rmse_pp": float(
                        np.sqrt(np.mean(error ** 2))
                    ),
                    "gt40_mae_pp": (
                        float(np.mean(np.abs(error[tail])))
                        if tail.any() else np.nan
                    ),
                    "n_gt40": int(tail.sum())
                })

    metrics = pd.DataFrame(metrics)

    comparisons = []

    pairs = [
        (
            "joint_real",
            "pred_zero_corrector_seq80",
            "observed_corrector"
        ),
        (
            "joint_real_total",
            "pred_zero_components_seq80",
            "observed_total"
        ),
        (
            "joint_real_total",
            "pred_zero_total_seq80",
            "observed_total"
        )
    ]

    for candidate, reference, target in pairs:
        comparisons.append(
            paired_gain(
                out, candidate, reference,
                target, nboot, 2608
            )
        )

    for k in range(shuffles):
        comparisons.append(
            paired_gain(
                out,
                "joint_real",
                f"joint_sham{k + 1}",
                "observed_corrector",
                nboot,
                2609 + k
            )
        )

    comp = pd.DataFrame(comparisons)

    OUT.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        OUT / "phase26h_oof.csv",
        index=False
    )

    choices.to_csv(
        OUT / "phase26h_inner_selections.csv",
        index=False
    )

    metrics.to_csv(
        OUT / "phase26h_metrics_by_repeat.csv",
        index=False
    )

    comp.to_csv(
        OUT / "phase26h_bootstrap.csv",
        index=False
    )

    print("\nA. MATCHED POSITION-LEVEL PERFORMANCE")

    summary = (
        metrics.groupby(
            ["target", "model"], as_index=False
        )[
            [
                "position_mae_pp",
                "rmse_pp",
                "gt40_mae_pp"
            ]
        ]
        .mean()
        .sort_values(["target", "position_mae_pp"])
    )

    print(
        summary.round(4).to_string(index=False)
    )

    print("\nB. PAIRED POSITION BOOTSTRAP")
    print("Positive gain = proposed model improved")

    print(
        comp.round(4).to_string(index=False)
    )

    print("\nC. INNER-CV ETI WEIGHT SELECTION")

    print(
        choices.groupby(
            ["model", "eti_weight"]
        ).size().rename("outer_splits").to_string()
    )

    print("\nD. OUTER-FOLD TRAINING DETAILS")

    print(
        choices.round(4).to_string(index=False)
    )

    print(
        "\nPASS: Phase26H-1 completed; "
        "TRAIN-only; outputs:", OUT
    )


def self_test():
    rng = np.random.default_rng(26)

    gp = np.repeat(np.arange(36), 2)
    n = len(gp)

    X = rng.normal(size=(n, 81))
    X[:, -1] = 1

    yc = (
        X[:, 0] * 4
        + X[:, 1] * 2
        + rng.normal(size=n)
    )

    eg = np.repeat(np.arange(44), 2)

    E = rng.normal(size=(len(eg), 81))
    E[:, -1] = 0

    ey = (
        E[:, 0] * 4
        + E[:, 1] * 2
        + rng.normal(size=len(E))
    )

    eysh = positional_shuffle(ey, eg, 3)

    assert len(eysh) == len(ey)
    assert np.isfinite(eysh).all()

    folds = list(
        GroupKFold(3).split(X, yc, gp)
    )

    for it, iv in folds:
        aux = ~np.isin(
            eg,
            np.unique(gp[iv])
        )

        weight, score = choose_weight(
            X[it], yc[it], gp[it],
            E[aux], ey[aux], eg[aux],
            weights=(0.25,)
        )

        mdl = fit_joint(
            X[it], yc[it],
            E[aux], ey[aux],
            weight
        )

        assert np.isfinite(
            mdl.predict(X[iv])
        ).all()

        assert np.isfinite(score)

        sw, ss = choose_weight(
            X[it], yc[it], gp[it],
            E[aux], ey[aux], eg[aux],
            weights=(0.25,),
            shuffle_seed=999
        )

        assert np.isfinite(ss)

    print(
        "PASS synthetic Phase26H: "
        "grouped CV, weighted joint regression, "
        "nested ETI exclusion, shuffled controls"
    )


def main():
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

    if not 1 <= args.shuffles <= 10:
        raise ValueError("--shuffles must be 1..10")

    if args.self_test:
        return self_test()

    print("=" * 100)
    print(
        "PHASE 26H-1 | ZERO-SHOT VTI CORRECTOR "
        "PREDICTION WITH ETI AUXILIARY SUPERVISION"
    )
    print("=" * 100)

    print(
        "Mutation-only prediction; ETI functional assays "
        "are training labels, not prediction inputs."
    )

    print(
        "Frozen position-disjoint outer folds; "
        "ETI supervision excluded at outer/inner "
        "held-out positions."
    )

    print(
        f"Fixed XGBoost={SPEC}; ETI weights selected "
        f"in inner grouped CV; shuffles={args.shuffles}"
    )

    print(
        "TRAIN ONLY. Protected ETI/VTI outcomes unopened."
    )

    vti, V80, eti, E80, folds = load_data()

    oof, choices = benchmark(
        vti, V80, eti, E80,
        folds, args.shuffles
    )

    report(
        oof, choices,
        args.bootstraps,
        args.shuffles
    )


if __name__ == "__main__":
    main()
