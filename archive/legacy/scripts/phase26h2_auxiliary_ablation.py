#!/usr/bin/env python3
"""Phase 26H-2: isolate where ETI auxiliary transfer originates.

Train-only, mutation-derived inference.
Same Phase26D frozen protein-position folds.
Reuse Phase26H-1 predictions for the all-ETI model.
Exclude ETI labels at all outer/inner held-out VTI positions.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

import phase26h_auxiliary_corrector as earlier

OUT = ROOT / "results/biology/phase26h2"
BASE = ROOT / "results/biology/phase26h/phase26h_oof.csv"


def position_mae(df, target, prediction):
    errors = (df[target] - df[prediction]).abs()

    return float(
        pd.DataFrame({
            "position": df.protein_position,
            "error": errors
        }).groupby("position").error.mean().mean()
    )


def gain_ci(df, candidate, reference, target, nboot, seed):
    q = df[
        ["repeat", "protein_position",
         target, candidate, reference]
    ].copy()

    q["gain"] = (
        (q[target] - q[reference]).abs()
        - (q[target] - q[candidate]).abs()
    )

    values = (
        q.groupby(["repeat", "protein_position"])
        .gain.mean()
        .groupby("protein_position")
        .mean()
        .to_numpy(float)
    )

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        values,
        size=(nboot, len(values)),
        replace=True
    ).mean(axis=1)

    return {
        "candidate": candidate,
        "reference": reference,
        "target": target,
        "gain_pp": float(values.mean()),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "n_independent_positions": len(values)
    }


def run(nsham, nboot):
    if not BASE.is_file():
        raise FileNotFoundError(
            f"Required Phase26H-1 OOF missing: {BASE}"
        )

    vti, V80, eti, E80, folds = earlier.load_data()

    vk = vti.variant_key.astype(str).to_numpy()
    ek = eti.variant_key.astype(str).to_numpy()

    vp = vti.protein_position.to_numpy(int)
    ep = eti.protein_position.to_numpy(int)

    vy = vti.corrector_rescue_pp.to_numpy(float)
    ey = eti.corrector_rescue_pp.to_numpy(float)

    novel = ~np.isin(ep, np.unique(vp))
    seen = ~novel
    exact = np.isin(ek, vk)

    assert len(vti) == 210
    assert len(eti) == 383
    assert int(novel.sum()) == 258
    assert int(exact.sum()) == 66
    assert not np.any(novel & exact)

    print("=" * 100)
    print(
        "PHASE 26H-2 | ETI AUXILIARY DATA SOURCE ABLATION"
    )
    print("=" * 100)

    print(
        f"ETI auxiliary: {len(eti)} variants / "
        f"{np.unique(ep).size} positions"
    )

    print(
        f"Novel VTI positions: {novel.sum()} variants / "
        f"{np.unique(ep[novel]).size} positions"
    )

    print(
        f"VTI-represented positions: {seen.sum()} variants / "
        f"{np.unique(ep[seen]).size} positions"
    )

    print(
        f"Exact matched variants: {exact.sum()} | "
        f"Other same-position variants: "
        f"{(seen & ~exact).sum()}"
    )

    print(
        "Mutation-only inference; frozen grouped outer CV; "
        "inner grouped ETI-weight selection"
    )

    print(
        "All protected outcomes unopened. "
        "Training-set findings exploratory."
    )

    vx = np.column_stack([
        V80, np.ones(len(vti))
    ])

    ex = np.column_stack([
        E80, np.zeros(len(eti))
    ])

    records = []
    selections = []

    for nfold, (rep, fold, trkeys, tekeys) in enumerate(
        folds, 1
    ):
        it = np.flatnonzero(
            np.isin(vk, list(trkeys))
        )

        iv = np.flatnonzero(
            np.isin(vk, list(tekeys))
        )

        assert len(it) + len(iv) == len(vti)
        assert set(vp[it]).isdisjoint(set(vp[iv]))

        outer_allowed = ~np.isin(ep, vp[iv])

        # Same fixed architecture as joint learning,
        # but VTI training data only.
        baseline_est = earlier.previous.model(
            earlier.SPEC
        )

        baseline_est.fit(
            vx[it], vy[it]
        )

        models = {
            "vti_fixed": baseline_est.predict(vx[iv])
        }

        for name, subset in [
            ("novel_real", novel),
            ("seen_real", seen)
        ]:
            ia = np.flatnonzero(
                outer_allowed & subset
            )

            if len(ia) == 0:
                raise ValueError(
                    f"Empty auxiliary subset: {name}"
                )

            assert set(ep[ia]).isdisjoint(
                set(vp[iv])
            )

            weight, inner_mae = earlier.choose_weight(
                vx[it], vy[it], vp[it],
                ex[ia], ey[ia], ep[ia]
            )

            estimator = earlier.fit_joint(
                vx[it], vy[it],
                ex[ia], ey[ia],
                weight
            )

            models[name] = estimator.predict(
                vx[iv]
            )

            selections.append({
                "repeat": rep,
                "fold": fold,
                "model": name,
                "weight": weight,
                "inner_mae_pp": inner_mae,
                "eti_variants": len(ia),
                "eti_positions": np.unique(ep[ia]).size
            })

        # Negative controls on the novel-position subset.
        ia = np.flatnonzero(
            outer_allowed & novel
        )

        for s in range(nsham):
            name = f"novel_sham{s+1}"
            seed = 26100 + nfold * 100 + s

            weight, inner_mae = earlier.choose_weight(
                vx[it], vy[it], vp[it],
                ex[ia], ey[ia], ep[ia],
                shuffle_seed=seed
            )

            shuffled = earlier.positional_shuffle(
                ey[ia], ep[ia], seed=seed
            )

            estimator = earlier.fit_joint(
                vx[it], vy[it],
                ex[ia], shuffled,
                weight
            )

            models[name] = estimator.predict(
                vx[iv]
            )

            selections.append({
                "repeat": rep,
                "fold": fold,
                "model": name,
                "weight": weight,
                "inner_mae_pp": inner_mae,
                "eti_variants": len(ia),
                "eti_positions": np.unique(ep[ia]).size
            })

        part = pd.DataFrame({
            "repeat": rep,
            "fold": fold,
            "variant_key": vk[iv],
            "protein_position": vp[iv]
        })

        for name, prediction in models.items():
            part[name] = prediction

        records.append(part)

        print(
            f"Completed {nfold}/{len(folds)} | "
            f"repeat={rep} fold={fold} | "
            f"test_positions={np.unique(vp[iv]).size} | "
            f"novel_ETI={int((outer_allowed & novel).sum())} | "
            f"seen_ETI={int((outer_allowed & seen).sum())}",
            flush=True
        )

    new_predictions = pd.concat(
        records, ignore_index=True
    )

    base = pd.read_csv(
        BASE,
        dtype={"variant_key": str}
    )

    keys = [
        "repeat", "fold",
        "variant_key", "protein_position"
    ]

    required = keys + [
        "observed_corrector",
        "observed_total",
        "pred_zero_corrector_seq80",
        "pred_zero_acute_seq80",
        "pred_zero_total_seq80",
        "joint_real",
        "joint_real_total"
    ]

    if not set(required).issubset(base.columns):
        raise ValueError(
            f"Missing Phase26H-1 columns: "
            f"{set(required) - set(base.columns)}"
        )

    base = base[required].copy()

    assert not base.duplicated(keys).any()
    assert not new_predictions.duplicated(keys).any()

    if (
        len(base) != 2 * len(vti)
        or len(new_predictions) != len(base)
    ):
        raise ValueError(
            "Incomplete repeated OOF predictions"
        )

    out = base.merge(
        new_predictions,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True
    )

    if (
        len(out) != len(base)
        or not out._merge.eq("both").all()
    ):
        raise ValueError(
            "Phase26H-1 and H-2 folds do not match"
        )

    out = out.drop(columns="_merge")

    for rep, group in out.groupby("repeat"):
        if (
            len(group) != len(vti)
            or group.variant_key.nunique() != len(vti)
        ):
            raise ValueError(
                f"Incomplete outer repeat {rep}"
            )

    new_names = [
        "vti_fixed",
        "novel_real",
        "seen_real"
    ] + [
        f"novel_sham{i+1}"
        for i in range(nsham)
    ]

    for name in new_names:
        out[name + "_total"] = (
            out[name]
            + out.pred_zero_acute_seq80
        )

    tasks = [
        (
            "Corrector",
            "observed_corrector",
            ["pred_zero_corrector_seq80",
             "joint_real"] + new_names
        ),
        (
            "Total",
            "observed_total",
            ["pred_zero_total_seq80",
             "joint_real_total"]
            + [name + "_total" for name in new_names]
        )
    ]

    metrics = []

    for rep, group in out.groupby("repeat"):
        for target_name, target, models in tasks:
            y = group[target].to_numpy(float)
            high = y > 40

            for name in models:
                pred = group[name].to_numpy(float)

                metrics.append({
                    "repeat": rep,
                    "target": target_name,
                    "model": name,
                    "position_mae_pp": position_mae(
                        group, target, name
                    ),
                    "rmse_pp": float(
                        np.sqrt(np.mean((y - pred)**2))
                    ),
                    "high_gt40_mae_pp": (
                        float(np.abs(y[high] - pred[high]).mean())
                        if high.any() else np.nan
                    ),
                    "n_gt40_variants": int(high.sum())
                })

    metrics = pd.DataFrame(metrics)

    comparisons = [
        ("novel_real", "vti_fixed", "observed_corrector"),
        ("joint_real", "vti_fixed", "observed_corrector"),
        ("novel_real", "pred_zero_corrector_seq80",
         "observed_corrector"),
        ("seen_real", "vti_fixed", "observed_corrector"),
        ("seen_real", "pred_zero_corrector_seq80",
         "observed_corrector"),
        ("joint_real", "novel_real", "observed_corrector"),
        ("novel_real", "seen_real", "observed_corrector"),
        ("novel_real_total", "vti_fixed_total",
         "observed_total"),
        ("novel_real_total", "pred_zero_total_seq80",
         "observed_total"),
        ("novel_real_total", "joint_real_total",
         "observed_total")
    ] + [
        (
            "novel_real",
            f"novel_sham{i+1}",
            "observed_corrector"
        )
        for i in range(nsham)
    ]

    gains = pd.DataFrame([
        gain_ci(
            out, candidate, reference,
            target, nboot, 26082+i
        )
        for i, (candidate, reference, target)
        in enumerate(comparisons)
    ])

    OUT.mkdir(parents=True, exist_ok=True)

    out.to_csv(
        OUT / "phase26h2_oof.csv",
        index=False
    )

    pd.DataFrame(selections).to_csv(
        OUT / "phase26h2_selections.csv",
        index=False
    )

    metrics.to_csv(
        OUT / "phase26h2_metrics.csv",
        index=False
    )

    gains.to_csv(
        OUT / "phase26h2_bootstrap.csv",
        index=False
    )

    print("\nA. MATCHED PERFORMANCE BY TARGET")

    summary = (
        metrics.groupby(
            ["target", "model"], as_index=False
        )[
            [
                "position_mae_pp",
                "rmse_pp",
                "high_gt40_mae_pp"
            ]
        ]
        .mean()
        .sort_values(["target", "position_mae_pp"])
    )

    print(
        summary.round(4).to_string(index=False)
    )

    print("\nB. PAIRED POSITION BOOTSTRAP")
    print("Positive gain = candidate improves")

    print(
        gains.round(4).to_string(index=False)
    )

    print("\nC. ETI WEIGHT SELECTIONS")

    print(
        pd.DataFrame(selections)
        .groupby(["model", "weight"])
        .size()
        .rename("folds")
        .to_string()
    )

    print("\nPASS: Phase26H-2 completed")
    print("Outputs:", OUT)


def self_test():
    df = pd.DataFrame({
        "repeat": [1, 1, 2, 2],
        "protein_position": [1, 2, 1, 2],
        "y": [2., 3., 2., 3.],
        "a": [1., 4., 1., 4.],
        "b": [2., 2., 2., 2.]
    })

    result = gain_ci(
        df, "a", "b", "y", 100, 42
    )

    assert result["n_independent_positions"] == 2
    assert np.isfinite(result["gain_pp"])

    print(
        "PASS synthetic: clustered "
        "position-bootstrap unit check"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--self-test", action="store_true"
    )
    parser.add_argument(
        "--shuffles", type=int, default=3
    )
    parser.add_argument(
        "--bootstraps", type=int, default=10000
    )

    args = parser.parse_args()

    if args.self_test:
        self_test()
    else:
        if not 1 <= args.shuffles <= 10:
            parser.error(
                "--shuffles must be between 1 and 10"
            )
        if args.bootstraps < 100:
            parser.error(
                "--bootstraps must be at least 100"
            )

        run(args.shuffles, args.bootstraps)
