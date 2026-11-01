"""Phase 19C: nested, position-disjoint acute IVA prediction benchmark."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cftr_theratype import model_zoo as mz
from cftr_theratype.transfer import ROOT


FEATURES = ("sequence", "sequence_structure")
FAMILIES = ("extra_trees", "xgboost")
TARGETS = ("acute_iva_vti_pp", "acute_iva_change_pp")


def position_mae(y, pred, positions):
    errors = np.abs(np.asarray(y) - np.asarray(pred))
    groups = np.asarray(positions)
    return float(np.mean([
        errors[groups == p].mean()
        for p in np.unique(groups)
    ]))


def main(smoke=False, target="acute_iva_vti_pp"):
    assert target in TARGETS

    print("=" * 96, flush=True)
    print("CFTR-THERATYPE | PHASE 19C — ACUTE IVACAFTOR NESTED CV", flush=True)
    print("=" * 96, flush=True)

    train, mats, _, outer = mz.prepare(ROOT)

    target_path = (
        ROOT / "results/boltz/phase19b/"
        "vti_train_acute_iva_targets.csv"
    )
    assert target_path.exists()

    labels = pd.read_csv(target_path)

    assert len(train) == len(labels) == 210
    assert train.variant_key.is_unique
    assert labels.variant_key.is_unique
    assert set(train.variant_key) == set(labels.variant_key)

    aligned = train[["variant_key", "protein_position"]].merge(
        labels[["variant_key", "protein_position", target]],
        on="variant_key",
        suffixes=("", "_target"),
        validate="one_to_one",
        sort=False,
    )

    assert np.array_equal(
        aligned.protein_position.to_numpy(),
        aligned.protein_position_target.to_numpy(),
    )

    assert aligned.variant_key.tolist() == train.variant_key.tolist()

    y = aligned[target].to_numpy(float)
    positions = train.protein_position.to_numpy(int)
    keys = train.variant_key.to_numpy()

    assert np.isfinite(y).all()
    assert len(np.unique(positions)) == 168
    assert len(outer) == 10

    for feature in FEATURES:
        assert feature in mats
        assert mats[feature].shape[0] == 210

    print("Target:", target, flush=True)
    print("TRAIN:", len(y), "variants,", len(np.unique(positions)),
          "positions", flush=True)
    print("Outer CV: original 2 x 5 frozen position-disjoint folds", flush=True)
    print("Inner CV: 3-fold position-disjoint model selection", flush=True)
    print("Families:", FAMILIES, flush=True)
    print("Features:", FEATURES, flush=True)
    print("Protected validation/test outcomes: NOT LOADED", flush=True)

    predictions = []
    selections = []

    folds = outer[:1] if smoke else outer

    for rep, fold, fit, held in folds:
        assert not set(positions[fit]) & set(positions[held])

        print(
            f"\n[OUTER] repeat={rep} fold={fold} "
            f"fit={len(fit)} held={len(held)}",
            flush=True,
        )

        # Constant median baseline, fit exclusively on outer training rows.
        median = float(np.median(y[fit]))
        median_pred = np.full(len(held), median)

        print(
            f"  {'median_baseline':39s} "
            f"position-MAE={position_mae(y[held], median_pred, positions[held]):.3f}",
            flush=True,
        )

        for idx, prediction in zip(held, median_pred):
            predictions.append({
                "repeat": rep,
                "fold": fold,
                "model": "median_baseline",
                "variant_key": str(keys[idx]),
                "protein_position": int(positions[idx]),
                "observed_pp": float(y[idx]),
                "predicted_pp": float(prediction),
            })

        for feature in FEATURES:
            X = mats[feature]

            for family in FAMILIES:
                seed = mz.SEED + rep * 100 + fold

                chosen, inner_mae = mz.inner_select(
                    X[fit],
                    y[fit],
                    positions[fit],
                    family,
                    seed=seed,
                )

                reg = mz.estimator(family, chosen, seed=seed)
                reg.fit(X[fit], y[fit])

                pred = np.asarray(
                    reg.predict(X[held]), dtype=float
                )

                assert pred.shape == (len(held),)
                assert np.isfinite(pred).all()

                name = f"{feature}__{family}"
                outer_mae = position_mae(
                    y[held], pred, positions[held]
                )

                print(
                    f"  {name:39s} "
                    f"inner={inner_mae:.3f} "
                    f"outer={outer_mae:.3f} "
                    f"params={chosen}",
                    flush=True,
                )

                selections.append({
                    "repeat": rep,
                    "fold": fold,
                    "model": name,
                    "inner_position_mae_pp": float(inner_mae),
                    "outer_position_mae_pp": outer_mae,
                    "hyperparams": json.dumps(chosen, sort_keys=True),
                })

                for idx, prediction in zip(held, pred):
                    predictions.append({
                        "repeat": rep,
                        "fold": fold,
                        "model": name,
                        "variant_key": str(keys[idx]),
                        "protein_position": int(positions[idx]),
                        "observed_pp": float(y[idx]),
                        "predicted_pp": float(prediction),
                    })

    if smoke:
        print("\nPASS: Smoke test; no outputs written", flush=True)
        return

    oof = pd.DataFrame(predictions)

    # Every variant must have one prediction per CV repeat, per model.
    assert not oof.duplicated(
        ["repeat", "model", "variant_key"]
    ).any()

    avg = (
        oof.groupby(["model", "variant_key"], as_index=False)
        .agg(
            protein_position=("protein_position", "first"),
            observed_pp=("observed_pp", "first"),
            predicted_pp=("predicted_pp", "mean"),
            n_repeats=("repeat", "nunique"),
        )
    )

    assert len(avg) == 5 * 210
    assert (avg.n_repeats == 2).all()

    summary = []

    for name, frame in avg.groupby("model"):
        actual = frame.observed_pp.to_numpy(float)
        prediction = frame.predicted_pp.to_numpy(float)
        pos = frame.protein_position.to_numpy(int)

        summary.append({
            "model": name,
            "position_mae_pp": position_mae(actual, prediction, pos),
            "variant_mae_pp": float(np.mean(np.abs(actual - prediction))),
            "rmse_pp": float(np.sqrt(np.mean((actual - prediction) ** 2))),
            "mean_bias_pp": float(np.mean(prediction - actual)),
        })

    summary = pd.DataFrame(summary).sort_values("position_mae_pp")

    print("\n" + "=" * 96, flush=True)
    print("PHASE 19C | REPEATED OOF RESULTS", flush=True)
    print("=" * 96, flush=True)
    print("Predictions averaged across repeats before scoring.")
    print("Primary metric: macro-average MAE across 168 protein positions.\n")
    print(summary.round(4).to_string(index=False))

    baseline = float(
        summary.loc[
            summary.model.eq("median_baseline"),
            "position_mae_pp",
        ].iloc[0]
    )

    summary["gain_vs_median_pp"] = (
        baseline - summary.position_mae_pp
    )

    print("\nGAIN VERSUS MEDIAN BASELINE")
    print(
        summary[[
            "model", "position_mae_pp", "gain_vs_median_pp"
        ]].round(4).to_string(index=False)
    )

    out = ROOT / "results/boltz/phase19c"
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "oof": out / f"{target}_oof.csv",
        "averaged": out / f"{target}_averaged.csv",
        "summary": out / f"{target}_summary.csv",
        "selection": out / f"{target}_inner_selection.csv",
    }

    oof.to_csv(paths["oof"], index=False)
    avg.to_csv(paths["averaged"], index=False)
    summary.to_csv(paths["summary"], index=False)
    pd.DataFrame(selections).to_csv(paths["selection"], index=False)

    print("\nOUTPUTS")
    for path in paths.values():
        print(" ", path.relative_to(ROOT))

    print("\nPASS: Frozen position-disjoint nested CV complete")
    print("PASS: Protected validation/test outcomes never loaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--target",
        choices=TARGETS,
        default="acute_iva_vti_pp",
    )
    args = parser.parse_args()

    main(smoke=args.smoke, target=args.target)
