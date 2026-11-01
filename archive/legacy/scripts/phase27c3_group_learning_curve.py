#!/usr/bin/env python3
"""Phase 27C-3: nested position-group learning curves, VTI TRAIN only.

Keep the original two-by-five position-disjoint outer evaluation folds fixed.
Within each outer training fold, use successively larger nested, approximately
class-balanced subsets of *positions*. Retain the exact two established binary
classifiers, features, and hyperparameters. Never use VTI VAL/TEST labels.

Report development-only performance; repeated subsets and OOF predictions
are NOT independent biological replication.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results/biology/phase27c"
FROZEN = ROOT / "results/biology/phase27b/phase27b1_repeated_oof.csv"
FRACTIONS = (0.40, 0.60, 0.80, 1.00)
TASKS = (("corrector_low", "xgb_seq80"),
         ("corrector_high", "logistic_seq68"))


def label_three(y):
    y = np.asarray(y, float)
    return np.where(y < 10, 0, np.where(y > 40, 2, 1)).astype(int)


def nested_position_order(positions, y, seed):
    """Balanced, fully nested position ordering, decided from training only.

    Priority high > low > middle if multiple substitutions at one position
    belong to different responder classes. Actual variant class counts are
    audited separately. This is for training-group selection only.
    """
    positions = np.asarray(positions, int)
    classes = label_three(y)
    unique = np.unique(positions)
    buckets = {k: [] for k in (0, 1, 2)}
    for p in unique:
        present = set(classes[positions == p].tolist())
        kind = 2 if 2 in present else 0 if 0 in present else 1
        buckets[kind].append(int(p))

    rng = np.random.default_rng(seed)
    for k in buckets:
        rng.shuffle(buckets[k])

    target = np.array([len(buckets[k]) for k in (0, 1, 2)], float)
    ratio = target / target.sum()
    taken = np.zeros(3, int)
    order = []
    for step in range(1, len(unique) + 1):
        deficit = ratio * step - taken
        for k in (0, 1, 2):
            if taken[k] >= len(buckets[k]):
                deficit[k] = -np.inf
        k = int(np.argmax(deficit))
        order.append(buckets[k][taken[k]])
        taken[k] += 1
    if len(order) != len(unique) or len(set(order)) != len(unique):
        raise AssertionError("Position order failed uniqueness check")
    return np.asarray(order, int)


def summarise(q):
    y = q.label.to_numpy(int)
    score = q.score.to_numpy(float)
    if len(np.unique(y)) != 2 or not np.isfinite(score).all():
        raise ValueError("One-class or nonfinite OOF predictions")
    n = int(np.ceil(0.2 * len(q)))
    hit_idx = np.argsort(-score, kind="stable")[:n]
    return dict(n=int(len(q)), positives=int(y.sum()),
                ap=float(average_precision_score(y, score)),
                auroc=float(roc_auc_score(y, score)),
                top20_hits=int(y[hit_idx].sum()),
                top20_recall=float(y[hit_idx].sum()/y.sum()))


def self_test():
    # 30 grouped positions, two variants each. Test nested counts and balance.
    p = np.repeat(np.arange(30), 2)
    y = np.repeat(np.array([5.]*8 + [20.]*17 + [45.]*5), 2)
    a = nested_position_order(p, y, 271)
    b = nested_position_order(p, y, 271)
    assert np.array_equal(a, b)
    assert sorted(a.tolist()) == list(range(30))
    assert set(a[:12]).issubset(set(a[:18]))
    assert set(a[:18]).issubset(set(a[:24]))
    assert len(set(a[:12][np.isin(a[:12], range(25, 30))])) >= 1
    q = pd.DataFrame({"label": [0, 0, 1, 1], "score": [.1, .2, .8, .9]})
    assert summarise(q)["ap"] == 1.0
    print("PASS synthetic: group integrity, nested subsets, high-class coverage, OOF metrics")


def run(subsamples):
    import phase26h_auxiliary_corrector as h
    import phase27b1_extreme_classifiers as b

    if not FROZEN.is_file():
        raise FileNotFoundError(f"Missing Phase27B-1 reference: {FROZEN}")
    OUT.mkdir(parents=True, exist_ok=True)
    v, X, _, _, folds = h.load_data()
    X = np.asarray(X, float)
    vk = v.variant_key.astype(str).to_numpy()
    groups = v.protein_position.to_numpy(int)
    yc = v.corrector_rescue_pp.to_numpy(float)
    if (len(v) != 210 or X.shape != (210, 80) or
        len(np.unique(groups)) != 168 or len(folds) != 10):
        raise ValueError("Unexpected fixed VTI TRAIN data or outer folds")
    if not np.isfinite(X[:, :68]).all() or not np.isfinite(yc).all():
        raise ValueError("Invalid sequence features or TRAIN labels")
    if not np.array_equal(np.bincount(label_three(yc), minlength=3),
                          [52, 129, 29]):
        raise ValueError("Class counts changed from precommitted training data")

    reference = pd.read_csv(FROZEN, dtype={"variant_key": str})
    if len(reference) != 1680 or reference.duplicated(
        ["repeat", "fold", "variant_key", "task"]
    ).any():
        raise ValueError("Frozen Phase27B1 OOF is incomplete")

    print("=" * 104)
    print("PHASE 27C-3 | FIXED-OUTER-POSITION LEARNING CURVES | TRAIN ONLY")
    print("=" * 104)
    print("Original frozen 2x5 outer folds, identical held-out positions at EVERY size")
    print("Nested, class-balanced group subsamples of each OUTER TRAIN partition")
    print("Same hyperparameters: low XGB seq80; high logistic seq68")
    print("Fractions of OUTER TRAIN positions:", FRACTIONS)
    print("Subset replicates:", subsamples)
    print("No VTI VAL/TEST labels; no fabricated biological variants")
    print("Not an independent prospective validation or model-selection endpoint")

    pred = []
    train_records = []
    full_parity = []

    for fold_id, (rep, fold, train_keys, test_keys) in enumerate(folds, 1):
        tr_all = np.flatnonzero(np.isin(vk, list(train_keys)))
        te = np.flatnonzero(np.isin(vk, list(test_keys)))
        if len(tr_all) + len(te) != 210 or set(groups[tr_all]) & set(groups[te]):
            raise ValueError(f"Outer position leakage, fold {fold_id}")
        if len(tr_all) == 0 or len(te) == 0:
            raise ValueError("Empty outer fold")
        frozen_test = reference.loc[
            (reference.repeat == rep) & (reference.fold == fold)
        ]
        for task, method in TASKS:
            q = frozen_test.loc[frozen_test.task == task]
            if set(q.variant_key.astype(str)) != set(vk[te]):
                raise ValueError("Frozen OOF test identities mismatch")

        for subset_seed in range(subsamples):
            order = nested_position_order(
                groups[tr_all], yc[tr_all],
                seed=273030 + 1000 * fold_id + subset_seed,
            )
            previous = set()
            for fraction in FRACTIONS:
                npos = max(1, int(round(fraction * len(order))))
                chosen_pos = set(order[:npos].tolist())
                if not previous.issubset(chosen_pos):
                    raise ValueError("Nested position subset violation")
                previous = chosen_pos
                tr = tr_all[np.isin(groups[tr_all], order[:npos])]
                if len(set(groups[tr])) != npos:
                    raise ValueError("Selected position-count mismatch")
                if set(groups[tr]) & set(groups[te]):
                    raise ValueError("TRAIN/TEST group overlap")
                cls = label_three(yc[tr])
                counts = np.bincount(cls, minlength=3)
                if counts.min() < 1:
                    raise ValueError(
                        f"Training subset missing a class: fold={fold_id}, "
                        f"seed={subset_seed}, fraction={fraction}, counts={counts}"
                    )
                train_records.append(dict(
                    repeat=int(rep), fold=int(fold), subset_seed=int(subset_seed),
                    train_fraction=float(fraction), train_variants=int(len(tr)),
                    train_positions=int(npos), train_low=int(counts[0]),
                    train_middle=int(counts[1]), train_high=int(counts[2]),
                    test_variants=int(len(te)), test_positions=int(len(set(groups[te]))),
                ))

                for task, method in TASKS:
                    labels = (yc < 10).astype(int) if task == "corrector_low" else (yc > 40).astype(int)
                    if len(np.unique(labels[tr])) != 2:
                        raise ValueError("No training positives or negatives")
                    # Same seed as original 27B1: class index 0=low, 1=high.
                    task_idx = 0 if task == "corrector_low" else 1
                    model_seed = 27000 + task_idx * 100 + fold_id
                    model = b.make_model(method, labels[tr], model_seed)
                    use = X[:, :68] if method == "logistic_seq68" else X
                    model.fit(use[tr], labels[tr])
                    probs = model.predict_proba(use[te])[:, 1]
                    if not np.isfinite(probs).all():
                        raise ValueError("Nonfinite predictions")
                    for ii, score in zip(te, probs):
                        pred.append(dict(
                            repeat=int(rep), fold=int(fold), subset_seed=int(subset_seed),
                            train_fraction=float(fraction),
                            variant_key=vk[ii], protein_position=int(groups[ii]),
                            task=task, label=int(labels[ii]), score=float(score),
                        ))

                    # Verify exact parity with original Phase27B1 when full positions used.
                    if fraction == 1.0:
                        expected = frozen_test.loc[frozen_test.task == task].set_index("variant_key")
                        compare = expected.loc[vk[te], method].to_numpy(float)
                        error = float(np.max(np.abs(probs - compare)))
                        full_parity.append(dict(repeat=rep, fold=fold,
                                                subset_seed=subset_seed, task=task,
                                                maximum_abs_probability_error=error))
                        if error > 1e-5:
                            raise ValueError(
                                f"Full-training model fails parity with original OOF "
                                f"({task}, repeat {rep}, fold {fold}, max error {error:.6g}). "
                                "Investigate model versions/feature order before comparison."
                            )
        print(f"completed outer fold {fold_id}/10 | rep={rep}, fold={fold}, "
              f"test positions={len(set(groups[te]))}", flush=True)

    d = pd.DataFrame(pred)
    audit = pd.DataFrame(train_records)
    parity = pd.DataFrame(full_parity)
    key = ["repeat", "subset_seed", "train_fraction", "variant_key", "task"]
    if d.duplicated(key).any():
        raise ValueError("OOF records duplicated")
    expected = len(FRACTIONS) * subsamples * 2 * 210 * len(TASKS)
    if len(d) != expected:
        raise ValueError(f"OOF count {len(d)} != expected {expected}")
    if not np.isfinite(d.score.to_numpy(float)).all():
        raise ValueError("Nonfinite results")

    rep_metrics = pd.DataFrame([
        dict(train_fraction=float(fr), subset_seed=int(s), repeat=int(rep), task=task, **summarise(q))
        for (fr, s, rep, task), q in d.groupby(
            ["train_fraction", "subset_seed", "repeat", "task"]
        )
    ])

    # Each mutation has two predictions, one per original repeat. Average exactly those two,
    # then compute pooled AP for each independent subset seed; never pool as 420 cases.
    avg = d.groupby(
        ["train_fraction", "subset_seed", "task", "variant_key", "protein_position", "label"],
        as_index=False,
    ).score.mean()
    per_seed = pd.DataFrame([
        dict(train_fraction=float(fr), subset_seed=int(s), task=task, **summarise(q))
        for (fr, s, task), q in avg.groupby(["train_fraction", "subset_seed", "task"])
    ])

    curve = per_seed.groupby(["task", "train_fraction"], as_index=False).agg(
        ap_mean=("ap", "mean"), ap_sd=("ap", "std"),
        auroc_mean=("auroc", "mean"), auroc_sd=("auroc", "std"),
        top20_hits_mean=("top20_hits", "mean"),
        top20_hits_min=("top20_hits", "min"),
        top20_hits_max=("top20_hits", "max"),
    )
    for task in curve.task.unique():
        base = per_seed.loc[
            (per_seed.task == task) & (per_seed.train_fraction == 1.0)
        ].sort_values("subset_seed").ap.to_numpy(float)
        if not np.allclose(base, base[0], atol=1e-9):
            raise ValueError("100% training results vary across subset seeds")

    diag = audit.groupby("train_fraction", as_index=False).agg(
        train_positions_mean=("train_positions", "mean"),
        train_variants_mean=("train_variants", "mean"),
        train_high_min=("train_high", "min"), train_high_median=("train_high", "median"),
        train_high_max=("train_high", "max"),
        train_low_min=("train_low", "min"), train_low_median=("train_low", "median"),
        train_low_max=("train_low", "max"),
    )

    change = per_seed.merge(
        per_seed.loc[per_seed.train_fraction == 1.0,
                     ["subset_seed", "task", "ap", "auroc"]].rename(
            columns={"ap": "full_ap", "auroc": "full_auroc"}),
        on=["subset_seed", "task"], validate="many_to_one",
    )
    change["ap_gain_vs_full"] = change.ap - change.full_ap
    change["auroc_gain_vs_full"] = change.auroc - change.full_auroc

    d.to_csv(OUT / "phase27c3_repeated_oof.csv", index=False)
    avg.to_csv(OUT / "phase27c3_two_repeat_avg_oof.csv", index=False)
    audit.to_csv(OUT / "phase27c3_training_counts.csv", index=False)
    rep_metrics.to_csv(OUT / "phase27c3_per_repeat_metrics.csv", index=False)
    per_seed.to_csv(OUT / "phase27c3_per_subsample_metrics.csv", index=False)
    curve.to_csv(OUT / "phase27c3_learning_curve.csv", index=False)
    change.to_csv(OUT / "phase27c3_paired_deltas.csv", index=False)
    parity.to_csv(OUT / "phase27c3_full_training_parity.csv", index=False)

    print("\nA. GROUPED TRAINING-SUBSET QC")
    print(diag.round(3).to_string(index=False))
    print("\nB. TWO-REPEAT AVERAGED OOF LEARNING CURVE (mean/sd over subset seeds)")
    print(curve.round(4).to_string(index=False))
    print("\nC. SUBSAMPLE-SPECIFIC METRICS AND PAIRED DELTAS FROM FULL TRAINING")
    print(change[["task", "train_fraction", "subset_seed", "ap", "auroc",
                  "top20_hits", "ap_gain_vs_full", "auroc_gain_vs_full"]]
          .sort_values(["task", "subset_seed", "train_fraction"])
          .round(4).to_string(index=False))
    print("\nD. 100% TRAINING REPRODUCTION QC")
    print("Max absolute error vs original 27B1 OOF:",
          f"{parity.maximum_abs_probability_error.max():.8g}")
    print("\nInterpretation: fixed outer folds isolate the effect of removing training positions;")
    print("this measures data sensitivity, not prospective benefit from new positions.")
    print("Many fraction/seed comparisons are exploratory; do not pick a lucky subset.")
    print("PASS: Phase27C-3 completed | TRAIN only | Saved:", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--subsamples", type=int, default=3)
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        if not (1 <= args.subsamples <= 20):
            ap.error("--subsamples must be between 1 and 20")
        run(args.subsamples)
