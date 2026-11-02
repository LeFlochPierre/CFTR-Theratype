"""Phase 15D: predict VTI <10 pp non-response explicitly, train-only.

Run: python -m cftr_theratype.nonresponse_classifier

The 210 frozen VTI *training* variants and the original repeated position-held-out
folds are the only supervised samples. Inner, position-disjoint CV chooses the
classification threshold using outer-TRAIN cases only. Both held-out partitions
remain sealed; the task is raw <10 pp rescue, not published EB categorisation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from cftr_theratype import structure_benchmark as sb

SEED = 20261023
THRESHOLD_PP = 10.0
INNER_SPLITS = 3
PROB_THRESHOLDS = np.linspace(0.10, 0.70, 13)
MIN_RESPONDER_SENSITIVITY = 0.80
CLASSIFIER_MODELS = ("sequence_balanced", "structure_balanced")
ALL_MODELS = (
    "sequence_regression", "structure_regression",
    "sequence_balanced_fixed", "sequence_balanced_inner",
    "structure_balanced_fixed", "structure_balanced_inner",
    "always_responder",
)
COHORTS = ("all", "ETI, VTI", "VTI")


def position_weights(positions: np.ndarray) -> np.ndarray:
    """Each unique residue contributes total weight one, even with multiple substitutions."""
    p = np.asarray(positions)
    if p.ndim != 1 or not len(p) or pd.isna(p).any():
        raise ValueError("Invalid position array")
    _, inverse, counts = np.unique(p, return_inverse=True, return_counts=True)
    return 1.0 / counts[inverse]


def balanced_accuracy_at_threshold(y_nr, score_nr, threshold, positions):
    truth = np.asarray(y_nr, dtype=bool)
    score = np.asarray(score_nr, dtype=float)
    w = position_weights(positions)
    if len(truth) != len(score) or len(truth) != len(w) or not np.isfinite(score).all():
        raise ValueError("Malformed threshold inputs")
    if not truth.any() or truth.all():
        raise ValueError("Both response classes are required")
    decisions = score >= float(threshold)
    recall_nr = float(w[truth & decisions].sum() / w[truth].sum())
    recall_responder = float(w[(~truth) & (~decisions)].sum() / w[~truth].sum())
    return (recall_nr + recall_responder) / 2., recall_nr, recall_responder


def choose_threshold_inner(y_nr, score_nr, positions, grid=PROB_THRESHOLDS,
                           min_responder_sensitivity=MIN_RESPONDER_SENSITIVITY) -> float:
    """Constrained inner-OOF BA optimisation; no outer held-out outcomes.

    Ties prefer a higher non-response threshold (fewer false alarms).
    Returns a number selected only using labelled inner-training OOF samples.
    """
    evaluated = []
    for threshold in grid:
        ba, recall_nr, recall_r = balanced_accuracy_at_threshold(
            y_nr, score_nr, threshold, positions)
        if recall_r + 1e-12 >= min_responder_sensitivity:
            evaluated.append((float(ba), float(recall_nr), float(threshold)))
    if not evaluated:
        raise ValueError("No inner-CV threshold meets responder-sensitivity floor")
    # BA primary, NR recall secondary, threshold conservative tie-break.
    return float(max(evaluated)[2])


def make_classifier(seed: int):
    """Predeclared classifier architecture; class imbalance handled at training."""
    return RandomForestClassifier(
        n_estimators=250, min_samples_leaf=4, max_features=0.75,
        class_weight="balanced_subsample", random_state=int(seed), n_jobs=-1,
    )


def predict_nonresponse_proba(model, x):
    if set(model.classes_.tolist()) != {0, 1}:
        raise ValueError("A CV training fold must contain both classes")
    col = int(np.flatnonzero(model.classes_ == 1)[0])
    p = np.asarray(model.predict_proba(x)[:, col], dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid non-responder scores")
    return p


def inner_oof_probability(xseq, xstruct, y_nr, positions, *, use_structure, seed,
                          n_splits=INNER_SPLITS):
    """Cross-fitted inner scores, strictly within the outer training fold."""
    xseq, xstruct = np.asarray(xseq, float), np.asarray(xstruct, float)
    truth, groups = np.asarray(y_nr, dtype=int), np.asarray(positions, int)
    if len(truth) != len(xseq) or len(truth) != len(xstruct) or len(truth) != len(groups):
        raise ValueError("Feature/label length mismatch")
    if len(np.unique(groups[truth == 1])) < n_splits or len(np.unique(groups[truth == 0])) < n_splits:
        raise ValueError("Insufficient independent positions of both classes for inner CV")
    splits = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    result = np.full(len(truth), np.nan)
    for j, (fit, held) in enumerate(splits.split(xseq, truth, groups)):
        if set(groups[fit]) & set(groups[held]):
            raise ValueError("Inner position leakage")
        if len(np.unique(truth[fit])) != 2:
            raise ValueError("Missing non-response or response in inner fitting fold")
        a, b = xseq[fit], xseq[held]
        if use_structure:
            st, sv = sb.impute_fold(xstruct[fit], xstruct[held])
            a, b = np.column_stack([a, st]), np.column_stack([b, sv])
        model = make_classifier(seed + j + 1)
        model.fit(a, truth[fit])
        result[held] = predict_nonresponse_proba(model, b)
    if not np.isfinite(result).all():
        raise ValueError("Incomplete inner OOF predictions")
    return result


def average_repeats(raw: pd.DataFrame, *, expected_repeats=2) -> pd.DataFrame:
    """Freeze two-repeat OOF evaluation; binary decision uses mean(p - threshold)."""
    required = {"repeat", "model", "variant_key", "protein_position", "study_cohort",
                "observed_pp", "nr_rank_score", "nr_decision_margin"}
    if not required.issubset(raw):
        raise ValueError(f"Missing OOF columns: {sorted(required-set(raw))}")
    if raw.duplicated(["repeat", "model", "variant_key"]).any():
        raise ValueError("Duplicate OOF model/variant/repeat")
    counts = raw.groupby(["model", "variant_key"]).repeat.nunique()
    if counts.ne(expected_repeats).any() or raw.groupby(["model", "variant_key"]).size().ne(expected_repeats).any():
        raise ValueError("Missing repeated OOF predictions")
    grouped = raw.groupby(["model", "variant_key"], as_index=False)
    for field in ("protein_position", "study_cohort"):
        if grouped[field].nunique()[field].ne(1).any():
            raise ValueError(f"Inconsistent OOF {field}")
    if grouped["observed_pp"].agg(lambda x: x.max()-x.min()).observed_pp.gt(1e-6).any():
        raise ValueError("Inconsistent observed outcomes")
    return grouped.agg(
        protein_position=("protein_position", "first"),
        study_cohort=("study_cohort", "first"),
        observed_pp=("observed_pp", "mean"),
        nr_rank_score=("nr_rank_score", "mean"),
        nr_decision_margin=("nr_decision_margin", "mean"),
    )


def summarise_threshold(averaged: pd.DataFrame, *, threshold_pp=THRESHOLD_PP):
    """Equal-position metrics and fixed 20%-assay-capacity retrieval benchmark."""
    metrics = []
    for cohort in COHORTS:
        frame = averaged if cohort == "all" else averaged.loc[averaged.study_cohort.eq(cohort)]
        for name in ALL_MODELS:
            part = frame.loc[frame.model.eq(name)].copy()
            if part.empty:
                raise ValueError(f"Missing predictions for {cohort}: {name}")
            truth_nr = part.observed_pp.to_numpy(float) < threshold_pp
            score = part.nr_rank_score.to_numpy(float)
            margins = part.nr_decision_margin.to_numpy(float)
            p = part.protein_position.to_numpy(int)
            decision = margins >= 0
            w = position_weights(p)
            if not truth_nr.any() or truth_nr.all():
                raise ValueError(f"Both classes required for {cohort}")
            nr_recall = float(w[truth_nr & decision].sum() / w[truth_nr].sum())
            r_recall = float(w[(~truth_nr) & (~decision)].sum() / w[~truth_nr].sum())
            tp = int(np.sum(truth_nr & decision))
            fp = int(np.sum((~truth_nr) & decision))
            nslots = max(1, int(np.ceil(.20*len(part))))
            # Rank-based triage; does not require threshold calibration.
            # Ties in the trivial baseline are not scientifically interpretable.
            if name == "always_responder":
                p_at_budget = None
                recall_at_budget = None
            else:
                order = np.argsort(-score, kind="stable")[:nslots]
                hits = int(truth_nr[order].sum())
                p_at_budget = float(hits / nslots)
                recall_at_budget = float(hits / truth_nr.sum())
            metrics.append({
                "cohort": cohort, "model": name, "variants": len(part),
                "positions": int(len(np.unique(p))), "nonresponders": int(truth_nr.sum()),
                "predicted_nonresponders": int(decision.sum()),
                "nonresponders_detected": tp, "false_nonresponse_alarms": fp,
                "nr_recall": nr_recall, "responder_recall": r_recall,
                "balanced_accuracy": (nr_recall+r_recall)/2,
                "nr_auroc": float(roc_auc_score(truth_nr, score, sample_weight=w)),
                "nr_average_precision": float(average_precision_score(truth_nr, score, sample_weight=w)),
                "assay_budget_20pct_variants": nslots,
                "precision_at_20pct": p_at_budget,
                "nr_recall_at_20pct": recall_at_budget,
            })
    return pd.DataFrame(metrics)


def validate_phase15c(phase15c: pd.DataFrame, train: pd.DataFrame):
    need = {"repeat", "fold", "model", "variant_key", "protein_position", "observed_pp", "predicted_pp"}
    if not need.issubset(phase15c):
        raise ValueError("Phase 15C OOF schema mismatch")
    if set(phase15c.model) != set(sb.MODELS):
        raise ValueError("Phase 15C models incomplete")
    if phase15c.duplicated(["repeat", "variant_key", "model"]).any():
        raise ValueError("Duplicate Phase 15C OOF")
    first = phase15c.loc[phase15c.model.eq("vti_forest")].copy()
    sb.validate_stored_forest(first, train)
    base = phase15c.loc[phase15c.model.eq("structural_forest")]
    if len(base) != len(first):
        raise ValueError("Structural regression OOF incomplete")
    k = ["repeat", "variant_key"]
    matched = base.merge(first[k+["fold", "observed_pp", "protein_position"]], on=k,
                         validate="one_to_one", suffixes=("_structure", "_baseline"))
    if (len(matched) != len(first) or not np.array_equal(matched.fold_structure, matched.fold_baseline)
            or not np.array_equal(matched.protein_position_structure, matched.protein_position_baseline)
            or not np.allclose(matched.observed_pp_structure, matched.observed_pp_baseline, rtol=0, atol=1e-6)):
        raise ValueError("Phase 15C classifier comparison fold/truth mismatch")
    return first


def run() -> None:
    from cftr_theratype.benchmark import make_features
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.splits import parse_substitution
    from cftr_theratype.transfer import (ROOT, VTI_PAIRS, SPLIT_OUT, MANIFEST, RESULTS,
                                         SEED as TRANSFER_SEED, _load_allowed_labels)

    paths = [
        ROOT / "results/structure/phase15b_position_features.csv",
        ROOT / "results/structure/phase15b_feature_summary.json",
        ROOT / "results/structure/phase15c_train_oof.csv",
        ROOT / "results/structure/phase15c_summary.json",
        RESULTS / "train_cv_predictions.csv", RESULTS / "summary.json",
        VTI_PAIRS, SPLIT_OUT, MANIFEST,
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    transfer_manifest = json.loads(MANIFEST.read_text())
    transfer_summary = json.loads((RESULTS / "summary.json").read_text())
    frozen = json.loads((ROOT / "results/structure/phase15c_summary.json").read_text())
    if (transfer_manifest.get("vti_source_sha256") != sb.sha256(VTI_PAIRS)
            or transfer_summary.get("vti_source_sha256") != sb.sha256(VTI_PAIRS)
            or transfer_summary.get("vti_partition_sha256") != sb.sha256(SPLIT_OUT)
            or frozen.get("vti_source_sha256") != sb.sha256(VTI_PAIRS)
            or frozen.get("vti_partition_sha256") != sb.sha256(SPLIT_OUT)
            or frozen.get("validation_or_test_outcomes_loaded") is not False
            or frozen.get("structure_csv_sha256") != sb.sha256(paths[0])):
        raise ValueError("Frozen train-only input provenance mismatch")

    feature_ref = sb.read_feature_table(paths[0], paths[1], ROOT)
    membership = pd.read_parquet(SPLIT_OUT, columns=[
        "variant_key", "protein_position", "transfer_split", "study_cohort",
        "variant", "variant_type", "hgvs_protein",
    ])
    train = membership.loc[membership.transfer_split.eq("train")].copy()
    if len(train) != 210 or train.protein_position.nunique() != 168 or train.variant_key.duplicated().any():
        raise ValueError("VTI training membership unexpected")
    keys = set(train.variant_key.astype(str))
    labels = _load_allowed_labels(VTI_PAIRS, keys, "vti_rescue_pp")
    train = train.merge(labels, on="variant_key", validate="one_to_one")
    if len(train) != 210 or train.vti_rescue_pp.isna().any():
        raise ValueError("Missing train-only VTI outcomes")
    sequence = bihler_2024_assay_sequence()
    parsed = [parse_substitution(row, sequence) for _, row in train.iterrows()]
    if any(p[-1] != "valid" or p[0] != pos for p, pos in zip(parsed, train.protein_position)):
        raise ValueError("Variant-to-experimental-reference mismatch")
    train["reference_aa"] = [p[1] for p in parsed]
    train["alternate_aa"] = [p[2] for p in parsed]
    train = train.sort_values("variant_key").reset_index(drop=True)
    xseq, feat_names = make_features(train, sequence)
    xseq = np.asarray(xseq, float)
    xstruct = sb.structural_matrix(train.protein_position.to_numpy(int), feature_ref)
    if xseq.shape != (210, 68) or not np.isfinite(xseq).all() or len(feat_names) != 68:
        raise ValueError("Unexpected sequence features")
    y_nr = (train.vti_rescue_pp.to_numpy(float) < THRESHOLD_PP).astype(int)
    positions = train.protein_position.to_numpy(int)
    if y_nr.sum() != 36:
        raise ValueError("Frozen VTI training non-responder count changed")
    phase15c = pd.read_csv(paths[2])
    reference = validate_phase15c(phase15c, train)
    if sb.sha256(RESULTS / "train_cv_predictions.csv") != frozen.get("baseline_oof_sha256"):
        raise ValueError("Stored regression baseline provenance mismatch")
    key_to_idx = {str(k): i for i, k in enumerate(train.variant_key)}

    print("="*95)
    print("CFTR-THERATYPE | PHASE 15D — DIRECT NON-RESPONSE CLASSIFICATION — TRAIN ONLY")
    print("="*95)
    print("[1] PRESPECIFIED DESIGN")
    print(f"  VTI train: {len(train)} variants / {len(np.unique(positions))} positions")
    print(f"  Nonresponse (rescue <10 pp WT): {int(y_nr.sum())}; responders: {int((1-y_nr).sum())}")
    print("  250-tree RF classifiers, class_weight=balanced_subsample, min_leaf=4")
    print("  Sequence-only vs reference-structure augmented; same outer 2x5 position-held-out folds")
    print("  Inner 3-fold position-grouped CV chooses threshold using outer TRAIN only")
    print("  Inner threshold objective: balanced accuracy; responder sensitivity >= 0.80")
    print("  0.5 default and inner-selected rule both reported; classifier scores NOT calibrated probabilities")
    print("  20% assay-capacity ranking also reported without threshold fitting")
    print("  2024/2026 validation/test outcomes and ETI response outcomes: NOT READ")

    rows = []
    print("\n[2] FIT CLASSIFIERS: REUSE FROZEN OUTER FOLDS")
    for (rep, fold), held in reference.groupby(["repeat", "fold"], sort=True):
        hi = np.array([key_to_idx[k] for k in held.variant_key], dtype=int)
        hpos = set(positions[hi])
        fi = np.flatnonzero(~np.isin(positions, list(hpos)))
        if len(fi) + len(hi) != len(train) or set(positions[fi]) & hpos:
            raise ValueError("Outer fold position leakage/incomplete membership")
        seed = TRANSFER_SEED + (int(rep)-1)*10 + int(fold)
        for model, use_structure in (("sequence_balanced", False), ("structure_balanced", True)):
            inners = inner_oof_probability(xseq[fi], xstruct[fi], y_nr[fi], positions[fi],
                                          use_structure=use_structure, seed=seed)
            cut = choose_threshold_inner(y_nr[fi], inners, positions[fi])
            a, b = xseq[fi], xseq[hi]
            if use_structure:
                st, sv = sb.impute_fold(xstruct[fi], xstruct[hi])
                a, b = np.column_stack([a, st]), np.column_stack([b, sv])
            classifier = make_classifier(seed)
            classifier.fit(a, y_nr[fi])
            p = predict_nonresponse_proba(classifier, b)
            for idx, prob in zip(hi, p):
                for mode, threshold in (("fixed", .5), ("inner", cut)):
                    rows.append({
                        "repeat": int(rep), "fold": int(fold), "model": model+"_"+mode,
                        "variant_key": str(train.iloc[idx].variant_key),
                        "protein_position": int(positions[idx]),
                        "study_cohort": str(train.iloc[idx].study_cohort),
                        "observed_pp": float(train.iloc[idx].vti_rescue_pp),
                        "nr_rank_score": float(prob),
                        "nr_decision_margin": float(prob - threshold),
                        "fold_nr_threshold": float(threshold),
                    })
            print(f"  repeat={rep}/2 fold={fold}/5 {model:19s} fit={len(fi):3d} held={len(hi):3d} "
                  f"inner-NR-threshold={cut:.2f}", flush=True)

    idx = train.set_index("variant_key")
    regress = phase15c.loc[phase15c.model.isin(["vti_forest", "structural_forest"])].copy()
    regress["model"] = regress.model.map({
        "vti_forest": "sequence_regression", "structural_forest": "structure_regression",
    })
    regress["study_cohort"] = regress.variant_key.map(idx.study_cohort)
    regress["nr_rank_score"] = -regress.predicted_pp.astype(float)
    regress["nr_decision_margin"] = THRESHOLD_PP - regress.predicted_pp.astype(float)
    regress["fold_nr_threshold"] = THRESHOLD_PP
    noop = regress.loc[regress.model.eq("sequence_regression")].copy()
    noop["model"] = "always_responder"
    noop["nr_rank_score"] = 0.0
    noop["nr_decision_margin"] = -1.0
    noop["fold_nr_threshold"] = np.nan
    columns = ["repeat", "fold", "model", "variant_key", "protein_position", "study_cohort",
               "observed_pp", "nr_rank_score", "nr_decision_margin", "fold_nr_threshold"]
    all_oof = pd.concat([pd.DataFrame(rows)[columns], regress[columns], noop[columns]],
                        ignore_index=True)
    if set(all_oof.model) != set(ALL_MODELS):
        raise ValueError("Incomplete model suite")
    averaged = average_repeats(all_oof)
    summary = summarise_threshold(averaged)
    print("\n[3] NON-RESPONSE DETECTION (10 PP, LOWER RESCUE = NR)")
    print(f"  {'Cohort':10s} {'Model':27s} {'BA':>6s} {'NR recall':>9s} {'R recall':>9s} "
          f"{'NR-AUC':>7s} {'NR-AP':>7s} {'NR caught':>10s} {'False NR':>9s}")
    for r in summary.itertuples(index=False):
        print(f"  {r.cohort:10s} {r.model:27s} {r.balanced_accuracy:6.3f} "
              f"{r.nr_recall:9.3f} {r.responder_recall:9.3f} "
              f"{r.nr_auroc:7.3f} {r.nr_average_precision:7.3f} "
              f"{r.nonresponders_detected:3d}/{r.nonresponders:<3d} {r.false_nonresponse_alarms:9d}")
    print("\n[4] EXPLORATORY 20% ASSAY TRIAGE: HIGHEST NR SCORES")
    print(f"  {'Cohort':10s} {'Model':27s} {'Tested':>6s} {'Precision':>10s} {'NR recall':>10s}")
    for r in summary.itertuples(index=False):
        if r.model == "always_responder":
            continue
        print(f"  {r.cohort:10s} {r.model:27s} {r.assay_budget_20pct_variants:6d} "
              f"{r.precision_at_20pct:10.3f} {r.nr_recall_at_20pct:10.3f}")
    print("\n  Ranking is retrospective TRAIN OOF, not prospective clinical prediction;")
    print("  20% is fixed in advance, not selected using these evaluation outcomes.")

    outdir = ROOT / "results/structure"
    outdir.mkdir(parents=True, exist_ok=True)
    p_oof = outdir / "phase15d_nonresponse_train_oof.csv"
    p_summary = outdir / "phase15d_nonresponse_summary.csv"
    p_json = outdir / "phase15d_nonresponse_summary.json"
    all_oof.to_csv(p_oof, index=False)
    summary.to_csv(p_summary, index=False)
    report = {
        "phase": "15D_direct_nonresponse_classification", "status": "exploratory_train_only_nested_thresholds",
        "threshold_pp_wt": THRESHOLD_PP, "class_positive": "nonresponse_below_10pp",
        "vti_source_sha256": sb.sha256(VTI_PAIRS), "partition_sha256": sb.sha256(SPLIT_OUT),
        "phase15c_oof_sha256": sb.sha256(paths[2]), "phase15c_summary_sha256": sb.sha256(paths[3]),
        "structure_features_sha256": sb.sha256(paths[0]),
        "outer_repeats": 2, "outer_folds": 5, "inner_position_folds": INNER_SPLITS,
        "class_weight": "balanced_subsample", "min_responder_sensitivity_inner": MIN_RESPONDER_SENSITIVITY,
        "inner_threshold_grid": [float(x) for x in PROB_THRESHOLDS],
        "classification_metrics": json.loads(summary.to_json(orient="records")),
        "val_test_outcomes_loaded": False,
        "cautions": [
            "Class-weighted Forest score is not a calibrated probability",
            "Class imbalance and 36 nonresponders yield substantial uncertainty",
            "Raw 10pp rescue threshold differs from empirical-Bayes response categories",
            "Only previously inspected train folds; results are exploratory, not independent validation",
            "No mutation-specific structures; structural feature table is reference-coordinate derived",
            "Triage ranking is in-vitro prioritisation only and not clinical eligibility advice",
        ],
    }
    p_json.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print("\n[5] SAVED")
    for p in (p_oof, p_summary, p_json):
        print(f"  {p.relative_to(ROOT)}")
    print("PASS: Only frozen VTI training outcomes read; no held-out/test outcomes accessed")
    print("="*95)


if __name__ == "__main__":
    run()
