"""Leakage-resistant exploratory ETI-to-VTI transfer benchmark.

Prepare: freeze a VTI test set at positions unseen in the 2024 missense split.
Fit: grouped CV on VTI train only; evaluate selected model on VTI val.
Original 2024 val/test and newly frozen VTI test outcomes are NEVER evaluated.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data/processed"
MEMBERSHIP = PROCESSED / "bihler_2026_cross_study_membership.parquet"
VTI_PAIRS = PROCESSED / "bihler_2026_vti.parquet"
ETI_PAIRS = PROCESSED / "bihler_2024_theratyping.parquet"
ETI_SPLITS = PROCESSED / "bihler_2024_missense_splits.parquet"
SPLIT_OUT = PROCESSED / "bihler_2026_transfer_splits.parquet"
MANIFEST = ROOT / "data/manifests/vti_transfer_protocol.json"
RESULTS = ROOT / "results/transfer"
SEED = 20261021


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_position_partitions(membership: pd.DataFrame, seed=SEED) -> pd.DataFrame:
    """Partition eligible VTI variants by position; no outcomes required.

    VTI test comprises ~20% of eligible positions not seen in the 2024
    missense split. VTI validation is ~20% of remaining eligible positions.
    """
    required = {
        "variant_key", "protein_position", "candidate_for_transfer_development",
        "frozen_2024_position_split", "position_parse_status",
    }
    if not required.issubset(membership.columns):
        raise ValueError(f"Missing membership columns: {sorted(required - set(membership.columns))}")
    if membership.variant_key.isna().any() or membership.variant_key.duplicated().any():
        raise ValueError("Membership requires unique non-null variant keys")

    result = membership.copy()
    result["transfer_split"] = "excluded"
    if result.groupby("protein_position", dropna=True)["frozen_2024_position_split"].nunique().max() > 1:
        raise ValueError("Inconsistent frozen 2024 partition for the same protein position")
    eligible = result.candidate_for_transfer_development.fillna(False).astype(bool)
    eligible_rows = result.loc[eligible].copy()
    if eligible_rows.protein_position.isna().any():
        raise ValueError("Eligible variant lacks a protein position")
    if not eligible_rows.position_parse_status.eq("missense_position_parsed").all():
        raise ValueError("Eligible variant is not a simple parsed missense")
    if eligible_rows.frozen_2024_position_split.isin(["val", "test"]).any():
        raise ValueError("2024 validation/test position leaked into candidates")

    new_positions = np.array(sorted(eligible_rows.loc[
        eligible_rows.frozen_2024_position_split == "unseen_2024_position",
        "protein_position",
    ].astype(int).unique()), dtype=int)
    if len(new_positions) < 10:
        raise ValueError("Too few entirely new positions to reserve a VTI test")

    test_count = max(1, int(np.ceil(0.20 * len(new_positions))))
    test_pos = set(np.random.default_rng(seed).permutation(new_positions)[:test_count])
    all_positions = set(eligible_rows.protein_position.astype(int))
    remaining = np.array(sorted(all_positions - test_pos), dtype=int)
    if len(remaining) < 15:
        raise ValueError("Too few positions left for model development")
    val_count = max(1, int(np.ceil(0.20 * len(remaining))))
    val_pos = set(np.random.default_rng(seed + 1).permutation(remaining)[:val_count])
    train_pos = all_positions - test_pos - val_pos

    assert train_pos.isdisjoint(val_pos)
    assert train_pos.isdisjoint(test_pos)
    assert val_pos.isdisjoint(test_pos)

    positions = result.protein_position
    result.loc[eligible & positions.isin(train_pos), "transfer_split"] = "train"
    result.loc[eligible & positions.isin(val_pos), "transfer_split"] = "val"
    result.loc[eligible & positions.isin(test_pos), "transfer_split"] = "test"

    if (result.loc[eligible, "transfer_split"] == "excluded").any():
        raise ValueError("Eligible records left unassigned")
    if result.loc[result.transfer_split.eq("test"), "frozen_2024_position_split"].ne(
        "unseen_2024_position"
    ).any():
        raise ValueError("VTI test must comprise only new ETI positions")
    return result


def prepare_transfer() -> None:
    """Use only identity columns; inspect 2026 experimental reference compatibility."""
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.splits import parse_substitution

    print("=" * 83)
    print("CFTR-THERATYPE | FREEZE 2026 VTI TRANSFER COHORT")
    print("=" * 83)
    for path in (MEMBERSHIP, VTI_PAIRS, ETI_SPLITS):
        if not path.exists():
            raise FileNotFoundError(path)

    membership = pd.read_parquet(MEMBERSHIP)
    vti_metadata = pd.read_parquet(
        VTI_PAIRS, columns=["variant_key", "variant", "variant_type", "hgvs_protein"]
    )
    etisplit = pd.read_parquet(
        ETI_SPLITS, columns=["variant_key", "protein_position", "split"]
    )
    if etisplit.groupby("protein_position").split.nunique().max() != 1:
        raise ValueError("Original ETI split is not position-disjoint")

    membership = membership.merge(
        vti_metadata, on="variant_key", how="left", validate="one_to_one",
        suffixes=("_membership", ""), indicator=True,
    )
    if not membership._merge.eq("both").all():
        raise ValueError("2026 identity data and cross-study mapping disagree")
    membership = membership.drop(columns="_merge")

    # Do not silently treat an HGVS position as a verified amino-acid variant.
    sequence = bihler_2024_assay_sequence()
    for row in membership.loc[membership.candidate_for_transfer_development].itertuples(index=False):
        # itertuples identifiers are not used for special column names here.
        r = pd.Series({"variant_type": row.variant_type,
                       "hgvs_protein": row.hgvs_protein, "variant": row.variant})
        position, wt, alt, status = parse_substitution(r, sequence)
        if status != "valid" or position != int(row.protein_position):
            raise ValueError(
                f"2026 reference/identity mismatch for {row.variant_key}: "
                f"{row.hgvs_protein} status={status}. Check 2026 assay WT background."
            )

    new = select_position_partitions(membership)
    expected_by_position = etisplit.drop_duplicates("protein_position").set_index(
        "protein_position"
    )["split"].to_dict()
    for row in new.loc[new.transfer_split.ne("excluded")].itertuples(index=False):
        original = expected_by_position.get(int(row.protein_position))
        if original in ("val", "test"):
            raise ValueError("Protected 2024 position assigned to VTI development/test")

    provenance = {
        "seed": SEED,
        "protocol": "position-disjoint VTI validation and VTI unseen-2024-position test",
        "vti_source_sha256": sha256(VTI_PAIRS),
        "membership_sha256": sha256(MEMBERSHIP),
        "eti_split_sha256": sha256(ETI_SPLITS),
        "vti_heldout_new_position_fraction": 0.20,
        "vti_validation_remaining_position_fraction": 0.20,
        "reference": "Bihler 2024 M470 assay protein background (2026 compatibility checked against variant HGVS)",
        "counts": {name: {
            "variants": int(new.transfer_split.eq(name).sum()),
            "positions": int(new.loc[new.transfer_split.eq(name), "protein_position"].nunique()),
        } for name in ["train", "val", "test", "excluded"]},
        "vti_test_positions": sorted(int(x) for x in new.loc[
            new.transfer_split.eq("test"), "protein_position"].unique()),
        "vti_validation_positions": sorted(int(x) for x in new.loc[
            new.transfer_split.eq("val"), "protein_position"].unique()),
        "eti_val_test_outcomes_loaded": False,
        "vti_outcomes_loaded": False,
    }

    if SPLIT_OUT.exists() or MANIFEST.exists():
        if not SPLIT_OUT.exists() or not MANIFEST.exists():
            raise ValueError("Existing transfer protocol is incomplete")
        existing = pd.read_parquet(SPLIT_OUT)
        old_manifest = json.loads(MANIFEST.read_text())
        keycols = ["variant_key", "protein_position", "transfer_split"]
        def normalise(frame):
            out = frame[keycols].copy().sort_values("variant_key")
            out["protein_position"] = out["protein_position"].astype("Int64")
            return out.reset_index(drop=True).astype(str)
        if old_manifest != provenance or not normalise(existing).equals(normalise(new)):
            raise ValueError("Transfer cohort is frozen and differs from regenerated data")
        print("PASS: Existing frozen transfer cohort matches source fingerprints")
    else:
        SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        new.to_parquet(SPLIT_OUT, index=False)
        MANIFEST.write_text(json.dumps(provenance, indent=2) + "\n")

    print("\n[1] FROZEN VTI PARTITIONS (NO RESPONSE LABELS READ)")
    for name in ("train", "val", "test", "excluded"):
        counts = provenance["counts"][name]
        print(f"{name.upper():8s} | {counts['variants']:3d} variants | {counts['positions']:3d} positions")
    print("\n[2] STUDY STRATA PER PARTITION")
    print(pd.crosstab(new["study_cohort"], new["transfer_split"]).to_string())
    print("\nVTI test restricted to mutation positions unseen in 2024 ETI split")
    print("2024 validation/test positions excluded; neither study test outcome loaded")
    print(f"Partition: {SPLIT_OUT.relative_to(ROOT)}")
    print(f"Protocol:  {MANIFEST.relative_to(ROOT)}")
    print("=" * 83)


def _estimator(name, seed):
    if name == "median":
        return DummyRegressor(strategy="median")
    if name == "vti_ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=100.0))
    if name in {"vti_forest", "eti_zero_shot_forest", "pooled_forest"}:
        return RandomForestRegressor(
            n_estimators=250, min_samples_leaf=4,
            max_features=0.75, n_jobs=-1, random_state=seed,
        )
    raise ValueError(name)


MODELS = ("median", "vti_ridge", "vti_forest", "eti_zero_shot_forest", "pooled_forest")


def _fit_and_predict(name, x_v, y_v, x_e, y_e, x_query, seed):
    """Train on labelled VTI and/or ETI *development* records only."""
    estimator = _estimator(name, seed)
    if name in {"median", "vti_ridge", "vti_forest"}:
        estimator.fit(x_v, y_v)
        return estimator.predict(x_query)
    if len(x_e) == 0:
        raise ValueError("No ETI train cases after held-out-position embargo")
    if name == "eti_zero_shot_forest":
        estimator.fit(x_e, y_e)
        return estimator.predict(x_query)
    if name == "pooled_forest":
        # Treatment identity is an INPUT, never the observed drug response.
        train_x = np.vstack([
            np.column_stack([x_e, np.zeros(len(x_e))]),
            np.column_stack([x_v, np.ones(len(x_v))]),
        ])
        estimator.fit(train_x, np.concatenate([y_e, y_v]))
        query = np.column_stack([x_query, np.ones(len(x_query))])
        return estimator.predict(query)
    raise ValueError(name)


def _load_allowed_labels(path, allowed, label):
    """Filter by allowed variant keys before exposing data to pandas."""
    result = pd.read_parquet(
        path, engine="pyarrow", columns=["variant_key", label],
        filters=[("variant_key", "in", sorted(set(allowed)))],
    )
    if not set(result.variant_key).issubset(allowed):
        raise ValueError("Unexpected excluded outcome returned by storage filter")
    if result.variant_key.duplicated().any():
        raise ValueError("Duplicate response labels")
    return result


def _feature_frame(metadata, sequence):
    from cftr_theratype.benchmark import make_features
    return make_features(metadata, sequence)[0]


def run_transfer_models() -> None:
    """5-fold grouped CV on VTI train; one exploratory VTI-val comparison."""
    from cftr_theratype.reference import bihler_2024_assay_sequence
    from cftr_theratype.cv import position_folds
    from cftr_theratype.benchmark import evaluate

    print("=" * 83)
    print("CFTR-THERATYPE | ETI → VTI TRANSFER BENCHMARK")
    print("=" * 83)
    if not SPLIT_OUT.exists() or not MANIFEST.exists():
        raise FileNotFoundError("First run: cftr-data transfer --stage prepare")
    protocol = json.loads(MANIFEST.read_text())
    hashes = {
        "vti_source_sha256": sha256(VTI_PAIRS),
        "membership_sha256": sha256(MEMBERSHIP),
        "eti_split_sha256": sha256(ETI_SPLITS),
    }
    if any(protocol.get(key) != val for key, val in hashes.items()):
        raise ValueError("Data changed after freezing VTI partitions")
    cohort = pd.read_parquet(SPLIT_OUT)
    if cohort.variant_key.duplicated().any():
        raise ValueError("Nonunique VTI variant_key")
    vti_train = cohort.loc[cohort.transfer_split == "train", [
        "variant_key", "protein_position", "variant_type", "hgvs_protein", "variant",
    ]].copy()
    vti_val = cohort.loc[cohort.transfer_split == "val", [
        "variant_key", "protein_position", "variant_type", "hgvs_protein", "variant",
    ]].copy()
    vti_test_positions = set(cohort.loc[
        cohort.transfer_split == "test", "protein_position"].dropna().astype(int))
    if set(vti_train.protein_position).intersection(vti_test_positions):
        raise ValueError("VTI test position leaked into train")
    if set(vti_val.protein_position).intersection(vti_test_positions):
        raise ValueError("VTI test position leaked into validation")
    if set(vti_train.protein_position).intersection(set(vti_val.protein_position)):
        raise ValueError("VTI val position leaked into train")

    # Select the model on training CV before accessing any VTI validation labels.
    vti_labels_train = _load_allowed_labels(
        VTI_PAIRS, set(vti_train.variant_key), "vti_rescue_pp"
    )
    vti_train = vti_train.merge(
        vti_labels_train, on="variant_key", how="left", validate="one_to_one"
    )
    if vti_train.vti_rescue_pp.isna().any():
        raise ValueError("VTI training responses missing")

    eti_split = pd.read_parquet(ETI_SPLITS)
    eti_train = eti_split.loc[eti_split.split == "train", [
        "variant_key", "protein_position", "reference_aa", "alternate_aa"
    ]].copy()
    allowed_eti_keys = set(eti_train.variant_key)
    eti_labels = _load_allowed_labels(ETI_PAIRS, allowed_eti_keys, "eti_rescue_pp")
    eti_train = eti_train.merge(eti_labels, on="variant_key", how="left", validate="one_to_one")
    if eti_train.eti_rescue_pp.isna().any():
        raise ValueError("ETI training responses missing")

    # All features from 2024 assay background, only if 2026 variants validate.
    from cftr_theratype.splits import parse_substitution
    seq = bihler_2024_assay_sequence()
    for frame in (vti_train, vti_val):
        parsed = [parse_substitution(row, seq) for _, row in frame.iterrows()]
        if any(p[-1] != "valid" for p in parsed):
            raise ValueError("VTI metadata invalid against experimental CFTR reference")
        frame["reference_aa"] = [p[1] for p in parsed]
        frame["alternate_aa"] = [p[2] for p in parsed]

    vti_train = vti_train.sort_values("variant_key").reset_index(drop=True)
    vti_val = vti_val.sort_values("variant_key").reset_index(drop=True)
    eti_train = eti_train.sort_values("variant_key").reset_index(drop=True)
    Xv = _feature_frame(vti_train, seq)
    Xa = _feature_frame(vti_val, seq)
    Xe = _feature_frame(eti_train, seq)
    yv = vti_train.vti_rescue_pp.to_numpy(float)
    ye = eti_train.eti_rescue_pp.to_numpy(float)
    pv = vti_train.protein_position.to_numpy(int)
    pa = vti_val.protein_position.to_numpy(int)
    pe = eti_train.protein_position.to_numpy(int)

    print("\n[1] SUPERVISED DATA ACCESS")
    print(f"VTI training labels:   {len(yv)}")
    print("VTI validation labels: SEALED until the CV winner is selected")
    print(f"ETI training labels:   {len(ye)}")
    print("VTI test and ETI val/test outcomes: NOT loaded")
    print("Every CV fold excludes its held-out positions from BOTH treatment trainings")

    rows = []
    print("\n[2] TWO REPEATS × FIVE POSITION-DISJOINT FOLDS")
    for repeat in range(2):
        for fold, tr_mask, va_mask in position_folds(pv, seed=SEED + 100 + repeat):
            fold_pos = set(pv[va_mask])
            blocked = fold_pos | set(pa) | vti_test_positions
            e_mask = ~np.isin(pe, list(blocked))
            if set(pe[e_mask]).intersection(blocked):
                raise ValueError("ETI data leaks into withheld VTI protein positions")
            print(f"Repeat {repeat+1}/2 fold {fold}/5 | "
                  f"VTI fit={tr_mask.sum():3d}, held={va_mask.sum():3d} | "
                  f"ETI fit={e_mask.sum():3d}", flush=True)
            for model in MODELS:
                preds = _fit_and_predict(
                    model, Xv[tr_mask], yv[tr_mask], Xe[e_mask], ye[e_mask],
                    Xv[va_mask], seed=SEED + repeat * 10 + fold,
                )
                for idx, pred in zip(np.flatnonzero(va_mask), preds):
                    rows.append({
                        "repeat": repeat + 1, "fold": fold,
                        "model": model, "variant_key": str(vti_train.iloc[idx].variant_key),
                        "protein_position": int(pv[idx]),
                        "absolute_error_pp": float(abs(yv[idx] - pred)),
                        "observed_pp": float(yv[idx]), "predicted_pp": float(pred),
                    })

    cv = pd.DataFrame(rows)
    errors = cv.groupby(["protein_position", "model"]).absolute_error_pp.mean().unstack()
    if errors.isna().any().any() or errors.shape[1] != len(MODELS):
        raise ValueError("Incomplete grouped-CV prediction coverage")
    scores = errors.mean().sort_values()
    winner = str(scores.index[0])
    print("\n[3] TRAIN-ONLY CV POSITION MAE (pp WT)")
    for key, value in scores.items():
        print(f"  {key:24s} {value:8.3f}")
    print(f"CV-selected model: {winner}")

    print("\n[3b] PAIRED POSITION-BOOTSTRAP COMPARISONS VS VTI-ONLY FOREST")
    base = errors["vti_forest"].to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(base), size=(3000, len(base)))
    for name in scores.index:
        if name == "vti_forest":
            continue
        gain = base - errors[name].to_numpy(float)
        boot = gain[draws].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        print(f"{name:24s} | gain={gain.mean():+.3f} pp | "
              f"95% CI [{low:+.3f}, {high:+.3f}]")

    # Access original VTI validation labels only after CV selection.
    vti_labels_val = _load_allowed_labels(
        VTI_PAIRS, set(vti_val.variant_key), "vti_rescue_pp"
    )
    vti_val = vti_val.merge(
        vti_labels_val, on="variant_key", how="left", validate="one_to_one"
    )
    if vti_val.vti_rescue_pp.isna().any():
        raise ValueError("VTI validation responses missing")
    ya = vti_val.vti_rescue_pp.to_numpy(float)

    # After model selection, the designated VTI validation subset is examined ONCE.
    # All ETI variants at validation positions are embargoed, even if from train.
    blocked_positions = set(pa) | vti_test_positions
    e_dev = ~np.isin(pe, list(blocked_positions))
    print("\n[4] PRE-SPECIFIED VALIDATION COMPARISON")
    targets = list(dict.fromkeys([winner, "vti_forest", "median"]))
    outputs = []
    for model in targets:
        prediction = _fit_and_predict(
            model, Xv, yv, Xe[e_dev], ye[e_dev], Xa, seed=SEED,
        )
        metric = evaluate(ya, prediction, pa)
        print(f"{model:24s} | Pos-MAE {metric['position_mae_pp']:8.3f} | "
              f"MAE {metric['mae_pp']:8.3f} | rho {metric['spearman']}")
        for key, pos, obs, pred in zip(vti_val.variant_key, pa, ya, prediction):
            outputs.append({"model": model, "variant_key": key,
                            "protein_position": int(pos),
                            "observed_pp": float(obs), "predicted_pp": float(pred)})

    RESULTS.mkdir(parents=True, exist_ok=True)
    cv.to_csv(RESULTS / "train_cv_predictions.csv", index=False)
    pd.DataFrame(outputs).to_csv(RESULTS / "validation_predictions.csv", index=False)
    (RESULTS / "summary.json").write_text(json.dumps({
        "vti_source_sha256": hashes["vti_source_sha256"],
        "vti_partition_sha256": sha256(SPLIT_OUT),
        "eti_split_sha256": hashes["eti_split_sha256"],
        "selection_metric": "train CV position-averaged MAE",
        "train_cv_scores": {str(k): float(v) for k, v in scores.items()},
        "selected_model": winner,
        "vti_test_evaluated": False, "original_eti_val_test_evaluated": False,
        "limitations": "Nonrandom ETI NR/BR cohort; ETI/VTI from different assay batches",
    }, indent=2) + "\n")
    print("\n[5] OUTPUTS")
    print(f"Results: {RESULTS.relative_to(ROOT)}")
    print("Both final test partitions remain sealed. TRANSFER BENCHMARK COMPLETE")
    print("=" * 83)




def diagnose_transfer():
    """Study-level comparability checks using TRAINING outcomes exclusively.

    The exact-pair response differences are descriptive: they are not causal
    head-to-head treatment effects across independent assay campaigns.
    """
    from scipy.stats import spearmanr

    print("=" * 83)
    print("CFTR-THERATYPE | TRAIN-ONLY ETI/VTI ASSAY COMPARABILITY")
    print("=" * 83)

    protocol = json.loads(MANIFEST.read_text())
    expected = {
        "vti_source_sha256": sha256(VTI_PAIRS),
        "membership_sha256": sha256(MEMBERSHIP),
        "eti_split_sha256": sha256(ETI_SPLITS),
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Frozen data fingerprint changed: {key}")

    cohort = pd.read_parquet(
        SPLIT_OUT,
        columns=["variant_key", "protein_position", "transfer_split"],
    )
    if cohort.variant_key.duplicated().any():
        raise ValueError("Nonunique VTI membership identifiers")
    vti_train = cohort.loc[cohort.transfer_split.eq("train"), [
        "variant_key", "protein_position"
    ]].copy()
    eti_groups = pd.read_parquet(
        ETI_SPLITS,
        columns=["variant_key", "protein_position", "split"],
    )
    eti_train = eti_groups.loc[eti_groups.split.eq("train"), [
        "variant_key", "protein_position"
    ]].copy()
    if not set(vti_train.protein_position).isdisjoint(
        set(cohort.loc[cohort.transfer_split.isin(["val", "test"]), "protein_position"].dropna())
    ):
        raise ValueError("VTI training shares a held-out protein position")
    if not set(eti_train.protein_position).isdisjoint(
        set(eti_groups.loc[eti_groups.split.isin(["val", "test"]), "protein_position"].dropna())
    ):
        raise ValueError("ETI training shares a held-out protein position")

    # Do not load ETI outcomes at VTI validation/test positions either.
    embargo = set(cohort.loc[
        cohort.transfer_split.isin(["val", "test"]), "protein_position"
    ].dropna().astype(int))
    eti_train = eti_train.loc[~eti_train.protein_position.isin(embargo)].copy()

    vti_cols = [
        "variant_key", "study_cohort", "baseline_fsk_pct_wt",
        "iva_treated_pct_wt", "vti_rescue_pp",
        "vti_increment_over_iva_pp",
    ]
    eti_cols = [
        "variant_key", "baseline_fsk_pct_wt", "iva_treated_pct_wt",
        "eti_rescue_pp", "eti_increment_over_iva_pp",
    ]
    # Predicate-pushdown ensures the materialised tables contain TRAIN keys only.
    vti = pd.read_parquet(
        VTI_PAIRS, engine="pyarrow", columns=vti_cols,
        filters=[("variant_key", "in", sorted(set(vti_train.variant_key)))],
    )
    eti = pd.read_parquet(
        ETI_PAIRS, engine="pyarrow", columns=eti_cols,
        filters=[("variant_key", "in", sorted(set(eti_train.variant_key)))],
    )
    if not set(vti.variant_key).issubset(set(vti_train.variant_key)):
        raise ValueError("Restricted VTI outcome loader returned held-out keys")
    if not set(eti.variant_key).issubset(set(eti_train.variant_key)):
        raise ValueError("Restricted ETI outcome loader returned held-out keys")
    if vti.variant_key.duplicated().any() or eti.variant_key.duplicated().any():
        raise ValueError("Duplicate assay labels")
    if set(vti.variant_key) != set(vti_train.variant_key):
        raise ValueError("Missing VTI training labels")
    if set(eti.variant_key) != set(eti_train.variant_key):
        raise ValueError("Missing ETI training labels")

    training = vti_train.merge(vti, on="variant_key", validate="one_to_one")
    matched = training.merge(
        eti_train.merge(eti, on="variant_key", validate="one_to_one"),
        on="variant_key", how="inner", validate="one_to_one",
        suffixes=("_vti", "_eti"),
    )
    if not matched.protein_position_vti.eq(matched.protein_position_eti).all():
        raise ValueError("Matched cDNA identifies different protein positions")
    if len(matched) < 10:
        raise ValueError("Insufficient exact training overlaps for diagnosis")
    matched = matched.rename(columns={"protein_position_vti": "protein_position"})
    matched = matched.drop(columns="protein_position_eti")
    matched["baseline_shift_pp"] = (
        matched.baseline_fsk_pct_wt_vti - matched.baseline_fsk_pct_wt_eti
    )
    matched["iva_activity_shift_pp"] = (
        matched.iva_treated_pct_wt_vti - matched.iva_treated_pct_wt_eti
    )
    matched["vti_minus_eti_rescue_pp"] = (
        matched.vti_rescue_pp - matched.eti_rescue_pp
    )
    matched["corrector_increment_shift_pp"] = (
        matched.vti_increment_over_iva_pp - matched.eti_increment_over_iva_pp
    )

    print("\n[1] TRAINING-ONLY COHORT DISTRIBUTION")
    print(f"VTI train:   {len(training)} variants at {training.protein_position.nunique()} positions")
    print(f"ETI train:   {len(eti)} variants at {eti_train.protein_position.nunique()} positions")
    print(f"Exact pairs: {len(matched)} variants at {matched.protein_position.nunique()} positions")
    for name, subset in training.groupby("study_cohort", dropna=False):
        y = subset.vti_rescue_pp
        print(f"  {str(name):12s} n={len(subset):3d} VTI median={y.median():8.3f} pp "
              f"IQR=[{y.quantile(.25):.3f}, {y.quantile(.75):.3f}]")

    print("\n[2] COMPARABILITY OF PAIRED ASSAY MEASUREMENTS")
    comparisons = [
        ("Baseline FSK", "baseline_fsk_pct_wt_eti", "baseline_fsk_pct_wt_vti", "baseline_shift_pp"),
        ("FSK + IVA (DMSO)", "iva_treated_pct_wt_eti", "iva_treated_pct_wt_vti", "iva_activity_shift_pp"),
        ("Modulator rescue", "eti_rescue_pp", "vti_rescue_pp", "vti_minus_eti_rescue_pp"),
        ("Corrector increment", "eti_increment_over_iva_pp", "vti_increment_over_iva_pp", "corrector_increment_shift_pp"),
    ]
    summary = {}
    for label, xcol, ycol, difference in comparisons:
        valid = matched[["protein_position", xcol, ycol, difference]].dropna()
        rho = float(spearmanr(valid[xcol], valid[ycol]).statistic)
        ci = _position_cluster_spearman_ci(valid, xcol, ycol, seed=SEED)
        diff = valid[difference]
        print(f"{label:22s} n={len(valid):3d} rho={rho:+.3f} "
              f"95% cluster CI=[{ci[0]:+.3f}, {ci[1]:+.3f}] "
              f"median paired shift={diff.median():+.3f} pp "
              f"median |shift|={diff.abs().median():.3f} pp")
        summary[label] = {
            "n": len(valid), "spearman": rho, "cluster_ci": ci,
            "median_paired_shift_pp": float(diff.median()),
            "median_absolute_shift_pp": float(diff.abs().median()),
        }

    print("\n[3] INTERPRETATION")
    print("Baseline and DMSO+IVA shifts provide an assay-comparability control.")
    print("Rescue shifts combine potential drug differences and study effects.")
    print("This is NOT a head-to-head estimate of clinical VTI versus ETI efficacy.")
    print("All outcome summaries use original training keys only.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    matched.to_csv(RESULTS / "training_exact_overlap.csv", index=False)
    (RESULTS / "training_overlap_diagnostics.json").write_text(json.dumps({
        "protocol_sha256": sha256(MANIFEST),
        "vti_train_n": len(training), "eti_train_n": len(eti),
        "exact_training_pairs": len(matched),
        "matched_positions": int(matched.protein_position.nunique()),
        "measurements": summary,
        "vti_or_eti_heldout_outcomes_used": False,
        "limitation": "Historical matched variants, different assays and nonrandom cohort; shifts not causal treatment effects",
    }, indent=2) + "\n")
    print("\nSaved: results/transfer/training_exact_overlap.csv")
    print("Saved: results/transfer/training_overlap_diagnostics.json")
    print("TRAINING-ONLY DIAGNOSTICS COMPLETE")
    print("=" * 83)


def _position_cluster_spearman_ci(frame, xcol, ycol, seed=0, n_boot=2000):
    """Cluster bootstrap to respect repeated substitutions at the same residue."""
    from scipy.stats import spearmanr

    groups = [g[[xcol, ycol]].to_numpy(dtype=float)
              for _, g in frame.groupby("protein_position", sort=True)]
    if len(groups) < 8:
        raise ValueError("Insufficient protein positions for bootstrap")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        sample_idx = rng.integers(0, len(groups), size=len(groups))
        sample = np.concatenate([groups[i] for i in sample_idx])
        with np.errstate(invalid="ignore"):
            rho = float(spearmanr(sample[:, 0], sample[:, 1]).statistic)
        if np.isfinite(rho):
            draws.append(rho)
    if len(draws) < n_boot * .9:
        raise ValueError("Too many undefined bootstrap rank correlations")
    return [float(x) for x in np.quantile(draws, [.025, .975])]

def run_transfer(stage="prepare"):
    if stage == "prepare":
        prepare_transfer()
    elif stage == "fit":
        run_transfer_models()
    elif stage == "diagnose":
        diagnose_transfer()
    elif stage == "mechanism":
        from cftr_theratype.mechanism import run_mechanism
        run_mechanism()
    elif stage == "reliability":
        from cftr_theratype.reliability import run_reliability
        run_reliability()
    else:
        raise ValueError(f"Unsupported transfer stage: {stage}")
