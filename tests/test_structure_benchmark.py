"""Phase 15C — structural features are per-residue, missingness safe, folds paired."""
import json
import hashlib

import numpy as np
import pandas as pd
import pytest

from cftr_theratype import structure_benchmark as sb


def fake_table():
    positions = [10, 20, 30, 40]
    frame = pd.DataFrame({"protein_position": positions})
    for i, col in enumerate(sb.DISTANCE_AND_GEOMETRY):
        frame[col] = [float(i + j + 1) for j in range(3)] + [np.nan]
    frame["has_exact_reference"] = [1, 1, 1, 0]
    frame["6O2P_resolved_exact"] = [1, 1, 1, 0]
    frame["8EIQ_resolved_exact"] = [1, 1, 1, 0]
    frame["assay_reference_M470_vs_PDB_V470"] = [0, 0, 0, 0]
    return frame


def test_unresolved_reference_kept_and_nan_until_fold_fit():
    vals = sb.structural_matrix(np.array([10, 40, 20]), fake_table())
    assert vals.shape == (3, len(sb.STRUCTURE_COLUMNS))
    assert np.isnan(vals[1, :8]).all()
    np.testing.assert_allclose(vals[1, 8:], [0, 0, 0, 0])
    train, val = sb.impute_fold(vals[[0, 2]], vals[[1]])
    np.testing.assert_allclose(val[0, :8], train[:, :8].mean(axis=0))
    np.testing.assert_allclose(val[0, 8:], [0, 0, 0, 0])


def test_imputation_medians_use_train_only_not_held_out():
    tr = np.array([[2., np.nan], [10., np.nan]])
    held = np.array([[np.nan, 999.]])
    xf, xv = sb.impute_fold(tr, held)
    np.testing.assert_allclose(xv, [[6., 999.]])
    np.testing.assert_allclose(xf[:, 1], [0., 0.])


def test_positionwise_shuffle_preserves_multiple_substitutions():
    positions = np.arange(10, 40)
    positions = np.insert(positions, 3, positions[2])
    x = np.column_stack([positions.astype(float), positions.astype(float) + 2])
    perm = sb.shuffled_by_position(x, positions, seed=123)
    assert perm[2].tolist() == perm[3].tolist()
    assert not np.allclose(perm, x)
    np.testing.assert_array_equal(np.sort(np.unique(perm[:, 0])), np.sort(np.unique(x[:, 0])))


def test_shuffled_features_reject_position_mismatch():
    with pytest.raises(ValueError, match="Structure differs"):
        sb.shuffled_by_position(np.array([[1.], [2.], [3.]]), np.array([10, 10, 20]))


def test_missing_position_rejected():
    with pytest.raises(ValueError, match="omitted"):
        sb.structural_matrix(np.array([10, 999]), fake_table())


def toy_oof():
    rows = []
    for rep in (1, 2):
        for i in range(8):
            cohort = "ETI, VTI" if i < 4 else "VTI"
            truth = [2., 4., 12., 16., 3., 7., 18., 24.][i]
            for model, error in zip(sb.MODELS, [5., 2., 4.]):
                rows.append({
                    "repeat": rep, "model": model,
                    "variant_key": f"v{i}", "protein_position": i + 1,
                    "study_cohort": cohort, "observed_pp": truth,
                    "predicted_pp": truth + error,
                })
    return pd.DataFrame(rows)


def test_paired_position_metrics_correct_and_threshold_metrics_finite():
    summary, averaged = sb.position_scores(toy_oof(), n_boot=100, seed=0)
    a = summary.loc[summary.model.eq("structural_forest") & summary.cohort.eq("all")].iloc[0]
    assert a.position_mae_pp == pytest.approx(2)
    assert a.gain_vs_forest_pp == pytest.approx(3)
    assert a.gain_vs_shuffled_pp == pytest.approx(2)
    assert len(averaged) == 24
    scores = sb.threshold_metrics(averaged)
    assert len(scores) == 9
    assert scores.auroc.notna().all()


def test_repeated_predictions_must_be_complete_and_unique():
    raw = toy_oof()
    with pytest.raises(ValueError, match="Duplicated"):
        sb.position_scores(pd.concat([raw, raw.head(1)], ignore_index=True), n_boot=100)
    missing = raw.loc[~(raw.variant_key.eq("v0") & raw.model.eq("structural_forest") & raw.repeat.eq(2))]
    with pytest.raises(ValueError, match="Incomplete"):
        sb.position_scores(missing, n_boot=100)


def test_stored_oof_rejects_position_leakage():
    oof = pd.DataFrame([
        {"repeat": rep, "fold": (i % 5) + 1, "model": "vti_forest",
         "variant_key": f"v{i}", "protein_position": (i+1),
         "observed_pp": float(i), "predicted_pp": float(i+1)}
        for rep in (1, 2) for i in range(10)
    ])
    train = pd.DataFrame({"variant_key": [f"v{i}" for i in range(10)],
                          "protein_position": list(range(1,11)),
                          "vti_rescue_pp": [float(i) for i in range(10)]})
    sb.validate_stored_forest(oof, train)
    invalid = oof.copy()
    invalid.loc[(invalid.repeat.eq(1)) & invalid.variant_key.eq("v0"), "protein_position"] = 2
    with pytest.raises(ValueError, match="position"):
        sb.validate_stored_forest(invalid, train)


def test_feature_provenance_fails_if_structure_changed(tmp_path):
    referenced = {
        "results/structure/phase15_coverage_summary.json": "summary",
        "results/structure/phase15_coverage_positions.csv": "map",
        "data/external/structures/6O2P.cif": "struct1",
        "data/external/structures/8EIQ.cif": "struct2",
    }
    hashes = {}
    for rel, value in referenced.items():
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(value)
        hashes[rel] = hashlib.sha256(value.encode()).hexdigest()
    inp = tmp_path / "results/structure/phase15b_position_features.csv"
    fake_table().to_csv(inp, index=False)
    man = tmp_path / "results/structure/phase15b_feature_summary.json"
    man.write_text(json.dumps({
        "phase": "15B_structure_features", "outcomes_loaded": False,
        "test_outcomes_loaded": False,
        "structural_predictor_columns": list(sb.STRUCTURE_COLUMNS),
        "phase15a_summary_sha256": hashes["results/structure/phase15_coverage_summary.json"],
        "phase15a_position_map_sha256": hashes["results/structure/phase15_coverage_positions.csv"],
        "structures_sha256": {p: hashes[f"data/external/structures/{p}.cif"] for p in ("6O2P", "8EIQ")},
    }))
    assert len(sb.read_feature_table(inp, man, tmp_path)) == 4
    (tmp_path / "data/external/structures/8EIQ.cif").write_text("altered")
    with pytest.raises(ValueError, match="provenance"):
        sb.read_feature_table(inp, man, tmp_path)
