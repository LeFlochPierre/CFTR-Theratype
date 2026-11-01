"""Phase 16C: ThermoMPNN feature provenance, fold safety and scoring."""
import json

import numpy as np
import pandas as pd
import pytest

from cftr_theratype import stability_benchmark as sb


def toy_data():
    train = pd.DataFrame({
        "variant_key": ["a", "b", "c", "d"],
        "protein_position": [10, 20, 20, 30],
        "reference_aa": ["A", "G", "G", "V"],
        "alternate_aa": ["V", "D", "E", "A"],
    })
    features = train.copy()
    features["thermompnn_ddg_pred"] = [1.1, -0.4, np.nan, 0.6]
    features["thermompnn_status"] = ["predicted", "predicted", "not_structurally_eligible", "predicted"]
    return train, features


def test_mutation_identity_join_is_exact_and_nan_preserved():
    train, scores = toy_data()
    m = sb.stability_matrix(train, scores)
    assert m.shape == (4, 2)
    np.testing.assert_allclose(m[[0, 1, 3], 0], [1.1, -0.4, 0.6])
    assert np.isnan(m[2, 0])
    np.testing.assert_array_equal(m[:, 1], [1, 1, 0, 1])


def test_fails_closed_on_wrong_substitution():
    train, scores = toy_data()
    scores.loc[1, "alternate_aa"] = "F"
    with pytest.raises(ValueError, match="incorrect mutation"):
        sb.stability_matrix(train, scores)


def test_fails_closed_on_missing_variant():
    train, scores = toy_data()
    with pytest.raises(ValueError, match="Incomplete stability join"):
        sb.stability_matrix(train, scores.iloc[:-1])


def test_shuffles_only_finite_stability_values():
    train, scores = toy_data()
    m = sb.stability_matrix(train, scores)
    out = sb.shuffled_stability(m, seed=2)
    np.testing.assert_array_equal(out[:, 1], m[:, 1])
    np.testing.assert_array_equal(np.isnan(out[:, 0]), np.isnan(m[:, 0]))
    np.testing.assert_array_equal(np.sort(out[np.isfinite(out[:, 0]), 0]),
                                  np.sort(m[np.isfinite(m[:, 0]), 0]))


def test_fails_closed_on_bad_shuffled_availability():
    m = np.array([[1., 1.], [np.nan, 1.], [2., 1.]])
    with pytest.raises(ValueError, match="availability mismatch"):
        sb.shuffled_stability(m, seed=10)


def toy_oof():
    truth = [2., 20., 5., 30., 6., 25., 3., 21.]
    models = [sb.MODEL_SEQUENCE, sb.MODEL_STRUCTURE, sb.MODEL_STABILITY] + [
        f"{sb.SHUFFLE_PREFIX}{k}" for k in range(sb.N_SHUFFLES)]
    rows = []
    for rep in (1, 2):
        for index, y in enumerate(truth):
            for model in models:
                offset = {sb.MODEL_SEQUENCE: 10., sb.MODEL_STRUCTURE: 5., sb.MODEL_STABILITY: 3.}.get(model, 8.)
                rows.append({"repeat": rep, "model": model,
                             "variant_key": f"v{index}", "protein_position": index+1,
                             "study_cohort": "ETI, VTI" if index < 4 else "VTI",
                             "observed_pp": y, "predicted_pp": y+offset})
    return pd.DataFrame(rows)


def test_equal_position_bootstrap_and_threshold_metrics():
    summary, cls, avg = sb.evaluate(toy_oof(), n_boot=100)
    sub = summary.loc[summary.cohort.eq("all") & summary.model.eq(sb.MODEL_STABILITY)].iloc[0]
    assert sub.n_positions == 8
    assert sub.position_mae_pp == pytest.approx(3.)
    assert sub.gain_vs_structural_pp == pytest.approx(2.)
    assert sub.gain_vs_mean_shuffled_pp == pytest.approx(5.)
    assert sub.gain_vs_structural_ci95 == pytest.approx([2., 2.])
    assert len(avg) == 8 * (3 + sb.N_SHUFFLES)
    c = cls.loc[cls.cohort.eq("all") & cls.model.eq(sb.MODEL_STABILITY)].iloc[0]
    assert c.triage_selected == 2
    assert 0 <= c.nonresponder_auroc <= 1
    assert c.nonresponders_total == 4


def test_partial_oof_refused():
    raw = toy_oof()
    raw = raw.iloc[1:]
    with pytest.raises(ValueError, match="Missing OOF predictions|Incomplete OOF"):
        sb.evaluate(raw, n_boot=50)


def test_stability_provenance_and_corruption_fail_closed(tmp_path):
    loc = tmp_path / "results/structure/phase16_thermompnn"
    loc.mkdir(parents=True)
    inputs = tmp_path / "results/structure/phase16_foldx"
    inputs.mkdir(parents=True)
    base = inputs / "6O2P_cftr_canonical_numbering.pdb"
    base.write_text("MODEL REFERENCE\n")
    raw = loc / "ThermoMPNN_inference_6O2P_cftr_canonical_numbering.csv"
    raw.write_text("raw scores\n")
    records = []
    for i in range(210):
        records.append({"variant_key": f"v{i}", "protein_position": i + 1,
                        "reference_aa": "A", "alternate_aa": "V",
                        "status": "eligible" if i < 195 else "unresolved_in_6O2P"})
    cand = pd.DataFrame(records)
    cp = inputs / "candidates_train_only.csv"
    cand.to_csv(cp, index=False)
    scored = cand.copy()
    scored["thermompnn_ddg_pred"] = [float(i)/100 for i in range(195)] + [np.nan]*15
    scored["thermompnn_status"] = ["predicted"]*195 + ["not_structurally_eligible"]*15
    fp = loc / "vti_train_stability_features.csv"
    scored.to_csv(fp, index=False)
    manifest = {
        "protocol": "phase16b_thermompnn_single_mutation_reference_6O2P_v1",
        "n_outcome_rows_accessed": 0, "n_train_variants": 210,
        "n_scored": 195, "n_missing_structure": 15,
        "pdb_sha256": sb.base.sha256(base),
        "prediction_file_sha256": sb.base.sha256(raw),
        "candidate_manifest_sha256": sb.base.sha256(cp),
    }
    (loc / "thermompnn_provenance.json").write_text(json.dumps(manifest))
    assert sb.verify_stability_artifacts(tmp_path, set(scored.variant_key)).shape[0] == 210
    raw.write_text("changed\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        sb.verify_stability_artifacts(tmp_path, set(scored.variant_key))
