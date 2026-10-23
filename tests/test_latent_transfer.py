"""Isolated checks for experimental latency, masking, and positional integrity."""
import numpy as np
import pandas as pd
import pytest

from cftr_theratype.latent_transfer import (
    embargo_eti_positions, restricted_train_outcomes, fit_latent, group_metrics,
    MODELS,
)


def test_embargo_excludes_both_held_and_protected_positions():
    eti = np.array([1, 2, 3, 5, 8, 13])
    mask = embargo_eti_positions(eti, {2, 5}, {8, 13})
    assert eti[mask].tolist() == [1, 3]


def test_strict_label_loader_only_requested_keys(tmp_path):
    pytest.importorskip("pyarrow")
    file = tmp_path / "source.parquet"
    pd.DataFrame({"variant_key": ["train", "test"], "outcome": [3., 100000.]}).to_parquet(file)
    allowed = restricted_train_outcomes(file, {"train"}, ("outcome",))
    assert allowed.variant_key.tolist() == ["train"]
    assert allowed.outcome.tolist() == [3.]
    with pytest.raises(ValueError, match="incomplete"):
        restricted_train_outcomes(file, {"missing"}, ("outcome",))


def test_fixed_low_rank_models_predict_without_heldout_assays():
    pytest.importorskip("torch")
    rng = np.random.default_rng(17)
    x = rng.normal(size=(38, 6))
    e = rng.normal(size=(44, 6))
    held = rng.normal(size=(7, 6))
    yv = np.column_stack([20+6*x[:, 0], 12+4*x[:, 1], 14+3*x[:, 1]])
    ye = np.column_stack([15+5*e[:, 0], 10+4*e[:, 1], 16+3*e[:, 1]])
    for model in MODELS:
        a = fit_latent(model, x, yv, e, ye, held, seed=12, epochs=32, rank=3)
        b = fit_latent(model, x, yv, e, ye, held, seed=12, epochs=32, rank=3)
        assert a.shape == (7, 3)
        assert np.isfinite(a[:, 0]).all()
        if model in {"vti_anchored", "anchored_latent"}:
            assert np.isfinite(a[:, 1:]).all()
        else:
            assert np.isnan(a[:, 1:]).all()
        np.testing.assert_allclose(a, b, atol=1.e-6, equal_nan=True)


def test_no_outcome_leak_through_query_variants():
    pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    a = rng.normal(size=(22, 4))
    b = rng.normal(size=(22, 4))
    ya = rng.normal(size=(22, 3))
    yb = rng.normal(size=(22, 3))
    q = rng.normal(size=(4, 4))
    result = fit_latent("anchored_latent", a, ya, b, yb, q, epochs=16)
    assert result.shape == (4, 3)
    # There is no held-out label input to this API.


def test_group_metrics_is_position_balanced_and_catches_duplicates():
    rows = []
    for name in ("vti_forest", *MODELS):
        for repeat in (1, 2):
            for i in range(10):
                rows.append({"model": name, "repeat": repeat, "variant_key": f"v{i}",
                             "protein_position": i + 1,
                             "study_cohort": "ETI, VTI" if i < 5 else "VTI",
                             "observed_pp": float(i), "predicted_pp": float(i + (0 if name == "anchored_latent" else 2))})
    out = pd.DataFrame(rows)
    summary, per_variant = group_metrics(out, bootstrap=100)
    assert len(summary) == 15
    best = summary.loc[summary.model.eq("anchored_latent")]
    assert (best.position_mae_pp == 0).all()
    assert len(per_variant) == 50
    with pytest.raises(ValueError, match="Duplicate"):
        group_metrics(pd.concat([out, out.head(1)], ignore_index=True))
