import numpy as np
import pandas as pd
import pytest

from cftr_theratype.maturation import (
    assemble, position_table, rank_corr, position_bootstrap_association,
    summarise_stratum,
)


def test_position_table_equal_position_means():
    df = pd.DataFrame({
        "variant_key": ["a", "b", "c"],
        "protein_position": [10, 10, 11],
        "marker": [0., 10., 7.], "response": [8., 12., 2.],
    })
    result = position_table(df, ["marker", "response"])
    assert len(result) == 2
    assert result.loc[result.protein_position.eq(10), "marker"].item() == 5.


def test_position_table_rejects_duplicate_keys_and_unknown_positions():
    df = pd.DataFrame({"variant_key": ["a", "a"], "protein_position": [12, 13], "x": [1., 2.]})
    with pytest.raises(ValueError, match="Duplicate"):
        position_table(df, ["x"])
    df.loc[1, "variant_key"] = "b"
    df.loc[1, "protein_position"] = np.nan
    with pytest.raises(ValueError, match="Missing"):
        position_table(df, ["x"])


def test_rank_corr_sensible_with_covariates():
    rng = np.random.default_rng(3)
    n = 60
    baseline = rng.normal(size=n)
    noise = rng.normal(size=n)
    marker = 2 * baseline + noise
    response = 3 * baseline + 2 * noise + rng.normal(scale=.15, size=n)
    assert rank_corr(marker, response) > .8
    assert rank_corr(marker, response, baseline[:, None]) > .7


def test_position_bootstrap_deterministic_and_missingness():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "marker": np.arange(32, dtype=float),
        "response": np.arange(32, dtype=float) + rng.normal(scale=.1, size=32),
        "baseline_fsk": rng.normal(size=32),
        "iva_treated": rng.normal(size=32),
    })
    result = position_bootstrap_association(df, "marker", "response", True,
                                             seed=11, bootstraps=120)
    same = position_bootstrap_association(df, "marker", "response", True,
                                           seed=11, bootstraps=120)
    assert result == same
    assert result["rho"] > .95
    assert result["ci_low"] > .85
    assert result["bootstrap_valid"] == 120
    df.loc[:25, "marker"] = np.nan
    small = position_bootstrap_association(df, "marker", "response", True)
    assert small["rho"] is None


def test_stratum_report_contains_predeclared_pairs():
    rng = np.random.default_rng(8)
    n = 36
    baseline = rng.normal(size=n)
    dmso = rng.normal(size=n)
    gain = rng.normal(size=n)
    iva = baseline + rng.normal(size=n)
    df = pd.DataFrame({
        "variant_key": [f"c.{i}" for i in range(n)],
        "protein_position": np.arange(1, n+1),
        "band_c_dmso_pct_wt": dmso,
        "band_c_treated_pct_wt": dmso + gain,
        "band_c_gain_pct_wt": gain,
        "band_c_gain_pct_total": gain * .5,
        "baseline_fsk": baseline,
        "iva_treated": iva,
        "absolute_treated": iva + gain,
        "total_rescue": iva + gain - baseline,
        "corrector_increment": gain,
    })
    summary, assocs = summarise_stratum(df, "synthetic")
    assert summary["n_positions"] == n
    assert len(assocs) == 5
    assert all(a["stratum"] == "synthetic" for a in assocs)
    assert assocs[0]["rho"] > .95


def test_assemble_only_training_and_exact_counts(monkeypatch, tmp_path):
    from cftr_theratype import maturation
    monkeypatch.setitem(maturation.SETTINGS, "MOCK", {
        "membership": "split.parquet", "split_col": "split", "expected_n": 2,
        "treated": "treated_response", "rescue": "rescue", "increment": "increment",
    })
    monkeypatch.setattr(maturation, "audit_one", lambda root, label: (
        pd.DataFrame({
            "variant_key": ["a", "b"], "baseline_fsk_pct_wt": [2., 3.],
            "iva_treated_pct_wt": [3., 4.], "treated_response": [9., 10.],
            "rescue": [7., 7.], "increment": [6., 6.],
            "dmso_band C, %WT": [1., 2.], "treated_band C, %WT": [3., 5.],
            "dmso_band C, %Total": [10., 11.],
            "treated_band C, %Total": [12., 15.],
            "approx_rescue_sem_independent": [3., 5.],
        }), {}
    ))
    def fake_read(path, *, columns):
        assert columns == ["variant_key", "protein_position", "split"]
        return pd.DataFrame({"variant_key": ["a", "b", "sealed"],
                             "protein_position": [10, 11, 12],
                             "split": ["train", "train", "test"]})
    monkeypatch.setattr(maturation.pd, "read_parquet", fake_read)
    out = assemble(tmp_path, "MOCK")
    assert out["variant_key"].tolist() == ["a", "b"]
    assert "sealed" not in set(out["variant_key"])
    assert out["band_c_gain_pct_wt"].tolist() == [2., 3.]
