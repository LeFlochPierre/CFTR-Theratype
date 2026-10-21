"""Unit tests for training-only repeated-CV reliability analysis."""

import numpy as np
import pandas as pd
import pytest

from cftr_theratype.reliability import normalise_oof, compute_reliability, cluster_mean_ci


def toy_oof():
    cohort = pd.DataFrame([
        {"variant_key": f"v{i}", "protein_position": i,
         "study_cohort": "VTI" if i < 6 else "ETI, VTI", "transfer_split": "train"}
        for i in range(12)
    ])
    rows = []
    for repeat in (1, 2):
        for i in range(12):
            for name, shift in (("median", 3.0), ("vti_forest", 2.0), ("pooled_forest", 1.5)):
                rows.append({"repeat": repeat, "fold": i % 5 + 1,
                             "model": name, "variant_key": f"v{i}",
                             "protein_position": i, "observed_pp": float(i),
                             "predicted_pp": float(i) + shift + i * 0.05})
    return pd.DataFrame(rows), cohort


def test_train_oof_complete_and_risk_curve():
    raw, cohort = toy_oof()
    joined = normalise_oof(raw, "total_rescue", cohort)
    rows, risk, extra = compute_reliability(joined)
    assert len(rows) == 6
    assert len(risk) == 12
    assert extra["risk_coverage"][-1]["positions"] == 12
    assert all(np.isfinite(r["position_mae_pp"]) for r in rows)


def test_rejects_any_nontrain_identifier():
    raw, cohort = toy_oof()
    cohort.loc[0, "transfer_split"] = "val"
    with pytest.raises(ValueError, match="training keys"):
        normalise_oof(raw, "total_rescue", cohort)


def test_rejects_repeat_duplication():
    raw, cohort = toy_oof()
    with pytest.raises(ValueError, match="Duplicate"):
        normalise_oof(pd.concat([raw, raw.iloc[[0]]], ignore_index=True),
                      "total_rescue", cohort)


def test_position_mean_ci_constant():
    score, lo, hi = cluster_mean_ci(np.repeat(4.0, 12))
    assert score == lo == hi == 4.0


def test_corrector_oof_with_cohort_metadata():
    raw, cohort = toy_oof()

    raw = raw.merge(
        cohort[["variant_key", "study_cohort"]],
        on="variant_key",
        validate="many_to_one",
    ).rename(columns={
        "observed_pp": "observed_corrector_pp",
        "predicted_pp": "predicted_corrector_pp",
    })

    result = normalise_oof(raw, "corrector_increment", cohort)

    assert len(result) == len(raw)
    assert "study_cohort" in result.columns

    # A conflicting source annotation must never be silently accepted.
    raw.loc[0, "study_cohort"] = "INCORRECT"

    with pytest.raises(
        ValueError,
        match="OOF study_cohort disagrees",
    ):
        normalise_oof(raw, "corrector_increment", cohort)
