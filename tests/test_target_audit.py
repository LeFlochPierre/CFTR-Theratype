import numpy as np
import pandas as pd
import pytest

from cftr_theratype.target_audit import (
    approximate_diff_sem, attach_condition_rows, selected_rows, summarise_numeric,
)


def test_filtered_reads_exclude_other_variants(tmp_path):
    pytest.importorskip("pyarrow")
    data = pd.DataFrame({"variant_key": ["trainA", "testB"], "value": [2.0, 999.0]})
    path = tmp_path / "paired.parquet"
    data.to_parquet(path, index=False)
    out = selected_rows(path, ["trainA"], ["variant_key", "value"])
    assert out["variant_key"].tolist() == ["trainA"]
    assert out["value"].tolist() == [2.0]


def test_independent_sem_approximation():
    out = approximate_diff_sem(pd.Series([3., np.nan]), pd.Series([4., 1.]))
    assert out.iloc[0] == 5.0
    assert np.isnan(out.iloc[1])


def test_negative_sem_rejected():
    with pytest.raises(ValueError, match="Negative SEM"):
        approximate_diff_sem(pd.Series([-1.]), pd.Series([2.]))


def test_condition_source_row_join_is_exact():
    paired = pd.DataFrame({"variant_key": ["v1"], "source_row_dmso": [21]})
    cond = pd.DataFrame({
        "variant_key": ["v1", "v1"], "source_excel_row": [20, 21],
        "band C, %WT": [999, 4], "band C, %Total": [1, 2],
        "C-B ratio, %WT": [2, 3], "Total protein, %WT": [3, 4],
        "band C (%WT) SEM": [4, 5],
    })
    out = attach_condition_rows(paired, cond, "source_row_dmso", "dmso_")
    assert out["dmso_band C, %WT"].iloc[0] == 4


def test_missing_condition_row_rejected():
    paired = pd.DataFrame({"variant_key": ["v1"], "source_row_dmso": [31]})
    cond = pd.DataFrame({
        "variant_key": ["v1"], "source_excel_row": [30],
        "band C, %WT": [5], "band C, %Total": [5],
        "C-B ratio, %WT": [5], "Total protein, %WT": [5],
        "band C (%WT) SEM": [5],
    })
    with pytest.raises(ValueError, match="missing"):
        attach_condition_rows(paired, cond, "source_row_dmso", "dmso_")


def test_numeric_summary():
    x = summarise_numeric(pd.Series([0., 1., -1., np.nan]))
    assert x["n"] == 3
    assert x["median"] == 0
    assert x["negative_fraction"] == pytest.approx(1/3)
