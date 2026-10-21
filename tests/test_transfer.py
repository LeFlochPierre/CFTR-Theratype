"""Position leakage and deterministic protocol tests for VTI transfer."""

import pandas as pd
import pytest
from cftr_theratype.transfer import select_position_partitions


def fake_membership():
    records = []
    for pos in range(100, 160):
        records.append({
            "variant_key": f"c.{pos}A>C",
            "protein_position": pos,
            "candidate_for_transfer_development": True,
            "frozen_2024_position_split": "unseen_2024_position" if pos < 135 else "train",
            "position_parse_status": "missense_position_parsed",
        })
    # Second variant at the same protein position must have the same split.
    records.append({
        "variant_key": "c.100A>G", "protein_position": 100,
        "candidate_for_transfer_development": True,
        "frozen_2024_position_split": "unseen_2024_position",
        "position_parse_status": "missense_position_parsed",
    })
    for pos, status in [(171, "val"), (180, "test")]:
        records.append({
            "variant_key": f"c.{pos}A>C", "protein_position": pos,
            "candidate_for_transfer_development": False,
            "frozen_2024_position_split": status,
            "position_parse_status": "missense_position_parsed",
        })
    return pd.DataFrame(records)


def test_reproducible_and_disjoint():
    cohort = fake_membership()
    first = select_position_partitions(cohort)
    second = select_position_partitions(cohort)
    pd.testing.assert_frame_equal(first, second)
    eligible = first[first.transfer_split.ne("excluded")]
    assert eligible.groupby("protein_position").transfer_split.nunique().max() == 1
    assert first.loc[first.frozen_2024_position_split.isin(["val", "test"]),
                     "transfer_split"].eq("excluded").all()
    assert first.loc[first.transfer_split.eq("test"),
                     "frozen_2024_position_split"].eq("unseen_2024_position").all()
    assert set(eligible.transfer_split) == {"train", "val", "test"}


def test_incorrect_protection_fails():
    cohort = fake_membership()
    cohort.loc[cohort.frozen_2024_position_split.eq("test"),
               "candidate_for_transfer_development"] = True
    with pytest.raises(ValueError, match="leaked"):
        select_position_partitions(cohort)


def test_no_duplicate_cdna_allowed():
    cohort = fake_membership()
    cohort.loc[0, "variant_key"] = cohort.loc[1, "variant_key"]
    with pytest.raises(ValueError, match="unique"):
        select_position_partitions(cohort)
