"""Cross-study mapping: exact overlap and position-level test protection."""
import pandas as pd
import pytest

from cftr_theratype.cross_study import map_cross_study_membership


def make_example():
    old = pd.DataFrame({
        "variant_key": ["c.1A>G", "c.2A>G", "c.3A>G"],
        "hgvs_protein": ["p.Ala10Val", "p.Gly20Ser", "p.Leu30Pro"],
    })
    splits = pd.DataFrame({
        "variant_key": ["c.1A>G", "c.2A>G", "c.3A>G"],
        "protein_position": [10, 20, 30],
        "split": ["train", "val", "test"],
    })
    vti = pd.DataFrame({
        "variant_key": ["c.1A>G", "new-at-test", "new-at-val", "new-unseen"],
        "hgvs_protein": ["p.Ala10Val", "p.Leu30Arg", "p.Gly20Ala", "p.Phe40Cys"],
        "variant_type": ["Missense"] * 4,
        "study_cohort": ["ETI, VTI", "VTI", "VTI", "VTI"],
        "vti_rescue_pp": [100000.0] * 4,  # must never enter output
    })
    return old, splits, vti


def test_position_protection_catches_nonidentical_variants():
    old, splits, vti = make_example()
    result = map_cross_study_membership(old, splits, vti)
    by_key = result.set_index("variant_key")
    assert bool(by_key.loc["c.1A>G", "candidate_for_transfer_development"])
    assert bool(by_key.loc["new-at-test", "protected_2024_validation_or_test_position"])
    assert bool(by_key.loc["new-at-val", "protected_2024_validation_or_test_position"])
    assert bool(by_key.loc["new-unseen", "candidate_for_transfer_development"])
    assert result["overlaps_2024_exact_cdna"].sum() == 1
    assert "vti_rescue_pp" not in result.columns


def test_reject_position_leakage_in_frozen_splits():
    old, splits, vti = make_example()
    splits.loc[0, "protein_position"] = 20
    with pytest.raises(ValueError, match="shared"):
        map_cross_study_membership(old, splits, vti)


def test_annotation_disagreement_is_fatal():
    old, splits, vti = make_example()
    vti.loc[0, "hgvs_protein"] = "p.Ala10Gly"
    with pytest.raises(ValueError, match="annotation"):
        map_cross_study_membership(old, splits, vti)
