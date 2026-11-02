import pytest
from cftr_theratype.boltz_pilot import mutate_reference, parse_hgvs, select_panel


def test_single_mutation_and_position():
    assert mutate_reference("ABCDE", 3, "C", "Y") == "ABYDE"


def test_no_reference_silent_errors():
    with pytest.raises(ValueError):
        mutate_reference("ABCDE", 3, "D", "Y")
    with pytest.raises(ValueError):
        mutate_reference("ABCDE", 6, "F", "Y")
    with pytest.raises(ValueError):
        mutate_reference("ABCDE", 3, "C", "C")


def test_hgvs():
    assert parse_hgvs("p.Leu61Pro") == ("L", 61, "P")
    assert parse_hgvs("p.(Gln2Pro)") == ("Q", 2, "P")
    with pytest.raises(ValueError):
        parse_hgvs("p.F508del")


def test_panel_unique_and_label_free():
    candidates = [{"variant_key": str(i), "protein_position": i,
                   "distance_to_vx7_A": float(i), "hgvs_protein": f"p.Leu{i}Pro"}
                  for i in range(1, 21)]
    panel = select_panel(candidates)
    assert len(panel) == 12
    assert [x[0] for x in panel].count("near") == 4
    assert [x[0] for x in panel].count("distant") == 4
    assert [x[0] for x in panel].count("intermediate") == 4
    assert len({r["protein_position"] for _,r in panel}) == 12
    assert [r["protein_position"] for c,r in panel if c=="near"] == [1,2,3,4]
    assert [r["protein_position"] for c,r in panel if c=="distant"] == [20,19,18,17]


def test_same_position_distinct_variants_select_once():
    items = [{"variant_key": str(i), "protein_position": i,
              "distance_to_vx7_A": float(i)} for i in range(1, 21)]
    items.append({"variant_key": "duplicate", "protein_position": 3, "distance_to_vx7_A": 3.0})
    assert len(select_panel(items)) == 12
