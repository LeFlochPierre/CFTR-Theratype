"""Phase 15B data isolation, construction and missingness checks."""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from cftr_theratype import structure_features as sf


def fake_map():
    # 6O2P exact at 10/20, mismatched 30, unresolved 40;
    # 8EIQ exact 10/30, unresolved elsewhere.
    return pd.DataFrame({
        "protein_position": [10, 20, 30, 40],
        "6O2P_status": ["resolved_exact", "resolved_exact", "resolved_construct_mismatch", "deposited_but_unresolved"],
        "8EIQ_status": ["resolved_exact", "deposited_but_unresolved", "resolved_exact", "not_in_deposited_construct"],
        "6O2P_ca_contacts_8A": [4, 5, 3, np.nan],
        "8EIQ_ca_contacts_8A": [7, np.nan, 10, np.nan],
        "6O2P_VX7_ca_distance_A": [8., 17., 3., np.nan],
        "8EIQ_VX7_ca_distance_A": [9., np.nan, 25., np.nan],
        "8EIQ_CV6_ca_distance_A": [11., np.nan, 27., np.nan],
        "8EIQ_WJX_ca_distance_A": [12., np.nan, 28., np.nan],
    })


def fake_geom():
    return {
        "6O2P": {
            10: {"6O2P_protein_sasa_A2": 100., "6O2P_protein_relative_sasa": .7, "6O2P_long_range_contacts_8A": 2},
            20: {"6O2P_protein_sasa_A2": 15., "6O2P_protein_relative_sasa": .1, "6O2P_long_range_contacts_8A": 1},
        },
        "8EIQ": {
            10: {"8EIQ_protein_sasa_A2": 55., "8EIQ_protein_relative_sasa": .5, "8EIQ_long_range_contacts_8A": 1},
            30: {"8EIQ_protein_sasa_A2": 90., "8EIQ_protein_relative_sasa": .8, "8EIQ_long_range_contacts_8A": 3},
        },
    }


def test_priority_and_fallback_without_mutant_assumptions():
    rows = sf.build_features(fake_map(), fake_geom()).set_index("protein_position")
    assert rows.loc[10, "primary_reference"] == "6O2P"
    assert rows.loc[30, "primary_reference"] == "8EIQ"
    assert rows.loc[40, "primary_reference"] == "unresolved"
    assert rows.loc[20, "reference_contacts_8A"] == 5
    assert rows.loc[30, "reference_contacts_8A"] == 10
    assert rows.loc[30, "reference_long_range_contacts_8A"] == 3
    assert rows.loc[10, "reference_protein_relative_sasa"] == .7
    assert rows.loc[30, "reference_protein_relative_sasa"] == .8


def test_mismatched_and_unresolved_positions_never_get_phenotype_proxy():
    rows = sf.build_features(fake_map(), fake_geom()).set_index("protein_position")
    assert np.isnan(rows.loc[30, "6O2P_ca_contacts_8A"])
    assert np.isnan(rows.loc[30, "6O2P_VX7_ca_distance_A"])
    assert rows.loc[40, "has_exact_reference"] == 0
    assert rows.loc[40, list(sf.STRUCTURE_ONLY_COLS)].isna().all()


def test_reference_geometry_shared_between_variants_at_same_position():
    positions = sf.build_features(fake_map(), fake_geom())
    variants = pd.DataFrame({"variant_key": ["A", "B"], "protein_position": [20, 20]})
    merged = variants.merge(positions, on="protein_position", validate="many_to_one")
    assert len(merged) == 2
    assert merged.reference_contacts_8A.nunique() == 1
    assert merged.reference_protein_sasa_A2.nunique() == 1


def test_contacts_only_include_spatially_close_and_sequence_distant_residues():
    xyz = {
        10: np.array([0., 0., 0.]),
        11: np.array([1., 0., 0.]),
        200: np.array([0., 6., 0.]),
        400: np.array([40., 40., 0.]),
    }
    c = sf.long_range_contacts(xyz)
    assert c == {10: 1, 11: 1, 200: 2, 400: 0}


def test_mapping_rejects_duplicate_positions():
    df = fake_map()
    df.loc[1, "protein_position"] = 10
    with pytest.raises(ValueError, match="duplicated"):
        sf.build_features(df, fake_geom())


def test_missing_necessary_fields_rejected():
    with pytest.raises(ValueError, match="Missing"):
        sf.build_features(fake_map().drop(columns=["8EIQ_WJX_ca_distance_A"]), fake_geom())


def test_geometry_missing_from_exact_position_rejected():
    geom = fake_geom()
    del geom["6O2P"][20]
    with pytest.raises(ValueError, match="Incomplete 6O2P"):
        sf.build_features(fake_map(), geom)


def test_assay_background_flag_at_470():
    m = fake_map()
    m.loc[0, "protein_position"] = 470
    g = fake_geom()
    g["6O2P"][470] = g["6O2P"].pop(10)
    g["8EIQ"][470] = g["8EIQ"].pop(10)
    out = sf.build_features(m, g).set_index("protein_position")
    assert out.loc[470, "assay_reference_M470_vs_PDB_V470"] == 1
    assert out.loc[20, "assay_reference_M470_vs_PDB_V470"] == 0


def test_provenance_rejects_modified_reference(tmp_path):
    entries = {}
    for name in (
        "data/external/structures/6O2P.cif",
        "data/external/structures/8EIQ.cif",
        "data/processed/bihler_2024_missense_splits.parquet",
        "data/processed/bihler_2026_transfer_splits.parquet",
    ):
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
        entries[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    report = {
        "file_sha256": entries,
        "pdb_sources": {pdb: {"file_sha256": entries[f"data/external/structures/{pdb}.cif"],
                              "ligands_without_coordinates": []}
                        for pdb in sf.PDB_IDS},
    }
    sf.validate_source_files(tmp_path, report)
    (tmp_path / "data/external/structures/6O2P.cif").write_text("changed")
    with pytest.raises(ValueError, match="provenance"):
        sf.validate_source_files(tmp_path, report)


def test_no_assay_or_response_tables_used_by_module():
    import inspect
    source = inspect.getsource(sf)
    assert 'read_parquet(' not in source
    assert 'results/transfer/' not in source
    assert 'bihler_2026_vti.parquet' not in source
    assert 'bihler_2024_theratyping.parquet' not in source
