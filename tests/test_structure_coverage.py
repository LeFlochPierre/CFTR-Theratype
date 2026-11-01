"""Phase 15A structure identity and alignment QC: no outcome values in any fixture."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cftr_theratype import structure_coverage as sc


def fake_cftr():
    # Canonical V470, F508; long enough to mirror full CFTR indexing.
    s = list("A" * 1480)
    s[469] = "V"
    s[507] = "F"
    return "".join(s)


def test_alignment_handles_f508_deletion_and_m470_construct_mismatch():
    ref = fake_cftr()
    dep = list(ref)
    dep[469] = "M"
    dep.pop(507)
    mapping, mismatches, identity = sc.align_to_canonical(ref, "".join(dep))
    assert identity > .99
    assert mapping[470] == 470
    assert mismatches[470] == ("V", "M")
    assert 508 not in mapping
    assert mapping[509] == 508


def test_reject_poor_alignment():
    with pytest.raises(ValueError, match="identity"):
        sc.align_to_canonical("A" * 1480, "Y" * 1480)


def fake_cif():
    ref = fake_cftr()
    dep = list(ref)
    dep[469] = "M"
    dep.pop(507)
    dep = "".join(dep)
    # Label_seq_id 470 is construct M470; 510 corresponds to canonical position 511.
    atom_rows = [
        ("ATOM", "1", "A", "470", "CA", "MET", "C", "0", "0", "0"),
        ("ATOM", "1", "A", "510", "CA", "ALA", "C", "8", "0", "0"),
        ("HETATM", "2", "B", ".", "C1", "VX7", "C", "2", "0", "0"),
        ("HETATM", "3", "C", ".", "C1", "CV6", "C", "11", "0", "0"),
        ("HETATM", "4", "D", ".", "C1", "WJX", "C", "14", "0", "0"),
    ]
    names = ["group_PDB", "label_entity_id", "label_asym_id", "label_seq_id",
             "label_atom_id", "label_comp_id", "type_symbol", "Cartn_x", "Cartn_y", "Cartn_z"]
    dic = {"_entity_poly.entity_id": ["2", "1"],
           "_entity_poly.pdbx_seq_one_letter_code_can": ["A" * 17, dep]}
    for j, field in enumerate(names):
        dic["_atom_site." + field] = [row[j] for row in atom_rows]
    return dic


def test_entity_selection_ignores_short_peptide():
    entity, sequence = sc.select_cftr_entity(fake_cif(), fake_cftr())
    assert entity == "1"
    assert len(sequence) == 1479


def test_coordinates_exclude_ligands_from_protein_residue_map():
    chain, coords, ligands = sc.parse_coordinates(fake_cif(), "1")
    assert chain == "A"
    assert set(coords) == {470, 510}
    assert set(ligands) == {"VX7", "CV6", "WJX"}
    np.testing.assert_allclose(ligands["VX7"], [[2, 0, 0]])


def test_full_structure_mapping_flags_deletion_unresolved_and_mismatch():
    rows, meta = sc.position_features(fake_cftr(), fake_cif(), "8EIQ", ["VX7", "CV6", "WJX"])
    assert rows[508]["8EIQ_status"] == "not_in_deposited_construct"
    assert rows[470]["8EIQ_status"] == "resolved_construct_mismatch"
    assert rows[511]["8EIQ_status"] == "resolved_exact"
    assert rows[512]["8EIQ_status"] == "deposited_but_unresolved"
    assert rows[470]["8EIQ_VX7_ca_distance_A"] == 2.0
    assert rows[511]["8EIQ_CV6_ca_distance_A"] == 3.0
    assert meta["cftr_chain_label_asym_id"] == "A"
    assert meta["resolved_mapped_residues"] == 2


def test_coverage_categories_are_conservative():
    def categorise(status1, status2, pos=10):
        return sc.categorise_coverage(pd.Series({"protein_position": pos,
                                                  "6O2P_status": status1,
                                                  "8EIQ_status": status2}))
    assert categorise("resolved_exact", "resolved_exact") == "exact_both"
    assert categorise("resolved_exact", "deposited_but_unresolved") == "exact_6O2P_only"
    assert categorise("resolved_construct_mismatch", "deposited_but_unresolved") == "construct_mismatch_only_or_unresolved"
    assert categorise("resolved_construct_mismatch", "resolved_exact") == "exact_8EIQ_only"
    assert categorise("deposited_but_unresolved", "not_in_deposited_construct") == "no_exact_reference_coordinates"
    assert categorise("resolved_exact", "resolved_exact", pos=pd.NA) == "unparseable_variant_position"


def test_position_denominator_and_variant_denominator():
    df = pd.DataFrame({"study": ["VTI"] * 4, "partition": ["train"] * 4,
                       "protein_position": [10, 10, 15, pd.NA],
                       "coverage_category": ["exact_both", "exact_both", "no_exact_reference_coordinates", "unparseable_variant_position"]})
    result = sc.summarise_coverage(df)
    assert result[0]["variants"] == 4
    assert result[0]["distinct_positions"] == 2
    assert result[0]["either_exact_variant_count"] == 2
    assert result[0]["either_exact_position_count"] == 1
    assert result[0]["either_exact_position_fraction"] == .5
    assert result[0]["missing_or_unparseable_variants"] == 1


def test_only_identity_columns_are_requested(tmp_path, monkeypatch):
    data = tmp_path / "data/processed"
    data.mkdir(parents=True)
    for study, (fname, _, expected) in sc.SOURCE_META.items():
        (data / fname).write_text(study)
    requested = []
    def reader(path, columns):
        requested.append(tuple(columns))
        study = "ETI" if "2024" in str(path) else "VTI"
        fname, split_col, n = sc.SOURCE_META[study]
        return pd.DataFrame({"variant_key": [f"{study}_{x}" for x in range(n)],
                             "protein_position": [x % 1480 + 1 for x in range(n)],
                             split_col: ["train"] * n})
    monkeypatch.setattr(sc.pd, "read_parquet", reader)
    frame, hashes = sc.load_identity_only(tmp_path)
    assert len(frame) == 1001
    assert requested == [
        ("variant_key", "protein_position", "split"),
        ("variant_key", "protein_position", "transfer_split"),
    ]
    assert len(hashes) == 2


def test_download_never_overwrites_existing(tmp_path, monkeypatch):
    dest = tmp_path / "6O2P.cif"
    dest.write_bytes(b"x" * 110_000)
    monkeypatch.setattr(sc, "urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("network accessed")))
    sc.download_reference(dest, "6O2P")
    assert dest.stat().st_size == 110_000


def test_real_mmcif_text_parser_roundtrip(tmp_path):
    # Exercises Bio's file reader, rather than only constructing dictionaries.
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    dic = fake_cif()
    out = ["data_synthetic\n", "loop_\n",
           "_entity_poly.entity_id\n", "_entity_poly.pdbx_seq_one_letter_code_can\n"]
    for entity, seq in zip(dic["_entity_poly.entity_id"], dic["_entity_poly.pdbx_seq_one_letter_code_can"]):
        out.append(f"{entity} '{seq}'\n")
    keys = ["group_PDB", "label_entity_id", "label_asym_id", "label_seq_id",
            "label_atom_id", "label_comp_id", "type_symbol", "Cartn_x", "Cartn_y", "Cartn_z"]
    out.append("loop_\n")
    out.extend("_atom_site." + k + "\n" for k in keys)
    for i in range(len(dic["_atom_site.group_PDB"])):
        out.append(" ".join(dic["_atom_site." + key][i] for key in keys) + "\n")
    path = tmp_path / "fixture.cif"
    path.write_text("".join(out))
    loaded = MMCIF2Dict(str(path))
    rows, _ = sc.position_features(fake_cftr(), loaded, "8EIQ", ["VX7", "CV6", "WJX"])
    assert rows[511]["8EIQ_status"] == "resolved_exact"
    assert rows[508]["8EIQ_status"] == "not_in_deposited_construct"
