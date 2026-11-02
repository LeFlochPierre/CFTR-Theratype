import math
from pathlib import Path

import pandas as pd
import pytest

from cftr_theratype.thermompnn_adapter import join_predictions, pdb_reference


def toy_pdb(tmp_path):
    pdb = tmp_path / "toy.pdb"
    lines = []
    for n, residue in [(11, "ALA"), (13, "GLY")]:
        for atom_idx, atom in enumerate(("N", "CA", "C", "O")):
            lines.append(f"ATOM  {n*10+atom_idx:5d}  {atom:<3} {residue:3} A{n:4d}    "
                         f"{n*1.0:8.3f}{atom_idx*1.0:8.3f}{n*1.0:8.3f}{1.0:6.2f}{10.:6.2f}           {atom[0]:>1}\n")
    pdb.write_text("".join(lines)+"TER\nEND\n")
    return pdb


def toy_manifest():
    rows=[]
    for n in range(210):
        status = "eligible" if n==0 else ("eligible" if n==1 else "unresolved_in_6O2P")
        rows.append(dict(variant_key=f"v{n}",hgvs_protein=f"p.Variant{n}",
                         protein_position=11 if n==0 else 13 if n==1 else n+30,
                         reference_aa="A" if n==0 else "G" if n==1 else "A",
                         alternate_aa="V" if n==0 else "D" if n==1 else "T",status=status))
    return pd.DataFrame(rows)


def toy_scores():
    # Upstream parser inserts a '-' at the missing position 12.
    return pd.DataFrame(dict(position=[0,2],wildtype=["A","G"],mutation=["V","D"],ddG_pred=[1.2,-0.4]))


def test_pdb_gap_mapping(tmp_path):
    start, coords = pdb_reference(toy_pdb(tmp_path))
    assert start==11
    assert coords=={11:"A",13:"G"}


def test_join_without_outcomes(tmp_path):
    result=join_predictions(toy_pdb(tmp_path),toy_manifest(),toy_scores())
    assert len(result)==210
    assert result.iloc[0].thermompnn_ddg_pred==1.2
    assert result.iloc[1].thermompnn_ddg_pred==-0.4
    assert result.thermompnn_ddg_pred.notna().sum()==2
    assert set(result.columns)=={"variant_key", "hgvs_protein", "protein_position", "reference_aa", "alternate_aa", "status", "thermompnn_ddg_pred", "thermompnn_status"}


def test_wildtype_mismatch_fails_closed(tmp_path):
    bad=toy_scores();bad.loc[0,"wildtype"]="R"
    with pytest.raises(ValueError,match="Unsafe ThermoMPNN"):
        join_predictions(toy_pdb(tmp_path),toy_manifest(),bad)


def test_missing_eligible_predictions_error(tmp_path):
    with pytest.raises(ValueError,match="No finite ThermoMPNN"):
        join_predictions(toy_pdb(tmp_path),toy_manifest(),toy_scores().iloc[:1])


def test_nonfinite_predictions_error(tmp_path):
    bad=toy_scores();bad.loc[1,"ddG_pred"]=float('nan')
    with pytest.raises(ValueError,match="No finite ThermoMPNN"):
        join_predictions(toy_pdb(tmp_path),toy_manifest(),bad)


def test_duplicate_predictions_error(tmp_path):
    bad=pd.concat([toy_scores(),toy_scores().iloc[[0]]],ignore_index=True)
    with pytest.raises(ValueError,match="Duplicated ThermoMPNN"):
        join_predictions(toy_pdb(tmp_path),toy_manifest(),bad)


def test_unknown_reference_position_error(tmp_path):
    bad=toy_scores();bad.loc[0,"position"]=1
    with pytest.raises(ValueError,match="unobserved PDB positions"):
        join_predictions(toy_pdb(tmp_path),toy_manifest(),bad)
