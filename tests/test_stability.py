"""Phase 16A safety/unit tests: do not require FoldX or outcome files."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cftr_theratype import stability as s


def test_hgvs_parser_and_rejections():
    assert s.parse_hgvs("p.Gly551Asp") == ("G",551,"D")
    assert s.parse_hgvs("p.(Leu1077Pro)") == ("L",1077,"P")
    assert s.parse_hgvs("p.Phe508del") is None
    assert s.parse_hgvs("p.Gly551Gly") is None
    assert s.parse_hgvs("p.Gly0Asp") is None
    assert s.parse_hgvs("nonsense") is None


def test_candidates_verify_structure_and_assay_identity():
    x=pd.DataFrame([
        {"variant_key":"1", "protein_position":551, "reference_aa":"G", "alternate_aa":"D", "status":"pending_structure_check"},
        {"variant_key":"2", "protein_position":470, "reference_aa":"M", "alternate_aa":"V", "status":"construct_background_difference"},
        {"variant_key":"3", "protein_position":900, "reference_aa":"A", "alternate_aa":"G", "status":"pending_structure_check"},
        {"variant_key":"4", "protein_position":800, "reference_aa":"D", "alternate_aa":"G", "status":"pending_structure_check"},
    ])
    result=s.annotate_candidates(x,{"observed_reference":{"551":"G","800":"A"}})
    assert result.status.tolist()==["eligible","construct_background_difference","unresolved_in_6O2P","structure_reference_mismatch"]
    assert result.foldx_mutation.tolist()==["GA551D;","","",""]


def test_pilot_deterministic_without_outcomes():
    x=pd.DataFrame({"variant_key":[str(x) for x in [1,2,3,4,5,6]],
                    "protein_position":[50,100,100,200,500,700],
                    "status":["eligible"]*6,
                    "foldx_mutation":["WA50Y;"]*6})
    y=s.select_pilot(x,3)
    assert y.protein_position.nunique()==3
    assert y.variant_key.tolist()==s.select_pilot(x,3).variant_key.tolist()
    assert len(s.select_pilot(x,0))==6


def test_parses_foldx_dif_output(tmp_path):
    p=tmp_path/"Dif_test.fxout"
    p.write_text("FoldX 6 build model\nPdb\tTotal Energy\tBackbone Hbond\nBM_run1.pdb\t2.35\t0.4\nBM_run2.pdb\t-1.2\t0.2\n")
    assert s.parse_foldx_dif(p)==[2.35,-1.2]
    p.write_text("FoldX\nPdb\tTotal Energy\n")
    with pytest.raises(ValueError,match="No FoldX"):
        s.parse_foldx_dif(p)


def test_identity_only_loader_no_outcomes(monkeypatch, tmp_path):
    pytest.importorskip("pyarrow")
    p=tmp_path/"data/processed/bihler_2026_transfer_splits.parquet"
    p.parent.mkdir(parents=True)
    values=[]
    for partition,n in s.EXPECTED_COUNTS.items():
        for k in range(n):
            values.append({"variant_key":f"{partition}_{k}", "hgvs_protein":"p.Gly551Asp", "protein_position":551,
                           "transfer_split":partition, "vti_rescue_pp":99999.0})
    pd.DataFrame(values).to_parquet(p,index=False)
    # M470 assay reference doesn't affect this Gly551 annotation
    assay="A"*1480
    df, filehash=s.load_train_identities(tmp_path,assay)
    assert len(df)==210
    assert set(df.status)=={"assay_reference_mismatch"}
    assert "vti_rescue_pp" not in df
    assert len(filehash)==64


def test_extract_reference_pdb_residue_canonical_numbering(monkeypatch,tmp_path):
    canonical=list("A"*1480)
    canonical[469]="V";canonical[507]="F";canonical[1370]="E"
    canonical="".join(canonical)
    deposited=canonical[:1370]+"Q"+canonical[1371:]
    atom_fields={k:[] for k in ("group_PDB","label_entity_id","label_asym_id",
        "label_seq_id","label_comp_id","label_atom_id","type_symbol","Cartn_x","Cartn_y","Cartn_z")}
    # Include 1170 resolved residues, including canonical sites 470, 508, 1371
    positions=list(range(1,1168))+[1371,1372,1373]
    for p in positions:
        aa=deposited[p-1]
        for atom,elem,x in [("N","N",0.),("CA","C",1.),("C","C",2.),("O","O",3.)]:
            vals=("ATOM","1","A",str(p),s.RES3[aa],atom,elem,str(x),str(float(p)),"0")
            for k,v in zip(atom_fields,vals):
                atom_fields[k].append(v)
    cif={"_entity_poly.entity_id":["1"],"_entity_poly.pdbx_seq_one_letter_code_can":[deposited]}
    cif.update({"_atom_site."+k:v for k,v in atom_fields.items()})
    text,meta=s.extract_reference_protein(cif,canonical)
    assert meta["canonical_positions_with_ca"]==1170
    assert meta["observed_reference"]["1371"]=="Q"
    assert meta["observed_reference"]["470"]=="V"
    p=tmp_path/"pdb.pdb"
    p.write_text(text)
    s.validate_prepared_pdb(p,meta["observed_reference"])
    assert "HETATM" not in text
    # Missing positions do not split the same chain into duplicate TER segments
    assert text.count("TER\n")==1


def test_pilot_results_fail_on_mismatched_identity(tmp_path):
    work=tmp_path/s.WORK_REL
    work.mkdir(parents=True)
    pd.DataFrame([{"variant_key":"abc","hgvs_protein":"p.Gly551Asp", "protein_position":551,
                   "foldx_mutation":"GA551D;"}]).to_csv(work/"pilot_train_only.csv",index=False)
    s.atomic_write_json(work/"manifest.json",{"pilot_sha256":s.sha(work/"pilot_train_only.csv")})
    d=work/"runs"/ __import__("hashlib").sha256(b"abc").hexdigest()[:16]
    d.mkdir(parents=True)
    s.atomic_write_json(d/"result.json",{"variant_key":"wrong", "foldx_mutation":"GA551D;"})
    with pytest.raises(ValueError,match="mismatched"):
        s.collect(tmp_path)


def test_prepare_refuses_unexpected_source_hash(tmp_path):
    d=tmp_path/"data/external/structures"
    d.mkdir(parents=True)
    (d/"6O2P.cif").write_text("FAKE")
    p=tmp_path/"results/structure/phase15_coverage_summary.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"pdb_sources":{"6O2P":{"file_sha256":"bad"}}}))
    with pytest.raises(ValueError,match="checksum"):
        s.prepare(tmp_path)


def test_no_foldx_executable_actionable_error():
    with pytest.raises(FileNotFoundError,match="academic license"):
        s.find_foldx("definitely-nonexistent-foldx")
