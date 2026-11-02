"""Phase 16A: leakage-safe FoldX mutation-stability pilot on CFTR 6O2P.

Commands:
    python -m cftr_theratype.stability prepare --pilot-size 12
    python -m cftr_theratype.stability run --foldx /path/to/foldx --limit 2
    python -m cftr_theratype.stability collect

Only identity/split metadata are loaded, never ETI/VTI response values. 6O2P is
an experimental *reference construct* (V470, E1371Q), not the assay's M470
background or a structure of the variant. Ligands are excluded from FoldX input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB import PDBParser

from cftr_theratype.structure_coverage import (
    ROOT, align_to_canonical, canonical_cftr_sequence,
    select_cftr_entity, sha256_file,
)

PDB_ID = "6O2P"
RES3 = {
    "A":"ALA", "C":"CYS", "D":"ASP", "E":"GLU", "F":"PHE",
    "G":"GLY", "H":"HIS", "I":"ILE", "K":"LYS", "L":"LEU",
    "M":"MET", "N":"ASN", "P":"PRO", "Q":"GLN", "R":"ARG",
    "S":"SER", "T":"THR", "V":"VAL", "W":"TRP", "Y":"TYR",
}
THREE_TO_ONE = {v: k for k,v in RES3.items()}
HGVS = re.compile(r"^p\.\(?([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)?$")
EXPECTED_COUNTS = {"train":210, "val":52, "test":33, "excluded":120}
WORK_REL = Path("results/structure/phase16_foldx")


def parse_hgvs(value: str):
    match = HGVS.fullmatch(str(value).strip())
    if not match:
        return None
    left, number, right = match.groups()
    code = {v.title(): k for k,v in RES3.items()}
    if left not in code or right not in code:
        return None
    p = int(number)
    if not 1 <= p <= 1480 or left == right:
        return None
    return (code[left], p, code[right])


def sha(path: Path) -> str:
    return sha256_file(path)


def _col(cif, name, n=None, default=None):
    v = cif.get(name)
    if v is None:
        if n is not None and default is not None:
            return [default]*n
        raise ValueError(f"mmCIF lacks {name}")
    return v if isinstance(v, list) else [v]


def extract_reference_protein(cif: dict, canonical: str):
    """Generate canonical-numbered, protein-only chain A PDB from mmCIF.

    Mapping is from deposited polymer via sequence alignment and mmCIF
    label_seq_id, NEVER author residue numbers. Keep observed residue identity.
    We do not add atoms, remodel residues, or silently alter E1371Q/V470.
    """
    entity, deposited = select_cftr_entity(cif, canonical)
    mapping, mismatches, identity = align_to_canonical(canonical, deposited)
    if identity < .99 or len(mapping) < 1450:
        raise ValueError("Unexpected 6O2P reference sequence alignment")
    if mismatches != {1371:("E","Q")}:
        raise ValueError(f"Unexpected construct differences: {mismatches}")
    if mapping.get(508) is None or mapping.get(470) is None:
        raise ValueError("F508 or V470 absent from 6O2P construct")
    dep_to_ref = {d:r for r,d in mapping.items()}
    chain_ids = _col(cif, "_atom_site.label_asym_id")
    n = len(chain_ids)
    keys = {
        k: _col(cif, "_atom_site."+k)
        for k in ("group_PDB", "label_entity_id", "label_seq_id",
                  "label_comp_id", "label_atom_id", "Cartn_x", "Cartn_y",
                  "Cartn_z", "type_symbol")
    }
    modelnums = _col(cif, "_atom_site.pdbx_PDB_model_num", n, "1")
    alt = _col(cif, "_atom_site.label_alt_id", n, ".")
    occupancy = _col(cif, "_atom_site.occupancy", n, "1.0")
    if any(len(z) != n for z in list(keys.values())+[modelnums,alt,occupancy]):
        raise ValueError("Malformed mmCIF atom table")
    # Select the highest-coverage label chain from the CFTR polymer entity.
    ca_positions = {}
    for i in range(n):
        if keys["group_PDB"][i] == "ATOM" and keys["label_entity_id"][i] == entity \
            and keys["label_atom_id"][i] == "CA" and modelnums[i] == "1":
            ca_positions.setdefault(chain_ids[i],set()).add(keys["label_seq_id"][i])
    if not ca_positions:
        raise ValueError("No CFTR CA coordinates")
    source_chain = sorted(ca_positions, key=lambda c: (-len(ca_positions[c]), c))[0]
    # Preserve only ATOM protein coordinates, no ligands or ions in energy model.
    accepted = {}
    residue_codes = {}
    for i in range(n):
        if (keys["group_PDB"][i] != "ATOM" or
            keys["label_entity_id"][i] != entity or
            chain_ids[i] != source_chain or modelnums[i] != "1" or
            alt[i] not in (".", "?", "A", "1")):
            continue
        dep_idx_s = keys["label_seq_id"][i]
        if dep_idx_s in (".", "?"):
            continue
        ref_idx = dep_to_ref.get(int(dep_idx_s))
        if ref_idx is None:
            continue
        aa3 = keys["label_comp_id"][i].upper()
        if aa3 not in THREE_TO_ONE:
            continue
        if THREE_TO_ONE[aa3] != deposited[int(dep_idx_s)-1]:
            raise ValueError(f"Atom sequence differs from deposited polymer at {ref_idx}")
        residue_codes[ref_idx] = aa3
        aname = keys["label_atom_id"][i]
        elem = keys["type_symbol"][i].upper()
        if elem in ("H", "D"):
            continue
        coord = tuple(float(keys[c][i]) for c in ("Cartn_x", "Cartn_y", "Cartn_z"))
        if not np.isfinite(coord).all():
            continue
        key = (ref_idx,aname)
        if key not in accepted or (alt[i] in (".","?") and accepted[key][3] not in (".","?")):
            accepted[key] = (aa3,coord,elem,alt[i],float(occupancy[i]))
    positions = sorted({p for p,a in accepted if a == "CA"})
    if len(positions) < 1100 or 508 not in positions or 470 not in positions:
        raise ValueError(f"Too few resolved CAs or missing F508/V470: {len(positions)}")
    records=[]
    serial=0
    for position, atom in sorted(accepted, key=lambda x:(x[0], x[1] != "N", x[1] != "CA", x[1] != "C", x[1] != "O", x[1])):
        if position not in positions:
            continue
        aa3,coord,elem,alt_id,occ=accepted[position,atom]
        serial+=1
        if serial > 99999:
            raise ValueError("Legacy PDB atom count overflows")
        # PDB atom name: standard right justified for one-character element names.
        atom_field = f" {atom:<3}" if len(atom) < 4 and len(elem) == 1 else f"{atom:<4}"
        line=(f"ATOM  {serial:5d} {atom_field} {aa3:>3} A{position:4d}    "
              f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
              f"{occ:6.2f}{0.0:6.2f}          {elem:>2}\n")
        records.append(line)
    records.append("TER\nEND\n")
    pdb_text="".join(records)
    return pdb_text, {
        "source_label_chain":source_chain,
        "canonical_positions_with_ca": len(positions),
        "canonical_positions":positions,
        "observed_reference":{str(p):THREE_TO_ONE[residue_codes[p]] for p in positions},
        "construct_mismatches":{str(k):list(v) for k,v in mismatches.items()},
        "sequence_identity":identity,
        "note":"6O2P contains V470 and engineered E1371Q; assay reference has M470",
    }


def validate_prepared_pdb(path: Path, observed: dict):
    struct=PDBParser(QUIET=True).get_structure("cftr", str(path))
    chain=struct[0]["A"]
    for position, expected in observed.items():
        res=chain[int(position)]
        if THREE_TO_ONE.get(res.resname) != expected or "CA" not in res:
            raise ValueError(f"PDB numbering/identity/CA mismatch at {position}")
    if len(chain) != len(observed):
        raise ValueError("Prepared PDB has unexpected number of resolved residues")
    if not all(r.id[0] == " " for r in chain):
        raise ValueError("Protein-only PDB contains hetero atoms")


def load_train_identities(root: Path, assay: str):
    p = root / "data/processed/bihler_2026_transfer_splits.parquet"
    # No paired functional/label file is read; whitelist four metadata columns.
    data=pd.read_parquet(p, columns=["variant_key","hgvs_protein","protein_position","transfer_split"])
    count=data.transfer_split.astype(str).str.lower().value_counts().to_dict()
    if count != EXPECTED_COUNTS or data.variant_key.duplicated().any():
        raise ValueError(f"Unexpected frozen VTI partitions: {count}")
    # No outcomes or prior ETI val/test accessed.
    data=data[data.transfer_split.astype(str).str.lower().eq("train")].copy()
    recs=[]
    for row in data.itertuples(index=False):
        hit=parse_hgvs(row.hgvs_protein)
        rec={"variant_key":row.variant_key,"hgvs_protein":row.hgvs_protein,
             "protein_position":int(row.protein_position), "reference_aa":None,
             "alternate_aa":None,"status":"unparseable_hgvs"}
        if hit is not None:
            wt,pos,mut=hit
            rec.update(reference_aa=wt,alternate_aa=mut)
            if pos != int(row.protein_position):
                rec["status"]="position_mismatch"
            elif assay[pos-1] != wt:
                rec["status"]="assay_reference_mismatch"
            elif pos in (470,1371):
                rec["status"]="construct_background_difference"
            else:
                rec["status"]="pending_structure_check"
        recs.append(rec)
    if len(recs)!=210:
        raise ValueError("Not all 210 frozen VTI training variants present")
    return pd.DataFrame(recs),sha(p)


def annotate_candidates(data, structure_meta):
    result=data.copy()
    observed=structure_meta["observed_reference"]
    for idx,row in result.iterrows():
        if row.status!="pending_structure_check":
            continue
        aa=observed.get(str(row.protein_position))
        if aa is None:
            result.at[idx,"status"]="unresolved_in_6O2P"
        elif aa!=row.reference_aa:
            result.at[idx,"status"]="structure_reference_mismatch"
        else:
            result.at[idx,"status"]="eligible"
    result["foldx_mutation"] = result.apply(lambda r: f"{r.reference_aa}A{int(r.protein_position)}{r.alternate_aa};" if r.status == "eligible" else "", axis=1)
    return result


def select_pilot(df, n: int):
    eligible=df.loc[df.status.eq("eligible")].sort_values(["protein_position","variant_key"])
    # Same-position substitutions do not increase positional coverage in pilot.
    unique=eligible.drop_duplicates("protein_position")
    if n == 0:
        return eligible.copy()   # optional expansion only after successful pilot
    if n < 1 or len(unique)<n:
        raise ValueError("Invalid pilot size or insufficient eligible positions")
    # Uniform quantiles by sequence position, not by measured outcome.
    indices=np.floor((np.arange(n)+.5)*len(unique)/n).astype(int)
    return unique.iloc[indices].copy()


def atomic_write_json(path, payload):
    path.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")


def prepare(root: Path, pilot_size: int=12, force=False):
    root=Path(root)
    src=root/"data/external/structures/6O2P.cif"
    report=root/"results/structure/phase15_coverage_summary.json"
    if not src.is_file() or not report.is_file():
        raise FileNotFoundError("Run Phase 15A first: 6O2P.cif and structural coverage summary required")
    meta=json.loads(report.read_text())
    expected=meta["pdb_sources"]["6O2P"]["file_sha256"]
    if sha(src)!=expected:
        raise ValueError("6O2P CIF checksum differs from frozen Phase15A reference")
    canonical,assay=canonical_cftr_sequence()
    pdbtext,structure_meta=extract_reference_protein(MMCIF2Dict(str(src)),canonical)
    data,splitsha=load_train_identities(root,assay)
    candidates=annotate_candidates(data,structure_meta)
    pilot=select_pilot(candidates,pilot_size)
    out=root/WORK_REL
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(f"{out} exists. Use --force only if you intentionally want to reset it")
    if out.exists() and force:
        # Refuse destructive reset of previous physical computations.
        if list(out.rglob("Dif_*.fxout")) or list(out.rglob("*_Repair.pdb")):
            raise ValueError("FoldX computations exist; refuse --force reset. Archive workdir manually")
    out.mkdir(parents=True,exist_ok=True)
    (out/"6O2P_cftr_canonical_numbering.pdb").write_text(pdbtext)
    validate_prepared_pdb(out/"6O2P_cftr_canonical_numbering.pdb",structure_meta["observed_reference"])
    candidates.to_csv(out/"candidates_train_only.csv",index=False)
    pilot.to_csv(out/"pilot_train_only.csv",index=False)
    manifest={
        "phase":"16A_foldx_stability_pilot", "source_pdb":"6O2P", "source_cif_sha256":sha(src),
        "phase15a_summary_sha256":sha(report), "vti_splits_sha256":splitsha,
        "prepared_pdb_sha256":sha(out/"6O2P_cftr_canonical_numbering.pdb"),
        "pilot_sha256":sha(out/"pilot_train_only.csv"), "n_train":len(candidates),
        "n_pilot":len(pilot), "status_counts":candidates.status.value_counts().to_dict(),
        "construct_differences":structure_meta["construct_mismatches"],
        "reference_background":"PDB: V470/E1371Q, VTI assay: M470/E1371",
        "ligands_in_foldx_model":False, "responses_loaded":False,"validation_test_outcomes_loaded":False,
        "foldx_executable_included":False,
        "interpretation":"Mutation effects on FoldX total energy in 6O2P protein-only reference conformation, NOT predicted ligand affinity or clinical response",
    }
    atomic_write_json(out/"manifest.json",manifest)
    print("="*83)
    print("CFTR-THERATYPE | PHASE 16A FOLDX STABILITY PILOT — PREPARED")
    print("="*83)
    print("  Source CIF:",src.relative_to(root))
    print("  Prepared protein-only PDB: 6O2P_cftr_canonical_numbering.pdb")
    print(f"  6O2P resolved CA positions: {structure_meta['canonical_positions_with_ca']}")
    print("  VTI train identities:",len(candidates))
    print("  Candidate status:",dict(candidates.status.value_counts()))
    print(f"  Selected pilot: {len(pilot)} mutations / {pilot.protein_position.nunique()} positions")
    print(pilot[["hgvs_protein","foldx_mutation","protein_position"]].to_string(index=False))
    print("  Source/reference differences: V470 vs assay M470; engineered E1371Q")
    print("  Excluded ligands/ions: no drug-binding energies are estimated")
    print("  No functional response, validation or test outcomes accessed")
    print("  Files:",out.relative_to(root))
    print("="*83)
    return out


def parse_foldx_dif(path: Path):
    """Read FoldX BuildModel Dif file. First numeric column is Total Energy ΔΔG."""
    rows=[]
    for line in path.read_text(errors="replace").splitlines():
        parts=[x.strip() for x in line.split("\t")]
        if len(parts)<2:
            continue
        # Header and metadata lines contain nonnumeric second column.
        try:
            val=float(parts[1])
        except (ValueError, TypeError):
            continue
        if not np.isfinite(val) or not re.search(r"[A-Za-z]", parts[0]):
            continue
        rows.append(val)
    if not rows:
        raise ValueError(f"No FoldX total energy differences in {path}")
    return rows


def find_foldx(binary: str):
    resolved=shutil.which(binary)
    if resolved is None:
        p=Path(binary).expanduser()
        if p.is_file():
            resolved=str(p.resolve())
    if not resolved:
        raise FileNotFoundError("FoldX executable not found. Obtain an academic license at https://foldxsuite.crg.eu/academic-license-info and pass --foldx /absolute/path/to/foldx")
    return Path(resolved).resolve()


def run_foldx(root: Path, foldx: str, limit: int, runs: int):
    root=Path(root)
    out=root/WORK_REL
    manifest=json.loads((out/"manifest.json").read_text())
    protein=out/"6O2P_cftr_canonical_numbering.pdb"
    pilot=out/"pilot_train_only.csv"
    if sha(protein)!=manifest["prepared_pdb_sha256"] or sha(pilot)!=manifest["pilot_sha256"]:
        raise ValueError("Prepared PDB/pilot differs from approved manifest")
    exe=find_foldx(foldx)
    candidates=pd.read_csv(pilot).head(limit if limit>0 else None)
    if runs not in (1,2,3,4,5):
        raise ValueError("Restrict pilot to 1–5 independent repeats")
    shared=out/"repair"
    shared.mkdir(exist_ok=True)
    repaired=shared/"6O2P_cftr_canonical_numbering_Repair.pdb"
    rotabase=exe.parent/"rotabase.txt"
    if rotabase.exists():
        shutil.copy2(rotabase,shared/"rotabase.txt")
    if not repaired.exists():
        shutil.copy2(protein,shared/protein.name)
        cmd=[str(exe),"--command=RepairPDB",f"--pdb={protein.name}"]
        proc=subprocess.run(cmd,cwd=shared,capture_output=True,text=True,timeout=5400,check=False)
        (shared/"repair_stdout.log").write_text(proc.stdout+"\nSTDERR:\n"+proc.stderr)
        if proc.returncode!=0 or not repaired.is_file():
            raise RuntimeError(f"FoldX RepairPDB failed (exit {proc.returncode}). See {shared/'repair_stdout.log'}")
    print("="*83)
    print("CFTR-THERATYPE | FOLDX PILOT, EXPERIMENTAL 6O2P CONSTRUCT")
    print("="*83)
    print("  FoldX path:",exe)
    print("  SHA256:",sha(exe))
    print("  Repaired PDB:",repaired)
    print("  Variants selected:",len(candidates),"; numberOfRuns:",runs)
    for row in candidates.itertuples(index=False):
        # Prevent collision or path issues from arbitrary HGVS cDNA string.
        folder=out/"runs"/hashlib.sha256(str(row.variant_key).encode()).hexdigest()[:16]
        folder.mkdir(parents=True,exist_ok=True)
        result=folder/"result.json"
        if result.exists():
            print("  SKIP already computed:",row.hgvs_protein)
            continue
        shutil.copy2(repaired,folder/repaired.name)
        if rotabase.exists():
            shutil.copy2(rotabase,folder/"rotabase.txt")
        (folder/"individual_list.txt").write_text(row.foldx_mutation+"\n")
        cmd=[str(exe),"--command=BuildModel",f"--pdb={repaired.name}",
             "--mutant-file=individual_list.txt",f"--numberOfRuns={runs}","--out-pdb=false"]
        proc=subprocess.run(cmd,cwd=folder,capture_output=True,text=True,timeout=5400,check=False)
        (folder/"foldx_stdout.log").write_text(proc.stdout+"\nSTDERR:\n"+proc.stderr)
        dif=list(folder.glob("Dif_*.fxout"))
        if proc.returncode != 0 or len(dif)!=1:
            print(f"  FAILURE: {row.hgvs_protein}, status={proc.returncode}; see {folder/'foldx_stdout.log'}")
            continue
        try:
            energies=parse_foldx_dif(dif[0])
            if len(energies) != runs:
                raise ValueError(f"Expected {runs} FoldX ΔΔG rows, found {len(energies)}")
        except ValueError as exc:
            print(f"  INVALID OUTPUT: {row.hgvs_protein}: {exc}")
            continue
        payload={"variant_key":row.variant_key,"hgvs_protein":row.hgvs_protein,
                 "foldx_mutation":row.foldx_mutation,"protein_position":int(row.protein_position),
                 "foldx_ddg_kcal_mol_mean":float(np.mean(energies)),
                 "foldx_ddg_kcal_mol_sd":float(np.std(energies,ddof=1)) if len(energies)>1 else None,
                 "runs":energies,"dif_sha256":sha(dif[0]),
                 "reference_pdb_sha256":sha(repaired),"foldx_executable_sha256":sha(exe),
                 "model":"ligand-free repaired experimental 6O2P protein construct"}
        atomic_write_json(result,payload)
        print(f"  PASS {row.hgvs_protein}: ΔΔG mean {np.mean(energies):+.3f} kcal/mol")
    print("  Next: python -m cftr_theratype.stability collect")


def collect(root: Path):
    root=Path(root)
    out=root/WORK_REL
    manifest=json.loads((out/"manifest.json").read_text())
    pilot=pd.read_csv(out/"pilot_train_only.csv")
    if sha(out/"pilot_train_only.csv")!=manifest["pilot_sha256"]:
        raise ValueError("Pilot membership file has changed")
    recs=[]
    for row in pilot.itertuples(index=False):
        folder=out/"runs"/hashlib.sha256(str(row.variant_key).encode()).hexdigest()[:16]
        p=folder/"result.json"
        if p.exists():
            r=json.loads(p.read_text())
            if r["variant_key"]!=row.variant_key or r["foldx_mutation"]!=row.foldx_mutation:
                raise ValueError("FoldX result mismatched mutation identity")
            recs.append({"variant_key":row.variant_key,"hgvs_protein":row.hgvs_protein,
                         "protein_position":int(row.protein_position),
                         "foldx_ddg_kcal_mol":r["foldx_ddg_kcal_mol_mean"],
                         "foldx_ddg_sd_kcal_mol":r["foldx_ddg_kcal_mol_sd"],
                         "status":"computed"})
        else:
            recs.append({"variant_key":row.variant_key,"hgvs_protein":row.hgvs_protein,
                         "protein_position":int(row.protein_position),
                         "foldx_ddg_kcal_mol":None,"foldx_ddg_sd_kcal_mol":None,
                         "status":"pending"})
    frame=pd.DataFrame(recs)
    frame.to_csv(out/"pilot_stability_results.csv",index=False)
    print("="*83)
    print("CFTR-THERATYPE | PHASE 16A FOLDX PILOT STATUS")
    print("="*83)
    print(f"  Computed: {(frame.status == 'computed').sum()}/{len(frame)}")
    for row in frame.loc[frame.status.eq("computed")].itertuples(index=False):
        print(f"  {row.hgvs_protein:20s} ΔΔG={row.foldx_ddg_kcal_mol:+.3f} kcal/mol")
    print("  Output:", (out/"pilot_stability_results.csv").relative_to(root))
    print("  Labels/test data: NOT ACCESSED")
    print("  Not yet an independent stability benchmark or CFTR response result")
    print("="*83)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["prepare","run","collect"])
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--pilot-size",type=int,default=12)
    p.add_argument("--force",action="store_true")
    p.add_argument("--foldx",default="foldx")
    p.add_argument("--limit",type=int,default=2,help="Only score first two pilot variants initially; 0 means all")
    p.add_argument("--runs",type=int,default=3)
    args=p.parse_args()
    if args.action=="prepare":prepare(args.root,args.pilot_size,args.force)
    elif args.action=="run":run_foldx(args.root,args.foldx,args.limit,args.runs)
    else:collect(args.root)

if __name__=="__main__":main()
