#!/usr/bin/env python3
"""Phase 27D-3A: audited 3D CFTR-ligand geometry, NO RESPONSE LABELS.

Inputs: 8EIG.pdb (VX-445/WJX), 7SV7.pdb (tezacaftor/CV6),
        P13569.fasta (UniProt canonical CFTR).
Coordinates are experimental conformations, NOT variant-specific structures.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser
from Bio.SeqIO import read as read_sequence
from Bio.SeqUtils import seq1
from scipy.spatial import cKDTree

SPEC = [
    ("8EIG", "WJX", "elexacaftor_VX445"),
    ("7SV7", "CV6", "tezacaftor_VX661"),
]
EXPECTED_LENGTH = 1480


def heavy_coordinates(residue):
    xyz = []
    for atom in residue.get_atoms():
        if atom.element.upper() not in ("H", "D"):
            xyz.append(np.asarray(atom.coord, dtype=float))
    return np.asarray(xyz, dtype=float).reshape(-1, 3)


def audited_structure(pdb_path, ligand_id, canonical):
    model = next(PDBParser(QUIET=True).get_structure(pdb_path.stem, str(pdb_path)).get_models())
    # Largest protein chain is CFTR; explicitly require its chain ID to be A.
    protein_chains = sorted(
        ((sum(1 for r in chain if r.id[0] == ' '), chain) for chain in model),
        key=lambda pair: pair[0], reverse=True,
    )
    if not protein_chains or protein_chains[0][0] < 900:
        raise ValueError(f"{pdb_path}: no CFTR-sized protein chain")
    _, chain = protein_chains[0]
    if chain.id != 'A':
        raise ValueError(f"{pdb_path}: expected CFTR auth chain A, got {chain.id}; inspect mapping")

    ligand_residues = [r for c in model for r in c if r.resname.upper() == ligand_id]
    if len(ligand_residues) != 1:
        raise ValueError(
            f"{pdb_path}: expected exactly one {ligand_id} ligand, found {len(ligand_residues)}"
        )
    ligand_coords = heavy_coordinates(ligand_residues[0])
    if len(ligand_coords) < 8 or not np.isfinite(ligand_coords).all():
        raise ValueError(f"{pdb_path}: invalid ligand heavy atoms: {len(ligand_coords)}")
    ligand_tree = cKDTree(ligand_coords)

    row_by_position = {}
    matches = 0
    mismatches = []
    for residue in chain:
        if residue.id[0] != ' ':
            continue
        position = int(residue.id[1])  # auth numbering only after sequence-identity audit
        if position < 1 or position > EXPECTED_LENGTH:
            continue
        aa = seq1(residue.resname, custom_map={"UNK": "X"})
        expected = canonical[position - 1]
        if aa != expected:
            mismatches.append((position, aa, expected))
        else:
            matches += 1
        atoms = heavy_coordinates(residue)
        if not len(atoms):
            continue
        if position in row_by_position:
            raise ValueError(f"{pdb_path}: duplicate author position {position}")
        dmin = float(ligand_tree.query(atoms, k=1)[0].min())
        ca_d = (
            float(ligand_tree.query(residue['CA'].coord, k=1)[0])
            if 'CA' in residue else np.nan
        )
        row_by_position[position] = (dmin, ca_d, aa)

    total = matches + len(mismatches)
    agreement = matches / max(total, 1)
    if total < 900 or agreement < 0.97:
        raise ValueError(
            f"{pdb_path}: unsafe author-number to UniProt mapping: "
            f"matches={matches}/{total} ({agreement:.2%}); "
            f"first mismatches={mismatches[:15]}. "
            "Do not silently shift PDB numbering."
        )

    rows = []
    for position in range(1, EXPECTED_LENGTH + 1):
        measurement = row_by_position.get(position)
        rows.append({
            "position": position,
            "dmin_A": measurement[0] if measurement else np.nan,
            "dca_A": measurement[1] if measurement else np.nan,
            "observed": int(measurement is not None),
        })

    frame = pd.DataFrame(rows)
    info = {
        "pdb": pdb_path.stem.upper(),
        "ligand_id": ligand_id,
        "file_sha256": hashlib.sha256(pdb_path.read_bytes()).hexdigest(),
        "cftr_auth_chain": chain.id,
        "modeled_residues": len(row_by_position),
        "matches_canonical": matches,
        "mismatches_canonical": len(mismatches),
        "canonical_agreement": agreement,
        "ligand_heavy_atoms": len(ligand_coords),
        "protein_contacts_lt5A": int((frame.dmin_A < 5).sum()),
        "first_mismatches": str(mismatches[:15]),
    }
    return frame, info


def self_test():
    from Bio.PDB.Atom import Atom
    from Bio.PDB.Residue import Residue
    r = Residue((' ', 1, ' '), 'ALA', '')
    r.add(Atom('CA', np.array([0., 0., 0.]), 1., 1., ' ', ' CA ', 1, element='C'))
    assert np.allclose(heavy_coordinates(r), [[0., 0., 0.]])
    tree = cKDTree(np.array([[3., 4., 0.]]))
    assert np.isclose(tree.query(heavy_coordinates(r))[0].min(), 5.)
    print('PASS synthetic: protein heavy atoms and 3D distance geometry')


def run(args):
    self_test()
    data = Path(args.data)
    fasta = data / 'P13569.fasta'
    if not fasta.is_file():
        raise FileNotFoundError(f"Missing UniProt FASTA: {fasta}")
    canonical = str(read_sequence(str(fasta), 'fasta').seq).upper()
    if len(canonical) != EXPECTED_LENGTH or canonical[1031] != 'Y':
        raise ValueError('Unexpected canonical CFTR sequence; require UniProt P13569 length 1480 and Y1032')

    print('=' * 104)
    print('PHASE 27D-3A | EXPERIMENTAL CFTR DRUG-POCKET 3D GEOMETRY AUDIT')
    print('=' * 104)
    print('NO VTI / ETI RESPONSE OUTCOMES READ. Coordinates describe deposited conformations only.')
    print('Mapping uses deposited author residue numbers ONLY IF >=97% match to UniProt P13569.')

    result = pd.DataFrame({'position': np.arange(1, EXPECTED_LENGTH + 1)})
    infos = []
    for structure, ligand, name in SPEC:
        path = data / f'{structure}.pdb'
        if not path.is_file():
            raise FileNotFoundError(f'Missing experimental structure: {path}')
        frame, info = audited_structure(path, ligand, canonical)
        infos.append(info)
        cols = {
            'dmin_A': f'{structure.lower()}_{name}_min_heavy_A',
            'dca_A': f'{structure.lower()}_{name}_ca_A',
            'observed': f'{structure.lower()}_observed',
        }
        result = result.merge(frame.rename(columns=cols), on='position', validate='one_to_one')
        print(f"\n{structure}: {name} ({ligand})")
        print(f"  SHA256: {info['file_sha256']}")
        print(f"  Protein residues mapped: {info['modeled_residues']}/{EXPECTED_LENGTH}")
        print(f"  Canonical identity at author numbering: {info['canonical_agreement']:.2%}")
        print(f"  Canonical mismatches: {info['first_mismatches']}")
        print(f"  Ligand heavy atoms: {info['ligand_heavy_atoms']}")
        print(f"  Residues <5 Å from ligand: {info['protein_contacts_lt5A']}")

    print('\nA. SOURCE / MAPPING QC')
    print(pd.DataFrame(infos).to_string(index=False))

    print('\nB. EXAMPLE FUNCTIONALLY INTERESTING RESIDUES')
    columns = ['position'] + [c for c in result if c != 'position']
    print(result[result.position.isin([68, 74, 508, 1032, 1240, 1303])][columns].round(2).to_string(index=False))

    print('\nC. COVERAGE')
    coverage = [c for c in result if c.endswith('_observed')]
    print('Modeled in both:', int((result[coverage].sum(axis=1) == len(coverage)).sum()), '/', EXPECTED_LENGTH)
    print('Not modeled in either:', int((result[coverage].sum(axis=1) == 0).sum()), '/', EXPECTED_LENGTH)
    print('Unmodeled positions are NaN, NOT zero distance; mutations retain canonical positions.')

    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)
    result.to_csv(dest / 'phase27d3_3d_distances_by_position.csv', index=False)
    pd.DataFrame(infos).to_csv(dest / 'phase27d3_structure_provenance.csv', index=False)
    print('\nPASS: 3D geometry audit completed. No label access, no model fitted.')
    print('Saved:', dest)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--self-test', action='store_true')
    ap.add_argument('--data', default='data/external/cftr_structures')
    ap.add_argument('--out', default='results/biology/phase27d')
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        run(args)
