"""Validation of Phase27D-3A CFTR structural-provenance files.

The PDB codes and distance feature field names are fixed historical research
parameters. Validation checks canonical sequence agreement, the documented
position-1371 mismatch and SHA256 identity of the original PDB structures.
Only local supplied provenance and PDB files are read; no downloads or
protected assay outcomes are accessed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


SPECS = {
    '8EIG': {
        'dmin': '8eig_elexacaftor_VX445_min_heavy_A',
        'dca': '8eig_elexacaftor_VX445_ca_A',
        'obs': '8eig_observed',
    },
    '7SV7': {
        'dmin': '7sv7_tezacaftor_VX661_min_heavy_A',
        'dca': '7sv7_tezacaftor_VX661_ca_A',
        'obs': '7sv7_observed',
    }
}


def verify_provenance(provenance_path, root, specs=SPECS):
    provenance_path = Path(provenance_path)
    root = Path(root)
    if not provenance_path.is_file():
        raise FileNotFoundError(f'Missing Phase 27D-3A structure provenance: {provenance_path}')
    info = pd.read_csv(provenance_path)
    if info.pdb.duplicated().any() or set(info.pdb) != set(specs):
        raise ValueError('Unexpected PDB provenance')
    for row in info.itertuples(index=False):
        if row.canonical_agreement < .99:
            raise ValueError(f'Insufficient sequence agreement for {row.pdb}')
        if row.mismatches_canonical != 1 or '(1371,' not in str(row.first_mismatches):
            raise ValueError(f'Unexpected mismatch set for {row.pdb}; inspect mapping')
        path = root / 'data/external/cftr_structures' / f'{row.pdb}.pdb'
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row.file_sha256:
            raise ValueError(f'Structure file missing or changed since Phase 27D-3A: {path}')
    return info
