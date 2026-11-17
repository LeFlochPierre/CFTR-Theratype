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

import numpy as np
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


def load_geometry(geometry_path):
    if not geometry_path.is_file():
        raise FileNotFoundError(f'Missing Phase 27D-3A geometry: {geometry_path}')
    raw = pd.read_csv(geometry_path)
    if len(raw) != 1480 or raw.position.duplicated().any() or set(raw.position) != set(range(1, 1481)):
        raise ValueError('Expected one distance row for all 1480 positions')
    for pdb, names in SPECS.items():
        obs = raw[names['obs']].to_numpy(int)
        heavy = raw[names['dmin']].to_numpy(float)
        ca = raw[names['dca']].to_numpy(float)
        if not np.isin(obs, [0, 1]).all():
            raise ValueError(f'Non-binary coverage flags: {pdb}')
        if ((obs == 1) & (~np.isfinite(heavy) | (heavy < 0))).any():
            raise ValueError(f'Invalid heavy distances at modelled sites: {pdb}')
        if ((obs == 1) & (~np.isfinite(ca) | (ca < 0))).any():
            raise ValueError(f'Invalid CA distances at modelled sites: {pdb}')
        if ((obs == 0) & np.isfinite(heavy)).any():
            raise ValueError(f'Unmodelled residues have numeric distances: {pdb}')
    return raw.set_index('position')


def feature_sets(geometry, positions, seq_proxy, chemistry):
    """Missing distance gets zero transformed proximity plus observed=0 flag.

    Zero is not treated as a zero-Angstrom distance. Structural mismatch
    residue 1371 is withheld from BOTH structures' geometry features.
    """
    p = np.asarray(positions, int)
    if (p < 1).any() or (p > 1480).any():
        raise ValueError('Invalid canonical position')
    indexed = geometry.loc[p]
    arrays = {'heavy': [], 'ca': [], 'contacts': []}
    coverage = []
    for pdb, names in SPECS.items():
        obs = indexed[names['obs']].to_numpy(int).astype(bool) & (p != 1371)
        heavy = indexed[names['dmin']].to_numpy(float)
        ca = indexed[names['dca']].to_numpy(float)
        if ((obs) & (~np.isfinite(heavy) | ~np.isfinite(ca))).any():
            raise ValueError(f'Inconsistent distance/coverage: {pdb}')
        # 0 transformed proximity means far OR missing, disambiguated by flag.
        near_heavy = np.where(obs, np.exp(-np.nan_to_num(heavy, nan=1e5) / 10.0), 0.0)
        near_ca = np.where(obs, np.exp(-np.nan_to_num(ca, nan=1e5) / 10.0), 0.0)
        arrays['heavy'].append(near_heavy)
        arrays['ca'].append(near_ca)
        for threshold in (5., 10., 20.):
            arrays['contacts'].append((obs & (heavy < threshold)).astype(float))
        arrays['heavy'].append(obs.astype(float))
        arrays['ca'].append(obs.astype(float))
        arrays['contacts'].append(obs.astype(float))
        coverage.append(obs.astype(float))
    out = {
        '3d_heavy': np.column_stack(arrays['heavy']),
        '3d_ca': np.column_stack(arrays['ca']),
        '3d_contacts': np.column_stack(arrays['contacts']),
    }
    out['seq_plus_3d'] = np.column_stack([seq_proxy, out['3d_heavy']])
    out['chem_plus_3d'] = np.column_stack([chemistry, out['3d_heavy']])
    for name, value in out.items():
        if not np.isfinite(value).all():
            raise ValueError(f'Nonfinite encoded 3D features: {name}')
    return out, np.column_stack(coverage)
