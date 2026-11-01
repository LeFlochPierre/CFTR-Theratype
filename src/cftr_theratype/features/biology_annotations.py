"""Sequence-level biological annotation features, not physical 3D distances.

Maintains Phase27D1 original frozen UniProt intervals and residue proxies.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import re

DOMAINS = {
    'uniprot_TMD1': (81, 365),
    'uniprot_NBD1_core': (423, 646),
    'uniprot_R_region': (654, 831),
    'uniprot_TMD2': (859, 1155),
    'uniprot_NBD2_core': (1210, 1443),
}

TM_HELICES = [
    (78, 98), (123, 146), (196, 216), (223, 243),
    (299, 319), (340, 358), (859, 879), (919, 939),
    (991, 1011), (1014, 1034), (1096, 1116), (1131, 1151),
]

CONTACT_SETS = {
    'typeI_corrector_seq': [68, 74, 195, 198, 364],
    'VX445_corrector_seq': [18, 21, 1098, 1102],
}

HYDROPATHY = dict(zip(
    'ACDEFGHIKLMNPQRSTVWY',
    [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8,
     1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]
))

CHARGE = {aa: (1 if aa in 'KR' else -1 if aa in 'DE' else 0)
          for aa in HYDROPATHY}

VOLUME = dict(zip(
    'ACDEFGHIKLMNPQRSTVWY',
    [88.6, 108.5, 111.1, 138.4, 189.9, 60.1, 153.2, 166.7,
     168.6, 166.7, 162.9, 114.1, 112.7, 143.8, 173.4, 89.0,
     116.1, 140.0, 227.8, 193.6]
))


def annotation_features(positions, codes):
    pp = np.asarray(positions, int)
    codes = list(codes)
    if len(pp) != len(codes):
        raise ValueError('Feature-label length mismatch')
    if (pp < 1).any() or (pp > 1480).any():
        raise ValueError('Protein position outside CFTR P13569 length')

    d = pd.DataFrame(index=np.arange(len(pp)))
    for name, (lo, hi) in DOMAINS.items():
        d[name] = ((pp >= lo) & (pp <= hi)).astype(float)
    d['uniprot_other_region'] = (
        d[list(DOMAINS)].sum(axis=1).to_numpy() == 0
    ).astype(float)

    tm = np.column_stack([
        ((pp >= lo) & (pp <= hi)).astype(float)
        for lo, hi in TM_HELICES
    ])
    d['in_transmembrane_helix'] = tm.max(axis=1)
    d['in_TM_1_to_6'] = tm[:, :6].max(axis=1)
    d['in_TM_7_to_12'] = tm[:, 6:].max(axis=1)
    d['nearest_TM_seq_distance'] = np.min(np.column_stack([
        np.maximum(np.maximum(lo - pp, pp - hi), 0)
        for lo, hi in TM_HELICES
    ]), axis=1).astype(float) / 100.

    pocket = pd.DataFrame(index=d.index)
    for name, residues in CONTACT_SETS.items():
        distances = np.min(
            np.abs(pp[:, None] - np.asarray(residues)[None, :]), axis=1
        )
        pocket[name + '_exp30'] = np.exp(-distances / 30.)
        pocket[name + '_within25'] = (distances <= 25).astype(float)
        pocket[name + '_distance100'] = distances / 100.

    chemistry = pd.DataFrame(index=d.index)
    for i, (pos, code) in enumerate(zip(pp, codes)):
        match = re.fullmatch(r'([A-Z])(\d+)([A-Z])', str(code))
        if match is None:
            raise ValueError(f'Mutation code not a missense substitution: {code}')
        wt, site, mut = match.group(1), int(match.group(2)), match.group(3)
        if site != pos:
            raise ValueError(f'Mutation/position mismatch: {code} != {pos}')
        if wt not in HYDROPATHY or mut not in HYDROPATHY:
            raise ValueError(f'Unknown amino acid in {code}')
        vals = {
            'delta_hydropathy': HYDROPATHY[mut] - HYDROPATHY[wt],
            'delta_charge': CHARGE[mut] - CHARGE[wt],
            'delta_volume': (VOLUME[mut] - VOLUME[wt]) / 100.,
            'abs_delta_volume': abs(VOLUME[mut] - VOLUME[wt]) / 100.,
            'gain_proline': float(mut == 'P' and wt != 'P'),
            'lose_proline': float(wt == 'P' and mut != 'P'),
            'gain_glycine': float(mut == 'G' and wt != 'G'),
            'lose_glycine': float(wt == 'G' and mut != 'G'),
            'gain_cysteine': float(mut == 'C' and wt != 'C'),
            'lose_cysteine': float(wt == 'C' and mut != 'C'),
            'aromatic_change': float((wt in 'FWY') != (mut in 'FWY')),
            'charge_sign_change': float(CHARGE[wt] != CHARGE[mut]),
        }
        for key, value in vals.items():
            chemistry.loc[i, key] = value

    groups = {'domains': d, 'pocket_proxy': pocket, 'chemistry': chemistry}
    groups['all'] = pd.concat([d, pocket, chemistry], axis=1)
    for name, frame in groups.items():
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite engineered features: {name}')
    return groups
