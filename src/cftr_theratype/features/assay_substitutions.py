"""Fixed mutation-only Phase31I.2 teacher representation (61 features).

No response, maturation, or assay-derived features enter the representation.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

AA = 'ACDEFGHIKLMNPQRSTVWY'
# Scalar physicochemical descriptors; fixed a priori, independent of outcomes.
HYDRO = dict(zip(AA, [1.8,2.5,-3.5,-3.5,2.8,-0.4,-3.2,4.5,-3.9,3.8,1.9,-3.5,-1.6,-3.5,-4.5,-0.8,-0.7,4.2,-0.9,-1.3]))
MASS = {'A':89.09,'C':121.15,'D':133.10,'E':147.13,'F':165.19,'G':75.07,
        'H':155.16,'I':131.17,'K':146.19,'L':131.17,'M':149.21,'N':132.12,
        'P':115.13,'Q':146.15,'R':174.20,'S':105.09,'T':119.12,'V':117.15,
        'W':204.23,'Y':181.19}
CHARGE = {a: (-1 if a in 'DE' else (1 if a in 'KR' else 0)) for a in AA}
AROM = {a: int(a in 'FWY') for a in AA}
PROLINE = {a: int(a == 'P') for a in AA}

def mutation_features(aa_codes: pd.Series) -> np.ndarray:
    """Common mutation-only teacher representation for ALL auxiliary/VTI variants.

    No sequence embeddings or assay-derived values are included. Deliberately
    modest features make this a proof-of-concept, not a definitive multi-omics test.
    """
    rows = []
    for value in aa_codes.astype(str):
        m = re.fullmatch(r'([A-Z])(\d{1,4})([A-Z])', value)
        if m is None or m.group(1) not in AA or m.group(3) not in AA:
            raise ValueError(f'Only canonical missense substitutions supported: {value!r}')
        w, pos, a = m.group(1), int(m.group(2)), m.group(3)
        if not 1 <= pos <= 1480 or w == a:
            raise ValueError(f'Invalid mutation: {value!r}')
        wvec = [float(w == x) for x in AA]
        avec = [float(a == x) for x in AA]
        phys = []
        for prop in (HYDRO, MASS, CHARGE, AROM, PROLINE):
            phys.extend([prop[w], prop[a], prop[a] - prop[w]])
        rel = pos / 1480.0
        loc = [rel, rel**2, np.sin(2*np.pi*rel), np.cos(2*np.pi*rel),
               np.sin(6*np.pi*rel), np.cos(6*np.pi*rel)]
        rows.append(wvec + avec + phys + loc)
    return np.asarray(rows, dtype=float)
