"""Exploratory sequence-position regions from Phase 31J (not validated CFTR domain boundaries).

The default intervals were used for retrospective diagnostic strata. The
optional CSV allows curated intervals; this module makes no clinical claims.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# APPROXIMATE exploratory structural regions, not a validated domain annotation.
# Optional --domain-map accepts source-curated intervals to replace these.
APPROX_REGIONS = [
    (1, 79, 'N-terminal'),
    (80, 380, 'MSD1~'),
    (381, 630, 'NBD1~'),
    (631, 849, 'R-domain~'),
    (850, 1176, 'MSD2~'),
    (1177, 1206, 'linker~'),
    (1207, 1440, 'NBD2~'),
    (1441, 1480, 'C-terminal'),
]


def domains_from_file(path):
    if path is None:
        segments = APPROX_REGIONS
    else:
        tab = pd.read_csv(path)
        needed={'start','end','domain'}
        if not needed.issubset(tab):
            raise ValueError(f'--domain-map requires columns {needed}')
        segments = [(int(r.start),int(r.end),str(r.domain)) for r in tab.itertuples(index=False)]
    if not segments:
        raise ValueError('Empty domain map')
    check = np.zeros(1481,dtype=np.int8)
    for lo,hi,name in segments:
        if not 1<=lo<=hi<=1480 or not name or (check[lo:hi+1]>0).any():
            raise ValueError('Domain intervals overlap, contain an invalid range or have empty name')
        check[lo:hi+1] = 1
    return segments


def domain_of(p, segments):
    for lo,hi,name in segments:
        if lo <= int(p) <= hi:
            return name
    return 'Unannotated'


