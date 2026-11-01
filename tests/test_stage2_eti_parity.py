"""Stage 2 real-data parity for Phase26H ETI auxiliary loader; reads TRAIN only.

Run: python -u tests/test_stage2_eti_parity.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import phase26h_auxiliary_corrector as old
from cftr_theratype.datasets.eti_auxiliary import load_data as new_load


def main():
    legacy = old.load_data()
    canonical = new_load(ROOT)
    lv, lx, le, lex, lf = legacy
    nv, nx, ne, nex, nf = canonical
    pd.testing.assert_frame_equal(lv, nv, check_exact=True)
    pd.testing.assert_frame_equal(le, ne, check_exact=True)
    np.testing.assert_array_equal(lx, nx)
    np.testing.assert_array_equal(lex, nex)
    assert len(lf) == len(nf) == 10
    for (r1, f1, tr1, te1), (r2, f2, tr2, te2) in zip(lf, nf):
        assert (r1, f1) == (r2, f2) and tr1 == tr2 and te1 == te2
    assert len(ne) == 383 and ne.protein_position.nunique() == 276
    assert len(set(nv.variant_key) & set(ne.variant_key)) == 66
    print('PASS: exact Phase26H ETI/VTI TRAIN auxiliary loader parity')
    print('383 ETI variants, 276 positions, 66 matched variants; 10 frozen folds')
    print('No original files changed, no VAL/TEST outcomes used')

if __name__ == '__main__':
    main()
