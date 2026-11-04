"""Opt-in real-data parity check against canonical Phase26D TRAIN loader.

This is a genuine on-disk data integration test, not a synthetic test. It must
not access protected validation/test outcomes or run during pytest collection.
"""
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import phase26d_corrector_bottleneck as legacy
from cftr_theratype.datasets.frozen_vti import load_train as new_load


def test_frozen_loader_parity():
    """Run on workstations with the ignored VTI TRAIN dataset; skip otherwise."""
    required = ROOT / "data/processed/bihler_2026_vti.parquet"
    if not required.is_file():
        message = (
            f"Real-data Phase26D parity requires local ignored dataset: {required}. "
            "Run this test in a checkout with the protected TRAIN-only inputs available."
        )
        if os.getenv("CFTR_REQUIRE_REAL_DATA") == "1":
            pytest.fail(message, pytrace=False)
        pytest.skip(message)

    # Load only the TRAIN cohort through the loader's existing filters.
    old = legacy.load_train()
    new = new_load(ROOT)
    a, ax, ah, af, ar = old
    b, bx, bh, bf, br = new

    pd.testing.assert_frame_equal(a, b, check_exact=True)
    pd.testing.assert_frame_equal(ah, bh, check_exact=True)
    for name in ("sequence", "sequence_structure"):
        np.testing.assert_array_equal(ax[name], bx[name])
    assert ar == br and len(af) == len(bf) == 10
    for (r1, f1, tr1, te1), (r2, f2, tr2, te2) in zip(af, bf):
        assert (r1, f1) == (r2, f2) and tr1 == tr2 and te1 == te2

    print("PASS: exact Phase26D parity — 210 variants, 168 positions, 10 frozen folds")
    print("Legacy and canonical dataframes, features, historical OOF and splits match.")
