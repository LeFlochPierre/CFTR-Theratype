import numpy as np
import pandas as pd

from cftr_theratype.transfer import _position_cluster_spearman_ci


def test_position_bootstrap_perfect_monotone_association():
    frame = pd.DataFrame({
        "protein_position": np.arange(1, 21).repeat(2),
        "eti": np.arange(40, dtype=float),
        "vti": np.arange(40, dtype=float) * 2 + 5,
    })
    low, high = _position_cluster_spearman_ci(
        frame, "eti", "vti", seed=42, n_boot=100
    )
    assert np.isclose(low, 1.0)
    assert np.isclose(high, 1.0)
