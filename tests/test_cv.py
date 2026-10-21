"""Tests for position-grouped cross-validation."""

import numpy as np

from cftr_theratype.cv import position_folds


def test_position_folds_are_disjoint_and_complete():
    positions = np.array([
        100, 100, 200, 200, 300, 400,
        500, 600, 700, 800, 900, 1000,
    ])

    for seed in (42, 43, 44):
        coverage = np.zeros(len(positions), dtype=int)

        for _, train, validation in position_folds(
            positions, seed=seed, n_folds=5
        ):
            assert not set(positions[train]) & set(
                positions[validation]
            )

            coverage[validation] += 1

        np.testing.assert_array_equal(
            coverage, np.ones(len(positions), dtype=int)
        )
