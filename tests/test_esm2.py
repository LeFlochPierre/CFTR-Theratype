"""Tests for mutation-centred ESM sequence windows."""

from cftr_theratype.features.esm2 import mutation_windows
from cftr_theratype.reference import bihler_2024_assay_sequence


def test_m470v_window():
    sequence = bihler_2024_assay_sequence()
    wt, mutant, index = mutation_windows(sequence, 470, "V")

    assert len(wt) == len(mutant) == 801
    assert wt[index] == "M"
    assert mutant[index] == "V"
    assert sum(a != b for a, b in zip(wt, mutant)) == 1


def test_terminal_positions():
    sequence = bihler_2024_assay_sequence()

    for position in (1, 1480):
        original = sequence[position - 1]
        alternate = "A" if original != "A" else "C"

        wt, mutant, index = mutation_windows(
            sequence, position, alternate
        )

        assert len(wt) == len(mutant) == 801
        assert wt[index] == original
        assert mutant[index] == alternate
        assert sum(a != b for a, b in zip(wt, mutant)) == 1
