"""Regression gates for Phase27D-derived, TRAIN-only dependency migration.

Only synthetic identifiers and metadata are used; no patient outcomes or
locked validation/test partitions are accessed.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cftr_theratype.features import biology_annotations, mutation_codes


ROOT = Path(__file__).resolve().parents[1]


def _numbered_imports(rel: str) -> list[str]:
    file = ROOT / rel
    tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return [name for name in names if name.startswith("phase")]


def test_later_train_only_analyses_do_not_import_numbered_phase_modules():
    for rel in (
        "scripts/phase27d1_biology_ablation.py",
        "scripts/phase27d3b_structure_ablation.py",
        "scripts/phase28a_trafficking_feature_audit.py",
        "scripts/phase29b_predictability_ceiling_audit.py",
    ):
        assert not _numbered_imports(rel), (rel, _numbered_imports(rel))


def test_strict_mutation_source_precedence_and_positions(capsys):
    frame = pd.DataFrame({
        "mutation": ["R74W", "bad label", "G551D"],
        "hgvs_protein": ["p.Arg74Trp", "p.Met470Val", "p.Gly551Asp"],
        "protein_position": [74, 470, 551],
    })
    result = mutation_codes.extract_codes(
        frame, np.array([74, 470, 551]),
    )
    assert result == ["R74W", "M470V", "G551D"]
    assert "Mutation-code source: hgvs_protein" in capsys.readouterr().out

    # A valid code with the wrong position is *not* silently accepted.
    mismatched = frame.drop(columns=["hgvs_protein"])
    with pytest.raises(ValueError, match="Could not parse all missense"):
        mutation_codes.extract_codes(mismatched, np.array([74, 470, 551]))

    with pytest.raises(ValueError, match="Could not parse all missense"):
        mutation_codes.extract_codes(
            pd.DataFrame({"variant_key": ["F508del"]}),
            np.array([508]),
        )


def test_phase27d1_exports_canonical_annotations_and_extractor():
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import phase27d1_biology_ablation as historical

    assert historical.DOMAINS is biology_annotations.DOMAINS
    assert historical.TM_HELICES is biology_annotations.TM_HELICES
    assert historical.CONTACT_SETS is biology_annotations.CONTACT_SETS
    assert historical.HYDROPATHY is biology_annotations.HYDROPATHY
    assert historical.CHARGE is biology_annotations.CHARGE
    assert historical.VOLUME is biology_annotations.VOLUME

    frame = pd.DataFrame({
        "hgvs_protein": ["p.Arg74Trp", "p.Met470Val"],
        "protein_position": [74, 470],
    })
    expected = mutation_codes.extract_codes(frame, np.array([74, 470]))
    observed = historical.extract_codes(frame, np.array([74, 470]))
    assert observed == expected == ["R74W", "M470V"]
    canon = biology_annotations.annotation_features([74, 470], expected)
    hist = historical.annotation_features([74, 470], expected)
    assert canon.keys() == hist.keys()
    for key in canon:
        pd.testing.assert_frame_equal(canon[key], hist[key], check_exact=True)
