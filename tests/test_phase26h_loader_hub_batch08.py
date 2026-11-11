"""Synthetic-only gates for eliminating Phase26H as a shared import hub."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cftr_theratype.datasets.eti_auxiliary import load_data
from cftr_theratype.evaluation.metrics import position_mae
from cftr_theratype.models.regressors import phase26d_model
from cftr_theratype.models.auxiliary import SPEC, WEIGHTS
from cftr_theratype.models.responder_classifiers import (
    make_labels, make_model,
)
from cftr_theratype.evaluation.responder_classification import (
    metrics, ap_gain_bootstrap,
)
from cftr_theratype.features.mutation_codes import (
    aa_symbol, mutation_code, position_from_code,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import phase26h_auxiliary_corrector as h  # noqa: E402
import phase27b1_extreme_classifiers as b1  # noqa: E402
import phase26k_trafficking_preflight as k  # noqa: E402


def _numbered_imports(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    return [name for name in imports if name.startswith("phase")]


def test_phase26h_and_later_loaders_no_longer_import_numbered_phase_hubs():
    no_phase_imports = [
        "scripts/phase26h_auxiliary_corrector.py",
        "scripts/phase27b1_extreme_classifiers.py",
        "archive/legacy/scripts/phase27b3_threeclass.py",
        "scripts/phase27d1_biology_ablation.py",
    ]
    for rel in no_phase_imports:
        assert not _numbered_imports(rel), (rel, _numbered_imports(rel))

    others = [
        "scripts/phase27d3b_structure_ablation.py",
        "scripts/phase27e1_mixture_of_experts.py",
        "archive/legacy/scripts/phase28a_trafficking_feature_audit.py",
        "archive/legacy/scripts/phase29b_predictability_ceiling_audit.py",
    ]
    for rel in others:
        assert "phase26h_auxiliary_corrector" not in _numbered_imports(rel)
    assert h.WEIGHTS == WEIGHTS
    assert h.SPEC == SPEC
    assert h.previous.model is phase26d_model
    assert h.previous.pos_mae is position_mae


def test_classifiers_match_original_wrappers_and_prespecified_estimators():
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 1], dtype=int)
    for direction, cutoff in [("low", 10), ("high", 40)]:
        vals = np.array([1, 10, 12, 41, 70, 39, 55, -3, 56, 41])
        np.testing.assert_array_equal(make_labels(vals, direction, cutoff),
                                      b1.make_labels(vals, direction, cutoff))
    for name in ("logistic_seq68", "xgb_seq80"):
        orig = b1.make_model(name, y, 27001)
        canon = make_model(name, y, 27001)
        assert type(orig) is type(canon)
        if name == "logistic_seq68":
            # Pipeline.get_params() includes step *objects*: separately
            # constructed estimators compare unequal by identity.
            assert list(orig.named_steps) == list(canon.named_steps)
            for step in orig.named_steps:
                left = orig.named_steps[step]
                right = canon.named_steps[step]
                assert type(left) is type(right)
                assert left.get_params(deep=True) == right.get_params(deep=True)
            assert orig.get_params(deep=False)["memory"] == canon.get_params(deep=False)["memory"]
        else:
            assert orig.get_params(deep=True) == canon.get_params(deep=True)
        if name == "xgb_seq80":
            assert canon.n_estimators == 140
            assert canon.max_depth == 2
            assert canon.random_state == 27001


def test_metrics_and_position_cluster_ap_bootstrap_are_deterministic():
    rng = np.random.default_rng(27008)
    pos = np.repeat(np.arange(40), 2)
    y = (pos % 3 == 0).astype(int)
    score = y * 0.6 + rng.uniform(0, 0.4, len(y))
    for probability in (False, True):
        old = b1.metrics(y, score, probability)
        new = metrics(y, score, probability)
        assert old.keys() == new.keys()
        for key in old:
            assert (pd.isna(old[key]) and pd.isna(new[key])) or old[key] == new[key]
    df = pd.DataFrame({
        "protein_position": pos,
        "label": y,
        "baseline": score,
        "candidate": score + rng.normal(0, 0.05, len(y)),
    })
    left = b1.ap_gain_bootstrap(df, "baseline", "candidate", 100, 31)
    right = ap_gain_bootstrap(df, "baseline", "candidate", 100, 31)
    assert left == right
    assert left["positions"] == 40


def test_mutation_parser_exact_compatibility_and_embedded_codes():
    for value in (
        "p.Ala74Val", "A74V", "CFTR p.Gly551Asp entry",
        "p.Phe508Leu", "p.Trp1282Ter", "F508del", "M470V", None, "",
    ):
        assert k.mutation_code(value) == mutation_code(value)
    assert mutation_code("CFTR p.Gly551Asp entry") == "G551D"
    assert mutation_code("p.Trp1282Ter") is None
    assert aa_symbol("Gly") == k.aa_symbol("Gly") == "G"
    for value in ("A74V", "W1282R", "F508del", None):
        actual = position_from_code(value)
        expected = k.position_from_code(value)
        assert (np.isnan(actual) and np.isnan(expected)) or actual == expected
