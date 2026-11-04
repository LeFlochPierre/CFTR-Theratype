"""Structural safeguards for Phase26 hub deduplication.

These checks are data-free. Full prediction parity is covered by the existing
Stage2/Stage3/Stage6 suites and must also be confirmed on RTX6.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Reusable logic moved to these existing canonical modules, with old signatures
# kept for backwards compatibility with archived experiments and tests.
EXPECTED = {
    "scripts/phase26d_corrector_bottleneck.py": {
        "pos_mae": "cftr_theratype.evaluation.metrics",
        "model": "cftr_theratype.models.regressors",
        "select_spec": "cftr_theratype.models.regressors",
        "folds_from_historical_oof": "cftr_theratype.evaluation.folds",
        "load_train": "cftr_theratype.datasets.frozen_vti",
    },
    "scripts/phase26g_feature_model_ablation.py": {
        "make_model": "cftr_theratype.models.regressors",
        "position_mae": "cftr_theratype.evaluation.metrics",
        "choose": "cftr_theratype.models.regressors",
        "paired_ci": "cftr_theratype.evaluation.metrics",
    },
    "scripts/phase26h_auxiliary_corrector.py": {
        "parsed_position": "cftr_theratype.datasets.eti_auxiliary",
        "load_data": "cftr_theratype.datasets.eti_auxiliary",
        "positional_shuffle": "cftr_theratype.models.auxiliary",
        "fit_joint": "cftr_theratype.models.auxiliary",
        "choose_weight": "cftr_theratype.models.auxiliary",
    },
    "scripts/phase27d1_biology_ablation.py": {
        "annotation_features": "cftr_theratype.features.biology_annotations",
    },
}


def test_all_15_legacy_helpers_delegate_to_canonical_package():
    n = 0
    for rel, funcs in EXPECT.items():
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        functions = {
            x.name: x for x in tree.body
            if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name, expected_module in funcs.items():
            assert name in functions, (rel, name)
            func = functions[name]
            # Wrappers stay small and preserve the public function name.
            assert len(func.body) == 2, (rel, name)
            assert isinstance(func.body[0], ast.ImportFrom), (rel, name)
            assert func.body[0].module == expected_module, (rel, name)
            assert isinstance(func.body[1], ast.Return), (rel, name)
            assert isinstance(func.body[1].value, ast.Call), (rel, name)
            n += 1
    assert n == 15


def test_migrated_late_scripts_do_not_import_numbered_phase26_modules():
    for rel in (
        "scripts/phase31l_predictability.py",
        "scripts/phase31m_three_tracks.py",
        "scripts/phase32a_adaptive_theratyping.py",
    ):
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not a.name.startswith(("phase26d_", "phase26g_"))
                    for a in node.names
                ), rel
            elif isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(
                    ("phase26d_", "phase26g_")
                ), rel


def test_canonical_phase26_entrypoints_exist():
    for rel, names in {
        "src/cftr_theratype/datasets/frozen_vti.py": {"load_train"},
        "src/cftr_theratype/datasets/eti_auxiliary.py": {"load_data", "parsed_position"},
        "src/cftr_theratype/models/regressors.py": {
            "phase26d_model", "phase26d_select_spec",
            "phase26g_model", "phase26g_choose",
        },
        "src/cftr_theratype/models/auxiliary.py": {
            "positional_shuffle", "fit_joint", "choose_weight",
        },
        "src/cftr_theratype/evaluation/metrics.py": {"position_mae", "paired_ci"},
        "src/cftr_theratype/evaluation/folds.py": {"from_historical_oof"},
        "src/cftr_theratype/features/biology_annotations.py": {"annotation_features"},
    }.items():
        tree = ast.parse((ROOT / rel).read_text(encoding="utf-8"), filename=rel)
        present = {
            x.name for x in tree.body
            if isinstance(x, ast.FunctionDef)
        }
        assert names <= present, (rel, names - present)
