"""Synthetic regressions for Phase31 assay source migration.

Does not load VTI outcomes, frozen clinical datasets, or protected partitions.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from cftr_theratype.datasets import assay_sources, assay_tables


ROOT = Path(__file__).resolve().parents[1]


def test_assay_tables_legacy_parity():
    import sys
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import phase31i_auxiliary_transfer as legacy

    raw = pd.DataFrame({
        "hgvs_protein": ["p.Ala50Val", "A50V", "A51V", "A52V"],
        "protein_position": [50, 50, 51, 52],
        "mutation_type": ["missense"] * 4,
        "study": ["P289", "P289", "P289", "U032"],
        "eti_rescue_pp": [1.0, 3.0, 4.0, 900.0],
    })
    canonical, report = assay_tables.table_aux(raw, "p289", "eti_rescue_pp", "P289")
    historical, old_report = legacy.table_aux(raw, "p289", "eti_rescue_pp", "P289")
    pd.testing.assert_frame_equal(canonical, historical, check_exact=True)
    assert report == old_report
    assert canonical.aa31i.tolist() == ["A50V", "A51V"]
    assert canonical.value31i.tolist() == [2.0, 4.0]
    assert canonical.assay.tolist() == ["p289", "p289"]

    a, b = assay_tables.agreement(canonical, canonical, "p289", "p289")
    c, d = legacy.agreement(historical, historical, "p289", "p289")
    assert a.keys() == c.keys()
    for key in a:
        assert (pd.isna(a[key]) and pd.isna(c[key])) or a[key] == c[key]
    pd.testing.assert_frame_equal(b, d, check_exact=True)
    rng = np.random.default_rng(3113)
    old = legacy.permute_by_position(canonical, np.random.default_rng(3113))
    new = assay_tables.permute_by_position(canonical, rng)
    pd.testing.assert_frame_equal(new, old, check_exact=True)


def test_root_explicit_file_source_manifest(tmp_path):
    (tmp_path / "data/processed").mkdir(parents=True)
    (tmp_path / "results/biology/phase31a").mkdir(parents=True)
    for name in ("bihler_2024_theratyping.parquet",
                 "bihler_2024_missense_splits.parquet",
                 "bihler_2026_transfer_splits.parquet"):
        (tmp_path / "data/processed" / name).write_text("synthetic")
    for name in (assay_tables.S5_NAME, assay_tables.S6_NAME):
        (tmp_path / "results/biology/phase31a" / name).write_text("synthetic")
    args = argparse.Namespace(
        bihler="data/processed/bihler_2024_theratyping.parquet",
        s5=None, s6=None,
        splits2024="data/processed/bihler_2024_missense_splits.parquet",
        vti_meta="data/processed/bihler_2026_transfer_splits.parquet",
    )
    actual = assay_sources.get_sources(args, tmp_path)
    assert set(actual) == {"bihler", "s5", "s6", "splits2024", "vti_meta"}
    assert all(path.is_file() for path in actual.values())
    assert actual["s5"].parent == tmp_path / "results/biology/phase31a"


def test_only_split_metadata_used_for_protected_position_mask(tmp_path, monkeypatch):
    splits2024 = pd.DataFrame({
        "variant_key": ["v50", "v51", "v60", "v62"],
        "hgvs_protein": ["A50V", "A51V", "A60V", "A62V"],
        "protein_position": [50, 51, 60, 62],
        "split": ["train", "train", "val", "test"],
    })
    vti_meta = pd.DataFrame({
        "variant_key": ["v50", "v70", "v72", "v90"],
        "protein_position": [50, 70, 72, 90],
        "transfer_split": ["train", "val", "test", "excluded"],
    })
    paths = {"splits2024": tmp_path / "splits.parquet",
             "vti_meta": tmp_path / "vti_meta.parquet"}
    requested = []

    def read_metadata(path, columns=None, **kwargs):
        assert columns is not None, "Never load all VTI columns"
        requested.append(tuple(columns))
        data = splits2024 if Path(path) == paths["splits2024"] else vti_meta
        return data.loc[:, columns].copy()

    monkeypatch.setattr(pd, "read_parquet", read_metadata)
    eligible, protected = assay_sources.protected_and_eligible_positions(paths)
    assert eligible == {50, 51}
    assert protected == {60, 62, 70, 72}
    assert requested == [
        ("variant_key", "hgvs_protein", "protein_position", "split"),
        ("variant_key", "protein_position", "transfer_split"),
    ]


def test_phase31j_has_no_numbered_module_imports():
    path = ROOT / "scripts/phase31j_error_diagnostics.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.append(node.module or "")
    assert not [name for name in modules if name.startswith("phase")]
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "insert"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "path"
        for node in ast.walk(tree)
    ), "Phase31J should not modify sys.path to load numbered scripts"

def test_teacher_crossfit_and_pilot_estimator_parity():
    import sys
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import phase31i2_repository as legacy_i2
    import phase31i_auxiliary_transfer as legacy_i

    from cftr_theratype.models.assay_teachers import (
        make_pilot_teacher, crossfit_teacher_predictions,
    )

    old_model = legacy_i.model(12.5)
    new_model = make_pilot_teacher(12.5)
    assert list(old_model.named_steps) == list(new_model.named_steps)
    for name in old_model.named_steps:
        assert (
            old_model.named_steps[name].get_params(deep=False)
            == new_model.named_steps[name].get_params(deep=False)
        )
    assert old_model.named_steps["imputer"].get_params()["keep_empty_features"]

    positions = np.arange(50, 80)
    table = pd.DataFrame({
        "aa31i": [f"A{p}V" for p in positions],
        "pos31i2": positions,
        "value31i": np.cos(positions / 13.0),
    })
    train_positions = np.arange(55, 67)
    training = pd.Series([f"A{p}V" for p in train_positions])
    held = pd.Series(["A80V", "A81V"])
    args = (
        table, training, train_positions, held, np.array([80, 81]),
        {80, 81}, 50.0,
    )
    old_oof, old_held = legacy_i2.teacher_predictions(
        *args, negative_seed=14, inner_folds=3,
    )
    new_oof, new_held = crossfit_teacher_predictions(
        *args, negative_seed=14, inner_folds=3,
    )
    np.testing.assert_array_equal(old_oof, new_oof)
    np.testing.assert_array_equal(old_held, new_held)
    assert np.isfinite(old_oof).all() and np.isfinite(old_held).all()
