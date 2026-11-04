"""Synthetic Phase26G/26K-2 canonical loader and parser regression gates.

Never reads the protected datasets. Parquet access is monkeypatched to
exercise TRAIN predicate filters and VTI protected-position exclusions.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cftr_theratype.datasets.phenotype_training import load_allowed
from cftr_theratype.features.mutation_codes import mutation_code, position_from_code
from cftr_theratype.features.phenotype_values import parse_value


ROOT = Path(__file__).resolve().parents[1]


def _phase_imports(path):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"), filename=path)
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return [name for name in names if name.startswith("phase")]


def test_phase26g_k2_and_e1_do_not_import_numbered_scripts():
    paths = (
        "scripts/phase26g_feature_model_ablation.py",
        "scripts/phase26k2_numeric_audit.py",
        "scripts/phase27e1_mixture_of_experts.py",
    )
    for rel in paths:
        assert not _phase_imports(rel), rel
        s = (ROOT / rel).read_text(encoding="utf-8")
        assert "sys.path.insert" not in s
    assert "from cftr_theratype.datasets.frozen_vti import load_train" in (
        ROOT / paths[0]
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("value,expected,kind", [
    (5.2, 5.2, "numeric"),
    (np.float64(1.0), 1.0, "numeric"),
    ("2.5", 2.5, "numeric_string"),
    ("12.1 ± 1.4", 12.1, "mean_plusminus"),
    ("5.2 +/- 0.3", 5.2, "mean_plusminus"),
    ("1.2e2", 120.0, "numeric_string"),
    ("  7.25 % ", 7.25, "numeric_string"),
    ("−5.2", -5.2, "numeric_string"),
    ("1.5, 2.6", np.nan, "UNRECOGNISED"),
    ("5.2 (3.1)", np.nan, "UNRECOGNISED"),
    ("=AVERAGE(A1:A3)", np.nan, "UNRECOGNISED"),
    ("ND", np.nan, "missing"),
    (None, np.nan, "missing"),
])
def test_numeric_formats_preserve_original_semantics(value, expected, kind):
    actual, label = parse_value(value)
    assert label == kind
    if np.isnan(expected):
        assert np.isnan(actual)
    else:
        assert actual == expected


def _fake_data():
    vkeys = [f"V{i:03d}" for i in range(210)]
    ekeys = [f"E{i:03d}" for i in range(422)]
    protected = list(range(1000, 1068))
    split = pd.DataFrame({
        "variant_key": vkeys + [f"P{i:03d}" for i in range(68)],
        "transfer_split": ["train"] * 210 + ["val"] * 34 + ["test"] * 34,
        "protein_position": list(range(1, 211)) + protected,
        "hgvs_protein": [f"A{i}V" for i in range(1, 211)] +
                        [f"A{i}V" for i in protected],
    })
    old = pd.DataFrame({"variant_key": ekeys, "split": ["train"] * 422})
    v = pd.DataFrame({
        "variant_key": vkeys + ["P000"],
        "hgvs_protein": [f"A{i}V" for i in range(1, 211)] + ["A1000V"],
        "corrector_rescue_pp": [float(i % 50) for i in range(210)] + [99.],
        "vti_rescue_pp": [float(i % 50 + 2) for i in range(210)] + [99.],
    })
    ep = list(range(1, 384)) + list(range(1000, 1039))
    e = pd.DataFrame({
        "variant_key": ekeys + ["not_train"],
        "hgvs_protein": [f"G{i}D" for i in ep] + ["G1400D"],
        "corrector_rescue_pp": [float(i % 60) for i in range(422)] + [99.],
    })
    return {
        "bihler_2026_transfer_splits.parquet": split,
        "bihler_2024_missense_splits.parquet": old,
        "bihler_2026_vti.parquet": v,
        "bihler_2024_theratyping.parquet": e,
    }


def test_train_only_loader_filters_before_loading_labels_and_excludes_protected(monkeypatch, tmp_path):
    sources = _fake_data()
    calls = []

    def fake_parquet(path, columns=None, filters=None):
        key = Path(path).name
        calls.append((key, columns, filters))
        assert key in sources, key
        frame = sources[key].copy()
        if key in ("bihler_2026_vti.parquet", "bihler_2024_theratyping.parquet"):
            assert filters is not None and len(filters) == 1
            field, op, ids = filters[0]
            assert (field, op) == ("variant_key", "in")
            assert len(ids) == (210 if key == "bihler_2026_vti.parquet" else 422)
            assert set(ids) == set(frame.variant_key) - (
                {"P000"} if key == "bihler_2026_vti.parquet" else {"not_train"}
            )
            frame = frame.loc[frame.variant_key.isin(ids)]
        assert columns is not None
        return frame.loc[:, columns].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_parquet)
    v, e, protected = load_allowed(tmp_path)
    assert len(v) == 210 and len(e) == 383
    assert len(protected) == 68 and min(protected) == 1000
    assert "A1000V" not in set(v.mutation)
    assert "G1000D" not in set(e.mutation)
    assert "G1400D" not in set(e.mutation)
    assert list(v.columns) == ["mutation", "vti_corrector", "vti_total"]
    assert list(e.columns) == ["mutation", "eti_corrector"]
    assert [item[0] for item in calls] == [
        "bihler_2026_transfer_splits.parquet",
        "bihler_2024_missense_splits.parquet",
        "bihler_2026_vti.parquet",
        "bihler_2024_theratyping.parquet",
    ]

    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import phase26k_trafficking_preflight as historical
    import phase26k2_numeric_audit as historical_k2

    monkeypatch.setattr(historical, "ROOT", tmp_path)
    monkeypatch.setattr(historical_k2, "ROOT", tmp_path)
    old_v, old_e, old_protected = historical.load_allowed()
    k2_v, k2_e, k2_protected = historical_k2.load_allowed()
    pd.testing.assert_frame_equal(v, old_v, check_exact=True)
    pd.testing.assert_frame_equal(e, old_e, check_exact=True)
    pd.testing.assert_frame_equal(v, k2_v, check_exact=True)
    pd.testing.assert_frame_equal(e, k2_e, check_exact=True)
    assert protected == old_protected == k2_protected

    for value in ["p.Ala74Val", "A74V", "p.Trp1282Ter", None]:
        assert historical.mutation_code(value) == mutation_code(value)
    assert historical_k2.position_from_code("G551D") == position_from_code("G551D")
    assert historical_k2.parse_value("3.2 +/- 1.4") == parse_value("3.2 +/- 1.4")


def test_unparseable_vti_train_mutation_fails_closed(monkeypatch, tmp_path):
    sources = _fake_data()
    sources["bihler_2026_vti.parquet"].loc[0, "hgvs_protein"] = "F508del"

    def fake_parquet(path, columns=None, filters=None):
        frame = sources[Path(path).name]
        if filters:
            frame = frame.loc[frame.variant_key.isin(filters[0][2])]
        return frame.loc[:, columns].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_parquet)
    with pytest.raises(ValueError, match="Unparseable training mutation"):
        load_allowed(tmp_path)
