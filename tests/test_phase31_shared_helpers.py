"""Regression checks for extracted Phase31J helpers and Phase31L imports.

Synthetic-only: no clinical cohorts, protected outcomes, or RTX6 result files.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase31j_error_diagnostics as legacy
from cftr_theratype.evaluation.historical_oof import load_m0, assert_oof_valid
from cftr_theratype.features.regions import APPROX_REGIONS, domains_from_file, domain_of


def test_region_defaults_and_legacy_api(tmp_path):
    assert legacy.domains_from_file is domains_from_file
    assert legacy.domain_of is domain_of
    assert legacy.APPROX_REGIONS is APPROX_REGIONS
    segments = domains_from_file(None)
    assert domain_of(1, segments) == "N-terminal"
    assert domain_of(450, segments) == "NBD1~"
    assert domain_of(1480, segments) == "C-terminal"
    assert domain_of(1481, segments) == "Unannotated"

    good = tmp_path / "curated.csv"
    good.write_text("start,end,domain\n1,10,A\n11,20,B\n")
    assert domain_of(15, domains_from_file(good)) == "B"

    bad = tmp_path / "overlap.csv"
    bad.write_text("start,end,domain\n1,10,A\n10,20,B\n")
    with pytest.raises(ValueError, match="overlap"):
        domains_from_file(bad)


def synthetic_historical_files(root):
    ids = [f"variant_{i}" for i in range(10)]
    d = pd.DataFrame({
        "variant_key": ids,
        "protein_position": list(range(101, 111)),
        "corrector_rescue_pp": np.linspace(3, 12, 10),
        "vti_rescue_pp": np.linspace(13, 22, 10),
    })
    phase_dir = root / "results" / "biology"
    (phase_dir / "phase26g").mkdir(parents=True)
    (phase_dir / "phase31i").mkdir(parents=True)
    shared = []
    for target, column in (("corrector", "corrector_rescue_pp"), ("total", "vti_rescue_pp")):
        rows = []
        for repeat in (1, 2):
            for index, row in d.iterrows():
                rows.append({
                    "repeat": repeat,
                    "fold": index % 5 + 1,
                    "variant_key": row.variant_key,
                    "protein_position": row.protein_position,
                    "observed": row[column],
                    "xgb_80": row[column] + 0.3,
                })
        frame = pd.DataFrame(rows)
        frame.to_csv(phase_dir / "phase26g" / f"phase26g_{target}_oof.csv", index=False)
        a = frame.rename(columns={"xgb_80": "prediction"}).copy()
        a["target"] = target
        a["model"] = "M0"
        shared.append(a)
    pd.concat(shared, ignore_index=True).to_csv(
        phase_dir / "phase31i" / "phase31i2_repo_oof.csv", index=False
    )
    return d


def test_historical_parity_loader_matches_legacy_wrapper(tmp_path, monkeypatch):
    d = synthetic_historical_files(tmp_path)
    actual, hashes = load_m0(d, root=tmp_path)
    assert len(actual) == 40
    assert set(actual.target) == {"corrector", "total"}
    assert len(hashes) == 3

    monkeypatch.setattr(legacy, "ROOT", tmp_path)
    historical, legacy_hashes = legacy.load_m0(d)
    pd.testing.assert_frame_equal(actual, historical, check_exact=True)
    assert hashes == legacy_hashes


def test_historical_loader_rejects_corrupted_observation(tmp_path):
    d = synthetic_historical_files(tmp_path)
    path = tmp_path / "results/biology/phase26g/phase26g_total_oof.csv"
    values = pd.read_csv(path)
    values.loc[0, "observed"] += 100
    values.to_csv(path, index=False)
    with pytest.raises(AssertionError, match="unexpected observed TRAIN target"):
        load_m0(d, root=tmp_path)


def test_direct_oof_validation_rejects_position_mismatch():
    d = pd.DataFrame({
        "variant_key": ["v1", "v2"],
        "protein_position": [1, 2],
        "corrector_rescue_pp": [1.0, 2.0],
        "vti_rescue_pp": [3.0, 4.0],
    })
    frame = pd.DataFrame({
        "repeat": [1, 1],
        "fold": [1, 2],
        "variant_key": ["v1", "v2"],
        "protein_position": [5, 2],
        "observed": [1.0, 2.0],
        "prediction": [1.1, 2.2],
    })
    with pytest.raises(AssertionError, match="position mismatch"):
        assert_oof_valid(frame, d, "corrector")
