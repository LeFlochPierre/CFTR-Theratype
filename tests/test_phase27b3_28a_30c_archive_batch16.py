"""Batch16 archival integrity and dependency-preserving synthetic CLI checks.

Original scientific scripts are preserved byte-for-byte. These tests never
open protected VTI/ETI outcomes or regenerate frozen scientific results.
"""
from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive/legacy/scripts"
RUNNER = ROOT / "scripts/run_legacy.py"

# Git blob hashes of exact original main-branch source, recorded before move.
CASES = {
    "phase27b3_threeclass.py": (
        "afcec504250456e3458f931831e3333f436de313",
        "PASS synthetic: labels, probabilities, ordinal consistency, metrics",
    ),
    "phase28a_trafficking_feature_audit.py": (
        "4e3e1ae3603691ad26c100495d14335c35e0ee2d",
        "PASS synthetic: HGVS substitution keys, exclusions",
    ),
    "phase30c_eti_incremental_ladder.py": (
        "0f1a2d0d345fe1b40bf2cd84a2b1462e6364bced",
        "PASS Phase30C synthetic tests",
    ),
}


def blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}".encode("ascii") + bytes([0])
    return hashlib.sha1(header + content).hexdigest()


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT / "src") + os.pathsep + str(ARCHIVE) +
        (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    return env


def _run(*args):
    return subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


@pytest.mark.parametrize("filename", sorted(CASES))
def test_source_byte_identical_and_no_active_duplicate(filename):
    fullpath = ARCHIVE / filename
    assert fullpath.is_file()
    assert not (ROOT / "scripts" / filename).exists()
    raw = fullpath.read_bytes()
    assert blob_sha1(raw) == CASES[filename][0]
    ast.parse(raw.decode("utf-8"), filename=str(fullpath))


@pytest.mark.parametrize("filename", sorted(CASES))
def test_original_experiment_self_test_still_runs(filename):
    run = _run(str(RUNNER), filename, "--self-test")
    assert run.returncode == 0, run.stdout + run.stderr
    assert CASES[filename][1] in run.stdout


def test_previous_phase27b3_consumer_import_survives_archive():
    # The unmodified archived Phase27B3B imports the historical evaluate API.
    consumer = (ARCHIVE / "phase27b3b_specialist_combination.py").read_text()
    assert "from phase27b3_threeclass import evaluate" in consumer
    run = _run(
        "-c",
        "from phase27b3_threeclass import evaluate; assert callable(evaluate)",
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_previous_phase28a_consumer_self_test_survives_archive():
    # Phase28B imports Phase28A from its archived directory at startup;
    # run only synthetic self-test, not the real workbook/audit pipeline.
    consumer = (ARCHIVE / "phase28b_multi_source_biology_audit.py").read_text()
    assert "from phase28a_trafficking_feature_audit import parse_mutation, load_training" in consumer
    run = _run(str(RUNNER), "phase28b_multi_source_biology_audit.py", "--self-test")
    assert run.returncode == 0, run.stdout + run.stderr
    assert "PASS synthetic: mutation matching" in run.stdout
