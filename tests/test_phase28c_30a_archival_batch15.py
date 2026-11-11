"""Batch15 archival regression: exact Phase28C/30A source and runnable CLI.

These are historical TRAIN-only experiment entrypoints, not fully migrated
canonical experiment runners. The underlying source is archived unchanged.
Synthetic self-tests require no protected assays, external PDBs, or results.
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
RUNNER = ROOT / "scripts/run_legacy.py"
ARCHIVED = {
    "phase28c_structure_biophysics.py": (
        "d3dd7a47515fae5a4506d92c4d48c192f15d16f8",
        "PASS synthetic: contact geometry, missing-coverage masks",
    ),
    "phase30a_target_information_ladder.py": (
        "45600ac61ba245de92191d8a9a4f099c497c647d",
        "PASS synthetic: target arithmetic, leakage-resistant fold inputs",
    ),
}


def git_blob_sha1(contents: bytes) -> str:
    header = f"blob {len(contents)}".encode("ascii") + bytes([0])
    return hashlib.sha1(header + contents).hexdigest()


@pytest.mark.parametrize("filename", sorted(ARCHIVED))
def test_archived_source_identical_and_active_file_absent(filename):
    original_blob, _ = ARCHIVED[filename]
    archived = ROOT / "archive/legacy/scripts" / filename
    assert archived.is_file()
    assert not (ROOT / "scripts" / filename).exists()
    contents = archived.read_bytes()
    assert git_blob_sha1(contents) == original_blob
    ast.parse(contents.decode("utf-8"), filename=str(archived))


@pytest.mark.parametrize("filename", sorted(ARCHIVED))
def test_archived_cli_synthetic_self_test(filename):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    run = subprocess.run(
        [sys.executable, str(RUNNER), filename, "--self-test"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stdout
    assert ARCHIVED[filename][1] in run.stdout
