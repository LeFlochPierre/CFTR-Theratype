"""Exact-source and historical CLI smoke checks for Batch14 archival.

Two exploratory TRAIN-only leaf experiment entry points were moved from
scripts/ to archive/legacy/scripts/ with their source bytes unchanged.
No protected assays, historical results, or clinical files are used.
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
LEGACY = ROOT / "archive/legacy/scripts"
RUNNER = ROOT / "scripts/run_legacy.py"

# Original Git blob IDs, recorded from main before the move. The Git blob
# identity commits to every byte, independent of file path and line endings.
ARCHIVED = {
    "phase28e5_foldx_predictive_ablation.py":
        ("8b54f2c30bb6c947802647ae4d831eee4717256a",
         "PASS synthetic: FoldX manifest"),
    "phase29b_predictability_ceiling_audit.py":
        ("14f026bc88a0703213b9f02d7b30ba70dccd8cf6",
         "PASS synthetic: position baselines"),
}


def _git_blob_id(data: bytes) -> str:
    header = f"blob {len(data)}".encode("utf-8") + bytes([0])
    return hashlib.sha1(header + data).hexdigest()


@pytest.mark.parametrize("name", sorted(ARCHIVED))
def test_original_source_bytes_preserved_and_active_copy_removed(name):
    path = LEGACY / name
    assert path.is_file()
    assert not (ROOT / "scripts" / name).exists()
    assert _git_blob_id(path.read_bytes()) == ARCHIVED[name][0]
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


@pytest.mark.parametrize("name", sorted(ARCHIVED))
def test_historical_self_test_still_runs_via_legacy_runner(name):
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(ROOT / "src")
        + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    )
    proc = subprocess.run(
        [sys.executable, str(RUNNER), name, "--self-test"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=100,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout
    assert ARCHIVED[name][1] in proc.stdout
