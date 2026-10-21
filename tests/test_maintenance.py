"""Safety regression tests for the standalone repository maintenance CLI."""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "maintenance.py"


def project(tmp_path):
    (tmp_path / "src/cftr_theratype").mkdir(parents=True)
    (tmp_path / "src/cftr_theratype/data.py").write_text("# must preserve\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='example'\n")
    (tmp_path / "scripts").mkdir()
    shutil.copy2(SOURCE, tmp_path / "scripts/maintenance.py")
    for directory in ("data/raw", "data/processed", "results", "logs"):
        (tmp_path / directory).mkdir(parents=True)
        (tmp_path / directory / "KEEP.txt").write_text("cannot lose this")
    (tmp_path / "README.md").write_text("readme preserve")
    (tmp_path / "README.txt").write_text("maybe useful")
    (tmp_path / "src/cftr_theratype/__pycache__").mkdir()
    (tmp_path / "src/cftr_theratype/__pycache__/data.cpython-311.pyc").write_bytes(b"cache")
    (tmp_path / "src/cftr_theratype.egg-info").mkdir()
    (tmp_path / "src/cftr_theratype.egg-info/PKG-INFO").write_text("generated")
    (tmp_path / "src/cftr_theratype/models").mkdir()
    return tmp_path


def run(root, *args):
    return subprocess.run(
        [sys.executable, "scripts/maintenance.py", *args],
        cwd=root, text=True, capture_output=True, check=True,
    )


def test_default_is_dry_run(tmp_path):
    root = project(tmp_path)
    output = run(root)
    assert "DRY RUN" in output.stdout
    assert (root / "src/cftr_theratype/__pycache__").exists()
    assert (root / "src/cftr_theratype/models").exists()
    assert (root / "results/KEEP.txt").read_text() == "cannot lose this"


def test_apply_preserves_all_research_files(tmp_path):
    root = project(tmp_path)
    output = run(root, "--apply")
    assert "PASS" in output.stdout
    assert not (root / "src/cftr_theratype/__pycache__").exists()
    assert not (root / "src/cftr_theratype.egg-info").exists()
    assert not (root / "src/cftr_theratype/models").exists()
    for path in (
        "data/raw/KEEP.txt", "data/processed/KEEP.txt", "results/KEEP.txt",
        "logs/KEEP.txt", "README.md", "README.txt",
        "src/cftr_theratype/data.py", "scripts/maintenance.py",
    ):
        assert (root / path).is_file(), path


def test_inventory_is_non_destructive(tmp_path):
    root = project(tmp_path)
    run(root, "--inventory")
    inventory = root / "logs/maintenance_inventory.json"
    assert inventory.exists()
    assert (root / "src/cftr_theratype/__pycache__").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="Symlinks need elevated privileges on Windows")
def test_symlink_to_external_data_is_never_deleted(tmp_path):
    root = project(tmp_path)
    external = tmp_path.parent / (tmp_path.name + "-external")
    external.mkdir()
    (external / "KEEP.txt").write_text("external")
    (root / "src/cftr_theratype/uncertainty").symlink_to(external, target_is_directory=True)
    run(root, "--apply")
    assert (external / "KEEP.txt").is_file()
    assert (root / "src/cftr_theratype/uncertainty").is_symlink()
