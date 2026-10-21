#!/usr/bin/env python3
"""Safe, dependency-free maintenance for the CFTR-Theratype repository.

Default: preview only. --apply removes ONLY reproducible Python build caches
and scaffold directories that are truly empty. --inventory writes a machine-
readable local file index under ignored logs/. Never touches research data,
results, logs, manifests, notebooks with content, or Git history.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "cftr_theratype"

EMPTY_SCAFFOLDS = (
    "configs", "notebooks", "scripts",
    "src/cftr_theratype/active_learning",
    "src/cftr_theratype/harmonise",
    "src/cftr_theratype/models",
    "src/cftr_theratype/splits",
    "src/cftr_theratype/uncertainty",
)

WATCHED_SOURCES = (
    "data/raw/bihler_2024/mmc3.xlsx",
    "data/raw/bihler_2026/mmc1.xlsx",
    "data/raw/mcdonald_2024/table_s3.xlsx",
    "data/raw/uniprot/P13569.fasta",
)
WATCHED_OUTPUTS = (
    "data/processed/bihler_2024_theratyping.parquet",
    "data/processed/bihler_2024_missense_splits.parquet",
    "data/processed/bihler_2026_vti.parquet",
    "data/processed/bihler_2026_cross_study_membership.parquet",
    "data/processed/bihler_2026_transfer_splits.parquet",
    "data/manifests/data_sources.json",
    "data/manifests/split_protocol.json",
    "data/manifests/cross_study_protocol.json",
    "data/manifests/vti_transfer_protocol.json",
    "data/interim/esm2_35m_development.npz",
    "results/baselines/cv_summary.json",
    "results/plm/summary.json",
    "results/transfer/reliability_summary.json",
    "results/transfer/training_overlap_diagnostics.json",
    "results/transfer/corrector_cv_summary.json",
)


def guarded(path: Path) -> Path:
    """Prevent deletion outside this repository or through symlinks."""
    if path.is_symlink():
        raise ValueError(f"Refusing to touch symlink: {path}")
    target = path.resolve()
    if target == ROOT or not target.is_relative_to(ROOT):
        raise ValueError(f"Unsafe path: {path}")
    return target


def check_project() -> None:
    if not (ROOT / "pyproject.toml").is_file():
        raise RuntimeError("Missing pyproject.toml; wrong project location")
    if not (PACKAGE / "data.py").is_file():
        raise RuntimeError("Missing cftr_theratype/data.py; wrong project location")


def candidates() -> tuple[list[Path], list[Path], list[Path]]:
    generated_dirs = []
    compiled_files = []
    roots = [ROOT / "src", ROOT / "tests", ROOT / "scripts"]
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_symlink():
                continue
            if path.is_dir() and (
                path.name in {"__pycache__", ".pytest_cache"}
                or path.name.endswith(".egg-info")
            ):
                generated_dirs.append(guarded(path))
            elif path.is_file() and path.suffix == ".pyc":
                compiled_files.append(guarded(path))

    # Python creates these at the project root too.
    for path in (ROOT / ".pytest_cache", ROOT / ".mypy_cache", ROOT / ".ruff_cache"):
        if path.exists() and not path.is_symlink() and path.is_dir():
            generated_dirs.append(guarded(path))

    generated_dirs = sorted(set(generated_dirs), key=lambda p: (len(p.parts), str(p)))
    # No need to traverse nested generated dirs individually.
    top_dirs = []
    for path in generated_dirs:
        if not any(path.is_relative_to(parent) for parent in top_dirs):
            top_dirs.append(path)
    compiled_files = sorted({path for path in compiled_files if not any(
        path.is_relative_to(parent) for parent in top_dirs
    )})

    removable = set(top_dirs) | set(compiled_files)
    scaffold_dirs = []
    for relative in EMPTY_SCAFFOLDS:
        path = ROOT / relative
        if not path.is_dir() or path.is_symlink():
            continue
        path = guarded(path)
        # Only remove if presently empty, or if every direct child is
        # one of the generated paths that this very run will remove.
        children = list(path.iterdir())
        if all(child in removable for child in children):
            scaffold_dirs.append(path)
    return top_dirs, compiled_files, scaffold_dirs


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def inventory() -> dict:
    def describe(name: str, with_checksum: bool = False) -> dict:
        path = ROOT / name
        item = {"path": name, "exists": path.is_file()}
        if path.is_file():
            item["bytes"] = path.stat().st_size
            if with_checksum:
                item["sha256"] = digest(path)
        return item

    git_status = None
    try:
        proc = subprocess.run(
            ["git", "status", "--short"], cwd=ROOT,
            capture_output=True, text=True, check=False, timeout=10,
        )
        git_status = proc.stdout.strip().splitlines() if proc.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Local file inventory. NOT a scientific results recalculation.",
        "raw_sources": [describe(s, True) for s in WATCHED_SOURCES],
        "expected_outputs": [describe(s) for s in WATCHED_OUTPUTS],
        "source_python_modules": len(list(PACKAGE.rglob("*.py"))),
        "test_python_files": len(list((ROOT / "tests").glob("test_*.py"))),
        "git_status_short": git_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply previewed cache cleanup")
    parser.add_argument("--inventory", action="store_true", help="Write logs/maintenance_inventory.json")
    args = parser.parse_args()
    check_project()

    gen_dirs, pycs, scaffolds = candidates()
    print("=" * 76)
    print("CFTR-THERATYPE | SAFE PROJECT MAINTENANCE")
    print("=" * 76)
    print("Mode:", "APPLY" if args.apply else "DRY RUN (no deletion)")
    print("\nRecreatable directories:")
    for path in gen_dirs:
        print("  ", relative(path))
    print("Standalone compiled files:", len(pycs))
    print("Empty scaffolding directories:")
    for path in scaffolds:
        print("  ", relative(path))
    if not any((gen_dirs, pycs, scaffolds)):
        print("  Nothing to clean")
    print("\nProtected: data/, results/, logs/, Git history, source .py, tests, README")

    if args.apply:
        for path in gen_dirs:
            guarded(path)
            shutil.rmtree(path)
        for path in pycs:
            guarded(path)
            path.unlink(missing_ok=True)
        for path in scaffolds:
            guarded(path)
            if path.is_dir() and not any(path.iterdir()):
                path.rmdir()
        print("\nPASS: Only generated caches and empty scaffolds removed")
    else:
        print("\nPreview only. Re-run with --apply to perform this cleanup.")

    if args.inventory:
        dest = ROOT / "logs" / "maintenance_inventory.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(inventory(), indent=2) + "\n", encoding="utf-8")
        print("Inventory written:", relative(dest))
    print("=" * 76)


if __name__ == "__main__":
    main()
