"""Immutable local archive of train-only CFTR-Theratype research artifacts.

Usage:
    python -m cftr_theratype.checkpoint --dry-run
    python -m cftr_theratype.checkpoint --create
    python -m cftr_theratype.checkpoint --verify

No data frames/assay values are read or processed: selected files are copied
byte-for-byte; raw workbooks, protected outcomes and frozen splits are hashed
but not copied. Snapshots cannot be overwritten by --create.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

CHECKPOINT_ID = "phase13_2026-10-23"
RELATIVE_DEST = Path("results/frozen") / CHECKPOINT_ID

# Training-only generated artifacts, never the ETI/VTI validation or test tables.
REQUIRED_COPY = (
    "results/transfer/latent_transfer_summary.json",
    "results/transfer/latent_transfer_oof.csv",
    "results/transfer/cohort_robustness_summary.json",
    "data/manifests/split_protocol.json",
    "data/manifests/vti_transfer_protocol.json",
)
OPTIONAL_COPY = (
    "results/transfer/latent_transfer_summary.csv",
    "results/transfer/summary.json",
    "results/transfer/train_cv_predictions.csv",
    "results/transfer/training_exact_overlap.csv",
    "results/transfer/training_exact_overlap.json",
    "results/transfer/corrector_cv_summary.json",
    "results/transfer/corrector_train_cv_predictions.csv",
    "results/transfer/reliability_summary.json",
    "results/transfer/reliability_position_scores.csv",
    "results/transfer/cohort_shift_summary.json",
    "results/transfer/cohort_shift_baseline_strata.csv",
    "results/transfer/cohort_shift_controls.json",
    "results/transfer/cohort_shift_controls_oof.csv",
    "results/transfer/cohort_robustness_summary.csv",
    "results/transfer/cohort_robustness_oof.csv",
    "data/manifests/cross_study_protocol.json",
    "data/manifests/data_sources.json",
    "docs/RESEARCH_CHECKPOINT.md",
    "docs/results_checkpoint_2026-10-21.json",
    "docs/LATENT_TRANSFER_PROTOCOL.md",
    "docs/PHASE13_FROZEN_RESULTS.md",
    "docs/PHASE14_BIOLOGY_TARGET_AUDIT.md",
    "docs/phase13_results_2026-10-23.json",
    "logs/latent_transfer.log",
    "logs/cohort_robustness.log",
    "logs/cohort_shift.log",
    "logs/cohort_shift_controls.log",
    "logs/transfer_reliability.log",
    "logs/transfer_mechanism.log",
)
HASH_ONLY = (
    "data/raw/bihler_2024/mmc3.xlsx",
    "data/raw/bihler_2026/mmc1.xlsx",
    "data/raw/mcdonald_2024/table_s3.xlsx",
    "data/processed/bihler_2024_missense_splits.parquet",
    "data/processed/bihler_2026_transfer_splits.parquet",
    "data/interim/esm2_35m_development.npz",
)
EXPECTED_RAW_SHA256 = {
    "data/raw/bihler_2024/mmc3.xlsx": "bbc759607c333e64fa6b228c4f29c62939d27fa5d4157a937bd9944476d395ce",
    "data/raw/bihler_2026/mmc1.xlsx": "92c4fdea737ba6e4499b1cb7a4f62fe21b56330323fe899908b63d5abdd93b23",
    "data/raw/mcdonald_2024/table_s3.xlsx": "28f721c1795b2516653c9f5383521697ed11b5fa851b75a68b4be1584c1ea269",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_file(root: Path, relative: str) -> Path | None:
    """Refuse symlinks or paths outside the repository."""
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Unsafe relative path: {relative}")
    target = root / p
    if target.is_symlink() or any(x.is_symlink() for x in target.parents if x != root.parent):
        raise ValueError(f"Symlink disallowed: {relative}")
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"File outside repository: {relative}")
    if not target.exists():
        return None
    if not target.is_file():
        raise ValueError(f"Not a regular file: {relative}")
    return target


def git_info(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        proc = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return proc.stdout.strip() if proc.returncode == 0 else None
    status = run("status", "--porcelain", "--untracked-files=normal")
    return {"head": run("rev-parse", "HEAD"), "branch": run("branch", "--show-current"),
            "working_tree_clean": status == "", "status_entries": status.splitlines() if status else []}


def collect(root: Path) -> tuple[dict[str, Any], list[str]]:
    copies = {}
    missing_required = []
    for name in REQUIRED_COPY + OPTIONAL_COPY:
        file = safe_file(root, name)
        if file is None:
            if name in REQUIRED_COPY:
                missing_required.append(name)
            continue
        copies[name] = {"sha256": sha256(file), "bytes": file.stat().st_size}

    hashes = {}
    raw_mismatch = []
    for name in HASH_ONLY:
        file = safe_file(root, name)
        if file is not None:
            actual = sha256(file)
            hashes[name] = {"sha256": actual, "bytes": file.stat().st_size}
            if name in EXPECTED_RAW_SHA256 and actual != EXPECTED_RAW_SHA256[name]:
                raw_mismatch.append(name)
        else:
            hashes[name] = {"missing": True}

    source_files = {}
    for group in ("src", "tests", "scripts"):
        base = root / group
        if base.exists():
            for file in sorted(base.rglob("*.py")):
                rel = file.relative_to(root).as_posix()
                file = safe_file(root, rel)
                if file is not None:
                    source_files[rel] = sha256(file)

    manifest = {
        "schema_version": "1.0", "checkpoint": CHECKPOINT_ID,
        "purpose": "Immutable training-only exploratory checkpoint, not confirmatory validation",
        "data_access": "Copies explicit allowlist of development outputs only; raw/processed are hashed but not copied; no test outcomes opened",
        "git": git_info(root), "copied_files": copies,
        "hash_only": hashes, "source_python_sha256": source_files,
        "missing_required": missing_required, "raw_hash_mismatch": raw_mismatch,
        "missing_optional": [n for n in OPTIONAL_COPY if n not in copies],
        "test_status": "ETI 108 and VTI 33 final test variants sealed (based on reported experiment protocol; not independently inspected)",
    }
    return manifest, missing_required + raw_mismatch


def create(root: Path, *, dry_run: bool) -> Path:
    root = root.resolve()
    out = root / RELATIVE_DEST
    if out.exists():
        raise FileExistsError(f"Snapshot already exists: {out}. Run --verify; never overwrite.")
    manifest, problems = collect(root)
    print("=" * 77)
    print("CFTR-THERATYPE | PHASE 13 FROZEN CHECKPOINT")
    print("=" * 77)
    print(f"Mode: {'PREVIEW' if dry_run else 'CREATE'}; files selected: {len(manifest['copied_files'])}; source Python: {len(manifest['source_python_sha256'])}")
    for name in manifest["copied_files"]:
        print("  COPY", name)
    for name in manifest["missing_optional"]:
        print("  OPTIONAL MISSING", name)
    print("Git HEAD:", manifest["git"]["head"])
    print("Working tree clean:", manifest["git"]["working_tree_clean"])
    for name in problems:
        print("  BLOCKER", name)
    if problems:
        raise RuntimeError("Missing required result/split or raw checksum mismatch; snapshot not created")
    if not manifest["git"]["working_tree_clean"]:
        print("WARNING: uncommitted/untracked changes are present; commit source and docs first for strongest provenance")
    if dry_run:
        print("Preview only; no files changed")
        return out

    out.mkdir(parents=True, exist_ok=False)
    manifest["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    for name in manifest["copied_files"]:
        dest = out / "files" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, dest)
        if sha256(dest) != manifest["copied_files"][name]["sha256"]:
            raise RuntimeError(f"Copy verification failure: {name}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print("PASS: archived byte-for-byte training-only files and verified SHA-256")
    print(f"Snapshot: {out}")
    print("NEXT: python -m cftr_theratype.checkpoint --verify")
    print("KEEP: back up results/frozen/ outside Git separately")
    return out


def verify(root: Path) -> int:
    out = root.resolve() / RELATIVE_DEST
    manifest = json.loads((out / "manifest.json").read_text())
    failures = []
    for name, metadata in manifest["copied_files"].items():
        file = safe_file(out / "files", name)
        if file is None or sha256(file) != metadata["sha256"]:
            failures.append(name)
    print("CFTR PHASE13 SNAPSHOT VERIFY")
    print("Archived files:", len(manifest["copied_files"]))
    print("Recorded source commit:", manifest["git"]["head"])
    print("Snapshot integrity:", "PASS" if not failures else "FAIL")
    for name in failures:
        print("  MISMATCH:", name)
    return int(bool(failures))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--verify", action="store_true")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project directory")
    args = parser.parse_args()
    if args.verify:
        raise SystemExit(verify(args.root))
    create(args.root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
