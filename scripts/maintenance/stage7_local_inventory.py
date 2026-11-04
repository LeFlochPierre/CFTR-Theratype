#!/usr/bin/env python3
"""Read-only CFTR-Theratype local-inventory audit.

Creates only an audit report under results/maintenance/cleanup_audit.
Never deletes, moves, extracts, or changes scientific inputs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import zipfile


def git(root: Path, *args: str) -> str:
    p = subprocess.run(["git", *args], cwd=root, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True, check=False)
    if p.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
    return p.stdout.strip()


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def zip_inventory(path: Path, do_hash: bool) -> dict:
    row = {"name": path.name, "bytes": path.stat().st_size,
           "sha256": digest(path) if do_hash else "NOT_COMPUTED",
           "members": None, "uncompressed_bytes": None,
           "unsafe_member_paths": None, "zip_error": ""}
    try:
        with zipfile.ZipFile(path) as z:
            entries = z.infolist()
            unsafe = [x.filename for x in entries if x.filename.startswith(("/", "\\"))
                      or ".." in Path(x.filename.replace("\\", "/")).parts]
            row["members"] = len(entries)
            row["uncompressed_bytes"] = sum(x.file_size for x in entries)
            row["unsafe_member_paths"] = len(unsafe)
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        row["zip_error"] = type(exc).__name__ + ": " + str(exc)
    return row


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--hash-zips", action="store_true",
                        help="Compute SHA256 for root ZIPs (slower, recommended for final archival)")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not (root / ".git").exists() or not (root / "pyproject.toml").exists():
        parser.error(f"Not a CFTR-Theratype Git working tree: {root}")

    branch = git(root, "branch", "--show-current")
    commit = git(root, "rev-parse", "HEAD")
    status = git(root, "status", "--porcelain=v1", "-uall")
    tracked = set(git(root, "ls-files").splitlines())

    root_zip_paths = sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() == ".zip")
    zips = [zip_inventory(p, args.hash_zips) for p in root_zip_paths]
    for r in zips:
        r["tracked_in_git"] = r["name"] in tracked

    focus_dirs = ["src", "scripts", "tests", "docs", "archive"]
    code_files = []
    for dirname in focus_dirs:
        folder = root / dirname
        if not folder.is_dir():
            continue
        for p in folder.rglob("*"):
            if not p.is_file() or "__pycache__" in p.parts:
                continue
            rel = p.relative_to(root).as_posix()
            code_files.append({"path": rel, "bytes": p.stat().st_size,
                               "tracked_in_git": rel in tracked})
    code_files.sort(key=lambda r: r["path"])

    root_rows = []
    for p in sorted(root.iterdir()):
        if p.name == ".git":
            continue
        root_rows.append({"name": p.name, "kind": "directory" if p.is_dir() else "file",
                          "bytes_if_file": p.stat().st_size if p.is_file() else "",
                          "tracked_in_git": p.name in tracked})

    outdir = root / "results" / "maintenance" / "cleanup_audit"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = outdir / stamp
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()

    write_csv(out / "root_entries.csv", root_rows, ["name", "kind", "bytes_if_file", "tracked_in_git"])
    write_csv(out / "root_zip_inventory.csv", zips,
              ["name", "bytes", "sha256", "members", "uncompressed_bytes",
               "unsafe_member_paths", "zip_error", "tracked_in_git"])
    write_csv(out / "source_inventory.csv", code_files, ["path", "bytes", "tracked_in_git"])
    (out / "git_status.txt").write_text(status + "\n", encoding="utf-8")
    summary = {"project_root": str(root), "timestamp_utc": stamp,
               "branch": branch, "commit": commit,
               "tracked_git_files": len(tracked),
               "git_status_entries": len(status.splitlines()) if status else 0,
               "root_zip_count": len(zips),
               "root_zip_untracked_count": sum(not r["tracked_in_git"] for r in zips),
               "source_files_scanned": len(code_files),
               "untracked_source_files": [r["path"] for r in code_files if not r["tracked_in_git"]],
               "zip_hashes_computed": args.hash_zips,
               "excluded_heavy_dirs": ["data", "results", "logs"],
               "note": "Inventory only; no files moved/deleted. data/results/logs not recursively scanned."}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("=" * 72)
    print("CFTR-Theratype | Stage 7 LOCAL INVENTORY | READ ONLY")
    print("=" * 72)
    for key in ("project_root", "branch", "commit", "tracked_git_files",
                "git_status_entries", "root_zip_count", "root_zip_untracked_count",
                "source_files_scanned"):
        print(f"{key}: {summary[key]}")
    print(f"untracked_source_files: {len(summary['untracked_source_files'])}")
    for name in summary["untracked_source_files"][:20]:
        print(f"  {name}")
    if len(summary["untracked_source_files"]) > 20:
        print("  ... (see summary.json)")
    print("Root ZIP archives:")
    for item in zips:
        status_s = "TRACKED" if item["tracked_in_git"] else "LOCAL-ONLY"
        print(f"  {status_s:10} {item['bytes'] / 1024**2:8.2f} MiB  {item['name']}")
    print("Audit output:", out)
    print("NO FILES DELETED, MOVED, OR EXTRACTED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())