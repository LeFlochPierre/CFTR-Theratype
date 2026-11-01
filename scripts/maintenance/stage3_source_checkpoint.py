#!/usr/bin/env python3
"""Non-destructive snapshot of CFTR research SOURCE files, including untracked files.

Writes a timestamped tar.gz OUTSIDE the repository plus a JSON manifest.
It never executes git add, modifies tracked files or includes data/results/logs.
Usage: python scripts/maintenance/stage3_source_checkpoint.py
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
from datetime import datetime, timezone

ROOTS = ("scripts", "src", "tests", "docs", "configs", "tools")
SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".json", ".txt", ".sh"}
TOP_LEVEL = ("README.md", "RECAP.md", "pyproject.toml", "setup.py",
             "setup.cfg", "requirements.txt", "environment.yml", ".gitignore")

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(root, *args):
    p = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    return p.stdout if p.returncode == 0 else f"ERROR: {p.stderr.strip()}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--backup-dir", type=Path)
    args = ap.parse_args(argv)
    root = args.root.resolve()
    if not (root / "src/cftr_theratype").is_dir() or not (root / "scripts").is_dir():
        raise SystemExit(f"Not the CFTR-Theratype repository: {root}")
    backup = (args.backup_dir or root.parent / "CFTR-Theratype-backups").resolve()
    if backup == root or root in backup.parents:
        raise SystemExit("Backup destination must be OUTSIDE repository")
    backup.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = backup / f"cftr_source_stage3_{tag}.tar.gz"
    if output.exists():
        raise FileExistsError(output)
    files = []
    for dirname in ROOTS:
        p = root / dirname
        if p.is_dir():
            files.extend(q for q in p.rglob("*") if q.is_file()
                         and q.suffix.lower() in SUFFIXES
                         and "__pycache__" not in q.parts)
    files.extend(root / n for n in TOP_LEVEL if (root / n).is_file())
    files = sorted(set(files))
    entries = []
    for path in files:
        if path.is_symlink():
            raise RuntimeError(f"Refusing to snapshot symlink: {path}")
        rel = str(path.relative_to(root))
        entries.append({"path": rel, "size": path.stat().st_size, "sha256": sha256(path)})
    # The canonical path manifest is written separately for easy audit.
    with tarfile.open(output, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    with tarfile.open(output, "r:gz") as archive:
        archived = {member.name for member in archive.getmembers()}
    if archived != {e["path"] for e in entries}:
        raise RuntimeError("Archive content verification failed")
    manifest = backup / f"cftr_source_stage3_{tag}.json"
    manifest.write_text(json.dumps({
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git(root, "rev-parse", "HEAD").strip(),
        "git_status": git(root, "status", "--short"),
        "archive": str(output), "file_count": len(entries),
        "files": entries,
        "excluded": ["data/", "results/", "logs/", "external binary assets"],
    }, indent=2))
    print(f"PASS: {len(entries)} source/config files backed up, including untracked source")
    print("Archive:", output)
    print("Manifest:", manifest)
    print("Git HEAD:", git(root, "rev-parse", "--short", "HEAD").strip())
    print("No existing source, dataset or results modified")

if __name__ == "__main__":
    main()
