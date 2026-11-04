#!/usr/bin/env python3
"""Stage 8: audit RTX6 numbered scripts/docs and cautiously archive local patch ZIPs.

PLAN is read-only except for CSV/JSON reports under ignored results/maintenance/.
Only explicit STAGE-ARTIFACTS --confirm relocates checksum-verified local patch ZIPs.
Never moves code, documents, datasets, clinical material, results or third-party ZIPs.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

PREFIXES = ("cftr_phase", "cftr_refactor_", "cftr_stage", "cftr_eti_", "cftr_theratype_phase")
EXTS = {".py", ".md", ".txt", ".sh", ".bash", ".toml", ".yaml", ".yml"}

def git(root, *args):
    p = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError(f"Git error: {p.stderr.strip()}")
    return p.stdout

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

def inventory(root):
    tracked = set(git(root, "ls-files", "-z").split("\x00"))
    text = {}
    for name in ("src", "scripts", "tests", "docs", "archive"):
        base = root / name
        if not base.is_dir():
            continue
        for f in base.rglob("*"):
            if f.is_file() and not f.is_symlink() and f.suffix in EXTS and "__pycache__" not in f.parts:
                text[f] = f.read_text(encoding="utf-8", errors="replace")
    scripts, docs, zips = [], [], []
    for f in sorted((root / "scripts").glob("phase*.py")):
        if not re.match(r"phase\d", f.stem):
            continue
        imports, mentions = [], []
        for other, content in text.items():
            if other == f or (f.stem not in content and f.name not in content):
                continue
            rel = other.relative_to(root).as_posix()
            found = False
            if other.suffix == ".py":
                try:
                    tree = ast.parse(content, filename=rel)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            found |= any(a.name.split(".")[0] == f.stem for a in node.names)
                        elif isinstance(node, ast.ImportFrom) and node.module:
                            found |= f.stem in node.module.split(".")
                except SyntaxError:
                    mentions.append(rel + " [syntax error]")
            (imports if found else mentions).append(rel)
        rel = f.relative_to(root).as_posix()
        scripts.append({"path": rel, "tracked": rel in tracked,
                        "importers": " | ".join(sorted(set(imports))),
                        "mentions": " | ".join(sorted(set(mentions))),
                        "sha256": sha256(f), "status": "KEEP_DEPENDENCY" if imports else "REVIEW"})
    for f in sorted((root / "docs").rglob("*.md")) + sorted(root.glob("*.md")):
        rel = f.relative_to(root).as_posix()
        docs.append({"path": rel, "tracked": rel in tracked,
                     "category": "HISTORIC_PHASE" if re.search(r"PHASE\d|STAGE\d", f.name, re.I) else "REVIEW"})
    for f in sorted(root.glob("*.zip")):
        if not f.is_file() or f.is_symlink():
            continue
        source_refs, doc_refs = [], []
        for other, content in text.items():
            if f.name in content:
                (doc_refs if other.suffix == ".md" else source_refs).append(other.relative_to(root).as_posix())
        valid = True
        unsafe = 0
        try:
            with zipfile.ZipFile(f) as z:
                unsafe = sum(i.filename.startswith(("/", "\\")) or ".." in Path(i.filename.replace("\\", "/")).parts for i in z.infolist())
        except (zipfile.BadZipFile, OSError, zipfile.LargeZipFile):
            valid = False
        candidate = (f.name.lower().startswith(PREFIXES) and f.name not in tracked
                     and not source_refs and valid and not unsafe)
        zips.append({"name": f.name, "bytes": f.stat().st_size,
                     "tracked": f.name in tracked, "candidate": bool(candidate),
                     "source_refs": " | ".join(source_refs), "doc_refs": " | ".join(doc_refs),
                     "zip_ok": valid, "unsafe_members": unsafe})
    return tracked, scripts, docs, zips

def reports(root, tracked, scripts, docs, zips):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = root / "results/maintenance/cleanup_audit/stage8" / timestamp
    out.mkdir(parents=True, exist_ok=False)
    write_csv(out / "phase_scripts.csv", scripts,
              ["path", "tracked", "status", "importers", "mentions", "sha256"])
    write_csv(out / "documentation.csv", docs, ["path", "tracked", "category"])
    write_csv(out / "root_zip_plan.csv", zips,
              ["name", "bytes", "tracked", "candidate", "source_refs", "doc_refs", "zip_ok", "unsafe_members"])
    summary = {"branch": git(root, "branch", "--show-current").strip(),
               "commit": git(root, "rev-parse", "HEAD").strip(),
               "tracked_files": len(tracked),
               "numbered_scripts": len(scripts),
               "imported_scripts": sum(bool(r["importers"]) for r in scripts),
               "phase_docs": sum(r["category"] == "HISTORIC_PHASE" for r in docs),
               "documents": len(docs),
               "root_zips": len(zips),
               "patch_candidates": sum(r["candidate"] for r in zips),
               "untracked_numbered_scripts": [r["path"] for r in scripts if not r["tracked"]],
               "untracked_docs": [r["path"] for r in docs if not r["tracked"]]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("=" * 76)
    print("CFTR Stage 8 | workstation docs / scripts / patch ZIP audit")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("Reports:", out)
    print("No scripts, docs or scientific data moved.")
    return out

def stage(root, zips):
    if git(root, "diff", "--name-only").strip() or git(root, "diff", "--cached", "--name-only").strip():
        raise RuntimeError("Tracked local changes exist; preserve or commit them before archive moves")
    manifest = root / "results/maintenance/cleanup_audit/stage8_moves.json"
    if manifest.exists():
        raise FileExistsError(manifest)
    dest = root.parent / "CFTR-Theratype-local-archive" / "root-patches"
    chosen = [x for x in zips if x["candidate"]]
    for x in chosen:
        if (dest / x["name"]).exists():
            raise FileExistsError(dest / x["name"])
    if not chosen:
        print("No eligible patch ZIPs; nothing staged.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    moves = []
    for x in chosen:
        src = root / x["name"]
        dst = dest / x["name"]
        checksum = sha256(src)
        shutil.copy2(src, dst)
        if sha256(dst) != checksum:
            raise RuntimeError("SHA256 mismatch; originals preserved: " + x["name"])
        moves.append({"source": str(src), "archive": str(dst), "sha256": checksum})
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"status": "copied_and_verified", "files": moves}, indent=2) + "\n")
    for x in moves:
        src = Path(x["source"])
        if sha256(src) != x["sha256"]:
            raise RuntimeError("Source changed before removal: " + str(src))
        src.unlink()
    manifest.write_text(json.dumps({"status": "archived", "files": moves}, indent=2) + "\n")
    print(f"Archived {len(moves)} verified local patch ZIPs to {dest}")
    print("Manifest:", manifest)

def restore(root):
    manifest = root / "results/maintenance/cleanup_audit/stage8_moves.json"
    info = json.loads(manifest.read_text())
    for x in info["files"]:
        dst, src = Path(x["source"]), Path(x["archive"])
        if not src.exists() or sha256(src) != x["sha256"]:
            raise RuntimeError("Missing or changed archive: " + str(src))
        if dst.exists() and sha256(dst) != x["sha256"]:
            raise FileExistsError("Existing conflicting file: " + str(dst))
    for x in info["files"]:
        dst, src = Path(x["source"]), Path(x["archive"])
        if not dst.exists():
            shutil.copy2(src, dst)
            if sha256(dst) != x["sha256"]:
                raise RuntimeError("Restore verification failed: " + str(dst))
    manifest.rename(manifest.with_name("stage8_restored_moves.json"))
    print("Original ZIPs restored; verified archived copies retained.")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("plan", "stage-artifacts", "restore-artifacts"))
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--confirm", action="store_true")
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    if not (root / ".git").exists() or not (root / "pyproject.toml").exists() or not (root / "src/cftr_theratype").is_dir():
        p.error("Expected CFTR-Theratype working tree")
    if args.action == "restore-artifacts":
        if not args.confirm:
            p.error("Explicit --confirm required for restore")
        restore(root)
        return
    tracked, scripts, docs, zips = inventory(root)
    reports(root, tracked, scripts, docs, zips)
    if args.action == "stage-artifacts":
        if not args.confirm:
            p.error("Explicit --confirm required for root ZIP archival")
        stage(root, zips)

if __name__ == "__main__":
    main()
