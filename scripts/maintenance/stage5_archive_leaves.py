#!/usr/bin/env python3
"""Conservative, reversible archive of historical *leaf* phase scripts.

Only files with zero detected Python/code references and no __file__ dependency
are eligible. This is a safe-first migration heuristic, NOT proof of safety.
The original source remains in Git; archived scripts retain their bytes.

Usage:
    python scripts/maintenance/stage5_archive_leaves.py plan
    python scripts/maintenance/stage5_archive_leaves.py apply --limit 20
    python scripts/maintenance/stage5_archive_leaves.py restore
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

SOURCE_EXT = {'.py', '.sh', '.bash', '.yaml', '.yml', '.toml', '.ini', '.cfg'}
SCAN_DIRS = ('scripts', 'src', 'tests', 'tools', 'configs')
PROTECTED = {
    'phase26d_corrector_bottleneck',
    'phase26g_feature_model_ablation',
    'phase26h_auxiliary_corrector',
    'phase26k_trafficking_preflight',
    'phase27b1_extreme_classifiers',
    'phase27d1_biology_ablation',
    'phase31i2_repository',
    'phase31i_auxiliary_transfer',
    'phase31j_error_diagnostics',
    'phase31k_response_generalisation',
    'phase31l_predictability',
    'phase31m_three_tracks',
    'phase32a_adaptive_theratyping',
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(['git', *args], cwd=root, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def sources(root: Path) -> list[Path]:
    all_files: list[Path] = []
    for folder in SCAN_DIRS:
        d = root / folder
        if d.exists():
            all_files.extend(p for p in d.rglob('*') if p.is_file()
                             and p.suffix in SOURCE_EXT
                             and '__pycache__' not in p.parts)
    return sorted(set(all_files))


def plan_repo(root: Path) -> list[dict]:
    scripts = sorted((root / 'scripts').glob('phase*.py'))
    candidates = [p for p in scripts if re.match(r'^phase\d', p.stem)]
    texts = {p: p.read_text(encoding='utf-8', errors='replace') for p in sources(root)}
    rows = []
    for candidate in candidates:
        stem = candidate.stem
        inbound = []
        text_refs = []
        parse_issues = []
        pat = re.compile(r'(?<![A-Za-z0-9_])' + re.escape(stem) + r'(?![A-Za-z0-9_])')
        for other, content in texts.items():
            if other == candidate:
                continue
            if not pat.search(content) and candidate.name not in content:
                continue
            if other.suffix == '.py':
                try:
                    tree = ast.parse(content, filename=str(other))
                except SyntaxError:
                    parse_issues.append(str(other.relative_to(root)))
                    continue
                found = False
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        found |= any(a.name.split('.')[0] == stem for a in node.names)
                    elif isinstance(node, ast.ImportFrom):
                        found |= bool(node.module and stem in node.module.split('.'))
                        found |= any(a.name == stem for a in node.names)
                if found:
                    inbound.append(str(other.relative_to(root)))
                else:
                    text_refs.append(str(other.relative_to(root)))
            else:
                text_refs.append(str(other.relative_to(root)))

        body = texts.get(candidate, '')
        reasons = []
        if stem in PROTECTED:
            reasons.append('protected_shared_or_current')
        if inbound:
            reasons.append('python_importers')
        if text_refs:
            reasons.append('code_or_config_references')
        if parse_issues:
            reasons.append('unparsed_reference_source')
        if '__file__' in body:
            reasons.append('file_location_dependency')
        # A source-only static scan cannot prove the absence of external callers.
        # Only the user-approved apply mode can move an eligible file.
        rows.append({
            'file': str(candidate.relative_to(root)),
            'status': 'CANDIDATE' if not reasons else 'RETAIN_OR_REVIEW',
            'reasons': '|'.join(reasons),
            'python_importers': '|'.join(sorted(inbound)),
            'code_references': '|'.join(sorted(text_refs)),
            'sha256': sha256(candidate),
        })
    return rows


def report(root: Path, rows: list[dict]) -> Path:
    folder = root / 'results/maintenance/refactor_audit/stage5'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'archive_plan.csv'
    with path.open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]) if rows else
                                ['file', 'status', 'reasons', 'python_importers', 'code_references', 'sha256'])
        writer.writeheader()
        writer.writerows(rows)
    return path


def require_clean_tracked_repo(root: Path) -> None:
    if git(root, 'rev-parse', '--is-inside-work-tree').stdout.strip() != 'true':
        raise RuntimeError('Not inside a Git repository')
    status = git(root, 'status', '--porcelain', '--untracked-files=no')
    if status.returncode or status.stdout.strip():
        raise RuntimeError('Tracked/staged changes exist. Commit your source checkpoint FIRST.\n' + status.stdout[:1000])


def do_apply(root: Path, rows: list[dict], limit: int) -> None:
    require_clean_tracked_repo(root)
    if limit < 1 or limit > 50:
        raise ValueError('--limit must be between 1 and 50 (small reviewable batches)')
    manifest_path = root / 'results/maintenance/refactor_audit/stage5/archive_manifest.json'
    if manifest_path.exists():
        raise RuntimeError('Existing archive manifest found. Commit/review or restore previous batch first.')
    chosen = [r for r in rows if r['status'] == 'CANDIDATE'][:limit]
    if not chosen:
        print('No candidates eligible for automatic archival. No changes made.')
        return
    moves = []
    try:
        for row in chosen:
            old = row['file']
            new = 'archive/legacy/scripts/' + Path(old).name
            (root / new).parent.mkdir(parents=True, exist_ok=True)
            if (root / new).exists():
                raise RuntimeError('Archive collision: ' + new)
            tracked = git(root, 'ls-files', '--error-unmatch', '--', old)
            if tracked.returncode:
                raise RuntimeError('File not tracked; checkpoint missing: ' + old)
            result = git(root, 'mv', '--', old, new)
            if result.returncode:
                raise RuntimeError('git mv failed: ' + result.stderr)
            if sha256(root / new) != row['sha256']:
                raise RuntimeError('Hash mismatch after move: ' + old)
            moves.append({'from': old, 'to': new, 'sha256': row['sha256']})
    except BaseException:
        for move in reversed(moves):
            git(root, 'mv', '--', move['to'], move['from'])
        raise
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'git_base': git(root, 'rev-parse', 'HEAD').stdout.strip(),
                'moves': moves}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'MOVED {len(moves)} leaf scripts (via git mv); verify tests before commit.')
    for item in moves:
        print('  ', item['from'], '->', item['to'])
    print('Manifest:', manifest_path)
    print('Historical scripts remain in Git and under archive/legacy/scripts/.')


def do_restore(root: Path) -> None:
    manifest_path = root / 'results/maintenance/refactor_audit/stage5/archive_manifest.json'
    if not manifest_path.exists():
        raise RuntimeError('No archive manifest to restore')
    moves = json.loads(manifest_path.read_text())['moves']
    for item in reversed(moves):
        old, new = item['from'], item['to']
        if (root / old).exists():
            raise RuntimeError('Original path already exists: ' + old)
        if not (root / new).is_file() or sha256(root / new) != item['sha256']:
            raise RuntimeError('Missing/modified archived script: ' + new)
        result = git(root, 'mv', '--', new, old)
        if result.returncode:
            raise RuntimeError('git mv restore failed: ' + result.stderr)
    manifest_path.unlink()
    print(f'RESTORED {len(moves)} scripts; original bytes and locations recovered.')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('plan', 'apply', 'restore'))
    ap.add_argument('--limit', type=int, default=20)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    if not (root / 'scripts').is_dir():
        raise RuntimeError('Repository scripts/ directory missing')
    if args.command == 'restore':
        do_restore(root)
        return
    rows = plan_repo(root)
    out = report(root, rows)
    possible = [r for r in rows if r['status'] == 'CANDIDATE']
    print(f'STAGE 5 | historical phase scripts: {len(rows)} | leaf candidates: {len(possible)} | retained/review: {len(rows)-len(possible)}')
    print('Plan:', out)
    print('This audit ignores documentation mentions (historical documents remain unchanged).')
    print('No code rewrite, no source deletion. External callers cannot be detected by static inspection.')
    for r in possible[:25]:
        print('  candidate:', r['file'])
    if args.command == 'apply':
        do_apply(root, rows, args.limit)


if __name__ == '__main__':
    main()
