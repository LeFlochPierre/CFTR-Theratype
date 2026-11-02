#!/usr/bin/env python3
"""Reversible AST-level migration of duplicate legacy helpers to canonical modules.

No file moves. The original sources are backed up verbatim and may be restored.
Usage: python scripts/maintenance/stage6_migrate_hubs.py plan|apply|restore
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[2]
EXPECT = Path(__file__).with_name('stage6_expected_hashes.json')
BACKUP = ROOT / 'archive/legacy/hub_snapshots/stage6'
MANIFEST = BACKUP / 'stage6_migration_manifest.json'

# Wrapper signatures retain legacy import/call behaviour, but implementation
# uses canonical production modules validated in Stages 1-3.
WRAPPERS = {
    'scripts/phase26d_corrector_bottleneck.py': {
        'pos_mae': 'def pos_mae(y, pred, group):\n    from cftr_theratype.evaluation.metrics import position_mae\n    return position_mae(y, pred, group)\n',
        'model': 'def model(spec, seed=26):\n    from cftr_theratype.models.regressors import phase26d_model\n    return phase26d_model(spec, seed=seed)\n',
        'select_spec': 'def select_spec(X, y, group, quick=False):\n    from cftr_theratype.models.regressors import phase26d_select_spec\n    return phase26d_select_spec(X, y, group, quick=quick)\n',
        'folds_from_historical_oof': 'def folds_from_historical_oof(oof, expected_keys, position_by_key):\n    from cftr_theratype.evaluation.folds import from_historical_oof\n    return from_historical_oof(oof, expected_keys, position_by_key)\n',
        'load_train': 'def load_train():\n    from cftr_theratype.datasets.frozen_vti import load_train as canonical_load_train\n    return canonical_load_train(ROOT)\n',
    },
    'scripts/phase26g_feature_model_ablation.py': {
        'make_model': 'def make_model(kind, params, topk=None):\n    from cftr_theratype.models.regressors import phase26g_model\n    return phase26g_model(kind, params, topk)\n',
        'position_mae': 'def position_mae(y, pred, groups):\n    from cftr_theratype.evaluation.metrics import position_mae as shared\n    return shared(y, pred, groups)\n',
        'choose': 'def choose(x, y, groups, kind, topk, fast=False):\n    from cftr_theratype.models.regressors import phase26g_choose\n    return phase26g_choose(x, y, groups, kind, topk, fast)\n',
        'paired_ci': 'def paired_ci(d, candidate, baseline, n=5000):\n    from cftr_theratype.evaluation.metrics import paired_ci as shared\n    return shared(d, candidate, baseline, n=n)\n',
    },
    'scripts/phase26h_auxiliary_corrector.py': {
        'parsed_position': 'def parsed_position(s):\n    from cftr_theratype.datasets.eti_auxiliary import parsed_position as shared\n    return shared(s)\n',
        'load_data': 'def load_data():\n    from cftr_theratype.datasets.eti_auxiliary import load_data as canonical_load_data\n    return canonical_load_data(ROOT)\n',
        'positional_shuffle': 'def positional_shuffle(y, group, seed):\n    from cftr_theratype.models.auxiliary import positional_shuffle as shared\n    return shared(y, group, seed)\n',
        'fit_joint': 'def fit_joint(Vx, Vy, Ex, Ey, weight):\n    from cftr_theratype.models.auxiliary import fit_joint as shared\n    return shared(Vx, Vy, Ex, Ey, weight)\n',
        'choose_weight': 'def choose_weight(Xv, yv, gv, Xe, ye, ge, weights=WEIGHTS, shuffle_seed=None):\n    from cftr_theratype.models.auxiliary import choose_weight as shared\n    return shared(Xv, yv, gv, Xe, ye, ge, weights=weights, shuffle_seed=shuffle_seed)\n',
    },
    'scripts/phase27d1_biology_ablation.py': {
        'annotation_features': 'def annotation_features(positions, codes):\n    from cftr_theratype.features.biology_annotations import annotation_features as shared\n    return shared(positions, codes)\n',
    },
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def rewrite_source(content, wrappers):
    """Replace only named top-level functions; other source bytes stay unchanged."""
    tree = ast.parse(content)
    lines = content.splitlines(keepends=True)
    edits = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wrappers:
            replacement = wrappers[node.name]
            ast.parse(replacement)
            edits.append((node.lineno - 1, node.end_lineno, replacement + '\n'))
    found = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if missing := (set(wrappers) - found):
        raise RuntimeError(f'Not all legacy functions found: {sorted(missing)}')
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start:end] = [replacement]
    result = ''.join(lines)
    ast.parse(result)
    return result


def git_clean_target(root, rel):
    r = subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--', rel], cwd=root)
    if r.returncode != 0:
        raise RuntimeError(f'Working source modified, refusing migration: {rel}')


def plan(root=ROOT):
    hashes = json.loads(EXPECT.read_text())
    for rel, funcs in WRAPPERS.items():
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(p)
        git_clean_target(root, rel)
        if sha(p.read_bytes()) != hashes[rel]:
            raise RuntimeError(f'Unexpected source SHA256: {rel} (check correct Git HEAD)')
        new = rewrite_source(p.read_text(), funcs)
        before = p.read_text().count('\n')
        after = new.count('\n')
        print(f'{rel}: {len(funcs)} functions migrated; {before} -> {after} lines')
    print('PASS: exact source hashes and syntax verified; no files changed')


def apply(root=ROOT):
    if MANIFEST.exists() or BACKUP.exists():
        raise RuntimeError('Stage6 snapshot already exists; inspect or restore before applying again')
    plan(root)
    pending = {}
    for rel, funcs in WRAPPERS.items():
        p = root / rel
        original = p.read_bytes()
        migrated = rewrite_source(original.decode('utf-8'), funcs).encode('utf-8')
        pending[rel] = (original, migrated)
    BACKUP.mkdir(parents=True)
    manifest = {'files': {rel: {'original_sha256': sha(original),
                               'migrated_sha256': sha(migrated),
                               'functions': sorted(WRAPPERS[rel])}
                          for rel,(original,migrated) in pending.items()}}
    for rel, (original, migrated) in pending.items():
        dest = BACKUP / Path(rel).name
        dest.write_bytes(original)
    MANIFEST.write_text(json.dumps(manifest, indent=2) + '\n')
    for rel, (_, migrated) in pending.items():
        (root / rel).write_bytes(migrated)
    print('PASS: migrated shared helpers, exact originals saved to', BACKUP)
    print('Run parity tests + historical CSV comparisons BEFORE committing changes.')


def restore(root=ROOT):
    if not MANIFEST.is_file():
        raise RuntimeError('Stage6 backup/manifest not found')
    manifest = json.loads(MANIFEST.read_text())
    for rel, entry in manifest['files'].items():
        cur = root / rel
        snapshot = BACKUP / Path(rel).name
        if not snapshot.is_file() or sha(snapshot.read_bytes()) != entry['original_sha256']:
            raise RuntimeError(f'Original snapshot invalid: {rel}')
        if sha(cur.read_bytes()) != entry['migrated_sha256']:
            raise RuntimeError(f'Source was edited after migration; refusing to overwrite: {rel}')
    for rel in manifest['files']:
        shutil.copy2(BACKUP / Path(rel).name, root / rel)
    print('PASS: four original hub scripts restored; preserved snapshots retained at', BACKUP)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan','apply','restore'))
    args = parser.parse_args()
    {'plan': plan, 'apply': apply, 'restore': restore}[args.action]()

if __name__ == '__main__':
    main()
