#!/usr/bin/env python3
"""Phase 31I.2: inspect CFTR-Theratype candidate asset schemas (never load outcome rows).

From repository root: python scripts/phase31i2_inspect_inputs.py
Prints names, column schemas and definition lines for Phase 26D/26G loaders.
Does NOT train models, read VTI outcome values or modify project assets.
"""
from pathlib import Path
import re

import pandas as pd

ROOT = Path.cwd()
REGEX = re.compile(r"bihler|castellani|transfer|vti|eti|feature|split|fold|phase26d|phase26g", re.I)
SKIP = re.compile(r"checkpoint|prediction|bootstrap|metric|plot|figure|readme|\.log$", re.I)

def get_columns(p):
    try:
        if p.suffix.lower() == '.csv':
            return list(pd.read_csv(p, nrows=0).columns)
        if p.suffix.lower() == '.parquet':
            import pyarrow.parquet as pq
            return pq.read_schema(p).names
    except Exception as e:
        return [f'[schema unavailable: {type(e).__name__}: {e}]']
    return []


def main():
    paths = []
    for base in ['data/processed', 'data', 'results/biology', 'scripts']:
        folder = ROOT / base
        if not folder.exists():
            continue
        for p in folder.rglob('*'):
            if not p.is_file() or p.suffix.lower() not in {'.csv', '.parquet', '.py'}:
                continue
            if not REGEX.search(str(p.relative_to(ROOT))) or SKIP.search(p.name):
                continue
            paths.append(p)
    # Dedupe data files from nested directories and cap output for readability.
    paths = sorted(set(paths), key=lambda p: (p.suffix == '.py', str(p)))
    print('PHASE 31I.2 INPUT SCHEMA DISCOVERY (metadata only)')
    print(f'Root: {ROOT}; candidates found: {len(paths)}')
    for p in paths[:100]:
        print('\n', p.relative_to(ROOT))
        if p.suffix == '.py':
            try:
                lines = p.read_text(errors='replace').splitlines()
                matches = [(i, line.strip()) for i, line in enumerate(lines,1)
                           if re.match(r'\s*(def (load_train|load_.*|get_.*fold|build_.*feature)|class )',line)]
                for i, line in matches[:12]:
                    print(f'  L{i}: {line[:130]}')
            except OSError as e:
                print('  [unreadable]', e)
        else:
            cols = get_columns(p)
            print(f'  columns ({len(cols)}): {", ".join(cols[:45])}')
            if len(cols)>45:
                print(f'  ... {len(cols)-45} more columns')
    if len(paths)>100:
        print(f'\n... {len(paths)-100} more candidate paths not shown')
    print('\nUse these schemas to supply --bihler-target, --vti-train, --fold-col,')
    print('--vti-target, and --features/--feature-cols. Do not select assay-derived')
    print('predictors as genotype features. Nothing was fitted or modified.')


if __name__=='__main__':
    main()
