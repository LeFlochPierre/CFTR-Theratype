#!/usr/bin/env python3
"""Execute an archived historical phase script with original project import paths.

Usage: python scripts/run_legacy.py phase29a_extreme_response_capacity.py [args...]
Run from the CFTR-Theratype repository root to preserve historical CWD assumptions.
"""
from __future__ import annotations
import runpy
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python scripts/run_legacy.py <phase-script.py> [args...]')
    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        raise SystemExit('Run from the repository root: cd ~/projects/CFTR-Theratype')
    filename = Path(sys.argv[1])
    if filename.name != str(filename) or not filename.name.startswith('phase') or filename.suffix != '.py':
        raise SystemExit('Expected a historical phase script filename, without directories')
    script = root / 'archive/legacy/scripts' / filename.name
    if not script.is_file():
        raise SystemExit(f'Not archived here: {script}')
    # The original scripts assumed both the scripts directory and package src
    # are importable. Both active and archived legacy modules remain accessible.
    for folder in [root / 'src', root / 'scripts', script.parent]:
        s = str(folder)
        if s not in sys.path:
            sys.path.insert(0, s)
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
