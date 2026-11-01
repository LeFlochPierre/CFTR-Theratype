"""Initial unified CFTR-Theratype command line interface.

Run as `python -m cftr_theratype.cli ...` from repository root.
"""
from __future__ import annotations
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="CFTR-Theratype canonical CLI (Stage 3)")
    sub = parser.add_subparsers(dest="action", required=True)

    bench = sub.add_parser("benchmark", help="Reproduce frozen Phase26G TRAIN-only CV")
    bench.add_argument("--target", choices=["corrector", "total"], default="total")
    bench.add_argument("--fast", action="store_true")
    bench.add_argument("--max-folds", type=int)
    bench.add_argument("--bootstraps", type=int, default=5000)
    bench.add_argument("--outdir", type=Path)
    bench.add_argument("--root", type=Path, default=Path.cwd())

    compare = sub.add_parser("compare", help="Compare new OOF CSV to historical frozen Phase26G CSV")
    compare.add_argument("--new", required=True, type=Path)
    compare.add_argument("--reference", required=True, type=Path)
    compare.add_argument("--atol", type=float, default=1e-6)
    compare.add_argument("--allow-partial", action="store_true")

    args = parser.parse_args(argv)
    if args.action == "benchmark":
        from cftr_theratype.experiments.feature_benchmark import run
        run(args.root, target=args.target, fast=args.fast,
            max_folds=args.max_folds, bootstraps=args.bootstraps,
            outdir=args.outdir)
    elif args.action == "compare":
        from cftr_theratype.experiments.feature_benchmark import compare_oof
        compare_oof(args.new, args.reference, atol=args.atol,
                    require_full=not args.allow_partial)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
