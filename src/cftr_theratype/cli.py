"""Installed CFTR-Theratype CLI. Only TRAIN-only / read-only research commands.

Install from project root with `python -m pip install -e . --no-deps` in CFTR_ENV.
Aliases retained for Stage 3: `benchmark --target ...` and `compare ...`.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys


def parser_factory():
    parser = argparse.ArgumentParser(prog='cftr', description='CFTR-Theratype canonical research CLI (Stage 6)')
    sub = parser.add_subparsers(dest='command', required=True)

    data = sub.add_parser('data', help='TRAIN-only dataset validation')
    dsub = data.add_subparsers(dest='operation', required=True)
    validate = dsub.add_parser('validate', help='Validate frozen VTI data / optional ETI training source')
    validate.add_argument('--root', type=Path, default=Path.cwd())
    validate.add_argument('--eti', action='store_true', help='Also validate ETI auxiliary TRAIN data')

    bench = sub.add_parser('benchmark', help='Frozen position-held-out Phase26G reproduction')
    bsub = bench.add_subparsers(dest='operation', required=True)
    run = bsub.add_parser('run')
    run.add_argument('--target', choices=('corrector','total'), default='total')
    run.add_argument('--fast', action='store_true')
    run.add_argument('--max-folds', type=int)
    run.add_argument('--bootstraps', type=int, default=5000)
    run.add_argument('--outdir', type=Path)
    run.add_argument('--root', type=Path, default=Path.cwd())
    comp = bsub.add_parser('compare', help='Check exact historical Phase26G OOF parity')
    comp.add_argument('--new', required=True, type=Path)
    comp.add_argument('--reference', required=True, type=Path)
    comp.add_argument('--atol', type=float, default=1e-6)
    comp.add_argument('--allow-partial', action='store_true')

    transfer = sub.add_parser('transfer', help='ETI->VTI TRAIN-only auxiliary experiment')
    tsub = transfer.add_subparsers(dest='operation', required=True)
    audit = tsub.add_parser('audit', help='Validate matched ETI/VTI features, no model training')
    audit.add_argument('--root', type=Path, default=Path.cwd())
    trun = tsub.add_parser('run', help='Canonical Phase26H real + shuffled negative controls')
    trun.add_argument('--root', type=Path, default=Path.cwd())
    trun.add_argument('--shuffles', type=int, default=2)
    trun.add_argument('--max-folds', type=int)
    trun.add_argument('--outdir', type=Path)
    tcomp = tsub.add_parser('compare', help='Compare Phase26H core OOF against archived output')
    tcomp.add_argument('--new', required=True, type=Path)
    tcomp.add_argument('--reference', required=True, type=Path)
    tcomp.add_argument('--atol', type=float, default=1e-6)
    tcomp.add_argument('--allow-partial', action='store_true')

    analyses = sub.add_parser('analysis', help='Read-only TRAIN results reporting')
    asub = analyses.add_subparsers(dest='operation', required=True)
    report = asub.add_parser('report', help='Read existing Phase26G TRAIN OOF, no retraining')
    report.add_argument('--root', type=Path, default=Path.cwd())
    report.add_argument('--out', type=Path)

    return parser


def main(argv=None):
    args_list = list(sys.argv[1:] if argv is None else argv)
    # Compatibility with Stage 3 `python -m cftr_theratype.cli benchmark --target total`.
    if args_list and args_list[0] == 'benchmark' and (len(args_list) == 1 or args_list[1].startswith('-')):
        args_list.insert(1, 'run')
    if args_list and args_list[0] == 'compare':
        args_list[:1] = ['benchmark', 'compare']
    args = parser_factory().parse_args(args_list)
    if args.command == 'data':
        from cftr_theratype.datasets.frozen_vti import load_train
        d, f, _, folds, reference = load_train(args.root)
        print(f'PASS: VTI TRAIN {len(d)} variants / {d.protein_position.nunique()} positions; '
              f'{len(folds)} frozen folds ({reference}); feature sets={sorted(f)}')
        if args.eti:
            from cftr_theratype.datasets.eti_auxiliary import load_data
            v, _, e, _, _ = load_data(args.root)
            print(f'PASS: ETI auxiliary TRAIN {len(e)} variants / {e.protein_position.nunique()} positions; '
                  f'{len(set(e.variant_key) & set(v.variant_key))} exact overlaps')
    elif args.command == 'benchmark':
        from cftr_theratype.experiments import feature_benchmark as fb
        if args.operation == 'run':
            fb.run(args.root, target=args.target, fast=args.fast,
                   max_folds=args.max_folds, bootstraps=args.bootstraps, outdir=args.outdir)
        else:
            fb.compare_oof(args.new, args.reference, atol=args.atol,
                           require_full=not args.allow_partial)
    elif args.command == 'transfer':
        if args.operation == 'audit':
            from cftr_theratype.datasets.eti_auxiliary import load_data
            v, X, e, E, folds = load_data(args.root)
            print(f'PASS: VTI TRAIN {len(v)} variants, ETI TRAIN {len(e)} variants, '
                  f'{len(folds)} frozen folds; {X.shape[1]}/{E.shape[1]} features')
        elif args.operation == 'run':
            from cftr_theratype.experiments.auxiliary_transfer import run
            run(args.root, shuffles=args.shuffles, max_folds=args.max_folds, outdir=args.outdir)
        else:
            from cftr_theratype.experiments.auxiliary_transfer import compare_oof
            compare_oof(args.new, args.reference, atol=args.atol,
                        full=not args.allow_partial)
    elif args.command == 'analysis':
        from cftr_theratype.experiments.summary_report import generate
        generate(args.root, output=args.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
