# CFTR-Theratype

Reproducible public-data research on CFTR in-vitro functional response, cross-assay transfer and prediction reliability. The genotype-only model development programme is an **exploratory benchmark**, not a clinical treatment recommendation or a demonstrated breakthrough.

**Reference state:** Git branch `refactor/cftr-modular-cleanup`, Stage 6 prepared against `6698a9045276`. This repository's VTI/ETI validation and test outcomes remain protected; canonical commands below operate on TRAIN-only measurements and position-disjoint OOF assignments.

## Quick start

```bash
conda activate CFTR_ENV
python -m pip install -e . --no-deps  # preserve the environment used for frozen parity
cftr --help
cftr data validate                   # 210 VTI TRAIN variants, 168 protein positions, 10 folds
cftr data validate --eti             # also 383 ETI TRAIN variants, 276 positions
cftr transfer audit                  # ETI-VTI matched-feature check without training
```

The old `cftr-data` CLI remains available; its build/split commands can overwrite exploratory derived files. Do not rebuild them casually while checking frozen parity.

## Reproducing frozen TRAIN-only benchmarks

```bash
# These may be slow; outputs go into a separate maintenance/refactor directory.
cftr benchmark run --target total
cftr benchmark run --target corrector

# Or compare the already generated Stage 3 OOF CSV files:
cftr benchmark compare \
  --new results/maintenance/refactor_audit/stage3/full/phase26g_total_oof.csv \
  --reference results/biology/phase26g/phase26g_total_oof.csv

# Existing Stage 3 flags remain supported:
python -m cftr_theratype.cli benchmark --target total
python -m cftr_theratype.cli compare --new NEW.csv --reference REFERENCE.csv
```

The full Phase26G OOF has **420 rows** (210 variants in two repeats) and nine regression model configurations. Both total and corrector predictions were verified to historical tolerance `1e-6` on the source workstation before Stage 6.

## ETI -> VTI auxiliary transfer

```bash
# Does not inspect protected validation/test response outcomes
cftr transfer audit

# Smoke test (use explicit output directory when repeating):
cftr transfer run --shuffles 0 --max-folds 1 \
  --outdir results/maintenance/refactor_audit/stage6/transfer_smoke

# Full frozen TRAIN-only Phase26H reproduction; independent outputs:
cftr transfer run --shuffles 2

# Once full results exist, validate against preserved Phase26H outputs:
cftr transfer compare \
  --new results/maintenance/refactor_audit/stage6/transfer/phase26h_oof.csv \
  --reference results/biology/phase26h/phase26h_oof.csv
```

The transfer runner uses all five original Phase26H ETI weight selection and negative-control conventions, excluding held-out positions from auxiliary supervision. The source-level protocol is described in `docs/refactor/STAGE6_MIGRATION.md`. Full historical parity must still be checked on the workstation.

## Existing-results report

```bash
cftr analysis report                # reads frozen TRAIN-only Phase26G CSV files
cftr analysis report --out results/maintenance/refactor_audit/stage6/train_report.md
```

Report generation does not retrain models or access final test outcomes. It refuses to overwrite an existing report.

## Source structure

- `src/cftr_theratype/datasets/`: canonical TRAIN-only cohort/feature loaders
- `src/cftr_theratype/models/`: shared regression and auxiliary training helpers
- `src/cftr_theratype/evaluation/`: frozen folds, metrics and leakage checks
- `src/cftr_theratype/experiments/`: canonical benchmark and transfer entry points
- `src/cftr_theratype/cli.py`: installed `cftr` CLI (plus old `cftr-data`)
- `scripts/maintenance/`: safe auditing / reversible source migration
- `scripts/phase*.py`: **legacy compatibility entry points**, not new code templates
- `archive/legacy/scripts/`: historical experiments preserved in Git
- `tests/`: synthetic unit tests and source-parity tests
- `results/biology/`: historical exploratory result tables; `results/maintenance/`: refactor checks

Historical modules remain importable to preserve old downstream experiments. Stage 6 provides an *optional*, hash-checked migration of 15 duplicated helpers; it does **not** automatically move all 29 remaining legacy scripts. Do not delete the hubs while archived scripts still import them.

## Research and data caveats

Read `docs/RESEARCH_CHECKPOINT.md` and `docs/REFACTOR_STAGE2.md` through `docs/REFACTOR_STAGE5.md` for provenance. Published datasets include Bihler 2024, Bihler 2026, and the McDonald-derived data (not an independent external replication). Raw spreadsheets and external assets may have separate redistribution licenses.

The target is **in-vitro CFTR rescue**, not clinical efficacy. The published 2026 VTI assay is not identical to the marketed vanzacaftor/tezacaftor/deutivacaftor combination. Development folds, even if repeated, are not equivalent to independent clinical cohorts. Keep final VAL/TEST partitions sealed until a confirmatory protocol is frozen.

See [Stage 6 migration guide](docs/refactor/STAGE6_MIGRATION.md) for installation, validation, rollback and next steps.
