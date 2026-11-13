# CFTR-Theratype

A Python research codebase for CFTR *in-vitro* functional response, cross-assay transfer and forecasting/prediction reliability. Analyses remain **exploratory**: this repository does not provide a clinically validated genotype-to-treatment recommendation.

**Research status (November 2026): paused / frozen exploratory work.** Frozen development folds and historical evaluation claims must not be presented as independently validated clinical performance. Historical Phase26N, Phase26O and Phase27F validation/final-test execution history should be audited before making any new independence claim.

## What this branch contains

- `src/cftr_theratype/` — retained reusable Python package for assay datasets, features, models, evaluation and selected canonical experiments
- `tests/` — maintained synthetic and compatibility regression tests
- `scripts/phase*.py` — **retained legacy compatibility/research entry points**, some of which still contain unique experiment logic
- `scripts/maintenance.py`, `scripts/maintenance/` — limited maintenance and workstation audit utilities
- `pyproject.toml` — editable package installation and `cftr` / `cftr-data` entry points

For a smaller, readable default branch, previous `archive/legacy/scripts/` and `docs/` trees and old migration-only tests were **removed from the tracked main tree**. Their historical versions are retained in Git history and in the explicit preservation branch:
[`preserve/complete-history-before-prune-2026-11-11`](https://github.com/LeFlochPierre/CFTR-Theratype/tree/preserve/complete-history-before-prune-2026-11-11).

**Important:** that branch retains the tracked research code, original methods, old protocols and historical source snapshots. It is not an independent backup, and Git does not contain most ignored research files. Preserve important source, data, results, logs, ZIP patches and third-party assets outside the active working tree before any local disk cleanup. The workstation may also hold untracked later Phase33–39 experiments not represented in this repository.

## Quick start

From a verified CFTR environment:

```bash
conda activate CFTR_ENV
python -m pip install -e . --no-deps
cftr --help
cftr data validate
cftr data validate --eti
cftr transfer audit
```

The `cftr-data` CLI remains available. Some build/split commands can overwrite derived scientific assets; do not rerun them during cleanup.

## Reproducing TRAIN-only development benchmarks

These commands use stored TRAIN cohorts and historical frozen development definitions; output locations should be confirmed before running.

```bash
# Canonical benchmark commands (may be slow)
cftr benchmark run --target total
cftr benchmark run --target corrector

# Compare new and historical development OOF outputs without refitting
cftr benchmark compare \
  --new results/maintenance/refactor_audit/stage3/full/phase26g_total_oof.csv \
  --reference results/biology/phase26g/phase26g_total_oof.csv

# Auxiliary ETI-to-VTI transfer checks
cftr transfer audit
cftr transfer run --shuffles 0 --max-folds 1 \
  --outdir results/maintenance/refactor_audit/stage6/transfer_smoke
```

The Phase26G full repeated OOF contains 420 observations (210 variants × two repeats). Historical parity checks had been performed under the source workstation's original environment; absence of the ignored data in an isolated clone means real-data parity cannot be independently reverified there.

## Tests

```bash
PYTHONPATH="$PWD/src" python -m pytest -q -rs tests/
git diff --check
```

The optional real-data `tests/test_frozen_loader_parity.py` gate skips when `data/processed/bihler_2026_vti.parquet` is absent; a passing synthetic suite **does not replace** exact scientific parity checks against preserved real TRAIN-only inputs.

## Scientific and preservation caveats

- The supervised task is *in-vitro* CFTR rescue; these assays are not direct clinical efficacy measurements or a validated treatment-selection system.
- Frozen TRAIN folds are grouped by protein position; repeated cross-validation is not an independent patient cohort.
- Earlier historical scripts may have accessed validation or test outcomes. Do not claim that protected partitions were never viewed without reconstructing execution history.
- The source and historical experiment protocol remain accessible through the preservation branch, but archived entry points are **not expected to run by their old active paths on a lean checkout**.
- The tracked-code cleanup does **not** delete or make backups of `data/`, `results/`, `logs/`, local ZIPs, or Git-ignored/untracked sources. Verify independent backups and inventories before pruning the RTX6 research checkout.
