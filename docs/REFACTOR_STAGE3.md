# Stage 3: canonical benchmark, historical comparison and source checkpoint

Stage 1 and 2 are required. This package **adds files only**; it does not modify
any of the 105 legacy scripts and does not move or delete data/results.

## 1. Protect untracked work (do this before any destructive cleanup)

```bash
cd ~/projects/CFTR-Theratype
python scripts/maintenance/stage3_source_checkpoint.py
```

This produces a new source-only tar.gz outside the repository, plus a SHA256
manifest and Git status. It includes untracked `.py`, docs and text configs,
not datasets or outputs. It is an extra backup, **not** a substitute for full
backups of raw data and expensive computations.

## 2. Test the new benchmark runner

```bash
python -m pytest -q tests/test_stage3_benchmark.py
```

## 3. One-fold smoke test (cannot compare to the full historical OOF)

```bash
python -m cftr_theratype.cli benchmark --target total --fast --max-folds 1
```

## 4. Full reproduction (TRAIN-only; exact Phase26G settings)

```bash
python -m cftr_theratype.cli benchmark --target total
```

Full results saved only under `results/maintenance/refactor_audit/stage3/full/`.
Existing `results/biology/phase26g` is NEVER overwritten.

## 5. Compare exact frozen OOF predictions

```bash
python -m cftr_theratype.cli compare \
  --new results/maintenance/refactor_audit/stage3/full/phase26g_total_oof.csv \
  --reference results/biology/phase26g/phase26g_total_oof.csv
```

Comparison validates the exact `(repeat, fold, variant_key)` membership,
protein positions, observed TRAIN labels and every matching predictor column;
repeated OOF rows are not treated as independent clinical samples.

Note: The full model refit is exploratory TRAIN-only CV, and parity can be
sensitive to changed machine-learning library versions. If parity fails, do
not delete the legacy code; inspect the maximum difference and environment.

## Before archiving old scripts

- Snapshot entire untracked source and record checksums.
- Check full Phase26G parity on both corrector and total targets.
- Commit untracked scripts explicitly to a protected Git branch (review staged
  files; never run `git add -A` blindly in a data-heavy repo).
- Audit *all* old script imports and subprocess/CLI invocations.
- Only then migrate Phase26D/26G/26H legacy callers. This Stage 3 does **not**
  claim that old scripts are now redundant or safely deletable.
