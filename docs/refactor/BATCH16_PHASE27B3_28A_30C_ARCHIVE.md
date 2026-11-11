# Batch16 — Preserve original Phase27B3 / Phase28A / Phase30C analyses

**Status:** draft; requires RTX6 verification before merge.

## Scope

Three active TRAIN-only historical entry points are moved unchanged to
`archive/legacy/scripts/`. Each original and archived Git blob ID
was verified identical **before deleting the original active path**.

| Original path in `scripts/` | SHA-1 of exact source Git blob |
|---|---|
| `phase27b3_threeclass.py` | `afcec504250456e3458f931831e3333f436de313` |
| `phase28a_trafficking_feature_audit.py` | `4e3e1ae3603691ad26c100495d14335c35e0ee2d` |
| `phase30c_eti_incremental_ladder.py` | `0f1a2d0d345fe1b40bf2cd84a2b1462e6364bced` |

These are *not* claimed to be fully reimplemented in `src/`.
Their full unique scientific functions, CLI, output schemas and
TRAIN-only split restrictions remain in the archive. No benchmark
results are regenerated in this branch.

## Dependency audit and compatibility

Existing AST/import-regression tests now inspect these archived files,
while keeping the original import-boundary assertions.

The archived `phase27b3b_specialist_combination.py` still imports
`phase27b3_threeclass.evaluate`, and archived Phase28B still
imports `phase28a_trafficking_feature_audit` helpers. The existing
`scripts/run_legacy.py` includes the archived directory on the
Python module path when running an archived script. New synthetic
tests verify the original Phase27B3 import and the archived Phase28B
self-test. All three migrated scripts have their own `--self-test`,
also exercised by the new tests.

No active phase-script importer of these three entry points was
identified in the remote code audit. The external RTX6 workstation
may contain additional untracked scripts: this is not a proof that
those have no old-path references.

## RTX6 validation

```bash
cd ~/projects/CFTR-Theratype-batch04-test
conda activate CFTR_ENV
git status --short
git fetch origin main
git fetch origin pull/22/head:refs/remotes/origin/pr-22

# Only if the test worktree is clean:
git switch --detach origin/pr-22

PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase27b3_28a_30c_archive_batch16.py \
  tests/test_phase26h_loader_hub_batch08.py \
  tests/test_phase27_downstream_imports_batch09.py \
  tests/test_phase28_30_canonical_batch10.py

PYTHONPATH="$PWD/src" python -m pytest -q -rs tests/
git diff --check origin/main...HEAD
git diff --summary origin/main...HEAD
```

The tested Batch15 baseline is 259 passed, one expected skipped
`test_frozen_loader_parity.py`, with the protected real-data TRAIN
parquet absent from the isolated worktree. Eight new Batch16 tests
should be collected; the actual result is unknown until RTX6 runs.

## Keep protected

- `scripts/phase26n_locked_validation.py`,
  `scripts/phase26o_final_test.py`, and
  `scripts/phase27f_locked_validation.py` are untouched.
- `data/`, `results/`, `logs/`, checkpoints and historical
  frozen OOF predictions are untouched.
- The main RTX6 checkout and its ignored/untracked Phase33–39
  research files must not be cleaned by these branch operations.
- Do not run these historical scripts without `--self-test`
  unless frozen output locations and source inputs have been
  independently backed up. Those runs can overwrite historical CSVs.

The target is an actively maintained `src/` package with a small
`scripts/` directory and an intact, reproducible historical archive.
