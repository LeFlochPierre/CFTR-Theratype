# Batch07 — canonical Phase31 assay workflows

**Status:** draft, preservation-first. Local workstation validation is required
before merge. No VTI validation/test labels should be read or re-evaluated.

## What moved into `src/cftr_theratype/`

- `datasets/assay_tables.py` — the original strict auxiliary numeric
  parsing, assay-level amino-acid substitution deduplication (median),
  canonical table reports, cross-assay agreement, position-wise permutation
  negative control, SHA256 and unambiguous input resolution. The file resolver
  now accepts an explicit repository root.
- `datasets/assay_sources.py` — Phase31I.2 source manifest,
  frozen Bihler2024/2026 protected-protein-position masks, and seven
  auxiliary teacher source tables. The original filters and audit columns
  remain unchanged.
- `models/assay_teachers.py` — original Phase31I.2 inner grouped
  cross-fit teacher prediction and original, *distinct* Phase31I pilot Ridge
  pipeline (`keep_empty_features=True`).
- `scripts/phase31i_auxiliary_transfer.py` and
  `scripts/phase31i2_repository.py` retain backward-compatible experiment
  entrypoints and wrapper signatures, but shared implementation delegates to
  canonical modules.
- `scripts/phase31j_error_diagnostics.py` now uses **only
  `cftr_theratype` package imports** for VTI TRAIN loading and auxiliary
  source selection. It no longer imports Phase26D, Phase31I or Phase31I.2
  numbered scripts, and does not alter `sys.path` to find them.
- Standalone Phase31I.2/J synthetic tests use the canonical loaders and
  preserve their original output schema and leakage guards.

This does **not** delete or overwrite numbered scripts, datasets, results,
frozen splits or archived research. It also does not claim all functionality
in Phase31I/J is yet migrated into `src/`; their run-specific analysis and
reporting still live in the entrypoint scripts.

## RTX6 verification

Use a separate clean worktree; preserve local untracked Phase33–39 work.

```bash
cd ~/projects/CFTR-Theratype
conda activate CFTR_ENV
git fetch origin main refactor/phase31-assay-workflows-batch07
git worktree add --detach ../CFTR-Theratype-batch07-test \
  origin/refactor/phase31-assay-workflows-batch07
cd ../CFTR-Theratype-batch07-test

# Canonical assaying, contamination barriers, original wrappers
PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase31_assay_workflows_batch07.py \
  tests/test_phase31_assay_helpers.py \
  tests/test_phase31_decoupling_batch04.py \
  tests/test_phase31_shared_helpers.py

# Run both historical synthetic integration harnesses
PYTHONPATH="$PWD/src" python scripts/test_phase31i2_repository.py
PYTHONPATH="$PWD/src" python scripts/test_phase31j_error_diagnostics.py
PYTHONPATH="$PWD/src" python scripts/test_phase31i2.py

# The maintained suite, including the provisioned-only test skip
PYTHONPATH="$PWD/src" python -m pytest -q -rs tests/
git diff --check origin/main...HEAD
```

The standalone integration scripts run only on generated temporary data,
not on patient TRAIN/VAL/TEST outcomes. The full `tests/` suite includes a
real-data loader parity test that skips when its ignored data are missing.

## Science parity gate before archival

Use existing frozen **TRAIN-only** run outputs in the authorized workstation
rather than overwriting results. Check Phase31I.2 and Phase31J teacher table
contents, exact variant and position keys, split boundaries, teacher OOF
rows/columns, model hyperparameters, same seeds and numerical predictions.
The old scripts remain supported until end-to-end comparisons pass.

**Still needed before removing `scripts/`:** migrating remaining Phase26H
and Phase27–30 dependency hubs, extracting maintained experiment entrypoints
and maintenance commands into `src/`, archiving historical scripts and
relocating any tests that import script paths.
