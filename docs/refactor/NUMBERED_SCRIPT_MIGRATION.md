# Numbered-script dependency removal (incremental, preservation-first)

**Status:** active, non-destructive refactor. This document describes a migration strategy, **not a claim that all scripts have been migrated**.

## Goal

A future CFTR-Theratype developer should import reusable scientific functions from `cftr_theratype`, not from `scripts/phaseXX*.py`. Keep one canonical implementation per function, with explicit stable inputs and outputs. Ultimately `scripts/` should contain command-line launchers and maintenance utilities, while historical numbered experiments live in `archive/legacy/scripts/`.

Avoid a monolithic `utils.py`: place functionality in `datasets/`, `features/`, `models/`, `evaluation/`, and `experiments/` by responsibility.

## Starting state (GitHub main, November 2026)

- 29 numbered `scripts/phase*.py` remain active; Stage 8 found 18 imported by other code.
- The previous Stages 1–6 already extracted canonical Phase26D/26G/26H functions into `datasets/`, `models/`, and `evaluation/`. Stage 6 also provides an **optional** reversible migration of 15 duplicated helpers in four legacy hubs. Do not assume this optional migration has been applied on RTX6.
- `archive/legacy/scripts/` contains 68 earlier experiments.
- Phase 33–39 source code and several late protocols are **untracked on the workstation**; preserve them before cleanup. Do not claim GitHub represents the complete research history.
- Historical scripts often assume execution from the repository root and import other top-level numbered modules. Some tests monkeypatch globals such as `ROOT`. Retain compatibility while moving implementation.

## Batch 01: Phase31J / Phase31L decoupling (this branch)

1. `features/regions.py`: the *original approximate* Phase31J domain intervals, optional CSV domain-map validation, and position-to-domain function.
2. `evaluation/historical_oof.py`: strict verification and loading of the already-frozen Phase26G/31I M0 OOF outputs (TRAIN only).
3. `scripts/phase31j_error_diagnostics.py`: re-exports region API and delegates OOF functions to canonical modules, retaining historical signatures and `ROOT` override.
4. `scripts/phase31l_predictability.py`: imports the two shared modules rather than depending on Phase31J for those operations.
5. `tests/test_phase31_shared_helpers.py`: synthetic test of defaults, invalid maps, exact legacy API parity, historical OOF alignment, and corruption rejection.

**Not yet done:** Phase31L still uses Phase26D/26G directly; other Phase31 tools still import numbered scripts. We have not reproduced real RTX6 predictions in this branch.

## Suggested further batches

1. **Foundational Phase26 hubs.** Apply/evaluate the previously committed Stage 6 wrapper migration (after its exact-hash and backup gates) and update direct dependents to import `datasets.frozen_vti`, `models.regressors`, `models.auxiliary`, and `evaluation.metrics`. Avoid changing train/test filtering, random seeds, fold membership, or output formats.
2. **Phase31 assay and teacher tools.** Extract shared mutation identifiers, assay-table provenance, cross-fitting and teacher constructors from `phase31i_auxiliary_transfer.py` and `phase31i2_repository.py` into small modules. Move functions only when exercised by real callers and tested against frozen outputs.
3. **Later diagnostic and learning scripts.** Move prediction diagnostics, bootstrap summaries and experimental-budget simulation into `evaluation/` or `experiments/` as appropriate. Preserve the original Phase31J/K/L/M and Phase32A experiment commands as thin wrappers or frozen historical entry points.
4. **Local-only Phase33–39.** Back up original source and result files, inspect for overlap and reuse, and decide which research functions are sufficiently general to keep in `src`. Keep exploratory experiments historical rather than converting them all to supported public APIs.
5. **Archive phase entry points.** Only when no active imported dependency remains, move old entry points with `git mv` and preserve their byte-identical originals/snapshots. Adjust `run_legacy.py` and historical docs if needed; verify both synthetic tests and real parity.
6. **Documentation consolidation.** Provide a concise project README, research-status and results-ledger docs, provenance/reproduction docs, and a history index. Archive rather than silently deleting original protocols.

## Testing and migration gates

For **every** code migration:
- Baseline `git status --short` and current source commit recorded.
- Protected data access stays restricted (TRAIN only where originally specified); do not rerun locked VAL/TEST.
- Synthetic tests prove equivalent outputs, mutation IDs, folds, label filtering, hash checks and error behaviour.
- On RTX6, rerun the relevant frozen OOF comparison when both datasets and reference outputs are available, with unchanged metrics and predictions.
- Run `python -m pytest -q` and `git diff --check` before merging.
- If a historical script is imported by later code, **do not archive or delete it yet**.
- All untracked Phase33–39 modules, relevant notebooks/notes, root ZIPs, data and 2.4GB of results need verified backup before any destructive local cleanup.

## Testing Batch 01 in a separate worktree

No checkout or overwrite of the active workstation tree is required:

```bash
cd ~/projects/CFTR-Theratype
git fetch origin refactor/decouple-numbered-scripts-01
git worktree add --detach ../CFTR-Theratype-refactor-test \
  origin/refactor/decouple-numbered-scripts-01
cd ../CFTR-Theratype-refactor-test
conda activate CFTR_ENV
PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase31_shared_helpers.py \
  tests/test_stage2_core_parity.py \
  tests/test_stage3_legacy_parity.py
```

A separate worktree has no ignored local datasets or `results/` files. These are **synthetic tests**. Scientific parity using historical real-data files must be performed separately from the live workstation after backup and review.
