# Refactor batch 02 — canonicalise Phase 26 shared helpers

**Status:** draft, requires RTX6 regression tests and parity checks. No deletions, no archive moves, no data or result changes.

## Objective

The active codebase should import reusable scientific functionality from the `cftr_theratype` Python package, **never** from another numbered experiment script. This change is a first substantial cleanup of the heavily imported Phase26D/26G/26H and Phase27D1 hubs.

It reuses the canonical implementations introduced in prior Stages 1–6, rather than reimplementing scientific calculations. The original Phase26* source is retained in the Git history before this branch.

## Changed scripts

| Legacy script | Delegated functions | Canonical module |
|---|---|---|
| Phase26D | `pos_mae`, `model`, `select_spec`, `folds_from_historical_oof`, `load_train` | `evaluation.metrics`, `models.regressors`, `evaluation.folds`, `datasets.frozen_vti` |
| Phase26G | `make_model`, `position_mae`, `choose`, `paired_ci` | `models.regressors`, `evaluation.metrics` |
| Phase26H | `parsed_position`, `load_data`, `positional_shuffle`, `fit_joint`, `choose_weight` | `datasets.eti_auxiliary`, `models.auxiliary` |
| Phase27D1 | `annotation_features` | `features.biology_annotations` |

The historical functions keep their names and call signatures as compatibility wrappers. Local module variables such as `ROOT`, `PARAMS`, `WEIGHTS`, `SPEC` and historical experiment entry points are retained.

The following downstream experiments now import canonical Phase26 functions directly:
- `scripts/phase31l_predictability.py`
- `scripts/phase31m_three_tracks.py`
- `scripts/phase32a_adaptive_theratyping.py`

These are not complete rewrites of the phase scripts; experiment-specific procedures are intentionally left unchanged.

## Important limitations

- Historical Phase26G/26H entry points still import Phase26D for compatibility with existing synthetic parity tests and external callers. These *remaining* legacy dependencies will be removed separately after the monkeypatch-based tests have been moved to canonical injection points.
- Phase31I/J/K and other experiments also continue to import legacy phase scripts. Do not archive or delete their imported modules yet.
- This branch does not contain the workstation's untracked Phase33–39 files. They **must be backed up** before broad reorganisations.
- `stage6_migrate_hubs.py plan/apply` expects the **original** pre-migration source checksums and is **not** intended to run on this already-migrated branch. It remains in the repository to document and revert the historical migration procedure. Revert this Git branch's changes if this batch does not pass parity.
- The exact historical train/validation/test outcome access must still be reconciled before claiming sealed prospective evaluation.

## Test on RTX6 in a separate Git worktree

Do not reset or clean the existing workstation checkout.

```bash
cd ~/projects/CFTR-Theratype
conda activate CFTR_ENV

git fetch origin refactor/phase26-hubs-batch02
git worktree add --detach ../CFTR-Theratype-phase26-test origin/refactor/phase26-hubs-batch02
cd ../CFTR-Theratype-phase26-test

PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase26_modular_hubs.py \
  tests/test_stage2_core_parity.py \
  tests/test_stage2_eti_parity.py \
  tests/test_stage3_legacy_parity.py \
  tests/test_stage6_transfer.py \
  tests/test_phase31_shared_helpers.py \
  tests/test_stage6_migration.py

git diff --check
```

This worktree intentionally contains no ignored local raw data or results. The listed tests are synthetic, not real-data validation. The Stage2/Stage3/Stage6 tests compare canonical and historical interfaces but cannot alone demonstrate numerical reproducibility of every archived experiment.

After all synthetic tests pass, perform **a separately reviewed real-data parity run** against the previously frozen TRAIN-only output CSVs without overwriting those reference files. In particular:
- Phase26G `total` and `corrector` grouped OOF predictions versus preserved historical CSVs (`atol=1e-6`).
- Phase26H ETI→VTI auxiliary transfer OOF versus the original independent Phase26H output, **only if that file is present and its provenance is verified**.
- Do not load or rerun locked validation/test outcomes.

## Subsequent batches

1. Relocate Phase26G/H/31I/J/K remaining import dependencies into explicitly parameterised canonical functions; adjust tests to patch canonical injection points.
2. Extract shared assay harmonisation, identifiers, cross-fitting and teacher helpers from Phase31I scripts.
3. Preserve and inventory local-only Phase33–39 code and results.
4. Finally archive remaining numbered experiment entry points with a legacy runner and complete history index.
5. Condense repository documentation, retain research conclusions and tag a reproducible research freeze.

**No files should be deleted merely to make the root listing shorter.**
