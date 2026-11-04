# Batch09 — Remove downstream Phase27 script dependencies

**Status:** draft. This branch includes the test-only Batch08 fix from PR #12.
Merge PR #12 and rerun its test before Batch09; GitHub will then show only
the additional Batch09 changes. Do not cherry-pick unrelated scripts/results.

## Canonical reuse

- `features.mutation_codes.extract_codes` now owns the exact original
  Phase27D1 source selection, free-text mutation parsing and matching of
  parsed amino-acid positions against frozen TRAIN metadata. The legacy
  `phase27d1.extract_codes` API is a thin wrapper.
- Phase27D1 now reuses the already canonical `biology_annotations`
  domain, helix, contact proxy and amino-acid descriptor constants without
  duplicate definitions.
- Phase27D3B imports canonical `extract_codes`,
  `annotation_features`, `DOMAINS`, fixed `make_model` and
  `ap_gain_bootstrap`; no Phase27B1 or D1 import.
- Phase28A and Phase29B import canonical `extract_codes` and no longer
  require Phase27D1. Their historical self-test and CLI entry points remain.

This is TRAIN-only refactoring. No frozen splits, model hyperparameters,
random seeds, outputs, data files, reference structures or numerical
algorithms have been altered. **No locked Phase27F run or protected VAL/TEST
evaluation was performed.**

## RTX6 verification

```bash
cd ~/projects/CFTR-Theratype-batch04-test
conda activate CFTR_ENV
git status --short
git fetch origin main fix/batch08-pipeline-param-comparison refactor/phase27-downstream-imports-batch09
# Use only a clean worktree:
git switch --detach origin/refactor/phase27-downstream-imports-batch09

# PR12 test fix and Batch09 structural/mutation checks
PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase26h_loader_hub_batch08.py \
  tests/test_phase27_downstream_imports_batch09.py \
  tests/test_stage2_core_parity.py \
  tests/test_stage6_transfer.py

# Exact synthetic self-tests, no real-data/locked experiment runs:
for script in \
  phase27d1_biology_ablation \
  phase27d3b_structure_ablation \
  phase28a_trafficking_feature_audit \
  phase29b_predictability_ceiling_audit
do
  PYTHONPATH="$PWD/src" python "scripts/${script}.py" --self-test || break
done

PYTHONPATH="$PWD/src" python -m pytest -q -rs tests/
git diff --check origin/main...HEAD
```

The ignored real-data VTI loader parity test should still skip in a fresh
worktree. Do **not** mistake this for real-data scientific parity, and never
re-run or overwrite historical prediction/locked validation outputs just to
verify a module move.

## Before archiving numbered scripts

The original Phase27F locked-validation script and later Phase28/29/30
experiments still include dependencies on numbered entry points and
`h.previous.model` compatibility APIs. Move these only through a separate
reviewed batch with explicit original-vs-canonical TRAIN-only output parity.
