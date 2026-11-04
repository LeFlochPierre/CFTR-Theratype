# Batch11 — Canonical Phase29A model capacity and response weighting

**Status:** draft; RTX6 synthetic suite and historical reference checks required
before merge. This patch does not rerun clinical or protected datasets.

## Goals

Phase29A was still importing `scripts/phase26h_auxiliary_corrector.py`
solely to load frozen TRAIN data and access the original fixed XGBoost
factory. Its regime weights, model-capacity variants, one-fold fitting and
response diagnostics also lived only in the historical script.

## Canonical modules

- `models/response_capacity.py` contains the original five fixed model
  configurations, responder regime labels (`<10` / `>40`), equal total
  TRAIN weight per regime, XGBoost capacity override, ExtraTrees imputation
  (median, indicator, `keep_empty_features=True`), fit/predict checks.
- `evaluation/response_capacity.py` contains the original position MAE,
  repeat-and-position cluster gain bootstrap, and per-variant repeat-mean
  aggregation, including the expected 210-variant guard.
- `scripts/phase29a_extreme_response_capacity.py` retains the original
  no-argument CLI and callable legacy helper wrappers. The script's run
  now directly invokes the canonical `datasets.eti_auxiliary.load_data(ROOT)`
  and `models.regressors.phase26d_model(SPEC)`; no numbered import.
  The unused historic `h` function argument in `make_model` and
  `fit_predict` is accepted for backwards compatibility.
- `tests/test_phase29a_capacity_batch11.py` performs **independent
  synthetic forward prediction comparison** of all five fixed models
  against a separate reference implementation restating the original
  Phase29A formula, not just its new wrappers. It verifies configuration,
  hyperparameters, deterministic cluster bootstrap and import boundaries.

## Scientific preservation

Retained exactly:
- Frozen outer 2×5 position-grouped TRAIN folds; 210 variants, 168 positions.
- Original 80-feature input plus a constant feature.
- XGBoost baseline `SPEC = (2, 5.0, 180)` with seed 26.
- Capacity override: depth 5, child weight 1, 350 trees, learning rate
  0.04, L2 penalty 1.0, subsample/colsample 0.9, seed 29.
- ExtraTrees 250 trees and its original median+indicator feature
  preprocessing, seed 29.
- Per-outer-TRAIN low/middle/high weights and negative/fail-fast checks.
- Frozen global and acute predictions as reference; report fields,
  hashes, seeds and output file names, including the global prediction
  exact-parity assertion (tolerance 1e-5).

The models and evaluation helpers have no filesystem access and never
load VAL or TEST outcomes. This patch does **not** touch Phase27F,
Phase26N or Phase26O locked validation/final-test scripts.

## Test in a clean RTX6 worktree

```bash
cd ~/projects/CFTR-Theratype-batch04-test
conda activate CFTR_ENV
git status --short
git fetch origin main refactor/phase29a-capacity-models-batch11
# Only switch when worktree is clean:
git switch --detach origin/refactor/phase29a-capacity-models-batch11

PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase29a_capacity_batch11.py \
  tests/test_phase28_30_canonical_batch10.py \
  tests/test_phase26h_loader_hub_batch08.py

PYTHONPATH="$PWD/src" python scripts/phase29a_extreme_response_capacity.py --self-test

PYTHONPATH="$PWD/src" python -m pytest -q -rs tests/
git diff --check origin/main...HEAD
```

Expected optional skip: ignored local `data/processed/bihler_2026_vti.parquet`
is missing from the isolated checkout. This means the real-data frozen
loader parity gate is **unverified**, not passed. Full frozen TRAIN output
parity must be checked in an authorised environment using preserved
historical outputs and without overwriting them.

## Remaining blockers to deleting scripts/

Phase28C still depends on the Phase27D3B structural provenance helper.
Phase27F and Phase26N/O contain locked-val/test pathways; do not migrate
or rerun these with exploratory scripts. Other historical CLI entrypoints
may still depend on numbered scripts; audit the import graph and
tracked/untracked local files before archival or deletion.
