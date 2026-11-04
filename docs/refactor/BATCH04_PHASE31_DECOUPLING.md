# Refactor batch 04 — decouple Phase31I.2/K/M/32A numbered imports

**Status:** draft. The assistant has not executed workstation tests or historical
real-data parity. Do not merge before the user verifies tests and outputs.

## Changes

- Phase31I.2 now uses `datasets.frozen_vti.load_train(root)` and
  `models.regressors.phase26g_model / PHASE26G_PARAMS`, not Phase26D/26G scripts.
- Phase31K uses the shared TRAIN-only VTI loader, verified historical M0 OOF
  loader and canonical Phase26G model factory. This removes its imports from
  Phase26D, Phase26G and Phase31J.
- Phase31M directly uses canonical assay identifiers and 61-dimensional teacher
  features, not Phase31I/31I.2 script helpers.
- The shared Phase31M/Phase32A VTI TRAIN input alignment and ESMRank score
  validation now live in `experiments.three_tracks_inputs`, with Phase31M
  compatibility wrappers.
- Phase32A reads ESMRank inputs directly from the canonical package instead of
  importing Phase31M. Existing filename conventions and output paths are unchanged.
- The standalone Phase31I.2 and Phase31K synthetic integration fixtures now
  patch the *canonical* loader/model functions rather than numbered scripts.
- Added synthetic alignment, ESMRank validation and static import tests.

## Tests on RTX6

```bash
cd ~/projects/CFTR-Theratype
git fetch origin refactor/remove-phase26-script-imports-batch04
git worktree add --detach ../CFTR-Theratype-batch04-test \
  origin/refactor/remove-phase26-script-imports-batch04
cd ../CFTR-Theratype-batch04-test
conda activate CFTR_ENV
PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase31_decoupling_batch04.py \
  tests/test_phase31_assay_helpers.py \
  tests/test_phase26_modular_hubs.py \
  tests/test_phase31_shared_helpers.py \
  tests/test_stage2_core_parity.py \
  tests/test_stage3_legacy_parity.py

PYTHONPATH="$PWD/src" python scripts/test_phase31i2_repository.py
PYTHONPATH="$PWD/src" python scripts/test_phase31k_response_generalisation.py
PYTHONPATH="$PWD/src" python scripts/test_phase31j_error_diagnostics.py

PYTHONPATH="$PWD/src" python -m pytest -q tests/
git diff --check origin/main...HEAD
```

Standalone integration tests use synthetic data and in-memory monkeypatching.
They may be slower due to XGBoost, but must not touch protected outcomes.
Do **not** run locked Phase26N/O or Phase27F validation/test workflows as
refactor validation.

## Verification gate

- Compare preserved, independently produced **TRAIN-only** Phase31I.2, K, M
  predictions with their frozen files, checking variant keys, positions,
  fold/repeat identities and numerical parity. Never overwrite historical files.
- Ensure the CFTR_ENV dependency versions match historical runs if exact
  floating-point parity is expected.
- Keep the workstation's untracked Phase33–39 code and output archives backed up.

## Remaining work before removing scripts/

Phase31I.2 and Phase31J still depend on Phase31I/31I.2 assay-table workflows.
Earlier Phase26–30 and Phase27F contain many cross-phase imports. Existing
test suites and the legacy launcher still reference numbered files.
Do not delete the directory, large results, or source ZIPs at this stage.
