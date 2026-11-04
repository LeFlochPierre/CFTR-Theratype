# Refactor batch 03 — Phase31 assay helper extraction

**Status:** draft. No RTX6 tests or independent real-data parity runs have been
performed by the assistant. This is a non-destructive code-organisation change.

## Scope

- `datasets/assay_identifiers.py` owns the original Phase31I HGVS-to-substitution
  normalisation and strict position/control checks.
- `features/assay_substitutions.py` owns Phase31I.2's original 61-feature
  mutation-only representation. No assay label or outcome enters this feature set.
- `models/assay_teachers.py` owns Phase31I.2's original fixed
  imputer/scaler/ridge teacher constructor.
- Phase31I and Phase31I.2 keep their existing function names as legacy wrappers.
- Phase31J imports the canonical pure helpers directly instead of importing
  another numbered script solely to obtain those functions.
- `tests/test_phase31_assay_helpers.py` checks synthetic identifier alignment,
  invalid substitutions, exact legacy delegation, feature dimensions and model
  parameter identity.

**Not done:** Phase31I.2 still imports numbered Phase26D/G scripts and
Phase31I's assay table operations; Phase31J's teacher-audit workflow still uses
Phase31I.2's path/loading functions. Phase31K retains monkeypatch-dependent
legacy imports. Do not delete or archive these scripts yet.

## Test in a separate worktree on RTX6

```bash
cd ~/projects/CFTR-Theratype
git fetch origin refactor/phase31-assay-helpers-batch03
git worktree add --detach ../CFTR-Theratype-phase31-test \
    origin/refactor/phase31-assay-helpers-batch03
cd ../CFTR-Theratype-phase31-test
conda activate CFTR_ENV

PYTHONPATH="$PWD/src" python -m pytest -q \
    tests/test_phase31_assay_helpers.py \
    tests/test_phase26_modular_hubs.py \
    tests/test_phase31_shared_helpers.py \
    tests/test_stage2_core_parity.py \
    tests/test_stage3_legacy_parity.py

git diff --check
```

The standalone `scripts/test_phase31i2_repository.py`,
`scripts/test_phase31j_error_diagnostics.py` and
`scripts/test_phase31k_response_generalisation.py` also use synthetic fixture
data, but can be expensive due to XGBoost. Run them separately as a regression
gate; they mutate module globals for synthetic tests only.

These tests are necessary but not sufficient: before merging, verify preserved
TRAIN-only historical OOF outputs when a trustworthy independent reference is
available, and confirm no changes in filtering, splits, hyperparameters or seeds.
Never load or re-evaluate the protected VAL/TEST outcomes during refactoring.

The workstation's Phase33–39 source and ~2.4 GB of results are local/untracked
and are not represented by this PR; preserve verified backups before any later
moves, archive operations or deletion.
