# Batch10 — Phase28/30 canonical TRAIN-only dependencies

**Status:** draft. This change is intended to remove additional numbered-script
import edges while preserving experiment entry points, frozen models, data
segregation, and historical output checks. No scientific reruns were done.

## Scope

Four historical TRAIN-only analyses now call canonical implementations:

| Historical script | Replacement |
|---|---|
| `phase28c_structure_biophysics.py` | `datasets.eti_auxiliary.load_data(ROOT)`, `features.mutation_codes.extract_codes`, `models.regressors.phase26d_model` |
| `phase28e5_foldx_predictive_ablation.py` | same canonical inputs/model; no numbered imports remain |
| `phase30a_target_information_ladder.py` | canonical frozen TRAIN loader and exact XGBoost factory; no numbered imports remain |
| `phase30c_eti_incremental_ladder.py` | canonical frozen TRAIN loader and exact XGBoost factory; no numbered imports remain |

The frozen XGBoost spec is **`SPEC = (2, 5.0, 180)`** from
`models.auxiliary` and the estimator constructor is
`models.regressors.phase26d_model`. The old
`h.previous.model(h.SPEC)` call was a compatibility shim pointing to this
same constructor. Seeds, folds, XGBoost settings and feature matrices
are not changed.

Phase28C still calls the historical Phase27D3B structural-provenance
helper; that is an explicitly **remaining** dependency for a later batch.
The Phase27F/26N/26O locked VALIDATION/TEST scripts are untouched.

The internal `strict_load()` (Phase30A) and `load_matched_data()`
(Phase30C) return one fewer value because callers no longer receive an
imported `phase26h_auxiliary_corrector` module solely to construct a
model. Their command-line interfaces and model factory behaviour are
preserved; the functions are script-internal, not published package APIs.

## Verification on RTX6 (synthetic only)

```bash
cd ~/projects/CFTR-Theratype-batch04-test
conda activate CFTR_ENV
git status --short
git fetch origin main refactor/phase28-30-direct-canonical-dependencies-batch10
# Only on a clean worktree:
git switch --detach origin/refactor/phase28-30-direct-canonical-dependencies-batch10

PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase28_30_canonical_batch10.py \
  tests/test_phase26h_loader_hub_batch08.py \
  tests/test_phase27_downstream_imports_batch09.py

for script in \
  phase28c_structure_biophysics \
  phase28e5_foldx_predictive_ablation \
  phase30a_target_information_ladder \
  phase30c_eti_incremental_ladder
do
  PYTHONPATH="$PWD/src" python "scripts/${script}.py" --self-test || break
done

PYTHONPATH="$PWD/src" python -m pytest -q -rs tests/
git diff --check origin/main...HEAD
```

The maintained suite's optional real-data loader parity check is expected
to **skip** when the local ignored VTI parquet is absent, which does not
constitute real-data parity. None of the above commands should open real
TRAIN or protected VAL/TEST clinical datasets or overwrite experiment
results. Do not run no-flag scripts in this test worktree.

## Scientific parity and archive gate

For any eventual real-data verification, compare the original **TRAIN-only**
output identities, ordered predictions, model spec and fold membership
against preserved historical runs in a separate authorised environment.
Do not overwrite old results. The code retains explicit frozen OOF parity
guards (max absolute discrepancy <= 1e-5) before saving new reports.

Phase29A and other Phase27–30 dependencies remain; do not delete
`scripts/` until those workflows and all tracked/ignored local research
assets are accounted for.
