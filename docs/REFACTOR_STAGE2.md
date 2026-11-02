# CFTR-Theratype Stage 2: shared core extraction (non-destructive)

This overlay **only adds new files**. It does not modify or remove legacy scripts, datasets or results. It has been generated from five user-supplied source files (Phase17A, Phase26D, Phase26G, Phase26H, Phase27D1), **not** from the full 47-file source ZIP. This is a transitional duplication of existing code until on-workstation parity and migration are complete.

## Extracted modules

- `cftr_theratype.evaluation.metrics`: position-weighted MAE, repeat-aware paired bootstrap used in Phase26G.
- `cftr_theratype.models.regressors`: exact Phase26D XGBoost, Phase26G pipelines, nested grouped selectors.
- `cftr_theratype.models.auxiliary`: position-wise label shuffling, weighted ETI joint-training, and inner selection; caller must enforce fold-specific auxiliary exclusions.
- `cftr_theratype.datasets.eti_auxiliary`: Phase26H train-only ETI loader, including global VTI protected-position exclusion, cross-study feature parity.
- `cftr_theratype.features.biology_annotations`: Phase27D1 UniProt sequence annotation features and proxy distances (not physical drug-pocket distances).

## After saving/unzipping under repository root

```
cd ~/projects/CFTR-Theratype
conda activate CFTR_ENV
python -m pytest -q tests/test_stage2_core_parity.py
python -u tests/test_stage2_eti_parity.py
python -u tests/test_frozen_loader_parity.py
```

Note: `test_stage2_eti_parity.py` loads only ETI and VTI TRAIN outcomes, but may be slower due to Stage 1 provenance checks. It compares *every* loaded dataframe and feature element with the original Phase26H implementation and checks fold identities.

## What not to do yet

- Do **not** delete, move, or rewrite any phase scripts; downstream callers still import them.
- Do **not** `git clean -fdx`, or run `git add -A` before reviewing external data, archives and credentials.
- Do **not** claim successful on-workstation parity until these checks pass.

## Next migration stage

After all three tests pass, migrate Phase26D/26G/26H call sites to canonical modules in a separate small commit, re-run frozen OOF checks and only then archive phase script entry points. Avoid changing fold memberships, XGBoost versions or hyperparameter search spaces during migration.
