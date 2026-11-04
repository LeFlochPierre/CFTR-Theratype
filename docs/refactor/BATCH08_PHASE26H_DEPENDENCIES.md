# Batch08 — Remove Phase26H as a dependency hub

**Status:** draft, workstation verification required; no numbered scripts deleted.

## Canonical implementations now in src

- `models/responder_classifiers.py` — original Phase27B fixed-design
  responder classifiers (logistic_seq68, xgb_seq80), exact hyperparameters,
  prevalence-dependent class weighting and task labels.
- `evaluation/responder_classification.py` — responder AUROC, AP, Brier,
  stable top-20% and position-cluster AP bootstrap, including original
  numerical and failure behaviour.
- `features/mutation_codes.py` — original Phase26K conservative
  free-text missense parser and position parser, **distinct from**
  `datasets.assay_identifiers.norm_aa` (which requires an exact ID).

## Import graph changes

- Phase26H no longer imports Phase26D. Its `position_mae`, SPEC and
  WEIGHTS are canonical `src` references. A temporary
  `h.previous.model(h.SPEC)` compatibility object resolves to the canonical
  `phase26d_model`; historical Phase27F/28C/28E5/29A/30A/30C callers
  therefore still work without a numbered Phase26D import.
- Phase27B1/B3, Phase27D3B, Phase27E1, Phase28A, Phase29B now directly
  import `datasets.eti_auxiliary.load_data(ROOT)` instead of Phase26H.
- Phase27D1 also uses canonical classifier and AP bootstrap functions,
  and the canonical missense parser instead of Phase27B1/Phase26K.
- Original command-line experiment files remain in `scripts/` for
  reproducibility and backwards-compatible usage. This is *not* a rewrite
  of historical science or its results.

## RTX6 verification in a clean worktree

```bash
cd ~/projects/CFTR-Theratype
conda activate CFTR_ENV
git fetch origin main refactor/phase26h-loader-dependencies-batch08
git worktree add --detach ../CFTR-Theratype-batch08-test \
  origin/refactor/phase26h-loader-dependencies-batch08
cd ../CFTR-Theratype-batch08-test

PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase26h_loader_hub_batch08.py \
  tests/test_phase26_modular_hubs.py \
  tests/test_stage2_core_parity.py \
  tests/test_stage6_transfer.py \
  tests/test_stage6_migration.py

# Synthetic-only self-tests; do not invoke their no-flag 'run' commands here.
PYTHONPATH="$PWD/src" python scripts/phase27b1_extreme_classifiers.py --self-test
PYTHONPATH="$PWD/src" python scripts/phase27b3_threeclass.py --self-test
PYTHONPATH="$PWD/src" python scripts/phase27d1_biology_ablation.py --self-test
PYTHONPATH="$PWD/src" python scripts/phase27d3b_structure_ablation.py --self-test
PYTHONPATH="$PWD/src" python scripts/phase27e1_mixture_of_experts.py --self-test
PYTHONPATH="$PWD/src" python scripts/phase28a_trafficking_feature_audit.py --self-test
PYTHONPATH="$PWD/src" python scripts/phase29b_predictability_ceiling_audit.py --self-test

# Full maintained synthetic suite (real-data parity skipped when unavailable).
PYTHONPATH="$PWD/src" python -m pytest -q -rs tests/
git diff --check origin/main...HEAD
```

Tests here do not run protected VAL/TEST or retrain a full historical
experiment. The real-data Phase26D frozen loader parity, historical
Phase26H/27B/27D1 OOF and metric parity still need independent verification
on an authorised TRAIN-only workstation before any later archive/deletion.

## Still pending before removing scripts

Phase27D3B/27F and Phase28C/28E5/29A/30A/30C still have other numbered
dependencies, plus `h.previous.model` compatibility calls. Their workflows,
frozen outputs and script-local tests must be migrated separately.
Phase33–39 local untracked scripts/results must be securely backed up.
