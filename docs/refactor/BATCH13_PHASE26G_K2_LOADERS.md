# Batch13 — Canonical Phase26G / Phase26K-2 data dependencies

**Status:** draft, RTX6 tests pending. No protected VTI/ETI outcomes
or historical result files accessed by this refactor.

## Canonical module changes

- `datasets/phenotype_training.py` owns the previously in-script
  `phase26k.load_allowed` logic, parameterised by `root` for reproducible
  file locations. It retains the original TRAIN-only Parquet predicate
  filters, 210 VTI TRAIN variants, 422 initial ETI TRAIN variants, 68
  protected VTI positions and 383 ETI records after position exclusion.
  It parses the original protein substitutions with the existing
  `features.mutation_codes` parser, and returns the same per-mutation
  grouped mean rescue targets. It does not open VAL/TEST assay outcomes.
- `features/phenotype_values.py` owns the unchanged Phase26K-2
  conservative numeric parser. Number/mean±SD text formats remain
  recognisable, while mixed values, formulas, unknown strings and
  insufficient evidence remain missing rather than being guessed.
- `phase26k_trafficking_preflight.py::load_allowed()` and
  `phase26k2_numeric_audit.py::load_allowed()` preserve zero-argument
  compatibility, each delegating to `datasets.phenotype_training`
  with the script's original `ROOT`. Phase26K-2 uses canonical free-text
  mutation helpers and retains its `parse_value` wrapper.
- `phase26g_feature_model_ablation.py` directly invokes
  `datasets.frozen_vti.load_train(ROOT)` rather than importing Phase26D
  for an already canonical loader. Original frozen folds, feature
  selection and grouped model comparisons are unchanged.
- `phase27e1_mixture_of_experts.py` drops its no-longer-needed
  `scripts/` sys.path mutation. This is an import hygiene change only.

## Synthetic regression checks

`tests/test_phase26g_k2_loader_batch13.py` checks:
- No numbered-script imports or script path mutations in the migrated
  Phase26G, Phase26K-2, Phase27E-1 entrypoints.
- Original exact phenotype numeric parsing and category labels.
- TRAIN-only Parquet predicates using monkeypatched in-memory data with
  intentionally supplied out-of-split rows; no protected outcome access.
- Exclusion of synthetic ETI TRAIN variants whose positions overlap the
  68 VTI protected positions, along with all 210/422/383 cardinality gates.
- Historical Phase26K and Phase26K-2 zero-argument loader compatibility.
- Fail-closed behaviour on unsupported mutation notation.

## RTX6 verification

```bash
cd ~/projects/CFTR-Theratype-batch04-test
conda activate CFTR_ENV
git status --short

# Prefer the PR ref, which still works if a feature branch is deleted.
git fetch origin main
git fetch origin pull/18/head:refs/remotes/origin/pr-18

# Only in a clean test worktree:
git switch --detach origin/pr-18

PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase26g_k2_loader_batch13.py \
  tests/test_phase27d3a_provenance_batch12.py \
  tests/test_phase29a_capacity_batch11.py

PYTHONPATH="$PWD/src" python -c \
'\''import sys; sys.path.insert(0, "scripts"); \
import phase26k2_numeric_audit as k2; k2.self_test()'\''
PYTHONPATH="$PWD/src" python scripts/phase27e1_mixture_of_experts.py --self-test

PYTHONPATH="$PWD/src" python -m pytest -q -rs tests/
git diff --check origin/main...HEAD
```

**Note:** Phase26K-2 historically has no CLI `--self-test`.
The command above calls its existing self-test directly without running
the full script. Do not execute the plain script from the clean worktree:
doing so would require the real Tedman workbook and TRAIN-only inputs.

As in earlier batches, a single skipped
`test_frozen_loader_parity.py` is expected in a clean worktree
without locally ignored clinical TRAIN data. This is *not*
scientific real-data parity. Do not overwrite archived OOF results.
