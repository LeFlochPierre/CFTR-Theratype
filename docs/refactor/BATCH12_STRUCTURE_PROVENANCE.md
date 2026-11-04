# Batch12 — Shared Phase27D-3A PDB provenance validation

**Draft and awaiting RTX6 validation.** Stacked on PR #16
(`fix/batch11-numerical-prediction-parity`), which corrects the false
bitwise-equality failure for independent repeated Phase29A model fits.
Merge PR #16 first. No historical experiment or locked-validation output
has been generated or modified by this patch.

## Changes
- `features/structure_provenance.py` now owns the unchanged fixed
  `SPECS` fields for the 8EIG/7SV7 pocket-proximity annotations.
  Its `verify_provenance(provenance_path, root, specs=SPECS)` verifies
  no duplicate PDB IDs, exact expected PDB sets, canonical agreement
  ≥0.99, the one recorded canonical mismatch involving residue 1371,
  and SHA256 of both local source PDB files. It refuses missing files
  or changed hashes.
- `scripts/phase27d3b_structure_ablation.py` preserves its public
  no-argument `verify_provenance()` as a wrapper, with historical
  `ROOT` and `PROVENANCE` constants and the same local input sources.
- `scripts/phase28c_structure_biophysics.py` calls that canonical
  validator directly. Phase28C no longer imports Phase27D3B or
  modifies `sys.path` to find numbered scripts.
- `tests/test_phase27d3a_provenance_batch12.py` verifies parity
  against the legacy wrapper using synthetic files, plus negative
  cases for duplicate/changed PDB identity, poor agreement, unexpected
  mismatch, missing residue 1371, altered hashes and absent PDBs.

## RTX6 verification

```bash
cd ~/projects/CFTR-Theratype-batch04-test
conda activate CFTR_ENV
git status --short
git fetch origin main refactor/phase27d3a-structure-provenance-batch12
# Only on a clean test worktree:
git switch --detach origin/refactor/phase27d3a-structure-provenance-batch12

PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase27d3a_provenance_batch12.py \
  tests/test_phase29a_capacity_batch11.py \
  tests/test_phase28_30_canonical_batch10.py

PYTHONPATH="$PWD/src" python scripts/phase27d3b_structure_ablation.py --self-test
PYTHONPATH="$PWD/src" python scripts/phase28c_structure_biophysics.py --self-test

PYTHONPATH="$PWD/src" python -m pytest -q -rs tests/
git diff --check origin/main...HEAD
```

An isolated worktree is expected to **skip the optional ignored-data
Phase26D parity test**; this is not verification on protected TRAIN data.
Do not run no-flag historical analysis scripts in this worktree or
overwrite frozen OOF/provenance data.

## Remaining
Many legacy scripts still remain as reproduce-only entrypoints.
Phase27F and Phase26N/O handle locked VAL/TEST workflows and must remain
separate. A full scripts/ deletion requires an inventory, preserved
historical outputs, and explicit scientific parity checks with original
data in an authorised environment.
