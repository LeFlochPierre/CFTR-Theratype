# Stage 6: Canonical CLI, ETI transfer, and safe hub consolidation

Source snapshot: `cftr_stage6_modular_migration.zip`, Git HEAD `6698a9045276f29ddb4e2759a08dca501eead56d` on `refactor/cftr-modular-cleanup`.

## Scope (what this patch actually implements)

1. Installs a real `cftr` console entry point while retaining `cftr-data` and compatibility with Stage 3 `python -m cftr_theratype.cli benchmark ...` / `compare ...`.
2. Adds `data validate`, `benchmark run/compare`, `transfer audit/run/compare`, and `analysis report` commands. No unimplemented `--config` or non-existent YAML interfaces advertised.
3. Adds the canonical Phase26H ETI->VTI auxiliary learning and shuffled negative-control runner using existing parity-tested canonical loaders, shared model functions and frozen outer folds. Outputs are independent and never overwrite historical result CSVs.
4. Provides an **optional, reversible** hub migration that removes 15 duplicated helper implementations from four legacy scripts but preserves original source snapshots. All other functions and downstream module names remain in place.
5. Adds synthetic CLI, transfer, and hub-migration regression tests.

**Not yet implemented:** a complete archive/migration of all 29 remaining phase scripts, independent real-data Phase26H parity (requires running on your workstation), new biological models, or retrospective alteration of sealed VAL/TEST outcomes.

## Safe installation

Download ZIP and extract into a temporary folder (NOT directly over your working source):

```bash
cd ~/projects/CFTR-Theratype
conda activate CFTR_ENV
unzip -q cftr_stage6_modular_patch.zip -d /tmp/cftr_stage6_patch
python /tmp/cftr_stage6_patch/install_stage6.py --project "$PWD"  # dry run
python /tmp/cftr_stage6_patch/install_stage6.py --project "$PWD" --apply
python -m pip install -e . --no-deps    # preserve existing scientific dependency versions
cftr --help
```

The installer verifies original file checksums against the submitted source snapshot, refuses to overwrite unexpected local edits, and preserves existing replaced files in a timestamped directory outside the repository under `../CFTR-Theratype-backups/`.

## Validation order

```bash
python -m pytest -q \
  tests/test_stage6_cli.py tests/test_stage6_transfer.py tests/test_stage6_migration.py \
  tests/test_frozen_loader_parity.py tests/test_stage2_core_parity.py \
  tests/test_stage2_eti_parity.py tests/test_stage3_benchmark.py \
  tests/test_stage3_legacy_parity.py

cftr data validate --eti
cftr transfer audit
cftr analysis report

cftr benchmark compare \
  --new results/maintenance/refactor_audit/stage3/full/phase26g_total_oof.csv \
  --reference results/biology/phase26g/phase26g_total_oof.csv

cftr benchmark compare \
  --new results/maintenance/refactor_audit/stage3/full/phase26g_corrector_oof.csv \
  --reference results/biology/phase26g/phase26g_corrector_oof.csv
```

### Phase26H transfer parity (go/no-go)

```bash
cftr transfer run --shuffles 2
cftr transfer compare \
  --new results/maintenance/refactor_audit/stage6/transfer/phase26h_oof.csv \
  --reference results/biology/phase26h/phase26h_oof.csv
```

If historical Phase26H `phase26h_oof.csv` is absent, do **not** claim real-data parity. Compare the canonical one-fold results to an equivalent controlled execution of the original script or re-execute the original separately. Never overwrite original output files while doing so.

## Optional source consolidation (AFTER prior tests pass)

```bash
python scripts/maintenance/stage6_migrate_hubs.py plan
python scripts/maintenance/stage6_migrate_hubs.py apply
python -m pytest -q \
  tests/test_stage2_core_parity.py tests/test_stage2_eti_parity.py \
  tests/test_stage3_legacy_parity.py tests/test_stage6_transfer.py

# If any test or frozen OOF check fails:
python scripts/maintenance/stage6_migrate_hubs.py restore
```

Four legacy hubs touched: Phase26D, Phase26G, Phase26H, Phase27D1. Snapshot sources (exact bytes) go to `archive/legacy/hub_snapshots/stage6/`; commit them alongside the wrappers if the migration is retained. A `stage6_migration_manifest.json` records hashes of original and rewritten versions.

The migrated legacy APIs still import through their old script names (compatibility); the canonical production code is shared. This is intentional, as archived experiments may import these module names. It is a **dependency consolidation**, not permission to delete legacy modules.

## Checkpoint

Before committing, review `git diff --check`, `git diff --stat`, test outputs, and verify no private or restricted data was staged. Commit only after verifying frozen benchmark predictions and auxiliary transfer parity. Do not use `git clean` or force overwrite during cleanup.
