# Stage 4: legacy callsite audit (read-only)

Stage 3 total-target frozen parity **passed on the workstation** (420 rows,
11 numeric columns, `atol=1e-6`). Stage 4 identifies how existing scripts
use legacy-phase APIs; it does **not** move, delete or rewrite any scripts.

Run the Stage 3 source checkpoint first:

```
python scripts/maintenance/stage3_source_checkpoint.py
```

Check the printed archive and JSON manifest are outside the repository.
The checkpoint excludes data, results and logs: preserve those separately.

Run the audit:

```
python -m pytest -q tests/test_stage4_callsite_audit.py
python scripts/maintenance/stage4_callsite_audit.py
```

Outputs under `results/maintenance/refactor_audit/stage4/`:
- `legacy_inventory.csv`: imported-by counts, symbols, conservative review tags
- `static_imports.csv`: import edges
- `used_symbols.csv`: legacy function/class accesses and source line numbers
- `dynamic_import_flags.csv`: some dynamic imports (not comprehensive)
- `report.json`: machine-readable overview

**No zero-import script is automatically safe to remove.** Some legacy scripts
are still standalone historical reproducibility assets, or invoked indirectly.
Any future migration must preserve the original source in a verified checkpoint,
then pass model and dataset parity against historical results.

Optional second independent parity check:

```
python -u -m cftr_theratype.cli benchmark --target corrector
python -m cftr_theratype.cli compare \
  --new results/maintenance/refactor_audit/stage3/full/phase26g_corrector_oof.csv \
  --reference results/biology/phase26g/phase26g_corrector_oof.csv
```
