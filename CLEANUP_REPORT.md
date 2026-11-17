# Final repository cleanup — 17 November 2026

Branch: `cleanup/final-repository`, based on `refactor/modular-scientific-hubs`. This cleanup does not refactor scientific functions or touch protected research assets.

## Verified deletions

Four tracked standalone Phase 31 synthetic test scripts were removed with `git rm` after migrating all their assertions into the maintained pytest suite:

| Deleted path | Bytes | Replacement test |
| --- | ---: | --- |
| `scripts/test_phase31i2.py` | 5,117 | `tests/test_phase31_assay_workflows_batch07.py::test_phase31i2_standalone_contract` |
| `scripts/test_phase31i2_repository.py` | 6,351 | `tests/test_phase31_integration.py::test_phase31i2_repository_end_to_end` |
| `scripts/test_phase31j_error_diagnostics.py` | 6,002 | `tests/test_phase31_integration.py::test_phase31j_end_to_end` |
| `scripts/test_phase31k_response_generalisation.py` | 6,015 | `tests/test_phase31_integration.py::test_phase31k_end_to_end` |

Gross removal: **4 files, 23,485 bytes**. The migrated tests add one pytest file and extend an existing test file, so source test coverage is retained rather than discarded. The versioned file count is **154 vs 156 at branch base** (net two fewer files). The versioned working-tree footprint is **1,181,562 vs 1,176,877 bytes** (net 4,685 bytes larger), because the exact checks and this report are retained.

No supported CLI, package import, active test, configuration or current README/research protocol invokes the four removed paths. Historical Stage 4–7 audit CSVs mention them as past inventory records and were preserved. No locked manifest lists these scripts. The detailed tracked review and deleted-file SHA-256 evidence are in `results/maintenance/cleanup_audit/final_2026_11_17/`.

## Validation

- Baseline before deletion: **348 tests passed**. After migration and deletion: **352 tests passed**.
- Phase 26N, 26O and 27F synthetic self-tests passed without locked scoring.
- Ten focused frozen-loader, Stage 2 and Stage 6 parity tests passed.
- `git diff --check` passed. Package discovery found seven packages; installed `cftr` and `cftr-data` entry points loaded and their `--help` commands passed.
- Stage 7 to cleanup dependency audit: top-level scripts **29 → 25**, scanned Python files **161 → 158**, static script imports **37 → 37**, imported-symbol occurrences **128 → 128**, dynamic flags and parse errors **0 → 0**.

## Retained history and unresolved lock risk

Phase 26N, 26O and 27F scripts, their manifests, frozen outputs and historical dependencies remain. The FoldX resume script and two distinct Boltz pose validators are retained as unique reproducibility paths. Other phase entry points and regression tests are retained or marked REVIEW in `tracked_file_review.csv`; static non-use alone was not grounds for deletion.

A read-only hash check found that the **existing** Phase 27F prediction manifest disagrees with six dependency-script SHA-256 values (`phase26h`, `phase27b1`, `phase27d1`, `phase27d3b`, `phase27e1`, `phase27e2`). Phase 26N and 26O script hashes match their manifests. This mismatch predates the cleanup and would make Phase 27F locked scoring refuse to run; no locked scoring was attempted. Restoring the exact historical dependency snapshot or documenting a new lock requires a separate reproducibility decision. This cleanup made no changes to those dependencies.

## Untracked local files

`all_file_inventory.csv` records 156 branch-base tracked paths, 55 visible untracked paths and 1,965 ignored local paths with sizes. `untracked_deletion_manifest.csv` records 160 selected local source, test, data, later-phase result and ZIP paths with exact sizes and SHA-256 hashes; `archive_contents.csv` lists 111 readable ZIP members. The manifests are under `results/maintenance/cleanup_audit/final_2026_11_17/` and are intentionally kept local with the ignored research results.

No untracked file was deleted. The only proposed removal after approval is the zero-byte `results/biology/phase27a/PMC13252174_supplementary.zip` (empty SHA-256, no members). The 386,138,112-byte `invalid_mendeley_download.zip` has a damaged central directory but contains a research filename in its payload, so it is **REVIEW/RETAIN**. Local datasets, ZIPs, documents, tools, results and Phase 33–39 modules/tests remain protected.
