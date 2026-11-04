# CFTR-Theratype: preservation-first repository freeze and cleanup

**Status:** planning and read-only local audit. **No removals or migrations are authorised by this document.**

## Purpose

Preserve all CFTR-Theratype research and negative results, document what can and cannot be reproduced, and leave a small maintainable scientific package that can be reopened when useful new datasets become available. Do not represent exploratory model comparisons as prospective clinical validation.

## Verified remote baseline (4 November 2026)

- GitHub: private repository `LeFlochPierre/CFTR-Theratype`.
- `main`: `404847690856ee6f5361941b2de3d9866ea7a08e`, a merge commit dated 2 November 2026.
- `refactor/cftr-modular-cleanup`: `857883e84e707ada63b20abadc0de20fa4781a04`, **already merged into `main`**. Do not repeat Stage 1–6 indiscriminately.
- GitHub `main` has 237 tracked files (68 archived historical script files, 45 files in `scripts/`, 48 in `src/`, 40 in `tests/`, 28 in `docs/`). It already contains `datasets/`, `models/`, `evaluation/`, `experiments/` and a `cftr` CLI.
- `data/raw`, `data/interim`, `data/processed`, `results` and `logs` are Git-ignored, so GitHub is **not** a backup of the inputs, outputs or run history.
- No tracked file path on `main` contains `phase33` through `phase39`. User workstation listing includes later-phase ZIPs, including Phase 39E; **their contents, actual installation status, and independent backups are not established by the GitHub audit**.
- There are several tracked backup and duplicate-name files (for example `.bak`, `.before_duplicate_fix`, and `(1)/(2)` copies). Review before removing; committed provenance is not evidence they are safe to delete now.

## Step 0: identify local truth (read-only)

On RTX6, without switching branches or pulling first:

```bash
cd ~/projects/CFTR-Theratype
conda activate CFTR_ENV

git status --short
git branch -vv
git remote -v
git log -1 --format='%H %ci %s'
du -sh data results logs archive scripts src tests docs 2>/dev/null

python scripts/maintenance/stage7_local_inventory.py
# Optional: hash all root ZIPs; this can be slower for large third-party software archives
python scripts/maintenance/stage7_local_inventory.py --hash-zips
```

Outputs: `results/maintenance/cleanup_audit/<UTC timestamp>/` with root ZIP inventory, tracked/untracked source-file inventory, `git_status.txt` and `summary.json`. **This script does not delete, move or extract files.** It does not recursively scan large `data/`, `results/` or `logs/`.

If the script is not yet in your workstation checkout, fetch the new audit branch *after checking your local status* (do not reset or overwrite local changes), or copy the script from GitHub as a standalone file.

## Preservation gates before any deletion

1. **Exact working state**: save the current commit, branch, `git status`, Git remotes (do not post tokens), installed dependency information, and local-file inventory. Local work might be ahead of GitHub or contain untracked source modifications.
2. **Historical source**: check whether Phase 33–39 ZIP contents include source, protocols or test files absent from GitHub. Identify whether the last executed source files match the archived ZIP contents (hash/provenance). Do not assume the ZIP is authoritative simply because it is newest.
3. **Scientific outputs**: preserve phase-wise CSV, JSON, figures, logs, and frozen OOF predictions from `results/` independently of Git history. Build a result manifest linking results to exact source/inputs when possible.
4. **Datasets**: store raw/derived data and reference material outside the lightweight code repository with a documented manifest, SHA256 sums, provenance, licensing and access restrictions. Do not commit data or proprietary FoldX/third-party software archives without explicit rights.
5. **Validation/test usage**: reconcile historical `scripts/phase26n_locked_validation.py`, `scripts/phase26o_final_test.py`, and subsequent evaluation logs. Existing README states VAL/TEST are sealed; this is an *unverified historical claim*, and previous run discussions reported at least some validation/test analysis. Distinguish original frozen split definition, actual outcome access, and truly independent future validation.
6. **Backups**: verify copied archives and data/results are readable, ideally in at least two locations, before removing redundant local files. No `rm -rf`, `git clean`, or reinitialisation.

## Proposed cleaned logical structure

```text
CFTR-Theratype/
  README.md                       # concise, honest frozen-project landing page
  pyproject.toml                  # package and CLI
  environment.yml                # tested environment specification (after audit)
  src/cftr_theratype/
    datasets/                     # loaders and split guards
    features/                     # mutation, structure, PLM features
    models/                       # reusable predictors and adaptation
    evaluation/                   # grouped folds, metrics, parity checks
    experiments/                  # canonical reproducible entry points
  scripts/
    maintenance/                  # reversible preservation and migration tools
    run_legacy.py                 # archived script compatibility
  configs/                        # frozen configs, only if actually used
  tests/                          # unit, smoke and parity tests
  docs/
    RESEARCH_STATUS.md            # paused project; scientific conclusions
    RESULTS_LEDGER.md             # results + exact provenance, no invented metrics
    DATA_PROVENANCE.md            # origins, checksums, licenses
    REPRODUCIBILITY.md            # reproducible commands and limitations
    protocols/                    # historical approved protocols, if desired
  archive/legacy/                 # immutable legacy scripts/snapshots
  data/                           # local/ignored; manifest files may be tracked
  results/                        # local/ignored
```

This is a **target**, not instructions to rename existing directories wholesale. Existing Stage 6 paths should be reused.

## Cleanup phases, in order

### A. Stop the root clutter (after preservation)

- Classify each root ZIP as historical patch, external dataset, proprietary software, or duplicated artifact. Keep a manifest of old/new location and SHA256.
- Keep external archives (for example `foldx_Linux_1.zip`) outside the source repository and out of Git; confirm redistribution rights.
- Move old patches to a **sibling, backed-up artifacts directory** rather than deleting them outright. Use copy + hash verification before removing the root original.
- Decide whether root `phase31a_*.csv`, `README_PHASE35.md`, `RECAP.md`, and installation notes are current research assets, historical documentation or redundant copies, after cross-checking referenced paths and contents.

### B. Separate current public interface from legacy experiment code

- Preserve CLI compatibility; define `cftr data`, `cftr benchmark`, `cftr transfer`, and `cftr analysis` as the supported minimal entry points until confirmed otherwise.
- Consolidate imports **only after** running baseline tests and checking historical OOF parity.
- Stage 6 offers an optional `stage6_migrate_hubs.py plan/apply/restore` for 15 helper implementations. Verify whether it was applied on RTX6 before repeating it.
- Do not move `scripts/phase*.py` or archived scripts based solely on a static zero-import scan. Legacy scripts may be launched by shell wrappers or documentation.

### C. Establish a reproducible frozen checkpoint

- Run the tests already present, without reprocessing the input data.
- Reproduce a representative canonical genotype-only Phase 26G benchmark and a Phase 26H transfer result against frozen historical OOF **if corresponding reference outputs actually exist**.
- Add source/manifest checksums to any documented reproduced result.
- Do **not** re-run `phase26n`/`phase26o` as part of a cleanup, and do not describe protected partitions as never observed without reconciling the history.

### D. Research documentation and final release

- `README.md`: clearly say project paused; not a validated clinical predictor; link the complete Overleaf-style archive and numerical ledger.
- `docs/RESEARCH_STATUS.md`: supported findings, unsuccessful approaches, limitations, reuse conditions.
- `docs/RESULTS_LEDGER.md`: output path, endpoint, cohort, folds, units, model, metric and provenance for each important result. Indicate missing files as missing.
- Run `git diff --check`, tests, and storage/security checks; tag a frozen checkpoint only after verification.

## Acceptance criteria

- No scientific or data artifact lost; local ZIPs and results backed up with hash manifests.
- `git status` and source provenance documented; active package `pip install -e . --no-deps`, `cftr --help`, and synthetic tests continue working.
- Frozen Phase 26G and Phase 26H parity matches reference outputs when available (or mismatch/unavailable is prominently documented).
- A future researcher can install, load the frozen public-data cohort, run one baseline and find the complete historical research history without wading through root ZIP files.
- Old scientific analyses remain accessible without conflating them with current supported package code.
- No changes to `main` until audit and verification; work is proposed on a separate cleanup branch.