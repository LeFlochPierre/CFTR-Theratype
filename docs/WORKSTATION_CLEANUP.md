# RTX6 workstation cleanup — CFTR-Theratype

**Purpose:** clean the *real* workstation directory together with GitHub, while preserving scientific data, results, experiments and provenance. GitHub alone is not a backup for `data/`, `results/` or `logs/`.

## 1. Inventory without changing working branches

```bash
cd ~/projects/CFTR-Theratype
conda activate CFTR_ENV

git status --short
git branch -vv
git remote -v
git log -1 --oneline

git fetch origin maintenance/research-freeze-2026-11
git show FETCH_HEAD:scripts/maintenance/stage7_local_inventory.py > /tmp/cftr_stage7.py
git show FETCH_HEAD:scripts/maintenance/stage8_workstation_cleanup.py > /tmp/cftr_stage8.py

python -u /tmp/cftr_stage7.py --root "$PWD" --hash-zips
python -u /tmp/cftr_stage8.py plan --root "$PWD"
du -sh archive data docs logs results scripts src tests tools 2>/dev/null
```

Neither script changes tracked files, local Python modules or scientific inputs. Stage 7 writes an inventory of Git status and root ZIP files; Stage 8 creates `phase_scripts.csv` (with importers), `documentation.csv`, `root_zip_plan.csv` and `summary.json` under `results/maintenance/cleanup_audit/stage8/`.

## 2. Preservation before moving anything

- Record current commit, environment versions, and uncommitted local work.
- Back up and hash **all** local patch ZIPs, phase scripts not tracked by Git, datasets, derived data, OOF predictions, results and logs. Inspect the Phase 33–39 ZIPs and compare them with whatever source actually ran on RTX6.
- Preserve third-party licence boundaries. Do not push FoldX, restricted clinical data or third-party dataset copies to GitHub without permission.
- Reconcile historical Phase 26N/26O protected validation/test usage before describing any outcomes as untouched.
- Do not use `git clean`, `git reset --hard`, bulk deletion or a force push.

## 3. Root patch ZIP relocation (optional and reversible)

**Only after** reviewing `root_zip_plan.csv` and securing an independent backup:

```bash
python -u /tmp/cftr_stage8.py stage-artifacts --root "$PWD" --confirm
cat results/maintenance/cleanup_audit/stage8_moves.json
ls -lh ../CFTR-Theratype-local-archive/root-patches/

# Roll back if necessary:
python -u /tmp/cftr_stage8.py restore-artifacts --root "$PWD" --confirm
```

The script only selects untracked local project patch ZIPs with verified ZIP structure and without detected Python/source references. It copies them into a sibling directory and verifies SHA256 **before removing root copies**. It writes a restore manifest. It never moves third-party archives, root CSV inputs, docs, Python source, results or data.

## 4. Numbered scripts and docs

```bash
# Establish baseline behaviour:
python -m pip install -e . --no-deps
python -m pytest -q tests/
cftr --help
cftr data validate --eti

# Existing dependency / archival tools:
python scripts/maintenance/stage4_callsite_audit.py
python scripts/maintenance/stage5_archive_leaves.py plan
python scripts/maintenance/stage6_migrate_hubs.py plan
```

The root `scripts/phase*.py` files are not all independent scripts. Some are dependency hubs used by other experiments, tests and archived scripts. **Do not relocate them simply because they have a phase number.** For each imported hub, extract shared functions into `src/cftr_theratype/`, leave a compatibility wrapper, and verify historical OOF parity before considering archival.

Many independent historical scripts have already been moved under `archive/legacy/scripts/` with a compatibility launcher. For remaining leaf scripts, use the existing Stage 5 `plan/apply/restore` flow in reviewed batches, not `mv` or deletion.

Keep historical `docs/PHASE*.md` readable at their existing paths initially. [The docs index](INDEX.md) provides navigability. Later, relocate old docs to `docs/history/` only after reviewing links and code references, using small reversible Git commits.

## 5. Sync RTX6 and GitHub

Review [draft PR #2](https://github.com/LeFlochPierre/CFTR-Theratype/pull/2). Do not merge until backups and tests pass. A clean GitHub branch does not clean the local workstation automatically. Only after local changes have been preserved and a refactor batch validated should you fast-forward the appropriate workstation branch or use a separate worktree.

**Done means:** no scientific output lost; late-phase source preserved; tested canonical API; numbered historical scripts still accessible; docs indexed; original results traceable; workstation and remote branch consistent.
