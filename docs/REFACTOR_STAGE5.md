# Stage 5 — Historical leaf archival

This stage separates historical experiment **entry points** from the small active `scripts/` tree, without modifying their contents. The migration is incremental, Git-backed and reversible.

1. Commit the Stage 1–4 source checkpoint (including previously untracked historical scripts).
2. Run `python -m pytest -q tests/test_stage5_archive_leaves.py`.
3. Run `python scripts/maintenance/stage5_archive_leaves.py plan`.
4. Inspect `results/maintenance/refactor_audit/stage5/archive_plan.csv`.
5. Move one small batch with `python scripts/maintenance/stage5_archive_leaves.py apply --limit 20`.
6. Re-run package tests and the canonical frozen benchmark comparison. If anything fails, use `python scripts/maintenance/stage5_archive_leaves.py restore` before committing.
7. Review `git diff --cached --stat`, then commit verified archive moves. The manifest may be retained as an audit record; once committed, rename it before running a further batch.

**Important**: static scans cannot find external shell commands, notebooks, bespoke scheduled jobs or dynamic imports. `CANDIDATE` means *not referenced by checked code*, not guaranteed unused. The audit ignores documentation references, so historic docs may still refer to old paths. Archived scripts can be invoked with `python scripts/run_legacy.py phaseXX_script.py [args...]` from the repository root, although a separate historical checkout at the checkpoint is the strongest guarantee of full reproducibility. Modules that other scripts import are protected until those imports have been migrated.

Do not use `git clean`. Do not archive the shared dependency hubs with this helper. Do not add raw data or huge model weights to Git. Existing data/results/logs remain unchanged.
