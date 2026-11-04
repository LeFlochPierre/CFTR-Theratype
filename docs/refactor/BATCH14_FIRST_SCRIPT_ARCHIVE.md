# Batch14 — First reversible archival of unused phase entry points

**Status: draft; RTX6 parity/test gate pending.** This is **archival**,
not deletion of unique scientific methods.

## Why Stage5 reported zero candidates

The Stage5 tool correctly rejects candidates that have any Python
or textual source reference, and also rejects scripts that use
`__file__`. This is intentionally conservative and is not proof
that all 29 scripts are actively required at their original locations.

A dependency audit of current GitHub `main` found two leaf scripts
with no inbound active numbered-script imports:
- `scripts/phase28e5_foldx_predictive_ablation.py`
- `scripts/phase29b_predictability_ceiling_audit.py`

Both still contain unique exploratory analysis CLI logic; **neither
has been fully reimplemented as a canonical package experiment**.
For reproducibility, their exact code is moved into the tracked
historical source archive instead of being discarded. This remains
accessible via `scripts/run_legacy.py`.

## Byte-identical source preservation

| Source in original main | New path | Original Git blob SHA1 |
|---|---|---|
| `scripts/phase28e5_foldx_predictive_ablation.py` | `archive/legacy/scripts/phase28e5_foldx_predictive_ablation.py` | `8b54f2c30bb6c947802647ae4d831eee4717256a` |
| `scripts/phase29b_predictability_ceiling_audit.py` | `archive/legacy/scripts/phase29b_predictability_ceiling_audit.py` | `14f026bc88a0703213b9f02d7b30ba70dccd8cf6` |

The created archive files were compared against the original
content-addressed Git blobs **before** the active copies were
deleted. Both blob IDs matched exactly. Git may report this as a
100%-similarity rename.

The only `__file__` uses in these scripts are to record their
own source hashes in scientific provenance files; those hashes
remain stable because the archived source bytes are identical.
Their dataset and output paths are based on `Path.cwd()`,
and `run_legacy.py` enforces execution from the repo root.

## Updated tests

- Historical import-boundary AST tests now read the archived source
  paths rather than the removed active paths.
- `tests/test_phase28e5_29b_archival_batch14.py` asserts that:
  - The original active files no longer exist.
  - The archived copies have exactly the original Git blob IDs.
  - Archived code still parses as Python.
  - The historical `--self-test` CLI runs through
    `scripts/run_legacy.py` for both scripts, without any
    protected data or outputs.

The broader 251-test baseline remains the test gate. The isolated
checkout's optional Phase26D parquet real-data parity test is
expected to skip; it is **not** proof of exact frozen scientific
parity.

## Test on RTX6, in the isolated worktree only

```bash
cd ~/projects/CFTR-Theratype-batch04-test
conda activate CFTR_ENV
git status --short
git fetch origin main
git fetch origin pull/20/head:refs/remotes/origin/pr-20
git switch --detach origin/pr-20

PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase28e5_29b_archival_batch14.py \
  tests/test_phase28_30_canonical_batch10.py \
  tests/test_phase27_downstream_imports_batch09.py \
  tests/test_phase26h_loader_hub_batch08.py

# Direct historical self-tests (the new test also calls both)
PYTHONPATH="$PWD/src" python scripts/run_legacy.py \
  phase28e5_foldx_predictive_ablation.py --self-test
PYTHONPATH="$PWD/src" python scripts/run_legacy.py \
  phase29b_predictability_ceiling_audit.py --self-test

PYTHONPATH="$PWD/src" python -m pytest -q -rs tests/
git diff --check origin/main...HEAD
git diff --summary origin/main...HEAD
```

## Not part of this batch

Never delete `data/`, `results/`, `logs/`, ZIP or other
Git-ignored research assets in the main RTX6 checkout. The local
Phase33–39 working state is not represented in the current remote
repository. Keep Phase26N, Phase26O and Phase27F locked validation
and final-test workflows unchanged. Do not remove active Phase26D,
Phase26G, Phase26H or other scripts that remain referenced until
they have their own source-parity and CLI migration gates.

Further archival batches should be small and reversible, updating
any original-path consumers and retaining full historical code
and source checksums.
