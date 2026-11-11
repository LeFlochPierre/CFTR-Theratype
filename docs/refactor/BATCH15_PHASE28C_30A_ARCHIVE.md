# Batch15 — Reversible archival of Phase28C and Phase30A

**Draft / RTX6 verification pending.** This batch archives two *complete*
historical TRAIN-only analyses. It does **not** claim their entire
experiment-specific workflows have been extracted into canonical
`src/cftr_theratype/experiments/` modules.

## What moved

| Previously active script | Archive path | Original Git blob SHA-1 |
|---|---|---|
| `scripts/phase28c_structure_biophysics.py` | `archive/legacy/scripts/phase28c_structure_biophysics.py` | `d3dd7a47515fae5a4506d92c4d48c192f15d16f8` |
| `scripts/phase30a_target_information_ladder.py` | `archive/legacy/scripts/phase30a_target_information_ladder.py` | `45600ac61ba245de92191d8a9a4f099c497c647d` |

The original and archived Git object hashes were verified identical
**before** the original active paths were removed. No source bytes
changed, and Git should identify both as 100% renames. Both modules
already use canonical `src/` imports for the shared dataset/model
pieces. Their experiment-specific geometry, assay-conditioning,
fold comparisons, frozen-result guards, outputs, and CLI are preserved
in the archive unchanged.

## Why these two

A remote-code reference audit did not identify imports of either
module from active scripts. The relevant existing tests inspect their
source text but do not require them to reside in `scripts/`; those
inspections now point at the original archived source. The original
`__file__` uses record the script source hash, not the source path;
the archived bytes are unchanged and can be reproduced with the
existing legacy runner.

This search does **not** prove that external workflows, local untracked
scripts, or Phase33–39 workstation sources have no callers. Inspect
those separately before touching the main research worktree.

## How to run the preserved original entrypoints

```bash
cd ~/projects/CFTR-Theratype
PYTHONPATH="$PWD/src" python scripts/run_legacy.py phase28c_structure_biophysics.py --self-test
PYTHONPATH="$PWD/src" python scripts/run_legacy.py phase30a_target_information_ladder.py --self-test
```

The no-`--self-test` entrypoints run their original experiments and
may overwrite their historical result paths. **Do not invoke them
without an output-preservation protocol.**

## Verification (isolated worktree)

```bash
cd ~/projects/CFTR-Theratype-batch04-test
conda activate CFTR_ENV
git status --short
git fetch origin main
git fetch origin pull/21/head:refs/remotes/origin/pr-21
git switch --detach origin/pr-21

PYTHONPATH="$PWD/src" python -m pytest -q \
  tests/test_phase28c_30a_archival_batch15.py \
  tests/test_phase28e5_29b_archival_batch14.py \
  tests/test_phase27d3a_provenance_batch12.py \
  tests/test_phase28_30_canonical_batch10.py

PYTHONPATH="$PWD/src" python scripts/run_legacy.py phase28c_structure_biophysics.py --self-test
PYTHONPATH="$PWD/src" python scripts/run_legacy.py phase30a_target_information_ladder.py --self-test

PYTHONPATH="$PWD/src" python -m pytest -q -rs tests/
git diff --check origin/main...HEAD
git diff --summary origin/main...HEAD
```

The prior maintained baseline had 251 passing tests, 1 expected skipped
real-TRAIN-loader parity test. Batch14 adds four synthetic archive
tests and Batch15 adds four more. The exact full-suite outcome must be
confirmed on the workstation, not inferred from these counts.

## Retained and out of scope

Phase26N/26O and Phase27F locked evaluation paths remain untouched.
Archived scripts remain accessible in Git and through `run_legacy.py`.
Nothing in this batch touches ignored `data/`, `results/`, `logs/`,
root ZIPs, user-owned local patches, or Phase33–39 working files.
No deletion or cleanup of the main RTX6 checkout is authorised.
