# Portable canonical CFTR reference and test-suite recovery

`tests/test_benchmark.py`, `tests/test_esm2.py`, and
`tests/test_reference.py` previously depended on the ignored
`data/raw/uniprot/P13569.fasta` file, causing seven failures in clean
Git worktrees even though the protein reference is public.

This patch packages the reviewed UniProt **P13569-1** canonical sequence in
`src/cftr_theratype/resources/P13569.fasta`, sourced from
https://rest.uniprot.org/uniprotkb/P13569.fasta (4 November 2026).

Scientific invariants verified:

- **1,480** residues; canonical residue 470 is **V**.
- Sequence MD5 **F29B8C0A9056A0F7680F3290D259B6AC**, matching UniProt's
  published checksum (sequence version 3).
- `bihler_2024_assay_sequence` preserves the original **V470M** background
  and remains a one-residue transformation.
- The previous local FASTA location remains preferred when present, but it
  **must** match the same pinned canonical sequence. Invalid references fail
  explicitly rather than silently producing different features.

`pyproject.toml` explicitly includes the FASTA in the installed wheel.

The Phase26 test `EXPECT` typo, accidentally present on merged main despite
being fixed in a previous feature branch, is also corrected.

## RTX6 verification

```bash
cd ~/projects/CFTR-Theratype-batch04-test
git status --short
git fetch origin main fix/portable-cftr-reference-batch06
# Only if this worktree is clean:
git switch --detach origin/fix/portable-cftr-reference-batch06
PYTHONPATH="$PWD/src" python -m pytest -q -rs \
  tests/test_reference_resource.py \
  tests/test_benchmark.py \
  tests/test_esm2.py \
  tests/test_reference.py \
  tests/test_phase26_modular_hubs.py
PYTHONPATH="$PWD/src" python -m pytest -q -rs tests/
git diff --check origin/main...HEAD
```

The separate real-data Phase26D frozen-loader check will still `skip` in an
isolated worktree without `data/processed/bihler_2026_vti.parquet`.
This is an **unverified** real-data parity gate, not a success.
