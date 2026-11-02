# Phase 15B — Experimental-reference structural descriptors

Status: **prespecified, label-free structural feature generation only**. The task is to construct interpretable, per-position reference structural descriptors for every CFTR position represented in the frozen 2024 ETI and 2026 VTI partitions. No functional outcome, model forecast or 10 pp responder label is read.

## Inputs and provenance

- Phase 15A outputs: `results/structure/phase15_coverage_positions.csv` and `phase15_coverage_summary.json`.
- Experimental reference structures: `data/external/structures/6O2P.cif`, `8EIQ.cif`, with SHA-256 verified against Phase 15A.
- Frozen ETI/VTI split **files are checked by hash only** (no outcomes or training partition measurements accessed).
- Canonical UniProt P13569 position numbering with special caution that the experimental functional assay has M470 rather than canonical V470; engineered Q1371 may differ.

## Features

1. Alpha-carbon contacts (8 Å, excluding positions ±3), calculated in Phase 15A.
2. Long-range alpha-carbon contacts (8 Å, residue sequence separation >100 amino acids). A *proxy* for long-distance structural interactions, **not** a validated domain-interface annotation.
3. Protein-only Shrake–Rupley SASA (100 sampling points per atom) and relative SASA (Biopython Sander maximum amino-acid accessibility). **These values ignore the membrane, lipids and ligands** and must not be interpreted as in-membrane exposure or mutant stability.
4. C-alpha distance to nearest observed ligand heavy atom: 6O2P–VX7 and 8EIQ–VX7/CV6/WJX. These are reference-state geometric distances, **not** ligand binding affinities. In particular, 8EIQ has ETI drug ligands, **not vanzacaftor**.
5. Explicit exact-coordinate coverage flags and an M470 assay-versus-structure flag.

Use 6O2P as primary for generic contacts and SASA; fall back to 8EIQ only where 6O2P is not exactly resolved. Retain individual-PDB ligand distances without pooling different drug identities. No contacts/exposure/distance values are assigned from construct-mismatched, deleted, or unresolved residues. All missing values stay missing in the feature file.

Structural features are *per wild-type/reference residue*, not individual mutant structures. Two substitutions at the same sequence position share these structural descriptors. Mutation-specific information remains in the original 68 sequence/biochemical features.

## Data isolation and scope

Run:

```bash
python -m pytest -q tests/
python -m cftr_theratype.structure_features 2>&1 | tee logs/phase15b_structure_features.log
```

Outputs:

- `results/structure/phase15b_position_features.csv`
- `results/structure/phase15b_feature_summary.json`

The resulting CSV contains all positions from the Phase 15A union, including positions without coordinates. These should never be discarded during performance evaluation.

## Next frozen modeling comparison (NOT implemented in Phase 15B)

On VTI 210 training variants only, use the same *original* 2x5 position-held-out partitions. Compare the original 68-feature sequence Forest to an identical Forest receiving 68 features plus the above prespecified structural features. Fit any missing-value imputation **inside each training fold** and include missingness indicators; do not drop unresolved positions. Report overall position-MAE, 10 pp AUROC and nonresponder recall, and cohort-specific position-MAE and bias, using identical cases. Avoid using validation/test outcomes during model selection; those sets remain sealed. Use the saved baseline forest OOF predictions and compare directly to the new model, with an appropriate check that the original CV folds/feature pipeline match.

If a structural feature does not improve results, do not reinterpret it as proven not biologically relevant: static reference structures omit mutant folding energetics, gating and conformation-dependent effects. AlphaFold/Boltz mutant-specific predictions remain a separate and more expensive hypothesis.
