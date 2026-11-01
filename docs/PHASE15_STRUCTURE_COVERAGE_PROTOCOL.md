# Phase 15A — CFTR reference-structure coverage audit

**Status:** frozen prospective audit protocol; identity-only; no structure-based model is trained.

## Scientific question

How many variants/positions in the *frozen* 2024 ETI missense and 2026 VTI transfer partitions can be confidently mapped to experimentally resolved CFTR residue coordinates? How many lack coverage, or map to a different reference construct?

## Inputs and isolation

- `data/processed/bihler_2024_missense_splits.parquet`: `variant_key`, `protein_position`, `split` only; frozen expected 586 variants.
- `data/processed/bihler_2026_transfer_splits.parquet`: `variant_key`, `protein_position`, `transfer_split` only; frozen expected 415 variants including excluded/unparseable.
- `data/external/structures/6O2P.cif`: RCSB experimental ivacaftor complex, with engineered construct differences.
- `data/external/structures/8EIQ.cif`: RCSB experimental *F508del* CFTR with elexacaftor (WJX), tezacaftor (CV6), ivacaftor (VX7).
- Canonical human CFTR UniProt P13569 (1480 aa): constructed from existing `bihler_2024_assay_sequence` after restoring V470. Assay construct has **M470**.

**No functional response labels, regression predictions, empirical-Bayes response categories, assay results, or validation/test outcomes are read.** Counting held-out positions for structure *coverage* does not evaluate their response.

## Alignment and quality controls

The mmCIF deposited CFTR polymer sequence is aligned globally (BLOSUM62, affine-gap penalty) to canonical P13569, not mapped by raw author residue numbers. `label_seq_id` identifies deposited-sequence positions; explicit C-alpha coordinates mark structurally resolved residues. Unresolved residues, construct deletions, and canonical-vs-construct residue differences are separate statuses. The CFTR chain is selected by the most resolved C-alpha coordinates in the aligned polymer entity.

The alignment rejects low identity; it also asserts that 8EIQ does **not** map canonical F508 (because the construct has F508del). Mutation at position 470 needs special care because of the V470 canonical/M470 assay reference difference. Any apparent discrepancy should be checked against RCSB/SIFTS before modeling.

Statuses:
- `resolved_exact`: coordinate exists and deposited residue equals canonical reference at that position.
- `resolved_construct_mismatch`: coordinate exists but amino acid differs from canonical reference.
- `deposited_but_unresolved`: sequence present, C-alpha coordinate absent.
- `not_in_deposited_construct`: reference site has no deposited sequence counterpart (deletion/omission).
- `unparseable_variant_position`: variant has no unambiguous position in frozen identity table.

The union of exact resolved coordinates from either structure is the **conservative coverage** numerator. The denominator is the number of distinct parsed variant positions in each split. The output reports both variant- and position-level coverage.

## Exploratory geometry

For exact- or mismatch-mapped residues with coordinates, CA contact count counts neighboring reference CFTR CA atoms within 8 Å, excluding ±3 sequence positions. Distances to listed experimental modulator heavy atoms are **CA-to-heavy-atom minimum distances in the reference structure** (Å), not docking, binding energy, or a representation of a mutant. For 6O2P, ivacaftor `VX7` is considered; for 8EIQ: `WJX` elexacaftor, `CV6` tezacaftor, `VX7` ivacaftor. Distances for absent ligand coordinates are missing, never imputed.

**No vanzacaftor complex is claimed.** The 2026 functional assay used vanzacaftor + tezacaftor pretreatment with acute ivacaftor, whereas 8EIQ contains the ETI modulators.

## Output

- `results/structure/phase15_coverage_variants.csv` — frozen variant IDs and mapped reference geometry/status.
- `results/structure/phase15_coverage_positions.csv` — union of observed variant positions and structural mapping.
- `results/structure/phase15_coverage_summary.json` — split-level coverage, alignment checks, construct and source SHA-256 fingerprints.
- `logs/phase15_structure_coverage.log` via `tee`.

Run `python -m cftr_theratype.structure_coverage --download 2>&1 | tee logs/phase15_structure_coverage.log`. Downloading uses Python (`urllib`), not curl, and preserves existing files. Run test suite before interpreting results.

## Gate to next phase

Inspect measured sequence-identity/position coverage, construct mismatches, ligand presence, and missing data. **Do not** engineer drug-binding labels or train a model until structural coverage/accuracy are checked. Subsequently test one fixed, position-disjoint, sequence-only Forest vs sequence + structural descriptors, using identical variant folds. This is retrospective experimental structure context, not an experimentally validated structure of every mutant.

Sources:
- 6O2P: https://www.rcsb.org/structure/6O2P
- 8EIQ: https://www.rcsb.org/structure/8EIQ
- UniProt P13569: https://www.uniprot.org/uniprotkb/P13569/entry
