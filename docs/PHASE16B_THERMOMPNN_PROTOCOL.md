# Phase 16B — CFTR single-substitution stability via ThermoMPNN

## Hypothesis and limits

Use the **pretrained** ThermoMPNN graph neural network to create mutation-specific
predicted stability features from the experimentally measured reference CFTR
structure 6O2P. This is an alternative to FoldX, not an experimental mutant
structure and **not** a modulator-binding-affinity or rescue predictor.

The reference has V470 and engineered E1371Q, whereas the functional assay uses
M470. Ligands are excluded from the prepared protein-only PDB. Interpret each
score as context-dependent, not a mutation's universal stability.

## Exact provenance

- PDB input: `results/structure/phase16_foldx/6O2P_cftr_canonical_numbering.pdb`.
- Candidate metadata: `results/structure/phase16_foldx/candidates_train_only.csv`.
- Predictive source: https://github.com/Kuhlman-Lab/ThermoMPNN
- Upstream script: `analysis/custom_inference.py`.
- Explicit pretrained checkpoint: `models/thermoMPNN_default.pt`, if present.
- Source Git commit, model/PDB/checkpoint CSV SHA-256s stored in output manifest.
- Only frozen VTI training identities (210). No assay outcome table read.

## Workflow

1. Clone official repository outside CFTR-Theratype.
2. Set up a separate Python 3.10 environment and verify `torch` and imports.
3. Run ThermoMPNN native saturation inference for **chain A** on prepared PDB.
4. Execute `python -m cftr_theratype.thermompnn_adapter collect ...` in `CFTR_ENV`.
5. Ensure `n_scored == 195`, and remaining 15 have unresolved coordinates.
6. Only after coverage/mapping validation, consider training-only matched CV with
   stability as one additional feature, using the unchanged models and folds.

### CRITICAL: mutation indices

ThermoMPNN's `custom_inference.py` reports 0-indexed positions from its own
PDB parser. The parser fills gaps in integer PDB residue numbering. The
adapter converts `position + first PDB residue number` to canonical CFTR
number, validates wild-type identity at every scored residue, and fails closed
on duplicate, non-finite, or absent predictions for eligible training variants.
Never join a ThermoMPNN score directly on the raw `position` field.

### IMPORTANT: full-length CFTR

The model was benchmarked mainly on soluble proteins. Full-length CFTR is a
large membrane protein, and missing backbones can create gaps. The trained
network's reliability and runtime on 6O2P require direct examination; abort
on CUDA OOM or malformed results rather than silently changing the model.

### Sign and physical interpretation

ThermoMPNN predictions are model-estimated stability-related `ddG_pred` scores.
Check the documented sign convention for the chosen upstream commit before
interpreting whether positive means destabilising. No score is clinical.
