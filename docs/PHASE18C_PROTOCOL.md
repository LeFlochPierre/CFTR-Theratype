# Phase 18C — IVACAFTOR BOLTZ PILOT

## Hypothesis and scope
Assess whether a known-pocket-conditioned ligand model responds reproducibly to
single substitutions in CFTR, before adding drug-aware features to supervised
theratyping models. This is NOT a validated predictor of acute potentiation.

## Fixed inputs
- Experimental 6O2P polymer, 1489 deposited residues with ligand VX7.
- Exact native 6O2P experimental CIF with SHA verified against the frozen Phase 16A manifest.
- Boltz guided WT YAML from Phase 18B. Keep template, binder VX7 and fixed five pocket
  contacts identical in all constructs.
- All candidate mutations are from the frozen 210 VTI TRAIN identities; 195 eligible.
- Map canonical positions to deposited polymer indices using the frozen Phase 15A global
  alignment; verify residue identity and resolved C-alpha coordinates.
- The template's engineered E1371Q and canonical V470 background do not match all assay
  construct variants. Positions 470 and 1371 are excluded by the Phase 16A manifest.

## Label-free panel
12 distinct sites: 4 closest to the native VX7 ligand, 4 intermediate-distance sites,
4 farthest sites. One deterministic mutation selected per site by sorted identity.
Wild-type experimental construct acts as internal control. No assay labels are loaded.

## Stage gates
1. Verify 1489 protein residues, 1179 C-alpha coordinates, 195 eligible TRAIN variants.
2. Verify every target residue's wild-type amino acid at the deposited index.
3. Run three smoke YAMLs with identical command options (WT, near-pocket, far-pocket).
4. Validate ligand pose, ligand contact patterns, model confidence and pose dispersion.
5. Compare mutant deltas against reference replicate dispersion. Do NOT infer affinity
   changes from raw score differences without external validation.
6. Only after this, optionally evaluate acute IVA increments from within-condition
   FSK+IVA minus FSK in training-only, position-safe CV.

## Caveats
Experiments use a ligand-bound WT experimental protein template and enforced pocket
constraints, so success is a conditional geometry perturbation not independent docking.
Boltz's affinity head is chiefly developed to compare small-molecule differences,
not necessarily protein-mutation effects for fixed ligand. CFTR is a membrane protein
and assay effects include maturation, expression, gating and drug interaction.
Predefined, label-blind selection avoids hidden tuning to the response.
