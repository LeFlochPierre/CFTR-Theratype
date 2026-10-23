# CFTR-Theratype | Phase 14, Step 1 — Biological inputs and target-definition audit

**Status: next planned work, NOT an executed experiment.** 23 October 2026. Before building any new model, freeze the candidate feature definitions and assess what information was missing in Phases 1–13.

## Research question

Which experimentally grounded mutation/structure features and outcome formulations are plausible determinants of ETI/VTI rescue at **unseen CFTR protein positions**, and which are available **at inference time without a wet-lab measurement**?

### A. Mandatory target-label audit — no modeling

1. For each study, verify units and exactly which pretreatment/acute potentiator are included in `FSK %WT`, `FSK+IVA %WT`, rescue and corrector-associated increment.
2. Decompose the algebra:  
   `total_rescue = (treated_FSK+IVA - control_FSK+IVA) + (control_FSK+IVA - control_FSK)`  
   The first term is corrector-associated activity, the second is acute IVA-associated activity. Confirm both terms exist for every eligible observation and that numerical differences reproduce stored targets to a strict tolerance.
3. Characterise negative functional readings, floor effects near 0, values above 100% WT, nonlinear ceiling/saturation and measurement uncertainty. **Never clamp observed values merely for convenience.**
4. Check technical replicates, number of wells, SEM/SD, normalisation batches and whether WT references are shared; establish which are measured in both 2024 and 2026.
5. Consider three *candidate estimands*, not three automatic targets: total rescue (original), corrector-associated increment (already tested), and absolute treated activity. Absolute treated activity may avoid baseline subtraction noise but answers a different biological question. Compare label noise, interpretability and clinical proxies **before** any model experiments.
6. Contrast study-specific normalisation, rescue definitions and retrospective ETI non-/borderline-response selection. Avoid inferring relative drug efficacy from between-study observational differences.

### B. Feature candidates — strictly annotation-only at inference

| Feature family | Example measurements | Source/status | Priority |
|---|---|---|---|
| Functional domain | TM1/TM2, NBD1/NBD2, disordered regulatory region | UniProt P13569 coordinates; reconcile assay M470 background | 1 |
| Local structural context | Solvent accessibility, secondary structure, residue packing/contact number | Specific experimental CFTR PDB structure(s), chain/coverage map required | 1 |
| Evolutionary / biochemical | Substitution scores, hydrophobicity, charge, residue conservation | Some already in existing 68-feature baseline; audit duplicates | 1 |
| Predicted structural confidence | Model pLDDT/prediction reliability, interface proximity | AlphaFold/PDB only when residue numbering and construct align | 2 |
| Protein conformational environment | Drug-binding vicinity and domain interfaces | Treatment/conformation-specific evidence required, no assumed binding sites | 2 |
| Clinical annotations | CFTR2 classification, CF-causing status, sweat chloride | Assess availability, rights, and whether they leak later clinical knowledge | 3 / separate task |
| Measured baseline or FSK+IVA | Functional activity measured in the same assay | **Diagnostic oracle only**, unavailable for untested variants | Not a sequence-only feature |
| Response stratum or source cohort | Prior ETI non-/borderline-response membership | **Forbidden** as a deployable sequence-only feature | Not a feature |

Important: most candidate mutation/structural annotations are **position-derived**. Use protein-position-disjoint CV and check coverage separately by domain/conformation to avoid overstating generalisation. Do not assume structure-derived features are nonredundant with existing protein-language embeddings.

### C. Data availability and novelty review

- **UniProt P13569** is a grounded source of domain/topology and annotated sequence features: https://www.uniprot.org/uniprotkb/P13569/entry
- **CFTR2** provides variant-specific disease/clinical descriptors. Check download/API permissions, licensing, cohort overlap and the difference between disease liability and drug response before using: https://cftr2.org/
- **Bihler et al. 2026** documents deliberate enrichment of ETI non-/borderline-responders, and reports in-vitro VNZ/TEZ + IVA response rather than clinical VTD therapy. DOI 10.1016/j.jcf.2026.08.010.
- Revisit independent publicly accessible functional rescue assays, patient-derived nasal epithelial/organoid measurements and assay licensing; do **not** count McDonald 2024 as external independent replication.

### D. Deliverables before another model

1. **Data dictionary:** each assayed condition, units, plate/control normalisation, measurement error and exact label formula for both studies.
2. **Domain/structural coverage table:** proportion of training variants at annotated domains/structure-covered positions; report multiple structure conformations if used.
3. **Target audit:** distributions, measurement variance, algebraic checks and the overlap of functional strata; only eligible *training* partitions are evaluated in discovery.
4. **Feature provenance and availability matrix:** exact accession/version/coordinate frame, missingness, test-time availability, and whether each candidate duplicates existing features.
5. **Fixed decision memo:** whether to add structural features, redefine the outcome, gather independent data, or stop cross-treatment modeling. No test access before this decision.

### E. Immediate first action

Run `python scripts/maintenance.py --inventory` if needed, then inspect **only column names/dtypes** of the study-derived `data/processed` tables and **training** rows, alongside the original publication legends. Do not print, load or analyse held-out test outcome columns. Establish the data dictionary before trying any new feature or representation.

## Stop/go gate

**GO** to a new modeling experiment only if at least one proposed feature/target supplies a *new biological signal*, has provenance and test-time availability, and can be evaluated with the existing position-disjoint protocol without repeated validation tuning. Otherwise, prioritise independent functional datasets or a reframed research aim.
