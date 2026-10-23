# Phase 14B: functional and maturation phenotype audit

**Purpose.** Test whether the mature CFTR protein (Band C) measurements supplied by Bihler 2024/2026 have an association with functional rescue not accounted for by untreated and ivacaftor-potentiated CFTR activity. This is **descriptive hypothesis generation**, not a new model architecture or a way to score unmeasured variants.

## Study scope and exact row joins

Reuse `target_audit.audit_one` to load only the existing **422 ETI TRAIN** variants and **210 VTI TRAIN** variants. That loader filters assay data to frozen training keys, checks response identities, and joins Band C using the workbook's exact `source_excel_row` (`source_row_dmso` and the corrector-treatment source row). Do not average control, starred, or alternate records. Add `protein_position` from the frozen train-only membership table. The 2026 training data is additionally described within its 66 previously ETI-characterised and 144 VTI-only cohort rows. No held-out outcomes are read.

## Mechanistic contrasts

- `band_c_dmso_pct_wt`: mature CFTR protein in DMSO control, normalized to %WT.
- `band_c_treated_pct_wt`: mature protein under the relevant corrector pretreatment.
- `band_c_gain_pct_wt`: treated minus DMSO mature protein (%WT difference).
- `band_c_gain_pct_total`: treated minus DMSO mature fraction (%Total difference; sensitivity check).

Test five prelisted biomarker–phenotype associations: (1) Band C gain with corrector-associated activity increment (primary), (2) Band C gain with total rescue, (3) treated Band C with absolute treated function, (4) untreated Band C with baseline FSK (positive-control association), and (5) mature-fraction gain with corrector increment. No ETI+VTI outcome pooling; associations are computed separately by study and 2026 selection cohort.

## Statistics

Average multiple variants at the same residue first, then use **equal-position rank correlations**. For associations (1), (2), (3), (5), partial rank correlations regress the biomarker and phenotype ranks on ranked baseline FSK and DMSO FSK+IVA plus an intercept; associations are descriptive and not causal. Compute 95% percentile intervals using 500 bootstrap resamples of **positions**, not variants. The overall VTI and VTI-stratum associations are overlapping descriptive summaries, not independent replication.

**Caveats:** Band C is a post-treatment assay measurement and must not become an inference-time feature. A Band C association can arise from trafficking, stability, abundance, shared normalisation or other confounding; it does not identify a causal mechanism. Band C is a proxy for mature protein, not a direct measurement of plasma membrane abundance or chloride conductance. Rescue targets share terms with baseline covariates (mathematical coupling). The association scan is exploratory; do not claim statistical discovery from unadjusted bootstrap intervals over multiple tests. CIs are not adjusted for measurement error or experimental dependence between studies.

## Decision gate

- A consistently positive **Band C gain ↔ corrector increment** association after conditioning would support evaluating Band C as *training-only auxiliary supervision* in a future pre-registered sequence-only model.
- Weak associations would argue against using Band C supervision merely because these columns are available.
- Either outcome requires independent validation before claims of improved out-of-position prediction. Do **not** unlock final tests or tune a model here.
