# Phase 16C — Does predicted mutation stability improve CFTR theratyping?

**Status:** exploratory, predeclared model comparison on **VTI frozen training only**. No ETI or VTI validation/test outcomes are opened or used to choose hyperparameters. The experimental goal is to test mutation-specific protein-stability *features*, not to infer drug binding or clinical benefit.

## Motivation and methods

- **Inputs:** 68 original sequence descriptors, 12 experimentally anchored reference-structure descriptors from 6O2P and 8EIQ, plus ThermoMPNN stability prediction and score-availability indicator (2 columns). ThermoMPNN was applied without CFTR rescue labels to the prepared protein-only 6O2P structure. It generated predictions for 195 of 210 eligible VTI training variants; the other 15 remain in the benchmark with missing score, imputed only from each training fold.
- **Reference:** 6O2P reflects V470 and engineered E1371Q, whereas experimental functional assays use an M470 background. No model here estimates modulator affinity, trafficking or gating. ThermoMPNN scores represent mutation-associated *predicted stability changes for this reference conformation*. In the upstream stability convention, negative values indicate stabilising predictions, positive values destabilising; this convention is not a guarantee of quantitative thermodynamic accuracy for full-length membrane CFTR.
- **Validation of inputs:** Check source PDB hash, full ThermoMPNN saturation-output hash, candidate identity-manifest hash, exact 210 variant keys and positions, amino-acid substitutions, 195 finite score values and 15 explicitly missing values. Also revalidate original Phase 15B structural provenance, Phase 15C cached model/protocol hashes and matching training labels.
- **CV:** *Exactly* the frozen Phase 15C 2 × 5 position-held-out VTI folds (210 variants, 168 positions), no new random split. The original structural Forest must refit and reproduce the stored OOF values within `1e-6` on each fold. Predictions from repeated folds are averaged for each variant, then errors averaged equally across protein positions; cohort-stratified results are reported.
- **Model:** Unchanged VTI Random Forest: 250 estimators, 4 minimum samples per leaf, 0.75 feature fraction, same fold-specific random seeds as previous studies. Compare the frozen sequence-only Forest; original sequence+structure Forest; sequence+structure+ThermoMPNN score/indicator.
- **Negative controls:** Five label-blind permutations of the finite ThermoMPNN scores *across scored variants*, retaining missingness at its original locations; fit the same Forest in each fold. Report their MAE mean/range, paired equal-position MAE gain relative to the real score, and position-cluster-bootstrap intervals. This is a diagnostic against arbitrary feature-count effects, not a model-selection distribution or a formal randomisation test.
- **Endpoints:** Main: position-averaged rescue MAE and cohort-specific bias. Secondary: continuous-score non-response AUROC/AP (raw `<10 pp` VTI rescue), fixed 10 pp decision rule counts and fixed 20% research-triage recovery. Positive/negative categories are raw assay thresholds, not the publication's empirical-Bayes classifications. No threshold optimization or direct classifier retraining.
- **Interpretation:** Do not claim improvement unless the real ThermoMPNN feature adds credible benefit versus the *structure-augmented* Forest and shuffled controls without harming the poorly responding cohort. Previously inspected training CV is exploratory, not independent confirmation. **Do not open final test outcomes**. Score reliability must be confirmed independently for membrane proteins before biological interpretation.

## Run

```bash
conda activate CFTR_ENV
python -m pytest -q tests/
python -m cftr_theratype.stability_benchmark 2>&1 | tee logs/phase16c_stability_benchmark.log
```

Outputs: `results/structure/phase16c_stability_train_oof.csv`, `phase16c_stability_summary.csv`, `phase16c_stability_nonresponse.csv`, `phase16c_stability_summary.json`.

## Source

Dieckhaus H, Brocidiacono M, Randolph NZ, Kuhlman B. *Transfer learning to leverage larger datasets for improved prediction of protein stability changes.* PNAS 2024;121:e2314853121. https://doi.org/10.1073/pnas.2314853121
