# Phase 14D — Raw 10 pp VTI responsiveness: frozen OOF evaluation

## Scientific question

Does the original sequence-only VTI Random Forest identify *assay-defined nonresponders* or responders more effectively than a trivial always-responsive and fold-trained median baseline?

## Fixed definition and caveats

- **Positive / responder:** measured `vti_rescue_pp >= 10` percentage points of wild-type CFTR activity.
- The score is the original out-of-fold **continuous** VTI-rescue prediction; call responsive when the predicted rescue is also `>=10`.
- This is a **raw numerical threshold on the experimental contrast**, not the source article's empirical-Bayes responder categorisation and not a direct estimate of patient benefit.
- The VTI experimental contrast was measured after VNZ+TEZ pretreatment with **acute ivacaftor**; it is not a measured clinical vanzacaftor/tezacaftor/deutivacaftor response.
- No threshold search, probability calibration, model retraining, feature selection or use of validation/test outcomes.

## Reused inputs

- `results/transfer/train_cv_predictions.csv`, models `vti_forest` and `median`, 2 repeats × 5 position-held-out folds, 210 VTI training variants at 168 positions.
- `data/processed/bihler_2026_transfer_splits.parquet`, **identity/split/cohort columns only**.
- Existing frozen manifest and stored transfer summary fingerprints.

## Metrics

- Responder and nonresponder prevalence, overall and separately in `ETI, VTI` and `VTI` cohorts.
- Sensitivity for responders, specificity for nonresponders, balanced accuracy, variant-level false responders (FP) and missed responders (FN).
- AUROC using continuous predicted VTI rescue; nonresponder average precision using **negative predicted rescue**. The latter must be compared against the cohort's nonresponder prevalence.
- Every protein position contributes equal total weight **within the cohort being evaluated**; multiple variants at the same position are averaged through their weights.
- Two out-of-fold repeats are averaged per variant **before** classification. Position-bootstrap confidence interval of the forest's balanced-accuracy gain over the trivial always-responsive baseline. These intervals describe variability within inspected development folds, not true prospective performance.
- A fixed observed-only 5–15 pp window to count boundary cases. **It is not calibrated uncertainty, and observed distance to threshold cannot be used for prospective deferral.**

## Critical interpretation

A classifier that always calls responder can achieve high raw accuracy in an enriched-positive cohort, but has **zero specificity and balanced accuracy 0.5**. Predicting raw continuous rescue above/below 10 pp does not produce calibrated responder probabilities. Do not use model-score distance as a claim of confidence.

This analysis remains exploratory because the model comparisons and development responses were already inspected. **Do not unlock either final test set.** The 10 pp task should be closed if sensitivity/specificity or negative-class ranking are not materially useful.

## Usage

```bash
python -m pytest -q tests/
python -m cftr_theratype.response_threshold 2>&1 | tee logs/phase14d_response10.log
```

Outputs: `results/biology/phase14d_response10_train_oof.csv` and `phase14d_response10_summary.json`.

## Next research decision

If the frozen Forest is little better than the trivial class predictor, then establish a **pre-registered structural feature study** using experimentally resolved CFTR structures. Begin with residue-level descriptors / distances to known ligand pockets, then validate a very small Boltz-2 wild-type/variant pilot against known cryo-EM ligand poses *before* scaling. A predicted protein–ligand pose or affinity is not the same as mutant channel rescue, folding correction, maturation, or clinical efficacy.
