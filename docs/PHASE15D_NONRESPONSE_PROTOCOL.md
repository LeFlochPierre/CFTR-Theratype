# Phase 15D — Explicit CFTR non-responder learning (train-only)

## Scientific question

The frozen VTI regression Forest assigns all 210 training variants a fitted rescue
>=10 pp in position-grouped OOF evaluation and therefore misses 36/36
measured low responders. Training a regressor to minimise MAE is not the
same objective as detecting a rare, subthreshold category.

This is a **new exploratory experiment** and not clinical responder classification.
The endpoint is `vti_rescue_pp < 10` vs `>=10` percentage points of WT CFTR
function, distinct from the paper's empirical-Bayes categories. This endpoint
can be noisy near threshold and is not a clinical treatment recommendation.

## Fixed approach

- VTI 2026 frozen TRAIN only (210 variants / 168 distinct residue positions);
  no ETI assays, VTI validation/test responses, or original ETI val/test responses.
- Reuse the original *2 repeats x 5 position-held-out folds* from transfer
  `train_cv_predictions.csv`, and the source/provenance-verified 15B structure table.
- Compare two direct binary random-forest classifiers: original **68 sequence**
  features, and **68 sequence + 12 experimental-reference structure** features.
- Class label 1 = *non-responder*. Both classifiers use 250 trees, minimum leaf 4,
  `max_features=0.75`, `class_weight="balanced_subsample"`; no cohort label or
  measured functional phenotype at inference.
- Fit preprocessing median values only on fitting positions. No position shared
  between fitting and validation in either inner or outer fold.
- Compare a naive probability decision boundary of 0.5 against a threshold
  chosen from 0.10..0.70 in 0.05 increments, using **three inner,
  position-held-out folds inside each outer training fold only**. Maximise
  inner balanced accuracy subject to >=80% responder sensitivity in the inner
  OOF data. We do **not** tune on held-out outer samples.
- Fixed comparators: frozen sequence and structure regressors, and the
  always-responder rule. Their OOF predictions are reused, not refitted.
- The 2 repeats are averaged per variant for scoring. For inner-threshold models,
  the averaged (score − selected threshold) margin determines the decision.
- Equal-position weighting for balanced accuracy, class-specific recall,
  non-responder AUROC and precision–recall AP. Report **both cohort strata**.
- Also rank the top 20% most likely non-responders as an experimental-assay
  prioritisation diagnostic. This budget was fixed before inspecting this run;
  its precision and recall are **variant-level**, unlike position-weighted metrics.

## What success means

The classifier should catch non-responders **without simply labelling almost
all variants nonresponsive**. Report non-responder recall, responder recall
(false-alarm cost), balanced accuracy, non-responder AP, and stratified results.
Better ranking alone is useful for experiments but is not calibrated prediction.
Class-weighted RF `predict_proba` is a ranking score, NOT validated probability.

The 36 non-responders are few and responses near 10 pp are noisy. Results are
exploratory and cannot be presented as independent evidence: architecture and
hypothesis choice followed inspected training experiments. A true confirmatory
assessment will require a strictly frozen procedure and an untouched partition.

## Run

```bash
python -m pytest -q tests/
python -m cftr_theratype.nonresponse_classifier \
    2>&1 | tee logs/phase15d_nonresponse_classifier.log
```

Outputs: `results/structure/phase15d_nonresponse_train_oof.csv`,
`phase15d_nonresponse_summary.csv`, `phase15d_nonresponse_summary.json`.
