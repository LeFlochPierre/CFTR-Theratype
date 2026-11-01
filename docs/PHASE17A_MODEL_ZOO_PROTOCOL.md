# Phase 17A — Group-nested model zoo (exploratory VTI TRAIN)

**Question.** Do alternative tabular regressors extract more generalisable VTI rescue information from (A) sequence alone, (B) sequence and static structure, or (C) sequence, static structure, and ThermoMPNN mutation stability?

**Inputs.** Original 210 VTI TRAIN missense variants, 168 protein positions. The original 2×5 frozen position-disjoint outer folds and prediction references from Phases 15C and 16C are verified before any model fit. Sealed VTI/ETI validation/test *outcomes* are never loaded. All label reads use the existing TRAIN-key-filtered parquet reader. The 15 missing ThermoMPNN stability scores are retained via an availability flag and training-fold imputation. Structural position coverage similarly remains explicit.

**Predeclared candidates.** Six family models × three nested feature sets = 18 algorithms. Each outer fold independently selects hyperparameters using a three-fold protein-position-disjoint GroupKFold (inner position-level MAE). Model parameters are a small fixed grid per family (2–3 options). Imputation and scaling are fit separately for each inner-fit and outer-fit fold. No tuning on outer-held-out responses.

| Family | Tuned options | Fixed regularisation / design |
|---|---|---|
| CatBoost | depth 2 / 3 / 5 | 250 iterations, learning rate .04, L2=8 |
| XGBoost | max-depth 2 / 3 / 4 | 250 trees, learning rate .035, min-child-weight=5, L2=12 |
| ExtraTrees | leaf-size 2 / 4 / 8 | 200 trees, max-features .75 |
| RBF-SVR | C = 1 / 10 / 100 | standardized features, epsilon=5 |
| ElasticNet | alpha .1 / 1 / 10 | standardized features, L1 fraction .5 |
| Gaussian Process | RBF lengthscale 4 / 12 | standardized features, fixed kernel, white noise |

**Metrics.** Average each variant's two independent repeat predictions, calculate variant absolute/signed errors, then average across protein positions equally. Report all-cohort and ETI-characterised/VTI-only subgroup position-MAE, signed bias, bootstrap paired differences versus frozen static-structure Forest and frozen stability-augmented Forest, 10-pp raw assay non-response AUROC/AP, and non-responders retrieved within the top 20% of variants by predicted risk. 2,500 protein-position resamples provide descriptive percentile CIs (not post-model-selection significance guarantees).

**Interpretation guardrails.** This is further exploration after earlier analyses used the same TRAIN cohort. The best among 18 outer CV scores will be optimistically selected. Confirm a *prespecified single finalist* on independent outcomes only once, with appropriate treatment-study and variant-position embargo; do not claim a clinical nonresponse classifier based solely on these results. The VTI outcome is an assay rescue measure involving acute ivacaftor, not the clinical VTD combination. No causal drug claims or mechanistic stability conclusions from predictive correlations.

**Requirements.** Existing Phases 15C, 16B, and 16C completed. Current project's environment with `catboost`, `xgboost`, `scikit-learn`, `numpy`, `pandas`, `pyarrow`, `pytest`. CPU execution is intentional for deterministic small-data benchmarking. Save terminal log with `tee`; complete structured results and fixed hyperparameter selections are written under `results/model_zoo`.
