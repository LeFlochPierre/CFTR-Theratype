# Phase 14C — Is Band C maturation useful *as training supervision*?

## Motivation
Phase 14B found strong position-aggregated, baseline-adjusted rank associations between Band C maturation gain and correction-associated CFTR function, including VTI rho +0.710 overall and +0.741 in VTI-only. This is **not** evidence that a model using sequence alone can predict either phenotype or that an unmeasured Band C value may be used at inference.

## Prespecified train-only comparison
The frozen VTI-only 250-tree Random Forest (250 estimators, min leaf 4, max features .75) provides the fixed reference. On its exact 2 × 5 protein-position-disjoint held-out folds, compare:

1. `vti_forest`: original frozen out-of-fold predictions (no retraining).
2. `rescue_plus_band_c`: same Forest in multi-output mode, training on *standardized* VTI rescue and treatment-induced Band C %WT gain with equal impurity weights.
3. `rescue_plus_shuffled_band_c`: same Forest, but permute **training-fold** Band C labels to break the biological association while preserving their marginal distribution.

At inference, each receives only the 68 pre-established sequence/mutation features. The observed held-fold Band C phenotype is never an inference feature; it is read only for *post hoc auxiliary-head evaluation*. No ETI outcomes, VTI validation outcomes, or either study's test outcomes are accessed.

### Success criterion
The genuine Band C auxiliary must show meaningful positive paired position-bootstrap MAE gains **against both** the original Forest and the shuffled-auxiliary negative control, without worsening the already disadvantaged previously ETI-characterised subgroup. Also report worst-cohort MAE and the signed cohort-bias gap. Because prior research choices used training-CV results, even success remains exploratory, not external confirmation.

A positive *correlation* between measured maturation and measured rescue, on its own, does not satisfy the criterion. Band C is a post-treatment protein maturation marker; it is not synonymous with ion channel function, nor a readily available feature for an untested clinical variant.

### Statistical methods and limitations
Average the two repeated held-out predictions per variant before calculating errors. Average within each protein position before the MAE and paired position bootstrap. The original reference's CV-fold identity is checked against the frozen train split and source hashes. Fold-training target means/SDs alone are used for task standardization. Study cohort is outcome-evaluation metadata only.

The random-auxiliary control is a useful diagnostic, but one fixed shuffle is not a definitive randomization test. This experiment does not prove mechanistic causality or clinical validity and should be frozen after one evaluation.
