# CFTR-Theratype: biologically anchored latent transfer protocol

**Status:** Exploratory, training-only experiment. Fixed before examining new output.

## Research hypothesis

Functional auxiliary targets shared between experimental studies will improve a
sequence-only representation used to predict rescue under VTI. This is NOT a
cohort-matching, adversarial-alignment, or causal treatment-effect experiment.
There is a common encoder for the same CFTR mutation sequence, plus separate
ETI and VTI heads for measurements from different studies.

## Datasets and separation

- **VTI**: frozen Bihler 2026 transfer training membership, 210 variants at
  168 protein positions. VTI validation 52 variants, final test 33 variants;
  none of those outcome rows are loaded.
- **ETI**: only Bihler 2024 original `train` protein positions. Never load
  original ETI validation/test assay outcomes. Within each VTI held-out CV
  fold, additionally exclude all ETI labels at those held-out VTI positions
  and any 2026 validation/test positions using Parquet row filtering.
- Position-grouped 2 repeats x 5 folds with the exact seeds and held-out
  membership used for the archived VTI-only Random Forest baseline.
- Source fingerprints and the archived Forest OOF records are checked.
- At inference each model accepts ONLY the original 68 sequence/mutation
  features, never study cohort, baseline assay measurements or rescue labels.
- VTI study-cohort labels are **evaluation strata only**.

## Fixed architecture, objective and ablations

- Rank-**4** shared linear bottleneck from standardized 68-dimensional input,
  with one linear head per outcome and study; separate ETI and VTI heads.
- Input standardization and per-target response normalization fit **within
  each CV training fold**, not from held-out observations.
- Fixed 260 Adam epochs, learning rate 0.02, L2 weight penalty 0.015;
  no hyperparameter tuning on VTI validation or final test.
- Fit one instance per candidate/fold with a deterministic seed.
- Per-head loss is mean squared standardized error, with weights:

| Candidate | VTI rescue | ETI rescue | VTI FSK / FSK+IVA | ETI FSK / FSK+IVA |
|---|---:|---:|---:|---:|
| VTI latent | 1.0 | — | — | — |
| VTI biological latent | 0.7 | — | 0.15 each | — |
| Shared latent | 0.5 | 0.5 | — | — |
| Anchored shared latent | 0.35 | 0.35 | 0.075 each | 0.075 each |

The **VTI biological latent** control is critical: it isolates the benefit of
adding *ETI* training data from simply predicting extra *VTI* assay targets.

## Fixed comparisons, all from out-of-fold VTI training predictions

1. Each candidate versus the original archived VTI-only Random Forest.
2. Anchored joint latent versus unanchored joint latent (functional auxiliary effect).
3. Anchored joint latent versus VTI-only biological latent (cross-study contribution).
4. Unanchored joint latent versus VTI-only latent (ETI response contribution).

Report overall position-balanced MAE, signed bias, paired protein-position
bootstrap 95% intervals, and cohort-stratified MAE/bias. Average two repeats
**per variant before computing absolute error**; average variants within
positions so each position carries equal weight. Auxiliary held-out FSK and
IVA prediction MAEs are descriptive, not a rescue inference input.

## Interpretation guardrails

- The studies do not disentangle clinical treatment effects from experimental
  assay differences; VTI is an in-vitro treatment combination proxy.
- Shared FSK/IVA auxiliary labels are partially algebraically related to the
  rescue outcomes. Any gain is NOT automatically transferable biology.
- Matching a mutation to itself in latent space is trivial with shared
  sequence features; the test is whether training supervision improves OOF
  performance on previously unseen mutation positions.
- The 2026 ETI-characterised cohort was selected retrospectively by prior
  response. Do not include cohort or response-group labels as predictor inputs.
- Training-only repeated CV is exploratory; no final generalisation claim
  until the evaluation design is frozen and an independent test is run once.

## Run

```bash
python -m pytest -q tests/
python -m cftr_theratype.latent_transfer --smoke
python -m cftr_theratype.latent_transfer 2>&1 | tee logs/latent_transfer.log
```

Outputs under `results/transfer/latent_transfer_{oof.csv,summary.csv,summary.json}`.
No existing split, model artifact or previous results are overwritten.
