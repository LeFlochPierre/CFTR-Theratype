# CFTR-Theratype Phase 14 — Freeze-preserving target and assay-quality audit

## Scope and rationale

Run only after the Phase 13 freeze has been verified. This audit uses the **422 frozen ETI training missense variants** and **210 frozen VTI transfer-training variants**, never the validation or test responses. It is descriptive, makes no model selection, and must not change splits.

## Target equations (units: percentage points of wild-type CFTR function)

Let `F` be untreated (DMSO) FSK, `I` be DMSO FSK+IVA, and `T` be FSK+IVA after ELX/TEZ (ETI) or VNZ/TEZ (VTI) pretreatment.

- **Absolute treated function**: `T`.
- **Total rescue**: `T − F`.
- **Corrector-associated increment**: `T − I`.
- **IVA-associated increment**: `I − F`.
- **Identity**: total rescue = corrector-associated increment + IVA-associated increment.

The existing `*_rescue_pp` target measures improvement over untreated FSK, while `*_increment_over_iva_pp` isolates the increment over potentiator-treated DMSO. These are not equivalent and should not be conflated with clinical sweat chloride or lung-function outcomes.

## SEM limitations

The two conditions' reported SEMs may be combined as `sqrt(SEM_A² + SEM_B²)` **only under an independence assumption**. The original data may contain plate-normalisation effects or within-variant correlations; the true contrast variance is `Var(A − B) = Var(A) + Var(B) − 2 Cov(A,B)`. Thus the computed values are **descriptive approximate uncertainty indicators**, not calibrated per-variant label SEMs or irreducible error estimates. Report SEM missingness and check whether the original authors provide Bayesian adjustments or measurement-level covariance.

## Biological columns

Use original `source_row_dmso` and treatment `source_row_*` IDs to match the precise rows of the conditions tables. `Band C, %WT`, `Band C, %Total`, `C-B ratio, %WT`, and `Total protein, %WT` represent additional *measured* CFTR maturation information. These columns must **never become sequence-only inference features** unless a separate precomputed, leakage-free model predicts them.

## Primary decisions after the audit

1. Establish missingness, units, negative values, assay dispersion, label identities and sample sizes.
2. Determine whether absolute treated function, total rescue and corrector-associated increment pose meaningfully different prediction problems.
3. Review plausible CFTR structural annotations (UniProt, available experimental structures) **without looking at sealed test responses**.
4. Freeze a biological-feature protocol and a single prespecified experimental comparison before training models again.

## Run

```bash
python -m pytest -q tests/
python -m cftr_theratype.target_audit 2>&1 | tee logs/phase14_target_audit.log
```

Outputs are written only to ignored `results/biology/`. The training-only CSV contains assay outcomes and should remain local.
