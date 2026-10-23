# CFTR-Theratype | Phase 13 frozen research checkpoint

**Frozen:** 23 October 2026  
**Scope:** Public functional datasets, sequence-only theratyping benchmarks, cross-treatment transfer, cohort-shift diagnostics and biologically anchored latent learning.  
**Status:** Completed *exploratory* research phase, not final model validation. ETI and VTI final tests remain sealed.  
**Machine-readable table:** [`phase13_results_2026-10-23.json`](phase13_results_2026-10-23.json)  
**Earlier complete provenance:** [`RESEARCH_CHECKPOINT.md`](RESEARCH_CHECKPOINT.md) and [`results_checkpoint_2026-10-21.json`](results_checkpoint_2026-10-21.json).

> Provenance: All numerical results here are transcribed from RTX6 terminal output supplied in the conversation, not rerun on the workstation during document generation. `python -m cftr_theratype.checkpoint --create` copies and hashes the **actual** local source results for independent reconciliation. Do not report this document as external validation.

## 1. Scientific question

**Can sequence-derived features support functional CFTR theratyping for variants at unseen protein positions, and is ETI→VTI transfer possible when study cohorts are selected differently?**

Targets are in-vitro rescue in **percentage points of WT CFTR function**; they are not clinical outcomes, a drug-prescribing recommendation, or a head-to-head treatment efficacy trial. The 2026 VTI experiment is vanzacaftor/tezacaftor with acute **ivacaftor**, not the marketed deutivacaftor-containing regimen.

## 2. Data and protected partitions

- **Bihler 2024 ETI:** 652 paired records; 586 eligible missense substitutions. Original ETI split: **422 train (298 positions), 56 validation (43), 108 sealed test (86)**.
- **Bihler 2026 VTI:** 415 unambiguously paired variants, 145 exact cross-study cDNA overlaps. VTI transfer split: **210 train (168 positions), 52 validation (42), 33 sealed test (26)**; 120 excluded from this transfer benchmark.
- **VTI training cohort split:** 66 previously ETI-characterised (`ETI, VTI`, 56 positions) and 144 VTI-only (`VTI`, 127 positions).
- **Training-matched assay comparison:** 66 identical variants (56 protein positions). Baseline FSK between-study Spearman ρ=**0.793**, DMSO FSK+IVA ρ=**0.822**, total modulator rescue ρ=**0.658**, corrector-associated increment ρ=**0.624**.
- ETI samples at a VTI held-out CV position are embargoed from that fold's training. The sealed final-test labels were not loaded in these experiments.

### Caveats on data construction

The previously ETI-characterised 2026 variants were deliberately selected for poor or borderline prior ETI response. The two studies have overlapping variants but distinct conditions and ascertainment. **McDonald 2024 is derived from the original Bihler measurements and is not external replication**. Experimental CFTR assay background at residue 470 must be respected when constructing protein features (M470 in the 2024 assay versus canonical UniProt V470).

## 3. Core benchmarks

| Experiment (train-only CV unless specified) | Model | Position MAE (pp) | Interpretation |
|---|---|---:|---|
| ETI 2024 | Sequence Forest | **22.983** | Modest advantage over median 24.128 |
| ETI 2024 | ESM WT, CV-selected | 22.587 | Did **not** improve over Forest on exploratory validation |
| VTI original transfer CV | VTI-only Forest | **17.871** | Best among tested original transfer models |
| VTI original transfer CV | Pooled ETI/VTI Forest | 18.214 | No demonstrated transfer gain |
| VTI original transfer CV | ETI zero-shot Forest | 22.150 | Negative transfer |
| VTI corrector-increment CV | VTI-only Forest | 15.889 | Effectively tied with median 15.922 |
| VTI corrector-increment CV | Pooled Forest | 16.286 | No benefit from ETI pooling |

### Metric-provenance warning

The original VTI transfer experiment reports **17.871 pp** for the Forest, whereas the later matched OOF comparison pipeline (cohort controls, robustness, and latent transfer) uses **17.798 pp**. Do **not** mix these values in paired comparisons. The latter uses a matched set/aggregation of repeated predictions; the exact source and aggregations should be reconciled against the archived `train_cv_predictions.csv` and downstream output files before publication. These are not independent datasets.

## 4. Selection shift and reliability

- Median VTI rescue: prior ETI-characterised **13.476 pp** versus VTI-only **39.555 pp**; a selection-affected comparison.
- Sequence-only cohort classification (grouped OOF): AUC **0.482**, balanced accuracy **0.509**, Brier **0.361** (class weighting limits calibration interpretation).
- Coarse baseline-function standardisation across 3 supported strata: ETI-characterised minus VTI-only rescue **−19.758 pp**, Forest bias difference **+18.970 pp**. These are **descriptive, not causal**.
- Retrospective, *non-deployable* oracles on matched folds: cohort-specific median MAE **15.902 pp**; measured FSK+IVA Ridge MAE **16.129 pp**; matched sequence Forest **17.798 pp**. Baseline measurements are unavailable for most new variants and mechanically related to the rescue label.
- Model-disagreement vs absolute-error Spearman: total-rescue **−0.070**; corrector-increment **+0.011**. Risk–coverage did not improve monotonically by selecting low-disagreement variants; this is not calibrated uncertainty.

## 5. Cohort-balanced Forest: a worst-group tradeoff, not shift correction

| Model | Overall pos-MAE | Prior ETI cohort pos-MAE | VTI-only pos-MAE | Prior ETI bias | VTI-only bias |
|---|---:|---:|---:|---:|---:|
| Forest | **17.798** | 20.395 | **17.323** | +14.446 | −5.161 |
| Half balancing | 17.629 | 19.375 | 17.500 | +13.024 | −6.452 |
| Full balancing | 17.748 | **18.562** | 17.978 | +11.947 | −7.668 |

Full balancing improves worst-group position MAE by 1.833 pp (bootstrap CI [+1.083,+2.582]) but worsens VTI-only by 0.654 pp (CI for *gain* [−1.154,−0.125]). Overall gain over Forest: +0.049 pp, CI [−0.432,+0.511]. The cross-cohort bias gap remains near **19.6 pp**. Mean prediction shifts are nearly identical across cohorts (~−2.50 pp), consistent with a global level adjustment plus variant-specific changes rather than improved discrimination between cohorts.

## 6. Phase 13: biologically anchored shared latent representation

**Predefined model:** four-dimensional regularised linear bottleneck, 68 mutation/sequence input features, study/treatment-specific response heads, optional FSK/FSK+IVA auxiliary supervision. Two repeats × five position-grouped folds. At inference, no cohort label or experimental functional measurement is supplied. See [`LATENT_TRANSFER_PROTOCOL.md`](LATENT_TRANSFER_PROTOCOL.md).

| Model | Overall CV position MAE (pp) | Previously ETI cohort | VTI-only cohort |
|---|---:|---:|---:|
| **VTI-only Forest comparator** | **17.798** | **20.395** | **17.323** |
| VTI latent | 21.497 | 23.649 | 20.622 |
| VTI biologically anchored | 21.384 | 23.545 | 20.511 |
| ETI/VTI shared latent | 21.097 | 23.206 | 20.259 |
| **ETI/VTI biologically anchored latent** | **20.873** | **22.949** | **20.068** |

Best latent vs Forest: **−3.075 pp gain** (95% position-bootstrap CI [−4.668,−1.570]); worse than even the simple original median and worse in both cohorts. Signed bias: Forest (+14.446, −5.161) versus anchored latent (+15.407, −4.988); between-cohort bias gap worsens from **19.607** to **20.395 pp**.

*Within the same latent-model family,* biologically anchored joint training improves over unanchored joint training by **+0.224 pp** (CI [+0.119,+0.328]); adding ETI to VTI-anchored training improves by **+0.511 pp** (CI [+0.292,+0.725]); adding ETI rescue without auxiliaries improves by **+0.400 pp** (CI [+0.233,+0.564]). These are incremental exploratory gains, **not** evidence that the method surpasses the strong Forest comparator. Auxiliary task position MAEs: baseline FSK 14.762 pp and FSK+IVA 18.343 pp.

### Frozen conclusion

1. ETI and VTI functional assays share some information, and a small shared representation can exploit it *relative to weaker latent baselines*.
2. **No tested cross-treatment method yields a compelling overall gain over VTI-only Forest** while reducing cohort-dependent errors.
3. Strong cohort bias tracks retrospective response-based study selection; the present analysis cannot identify a causal or drug-specific explanation.
4. Linear latent learning and auxiliary biology are **completed negative ablations**. Do not perform an endless search over latent dimension or balancing weights on the same exploratory data.
5. The next research phase is **an evidence-driven audit of biological features, target definitions, and additional genuinely independent functional measurements**, before any further model development.

## 7. Reproducibility and what is frozen

Run `python -m cftr_theratype.checkpoint --dry-run` after checking Git source status. After review, `--create` archives an immutable **local-only** copy of allowed training-only outputs, protocol JSON, source-document snapshots and logs in `results/frozen/phase13_2026-10-23/`, with checksums and Git provenance. `--verify` rechecks the archived bytes. It refuses to overwrite the snapshot. It **does not copy raw workbooks, processed assay tables or sealed test labels**, and it does not rerun experiments.

**Version control:** Commit source, tests, docs and the structured recap JSON to Git. The full frozen result archive stays under locally ignored `results/` and should be backed up separately. Do not force-add raw data or local outcome files to GitHub without license/privacy review.

## 8. Appropriate current claims

**Reportable:** dataset/protocol construction; sequence vs frozen ESM findings; negative transfer; matched-study functional assay correlations; sample-selection-linked cohort bias; balancing tradeoff; latent transfer ablations.

**Not supported:** a superior theratyping AI model; clinical response prediction; causality of cross-cohort differences; calibrated uncertainty; unselected-population generalisation; any conclusion from sealed final tests.
