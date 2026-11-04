# CFTR-Theratype documentation index

This index separates **scientific results and protocols** from **repository maintenance**. Phase numbers denote research history, not versions of a deployed predictive model.

## First reads

- [README](../README.md): paused research status and supported CLI.
- [Research checkpoint](RESEARCH_CHECKPOINT.md): results and scientific provenance.
- [Phase 13 frozen results](PHASE13_FROZEN_RESULTS.md) and [structured results](phase13_results_2026-10-23.json).
- [Preservation-first freeze and cleanup](REPOSITORY_FREEZE_AND_CLEANUP.md).
- [RTX6 workstation cleanup guide](WORKSTATION_CLEANUP.md).

## Historical scientific protocols

**Functional targets and ETI/VTI transfer**

- [Latent transfer](LATENT_TRANSFER_PROTOCOL.md)
- [Phase 14 target audit](PHASE14_TARGET_AUDIT.md); [biology target audit](PHASE14_BIOLOGY_TARGET_AUDIT.md)
- [Phase 14B maturation](PHASE14B_MATURATION_PROTOCOL.md)
- [Phase 14C Band-C auxiliary](PHASE14C_BAND_C_AUX_PROTOCOL.md); [closeout](PHASE14C_CLOSEOUT.md)
- [Phase 14D response threshold](PHASE14D_RESPONSE_10PP_PROTOCOL.md)
- [Phase 15D nonresponse](PHASE15D_NONRESPONSE_PROTOCOL.md)

**Protein structure, sequence, stability and molecular modelling**

- [Phase 15 structure coverage](PHASE15_STRUCTURE_COVERAGE_PROTOCOL.md)
- [Phase 15B reference features](PHASE15B_REFERENCE_FEATURES_PROTOCOL.md)
- [Phase 15C structure benchmarks](PHASE15C_STRUCTURE_BENCHMARK_PROTOCOL.md)
- [Phase 16A FoldX](PHASE16A_FOLDX_STABILITY_PROTOCOL.md)
- [Phase 16B ThermoMPNN](PHASE16B_THERMOMPNN_PROTOCOL.md); [Phase 16C benchmark](PHASE16C_THERMOMPNN_BENCHMARK_PROTOCOL.md)
- [Phase 17A model zoo](PHASE17A_MODEL_ZOO_PROTOCOL.md)
- [Phase 18C Boltz pocket pilot](PHASE18C_PROTOCOL.md)

**Late stages**

The remote GitHub tree currently has no tracked file paths for Phases 33–39. Their local ZIP patches and possibly installed source/results on RTX6 must be inventoried and backed up before consolidation. Do not infer successful parity from the existence of an experiment ZIP.

## Previous refactoring

- [Stage 2](REFACTOR_STAGE2.md), [Stage 3](REFACTOR_STAGE3.md), [Stage 4](REFACTOR_STAGE4.md), [Stage 5](REFACTOR_STAGE5.md).
- [Stage 6 migration](refactor/STAGE6_MIGRATION.md): CLI, canonical helpers, optional hub consolidation.
- `refactor/archive_batch01_manifest.json`, `archive_batch02_manifest.json` and `archive_batch03_manifest.json`: preserved legacy-script moves.
- [Stage 7/8 workstation preservation](WORKSTATION_CLEANUP.md): scripts, docs and patch inventory.

## Why numbered documents are not moved yet

External commands or historical scripts may refer to the original document paths. This index gives the repository a clean entry point **without** changing those paths. Move older protocols in a subsequent small, reviewed `git mv` batch after references are checked.

## Historical validation/test caveat

Some older documents refer to VTI and ETI held-out sets as sealed, although Phase 26N/26O scripts and previous run discussions indicate that protected-outcome evaluation may have occurred. The split design and actual outcome-access history are different matters; reconcile the execution logs before making claims of independent final evaluation.
