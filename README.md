# CFTR-Theratype

**Public-data research on CFTR functional-response prediction, cross-treatment transfer, and prediction reliability.**

Current status (21 October 2026): reproducible exploratory ETI/VTI benchmarks; final held-out ETI and VTI test sets remain sealed.

## Research checkpoint

Read **[the full scientific recap](docs/RESEARCH_CHECKPOINT.md)** before running further experiments. A corresponding **[machine-readable snapshot](docs/results_checkpoint_2026-10-21.json)** records cohort sizes, target definitions, model scores, paired intervals, dataset hashes and remaining limitations. These are transcribed from terminal results and should be checked against local experiment logs before publication.

## Public datasets

- **Bihler et al. 2024:** 652 unambiguous paired ETI condition records; 586 eligible missense variants.
- **Bihler et al. 2026:** 415 usable VTI pairs; 145 exact cDNA overlaps with the 2024 study.
- **McDonald et al. 2024:** derived Bihler subset, **not** independent replication.

Raw published workbooks are kept under `data/raw/` and excluded from Git by design. See `docs/RESEARCH_CHECKPOINT.md` for exact filenames and SHA-256 hashes.

## Reproducibility

```bash
conda activate CFTR_ENV
python -m pip install -e .
python -m pytest -q tests/

cftr-data inspect
cftr-data build
cftr-data split
cftr-data baseline
cftr-data cv
cftr-data plm
cftr-data build --study 2026
cftr-data crossstudy
cftr-data transfer --stage prepare
cftr-data transfer --stage fit
cftr-data transfer --stage diagnose
cftr-data transfer --stage mechanism
cftr-data transfer --stage reliability
```

Read the full recap before rebuilding files because some commands overwrite exploratory outputs. Do not evaluate final test data until a confirmatory protocol is frozen.

## Safe project maintenance

```bash
python scripts/maintenance.py                  # dry run
python scripts/maintenance.py --apply          # generated caches + empty scaffolds only
python scripts/maintenance.py --inventory      # write local paths/hashes under logs/
```

The maintenance command never deletes datasets, experiments, logs, manifests, source files, tests or Git history.

## Interpretation limits

The prediction targets are **in-vitro functional rescue** in percentage points of wild-type CFTR function. They are not clinical-treatment response predictions. The 2026 functional VTI assay is not identical to the clinical drug combination using deutivacaftor. Cross-study differences are not head-to-head randomized estimates. Pretrained ESM-2, naive ETI→VTI pooling and simple model disagreement have not shown reliable improvements in the present experiments.

## License and sources

The project currently uses publicly published research data. Check each dataset's redistribution terms before uploading raw spreadsheets or derived data. Consult the publications and supplementary materials for experimental details and provenance.
