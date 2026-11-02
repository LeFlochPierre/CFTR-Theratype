# CFTR-Theratype — Phase 16A: mutation-specific stability pilot

**Frozen design — 23 October 2026**

## Question

Can mutation-specific structural energetics provide information beyond the position-level structural descriptors and sequence features used in Phases 15A–D? **This phase is preparation and technical feasibility, not response prediction.**

## Inputs and safety boundaries

- Experimental reference: **RCSB 6O2P** (CFTR–ivacaftor cryo-EM structure); SHA-256 must match `results/structure/phase15_coverage_summary.json`.
- Variant identities only: `data/processed/bihler_2026_transfer_splits.parquet`, using **exactly 210 frozen VTI TRAIN variants**, with an allowlist of four fields: variant key, protein HGVS, protein position, transfer split.
- No 2024/2026 assay response tables, no observed label values, no validation/test outcomes and no empirical-Bayes labels.
- Reference sequence comes from the repository's assay-reference function: canonical UniProt P13569 has **V470**; the functional assay reference has **M470**. 6O2P has V470 and a deliberately engineered **E1371Q** substitution. This construct mismatch affects the interpretation of all energies. Mutations at these two mismatched positions are excluded in this pilot.
- 6O2P retains residue F508, unlike 8EIQ, so it is a less-complicated reference for an initial stability pilot. **Do not silently pool energy estimates from different structural states or constructs.**

## Phase 16A steps

1. Validate mmCIF reference against the frozen Phase 15A source hash, and align CFTR entity polymer to canonical UniProt positions (never assume author numbering).
2. Write a **protein-only**, canonical-numbered PDB chain A, excluding ligands, ions and other polymers. Verify PDB parsing and each residue's one-letter identity.
3. Select a deterministic *label-free* pilot of 12 resolved VTI TRAIN missense variants spanning sequence positions (at most one substitution per position).
4. Inspect candidate exclusions and FoldX mutation strings in `pilot_train_only.csv`; there are no experimental response labels in it.
5. Obtain a personal/institutional FoldX licence and executable yourself. The FoldX binary and `rotabase.txt` **must never be committed, bundled, or published**.
6. Run FoldX `RepairPDB` on the prepared **protein-only reference** and `BuildModel` for the first two pilot substitutions. Use multiple independent runs to assess numerical variability. FoldX mutation file format is **WT residue – chain – residue number – mutated residue**, e.g. `GA551D;`.
7. Inspect logs, PDB repair, number of energy-difference rows, executable checksum and the magnitudes/signs. Continue only if both pilot calculations are technically valid.
8. Only after success, consider running the remaining training-only variants and one predeclared model ablation. Do **not** unlock any final test label during feature generation or exploratory analysis.

## Exact CLI

```bash
python -m cftr_theratype.stability prepare --pilot-size 12
python -m cftr_theratype.stability run --foldx /path/to/your/licensed/foldx --limit 2 --runs 3
python -m cftr_theratype.stability collect
```

The `run` command fails with an actionable error when FoldX is not installed. `prepare` works without FoldX.

## Output location

`results/structure/phase16_foldx/` (local, intentionally not committed):

- `6O2P_cftr_canonical_numbering.pdb` — prepared protein-only construct
- `candidates_train_only.csv` — all 210 train variants, eligibility and exclusion reason
- `pilot_train_only.csv` — selected variants and exact mutation strings
- `manifest.json` — input hashes, construct differences, phase and label-access guarantees
- `repair/` — FoldX repaired protein PDB, output and logs
- `runs/<hashed variant key>/` — separate FoldX runs, stdout, `Dif_*.fxout`, parsed `result.json`
- `pilot_stability_results.csv` — one row per selected variant, calculated or pending

**Back up FoldX results privately**; generated raw structures and computation outputs may be large, are not for GitHub, and FoldX binary/rotabase licensing is separate.

## Scientific limitations

- FoldX estimates an energy change of a local model, **not** a biochemical measurement of CFTR folding stability in the ER, trafficking, gating, or drug response. The sign convention and model energy terms must be documented per FoldX version.
- 6O2P captures a particular cryo-EM conformation with drug-bound history; ligands are deliberately omitted from FoldX scoring. **No change in ligand affinity is calculated.** This is not a substitute for Boltz-2 or measured drug affinities.
- Resolved residues only. Unresolved, insertion/deletion, unparseable and construct-mismatched variants are explicit missing values, not excluded from a later predictive evaluation.
- The PDB lacks unresolved loops and likely has incomplete/uncertain side-chain detail in some regions. RepairPDB may fail. Such failures must be reported rather than silently recovered.
- The small set of 210 VTI train variants and 36 training non-responders makes extra-feature overfitting a serious risk. After technical validation, only add a few **prespecified** features to the same stored 2 × 5 position-disjoint CV folds, with negative controls and cohort-specific metrics.
- The VTI assay pretreats with vanzacaftor/tezacaftor and then acutely uses ivacaftor; 6O2P is the ivacaftor-bound *reference*, not a vanzacaftor-bound construct.

## Further work, not yet undertaken

- Map computed FoldX energies against experimentally measured Band C maturation **using TRAIN only**, using results only after feature selection is frozen.
- Compare `sequence+reference structure` to the same features plus FoldX ΔΔG, with fold-local imputation and structural-coverage indicators. For non-response, evaluate balanced accuracy, non-response AUROC/AP, position-bootstrap and triage enrichment alongside the unchanged Forest comparator.
- Pilot Boltz-2 ligand pose/affinity analysis separately, and do not interpret affinity predictions as CFTR rescue percentages.

## References

- FoldX `RepairPDB` / `BuildModel` docs: https://foldxsuite.crg.eu/command/BuildModel
- FoldX mutant-file format: https://foldxsuite.crg.eu/node/239
- FoldX academic licensing: https://foldxsuite.crg.eu/academic-license-info
- RCSB 6O2P: https://www.rcsb.org/structure/6O2P
